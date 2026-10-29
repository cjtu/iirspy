"""Raw -> L1 radiance -> L2 reflectance, warped onto an already-solved scene's GCPs.

Takes the merged GCPs an `iirs-solve-scene` run produced and turns them into science products:
full-band L1 radiance, L2 reflectance and the OBS backplane in camera space, plus the GLT that projects them into the
group's CRS. No re-registration -- if the scene isn't solved yet, this refuses to run.

    iirs-compute-refl <sid> --group south|north|equatorial [--out WORKDIR] [--keep DIR]
        [--keep-l1 DIR] [--clean] [--gcps PATH] [--l1 PRESET] [--l1-<field> VALUE ...]
        [--l2 PRESET] [--l2-<field> VALUE ...]

`--l1` picks an `iirspy.config.L1Config` preset (default iirspy; `p0` is the frozen derivation basis) and each
`--l1-<field>` overrides one field, same JSON-value rules as `--l2-<field>`; `--fix-frames`/`--fill-frames` override
both. The resolved config is part of the L1 cache key. Ignored for `--l2 issdc-nci`.

`--l2` picks an `iirspy.config.L2Config` preset (default iirspy); `issdc-nci` is the `issdc` L2 on ISSDC's own
L1 (the nci cube under `IIRS_ARCHIVE/zips/nci`, unzipped into the archive's ISSDC tree on first use) instead of our
nri calibration -- same GCPs, GLT and OBS. Each `--l2-<field>` overrides one
`L2Config` field, e.g. `--l2-photom lunar_lambert --l2-cos-min 0.0872 --l2-thermal none`. Values
parse as JSON (`--l2-thermal-bands [225,246]`, `--l2-polish null`), else as a string.
`--l2-use-slope-az false` skips rendering the DEM slope/aspect product.

`--gcps` defaults to the standard `iirs-solve-scene --keep` layout:
geometry/recalibrated/<day>/<sid>_<group>.gcps under `IIRS_RECAL_ROOT`.

Paths come from `IIRS_ARCHIVE`, `IIRS_DEM_ROOTS`, `IIRS_SPICE`, `IIRS_STAGE` and `IIRS_RECAL_ROOT`,
same as `iirspy.solve`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

from iirspy import chunks as ck
from iirspy import solve, utils
from iirspy.config import L1Config, L2Config


def _full_bands() -> list[int]:
    """Every band calibration can produce -- the same OSF/invalid sets `calibrate_to_rad` nulls."""
    return sorted(set(range(1, 257)) - set(utils.OSF) - set(utils.INVALID))


def _topo_info(ftopo: Path | None) -> dict:
    """topo path plus the bands/piece count it used, from its own tags -- for the summary."""
    if not ftopo:
        return {}
    import rasterio

    with rasterio.open(ftopo) as src:
        t = src.tags()
    return {"path": str(ftopo), "bands_used": t.get("bands_used"), "n_pieces": t.get("n_pieces")}


def _solve_quality(fgcps: Path, group: str) -> dict:
    """converged/corr/n_merged_gcps/overlap agreement from the solve's own summary json, if kept
    next to the GCPs -- so a suspect refl product points at the registration it was built on."""
    fsummary = fgcps.parent / f"{fgcps.stem}_solve_summary.json"
    if not fsummary.exists():
        return {}
    s = json.loads(fsummary.read_text())
    fits = s.get("per_chunk_fit", [])
    return {
        "converged": all(f.get("converged") for f in fits) if fits else None,
        "corr": [f.get("corr") for f in fits],
        "n_merged_gcps": s.get("n_merged_gcps"),
        "overlap_agreement": s.get("overlap_agreement"),
    }


def _chunk_report(img) -> str:
    """Whether the cube's chunking can actually keep `ncpu` dask threads busy.

    Blocks are split along y only, so y-blocks is the real parallelism ceiling: fewer of them than
    workers and the extra cores idle no matter what `--cpus-per-task` says. Block MB x workers is
    the floor under `--mem`, since `_compute_reflectance` holds several cube-shaped temporaries.
    """
    n, ny, nx = img.shape
    cube_mb = n * ny * nx * 4 / 2**20
    if img.chunks is None:
        return f"L1 {img.shape} UNCHUNKED, {cube_mb:.0f} MB resident -- expect cube-sized temporaries"
    ychunks = img.chunks[img.dims.index("y")]
    block_mb = n * max(ychunks) * nx * 4 / 2**20
    return (
        f"L1 {img.shape} = {cube_mb:.0f} MB in {len(ychunks)} y-blocks of <={block_mb:.0f} MB "
        f"for {solve.ncpu()} workers ({'ok' if len(ychunks) >= solve.ncpu() else 'STARVED: fewer blocks than cores'})"
    )


def _peak_rss_report() -> str:
    """Peak RSS against the Slurm allocation, so `--mem` can be right-sized off a real run."""
    import resource

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    limit = os.environ.get("SLURM_MEM_PER_NODE")
    of = f" of --mem={int(limit) / 1024:.0f}G ({100 * peak_mb / int(limit):.0f}%)" if limit else ""
    return f"peak RSS {peak_mb:.0f} MB{of}"


def _keep(dest_dir: str | None, *files: Path) -> None:
    """Copy a product and its sidecars out of the work dir. The vrt names its tif by relative path,
    so it has to land in the same dir -- which is why L1 and L2 keep separately."""
    if not dest_dir:
        return
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copyfile(f, dest / f.name)
    solve.log(f"kept {len(files)} file(s) -> {dest}")


def _parser():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sid", help="scene id, e.g. 20201202T2319552644")
    ap.add_argument("--group", required=True, choices=list(ck.GROUPS))
    ap.add_argument("--out", default=None, help="work dir and restart cache (default runs/<sid>_<group>)")
    ap.add_argument("--keep", default=None, help="copy the L2 product, its vrt, summary and log here on success")
    ap.add_argument("--keep-l1", default=None, help="copy the L1 radiance product and its vrt here on success")
    ap.add_argument(
        "--l1-only", action="store_true", help="stop after the L1: keep only its _empirical.npz frames and provenance"
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="delete the work dir after --keep; drops the camera-space L1/L2 intermediates",
    )
    ap.add_argument(
        "--gcps",
        default=None,
        help="merged GCPs from iirs-solve-scene --keep (default geometry/recalibrated/<day>/"
        "<sid>_<group>.gcps under IIRS_RECAL_ROOT)",
    )
    for step, h in (
        ("fix", "move each 4-band group onto true frames"),
        ("fill", "rebuild the lost frames (implies fix)"),
    ):
        ap.add_argument(
            f"--{step}-frames",
            nargs="?",
            const=True,
            type=_json_value,
            default=None,
            help=f"{h}; unset keeps the L1 preset (iirspy L1: on, issdc-nci: off)",
        )
    ap.add_argument("--l1", default="iirspy", choices=list(L1Config.PRESETS), help="L1 preset (iirspy.config.L1Config)")
    for f in fields(L1Config):
        ap.add_argument(
            f"--l1-{f.name.replace('_', '-')}",
            dest=f"l1_{f.name}",
            type=_json_value,
            default=argparse.SUPPRESS,
            help=f"override L1Config.{f.name}",
        )
    ap.add_argument("--l2", default="iirspy", choices=list(L2Config.PRESETS), help="L2 preset (iirspy.config.L2Config)")
    for f in fields(L2Config):
        ap.add_argument(
            f"--l2-{f.name.replace('_', '-')}",
            dest=f"l2_{f.name}",
            type=_json_value,
            default=argparse.SUPPRESS,  # absent, not None: `--l2-polish null` is a real override
            help=f"override L2Config.{f.name}",
        )
    return ap


def _json_value(v: str):
    """A `--l2-<field>` value: JSON when it parses (numbers, true/false, null, lists), else the string."""
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def l2_config(args) -> L2Config:
    """The preset `args.l2` with every given `--l2-<field>` applied."""
    over = {f.name: getattr(args, f"l2_{f.name}") for f in fields(L2Config) if hasattr(args, f"l2_{f.name}")}
    return L2Config.preset(args.l2, **over)


def l1_config(args) -> L1Config:
    """The preset `args.l1` with every given `--l1-<field>` and then `--fix-frames`/`--fill-frames` applied."""
    over = {f.name: getattr(args, f"l1_{f.name}") for f in fields(L1Config) if hasattr(args, f"l1_{f.name}")}
    over |= {k: v for k in ("fix_frames", "fill_frames") if (v := getattr(args, k)) is not None}
    return L1Config.preset(args.l1, **over)


def _file_info(f, sha=False) -> dict:
    """Path and mtime (UTC) of an input, plus its sha256 when `sha`."""
    import datetime

    f = Path(f)
    mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime, datetime.UTC).isoformat(timespec="seconds")
    return {"path": str(f), "mtime": mtime, **({"sha256": hashlib.sha256(f.read_bytes()).hexdigest()} if sha else {})}


def main(argv: list[str] | None = None) -> None:

    args = _parser().parse_args(argv)
    args.l1_config = l1_config(args)
    args.l2_config = l2_config(args)  # fail on a bad preset/field value before any work
    sid, group = args.sid, args.group
    if args.clean and not (args.keep or args.keep_l1):
        sys.exit("--clean without --keep/--keep-l1 would discard the whole run")

    fgcps = Path(args.gcps) if args.gcps else solve.merged_gcps_path(sid, group)
    if not fgcps.is_file() or fgcps.stat().st_size == 0:
        sys.exit(f"{fgcps} missing or empty -- {sid} {group} is not solved yet (run iirs-solve-scene first)")

    import dask

    dask.config.set(scheduler="threads", num_workers=solve.ncpu())

    solve.SID, solve.GROUP = sid, group
    solve.FRAMES = solve.frame_offsets_path(sid, fgcps.parent)
    if solve.predates_frames(fgcps, solve.FRAMES):
        sys.exit(f"{fgcps} was solved before {solve.FRAMES.name} existed -- re-run iirs-solve-scene first")
    solve._open_run(args.out, "refl")
    t_start = time.time()
    solve._log_provenance()
    solve.log(f"{sid} group={group} zip={solve.ZIP.name} ncpu={solve.ncpu()} stage={solve.STAGE} gcps={fgcps}")

    try:
        _run(args, sid, group, fgcps, t_start)
    except Exception as e:
        if solve.is_disk_full(e):
            solve.log(f"FATAL: disk full/quota exceeded -- run `diskusage_report` to check usage. ({e})")
            sys.exit(f"{sid} {group}: disk full/quota exceeded -- run `diskusage_report` to check usage")
        raise


def _run(args, sid: str, group: str, fgcps: Path, t_start: float) -> None:  # noqa: C901
    from iirspy import georef
    from iirspy.iirs import L1, L2

    gcps, aoi = georef._gcps_and_aoi(fgcps)
    lat_range = ck.l1_lat_range(group)
    cfg = replace(ck.chunk_cfg(group), aoi=aoi, lat_band=lat_range)
    solve.log(f"aoi from {len(gcps)} merged gcps: {[round(v / 1000, 1) for v in aoi]} km")

    bands = _full_bands()
    nci = args.l2 == "issdc-nci"
    kw = {k: v for k in ("fix_frames", "fill_frames") if (v := getattr(args, k)) is not None}
    if nci:
        ftif, scan0, lat_range = solve.build_l1_nci(lat_range, bands, solve._work("l1_rad.tif"), **kw)
    else:
        ftif, scan0, lat_range = solve.build_l1(lat_range, bands, solve._work("l1_rad.tif"), l1_config=args.l1_config)
    solve.log(f"L1 built ({'ISSDC nci' if nci else 'iirspy'}): {ftif} ({len(bands)} bands), scan0={scan0}")
    # GCP rows are cube rows of the solve's crop: a different crop start would shift every GCP row.
    fsum = fgcps.parent / f"{fgcps.stem}_solve_summary.json"
    solve_scan0 = json.loads(fsum.read_text()).get("scan0") if fsum.exists() else None
    if not nci and solve_scan0 is not None and solve_scan0 != scan0:
        sys.exit(
            f"{sid} {group}: L1 crop starts at scan {scan0} but the solve's started at {solve_scan0} -- re-solve with this iirspy"
        )
    if args.l1_only:
        fnpz, fprov = ftif.with_name(ftif.stem + "_empirical.npz"), solve._work("provenance.json")
        prov = utils.provenance(
            sid=sid, group=group, l1_preset=args.l1, l1_config=args.l1_config.to_attrs(), l1_only=True
        )
        fprov.write_text(json.dumps(prov, indent=1, default=str))
        _keep(args.keep_l1, *[f for f in (fnpz, fprov) if f.exists()])
        solve.log(
            f"--l1-only: kept {fnpz.name if fnpz.exists() else 'no empirical frames'}, done in {time.time() - t_start:.0f} s"
        )
        if args.clean:
            shutil.rmtree(solve.OUT, ignore_errors=True)
        return

    anc = ck.ancillary(sid)
    fgeom, flabel = anc["geometry/calibrated"], ck.label(sid, solve.STAGE)
    if fgeom is None or flabel is None:
        sys.exit(f"missing ancillary for {sid}: geometry={fgeom} label={flabel}")
    kernels = ck.kernels(sid[:8])
    az, elev, r_sun = georef.sun_geometry(fgeom, flabel, cfg, kernels)
    solve.log(f"sun (grid frame): az={az:.1f} elev={elev:.2f} r_sun={r_sun:.4f} deg")

    l1 = L1.from_file(ftif, sid, str(ck.ARCHIVE if nci else solve.STAGE))
    if nci:
        l1.img.attrs["calibration_source"] = "issdc"
    solve.log(_chunk_report(l1.img))

    l2cfg = args.l2_config
    ftopo = (
        georef.scene_topo(sid, group, gcps, l1.img.shape[-2:], fgeom, flabel, cfg, kernels, solve.OUT, scan0=scan0)
        if l2cfg.use_slope_az
        else None
    )
    solve.log(f"topo: {ftopo or 'skipped (--l2-use-slope-az false)'}")

    # No topo product to carry the per-pixel sun: give L2 the exact per-pixel incidence instead of the spm's one
    # value per line, which misses the up-to-2% cross-track cos(i) change ISSDC's own L2 divides by.
    inc = None
    if not ftopo:
        _, el = georef.sun_geometry_rows(gcps, l1.img.shape[-2:], flabel, cfg, scan0, kernels)
        inc = xr.DataArray(90 - np.abs(el), dims=("y", "x"))
    l2 = l1.calibrate(l2cfg, inc=inc, slope_az=str(ftopo) if ftopo else None)
    solve.log(f"L2 computed ({args.l2}: {json.dumps(l2cfg.to_attrs())})")

    # Camera space keeps one unblended spectrum per pixel; the GLT projects on demand and the
    # .vrt beside each product carries the GCPs for other GDAL readers.
    fl2 = solve._work("l2_refl.tif")
    l2.save(str(fl2))
    solve.log(f"L2 saved: {fl2}")
    for f in (Path(ftif), fl2):
        solve.log(f"gcp vrt -> {georef.save_gcp_vrt(f, gcps, georef.stereo_crs(group))}")

    # OBS from the same GCPs/topo/L1 already in hand; no LOC product (see backplanes.scene_obs)
    from iirspy import backplanes

    fobs = solve._work("obs.tif")
    obs = backplanes.scene_obs(sid, group, gcps, l1.img.shape[-2:], scan0, ftopo, ftif, solve.STAGE)
    backplanes.write_obs(fobs, obs, sid, row0=scan0)
    solve.log(f"OBS saved: {fobs} (sensor geometry from {obs['sensor_source']})")

    from iirspy import l2_stats

    t_stats = time.time()
    try:
        stats = l2_stats.l2_stats(
            fl2,
            ftif,
            np.asarray(l2.img.solar_inc),
            obs["lat"],
            emi=obs["bands"]["sensor_zenith"],
            phase=obs["bands"]["phase"],
            sdist=float(l2.img.attrs["solar_distance_au"]),
        )
    except Exception as e:  # sanity stats must never fail a refl run
        stats = {"error": f"{type(e).__name__}: {e}"}
    solve.log(f"l2_stats: {time.time() - t_stats:.1f}s {stats.get('error', 'ok')}")

    fglt = fgcps.parent / f"{sid}_{group}_glt.tif"
    if not fglt.is_file():
        solve.log(f"glt: {fglt} not found -- iirs-solve-scene should have built it alongside {fgcps}")
    else:
        solve.log(f"glt: {fglt}")

    resources = f"{_chunk_report(l1.img)}; {_peak_rss_report()}; ncpu={solve.ncpu()}"
    solve.log(resources)

    summary = {
        "sid": sid,
        "group": group,
        "resources": resources,
        "gcps": str(fgcps),
        "n_gcps": len(gcps),
        "solve_quality": _solve_quality(fgcps, group),
        "lat_range": list(lat_range),
        "bands": bands,
        "l2_preset": args.l2,
        "l1_source": "issdc-nci" if nci else "iirspy",
        "l2_config": l2cfg.to_attrs(),
        "topo": _topo_info(ftopo),
        "aoi_m": list(cfg.aoi),
        "scan0": scan0,
        "l1": ftif.name,
        "l2": fl2.name,
        "obs": fobs.name,
        "obs_sensor_source": obs["sensor_source"],
        "glt": str(fglt),
        "sun_az_grid_deg": az,
        "sun_elev_deg": elev,
        "sun_angular_radius_deg": r_sun,
        "l1_empirical_notes": json.loads(l1.img.attrs["empirical_notes"])
        if "empirical_notes" in l1.img.attrs
        else None,
        "l2_diagnostics": {k: l2.img.attrs.get(k) for k in L2.DIAGNOSTICS},
        "l2_stats": stats,
        "total_s": round(time.time() - t_start, 1),
    }
    fsummary = solve._work("refl_summary.json")
    fsummary.write_text(json.dumps(summary, indent=1, default=str))
    solve.log(f"\ndone in {summary['total_s']}s.")

    inputs = {
        "nri_zip": _file_info(solve.ZIP),
        "gcps": _file_info(fgcps, sha=True),
        "frames": {**_file_info(solve.FRAMES), "key": solve._frames_key()},
        "label": _file_info(flabel),
        "geometry": _file_info(fgeom),
        "spice_kernels": [str(k) for k in kernels],
        "topo": _file_info(ftopo) if ftopo else None,
    }
    prov = utils.provenance(
        sid=sid,
        group=group,
        l1_preset=None if nci else args.l1,
        l1_config=None if nci else args.l1_config.to_attrs(),
        l2_preset=args.l2,
        l2_config=l2cfg.to_attrs(),
        inputs=inputs,
        empirical_notes=summary["l1_empirical_notes"],
    )
    fprov = solve._work("provenance.json")
    fprov.write_text(json.dumps(prov, indent=1, default=str))
    for f in (ftif, fl2):
        with rasterio.open(f, "r+") as dst:
            dst.update_tags(IIRS_PROVENANCE=json.dumps(prov, default=str))

    sidecars = [
        f for f in (ftif.with_name(ftif.stem + "_empirical.npz"), ftif.with_name(ftif.stem + "_snr.tif")) if f.exists()
    ]
    _keep(args.keep_l1, ftif, ftif.with_suffix(".vrt"), *sidecars, fprov)
    _keep(args.keep, fl2, fl2.with_suffix(".vrt"), fobs, fsummary, fprov, solve.LOG)
    if ftopo:  # geometry, so it lands beside the GCPs rather than with L2
        _keep(str(fgcps.parent) if args.keep else None, ftopo)

    if args.clean:
        shutil.rmtree(solve.OUT, ignore_errors=True)
        print(f"removed work dir {solve.OUT}", flush=True)


if __name__ == "__main__":
    main()
