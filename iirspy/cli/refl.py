"""Raw -> L1 radiance -> L2 reflectance, warped onto an already-solved scene's GCPs.

Takes the merged GCPs an `iirs-solve-scene` run produced and turns them into science products:
full-band L1 radiance, L2 reflectance and the OBS backplane in camera space, plus the GLT that projects them into the
group's CRS. No re-registration -- if the scene isn't solved yet, this refuses to run.

    iirs-compute-refl <sid> --group south|north|equatorial [--out WORKDIR] [--keep DIR]
        [--keep-l1 DIR] [--clean] [--gcps PATH] [--l1 PRESET] [--l1-<field> VALUE ...]
        [--l2 PRESET] [--l2-<field> VALUE ...]

`--l1` picks an `iirspy.l1.config.L1Config` preset (default iirspy; `p0` is the frozen derivation basis) and each
`--l1-<field>` overrides one field, same JSON-value rules as `--l2-<field>`; `--fix-frames`/`--fill-frames` override
both. The resolved config is part of the L1 cache key. Ignored for `--l2 issdc-nci`.

`--l2` picks an `iirspy.l2.config.L2Config` preset (default iirspy); `issdc-nci` is the `issdc` L2 on ISSDC's own
L1 (the nci cube under `IIRS_ARCHIVE/zips/nci`, unzipped into the archive's ISSDC tree on first use) instead of our
nri calibration -- same GCPs, GLT and OBS. Each `--l2-<field>` overrides one
`L2Config` field, e.g. `--l2-photom lunar_lambert --l2-cos-min 0.0872 --l2-thermal none`. Values
parse as JSON (`--l2-thermal-bands [225,246]`, `--l2-polish null`), else as a string.
`--l2-use-slope-az false` skips rendering the DEM slope/aspect product.

`--gcps` defaults to the standard `iirs-solve-scene --keep` layout:
geometry/recalibrated/<day>/<sid>_<group>.gcps under `IIRS_RECAL_ROOT`.

Paths come from `IIRS_ARCHIVE`, `IIRS_DEM_ROOTS`, `IIRS_SPICE`, `IIRS_STAGE` and `IIRS_RECAL_ROOT`,
same as `iirs-solve-scene`.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from iirspy import instrument, io
from iirspy.cli._run import (
    Job,
    build_l1,
    build_l1_nci,
    ensure_frame_offsets,
    file_info,
    frame_offsets_path,
    is_disk_full,
    keep,
    log_provenance,
    open_run,
    paths_from_env,
    predates_frames,
    resource_report,
)
from iirspy.georef import chunks as ck
from iirspy.instrument import PAN_BANDS
from iirspy.l1.config import L1Config
from iirspy.l1.frames import table_key as frames_key
from iirspy.l2.config import L2Config
from iirspy.workspace import ncpu


def _solve_quality(fgcps: Path) -> dict:
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
        "--topo-only",
        action="store_true",
        help="build the pan-band L1 for the scan crop, write only <sid>_<group>_topo.tif beside the GCPs (needs --keep)",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="delete the work dir after --keep; drops the camera-space L1/L2 intermediates",
    )
    ap.add_argument(
        "--georef",
        default=None,
        choices=["arosics", "issdc"],
        help="georef method: arosics (the solved GCPs) or issdc (ISSDC bundle files only, no SPICE/DEM/solve); "
        "default issdc for the issdc and issdc-nci L2 presets, else arosics",
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
    ap.add_argument(
        "--l1", default="iirspy", choices=list(L1Config.PRESETS), help="L1 preset (iirspy.l1.config.L1Config)"
    )
    for f in fields(L1Config):
        ap.add_argument(
            f"--l1-{f.name.replace('_', '-')}",
            dest=f"l1_{f.name}",
            type=_json_value,
            default=argparse.SUPPRESS,
            help=f"override L1Config.{f.name}",
        )
    ap.add_argument(
        "--l2", default="iirspy", choices=list(L2Config.PRESETS), help="L2 preset (iirspy.l2.config.L2Config)"
    )
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


def main(argv: list[str] | None = None) -> None:  # noqa: C901
    args = _parser().parse_args(argv)
    args.l1_config = l1_config(args)
    args.l2_config = l2_config(args)  # fail on a bad preset/field value before any work
    sid, group = args.sid, args.group
    if args.topo_only and not args.keep:
        sys.exit("--topo-only needs --keep: the topo is kept beside the GCPs")
    if args.clean and not (args.keep or args.keep_l1):
        sys.exit("--clean without --keep/--keep-l1 would discard the whole run")

    args.georef = args.georef or ("issdc" if args.l2 in ("issdc", "issdc-nci") else "arosics")
    if args.georef == "issdc" and args.topo_only:
        sys.exit("--topo-only needs --georef arosics: the issdc method has no DEM topo")

    paths = paths_from_env()
    fgcps = frames = None
    if args.georef == "arosics":
        fgcps = Path(args.gcps) if args.gcps else ck.merged_gcps_path(sid, group, paths)
        if not fgcps.is_file() or fgcps.stat().st_size == 0:
            sys.exit(f"{fgcps} missing or empty -- {sid} {group} is not solved yet (run iirs-solve-scene first)")
        frames = frame_offsets_path(sid, paths, fgcps.parent)
        if predates_frames(fgcps, frames):
            sys.exit(f"{fgcps} was solved before {frames.name} existed -- re-run iirs-solve-scene first")

    import dask

    dask.config.set(scheduler="threads", num_workers=ncpu())

    job = open_run(sid, group, args.out, "refl", paths)
    if frames:
        job = replace(job, frames=frames)
    elif args.l2 != "issdc-nci" or args.fix_frames or args.fill_frames:
        job = replace(job, frames=ensure_frame_offsets(job, job.ws.dir))
    log_provenance(job.ws)
    job.ws.log(
        f"{sid} group={group} zip={job.zip.name} ncpu={ncpu()} stage={job.stage} georef={args.georef} gcps={fgcps}"
    )
    try:
        _run(job, args, fgcps, time.time())
    except Exception as e:
        if is_disk_full(e):
            job.ws.log(f"FATAL: disk full/quota exceeded -- run `diskusage_report` to check usage. ({e})")
            sys.exit(f"{sid} {group}: disk full/quota exceeded -- run `diskusage_report` to check usage")
        raise


def _run(job: Job, args, fgcps: Path | None, t_start: float) -> None:  # noqa: C901
    from iirspy.georef import arosics, backplanes, georef
    from iirspy.iirs import L1, L2

    ws, sid, group, paths = job.ws, job.sid, job.group, job.paths
    lat_range = ck.l1_lat_range(group)
    gcps = cfg = None
    merged: dict = {}
    if fgcps is not None:
        try:
            gcps, aoi = georef.gcps_and_aoi(fgcps)
        except ValueError as e:
            sys.exit(str(e))
        merged = arosics.load_gcps(fgcps)
        cfg = replace(ck.chunk_cfg(group, paths.dem_roots), aoi=aoi, lat_band=lat_range)
        ws.log(f"aoi from {len(gcps)} merged gcps: {[round(v / 1000, 1) for v in aoi]} km")

    # topo-only needs the scan crop, not the spectrum
    bands = PAN_BANDS if args.topo_only else sorted(set(range(1, 257)) - set(instrument.OSF) - set(instrument.INVALID))
    nci = args.l2 == "issdc-nci"
    ftif = ws.path(f"{sid}_{group}_l1_rad.tif")
    if nci:
        kw = {k: v for k in ("fix_frames", "fill_frames") if (v := getattr(args, k)) is not None}
        ftif, scan0, lat_range = build_l1_nci(job, lat_range, bands, ftif, **kw)
    else:
        ftif, scan0, lat_range = build_l1(job, lat_range, bands, ftif, l1_config=args.l1_config)
    ws.log(f"L1 built ({'ISSDC nci' if nci else 'iirspy'}): {ftif} ({len(bands)} bands), scan0={scan0}")
    # GCP rows are cube rows of the solve's crop: a different crop start would shift every GCP row.
    fsum = fgcps.parent / f"{fgcps.stem}_solve_summary.json" if fgcps else None
    solve_scan0 = json.loads(fsum.read_text()).get("scan0") if fsum and fsum.exists() else None
    if not nci and solve_scan0 is not None and solve_scan0 != scan0:
        sys.exit(
            f"{sid} {group}: L1 crop starts at scan {scan0} but the solve's started at {solve_scan0} -- re-solve with this iirspy"
        )
    prov_kw = {"sid": sid, "group": group, "l1_preset": None if nci else args.l1}
    prov_kw["l1_config"] = None if nci else args.l1_config.to_attrs()
    if args.l1_only:
        fnpz, fprov = ftif.with_name(ftif.stem + "_empirical.npz"), ws.path(f"{sid}_{group}_provenance.json")
        fprov.write_text(json.dumps(io.provenance(**prov_kw, l1_only=True), indent=1, default=str))
        keep(ws, args.keep_l1, *[f for f in (fnpz, fprov) if f.exists()])
        ws.log(f"--l1-only: done in {time.time() - t_start:.0f} s")
        return _finish(ws, args)

    anc = ck.ancillary(sid, paths)
    fgeom, flabel = anc["geometry/calibrated"], ck.label(sid, paths, job.stage)
    if fgeom is None or flabel is None:
        sys.exit(f"missing ancillary for {sid}: geometry={fgeom} label={flabel}")
    l2cfg = args.l2_config
    kernels: list[Path] = []
    az = elev = r_sun = None
    if args.georef == "arosics":
        kernels = ck.kernels(sid[:8], paths)
        az, elev, r_sun = georef.sun_geometry(fgeom, flabel, cfg, kernels)
        ws.log(f"sun (grid frame): az={az:.1f} elev={elev:.2f} r_sun={r_sun:.4f} deg")

    if args.topo_only:
        assert fgcps is not None  # noqa: S101 -- main() routes only the arosics method here
        with rasterio.open(ftif) as src:
            nrow, ncol = src.height, src.width
        lon, lat, _ = backplanes.group_loc_core(merged, group, paths.dem_roots, nrow=nrow, ncol=ncol)
        topo = georef.scene_topo(
            sid, group, lon, lat, fgeom, flabel, cfg, kernels, ws.dir, paths.dem_roots, scan0=scan0
        )
        keep(ws, str(fgcps.parent), topo)
        ws.log(f"--topo-only: {topo} -> {fgcps.parent}, done in {time.time() - t_start:.0f} s")
        return _finish(ws, args)

    l1 = L1.from_file(ftif, sid, str(paths.archive if nci else job.stage))
    if nci:
        l1.img.attrs["calibration_source"] = "issdc"
    ws.log(resource_report(l1.img))
    inputs: dict[str, Any]
    if args.georef == "issdc":
        inputs = {"fspm": ck.spm(sid, paths, job.stage)}
    else:
        inputs = {
            "merged": merged,
            "paths": paths,
            "cfg": cfg,
            "kernels": kernels,
            "fl1": ftif,
            "stage_dir": job.stage,
            "fgcps": fgcps,
            "topo_dir": ws.dir if l2cfg.use_slope_az else None,
        }
    geo = l1.georef(args.georef, ws=ws, group=group, fgeom=fgeom, flabel=flabel, **inputs)
    if geo.row0 != scan0:
        sys.exit(f"geometry row0 {geo.row0} != L1 scan0 {scan0}")
    ftopo = geo.sidecars.get("topo")
    ws.log(f"geometry: {geo.method} {geo.quality}; topo: {ftopo or 'none'}")

    l2 = l1.calibrate(l2cfg, geom=geo, ws=ws)
    ws.log(f"L2 computed ({args.l2}: {json.dumps(l2cfg.to_attrs())})")

    # Camera space keeps one unblended spectrum per pixel; the GLT projects on demand and the
    # .vrt beside each product carries the GCPs for other GDAL readers.
    fl2 = ws.path(f"{sid}_{group}_l2_refl.tif")
    l2.save(str(fl2))
    ws.log(f"L2 saved: {fl2}")
    if gcps:
        for f in (ftif, fl2):
            ws.log(f"gcp vrt -> {georef.save_gcp_vrt(f, gcps, georef.stereo_crs(group))}")
    gfiles = geo.save(ws.dir)
    ws.log(
        f"geometry saved: {', '.join(f.name for f in gfiles.values())} (sensor geometry from {geo.tags['SENSOR_SOURCE']})"
    )

    from iirspy.l2 import stats as l2_stats

    t_stats = time.time()
    try:
        stats = l2_stats.l2_stats(
            fl2,
            ftif,
            np.asarray(l2.img.solar_inc),
            geo.loc.lat.values,
            emi=geo.obs.sensor_zenith.values,
            phase=geo.obs.phase.values,
            sdist=float(l2.img.attrs["solar_distance_au"]),
        )
    except Exception as e:  # sanity stats must never fail a refl run
        stats = {"error": f"{type(e).__name__}: {e}"}
    ws.log(f"l2_stats: {time.time() - t_stats:.1f}s {stats.get('error', 'ok')}")
    fglt = fgcps.parent / f"{sid}_{group}_glt.tif" if fgcps else None
    if fglt is not None:
        ws.log(f"glt: {fglt}" if fglt.is_file() else f"glt: {fglt} not found -- iirs-solve-scene should have built it")
    resources = f"{resource_report(l1.img)}; ncpu={ncpu()}"
    ws.log(resources)

    notes = json.loads(l1.img.attrs["empirical_notes"]) if "empirical_notes" in l1.img.attrs else None
    topo_tags = {}
    if ftopo:
        with rasterio.open(ftopo) as src:
            topo_tags = {
                "path": str(ftopo),
                "bands_used": src.tags().get("bands_used"),
                "n_pieces": src.tags().get("n_pieces"),
            }
    summary = {
        "sid": sid,
        "group": group,
        "resources": resources,
        "georef": geo.method,
        "geometry_quality": geo.quality,
        "gcps": str(fgcps) if fgcps else None,
        "n_gcps": len(gcps) if gcps else None,
        "solve_quality": _solve_quality(fgcps) if fgcps else None,
        "lat_range": list(lat_range),
        "bands": bands,
        "l2_preset": args.l2,
        "l1_source": "issdc-nci" if nci else "iirspy",
        "l2_config": l2cfg.to_attrs(),
        "topo": topo_tags,
        "aoi_m": list(cfg.aoi) if cfg else None,
        "scan0": scan0,
        "l1": ftif.name,
        "l2": fl2.name,
        "obs": gfiles["obs"].name,
        "obs_sensor_source": geo.tags["SENSOR_SOURCE"],
        "geometry": {k: f.name for k, f in gfiles.items()},
        "glt": str(fglt) if fglt else None,
        "sun_az_grid_deg": az,
        "sun_elev_deg": elev,
        "sun_angular_radius_deg": r_sun,
        "l1_empirical_notes": notes,
        "l2_diagnostics": {k: l2.img.attrs.get(k) for k in L2.DIAGNOSTICS},
        "l2_stats": stats,
        "total_s": round(time.time() - t_start, 1),
    }
    fsummary = ws.path(f"{sid}_{group}_refl_summary.json")
    fsummary.write_text(json.dumps(summary, indent=1, default=str))
    ws.log(f"\ndone in {summary['total_s']}s.")

    prov_inputs = {
        "nri_zip": file_info(job.zip),
        "gcps": file_info(fgcps, sha=True) if fgcps else None,
        "frames": {**file_info(job.frames), "key": frames_key(job.frames)} if job.frames else None,
        "label": file_info(flabel),
        "geometry_csv": file_info(fgeom),
        "geometry": geo.provenance,
        "spice_kernels": [str(k) for k in kernels],
        "topo": file_info(ftopo) if ftopo else None,
    }
    prov = io.provenance(
        **prov_kw,
        l2_preset=args.l2,
        l2_config=l2cfg.to_attrs(),
        inputs=prov_inputs,
        empirical_notes=notes,
        steps=json.loads(l2.img.attrs["provenance"]),
    )
    fprov = ws.path(f"{sid}_{group}_provenance.json")
    fprov.write_text(json.dumps(prov, indent=1, default=str))
    for f in (ftif, fl2):
        with rasterio.open(f, "r+") as dst:
            dst.update_tags(IIRS_PROVENANCE=json.dumps(prov, default=str))

    sidecars = [
        f for f in (ftif.with_name(ftif.stem + "_empirical.npz"), ftif.with_name(ftif.stem + "_snr.tif")) if f.exists()
    ]
    vrt = lambda f: [f.with_suffix(".vrt")] if f.with_suffix(".vrt").exists() else []
    keep(ws, args.keep_l1, ftif, *vrt(ftif), *sidecars, fprov)
    keep(ws, args.keep, fl2, *vrt(fl2), *gfiles.values(), fsummary, fprov, job.logfile)
    if ftopo and fgcps:  # geometry, so it lands beside the GCPs rather than with L2
        keep(ws, str(fgcps.parent) if args.keep else None, ftopo)
    _finish(ws, args)


def _finish(ws, args) -> None:
    if args.clean:
        shutil.rmtree(ws.dir, ignore_errors=True)
        print(f"removed work dir {ws.dir}", flush=True)


if __name__ == "__main__":
    main()
