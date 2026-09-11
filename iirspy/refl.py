"""Raw -> L1 radiance -> L2 reflectance, warped onto an already-solved scene's GCPs.

Takes the merged GCPs an `iirs-solve-scene` run produced and turns them into science products:
full-band L1 radiance and L2 reflectance in camera space, plus the GLT that projects them into the
group's CRS. No re-registration -- if the scene isn't solved yet, this refuses to run.

    iirs-compute-refl <sid> --group south|north|equatorial [--out WORKDIR] [--keep DIR]
        [--keep-l1 DIR] [--clean] [--gcps PATH] [--thermal-corr ""|verma]
        [--photom lambert|lommel_seeliger|lunar_lambert] [--no-topo] [--min-lit FRAC]

`--gcps` defaults to the standard `iirs-solve-scene --keep` layout:
geometry/recalibrated/<day>/<sid>_<group>/<sid>_<group>.gcps under `IIRS_RECAL_ROOT`.

Paths come from `IIRS_ARCHIVE`, `IIRS_DEM_ROOTS`, `IIRS_SPICE`, `IIRS_STAGE` and `IIRS_RECAL_ROOT`,
same as `iirspy.solve`.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

from iirspy import chunks as ck
from iirspy import solve

L2_DIAGNOSTIC_ATTRS = (
    "thermal_corr",
    "photom",
    "min_lit",
    "sun_az_offset",
    "topo_used",
    "solar_distance_au",
    "inc_min_deg",
    "inc_max_deg",
    "inc_mean_deg",
)


def _full_bands() -> list[int]:
    """Every band calibration can produce -- the same OSF/invalid sets `calibrate_to_rad` nulls."""
    from iirspy import utils

    return sorted(set(range(1, 257)) - set(utils.OSF) - set(utils.INVALID))


def _topo_info(ftopo: Path | None) -> dict:
    """topo path plus the bands/piece count it used, from its own tags -- for the summary."""
    if not ftopo:
        return {}
    import rasterio

    with rasterio.open(ftopo) as src:
        t = src.tags()
    return {"path": str(ftopo), "bands_used": t.get("bands_used"), "n_pieces": t.get("n_pieces")}


def _solve_quality(fgcps: Path) -> dict:
    """converged/corr/n_merged_gcps/overlap agreement from the solve's own summary.json, if kept
    next to the GCPs -- so a suspect refl product points at the registration it was built on."""
    fsummary = fgcps.parent / "summary.json"
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
    """Peak RSS against the Slurm allocation, so `--mem` can be right-sized off a real run.

    Self only: the staging subprocess peaks separately and well below this.
    """
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
        "--clean",
        action="store_true",
        help="delete the work dir after --keep; drops the camera-space L1/L2 intermediates",
    )
    ap.add_argument(
        "--gcps",
        default=None,
        help="merged GCPs from iirs-solve-scene --keep (default geometry/recalibrated/<day>/"
        "<sid>_<group>/<sid>_<group>.gcps under IIRS_RECAL_ROOT)",
    )
    ap.add_argument("--thermal-corr", default="", choices=("", "verma"), help="L2 thermal correction")
    ap.add_argument("--photom", default="lambert", help="photometric model name (iirspy.photometry.MODELS)")
    ap.add_argument(
        "--no-topo",
        action="store_true",
        help="flat local-horizontal correction; skip the DEM slope/aspect/lit (comparison baseline)",
    )
    ap.add_argument(
        "--min-lit",
        type=float,
        default=0.9,
        help="null pixels the terrain leaves less than this fraction lit; no-op without topo",
    )
    return ap


def main(argv: list[str] | None = None) -> None:

    args = _parser().parse_args(argv)
    sid, group = args.sid, args.group
    if args.clean and not (args.keep or args.keep_l1):
        sys.exit("--clean without --keep/--keep-l1 would discard the whole run")

    fgcps = Path(args.gcps) if args.gcps else solve.merged_gcps_path(sid, group)
    if not fgcps.is_file() or fgcps.stat().st_size == 0:
        sys.exit(f"{fgcps} missing or empty -- {sid} {group} is not solved yet (run iirs-solve-scene first)")

    import dask

    dask.config.set(scheduler="threads", num_workers=solve.ncpu())

    solve.SID, solve.GROUP = sid, group
    solve._open_run(args.out)
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


def _run(args, sid: str, group: str, fgcps: Path, t_start: float) -> None:
    from iirspy import georef
    from iirspy.iirs import L1

    gcps, aoi = georef._gcps_and_aoi(fgcps)
    lat_range = ck.l1_lat_range(group)
    cfg = replace(ck.chunk_cfg(group), aoi=aoi, lat_band=lat_range)
    solve.log(f"aoi from {len(gcps)} merged gcps: {[round(v / 1000, 1) for v in aoi]} km")

    bands = _full_bands()
    ftif, scan0, lat_range = solve.build_l1(lat_range, bands, solve.OUT / f"{sid}_{group}_l1_rad.tif")
    solve.log(f"L1 built: {ftif} ({len(bands)} bands), scan0={scan0}")

    anc = ck.ancillary(sid)
    fgeom, fspm = anc["geometry/calibrated"], anc["miscellaneous/raw"]
    if fspm is None:
        fspm = next(solve.STAGE.glob(f"miscellaneous/raw/{sid[:8]}/*{sid}*.spm"), None)
    if fgeom is None or fspm is None:
        sys.exit(f"missing ancillary for {sid}: geometry={fgeom} spm={fspm}")
    kernels = ck.kernels(sid[:8])
    az, elev, r_sun = georef.sun_geometry(fgeom, fspm, cfg, kernels)
    solve.log(f"sun (grid frame): az={az:.1f} elev={elev:.2f} r_sun={r_sun:.4f} deg")

    l1 = L1.from_file(ftif, sid, str(solve.STAGE))
    solve.log(_chunk_report(l1.img))

    ftopo = (
        None
        if args.no_topo
        else georef.scene_topo(
            sid, group, gcps, l1.img.shape[-2:], fgeom, fspm, cfg, kernels, fgcps.parent, scan0=scan0
        )
    )
    solve.log(f"topo: {ftopo or 'skipped (--no-topo)'}")

    l2 = l1.calibrate(
        thermal_corr=args.thermal_corr, photom=args.photom, topo=str(ftopo) if ftopo else None, min_lit=args.min_lit
    )
    solve.log(
        f"L2 computed (thermal_corr={args.thermal_corr or 'none'}, photom={args.photom}, "
        f"topo={'yes' if ftopo else 'no'})"
    )

    # Camera space keeps one unblended spectrum per pixel; the GLT projects on demand and the
    # .vrt beside each product carries the GCPs for other GDAL readers.
    fl2 = solve.OUT / f"{sid}_{group}_l2_refl.tif"
    l2.save(str(fl2))
    solve.log(f"L2 saved: {fl2}")
    for f in (Path(ftif), fl2):
        solve.log(f"gcp vrt -> {georef.save_gcp_vrt(f, gcps, georef.stereo_crs(group))}")

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
        "solve_quality": _solve_quality(fgcps),
        "lat_range": list(lat_range),
        "bands": bands,
        "thermal_corr": args.thermal_corr,
        "photom": args.photom,
        "topo": _topo_info(ftopo),
        "aoi_m": list(cfg.aoi),
        "scan0": scan0,
        "l1": ftif.name,
        "l2": fl2.name,
        "glt": str(fglt),
        "sun_az_grid_deg": az,
        "sun_elev_deg": elev,
        "sun_angular_radius_deg": r_sun,
        "l1_empirical_notes": json.loads(l1.img.attrs["empirical_notes"])
        if "empirical_notes" in l1.img.attrs
        else None,
        "l2_diagnostics": {k: l2.img.attrs.get(k) for k in L2_DIAGNOSTIC_ATTRS},
        "total_s": round(time.time() - t_start, 1),
    }
    (solve.OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    solve.log(f"\ndone in {summary['total_s']}s.")

    _keep(args.keep_l1, ftif, ftif.with_suffix(".vrt"))
    _keep(args.keep, fl2, fl2.with_suffix(".vrt"), solve.OUT / "summary.json", solve.LOG)

    if args.clean:
        shutil.rmtree(solve.OUT, ignore_errors=True)
        print(f"removed work dir {solve.OUT}", flush=True)


if __name__ == "__main__":
    main()
