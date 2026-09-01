"""Raw -> L1 radiance -> L2 reflectance, warped onto an already-solved scene's GCPs.

Takes the merged GCPs an `iirs-solve-scene` run produced and turns them into science products:
full-band L1 radiance and L2 reflectance in camera space, plus the GLT that projects them into the
group's CRS. No re-registration -- if the scene isn't solved yet, this refuses to run.

    iirs-compute-refl <sid> --group south|north|equatorial [--out WORKDIR] [--keep DIR] [--clean]
        [--gcps PATH] [--thermal-corr ""|verma] [--photom lambert|lommel_seeliger|lunar_lambert]

`--gcps` defaults to the standard `iirs-solve-scene --keep` layout:
$HOME/data/iirs/gcps/<sid>_<group>/<sid>_<group>_merged.gcps

Paths come from `IIRS_ARCHIVE`, `IIRS_DEM_ROOTS`, `IIRS_SPICE` and `IIRS_STAGE`, same as `iirspy.solve`.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

from rasterio.control import GroundControlPoint

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


def _gcps_and_aoi(fgcps: Path) -> tuple[list[GroundControlPoint], tuple[float, float, float, float]]:
    """GroundControlPoints from a merged `.gcps` file, plus the AOI their own x/y span."""
    merged = solve._load_gcps(fgcps)
    if not merged:
        sys.exit(f"{fgcps} has no GCPs")
    gcps = [GroundControlPoint(row=float(r), col=float(c), x=float(x), y=float(y)) for (r, c), (x, y) in merged.items()]
    xs, ys = [g.x for g in gcps], [g.y for g in gcps]
    return gcps, (min(xs), min(ys), max(xs), max(ys))


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


def _parser():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sid", help="scene id, e.g. 20201202T2319552644")
    ap.add_argument("--group", required=True, choices=list(ck.GROUPS))
    ap.add_argument("--out", default=None, help="work dir and restart cache (default runs/<sid>_<group>)")
    ap.add_argument("--keep", default=None, help="copy warped L1/L2, summary and log here when the run completes")
    ap.add_argument(
        "--clean",
        action="store_true",
        help="delete the work dir after --keep; drops the camera-space L1/L2 intermediates",
    )
    ap.add_argument(
        "--gcps",
        default=None,
        help="merged GCPs from iirs-solve-scene --keep (default $HOME/data/iirs/gcps/<sid>_<group>/"
        "<sid>_<group>_merged.gcps)",
    )
    ap.add_argument("--glts", default=str(Path.home() / "data" / "iirs" / "glts"), help="where scene GLTs live")
    ap.add_argument("--thermal-corr", default="", choices=("", "verma"), help="L2 thermal correction")
    ap.add_argument("--photom", default="lambert", help="photometric model name (iirspy.photometry.MODELS)")
    return ap


def main(argv: list[str] | None = None) -> None:
    from iirspy import georef, glt
    from iirspy.iirs import L1

    args = _parser().parse_args(argv)
    sid, group = args.sid, args.group

    fgcps = (
        Path(args.gcps)
        if args.gcps
        else Path.home() / "data" / "iirs" / "gcps" / f"{sid}_{group}" / f"{sid}_{group}_merged.gcps"
    )
    if not fgcps.is_file() or fgcps.stat().st_size == 0:
        sys.exit(f"{fgcps} missing or empty -- {sid} {group} is not solved yet (run iirs-solve-scene first)")

    import dask

    dask.config.set(scheduler="threads", num_workers=solve.ncpu())

    solve.SID, solve.GROUP = sid, group
    solve._open_run(args.out)
    t_start = time.time()
    solve._log_provenance()
    solve.log(f"{sid} group={group} zip={solve.ZIP.name} ncpu={solve.ncpu()} stage={solve.STAGE} gcps={fgcps}")

    gcps, aoi = _gcps_and_aoi(fgcps)
    lat_range = ck.l1_lat_range(group)
    cfg = replace(ck.chunk_cfg(group), aoi=aoi, lat_band=lat_range)
    solve.log(f"aoi from {len(gcps)} merged gcps: {[round(v / 1000, 1) for v in aoi]} km")

    bands = _full_bands()
    ftif, scan0, lat_range = solve.build_l1(lat_range, bands, solve.OUT / f"{sid}_l1_{group}_full.tif")
    solve.log(f"L1 built: {ftif} ({len(bands)} bands), scan0={scan0}")

    l1 = L1.from_file(ftif, sid, str(solve.STAGE))
    l2 = l1.calibrate(thermal_corr=args.thermal_corr, photom=args.photom)
    solve.log(f"L2 computed (thermal_corr={args.thermal_corr or 'none'}, photom={args.photom})")

    anc = ck.ancillary(sid)
    fgeom, fspm = anc["geometry/calibrated"], anc["miscellaneous/raw"]
    if fspm is None:
        fspm = next(solve.STAGE.glob(f"miscellaneous/raw/{sid[:8]}/*{sid}*.spm"), None)
    if fgeom is None or fspm is None:
        sys.exit(f"missing ancillary for {sid}: geometry={fgeom} spm={fspm}")
    az, elev, r_sun = georef.sun_geometry(fgeom, fspm, cfg, ck.kernels(sid[:8]))
    solve.log(f"sun (grid frame): az={az:.1f} elev={elev:.2f} r_sun={r_sun:.4f} deg")

    # Camera space keeps one unblended spectrum per pixel; the GLT projects on demand and the
    # .vrt beside each product carries the GCPs for other GDAL readers.
    fl2 = solve.OUT / f"{sid}_{group}_l2_refl.tif"
    l2.save(str(fl2))
    solve.log(f"L2 saved: {fl2}")
    for f in (Path(ftif), fl2):
        solve.log(f"gcp vrt -> {glt.save_gcp_vrt(f, gcps, georef.stereo_crs(group))}")

    fglt = glt.scene_glt(sid, group, gcps, cfg, scan0, args.glts)
    solve.log(f"glt: {fglt}")

    summary = {
        "sid": sid,
        "group": group,
        "gcps": str(fgcps),
        "n_gcps": len(gcps),
        "solve_quality": _solve_quality(fgcps),
        "lat_range": list(lat_range),
        "bands": bands,
        "thermal_corr": args.thermal_corr,
        "photom": args.photom,
        "aoi_m": list(cfg.aoi),
        "scan0": scan0,
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

    if args.keep:
        dest = Path(args.keep)
        dest.mkdir(parents=True, exist_ok=True)
        for f in (fl2, fl2.with_suffix(".vrt"), solve.OUT / "summary.json", solve.LOG):
            shutil.copyfile(f, dest / f.name)
        solve.log(f"kept 4 file(s) -> {dest}")
        if args.clean:
            shutil.rmtree(solve.OUT, ignore_errors=True)
            print(f"removed work dir {solve.OUT}", flush=True)
    elif args.clean:
        sys.exit("--clean without --keep would discard the whole run")


if __name__ == "__main__":
    main()
