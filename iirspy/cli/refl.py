"""Raw -> L1 radiance -> L2 reflectance on an already-solved scene's geometry.

    iirs-compute-refl <sid> --group south|north|equatorial [--out WORKDIR] [--keep DIR] [--keep-l1 DIR]
        [--clean] [--georef arosics|issdc] [--gcps PATH] [--l1 PRESET] [--l1-<field> VALUE ...]
        [--l2 PRESET] [--l2-<field> VALUE ...]

Writes full-band L1 radiance, L2 reflectance and the LOC/OBS/QA `Geometry` in camera space. `--georef arosics`
(default) builds the geometry from the GCPs `iirs-georef-solve` produced and refuses to run on an unsolved scene;
`--georef issdc` (default for the `issdc`/`issdc-nci` L2 presets) needs only the ISSDC bundle files.

`--l1`/`--l2` pick an `L1Config`/`L2Config` preset (`lut-only` is the LUT median gain with no empirical corrections); each
`--l1-<field>`/`--l2-<field>` overrides one field, parsed as JSON else a string, e.g.
`--l2-photom lunar_lambert --l2-thermal-bands [225,246] --l2-polish null`. The resolved L1 config is part of the L1
cache key. `--l2 issdc-nci` runs the `issdc` L2 on ISSDC's own nci cube (unzipped into the archive tree on first use).

`--gcps` defaults to geometry/recalibrated/<day>/<sid>_<group>.gcps under `IIRS_RECAL_ROOT`. Paths come from
`IIRS_ARCHIVE`, `IIRS_DEM_ROOTS`, `IIRS_SPICE`, `IIRS_STAGE` and `IIRS_RECAL_ROOT`, as for `iirs-georef-solve`.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import rasterio

from iirspy import instrument, io
from iirspy.cli._run import (
    Job,
    add_config_args,
    build_l1,
    build_l1_nci,
    clean_work,
    ensure_frame_offsets,
    frame_offsets_path,
    is_disk_full,
    keep,
    l1_config,
    l2_config,
    log_provenance,
    open_run,
    predates_frames,
    resource_report,
    write_refl_outputs,
)
from iirspy.georef import chunks as ck
from iirspy.instrument import PAN_BANDS
from iirspy.io import ncpu

log = logging.getLogger(__name__)


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
        help="merged GCPs from iirs-georef-solve --keep (default geometry/recalibrated/<day>/"
        "<sid>_<group>.gcps under IIRS_RECAL_ROOT)",
    )
    add_config_args(ap)
    return ap


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

    paths = io.Paths()
    fgcps = frames = None
    if args.georef == "arosics":
        fgcps = Path(args.gcps) if args.gcps else ck.merged_gcps_path(sid, group, paths)
        if not fgcps.is_file() or fgcps.stat().st_size == 0:
            sys.exit(f"{fgcps} missing or empty -- {sid} {group} is not solved yet (run iirs-georef-solve first)")
        frames = frame_offsets_path(sid, paths, fgcps.parent)
        if predates_frames(fgcps, frames):
            sys.exit(f"{fgcps} was solved before {frames.name} existed -- re-run iirs-georef-solve first")

    import dask

    dask.config.set(scheduler="threads", num_workers=ncpu())

    job = open_run(sid, group, args.out, "refl", paths)
    if frames:
        job = replace(job, frames=frames)
    elif args.l2 != "issdc-nci" or args.fix_frames or args.fill_frames:
        job = replace(job, frames=ensure_frame_offsets(job, job.work))
    log_provenance()
    log.info(
        f"{sid} group={group} zip={job.zip.name} ncpu={ncpu()} stage={job.stage} georef={args.georef} gcps={fgcps}"
    )
    try:
        _run(job, args, fgcps, time.time())
    except Exception as e:
        if is_disk_full(e):
            log.info(f"FATAL: disk full/quota exceeded -- run `diskusage_report` to check usage. ({e})")
            sys.exit(f"{sid} {group}: disk full/quota exceeded -- run `diskusage_report` to check usage")
        raise


def _run(job: Job, args, fgcps: Path | None, t_start: float) -> None:  # noqa: C901
    from iirspy.georef import arosics, backplanes, georef
    from iirspy.iirs import L1

    work, sid, group, paths = job.work, job.sid, job.group, job.paths
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
        log.info(f"aoi from {len(gcps)} merged gcps: {[round(v / 1000, 1) for v in aoi]} km")

    # topo-only needs the scan crop, not the spectrum
    if args.topo_only:
        bands = PAN_BANDS
    elif args.l2_config.polish == "issdc":  # its spike step reads each band's neighbours, over the whole cube
        bands = list(range(1, 257))
    else:
        bands = sorted(set(range(1, 257)) - set(instrument.OSF) - set(instrument.INVALID))
    nci = args.l2 == "issdc-nci"
    ftif = work / f"{sid}_{group}_l1_rad.tif"
    if nci:
        kw = {k: v for k in ("fix_frames", "fill_frames") if (v := getattr(args, k)) is not None}
        ftif, scan0, lat_range = build_l1_nci(job, lat_range, bands, ftif, **kw)
    else:
        ftif, scan0, lat_range = build_l1(job, lat_range, bands, ftif, l1_config=args.l1_config)
    log.info(f"L1 built ({'ISSDC nci' if nci else 'iirspy'}): {ftif} ({len(bands)} bands), scan0={scan0}")
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
        fnpz, fprov = ftif.with_name(ftif.stem + "_empirical.npz"), work / f"{sid}_{group}_provenance.json"
        fprov.write_text(json.dumps(io.provenance(**prov_kw, l1_only=True), indent=1, default=str))
        keep(work, args.keep_l1, *[f for f in (fnpz, fprov) if f.exists()])
        log.info(f"--l1-only: done in {time.time() - t_start:.0f} s")
        return clean_work(work, args.clean)

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
        log.info(f"sun (grid frame): az={az:.1f} elev={elev:.2f} r_sun={r_sun:.4f} deg")

    if args.topo_only:
        assert fgcps is not None  # noqa: S101 -- main() routes only the arosics method here
        with rasterio.open(ftif) as src:
            nrow, ncol = src.height, src.width
        lon, lat, _ = backplanes.group_loc_core(merged, group, paths.dem_roots, nrow=nrow, ncol=ncol)
        topo = georef.scene_topo(sid, group, lon, lat, fgeom, flabel, cfg, kernels, work, paths.dem_roots, scan0=scan0)
        keep(work, str(fgcps.parent), topo)
        log.info(f"--topo-only: {topo} -> {fgcps.parent}, done in {time.time() - t_start:.0f} s")
        return clean_work(work, args.clean)

    l1 = L1.from_file(ftif, sid, str(paths.archive if nci else job.stage))
    if nci:
        l1.img.attrs["calibration_source"] = "issdc"
    log.info(resource_report(l1.img))
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
            "topo_dir": work if l2cfg.use_slope_az else None,
        }
    geo = l1.georef(args.georef, work=work, group=group, fgeom=fgeom, flabel=flabel, **inputs)
    if geo.row0 != scan0:
        sys.exit(f"geometry row0 {geo.row0} != L1 scan0 {scan0}")
    ftopo = geo.sidecars.get("topo")
    log.info(f"geometry: {geo.method} {geo.quality}; topo: {ftopo or 'none'}")

    l2 = l1.calibrate(l2cfg, geom=geo)
    log.info(f"L2 computed ({args.l2}: {json.dumps(l2cfg.to_attrs())})")

    # Camera space keeps one unblended spectrum per pixel; the GLT projects on demand and the
    # .vrt beside each product carries the GCPs for other GDAL readers.
    fl2 = work / f"{sid}_{group}_l2_refl.tif"
    l2.save(str(fl2))
    log.info(f"L2 saved: {fl2}")
    if gcps:
        for f in (ftif, fl2):
            log.info(f"gcp vrt -> {georef.save_gcp_vrt(f, gcps, georef.stereo_crs(group))}")
    gfiles = geo.save(work)
    log.info(
        f"geometry saved: {', '.join(f.name for f in gfiles.values())} (sensor geometry from {geo.tags['SENSOR_SOURCE']})"
    )

    write_refl_outputs(
        job,
        args,
        l1=l1,
        l2=l2,
        geo=geo,
        ftif=ftif,
        fl2=fl2,
        fgcps=fgcps,
        gcps=gcps,
        cfg=cfg,
        lat_range=lat_range,
        bands=bands,
        scan0=scan0,
        gfiles=gfiles,
        sun=(az, elev, r_sun),
        kernels=kernels,
        fgeom=fgeom,
        flabel=flabel,
        prov_kw=prov_kw,
        t_start=t_start,
    )
    clean_work(work, args.clean)


if __name__ == "__main__":
    main()
