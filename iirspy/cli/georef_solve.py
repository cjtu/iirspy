"""Chunked AROSICS registration of one CRS group of one scene (`L1.georef("arosics")`).

    iirs-georef-solve <sid> --group south|north|equatorial [--out WORKDIR] [--keep DIR] [--clean]

`--out` holds everything the run produces and doubles as its restart cache: a re-run reuses any
chunk already solved at the same shape, so a rerun only re-solves what it did not finish.
`--keep` copies the durable products out of it -- GCPs, per-chunk fits, summary, log -- and
`--clean` then deletes the rest, which is rebuildable from the zip.

Paths come from `IIRS_ARCHIVE`, `IIRS_DEM_ROOTS`, `IIRS_SPICE` and `IIRS_STAGE`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

from iirspy import io
from iirspy.cli._run import (
    Job,
    build_l1,
    clean_work,
    ensure_frame_defects,
    log_provenance,
    open_run,
    predates_frame_defects,
)
from iirspy.georef import ancillary, arosics
from iirspy.georef import chunks as ck
from iirspy.georef.georef import scan_order_reversed
from iirspy.iirs import L1
from iirspy.instrument import PAN_BANDS
from iirspy.io import ncpu
from iirspy.l1.frame_defects import BAND_GAP

log = logging.getLogger(__name__)


# (flag, GeorefConfig field, type, help); an unset flag takes the field's tuned default.
TUNING = (
    ("decay-m", "decay_m", float, "how far (m) local tie points are trusted before blending to the chunk's bulk shift"),
    ("gcp-row-margin", "gcp_row_margin", int, "GCP lattice rows kept either side of a chunk (-1 = whole cube)"),
    (
        "edge-reject-m",
        "edge_reject_m",
        float,
        "drop tie points closer than this (m) to the swath's nodata edge (0 = off)",
    ),
    ("shadow-max", "shadow_max", float, "drop tie points whose match window is more shadowed than this (1 = off)"),
    ("plateau-frac", "p95_plateau_frac", float, "stop when p95 improves by less than this fraction (0 = off)"),
    ("min-iter", "min_iter", int, "never declare convergence before this iteration (0 = off)"),
    ("edge-fit-k", "edge_fit_k", int, "off-support fallback plane fit to the k nearest tie points (0 = off)"),
    ("edge-dense-m", "edge_dense_m", float, "denser tie-point pass within this (m) of the swath edge (0 = off)"),
)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sid", help="scene id, e.g. 20201202T2319552644")
    ap.add_argument("--group", required=True, choices=list(ck.GROUPS))
    ap.add_argument("--out", default=None, help="work dir and restart cache (default runs/<sid>_<group>)")
    ap.add_argument(
        "--keep",
        default=None,
        help="copy the merged GCPs, GLT, LOC, summary and log here when the run completes; an already-solved "
        "scene's GLT/LOC backfill defaults to geometry/recalibrated/<day> under IIRS_RECAL_ROOT",
    )
    ap.add_argument("--resolve", action="store_true", help="re-solve even if merged GCPs already exist")
    ap.add_argument("--clean", action="store_true", help="delete the work dir after --keep")
    ap.add_argument(
        "--chunks", default=None, help="comma-separated chunk indices to solve, e.g. 0,3,8; skips merge/warp"
    )
    ap.add_argument("--hillshade-only", action="store_true", help="write each chunk's reference hillshade, then stop")
    ap.add_argument(
        "--chunk-y",
        type=int,
        default=int(os.environ.get("IIRS_CHUNK_Y", 1024)),
        help="dask y-chunk and write block size for the L1 (env IIRS_CHUNK_Y)",
    )
    ap.add_argument("--edge-reject-axis", choices=("cross", "all"), default=None, help="edge for --edge-reject-m")
    for flag, field, typ, h in TUNING:
        ap.add_argument(f"--{flag}", dest=field, type=typ, default=None, help=h)
    return ap


def _band_window(band: int) -> list[int]:
    return sorted(range(max(1, band - BAND_GAP), min(256, band + BAND_GAP) + 1))


def _write_reversed_csv(job: Job, fgeom: Path) -> Path:
    """`fgeom` with Scan renumbered max - Scan, in the recal tree where `ck.ancillary` prefers it."""
    df = ancillary.read_geom_csv(fgeom)
    df["Scan"] = df["Scan"].max() - df["Scan"]
    dst = ck.recal_dir(job.sid, job.paths) / fgeom.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["Scan", "Pixel"]).to_csv(dst, index=False, float_format="%.9g")
    log.info(f"csv Scan runs against time (matches SPICE only reversed): using {dst}")
    return dst


def _glt_only(job: Job, defects: Path, args, cfg, fgeom, lat_range) -> bool:
    """Already solved: build the GLT (and LOC, if missing) from the saved GCPs instead of restaging and re-solving."""
    work, sid, group = job.work, job.sid, job.group
    name = f"{sid}_{group}.gcps"
    fdone = next((f for f in (work / name, *([Path(args.keep) / name] if args.keep else [])) if f.exists()), None)
    if fdone is None or args.resolve or args.chunks or args.hillshade_only:
        return False
    if predates_frame_defects(fdone, defects):
        log.info(f"{fdone} predates {defects.name} -- re-solving on the re-timed cube")
        return False
    if fgeom is None:
        sys.exit(f"missing geometry csv for {sid}")
    merged = arosics.load_gcps(fdone)
    keep = Path(args.keep or ck.recal_dir(sid, job.paths))
    scan0 = arosics.scene_scan0(fgeom, lat_range)
    log.info(f"{fdone} exists ({len(merged)} gcps) -- skipped solve")
    # GCPs found only in the work dir are an interrupted run's, so land them before anything else.
    arosics.keep_products(sid, group, work, keep)
    ksummary = keep / f"{sid}_{group}_solve_summary.json"
    fits = json.loads(ksummary.read_text()).get("per_chunk_fit", []) if ksummary.exists() else []
    builds = {
        "glt": lambda: arosics.make_glt(sid, group, merged, replace(cfg, lat_band=lat_range), scan0, str(work)),
        "loc": lambda: arosics.make_loc(work, sid, group, job.paths.dem_roots, merged, scan0, str(work), fits),
    }
    for kind, build in builds.items():
        f = keep / f"{sid}_{group}_{kind}.tif"
        if not f.exists():
            build()
            arosics.keep_products(sid, group, work, keep)
        log.info(f"{kind}: {f}")
    return True


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.clean and not args.keep:
        sys.exit("--clean without --keep would discard the whole run")
    paths = io.Paths()

    import dask

    dask.config.set(scheduler="threads", num_workers=ncpu())

    job = open_run(args.sid, args.group, args.out, "solve", paths)
    work, sid, group = job.work, job.sid, job.group
    log_provenance()
    log.info(f"{sid} group={group} zip={job.zip.name} ncpu={ncpu()} stage={job.stage}")

    cfg = ck.chunk_cfg(group, paths.dem_roots)
    fgeom = ck.ancillary(sid, paths)["geometry/calibrated"]
    keep = Path(args.keep) if args.keep else None
    defects = ensure_frame_defects(job, keep)
    job = replace(job, frame_defects=defects)
    lat_range = ck.l1_lat_range(group)
    if _glt_only(job, defects, args, cfg, fgeom, lat_range):
        return clean_work(work, args.clean)

    # Calibration only masks/interpolates the registration band's +-BAND_GAP window; the PAN_BANDS beyond it
    # feed the empirical dark/flat/smile derivation. Only the registration band is written.
    ftif, scan0, lat_range = build_l1(
        job,
        lat_range,
        sorted(set(PAN_BANDS) | set(_band_window(cfg.band))),
        work / f"{sid}_{group}_l1_group.tif",
        calibrate_kwargs={"output_bands": _band_window(cfg.band)},
        out_bands=[cfg.band],
        chunk_y=args.chunk_y,
    )
    cfg = replace(cfg, lat_band=lat_range)
    flabel = ck.label(sid, paths, job.stage)
    if fgeom is None or flabel is None:
        sys.exit(f"missing ancillary for {sid}: geometry={fgeom} label={flabel}")

    l1 = L1.from_file(ftif, sid, str(job.stage))
    assert int(l1.img.y.values[0]) == scan0, "L1 crop does not start at the scan build_l1 reported"  # noqa: S101
    ny, nx = l1.img.shape[-2:]
    log.info(f"L1 cube ({group} group, lat {lat_range}): {ny} rows x {nx} cols, scan0={scan0}")

    kernels = ck.kernels(sid[:8], paths)
    try:
        reversed_, scan_source = scan_order_reversed(fgeom, flabel, ck.spm(sid, paths, job.stage), kernels)
    except FileNotFoundError as e:
        sys.exit(str(e))
    log.info(f"scan order reversed={reversed_} (track from {scan_source})")
    if reversed_:
        fgeom = _write_reversed_csv(job, fgeom)

    inputs = {
        "paths": paths,
        "cfg": cfg,
        "fgeom": fgeom,
        "flabel": flabel,
        "kernels": kernels,
        "frame_defects": defects,
        "ftif": ftif,
        "decay_m": args.decay_m,
        "tweaks": {k: getattr(args, k) for k in arosics.TWEAK_DEFAULTS},
    }
    if args.chunks or args.hillshade_only:
        chunks, results, _ = arosics.solve_group(
            work,
            sid=sid,
            group=group,
            row0=scan0,
            shape=(ny, nx),
            only=args.chunks,
            hillshade_only=args.hillshade_only,
            **inputs,
        )
        return log.info(
            f"\n{len(results)} of {len(chunks)} chunk(s) solved; no merge/warp for --chunks/--hillshade-only."
        )
    l1.georef("arosics", work=work, group=group, scan_source=scan_source, lat_range=lat_range, keep=keep, **inputs)
    clean_work(work, args.clean)


if __name__ == "__main__":
    main()
