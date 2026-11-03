"""Chunked AROSICS registration of one CRS group of one scene.

Builds an L1 product cropped to the group's latitudes, splits it into chunks via
`iirspy.georef.chunks.plan_chunks`, solves each chunk against its own bounded hillshade, then merges the
chunks with a cosine cross-fade over each row overlap and warps the merged result.

    iirs-solve-scene <sid> --group south|north|equatorial [--out WORKDIR] [--keep DIR] [--clean]

`--out` holds everything the run produces and doubles as its restart cache: a re-run reuses any
chunk already solved at the same shape, so a rerun only re-solves what it did not finish.
`--keep` copies the durable products out of it -- GCPs, per-chunk fits, summary, log -- and
`--clean` then deletes the rest, which is rebuildable from the zip.

Paths come from `IIRS_ARCHIVE`, `IIRS_DEM_ROOTS`, `IIRS_SPICE` and `IIRS_STAGE`.
"""

from __future__ import annotations

import filecmp
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

from iirspy import io
from iirspy.cli._run import (
    Job,
    build_l1,
    ensure_frame_offsets,
    frame_offsets_path,
    log_provenance,
    open_run,
    paths_from_env,
    predates_frames,
)
from iirspy.georef import ancillary, arosics
from iirspy.georef import chunks as ck
from iirspy.workspace import DeadlineReached, ncpu

# Exit code for "stopped early on purpose, work is checkpointed, run me again" -- distinct from 0
# (finished) and from any real failure, so a submission script can tell a resumable stop from a failure.
EXIT_RESUME = 75


# Copied to `--keep`, each after the `<sid>_<group>` prefix every work-dir file carries (the keep dir
# is a shared day dir). Everything else -- chunk gcps/fits/plan, hillshades, the L1 and its vrt -- is
# the scratch restart cache; the summary carries the chunk plan and every chunk's fit.
# - `_L1_b*.tif`: the final warped raster at the registration band (only 1 band, 10s of MB)
KEEP_GLOBS = (".gcps", "_glt.tif", "_loc.tif", "_solve_summary.json", "_solve.log", "_L1_b*.tif")


def _band_window(band: int) -> list[int]:
    """+-BAND_GAP bands around `band` -- as far as `interpolate_na(..., max_gap=BAND_GAP)` ever reaches to fill a
    masked `band` pixel from its true neighbours, so it's also as much of the band axis as
    `iirspy.l1.calibrate`'s masking/interpolation stage needs to touch to get `band` right.
    """
    from iirspy.l1.fill import BAND_GAP

    return sorted(range(max(1, band - BAND_GAP), min(256, band + BAND_GAP) + 1))


def _cube_bands(band: int) -> list[int]:
    """Bands `build_l1`'s calibration actually touches: PAN_BANDS (broadband dark/flat/smile) plus
    `_band_window(band)` (see there). Wider than what the masking/interpolation stage runs on
    (`_band_window` alone, via `output_bands`) -- PAN_BANDS beyond the window only feed the
    empirical dark/flat/smile derivation, which sees the full staged cube regardless.
    """
    from iirspy.instrument import PAN_BANDS

    return sorted(set(PAN_BANDS) | set(_band_window(band)))


def _write_reversed_csv(job: Job, fgeom: Path) -> Path:
    """`fgeom` with Scan renumbered max - Scan, in the recal tree where `ck.ancillary` prefers it."""
    df = ancillary.read_geom_csv(fgeom)
    df["Scan"] = df["Scan"].max() - df["Scan"]
    dst = ck.recal_dir(job.sid, job.paths) / fgeom.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["Scan", "Pixel"]).to_csv(dst, index=False, float_format="%.9g")
    job.ws.log(f"csv Scan runs against time (matches SPICE only reversed): using {dst}")
    return dst


def _merged_gcps_paths(job: Job, keep: str | None) -> list[Path]:
    """Where a previous run's merged GCPs could be, work dir first."""
    name = f"{job.sid}_{job.group}.gcps"
    return [job.ws.dir / name, *([Path(keep) / name] if keep else [])]


def already_solved(sid: str, group: str, paths: io.Paths) -> bool:
    """Whether `sid`/`group` has non-empty merged GCPs solved after its frame table -- what a submission script
    checks before deciding a scene needs a solve job at all (job-count control belongs there, not in a
    job that's already been spawned)."""
    f = ck.merged_gcps_path(sid, group, paths)
    return f.is_file() and f.stat().st_size > 0 and not predates_frames(f, frame_offsets_path(sid, paths))


def keep_products(sid: str, group: str, out: Path, dest: Path) -> list[Path]:
    """Copy the durable products in `out` (see `KEEP_GLOBS`) to `dest`, skipping any already there
    unchanged, so calling it after each new product lands just that product. Returns what was copied."""
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for pattern in KEEP_GLOBS:
        for f in sorted(out.glob(f"{sid}_{group}{pattern}")):
            d = dest / f.name
            # copy2 keeps mtime, so a file this kept before matches on filecmp's size+mtime alone.
            if d.exists() and filecmp.cmp(f, d):
                continue
            shutil.copy2(f, d)
            copied.append(f)
    return copied


def _write_and_keep(job: Job, summary: dict, dest: Path | None) -> None:
    """Write the solve summary to the work dir and, with `dest`, copy the products there. Before LOC
    (`summary["loc"]` None) any earlier solve's LOC is deleted from both, since it would pair with the
    wrong GCPs."""
    if summary["loc"] is None:
        for d in (job.ws.dir, dest or job.ws.dir):
            (d / f"{job.sid}_{job.group}_loc.tif").unlink(missing_ok=True)
    job.ws.path(f"{job.sid}_{job.group}_solve_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if dest:
        job.ws.log(f"kept {len(keep_products(job.sid, job.group, job.ws.dir, dest))} file(s) -> {dest}")


def _kept_fits(job: Job, keep: Path) -> list[dict]:
    """The per-chunk fits from a kept solve summary, or [] for a solve that predates it."""
    f = keep / f"{job.sid}_{job.group}_solve_summary.json"
    return json.loads(f.read_text()).get("per_chunk_fit", []) if f.exists() else []


def _glt_only(job: Job, frames: Path, args, cfg0, fgeom, lat_range) -> bool:
    """Already solved: build the GLT (and LOC, if missing) from the saved GCPs instead of restaging and re-solving."""
    fdone = next((f for f in _merged_gcps_paths(job, args.keep) if f.exists()), None)
    if fdone is None or args.resolve or args.chunks or args.hillshade_only:
        return False
    if predates_frames(fdone, frames):
        job.ws.log(f"{fdone} predates {frames.name} -- re-solving on the re-timed cube")
        return False
    if fgeom is None:
        sys.exit(f"missing geometry csv for {job.sid}")
    merged = arosics.load_gcps(fdone)
    keep = Path(args.keep or ck.recal_dir(job.sid, job.paths))
    cfg = replace(cfg0, lat_band=lat_range)
    scan0 = arosics.scene_scan0(fgeom, lat_range)
    job.ws.log(f"{fdone} exists ({len(merged)} gcps) -- skipped solve")
    # GCPs found only in the work dir are an interrupted run's, so land them before anything else.
    keep_products(job.sid, job.group, job.ws.dir, keep)
    # Built on scratch, then kept: nothing is written into or read back from keep as a cache.
    for name, build in (
        ("glt.tif", lambda: arosics.make_glt(job.sid, job.group, merged, cfg, scan0, str(job.ws.dir))),
        (
            "loc.tif",
            lambda: arosics.make_loc(
                job.ws,
                job.sid,
                job.group,
                job.paths.dem_roots,
                merged,
                scan0,
                str(job.ws.dir),
                _kept_fits(job, keep),
            ),
        ),
    ):
        f = keep / f"{job.sid}_{job.group}_{name}"
        if not f.exists():
            build()
            keep_products(job.sid, job.group, job.ws.dir, keep)
        job.ws.log(f"{name.split('.')[0]}: {f}")
    if args.clean:
        shutil.rmtree(job.ws.dir, ignore_errors=True)
        print(f"removed work dir {job.ws.dir}", flush=True)
    return True


def _parser():
    import argparse

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
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=float(os.environ.get("IIRS_MAX_SECONDS", 0)) or None,
        help=f"stop before starting any step that would not finish inside this budget and exit "
        f"{EXIT_RESUME}, leaving the work dir resumable, instead of being killed at the wall clock "
        f"(env IIRS_MAX_SECONDS; default unbounded). Set it a few minutes under the SBATCH --time.",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="delete the work dir after --keep; drops the L1, hillshades and per-chunk rasters",
    )
    ap.add_argument(
        "--chunks",
        default=None,
        help="comma-separated chunk indices to solve, e.g. 0,3,8 -- skips merge/warp, "
        "for quick before/after comparison on a representative subset",
    )
    ap.add_argument(
        "--hillshade-only",
        action="store_true",
        help="render and write each chunk's reference hillshade, then stop -- for eyeballing a DEM "
        "rewire before paying for the solve",
    )
    ap.add_argument(
        "--chunk-y",
        type=int,
        default=int(os.environ.get("IIRS_CHUNK_Y", 1024)),
        help="dask y-chunk for the L1 read/calibrate/write, and so the write block size "
        "(env IIRS_CHUNK_Y; default 1024 rows)",
    )
    ap.add_argument(
        "--gcp-row-margin",
        type=int,
        default=None,
        help="rows of GCP lattice kept either side of a chunk's own rows (default GeorefConfig."
        "gcp_row_margin; -1 = the whole cube). Clipping cuts per-iteration TPS cost.",
    )
    ap.add_argument(
        "--decay-m",
        type=float,
        default=None,
        help="override GeorefConfig.decay_m (default 5000m) -- how far the displacement field "
        "trusts local tie points before blending to the chunk's bulk median shift",
    )
    ap.add_argument(
        "--edge-reject-m",
        type=float,
        default=None,
        help="drop tie points whose centre is closer than this to the swath's own nodata edge (default 1000; 0 = off)",
    )
    ap.add_argument(
        "--edge-reject-axis",
        choices=("cross", "all"),
        default=None,
        help="measure --edge-reject-m from the left/right swath edges only ('cross', default) or from any nodata ('all')",
    )
    ap.add_argument(
        "--shadow-max",
        type=float,
        default=None,
        help="drop tie points whose reference match window is more shadowed than this fraction "
        "(default GeorefConfig.shadow_max; 1 = off)",
    )
    ap.add_argument(
        "--plateau-frac",
        type=float,
        default=None,
        help="stop when p95 improves by less than this fraction of the previous iteration's p95, "
        "independent of GeorefConfig.p95_stop_m (default GeorefConfig.p95_plateau_frac; 0 = off)",
    )
    ap.add_argument(
        "--min-iter",
        type=int,
        default=None,
        help="never declare convergence before this iteration; mad_from_iter+1 (=4) guarantees one "
        "MAD-culled field is applied and measured (default GeorefConfig.min_iter; 0 = off)",
    )
    ap.add_argument(
        "--edge-fit-k",
        type=int,
        default=None,
        help="off-support displacement fallback: fit a plane to the k nearest tie points instead "
        "of using the chunk-wide bulk median (default GeorefConfig.edge_fit_k; 0 = off)",
    )
    ap.add_argument(
        "--edge-dense-m",
        type=float,
        default=None,
        help="second, denser tie-point pass within this far of the swath edge (default GeorefConfig.edge_dense_m; 0 = off)",
    )
    return ap


def main(argv: list[str] | None = None) -> None:  # noqa: C901
    from iirspy import georef
    from iirspy.georef.georef import read_band, scan_order_reversed

    args = _parser().parse_args(argv)
    paths = paths_from_env()
    deadline = time.monotonic() + args.max_seconds if args.max_seconds else None

    # Bind dask's threadpool to the allocation before anything touches a chunked array (see `ncpu`).
    import dask

    dask.config.set(scheduler="threads", num_workers=ncpu())

    job = open_run(args.sid, args.group, args.out, "solve", paths, deadline)
    ws, sid, group = job.ws, job.sid, job.group
    t_start = time.time()
    log_provenance(ws)
    ws.log(f"{sid} group={group} zip={job.zip.name} ncpu={ncpu()} stage={job.stage}")

    cfg0 = ck.chunk_cfg(group, paths.dem_roots)
    anc = ck.ancillary(sid, paths)
    fgeom = anc["geometry/calibrated"]

    frames = ensure_frame_offsets(job, Path(args.keep) if args.keep else None)
    job = replace(job, frames=frames)
    lat_range0 = ck.l1_lat_range(group)
    if _glt_only(job, frames, args, cfg0, fgeom, lat_range0):
        return

    # The crop carries a buffer past the solved bands; the chunk plan does not.
    # A chunk solve only ever reads `cfg0.band` back out, so calibration only masks/interpolates
    # its own +-6 window (`output_bands`), not every PAN_BANDS the empirical search also staged;
    # `out_bands` then narrows what's actually written to just that one band.
    ftif, scan0, lat_range = build_l1(
        job,
        lat_range0,
        _cube_bands(cfg0.band),
        ws.path(f"{sid}_{group}_l1_group.tif"),
        calibrate_kwargs={"output_bands": _band_window(cfg0.band)},
        out_bands=[cfg0.band],
        chunk_y=args.chunk_y,
    )
    cfg0 = replace(cfg0, lat_band=lat_range)

    flabel = ck.label(sid, paths, job.stage)
    if fgeom is None or flabel is None:
        sys.exit(f"missing ancillary for {sid}: geometry={fgeom} label={flabel}")

    band = read_band(ftif, cfg0.band)
    ny, nx = band.shape
    del band
    ws.log(f"L1 cube ({group} group, lat {lat_range}): {ny} rows x {nx} cols, scan0={scan0}")

    kernels = ck.kernels(sid[:8], paths)
    try:
        reversed_, scan_source = scan_order_reversed(fgeom, flabel, ck.spm(sid, paths, job.stage), kernels)
    except FileNotFoundError as e:
        sys.exit(str(e))
    ws.log(f"scan order reversed={reversed_} (track from {scan_source})")
    if reversed_:
        fgeom = _write_reversed_csv(job, fgeom)

    try:
        chunks = arosics.plan_group_chunks(ws, sid, group, fgeom, paths.dem_roots, scan0, ny, args.chunks, cfg0.niter)
    except ValueError as e:
        sys.exit(str(e))

    decay_m = args.decay_m if args.decay_m is not None else cfg0.decay_m
    # An unset flag takes GeorefConfig's tuned default, recorded in each fit so the cache key sees it.
    tweaks = {
        k: getattr(cfg0, k) if v is None else v
        for k, v in {
            "edge_reject_m": args.edge_reject_m,
            "edge_reject_axis": args.edge_reject_axis,
            "shadow_max": args.shadow_max,
            "edge_dense_m": args.edge_dense_m,
            "edge_fit_k": args.edge_fit_k,
            "p95_plateau_frac": args.plateau_frac,
            "min_iter": args.min_iter,
            "gcp_row_margin": args.gcp_row_margin,
        }.items()
    }

    try:
        results, consensus_info = arosics.solve_chunks(
            ws, sid, group, frames, kernels, chunks, cfg0, ftif, fgeom, flabel, decay_m, tweaks, args.hillshade_only
        )
    except DeadlineReached:
        ws.log(f"chunks solved and cached; exiting {EXIT_RESUME} to resume.")
        sys.exit(EXIT_RESUME)

    if args.hillshade_only:
        ws.log(f"\n--hillshade-only: wrote {len(chunks)} hillshade(s), no solve.")
        return

    if args.chunks:
        # A --chunks subset is a before/after comparison, not a strip solve: its chunks are
        # typically non-adjacent, and merge/warp cost scales with the union AOI regardless of how
        # few chunks fed it.
        ws.log(f"\n--chunks given: solved {len(results)} chunk(s), skipping merge/warp.")
        return

    if len(results) < min(2, len(chunks)):
        sys.exit("fewer than 2 chunks solved -- nothing to merge")

    results.sort(key=lambda r: r["chunk"]["i"])
    used_chunks = [r["chunk"] for r in results]
    merged, agree_summary = arosics.merge_gcps(ws, results)

    fgcps = ws.path(f"{sid}_{group}.gcps")
    arosics.save_gcps(fgcps, merged)
    ws.log(f"\nmerged gcps: {len(merged)} points -> {fgcps}")

    # The L1 raster holds a geotransform for `scan0`, so its GCPs have to ride in a sidecar; without
    # it the kept raster is camera-space only and nothing but iirspy can place it.
    fvrt = georef.save_gcp_vrt(ftif, arosics.gcp_list(merged), georef.stereo_crs(group))
    ws.log(f"gcp vrt: {fvrt}")

    final_cfg, final = arosics.warp_merged(ws, sid, group, merged, used_chunks, cfg0, ftif)
    final_shape = list(final.shape)
    del final  # written to disk by warp_merged; only the shape is still needed below
    fglt = arosics.make_glt(sid, group, merged, cfg0, scan0, str(ws.dir), overwrite=True)
    ws.log(f"glt: {fglt}")

    summary = {
        "sid": sid,
        "group": group,
        "scan_source": scan_source,
        "lat_range": lat_range,
        "solve_lat_range": ck.group_lat_range(group),
        "width_range_km": arosics.WIDTH_RANGE_KM,
        "overlap_frac": arosics.OVERLAP_FRAC,
        "n_chunks": len(chunks),
        "n_chunks_solved": len(results),
        "chunks": chunks,
        "per_chunk_fit": [r["fit"] for r in results],
        "overlap_agreement": agree_summary,
        "solve_grade": arosics.solve_grade(agree_summary, max(c["p95_stop_m"] for c in used_chunks)),
        **consensus_info,
        "n_merged_gcps": len(merged),
        "scan0": scan0,
        "glt": fglt.name,
        "loc": None,
        "final_aoi_m": final_cfg.aoi,
        "final_shape": final_shape,
        "total_s": round(time.time() - t_start, 1),
    }
    dest = Path(args.keep) if args.keep else None
    # Keep the GCPs and GLT before LOC: a LOC that dies (OOM, wall clock) must not take the solve with it.
    _write_and_keep(job, summary, dest)
    floc = arosics.make_loc(
        ws, sid, group, paths.dem_roots, merged, scan0, str(ws.dir), [r["fit"] for r in results], overwrite=True
    )
    summary |= {"loc": floc.name, "total_s": round(time.time() - t_start, 1)}
    ws.log(f"\ndone in {summary['total_s']}s.")
    _write_and_keep(job, summary, dest)

    if dest:
        if args.clean:
            shutil.rmtree(ws.dir, ignore_errors=True)
            print(f"removed work dir {ws.dir}", flush=True)
    elif args.clean:
        sys.exit("--clean without --keep would discard the whole run")


if __name__ == "__main__":
    main()
