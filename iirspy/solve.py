"""Chunked AROSICS registration of one CRS group of one scene.

Builds an L1 product cropped to the group's latitudes, splits it into chunks via
`iirspy.chunks.plan_chunks`, solves each chunk against its own bounded hillshade, then merges the
chunks with a cosine cross-fade over each row overlap and warps the merged result.

    iirs-solve-scene <sid> --group south|north|equatorial [--out WORKDIR] [--keep DIR] [--clean]

`--out` holds everything the run produces and doubles as its restart cache: a re-run reuses any
chunk already solved at the same shape, so a requeued job only re-solves what it did not finish.
`--keep` copies the durable products out of it -- GCPs, per-chunk fits, summary, log -- and
`--clean` then deletes the rest, which is rebuildable from the zip.

Paths come from `IIRS_ARCHIVE`, `IIRS_DEM_ROOTS`, `IIRS_SPICE` and `IIRS_STAGE`.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from iirspy import chunks as ck

WIDTH_RANGE_KM = (100.0, 150.0)
OVERLAP_FRAC = 0.10

# Exit code for "stopped early on purpose, work is checkpointed, run me again" -- distinct from 0
# (finished) and from any real failure, so a submission script can requeue on it alone.
EXIT_RESUME = 75

# Copied to `--keep`; everything else in the work dir is rebuildable from the zip.
# - all gcp files generated and run log / json stats
# - `*_L1_b*.tif`: the final warped raster at the registration band (only 1 band, 10s of MB)
KEEP_GLOBS = ("*.gcps", "chunk*_fit.json", "chunks.json", "summary.json", "run.log", "*_L1_b*.tif")

# Set by `main` from argv; module-level because `build_l1` and `log` both need them.
SID = ""
GROUP = ""
ZIP = Path()
OUT = Path()
LOG = Path()
# Where the zip is extracted and the one-root PDS tree assembled. Calibration reads the raw cube in
# millions of small pieces, so this wants the fastest local filesystem available: node-local disk
# on a cluster, native ext4 rather than a drvfs mount on WSL. Shared across scenes.
STAGE = Path()
# When this run must be finished by (`time.monotonic()`), or None for no limit. See `_time_left`.
DEADLINE: float | None = None


def _time_left() -> float:
    """Seconds until `DEADLINE`, or inf when the run is unbounded."""
    return np.inf if DEADLINE is None else DEADLINE - time.monotonic()


def ncpu() -> int:
    """Cores this process may use.

    `os.cpu_count()` reports the machine, not the cgroup, so under Slurm it reads the whole node
    however few `--cpus-per-task` were granted. Both dask's default threadpool and arosics' `CPUs`
    size themselves from it, so left alone a small allocation spawns a node's worth of workers.
    """
    return len(os.sched_getaffinity(0))


def log(msg):
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(str(msg) + "\n")


def is_disk_full(exc: BaseException) -> bool:
    """Disk quota exceeded surfaces as OSError(errno) from shutil.copy*, or as a message-only
    error from GDAL/rasterio writes -- check both so either path gets the same clear diagnosis."""
    if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):
        return True
    msg = str(exc).lower()
    return "disk quota exceeded" in msg or "no space left on device" in msg


class phase:
    """Time one phase and log wall, CPU and the mean cores it actually kept busy.

    `cores` is the number that answers "would more `--cpus-per-task` help here": a stage pinned near
    1.0 is serial (GIL-bound Python, or waiting on disk) and more cores buy it nothing, while one
    that tracks `ncpu()` is already scaling. CPU time sums this process *and* its waited-for
    children, so a subprocess stage (zip extraction) is measured on the same footing as an in-process
    one, not reported as idle.
    """

    def __init__(self, name: str):
        self.name = name

    def _cpu_s(self) -> float:
        import resource

        return sum(
            r.ru_utime + r.ru_stime
            for r in (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN))
        )

    def __enter__(self):
        self.t0, self.c0 = time.monotonic(), self._cpu_s()
        return self

    def __exit__(self, *exc):
        self.wall, self.cpu = time.monotonic() - self.t0, self._cpu_s() - self.c0
        self.cores = self.cpu / max(self.wall, 1e-9)
        log(f"{self.name}: {self.wall:.0f}s wall, {self.cpu:.0f}s cpu, {self.cores:.1f} cores of {ncpu()}")
        return False


def _cross_fade(t: float) -> float:
    """Weight for the earlier of two overlapping chunks: 1 at the overlap's start, 0 at its end.

    >>> [float(_cross_fade(t)) for t in (0.0, 0.5, 1.0)]
    [1.0, 0.5, 0.0]
    >>> [float(_cross_fade(t)) for t in (-1.0, 2.0)]  # clipped outside [0, 1]
    [1.0, 0.0]
    """
    return float(0.5 * (1 + np.cos(np.pi * np.clip(t, 0, 1))))


def _save_gcps(path: Path, gcps_by_rc: dict) -> None:
    lines = ["row,col,x,y\n"] + [f"{r},{c},{x},{y}\n" for (r, c), (x, y) in sorted(gcps_by_rc.items())]
    path.write_text("".join(lines))


def _load_gcps(path: Path) -> dict:
    gcps_by_rc = {}
    for line in path.read_text().splitlines()[1:]:
        r, c, x, y = (float(v) for v in line.split(","))
        gcps_by_rc[(r, c)] = (x, y)
    return gcps_by_rc


# How a fit json written before one of these knobs existed must be read, so an old cache is only
# reused by a run that also leaves them off.
TWEAK_DEFAULTS = {
    "edge_reject_m": 0.0,
    "shadow_max": 1.0,
    "edge_dense_m": 0.0,
    "edge_fit_k": 0,
    "p95_plateau_frac": 0.0,
    "min_iter": 0,
    "gcp_row_margin": 250,
}


def _dem_key(path: str) -> str:
    """Cache-comparable identity for a DEM path: the product filename, not the mount root.

    `IIRS_DEM_ROOTS` differs between a cluster job and a local rerun (e.g. cluster `/scratch/...`
    vs local `/mnt/d/...`), so comparing full paths would call every chunk's cache stale on a
    different machine even though the same DEM product solved it.

    >>> _dem_key("/scratch/cjtu/lro_lola/LOLA_GDR/POLAR/SOUTH_POLE/LDEM_75S_30M.tif")
    'LDEM_75S_30M.tif'
    >>> _dem_key("/mnt/d/ddata/moon/lro_lola/LOLA_GDR/POLAR/SOUTH_POLE/LDEM_75S_30M.tif")
    'LDEM_75S_30M.tif'
    """
    return Path(path).name


def _cached_chunk(out: Path, c: dict, good_corr: float, decay_m: float, tweaks: dict):
    """The prior solve of chunk `c['i']` if it is still valid, else None.

    Valid means its band, DEM pair, row bounds, aoi, decay_m and tweaks all match `c` and it
    wasn't rejected, at corr >= `good_corr`. Convergence isn't required -- a non-converged chunk
    that still passed quality (5-iter cap, no plateau) re-solves to the same result every time, so
    treating it as cache-invalid just burns wall-clock re-deriving an answer that won't change.
    The DEM pair is part of the key (by product name, see `_dem_key`) because `plan_chunks` picks
    the far tier per chunk, so two runs can agree on shape and still have used different
    references. Returns (fit, gcps_by_rc).
    """
    ffit, fgcp = out / f"chunk{c['i']}_fit.json", out / f"chunk{c['i']}.gcps"
    if not (ffit.exists() and fgcp.exists()):
        return None
    fit = json.loads(ffit.read_text())
    same = (
        fit.get("band") == c["band"]
        and _dem_key(fit.get("dem_near", "")) == _dem_key(c["dem_near"])
        and _dem_key(fit.get("dem_far", "")) == _dem_key(c["dem_far"])
        and fit.get("row0") == c["row0"]
        and fit.get("row1") == c["row1"]
        and fit.get("decay_m") == decay_m
        and all(fit.get(k, TWEAK_DEFAULTS[k]) == v for k, v in tweaks.items())
        and all(abs(a - b) < 1.0 for a, b in zip(fit.get("aoi", []), c["aoi"], strict=True))
    )
    good = fit.get("corr") is not None and fit["corr"] >= good_corr and not fit.get("quality", {}).get("rejected")
    if same and good:
        return fit, _load_gcps(fgcp)
    return None


def _band_window(band: int) -> list[int]:
    """+-6 bands around `band` -- as far as `interpolate_na(..., max_gap=6)` ever reaches to fill a
    masked `band` pixel from its true neighbours, so it's also as much of the band axis as
    `calibrate_to_rad`'s masking/interpolation stage needs to touch to get `band` right.
    """
    return sorted(range(max(1, band - 6), min(256, band + 6) + 1))


def _cube_bands(band: int) -> list[int]:
    """Bands `build_l1`'s calibration actually touches: PAN_BANDS (broadband dark/flat/smile) plus
    `_band_window(band)` (see there). Wider than what the masking/interpolation stage runs on
    (`_band_window` alone, via `output_bands`) -- PAN_BANDS beyond the window only feed the
    empirical dark/flat/smile derivation, which sees the full staged cube regardless.
    """
    from iirspy.empirical import PAN_BANDS

    return sorted(set(PAN_BANDS) | set(_band_window(band)))


def _stage_inputs(day: str, bands: list[int]) -> Path:
    """Extract the zip and assemble the one-root PDS tree `L0` reads. Returns the raw cube's path.

    Ancillary is staged from every `ANC_ROOT`, not just the archive: without the nci geometry csv
    here `l1.csv` is None and the crop dies in `parse_geom` after the whole cube has calibrated.
    """
    raw_qub = STAGE / f"data/raw/{day}/{ZIP.stem}.qub"
    if not raw_qub.exists():
        log(f"extracting {ZIP.name} ({len(bands)} bands) -> {STAGE}")
        band_arg = ",".join(str(b) for b in bands)
        with phase("extraction"):
            subprocess.run(  # noqa: S603
                [sys.executable, "-m", "issdc_iirs", str(ZIP), "-o", str(STAGE), "--bands", band_arg], check=True
            )
    for sub in (
        f"geometry/calibrated/{day}",
        f"miscellaneous/raw/{day}",
        f"miscellaneous/calibrated/{day}",
        f"data/calibrated/{day}",
    ):
        src_dir = ck.ARCHIVE / sub
        if not src_dir.exists():
            continue
        dst_dir = STAGE / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.glob(f"*{SID}*"):
            if not (dst_dir / f.name).exists():
                shutil.copyfile(f, dst_dir / f.name)
    for key, anc in ck.ancillary(SID).items():
        if anc is None:
            continue
        dst = STAGE / key / day / anc.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(anc, dst)
    return raw_qub


CALIBRATE_KWS = {"empirical": True, "interp_bands": "linear", "interp_spatial": True, "bad_pixel_mask": True}


def build_l1(
    lat_range: tuple[float, float],
    bands: list[int],
    ftif_out: Path,
    calibrate_kwargs: dict | None = None,
    out_bands: list[int] | None = None,
    chunk_y: int = 1024,
) -> tuple[Path, int, tuple[float, float]]:
    """Extract the nri zip and calibrate it, cropped to `lat_range`.

    The zip's internal layout already matches the archive's PDS tree, so it extracts in place.
    Returns the L1 path, the crop's first `Scan` value (row 0's scan id, for turning a chunk's
    scan_lo/scan_hi into row indices), and the latitudes the returned file actually covers, which
    for a cache hit may be a wider earlier build. `bands` is part of the cache key -- a cube staged
    for one band list is never reused for a different one, e.g. a single-band solve cache can't
    satisfy a full-spectrum refl run.

    `out_bands` narrows what actually gets *written*, after `bands` has done its job of giving the
    dark/flat/smile search enough of the spectrum to work with -- a chunk solve only ever reads one
    band back out (`read_band(ftif, cfg.band)`), so writing all of `bands` to disk for that case is
    pure I/O nobody uses. None (default, what `iirspy.refl` wants) writes every calibrated band.

    calibrate_kwargs overrides CALIBRATE_KWS (e.g. interp_bands=None, exclude_wl=[...]) for a
    one-off reprocessing run without touching the production default. `chunk_y` is the dask y-chunk
    the cube is read and written in, and so also the write/resume block size.
    """
    # Ancillary only (spm, oat, xml, csv -- excludes the qub by default): a few MB, so cheap enough
    # to always re-run even on an L1 cache hit. Without this, a cache hit skips `_stage_inputs`
    # below and STAGE never gets the spm the zip carries, since it does not live in ARCHIVE.
    subprocess.run([sys.executable, "-m", "issdc_iirs", str(ZIP), "-o", str(STAGE)], check=True)  # noqa: S603
    # The nri zip's own layout only ever yields .../raw, but iirspy.L1 (level=1) always looks
    # under .../calibrated -- for spm (calibrate()'s FileNotFoundError check) and for the xml
    # label (from_xarray's solar_inc/solar_az lookup, iirs.py `paths.get("qub"/"xml")` at
    # level=1). Mirror both so a stage-only run (no full nci archive copy) still satisfies them.
    day = SID[:8]
    for root, ext in (("miscellaneous", "spm"), ("data", "xml")):
        src_dir = STAGE / root / "raw" / day
        dst_dir = STAGE / root / "calibrated" / day
        for fsrc in src_dir.glob(f"*{SID}*.{ext}"):
            dst_dir.mkdir(parents=True, exist_ok=True)
            if not (dst_dir / fsrc.name).exists():
                shutil.copyfile(fsrc, dst_dir / fsrc.name)

    saved_bands = out_bands if out_bands is not None else bands
    fmeta = ftif_out.with_suffix(".meta.json")
    if ftif_out.exists() and fmeta.exists():
        meta = json.loads(fmeta.read_text())
        cached_range = tuple(meta["lat_range"])
        if cached_range[0] <= lat_range[0] and lat_range[1] <= cached_range[1] and meta.get("bands") == saved_bands:
            log(f"L1 already built: {ftif_out} (covers {cached_range}, requested {lat_range})")
            return ftif_out, meta["scan0"], cached_range
        log(
            f"cached L1 {cached_range}/bands={meta.get('bands')} does not cover requested "
            f"{lat_range}/{bands} -- rebuilding"
        )
    import iirspy.utils as utils
    from iirspy import L0

    with phase("staging"):
        raw_qub = _stage_inputs(day, bands)

    # `chunk_y` sets both the dask y-chunk and, through `_row_block`, the write/resume block.
    with phase("L0 read + calibrate (graph build; the compute lands in the write)"):
        l1 = L0(SID, STAGE, chunk={"band": -1, "y": chunk_y, "x": -1}).calibrate(**{
            **CALIBRATE_KWS,
            **(calibrate_kwargs or {}),
        })
    # l1.img already holds exactly `bands` -- that's what _stage_inputs staged.
    _, xyext = utils.parse_geom(l1.csv, latlonextent=(-180, 180, *lat_range))
    ymin, ymax = xyext[2], xyext[3]
    log(f"L1 crop lat={lat_range}: scan {ymin}-{ymax} ({ymax - ymin} rows)")
    l1.img = l1.img.sel(y=slice(ymin, ymax))
    l1.img = l1.img / utils.RAD_NATIVE_SCALE
    if out_bands is not None:
        l1.img = l1.img.sel(band=out_bands)
    with phase(f"L1 write ({l1.img.shape[0]} band x {l1.img.shape[1]} rows, chunk_y={chunk_y})"):
        l1.save(str(ftif_out))
    fmeta.write_text(json.dumps({"scan0": int(ymin), "lat_range": list(lat_range), "bands": saved_bands}))
    raw_qub.unlink(missing_ok=True)
    raw_qub.with_suffix(".hdr").unlink(missing_ok=True)
    log(f"L1 saved: {ftif_out}")
    return ftif_out, int(ymin), lat_range


def _merged_gcps_paths(keep: str | None) -> list[Path]:
    """Where a previous run's merged GCPs could be, work dir first."""
    name = f"{SID}_{GROUP}.gcps"
    return [OUT / name, *([Path(keep) / name] if keep else [])]


def merged_gcps_path(sid: str, group: str) -> Path:
    """Where `sid`/`group`'s merged GCPs land once solved -- the durable `--keep` layout every
    cluster script and CLI default assumes, independent of any particular run's `--out`."""
    return ck.recal_dir(sid, group) / f"{sid}_{group}.gcps"


def already_solved(sid: str, group: str) -> bool:
    """Whether `sid`/`group` has a non-empty merged-GCPs file -- what a submission script checks
    before deciding a scene needs a solve job at all (job-count control belongs there, not in a
    job that's already been spawned)."""
    f = merged_gcps_path(sid, group)
    return f.is_file() and f.stat().st_size > 0


def keep_products(out: Path, dest: Path) -> list[Path]:
    """Copy the durable products in `out` (see `KEEP_GLOBS`) to `dest`. Returns what was copied."""
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for pattern in KEEP_GLOBS:
        for f in sorted(out.glob(pattern)):
            shutil.copyfile(f, dest / f.name)
            copied.append(f)
    return copied


def _scene_scan0(fgeom: Path, lat_range: tuple[float, float]) -> int:
    """First Scan of the group's L1 crop, from the geometry csv alone -- no cube needed."""
    import iirspy.utils as utils

    _, xyext = utils.parse_geom(fgeom, latlonextent=(-180, 180, *lat_range))
    return int(xyext[2])


def _make_glt(merged: dict, cfg0, scan0: int, glt_dir: str, overwrite: bool = False) -> Path:
    """Build the scene's GLT from the merged GCPs."""
    from rasterio.control import GroundControlPoint

    from iirspy import georef

    gcps = [GroundControlPoint(row=r, col=c, x=x, y=y) for (r, c), (x, y) in merged.items()]
    xs, ys = [g.x for g in gcps], [g.y for g in gcps]
    cfg = replace(cfg0, aoi=(min(xs), min(ys), max(xs), max(ys)))
    return georef.scene_glt(SID, GROUP, gcps, cfg, scan0, glt_dir, overwrite)


def _glt_only(args, cfg0, fgeom, lat_range) -> bool:
    """Already solved: build the GLT from the saved GCPs instead of restaging and re-solving."""
    fdone = next((f for f in _merged_gcps_paths(args.keep) if f.exists()), None)
    if fdone is None or args.resolve or args.chunks or args.hillshade_only:
        return False
    if fgeom is None:
        sys.exit(f"missing geometry csv for {SID}")
    merged = _load_gcps(fdone)
    glt_dir = args.keep or str(ck.recal_dir(SID, GROUP))
    f = _make_glt(merged, replace(cfg0, lat_band=lat_range), _scene_scan0(fgeom, lat_range), glt_dir)
    log(f"{fdone} exists ({len(merged)} gcps) -- skipped solve, glt: {f}")
    if args.clean:
        shutil.rmtree(OUT, ignore_errors=True)
        print(f"removed work dir {OUT}", flush=True)
    return True


def _warp_merged(merged: dict, used_chunks: list[dict], cfg0, ftif: Path):
    """Warp the band once through the merged GCPs, cropped to the union of the solved chunk AOIs."""
    from rasterio.control import GroundControlPoint

    from iirspy.georef import project, read_band, save_grid

    gcps = [
        GroundControlPoint(row=float(row), col=float(col), x=float(x), y=float(y))
        for (row, col), (x, y) in merged.items()
    ]
    final_cfg = replace(
        cfg0,
        aoi=(
            min(c["aoi"][0] for c in used_chunks),
            min(c["aoi"][1] for c in used_chunks),
            max(c["aoi"][2] for c in used_chunks),
            max(c["aoi"][3] for c in used_chunks),
        ),
    )
    final = project(read_band(ftif, final_cfg.band), gcps, final_cfg)
    ffinal = OUT / f"{SID}_{GROUP}_L1_b{final_cfg.band}.tif"
    save_grid(ffinal, final, final_cfg)
    log(f"final merged+warped: shape={final.shape} -> {ffinal}, aoi_km={[round(v / 1000, 1) for v in final_cfg.aoi]}")
    return final_cfg, final


def _blend_point(pa, pb, w_a: float):
    """One merged point from two owners' GCPs, falling back to whichever side has it.

    A tie point can be MAD-rejected in one chunk's solve but not the other -- don't assume both
    grids are dense at every (row, col). Returns (point, disagreement_m or None).
    """
    if pa is None and pb is None:
        return None, None
    if pa is None:
        return pb, None
    if pb is None:
        return pa, None
    xa, ya = pa
    xb, yb = pb
    return (w_a * xa + (1 - w_a) * xb, w_a * ya + (1 - w_a) * yb), float(np.hypot(xa - xb, ya - yb))


def _merge_gcps(results) -> tuple[dict, dict]:
    """Blend the per-chunk GCP fields into one, cosine cross-fading each row overlap.

    Returns the merged {(row, col): (x, y)} and a per-overlap summary of how far the two
    independent solves disagreed before blending.
    """
    used_chunks = [r["chunk"] for r in results]
    all_cols = sorted({col for r in results for (_row, col) in r["gcps"]})
    all_rows = sorted({row for r in results for (row, _col) in r["gcps"]})
    # chunk index does not track row direction -- some tracks scan pole-to-equator, others
    # equator-to-pole, so the bound has to come from the rows themselves, not chunks[-1].
    max_row = max(c["row1"] for c in used_chunks)
    merged_rows = [row for row in all_rows if row <= max_row]

    agree: dict[tuple[int, int], list[float]] = {}
    merged: dict[tuple[float, float], tuple[float, float]] = {}
    for row in merged_rows:
        owners = [r for r in results if r["chunk"]["row0"] <= row <= r["chunk"]["row1"]]
        if not owners:
            continue
        if len(owners) == 1:
            o = owners[0]
            for col in all_cols:
                pt = o["gcps"].get((row, col))
                if pt is not None:
                    merged[(row, col)] = pt
            continue
        owners.sort(key=lambda r: r["chunk"]["row0"])
        c_a, c_b = owners[0], owners[-1]
        row0_ov, row1_ov = c_b["chunk"]["row0"], c_a["chunk"]["row1"]
        t = (row - row0_ov) / max(row1_ov - row0_ov, 1e-6)
        w_a = _cross_fade(t)
        key = (c_a["chunk"]["i"], c_b["chunk"]["i"])
        for col in all_cols:
            pa, pb = c_a["gcps"].get((row, col)), c_b["gcps"].get((row, col))
            pt, dist = _blend_point(pa, pb, w_a)
            if pt is None:
                continue
            merged[(row, col)] = pt
            if dist is not None:
                agree.setdefault(key, []).append(dist)

    log("\noverlap agreement (independent chunk solves, before blending):")
    agree_summary = {}
    for (ai, bi), dists in agree.items():
        arr = np.array(dists)
        agree_summary[f"{ai}-{bi}"] = {
            "n": len(arr),
            "median_m": round(float(np.median(arr)), 1),
            "p95_m": round(float(np.percentile(arr, 95)), 1),
        }
        log(f"  chunk{ai}<->chunk{bi}: n={len(arr)} median={np.median(arr):.1f}m p95={np.percentile(arr, 95):.1f}m")

    return merged, agree_summary


def _solve_chunks(chunks, cfg0, ftif, fgeom, fspm, decay_m, tweaks, hillshade_only):
    """Solve each chunk against its own hillshade, reusing any prior good solve of the same shape.

    Returns one {"chunk", "gcps", "fit"} per solved chunk, empty when `hillshade_only`.

    Exits `EXIT_RESUME` if the next chunk would not fit in the time left (see `--max-seconds`).
    Stopping *before* a chunk rather than being killed inside one keeps every finished chunk's
    cache intact for the requeue; exiting here rather than returning short also means a partial set
    can never reach `_merge_gcps` and write a truncated `<sid>_<group>.gcps`, which `already_solved`
    would then read as a finished scene.
    """
    from iirspy.georef import project, read_band, register, render_reference, save_grid

    results = []
    for c in chunks:
        i = c["i"]
        if hillshade_only:
            cfg = replace(cfg0, aoi=c["aoi"], dem_near=c["dem_near"], dem_far=c["dem_far"])
            t0 = time.time()
            ref, hs_info = render_reference(fgeom, fspm, cfg, ck.kernels(SID[:8]))
            save_grid(OUT / f"chunk{i}_hs.tif", ref, cfg)
            log(
                f"chunk {i} [{c['band']}] hillshade {time.time() - t0:.1f}s shape={ref.shape} "
                f"near={Path(c['dem_near']).name} far={Path(c['dem_far']).name} "
                f"az={hs_info['az_grid']:.1f} elev={hs_info['elev']:.2f} lit={hs_info['lit_frac']:.3f}"
            )
            continue
        cached = _cached_chunk(OUT, c, ck.GOOD_CORR, decay_m, tweaks)
        if cached is not None:
            fit, gcps_by_rc = cached
            log(f"chunk {i} [{c['band']}]: reusing cached solve, corr={fit['corr']}")
            results.append({"chunk": c, "gcps": gcps_by_rc, "fit": fit})
            continue

        need_s = _band_estimate_s(c["band"], cfg0.niter)
        if _time_left() < need_s:
            log(
                f"chunk {i} [{c['band']}]: {_time_left() / 60:.0f} min left < ~{need_s / 60:.0f} min needed. "
                f"{len(results)}/{len(chunks)} chunks solved and cached; exiting {EXIT_RESUME} to resume."
            )
            sys.exit(EXIT_RESUME)

        cfg = replace(
            cfg0,
            aoi=c["aoi"],
            dem_near=c["dem_near"],
            dem_far=c["dem_far"],
            p95_stop_m=c["p95_stop_m"],
            decay_m=decay_m,
            gcp_rows=(c["row0"], c["row1"]),
            **tweaks,
        )
        log(
            f"\n--- chunk {i} [{c['band']}] solve: aoi={c['aoi']} far={Path(c['dem_far']).name} "
            f"decay_m={decay_m} {tweaks} ---"
        )

        t0 = time.time()
        ref, hs_info = render_reference(fgeom, fspm, cfg, ck.kernels(SID[:8]))
        hs_s = time.time() - t0
        save_grid(OUT / f"chunk{i}_hs.tif", ref, cfg)
        log(f"chunk {i}: hillshade {hs_s:.1f}s, shape {ref.shape}")

        with phase(f"chunk {i} register") as ph:
            try:
                reg = register(ftif, fgeom, fspm, cfg, reference=ref, verbose=True)
            except Exception as e:
                # corr=None keeps `_cached_chunk` from reusing the failure, so a re-run re-solves
                # this chunk; the merge just has no GCPs from its rows. Traceback to the log only.
                err = f"{type(e).__name__}: {e}"
                log(f"chunk {i}: FAILED -- {err}")
                log(traceback.format_exc())
                fit = {"chunk": i, "band": c["band"], "row0": c["row0"], "row1": c["row1"], "corr": None, "error": err}
                (OUT / f"chunk{i}_fit.json").write_text(json.dumps(fit, indent=1))
                continue
        solve_s = ph.wall

        band_i = read_band(ftif, cfg.band)
        after = project(band_i, reg.gcps, cfg)
        m = np.isfinite(after) & np.isfinite(ref) & (after > 0) & (ref > 0)
        corr = float(np.corrcoef(after[m], ref[m])[0, 1]) if m.sum() > 1000 else None
        save_grid(OUT / f"chunk{i}_final.tif", after, cfg)

        fit = {
            "chunk": i,
            "band": c["band"],
            "dem_near": c["dem_near"],
            "dem_far": c["dem_far"],
            "aoi": c["aoi"],
            "row0": c["row0"],
            "row1": c["row1"],
            "decay_m": decay_m,
            **tweaks,
            "hillshade_s": round(hs_s, 2),
            # The sun geometry the reference was rendered under: the first thing wanted when a
            # chunk's shadows look wrong.
            "az_grid": round(hs_info["az_grid"], 2),
            "elev": round(hs_info["elev"], 3),
            "lit_frac": round(hs_info["lit_frac"], 4),
            "solve_s": round(solve_s, 2),
            # Mean cores `register` kept busy, against `ncpu`: what a --cpus-per-task sweep reads.
            "solve_cores": round(ph.cores, 2),
            "ncpu": ncpu(),
            "corr": corr,
            "converged": reg.stats.get("converged"),
            "converged_reason": reg.stats.get("converged_reason"),
            "quality": reg.stats.get("quality"),
            "stats": reg.stats,
        }
        (OUT / f"chunk{i}_fit.json").write_text(json.dumps(fit, indent=1, default=str))
        q = reg.stats.get("quality", {})
        verdict = f"REJECTED ({q['reason']})" if q.get("rejected") else f"ok match_frac={q.get('match_frac')}"
        log(
            f"chunk {i}: {verdict}, corr={corr}, converged={reg.stats.get('converged')} "
            f"({reg.stats.get('converged_reason')}), solve={solve_s:.1f}s "
            f"[setup={reg.stats.get('setup_s')}s project={reg.stats.get('project_s')}s "
            f"coarse={reg.stats.get('coarse_s')}s, {len(reg.stats.get('iters', []))} iters]"
        )

        gcps_by_rc = {(g.row, g.col): (g.x, g.y) for g in reg.gcps}
        _save_gcps(OUT / f"chunk{i}.gcps", gcps_by_rc)
        results.append({"chunk": c, "gcps": gcps_by_rc, "fit": fit})

    return results


# Per-band solve-time model: {hillshade_s, overhead_s, iter_s} @ 4 cores/24GB SBATCH defaults.
# `overhead_s` is coarse_shift plus GCP-lattice setup, i.e. solve_s minus the sum of that chunk's
# own iter_s; it dominates for equatorial, whose coarse search over SLDEM costs ~2x LOLA's.
# north/north_midlat mirror south/south_midlat (same DEM tier structure) pending their own data.
BAND_TIMING: dict[str, dict[str, float]] = {
    "south": {"hillshade_s": 18.0, "overhead_s": 77.0, "iter_s": 43.0},
    "south_midlat": {"hillshade_s": 2.0, "overhead_s": 81.0, "iter_s": 57.0},
    "equatorial": {"hillshade_s": 3.0, "overhead_s": 156.0, "iter_s": 47.0},
    "north": {"hillshade_s": 18.0, "overhead_s": 77.0, "iter_s": 43.0},
    "north_midlat": {"hillshade_s": 2.0, "overhead_s": 81.0, "iter_s": 57.0},
}
# Iterations a plateau-converged chunk lands around; a placeholder until measured under the
# current convergence rule.
TYPICAL_NITER = 4


def _band_estimate_s(band: str, niter: int) -> float:
    """Solve time for one `band` chunk run to `niter` iterations, from `BAND_TIMING`.

    >>> round(_band_estimate_s("equatorial", 5))
    394
    """
    t = BAND_TIMING[band]
    return t["hillshade_s"] + t["overhead_s"] + niter * t["iter_s"]


def _plan_group_chunks(fgeom, scan0: int, ny: int, only: str | None, niter_max: int) -> list[dict]:
    """This group's chunks, with cube row bounds attached and optionally filtered to `only`.

    `only` is a comma-separated list of chunk indices. `niter_max` is `GeorefConfig.niter`, for the
    worst-case estimate.
    """
    all_chunks = ck.plan_chunks(fgeom, WIDTH_RANGE_KM, OVERLAP_FRAC)
    chunks = [c for c in all_chunks if c["group"] == GROUP]
    for c in chunks:
        c["row0"] = max(0, c["scan_lo"] - scan0)
        c["row1"] = min(ny - 1, c["scan_hi"] - scan0)
    if only:
        wanted = {int(x) for x in only.split(",")}
        # `i` is a global index across the whole strip, not per-group -- a chunk that exists but
        # belongs to another group would otherwise silently vanish here, leaving "0 chunks planned"
        # to look like a clean no-op run instead of a mistyped --chunks/--group pair.
        missing = wanted - {c["i"] for c in chunks}
        if missing:
            other_group = {c["i"]: c["group"] for c in all_chunks}
            detail = ", ".join(
                f"{i} (group={other_group[i]!r})" if i in other_group else f"{i} (no such chunk)"
                for i in sorted(missing)
            )
            sys.exit(f"--chunks {sorted(missing)} not in group {GROUP!r}: {detail}")
        chunks = [c for c in chunks if c["i"] in wanted]
    (OUT / "chunks.json").write_text(json.dumps(chunks, indent=1))
    log(f"{len(chunks)} chunks planned ({GROUP} group, {WIDTH_RANGE_KM[0]:.0f}-{WIDTH_RANGE_KM[1]:.0f}km):")
    for c in chunks:
        log(
            f"  chunk {c['i']} [{c['band']}]: rows {c['row0']}-{c['row1']}, "
            f"s {c['s0'] / 1000:.0f}-{c['s1'] / 1000:.0f}km (w={(c['s1'] - c['s0']) / 1000:.0f}km), "
            f"aoi_km={[round(v / 1000, 1) for v in c['aoi']]}"
        )
    bands_seen = list(dict.fromkeys(c["band"] for c in chunks))
    log("per-band summary:")
    typical_s = worst_s = 0.0
    for band in bands_seen:
        bc = [c for c in chunks if c["band"] == band]
        rows = sum(c["row1"] - c["row0"] + 1 for c in bc)
        band_typical = len(bc) * _band_estimate_s(band, TYPICAL_NITER)
        band_worst = len(bc) * _band_estimate_s(band, niter_max)
        typical_s += band_typical
        worst_s += band_worst
        log(
            f"  {band}: {len(bc)} chunk(s), {rows} rows, near={Path(bc[0]['dem_near']).name}, "
            f"far={Path(bc[0]['dem_far']).name}, ~{band_typical / 60:.0f} min typical / "
            f"~{band_worst / 60:.0f} min worst-case"
        )
    log(
        f"estimated solve time: ~{typical_s / 60:.0f} min typical ({TYPICAL_NITER} iters/chunk, "
        f"plateau-converged) / ~{worst_s / 60:.0f} min worst-case (every chunk hits the "
        f"{niter_max}-iter cap) for {len(chunks)} chunks"
    )
    return chunks


def _parser():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sid", help="scene id, e.g. 20201202T2319552644")
    ap.add_argument("--group", required=True, choices=list(ck.GROUPS))
    ap.add_argument("--out", default=None, help="work dir and restart cache (default runs/<sid>_<group>)")
    ap.add_argument(
        "--keep",
        default=None,
        help="copy GCPs, fits, summary, log and the GLT here when the run completes "
        "(default geometry/recalibrated/<day>/<sid>_<group> under IIRS_RECAL_ROOT)",
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
        help="dask y-chunk for the L1 read/calibrate/write, and so the write/resume block size "
        "(env IIRS_CHUNK_Y; default 1024 rows)",
    )
    ap.add_argument(
        "--gcp-row-margin",
        type=int,
        default=250,
        help="rows of GCP lattice kept either side of a chunk's own rows (GeorefConfig."
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
        default=0.0,
        help="drop tie points whose centre is closer than this to the swath's own nodata edge "
        "(GeorefConfig.edge_reject_m; 0 = off)",
    )
    ap.add_argument(
        "--shadow-max",
        type=float,
        default=1.0,
        help="drop tie points whose reference match window is more shadowed than this fraction "
        "(GeorefConfig.shadow_max; 1 = off)",
    )
    ap.add_argument(
        "--plateau-frac",
        type=float,
        default=0.02,
        help="stop when p95 improves by less than this fraction of the previous iteration's p95, "
        "independent of GeorefConfig.p95_stop_m (GeorefConfig.p95_plateau_frac; 0 = off)",
    )
    ap.add_argument(
        "--min-iter",
        type=int,
        default=0,
        help="never declare convergence before this iteration; mad_from_iter+1 (=4) guarantees one "
        "MAD-culled field is applied and measured (GeorefConfig.min_iter; 0 = off)",
    )
    ap.add_argument(
        "--edge-fit-k",
        type=int,
        default=0,
        help="off-support displacement fallback: fit a plane to the k nearest tie points instead "
        "of using the chunk-wide bulk median (GeorefConfig.edge_fit_k; 0 = off)",
    )
    ap.add_argument(
        "--edge-dense-m",
        type=float,
        default=0.0,
        help="second, denser tie-point pass within this far of the swath edge (GeorefConfig.edge_dense_m; 0 = off)",
    )
    return ap


def _log_provenance() -> None:
    """Slurm allocation + host details, `seff`-style, so a slow/failed run can be triaged later.

    Env vars are unset outside Slurm (e.g. a dev-box run), so every lookup falls back to "n/a" or
    the plain hostname/cpu count instead of raising.
    """
    import getpass
    import platform

    def _gb(mb: str) -> str:
        try:
            return f"{int(mb) / 1024:.0f}G"
        except ValueError:
            return "n/a"

    job_id = os.environ.get("SLURM_JOB_ID")
    array_job, array_task = os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    log("--- job ---")
    log(f"Job ID: {job_id or 'n/a'}")
    if array_job:
        log(f"Array Job ID: {array_job}_{array_task}")
    log(f"Cluster: {os.environ.get('SLURM_CLUSTER_NAME', 'n/a')}")
    log(f"User: {getpass.getuser()}")
    log(f"Node: {os.environ.get('SLURMD_NODENAME', platform.node())}")
    log(f"Cores per node: {os.environ.get('SLURM_CPUS_PER_TASK', ncpu())} (ncpu()={ncpu()})")
    log(f"Memory: {_gb(os.environ.get('SLURM_MEM_PER_NODE', ''))}/node")
    log(f"Python: {platform.python_version()}")


def _resolve_stage(out: Path) -> Path:
    """`IIRS_STAGE`, else `$SLURM_TMPDIR/stage`, else a `stage/` sibling of the work dir."""
    if os.environ.get("IIRS_STAGE"):
        return Path(os.environ["IIRS_STAGE"])
    if os.environ.get("SLURM_TMPDIR"):
        return Path(os.environ["SLURM_TMPDIR"]) / "stage"
    return out.parent / "stage"


def _open_run(out: str | None) -> None:
    """Locate the scene's zip and prepare the work dir, log and stage. Sets the module globals."""
    global ZIP, OUT, LOG, STAGE

    if not ck.ARCHIVE.exists():
        sys.exit(f"{ck.ARCHIVE} missing -- archive not mounted, set IIRS_ARCHIVE")
    zip_path = ck.nri_zip(SID)
    if zip_path is None:
        sys.exit(f"no nri zip for {SID} under {ck.ARCHIVE / 'zips'}")
    ZIP = zip_path
    OUT = Path(out) if out else Path.cwd() / "runs" / f"{SID}_{GROUP}"
    LOG = OUT / "run.log"
    OUT.mkdir(parents=True, exist_ok=True)
    STAGE = _resolve_stage(OUT)
    STAGE.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")


def main(argv: list[str] | None = None) -> None:

    from iirspy.georef import read_band

    args = _parser().parse_args(argv)

    global SID, GROUP, DEADLINE
    SID, GROUP = args.sid, args.group
    DEADLINE = time.monotonic() + args.max_seconds if args.max_seconds else None

    # Bind dask's threadpool to the allocation before anything touches a chunked array (see `ncpu`).
    import dask

    dask.config.set(scheduler="threads", num_workers=ncpu())

    _open_run(args.out)
    t_start = time.time()
    _log_provenance()
    log(f"{SID} group={GROUP} zip={ZIP.name} ncpu={ncpu()} stage={STAGE}")

    cfg0 = ck.chunk_cfg(GROUP)
    anc = ck.ancillary(SID)
    fgeom, fspm = anc["geometry/calibrated"], anc["miscellaneous/raw"]

    lat_range0 = ck.l1_lat_range(GROUP)
    if _glt_only(args, cfg0, fgeom, lat_range0):
        return

    # The crop carries a buffer past the solved bands; the chunk plan does not.
    # A chunk solve only ever reads `cfg0.band` back out, so calibration only masks/interpolates
    # its own +-6 window (`output_bands`), not every PAN_BANDS the empirical search also staged;
    # `out_bands` then narrows what's actually written to just that one band.
    ftif, scan0, lat_range = build_l1(
        lat_range0,
        _cube_bands(cfg0.band),
        OUT / f"{SID}_l1_{GROUP}_group.tif",
        calibrate_kwargs={"output_bands": _band_window(cfg0.band)},
        out_bands=[cfg0.band],
        chunk_y=args.chunk_y,
    )
    cfg0 = replace(cfg0, lat_band=lat_range)

    # The spm rides inside the nri zip, so a scene whose ancillary was never synced to the archive
    # still has one under STAGE once build_l1 has extracted it.
    if fspm is None:
        fspm = next(STAGE.glob(f"miscellaneous/raw/{SID[:8]}/*{SID}*.spm"), None)
    if fgeom is None or fspm is None:
        sys.exit(f"missing ancillary for {SID}: geometry={fgeom} spm={fspm}")

    band = read_band(ftif, cfg0.band)
    ny, nx = band.shape
    del band
    log(f"L1 cube ({GROUP} group, lat {lat_range}): {ny} rows x {nx} cols, scan0={scan0}")

    chunks = _plan_group_chunks(fgeom, scan0, ny, args.chunks, cfg0.niter)

    decay_m = args.decay_m if args.decay_m is not None else cfg0.decay_m
    tweaks = {
        "edge_reject_m": args.edge_reject_m,
        "shadow_max": args.shadow_max,
        "edge_dense_m": args.edge_dense_m,
        "edge_fit_k": args.edge_fit_k,
        "p95_plateau_frac": args.plateau_frac,
        "min_iter": args.min_iter,
        "gcp_row_margin": args.gcp_row_margin,
    }

    results = _solve_chunks(chunks, cfg0, ftif, fgeom, fspm, decay_m, tweaks, args.hillshade_only)

    if args.hillshade_only:
        log(f"\n--hillshade-only: wrote {len(chunks)} hillshade(s), no solve.")
        return

    if args.chunks:
        # A --chunks subset is a before/after comparison, not a strip solve: its chunks are
        # typically non-adjacent, and merge/warp cost scales with the union AOI regardless of how
        # few chunks fed it.
        log(f"\n--chunks given: solved {len(results)} chunk(s), skipping merge/warp.")
        return

    if len(results) < 2:
        sys.exit("fewer than 2 chunks solved -- nothing to merge")

    results.sort(key=lambda r: r["chunk"]["i"])
    used_chunks = [r["chunk"] for r in results]
    merged, agree_summary = _merge_gcps(results)

    fgcps = OUT / f"{SID}_{GROUP}.gcps"
    _save_gcps(fgcps, merged)
    log(f"\nmerged gcps: {len(merged)} points -> {fgcps}")

    final_cfg, final = _warp_merged(merged, used_chunks, cfg0, ftif)
    fglt = _make_glt(merged, cfg0, scan0, args.keep or str(ck.recal_dir(SID, GROUP)), overwrite=True)
    log(f"glt: {fglt}")

    summary = {
        "sid": SID,
        "group": GROUP,
        "lat_range": lat_range,
        "solve_lat_range": ck.group_lat_range(GROUP),
        "width_range_km": WIDTH_RANGE_KM,
        "overlap_frac": OVERLAP_FRAC,
        "n_chunks": len(chunks),
        "n_chunks_solved": len(results),
        "chunks": chunks,
        "per_chunk_fit": [r["fit"] for r in results],
        "overlap_agreement": agree_summary,
        "n_merged_gcps": len(merged),
        "scan0": scan0,
        "glt": str(fglt),
        "final_aoi_m": final_cfg.aoi,
        "final_shape": list(final.shape),
        "total_s": round(time.time() - t_start, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"\ndone in {summary['total_s']}s.")

    if args.keep:
        dest = Path(args.keep)
        copied = keep_products(OUT, dest)
        log(f"kept {len(copied)} file(s) -> {dest}")
        if args.clean:
            shutil.rmtree(OUT, ignore_errors=True)
            print(f"removed work dir {OUT}", flush=True)
    elif args.clean:
        sys.exit("--clean without --keep would discard the whole run")


if __name__ == "__main__":
    main()
