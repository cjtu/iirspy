"""Chunked AROSICS registration of one CRS group of one scene.

Splits an L1 product cropped to the group's latitudes into chunks via `iirspy.georef.chunks.plan_chunks`, solves each
chunk against its own bounded hillshade, then merges the chunks with a cosine cross-fade over each row overlap and
warps the merged result.

The work dir (`Workspace.dir`) doubles as the restart cache: a re-run reuses any chunk already solved at the same
shape, so a rerun only re-solves what it did not finish. A step that would not finish inside `Workspace.deadline`
raises `DeadlineReached` before starting.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from iirspy.georef import ancillary
from iirspy.georef import chunks as ck
from iirspy.io import IIRSPY_VERSION
from iirspy.l1.frames import table_key as frames_key
from iirspy.workspace import Workspace, ncpu, phase

WIDTH_RANGE_KM = (100.0, 150.0)
OVERLAP_FRAC = 0.10
# How far a chunk's own coarse shift may sit from the strip consensus before it is re-solved seeded
# from it. Measured over 9 strips the true correction moves <=1.4 km chunk-to-chunk while a spurious
# correlation peak jumps 6-28 km, and 2 km stays under the 2.0 km iteration-1 capture radius, so
# refinement absorbs the residual a reseeded chunk is left with.
CONSENSUS_TOL_M = 2000.0
# Grades a merged product from the worst overlap-boundary p95 (see `solve_grade`).
GRADE_ACCEPTABLE_M = 500.0
# The consensus is two lines in along-track distance, one per component of the shift in the chunk's
# (along-track, cross-track) frame: line timing and pointing have independent causes. Over 91
# a3_9regions run-2 strips the accepted chunks drift |slope| p90 1.0 m/km on both components; cross-track
# never exceeds 5.6 m/km, along-track reaches 8.5 (20210103T0245) and 45 (20201226T1745 south, a line-
# timing drift that walks its equatorial group out of coarse capture entirely).
CONSENSUS_MAX_DRIFT_ALONG = 0.060
CONSENSUS_MAX_DRIFT_CROSS = 0.012
# Anchors per local line; beyond the outermost inlier the line is held flat after this many chunk spans.
CONSENSUS_LOCAL_K = 5
CONSENSUS_HOLD_CHUNKS = 2.0


def _cross_fade(t: float) -> float:
    """Weight for the earlier of two overlapping chunks: 1 at the overlap's start, 0 at its end.

    >>> [float(_cross_fade(t)) for t in (0.0, 0.5, 1.0)]
    [1.0, 0.5, 0.0]
    >>> [float(_cross_fade(t)) for t in (-1.0, 2.0)]  # clipped outside [0, 1]
    [1.0, 0.0]
    """
    return float(0.5 * (1 + np.cos(np.pi * np.clip(t, 0, 1))))


def save_gcps(path: Path, gcps_by_rc: dict) -> None:
    lines = ["row,col,x,y\n"] + [f"{r},{c},{x},{y}\n" for (r, c), (x, y) in sorted(gcps_by_rc.items())]
    path.write_text("".join(lines))


def load_gcps(path: Path) -> dict:
    gcps_by_rc = {}
    for line in path.read_text().splitlines()[1:]:
        r, c, x, y = (float(v) for v in line.split(","))
        gcps_by_rc[(r, c)] = (x, y)
    return gcps_by_rc


# How a fit json written before one of these knobs existed must be read, so an old cache is only
# reused by a run that also leaves them off.
TWEAK_DEFAULTS = {
    "edge_reject_m": 0.0,
    "edge_reject_axis": "all",
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


def _cached_chunk(
    ws: Workspace, sid: str, group: str, frames: Path, c: dict, good_corr: float, decay_m: float, tweaks: dict
):
    """The prior solve of chunk `c['i']` if it is still valid, else None.

    Valid means its band, DEM pair, row bounds, aoi, decay_m, frame table, tweaks and the iirspy version that
    wrote it all match `c` and it wasn't rejected, at corr >= `good_corr`. The version is in the key
    because none of the other fields say anything about the *code* that solved the chunk, so without
    it a rerun after an algorithm change mixes old and new solves in one
    product. `"0+source"` (not installed as a distribution, so the version cannot invalidate
    anything) always misses; the cluster venv picks up a bump when iirspy is reinstalled into it.

    Convergence isn't required -- a non-converged chunk that still passed quality (5-iter cap, no
    plateau) re-solves to the same result every time, so treating it as cache-invalid just burns
    wall-clock re-deriving an answer that won't change.
    The DEM pair is part of the key (by product name, see `_dem_key`) because `plan_chunks` picks
    the far tier per chunk, so two runs can agree on shape and still have used different
    references. Returns (fit, gcps_by_rc).
    """
    ffit, fgcp = ws.path(f"{sid}_{group}_chunk{c['i']}_fit.json"), ws.path(f"{sid}_{group}_chunk{c['i']}.gcps")
    if not (ffit.exists() and fgcp.exists()):
        return None
    fit = json.loads(ffit.read_text())
    same = (
        IIRSPY_VERSION != "0+source"
        and fit.get("version") == IIRSPY_VERSION
        and fit.get("band") == c["band"]
        and _dem_key(fit.get("dem_near", "")) == _dem_key(c["dem_near"])
        and _dem_key(fit.get("dem_far", "")) == _dem_key(c["dem_far"])
        and fit.get("row0") == c["row0"]
        and fit.get("row1") == c["row1"]
        and fit.get("decay_m") == decay_m
        and fit.get("frames") == frames_key(frames)
        and all(fit.get(k, TWEAK_DEFAULTS[k]) == v for k, v in tweaks.items())
        and all(abs(a - b) < 1.0 for a, b in zip(fit.get("aoi", []), c["aoi"], strict=True))
    )
    good = fit.get("corr") is not None and fit["corr"] >= good_corr and not fit.get("quality", {}).get("rejected")
    if same and good:
        return fit, load_gcps(fgcp)
    return None


def scene_scan0(fgeom: Path, lat_range: tuple[float, float]) -> int:
    """First Scan of the group's L1 crop, from the geometry csv alone -- no cube needed."""

    _, xyext = ancillary.parse_geom(fgeom, latlonextent=(-180, 180, *lat_range))
    return int(xyext[2])


def gcp_list(merged: dict) -> list:
    """The merged `{(row, col): (x, y)}` mapping as the GCP list rasterio and the VRT writer want."""
    from rasterio.control import GroundControlPoint

    return [
        GroundControlPoint(row=float(row), col=float(col), x=float(x), y=float(y))
        for (row, col), (x, y) in merged.items()
    ]


def make_glt(sid: str, group: str, merged: dict, cfg0, scan0: int, glt_dir: str, overwrite: bool = False) -> Path:
    """Build the scene's GLT from the merged GCPs."""
    from iirspy import georef

    gcps = gcp_list(merged)
    xs, ys = [g.x for g in gcps], [g.y for g in gcps]
    cfg = replace(cfg0, aoi=(min(xs), min(ys), max(xs), max(ys)))
    return georef.scene_glt(sid, group, gcps, cfg, scan0, glt_dir, overwrite)


def _peak_mb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def make_loc(
    ws: Workspace,
    sid: str,
    group: str,
    dem_roots: tuple[Path, ...],
    merged: dict,
    scan0: int,
    loc_dir: str,
    fits: list[dict],
    overwrite: bool = False,
) -> Path:
    """Write `<sid>_<group>_loc.tif` (lon, lat, LOLA radius per camera pixel) from the merged GCP lattice."""
    from iirspy.georef import backplanes

    floc = Path(loc_dir) / f"{sid}_{group}_loc.tif"
    if floc.exists() and not overwrite:
        return floc
    peak0 = _peak_mb()
    with phase(ws, "loc"):
        lon, lat, radius = backplanes.group_loc_core(merged, group, dem_roots)
        loc = {"lon": lon, "lat": lat, "radius": radius, "groups": {group: (scan0, scan0 + lon.shape[0] - 1)}}
        backplanes.write_loc(floc, loc, sid, row0=scan0, fits=fits, dem_roots=dem_roots)
    ws.log(f"loc: {floc} {lon.shape}, peak RSS {peak0:.0f} -> {_peak_mb():.0f} MB")
    return floc


GRADE_OF = {"tight": "pass", "acceptable": "warn", "failed": "fail", None: "warn"}


def geometry(
    *,
    sid: str,
    group: str,
    merged: dict,
    row0: int,
    shape: tuple[int, int],
    paths,
    cfg,
    fgeom,
    flabel,
    kernels,
    fl1=None,
    stage_dir=None,
    fgcps=None,
    topo_dir=None,
    ws=None,
):
    """`Geometry` of one solved group from its merged GCP lattice: LOC from the lattice + LOLA radius, topo
    (when `topo_dir` is given) and SPICE sun/sensor terms from that LOC, QA from the lattice and the solve summary
    beside `fgcps`. `shape` is the (ny, nx) of the L1 crop starting at absolute Scan `row0`; `fl1` the L1 cube
    for SNR and the PAN-incomplete flag."""
    from iirspy.georef import backplanes as bp
    from iirspy.georef.geometry import Geometry, quality
    from iirspy.georef.georef import scene_topo

    t_start = time.time()
    ny, nx = shape
    lon, lat, radius = bp.group_loc_core(merged, group, paths.dem_roots, nrow=ny, ncol=nx)
    loc = {"lon": lon, "lat": lat, "radius": radius}
    ftopo = (
        scene_topo(sid, group, lon, lat, fgeom, flabel, cfg, kernels, topo_dir, paths.dem_roots, scan0=row0)
        if topo_dir
        else None
    )
    obs = bp.obs_from_loc(sid, group, loc, row0, merged, paths, ftopo, fl1, stage_dir)

    fsum = Path(fgcps).with_name(f"{Path(fgcps).stem}_solve_summary.json") if fgcps else None
    summary = json.loads(fsum.read_text()) if fsum is not None and fsum.exists() else {}
    qa = bp.build_qa(
        loc,
        obs,
        merged,
        bp.chunk_overlaps(summary.get("chunks", [])),
        bp.pan_incomplete(fl1) if fl1 else None,
    )
    if ftopo is None:
        qa |= bp.qa_mask("no_topo")

    grade = summary.get("solve_grade")
    agree = [v["p95_m"] for v in summary.get("overlap_agreement", {}).values()]
    tie = bp._tie_point_residual(summary.get("per_chunk_fit", []))
    sidecars = {"gcps": fgcps, "summary": fsum if summary else None, "topo": ftopo}
    return Geometry.from_arrays(
        sid,
        group,
        row0,
        "arosics",
        loc,
        obs["bands"],
        qa,
        quality("arosics", GRADE_OF[grade], max(agree) if agree else None),
        sidecars={k: Path(v) for k, v in sidecars.items() if v},
        provenance=[
            {
                "step": "georef",
                "method": "arosics",
                "version": IIRSPY_VERSION,
                "inputs": {"gcps": str(fgcps), "solve_grade": grade, "n_gcps": len(merged)},
                "wall_s": round(time.time() - t_start, 1),
            }
        ],
        tags={
            "SENSOR_SOURCE": obs["sensor_source"],
            "TIE_POINT_RESIDUAL_P50_M": tie["p50_m"] if tie else "unavailable",
            "TIE_POINT_RESIDUAL_P95_M": tie["p95_m"] if tie else "unavailable",
            "PROVENANCE_DEM_ROOTS": ":".join(str(p) for p in paths.dem_roots),
        },
    )


def warp_merged(ws: Workspace, sid: str, group: str, merged: dict, used_chunks: list[dict], cfg0, ftif: Path):
    """Warp the band once through the merged GCPs, cropped to the union of the solved chunk AOIs."""
    from iirspy.georef.georef import project, read_band, save_grid

    gcps = gcp_list(merged)
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
    ffinal = ws.path(f"{sid}_{group}_L1_b{final_cfg.band}.tif")
    save_grid(ffinal, final, final_cfg)
    ws.log(
        f"final merged+warped: shape={final.shape} -> {ffinal}, aoi_km={[round(v / 1000, 1) for v in final_cfg.aoi]}"
    )
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


def merge_gcps(ws: Workspace, results) -> tuple[dict, dict]:
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

    ws.log("\noverlap agreement (independent chunk solves, before blending):")
    agree_summary = {}
    for (ai, bi), dists in agree.items():
        arr = np.array(dists)
        agree_summary[f"{ai}-{bi}"] = {
            "n": len(arr),
            "median_m": round(float(np.median(arr)), 1),
            "p95_m": round(float(np.percentile(arr, 95)), 1),
        }
        ws.log(f"  chunk{ai}<->chunk{bi}: n={len(arr)} median={np.median(arr):.1f}m p95={np.percentile(arr, 95):.1f}m")

    return merged, agree_summary


def solve_grade(agree_summary, p95_stop_m: float):
    """Grade the merged product from the worst chunk-boundary p95 in `agree_summary`.

    The per-chunk `rejected`/`converged` flags do not track whether the product landed -- every
    human-verified-good scene has rejected chunks too -- while this does: over 11 verified scenes
    the worst good scene sat at 383 m and the best bad one at 6487 m.

    >>> [solve_grade({"0-1": {"p95_m": v}}, 16.5) for v in (12.0, 382.9, 6486.6)]
    ['tight', 'acceptable', 'failed']
    >>> solve_grade({}, 16.5) is None
    True
    """
    worst = max((v["p95_m"] for v in agree_summary.values()), default=None)
    if worst is None:
        return None
    if worst <= p95_stop_m:
        return "tight"
    return "acceptable" if worst <= GRADE_ACCEPTABLE_M else "failed"


def _accepted(r):
    """Whether a `_solve_one` result solved cleanly (not None, not quality-rejected)."""
    return r is not None and not r["fit"].get("quality", {}).get("rejected")


def _shift_of(r):
    """A solved chunk's total coarse shift in metres, or (0, 0) if `r` is None."""
    return tuple(r["fit"]["stats"]["coarse"].get("total_shift_m", (0.0, 0.0))) if r else (0.0, 0.0)


def _is_anchor(r, cfg0) -> bool:
    """Whether `r` registered well enough to vote on the strip consensus: match rate alone.

    How the chunk arrived at its shift does not matter, only that the shift it ended up with
    *registers*. A chunk that refused its coarse peak and kept its seed (`coarse.accepted=False`)
    and then reached `match_frac=0.21` has corroborated that seed as well as anything can; requiring
    `coarse.accepted` too left both dead E1 strips with no anchor at all, because there the only
    accepted chunks were the mis-latched ones. Those are excluded by `min_match_frac` regardless --
    a chunk kilometres off keeps 0.003-0.017 of its tie-point candidates.

    >>> from dataclasses import replace
    >>> from iirspy.georef import GeorefConfig
    >>> cfg = replace(GeorefConfig(), min_match_frac=0.10)
    >>> _is_anchor({"fit": {"quality": {"match_frac": 0.167}}}, cfg)  # refused its peak, matched
    True
    >>> _is_anchor({"fit": {"quality": {"match_frac": 0.004}}}, cfg)  # accepted a 15 km peak
    False
    >>> _is_anchor(None, cfg)
    False
    """
    if r is None:
        return False
    frac = (r["fit"].get("quality") or {}).get("match_frac")
    return frac is not None and frac >= cfg0.min_match_frac


def _consensus_model(anchors, hold_m: float, max_drift: float, tol_m: float = CONSENSUS_TOL_M):
    """One shift component as a function of along-track distance, from `anchors` = [(s_m, v_m, weight)].

    Inliers: the line (through an anchor pair, or flat through one anchor) with |slope| <= `max_drift`
    that the most anchor weight lies within `tol_m` of. Then, per query, a weighted line through the
    `CONSENSUS_LOCAL_K` nearest inliers (slope clamped to `max_drift`; flat below 3 inliers), held flat
    beyond `hold_m` past the outermost inlier. Returns (f(s) -> v, inlier mask), or None when there
    is nothing to vote.

    >>> f, inl = _consensus_model([(0, 0, 1), (1e5, 500, 1), (2e5, 1000, 1), (3e5, 15000, 1)], 2e5, 0.012)
    >>> inl.tolist(), round(f(3e5)), round(f(9e5))
    ([True, True, True, False], 1500, 2000)
    >>> _consensus_model([], 1.0, 0.012) is None
    True
    """
    if not anchors:
        return None
    s, v, w = np.asarray(anchors, float).T
    best, inliers = None, None
    for i in range(len(s)):
        for j in range(i, len(s)):
            ds = s[j] - s[i]
            if j > i and ds == 0:
                continue
            slope = (v[j] - v[i]) / ds if j > i else 0.0
            if abs(slope) > max_drift:
                continue
            dev = np.abs(v - v[i] - (s - s[i]) * slope)
            inl = dev <= tol_m
            score = (w[inl].sum(), -dev[inl].sum())
            if best is None or score > best:
                best, inliers = score, inl
    si, vi, wi = s[inliers], v[inliers], w[inliers]

    def f(q):
        q = float(np.clip(q, si.min() - hold_m, si.max() + hold_m))
        near = np.argsort(np.abs(si - q))[:CONSENSUS_LOCAL_K]
        sn, vn, wn = si[near], vi[near], wi[near]
        if len(sn) < 3 or np.ptp(sn) == 0:  # two points are a slope of pure noise; hold flat
            return float(np.average(vn, weights=wn))
        m = float(np.clip(np.polyfit(sn, vn, 1, w=np.sqrt(wn))[0], -max_drift, max_drift))
        return float(np.average(vn - m * sn, weights=wn) + m * q)

    return f, inliers


def _frame(c) -> np.ndarray:
    """Rows: chunk `c`'s unit along-track `t` and cross-track `n` (t rotated +90 deg), in its grid CRS."""
    t = np.asarray(c["t"], float)
    return np.array([t, [-t[1], t[0]]])


def _dev_m(shift, consensus) -> float:
    """Distance in metres from a chunk's own coarse shift to the strip consensus."""
    return float(np.hypot(shift[0] - consensus[0], shift[1] - consensus[1]))


def _chunk_grade(r) -> str:
    """How much a chunk's own solve can be trusted: `ok`, `fallback` (refused the coarse shift, so
    it kept its seed rigidly -- safe), or `suspect` (accepted a coarse shift and was then rejected
    on match rate -- the case that lands a chunk kilometres off)."""
    if _accepted(r):
        return "ok"
    return "suspect" if r["fit"].get("stats", {}).get("coarse", {}).get("accepted") else "fallback"


def _solve_one(ws, sid, group, frames, kernels, c, seed, cfg0, ftif, fgeom, flabel, decay_m, tweaks, pin_coarse=False):
    """Solve chunk `c` seeded from `seed` (a prior coarse shift in metres), or reuse its cache.

    `pin_coarse` holds the chunk at `seed` exactly: `coarse_max_m=0` leaves the coarse search one
    zero-lag bin to pick from, so it adds nothing and `total_shift_m == seed`, while the fine
    tie-point iterations still run. The cache is keyed on everything but the seed, so a pinned
    re-solve must not read it -- it would hand back the very fit the reseed exists to replace.

    Returns {"chunk", "gcps", "fit"}, or None on total failure. Raises `DeadlineReached` if this chunk
    would not fit in the time left (`ws.deadline`) -- stopping *before* a chunk rather than
    being killed inside one keeps every finished chunk's cache intact for the rerun.
    """
    from iirspy.georef.georef import (
        project,
        read_band,
        register,
        render_reference,
        render_wac,
        save_grid,
        sun_geometry,
        use_wac,
    )

    i = c["i"]
    cached = None if pin_coarse else _cached_chunk(ws, sid, group, frames, c, ck.GOOD_CORR, decay_m, tweaks)
    if cached is not None:
        fit, gcps_by_rc = cached
        ws.log(f"chunk {i} [{c['band']}]: reusing cached solve, corr={fit['corr']}")
        return {"chunk": c, "gcps": gcps_by_rc, "fit": fit}

    need_s = _band_estimate_s(c["band"], cfg0.niter)
    ws.check_deadline(need_s, f"chunk {i} [{c['band']}]")

    cfg = replace(
        cfg0,
        aoi=c["aoi"],
        dem_near=c["dem_near"],
        dem_far=c["dem_far"],
        p95_stop_m=c["p95_stop_m"],
        decay_m=decay_m,
        gcp_rows=(c["row0"], c["row1"]),
        seed_m=seed,
        **({"coarse_max_m": 0.0} if pin_coarse else {}),
        **tweaks,
    )
    ws.log(
        f"\n--- chunk {i} [{c['band']}] solve: aoi={c['aoi']} far={Path(c['dem_far']).name} "
        f"decay_m={decay_m} seed_m={seed}{' PINNED' if pin_coarse else ''} {tweaks} ---"
    )

    t0 = time.time()
    # Which reference is decided by the sun, not by trying one and detecting failure: above
    # `georef.WAC_MIN_ELEV_DEG` there is no shadow contrast for a hillshade to carry (0 of 16 chunks
    # above 70 deg ever matched one) and the scene reads as albedo, which is what WAC shows.
    _, elev, _ = sun_geometry(fgeom, flabel, cfg, kernels)
    wac = use_wac(elev, cfg)
    hit = _cached_reference(ws, sid, group, c, "wac" if wac else "hillshade")
    if hit is None:
        ref, hs_info = (render_wac if wac else render_reference)(fgeom, flabel, cfg, kernels)
        _save_reference(ws, sid, group, c, cfg, ref, hs_info, "wac" if wac else "hillshade")
    else:
        ref, hs_info = hit
    hs_s = time.time() - t0
    ws.log(f"chunk {i}: {'wac' if wac else 'hillshade'} ref {hs_s:.1f}s{' (cached)' if hit else ''}, shape {ref.shape}")

    with phase(ws, f"chunk {i} register") as ph:
        try:
            reg = register(ftif, fgeom, flabel, cfg, reference=ref, verbose=True)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            ws.log(f"chunk {i}: FAILED -- {err}")
            ws.log(traceback.format_exc())
            fit = {"chunk": i, "band": c["band"], "row0": c["row0"], "row1": c["row1"], "corr": None, "error": err}
            ws.path(f"{sid}_{group}_chunk{i}_fit.json").write_text(json.dumps(fit, indent=1))
            return None
    solve_s = ph.wall

    band_i = read_band(ftif, cfg.band)
    after = project(band_i, reg.gcps, cfg)
    m = np.isfinite(after) & np.isfinite(ref) & (after > 0) & (ref > 0)
    corr = float(np.corrcoef(after[m], ref[m])[0, 1]) if m.sum() > 1000 else None
    save_grid(ws.path(f"{sid}_{group}_chunk{i}_final.tif"), after, cfg)

    fit = {
        "chunk": i,
        "version": IIRSPY_VERSION,
        "reference": "wac" if wac else "hillshade",
        "band": c["band"],
        "dem_near": c["dem_near"],
        "dem_far": c["dem_far"],
        "aoi": c["aoi"],
        "row0": c["row0"],
        "row1": c["row1"],
        "decay_m": decay_m,
        "frames": frames_key(frames),
        **tweaks,
        "hillshade_s": round(hs_s, 2),
        "az_grid": round(hs_info["az_grid"], 2),
        "elev": round(hs_info["elev"], 3),
        "lit_frac": round(hs_info["lit_frac"], 4),
        "solve_s": round(solve_s, 2),
        "solve_cores": round(ph.cores, 2),
        "ncpu": ncpu(),
        "corr": corr,
        "converged": reg.stats.get("converged"),
        "converged_reason": reg.stats.get("converged_reason"),
        "quality": reg.stats.get("quality"),
        "stats": reg.stats,
    }
    ws.path(f"{sid}_{group}_chunk{i}_fit.json").write_text(json.dumps(fit, indent=1, default=str))
    q = reg.stats.get("quality", {})
    verdict = f"REJECTED ({q['reason']})" if q.get("rejected") else f"ok match_frac={q.get('match_frac')}"
    ws.log(
        f"chunk {i}: {verdict}, corr={corr}, converged={reg.stats.get('converged')} "
        f"({reg.stats.get('converged_reason')}), solve={solve_s:.1f}s "
        f"[setup={reg.stats.get('setup_s')}s project={reg.stats.get('project_s')}s "
        f"coarse={reg.stats.get('coarse_s')}s, {len(reg.stats.get('iters', []))} iters]"
    )

    gcps_by_rc = {(g.row, g.col): (g.x, g.y) for g in reg.gcps}
    save_gcps(ws.path(f"{sid}_{group}_chunk{i}.gcps"), gcps_by_rc)
    return {"chunk": c, "gcps": gcps_by_rc, "fit": fit}


def _reference_key(c: dict, reference: str) -> dict:
    return {
        "reference": reference,
        "aoi": list(c["aoi"]),
        "dem_near": _dem_key(c["dem_near"]),
        "dem_far": _dem_key(c["dem_far"]),
    }


def _save_reference(ws: Workspace, sid: str, group: str, c: dict, cfg, ref, hs_info: dict, reference: str) -> None:
    """Write chunk `c`'s reference image and the key `_cached_reference` reads it back by."""
    from iirspy.georef.georef import save_grid

    save_grid(ws.path(f"{sid}_{group}_chunk{c['i']}_hs.tif"), ref, cfg)
    info = {k: hs_info[k] for k in ("az_grid", "elev", "lit_frac")}
    ws.path(f"{sid}_{group}_chunk{c['i']}_hs.json").write_text(json.dumps({**_reference_key(c, reference), **info}))


def _cached_reference(ws: Workspace, sid: str, group: str, c: dict, reference: str):
    """(image, info) of chunk `c`'s reference from an earlier run in this work dir, or None.

    A reference depends on the DEMs, the sun and the chunk's grid, never on the cube, so a re-solve on a re-timed
    cube or a new iirspy version reads it back. Its key is the `_hs.json` sidecar, else (runs before the sidecar)
    the chunk's fit, which records the same fields.
    """
    import rasterio

    fhs = ws.path(f"{sid}_{group}_chunk{c['i']}_hs.tif")
    finfo = next(
        (
            f
            for f in (ws.path(f"{sid}_{group}_chunk{c['i']}_hs.json"), ws.path(f"{sid}_{group}_chunk{c['i']}_fit.json"))
            if f.exists()
        ),
        None,
    )
    if not fhs.exists() or finfo is None:
        return None
    info = json.loads(finfo.read_text())
    seen = {**info, "dem_near": _dem_key(info.get("dem_near", "")), "dem_far": _dem_key(info.get("dem_far", ""))}
    key = _reference_key(c, reference)
    if "az_grid" not in info or any(seen.get(k) != v for k, v in key.items()):
        return None
    with rasterio.open(fhs) as src:
        return src.read(1), info


def _solve_chunks_hillshade(ws, sid, group, kernels, chunks, cfg0, fgeom, flabel):
    """Render and save each chunk's reference hillshade only, no registration."""
    from iirspy.georef.georef import render_reference

    for c in chunks:
        cfg = replace(cfg0, aoi=c["aoi"], dem_near=c["dem_near"], dem_far=c["dem_far"])
        t0 = time.time()
        ref, hs_info = render_reference(fgeom, flabel, cfg, kernels)
        _save_reference(ws, sid, group, c, cfg, ref, hs_info, "hillshade")
        ws.log(
            f"chunk {c['i']} [{c['band']}] hillshade {time.time() - t0:.1f}s shape={ref.shape} "
            f"near={Path(c['dem_near']).name} far={Path(c['dem_far']).name} "
            f"az={hs_info['az_grid']:.1f} elev={hs_info['elev']:.2f} lit={hs_info['lit_frac']:.3f}"
        )


def _s_mid(c) -> float:
    """A chunk's along-track position [m]: the centre of its planned `s0`..`s1` span."""
    return 0.5 * float(c["s0"] + c["s1"])


def _consensus_pass(ws, sid, group, frames, kernels, chunks, results, cfg0, ftif, fgeom, flabel, decay_m, tweaks):
    """Re-solve every chunk that disagrees with the strip's consensus coarse shift, seeded from it.

    The coarse search estimates a quantity that is near-constant along a strip, so running it
    independently per chunk gives one chance per chunk to latch a spurious correlation peak -- the
    entire catastrophic failure mode. The consensus of the corroborated chunks (`_is_anchor`)
    overrides the rest, and the re-solve is pinned to it: re-running the coarse search from the
    consensus seed just hands the chunk a second chance to latch the same peak on top of it
    (`20231204T2004426846` chunk 1 did exactly that, reseeded to -80 m and back out to 6.4 km). With no anchor there is no consensus and the forward pass stands, which
    means the product is uncorrected raw geometry rather than solved.

    Returns (results, info) with `info` for the summary.
    """
    idx_a = [k for k, r in enumerate(results) if _is_anchor(r, cfg0)]
    if not idx_a:
        ws.log("\nno anchor chunk in this strip: no consensus coarse shift, product is UNCORRECTED")
        return results, {"coarse_consensus_m": None, "uncorrected": True}

    spans = [c["s1"] - c["s0"] for c in chunks]
    hold_m = CONSENSUS_HOLD_CHUNKS * float(np.median(spans))
    ac = {k: _frame(chunks[k]) @ _shift_of(results[k]) for k in idx_a}
    fits, outliers = [], {}
    for ax, name, cap in ((0, "along", CONSENSUS_MAX_DRIFT_ALONG), (1, "cross", CONSENSUS_MAX_DRIFT_CROSS)):
        anchors = [(_s_mid(chunks[k]), ac[k][ax], results[k]["fit"]["quality"]["match_frac"]) for k in idx_a]
        f, inl = _consensus_model(anchors, hold_m, cap)
        fits.append(f)
        outliers[name] = [chunks[k]["i"] for k, ok in zip(idx_a, inl, strict=True) if not ok]
        if name == "along":
            along_inl = [(a[0], a[2]) for a, ok in zip(anchors, inl, strict=True) if ok]

    def outmatched(k) -> bool:
        """Whether anchor `k` matched at least as well as the weakest inlier its local along-track line rests on.

        Along-track error can step (dropped/duplicated frames), not only drift, so a well-matched anchor
        off the along-track line is evidence against the line, not a mis-latch.
        """
        if k not in ac:
            return False
        near = sorted(along_inl, key=lambda a: abs(a[0] - _s_mid(chunks[k])))[:CONSENSUS_LOCAL_K]
        return bool(results[k]["fit"]["quality"]["match_frac"] >= min(a[1] for a in near))

    consensus = {c["i"]: [round(float(v), 1) for v in _frame(c).T @ [f(_s_mid(c)) for f in fits]] for c in chunks}
    ws.log(
        f"\nconsensus coarse shift (local lines in along-track distance) from {len(idx_a)} anchor chunk(s), "
        f"outlier anchors {outliers}, tol={CONSENSUS_TOL_M:.0f}m per component, max drift along/cross "
        f"{CONSENSUS_MAX_DRIFT_ALONG * 1000:.0f}/{CONSENSUS_MAX_DRIFT_CROSS * 1000:.0f} m/km"
    )
    for idx, c in enumerate(chunks):
        seed = tuple(consensus[c["i"]])
        dev = _frame(c) @ np.subtract(_shift_of(results[idx]), seed)
        # A chunk already sitting on the consensus (a `fallback` that inherited it as its seed)
        # would re-solve to exactly what it has, so the 90 s it costs buys nothing.
        worst = float(np.abs(dev).max())
        far = abs(dev[1]) > CONSENSUS_TOL_M or (abs(dev[0]) > CONSENSUS_TOL_M and not outmatched(idx))
        off = worst > 1.0 and (not _accepted(results[idx]) or far)
        ws.log(
            f"  chunk {c['i']}: shift={_shift_of(results[idx])} consensus={seed} "
            f"dev along/cross={dev[0]:.0f}/{dev[1]:.0f}m {'RESEED' if off else 'keep'}"
        )
        if not off:
            continue
        results[idx] = _solve_one(
            ws, sid, group, frames, kernels, c, seed, cfg0, ftif, fgeom, flabel, decay_m, tweaks, pin_coarse=True
        )
        if results[idx] is not None:
            results[idx]["fit"]["reseeded"] = True
    return results, {"coarse_consensus_m": consensus, "consensus_outlier_anchors": outliers, "uncorrected": False}


def solve_chunks(ws, sid, group, frames, kernels, chunks, cfg0, ftif, fgeom, flabel, decay_m, tweaks, hillshade_only):
    """Solve each chunk against its own hillshade, reusing any prior good solve of the same shape.

    A forward pass seeds each chunk with the last accepted chunk's coarse shift, then
    `_consensus_pass` re-solves whatever disagrees with the strip as a whole.

    Returns (results, info): one {"chunk", "gcps", "fit"} per solved chunk plus the consensus
    decision, both empty when `hillshade_only`.
    """
    if hillshade_only:
        _solve_chunks_hillshade(ws, sid, group, kernels, chunks, cfg0, fgeom, flabel)
        return [], {}

    results: list[dict | None] = [None] * len(chunks)
    seed = (0.0, 0.0)
    for idx, c in enumerate(chunks):
        results[idx] = _solve_one(ws, sid, group, frames, kernels, c, seed, cfg0, ftif, fgeom, flabel, decay_m, tweaks)
        if _accepted(results[idx]):
            seed = _shift_of(results[idx])

    results, info = _consensus_pass(
        ws, sid, group, frames, kernels, chunks, results, cfg0, ftif, fgeom, flabel, decay_m, tweaks
    )
    solved = [r for r in results if r is not None]
    consensus = info.get("coarse_consensus_m")
    for r in solved:
        r["fit"]["chunk_grade"] = _chunk_grade(r)
        r["fit"].setdefault("reseeded", False)
        r["fit"]["shift_dev_m"] = round(_dev_m(_shift_of(r), consensus[r["chunk"]["i"]]), 1) if consensus else None
    return solved, info


# Per-band solve-time model: {hillshade_s, overhead_s, iter_s} @ the 4-core SBATCH default.
# `overhead_s` is coarse_shift plus GCP-lattice setup, i.e. solve_s minus the sum of that chunk's
# own iter_s. `--max-seconds` reads this to decide whether a chunk fits in the time left, so it
# wants to run a little long rather than short.
# north/north_midlat mirror south/south_midlat (same DEM tier structure) pending their own data.
# `equatorial` re-measured 2026-09-04 (iirs-cluster/jobs/profiling): the old {3, 156, 47} predated
# the gcp_lattice fix (428s -> 1.9s) and the bounded coarse search, and overshot 394s against a
# measured 146s. south/south_midlat were fit the same way and are probably stale too -- unmeasured.
BAND_TIMING: dict[str, dict[str, float]] = {
    "south": {"hillshade_s": 18.0, "overhead_s": 77.0, "iter_s": 43.0},
    "south_midlat": {"hillshade_s": 2.0, "overhead_s": 81.0, "iter_s": 57.0},
    "equatorial": {"hillshade_s": 3.0, "overhead_s": 10.0, "iter_s": 30.0},
    "north": {"hillshade_s": 18.0, "overhead_s": 77.0, "iter_s": 43.0},
    "north_midlat": {"hillshade_s": 2.0, "overhead_s": 81.0, "iter_s": 57.0},
}


# Iterations a plateau-converged chunk lands around; a placeholder until measured under the
# current convergence rule.
TYPICAL_NITER = 4


def _band_estimate_s(band: str, niter: int) -> float:
    """Solve time for one `band` chunk run to `niter` iterations, from `BAND_TIMING`.

    >>> round(_band_estimate_s("equatorial", 5))
    163
    """
    t = BAND_TIMING[band]
    return t["hillshade_s"] + t["overhead_s"] + niter * t["iter_s"]


def plan_group_chunks(
    ws: Workspace,
    sid: str,
    group: str,
    fgeom,
    dem_roots: tuple[Path, ...],
    scan0: int,
    ny: int,
    only: str | None,
    niter_max: int,
) -> list[dict]:
    """This group's chunks, with cube row bounds attached and optionally filtered to `only`.

    `only` is a comma-separated list of chunk indices. `niter_max` is `GeorefConfig.niter`, for the
    worst-case estimate.
    """
    all_chunks = ck.plan_chunks(fgeom, dem_roots, WIDTH_RANGE_KM, OVERLAP_FRAC)
    chunks = [c for c in all_chunks if c["group"] == group]
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
            raise ValueError(f"--chunks {sorted(missing)} not in group {group!r}: {detail}")
        chunks = [c for c in chunks if c["i"] in wanted]
    ws.path(f"{sid}_{group}_chunks.json").write_text(json.dumps(chunks, indent=1))
    ws.log(f"{len(chunks)} chunks planned ({group} group, {WIDTH_RANGE_KM[0]:.0f}-{WIDTH_RANGE_KM[1]:.0f}km):")
    for c in chunks:
        ws.log(
            f"  chunk {c['i']} [{c['band']}]: rows {c['row0']}-{c['row1']}, "
            f"s {c['s0'] / 1000:.0f}-{c['s1'] / 1000:.0f}km (w={(c['s1'] - c['s0']) / 1000:.0f}km), "
            f"aoi_km={[round(v / 1000, 1) for v in c['aoi']]}"
        )
    bands_seen = list(dict.fromkeys(c["band"] for c in chunks))
    ws.log("per-band summary:")
    typical_s = worst_s = 0.0
    for band in bands_seen:
        bc = [c for c in chunks if c["band"] == band]
        rows = sum(c["row1"] - c["row0"] + 1 for c in bc)
        band_typical = len(bc) * _band_estimate_s(band, TYPICAL_NITER)
        band_worst = len(bc) * _band_estimate_s(band, niter_max)
        typical_s += band_typical
        worst_s += band_worst
        ws.log(
            f"  {band}: {len(bc)} chunk(s), {rows} rows, near={Path(bc[0]['dem_near']).name}, "
            f"far={Path(bc[0]['dem_far']).name}, ~{band_typical / 60:.0f} min typical / "
            f"~{band_worst / 60:.0f} min worst-case"
        )
    ws.log(
        f"estimated solve time: ~{typical_s / 60:.0f} min typical ({TYPICAL_NITER} iters/chunk, "
        f"plateau-converged) / ~{worst_s / 60:.0f} min worst-case (every chunk hits the "
        f"{niter_max}-iter cap) for {len(chunks)} chunks"
    )
    return chunks
