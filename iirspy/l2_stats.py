"""Cheap sanity statistics of a saved L2 reflectance cube, for judging a refl run before the cube is synced.

Reads a strided sample of ~10 bands from the L2 tif (and the thermal bands of the L1 tif), never the whole cube, so it
costs seconds. Everything is a plain float/list for the refl summary json.

>>> round(float(_hapke_ssa(np.array([0.3]), np.array([60.0]), np.array([0.0]), np.array([60.0]))[0]), 2) > 0.4
True
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import rasterio

from iirspy import utils

TARGET_NM = (750, 1000, 1250, 1500, 1800, 2000, 2400, 2700, 3500, 4200, 4800)
REF_NM, SSA_NM = 1500, 1000  # level / drift / correlation band, SSA band
SSA_INC_MAX = 80.0  # deg; the Hapke inversion is ill-conditioned toward grazing incidence
LIT_INC = 85.0  # deg; flat-model incidence above this is grazing, not lit
L1_TIF_SCALE = 0.01  # the L1 tif stores 100 x W/m^2/sr/um (measured on 7 scenes: rad / (refl flux cos i) = 99.5 +- 0.5)
HAPKE = {
    "b": 0.3,
    "B0": 1.0,
    "h": 0.07,
}  # single-term HG (b>0 backscatter), SHOE; no roughness. Provisional until calval.


def _bands(src) -> tuple[np.ndarray, np.ndarray]:
    """(IIRS band numbers 1..n, wavelengths [nm]) of every band, from the band descriptions."""
    return np.arange(1, src.count + 1), np.array([float(d) for d in src.descriptions])


def _valid(src) -> set[int]:
    """IIRS band numbers actually in the cube (`Band_k` tags); the file is padded to 256 with fill bands."""
    nums = {int(v.split()[0]) for k, v in src.tags().items() if k.startswith("Band_")}
    return nums or set(range(1, src.count + 1))


def _read(src, idx, step):
    """(len(idx), ny/step, nx/step) strided float32 read of the 1-based band `idx`."""
    return src.read(list(idx), out_shape=(len(idx), src.height // step, src.width // step)).astype("float32")


def _hapke_h(x, w):
    r0 = (1 - np.sqrt(1 - w)) / (1 + np.sqrt(1 - w))
    return 1 / (1 - w * x * (r0 + (1 - 2 * r0 * x) / 2 * np.log((1 + x) / x)))


def _hapke_model(w, mu0, mu, g, b=HAPKE["b"], B0=HAPKE["B0"], h=HAPKE["h"]):
    """Reflectance as `iirspy` defines it (I / (F/pi * cos i)) = REFF / mu0, for single scattering albedo `w`."""
    gr = np.radians(g)
    P = (1 - b * b) / (1 - 2 * b * np.cos(gr) + b * b) ** 1.5
    B = B0 / (1 + np.tan(gr / 2) / h)
    return w / 4 / (mu0 + mu) * ((1 + B) * P + _hapke_h(mu0, w) * _hapke_h(mu, w) - 1)


def _hapke_ssa(refl, inc, emi, phase):
    """Single scattering albedo per sample by inverting `_hapke_model` over a w grid (monotonic); NaN outside the grid."""
    mu0, mu = np.cos(np.radians(inc))[:, None], np.cos(np.radians(emi))[:, None]
    wg = np.linspace(0.02, 0.999, 120)[None]
    m = _hapke_model(wg, mu0, mu, phase[:, None])
    k = np.clip((m < refl[:, None]).sum(1), 1, m.shape[1] - 1)
    r = np.arange(len(refl))
    f = (refl - m[r, k - 1]) / (m[r, k] - m[r, k - 1])
    w = wg[0, k - 1] + f * (wg[0, k] - wg[0, k - 1])
    return np.where((f >= 0) & (f <= 1), w, np.nan)


def l2_stats(fl2, fl1, inc, lat, emi=None, phase=None, sdist=1.0, step=8, t_bins=2.0) -> dict:
    """Statistics of `fl2` (reflectance tif) with `fl1` (radiance tif, for the isothermal T).

    `inc` [deg] and `lat` [deg] are per row (ny,) or per pixel (ny, nx); `emi`/`phase` per pixel or None
    (nadir, phase = incidence). `step` strides rows and columns.
    """
    t0 = time.time()
    out: dict = {}
    with rasterio.open(fl2) as src:
        _, wl = _bands(src)
        ok_b = np.array(sorted(_valid(src)))
        sel = {t: int(ok_b[np.argmin(abs(wl[ok_b - 1] - t))]) for t in TARGET_NM}
        extra = [sel[REF_NM] - 1, sel[REF_NM] + 1]
        cube = _read(src, sorted({*sel.values(), *extra}), step)
        order = {b: i for i, b in enumerate(sorted({*sel.values(), *extra}))}
        ny, nx = src.height, src.width
    sub = lambda a: (
        np.broadcast_to(np.asarray(a, "float32")[:, None], (ny, nx)) if np.ndim(a) == 1 else np.asarray(a, "float32")
    )[::step, ::step][: cube.shape[1], : cube.shape[2]]
    inc_s, lat_s = sub(inc), sub(lat)
    ref = cube[order[sel[REF_NM]]]
    foot = np.isfinite(inc_s)
    lit = foot & (inc_s < LIT_INC)
    out["sample"] = {"step": step, "n_px": int(foot.sum()), "wl_nm": {str(t): float(wl[b - 1]) for t, b in sel.items()}}

    # 1. coverage: NaN / exactly 0 (fill_nan, clip_negative) / negative, all bands, and lit pixels that came out unusable
    cov = {}
    for t, b in sel.items():
        c = cube[order[b]]
        cov[str(t)] = {
            "nan": float(np.isnan(c[foot]).mean()),
            "zero": float((c[foot] == 0).mean()),
            "lit_unusable": float((~np.isfinite(c[lit]) | (c[lit] <= 0)).mean()) if lit.any() else None,
        }
    out["coverage"] = cov
    out["lit_frac"] = float(lit.sum() / max(foot.sum(), 1))
    out["lit_px_full"] = int(lit.sum() * step * step)

    # 2. level and spectrum on lit, usable pixels
    ok = lit & np.isfinite(ref) & (ref > 0)
    v = ref[ok]
    out["level"] = {"n": int(ok.sum())}
    if v.size:
        out["level"] |= {f"p{p}": float(np.percentile(v, p)) for p in (5, 50, 95)} | {
            "gt1": float((v > 1).mean()),
            "gt05": float((v > 0.5).mean()),
        }
    spec = {}
    for t, b in sel.items():
        c = cube[order[b]][ok]
        spec[str(t)] = float(np.nanmedian(c[c > 0])) if (c > 0).any() else None
    out["median_spectrum"] = spec
    a, bb, c = (cube[order[k]][ok] for k in (extra[0], sel[REF_NM], extra[1]))
    out["roughness"] = float(np.nanmedian(abs(a - 2 * bb + c)) / np.nanmedian(bb)) if v.size else None
    s24, s48 = spec["2400"], spec["4800"]
    out["thermal_tail"] = (
        float(s48 / s24) if s24 and s48 else None
    )  # b4800 / b2400 median: >> its expected value = leftover thermal

    # 3. along-track drift and steps, across-track profile (REF band, lit usable pixels)
    def blocks(axis, n):
        edges = np.linspace(0, ref.shape[axis], n + 1).astype(int)
        res = []
        for lo, hi in itertools.pairwise(edges):
            m = ok[lo:hi] if axis == 0 else ok[:, lo:hi]
            r = ref[lo:hi] if axis == 0 else ref[:, lo:hi]
            res.append(float(np.median(r[m])) if m.sum() > 50 else None)
        return res

    along, across = blocks(0, 10), blocks(1, 10)
    out["along_track"], out["across_track"] = along, across
    al = np.array([x for x in along if x is not None])
    out["along_max_step"] = float(np.max(abs(np.diff(al))) / np.median(al)) if al.size > 1 else None
    ac = [x for x in across if x is not None]
    out["edge_to_centre"] = float(np.mean([ac[0], ac[-1]]) / np.median(ac[3:-3])) if len(ac) >= 8 else None

    # 4. dead rows (lit rows with nothing usable at any sampled band)
    rows = lit.any(1)
    usable = np.isfinite(cube) & (cube > 0)
    out["dead_row_frac"] = float((~usable[:, rows][:, :].any(axis=(0, 2))).mean()) if rows.any() else None

    # 5. geometry consistency: refl vs cos(incidence) over lit usable pixels (flat model: ~0 if photometry is right)
    if v.size > 100:
        ci = np.cos(np.radians(inc_s[ok]))
        out["corr_refl_cosinc"] = float(np.corrcoef(v, ci)[0, 1])

    # 6. isothermal T [K] vs latitude from the L1 thermal bands, and the radiative-equilibrium flat-surface T
    with rasterio.open(fl1) as s1:
        n1, wl1 = _bands(s1)
        tb = [i + 1 for i in range(s1.count) if 227 <= n1[i] <= 247]
        rad = _read(s1, tb, step)[:, : cube.shape[1], : cube.shape[2]] * L1_TIF_SCALE
    import xarray as xr

    da = xr.DataArray(
        rad, dims=("band", "y", "x"), coords={"band": n1[np.array(tb) - 1], "wl": ("band", wl1[np.array(tb) - 1])}
    )
    _, t_k = utils.thermal_radiance(
        da, 1.0, (227, 247), inc=xr.DataArray(inc_s, dims=("y", "x")), sdist=sdist, t_min=0.0
    )
    temp = np.asarray(t_k)
    teq = utils.flat_surface_temperature(inc_s, sdist)
    out["T"] = {
        "median": float(np.nanmedian(temp[lit])),
        "p5": float(np.nanpercentile(temp[lit], 5)),
        "p95": float(np.nanpercentile(temp[lit], 95)),
    }
    edges = np.arange(np.floor(np.nanmin(lat_s[lit])), np.nanmax(lat_s[lit]) + t_bins, t_bins)
    ib = np.digitize(lat_s, edges)
    prof = []
    for k in np.unique(ib[lit]):
        m = lit & (ib == k)
        if m.sum() > 100:
            prof.append([
                float(edges[k - 1] + t_bins / 2),
                float(np.nanmedian(temp[m])),
                float(np.nanmedian(teq[m])),
                int(m.sum()),
            ])
    out["T_vs_lat"] = prof  # [lat, T fitted, T radiative equilibrium, n]

    # 7. single scattering albedo (provisional Hapke, fixed params)
    with rasterio.open(fl2) as src:
        rssa = _read(src, [sel[SSA_NM]], step)[0]
    m = lit & (inc_s < SSA_INC_MAX) & np.isfinite(rssa) & (rssa > 0)
    if m.sum() > 100:
        e = np.nan_to_num(sub(emi)[m]) if emi is not None else np.zeros(m.sum(), "float32")
        g = np.where(np.isfinite(sub(phase)[m]), sub(phase)[m], inc_s[m]) if phase is not None else inc_s[m]
        w = _hapke_ssa(rssa[m].astype("float64"), inc_s[m].astype("float64"), e.astype("float64"), g.astype("float64"))
        out["ssa"] = {
            "nm": SSA_NM,
            "mean": float(np.nanmean(w)),
            "std": float(np.nanstd(w)),
            "valid": float(np.isfinite(w).mean()),
            "params": HAPKE,
        }
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out
