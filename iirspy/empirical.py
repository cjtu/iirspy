"""
On-orbit empirical dark / flat / smile correction for IIRS L0->L1 calibration.

The IIRS gain/offset LUT is applied per detector element (band, x), but its cross-track
structure is poorly correlated with the true on-orbit detector response, injecting striping
and speckle. This module derives three corrections from the scene itself and lets the L0->L1
step use the LUT only for the per-band absolute scale:

  dark(band, x)  : median DN over auto-detected shadow rows (0 if the scene has no shadow).
  flat(band, x)  : sensor high-frequency response, R / lowpass_x(R), median-composited over the
                   flattest scene region in each of several DN bins. A hardware property, so it
                   is transferable across scenes (though here it is recomputed per scene).
  smile(band, x) : per-scene smooth cross-track field (solar geometry + regional detector
                   response), from the whole-scene along-track median after dark+flat.

The calibrated radiance is then (see iirs.L0.calibrate_to_rad):

    rad = 10 * ((DN - dark) / flat / smile * gain_abs(band) + off_abs(band))

Everything here operates on the raw DN DataArray (band, y, x) and depends only on iirspy.utils.
"""

import warnings

import numpy as np
import xarray as xr

import iirspy.utils as utils

# Panchromatic band spread (avoids OSF/invalid) used to beat single-band noise
_OSF, _INV = set(utils.OSF), set(utils.INVALID)
PAN_BANDS = [b for b in range(10, 251, 8) if b not in _OSF and b not in _INV]
CLEAN_BANDS = [42, 110, 111, 112, 113, 249, 250, 251]  # fewest bad pixels, for crater detection

# Dark / shadow detection (location-agnostic: shadow is found wherever it is, if present at all)
DARK_PCT = 0.5  # darkest percentile of rows used to estimate the shadow floor + noise
DARK_K = 8.0  # row shadow threshold = floor + DARK_K * spread
DARK_SPREAD_FLOOR = 0.15  # min shadow-cluster spread [DN]
DARK_MAX_FRAC = 0.3  # darkest rows count as shadow only if floor < this * scene median
MIN_DARK_ROWS = 100  # need at least this many contiguous shadow rows to derive a dark
SHADOW_SNR = 2.0  # broadband SNR above the dark level below which a pixel is nulled (shadow)
NOISE_FLOOR = 0.5  # min noise [DN] to avoid divide-by-zero in the SNR

# Flat-region detection (per-row roughness -> variable-length runs)
XSMOOTH = 25  # cross-track running-mean window isolating local scene features
YSMOOTH = 51  # along-track smoothing of the per-row roughness profile
EDGE_MARGIN = 15  # extra rows trimmed off each run end after the raw-roughness trim
MINLEN = 200  # minimum flat-window length (rows)
TOL_FAC = 1.4  # accept rows with roughness < TOL_FAC * floor
MIN_LIT_FRAC = 0.25  # lit rows are brighter than this fraction of the scene median
DN_BINS = ((50, 100), (100, 150), (150, 200), (200, np.inf))  # flat sampled across DN levels

# Flat / smile
SMOOTH_X = 31  # cross-track lowpass window separating sensor (high-freq) from smile (smooth)
SMILE_STRIDE = 8  # along-track subsample stride for the whole-scene smile estimate


def lowpass_x(da, w=SMOOTH_X):
    """Smooth a (..., x) DataArray across x with a centered rolling median."""
    return da.rolling(x=w, center=True, min_periods=1).median()


def robust_z(resid, axis):
    """Nan-safe robust (MAD-based) z-score of resid along axis."""
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid, axis=axis, keepdims=True)), axis=axis, keepdims=True)
    robust_std = 1.4826 * mad
    return np.divide(resid, robust_std, out=np.full_like(resid, np.nan), where=robust_std > 0)


def panchromatic(img):
    """Band-averaged (y, x) brightness over PAN_BANDS (eager)."""
    return img.sel(band=PAN_BANDS).mean("band").compute()


def _xr_yx(mask_yx, like):
    """Wrap a (y, x) numpy bool mask as a DataArray aligned to like's y, x coords."""
    return xr.DataArray(mask_yx, coords={"y": like.y, "x": like.x}, dims=("y", "x"))


def longest_run(mask):
    """Return (start, stop) of the longest contiguous True run in a 1-D bool array."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return (0, 0)
    splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    best = max(splits, key=len)
    return int(best[0]), int(best[-1]) + 1


def detect_dark_rows(P):
    """
    Return (dark_mask_y, threshold, row_bright). Shadow rows via a location-agnostic threshold.

    Estimate the dark floor + noise from the darkest DARK_PCT of rows (wherever they are - no
    assumption that the scene is a pole crossing with leading shadow) and cut DARK_K spreads
    above it. The darkest rows only count as shadow if they are well below the scene's overall
    brightness (floor < DARK_MAX_FRAC * scene median); otherwise the scene has no usable shadow
    (e.g. an equatorial pass) and the dark step is skipped.
    """
    row_bright = np.nanmedian(P.values, axis=1)
    lo = row_bright[row_bright <= np.nanpercentile(row_bright, DARK_PCT)]
    floor = np.nanmedian(lo)
    spread = max(np.nanmedian(np.abs(lo - floor)) * 1.4826, DARK_SPREAD_FLOOR)
    if floor >= DARK_MAX_FRAC * np.nanmedian(row_bright):
        return np.zeros_like(row_bright, dtype=bool), floor, row_bright
    thresh = floor + DARK_K * spread
    return row_bright < thresh, thresh, row_bright


def shadow_context(P, d0, d1):
    """
    Return a (y, x) bool mask, True where a pixel's broadband signal is at the dark level (shadow).

    Uses the panchromatic (27-band-averaged, high-SNR) brightness so it works even where individual
    long-wavelength bands are pure noise: pixels whose broadband SNR above the dark level is below
    SHADOW_SNR carry no real signal in any band and are nulled to give a uniformly black shadow.
    """
    ref = P.isel(y=slice(d0, d1))
    pan_dark = ref.median("y")  # (x,) broadband dark level
    pan_noise = ref.std("y").clip(min=NOISE_FLOOR)  # (x,) broadband noise
    return (P - pan_dark) / pan_noise < SHADOW_SNR


def _runs_below(mask):
    """Return list of (start, stop) contiguous True runs in a 1-D bool array."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [(int(s[0]), int(s[-1]) + 1) for s in splits]


def _trim_run(a, b, r_raw, tol, margin=EDGE_MARGIN):
    """Trim a run inward while the un-smoothed roughness is above tol, then inset by margin."""
    while a < b and r_raw[a] >= tol:
        a += 1
    while b > a and r_raw[b - 1] >= tol:
        b -= 1
    return a + margin, b - margin


def row_roughness(P):
    """
    Per-row scene roughness rs(y) (smoothed), raw roughness r(y), and the lit mask.

    N = P / global_prof removes the fixed detector stripe/vignette (dead columns dropped). Each
    row's cross-track high-pass isolates local scene features (craters); the row's tail-sensitive
    (5-95 pctile) spread relative to its median is its roughness.
    """
    from scipy.ndimage import uniform_filter1d

    vals = P.values
    global_prof = np.nanmedian(vals, axis=0)
    good = global_prof > 0.2 * np.nanmedian(global_prof)
    N = vals[:, good] / global_prof[good]
    hp = N - uniform_filter1d(N, XSMOOTH, axis=1, mode="nearest")
    p5, p95 = np.nanpercentile(hp, [5, 95], axis=1)
    r = (p95 - p5) / np.nanmedian(N, axis=1)
    rs = uniform_filter1d(r, YSMOOTH, mode="nearest")
    row_bright = np.nanmedian(vals, axis=1)
    lit = row_bright > MIN_LIT_FRAC * np.nanmedian(vals)
    return rs, r, lit


def find_flat_runs(rs, r_raw, lit, minlen=MINLEN, tol_fac=TOL_FAC, min_runs=1):
    """
    Adaptive-threshold contiguous flat runs (variable length >= minlen), flattest-first.

    The acceptance threshold grows from tol_fac * floor until at least min_runs distinct lit
    runs (edge-trimmed) reach minlen.
    """
    masked = np.where(lit, rs, np.inf)
    floor = float(np.nanpercentile(rs[lit], 5))
    tol = tol_fac * floor
    runs = []
    for _ in range(40):
        trimmed = (_trim_run(a, b, r_raw, tol) for a, b in _runs_below(masked < tol))
        runs = [(a, b) for a, b in trimmed if b - a >= minlen]
        if len(runs) >= min_runs:
            break
        tol *= 1.15
    runs.sort(key=lambda ab: float(np.nanmean(rs[ab[0] : ab[1]])))
    return runs


def flattest_run_in_bin(rs, r_raw, lit, row_bright, lo, hi):
    """The single flattest flat run whose rows fall in DN bin [lo, hi); None if none qualify."""
    binmask = lit & (row_bright >= lo) & (row_bright < hi)
    if binmask.sum() < MINLEN:
        return None
    runs = find_flat_runs(rs, r_raw, binmask)
    return runs[0] if runs else None


def crater_mask(fsub):
    """(y, x) bool mask, True where a crater/anomaly is detected via the clean-band spatial stack."""
    cb = fsub.sel(band=CLEAN_BANDS)
    struct = (cb / cb.mean(("y", "x"))).median("band").values
    z = robust_z(struct - np.nanmedian(struct), axis=(0, 1))
    return np.abs(z) > 4.0


def derive_sensor_flat(img, dark, a, b):
    """Sensor flat (band, x) = R / lowpass_x(R) from rows [a, b): high-frequency response only."""
    fsub = (img.sel(y=slice(a, b)).astype("float32") - dark).compute()
    mask = crater_mask(fsub)
    R = fsub.where(~_xr_yx(mask, fsub)).median("y")
    return (R / lowpass_x(R)).astype("float32")


def build_flat(img, dark, rs, r_raw, lit, row_bright):
    """Median composite sensor flat over the flattest region of each DN bin."""
    flats = []
    for lo, hi in DN_BINS:
        run = flattest_run_in_bin(rs, r_raw, lit, row_bright, lo, hi)
        if run is not None:
            flats.append(derive_sensor_flat(img, dark, *run))
    if not flats:
        raise ValueError("no flat regions found for empirical flat")
    flat = xr.concat(flats, dim="region").median("region")
    return flat.where(~utils.load_bad_pixel_mask()).astype("float32")


def estimate_smile(img, dark, flat, lit, stride=SMILE_STRIDE):
    """Per-scene smooth cross-track field (band, x) from the whole scene's along-track median."""
    ys = np.flatnonzero(lit)[::stride]
    sub = ((img.isel(y=ys).astype("float32") - dark) / flat).compute()
    M = sub.median("y")
    smile = lowpass_x(M)
    return (smile / smile.median("x")).astype("float32")


def empirical_frames(img):
    """
    Derive the empirical correction frames from a raw DN cube (band, y, x).

    Returns (dark, flat, smile, shadow). dark is 0.0 and shadow is None when the scene has no
    usable shadow rows; otherwise dark is the (band, x) shadow median and shadow is a (y, x) bool
    mask of broadband-shadow pixels to null. flat and smile are always (band, x). See the module
    docstring for the correction each applies.
    """
    P = panchromatic(img)
    dark_mask, _, row_bright = detect_dark_rows(P)
    d0, d1 = longest_run(dark_mask)
    if d1 - d0 >= MIN_DARK_ROWS:
        dark = img.sel(y=slice(d0, d1)).median("y").compute().astype("float32")
        shadow = shadow_context(P, d0, d1)
    else:
        warnings.warn("no shadow rows found; skipping empirical dark subtraction", UserWarning, stacklevel=2)
        dark = xr.zeros_like(P.isel(y=0), dtype="float32").drop_vars("y")
        shadow = None

    rs, r_raw, lit = row_roughness(P)
    flat = build_flat(img, dark, rs, r_raw, lit, row_bright)
    smile = estimate_smile(img, dark, flat, lit)
    return dark, flat, smile, shadow
