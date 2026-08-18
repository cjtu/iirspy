"""
On-orbit empirical residual-dark / flat / smile correction for IIRS L0->L1 calibration.

The IIRS gain/offset LUT is applied per detector element (band, x), but its cross-track
structure is poorly correlated with the true on-orbit detector response, injecting striping
and speckle. This module derives three corrections from the scene itself and lets the L0->L1
step use the LUT only for the per-band absolute scale (see iirs.L0.calibrate_to_rad):

  dark_resid(band, x) : median DN over auto-detected shadow rows. L0 counts are already
                        dark-subtracted onboard (pre-imaging night-side dark), so this is a
                        *residual* dark - drift + hot/warm-pixel residuals + stray light. Its
                        real job is per-element hot-pixel refinement. 0 when no shadow is found.
  flat(band, x)       : sensor high-frequency response, R / lowpass_x(R), median-composited over
                        the flattest scene region in each of several DN bins. A hardware property,
                        so it is transferable; falls back to a packaged reference flat when the
                        scene has no qualifying flat region.
  smile(band, x)      : band-relative smooth cross-track field. Only spectrally-varying smooth
                        structure is divided out; the broadband scene gradient is preserved.

Correction model (shadow anchors the zero point; see SPEC / calibrate_to_rad):

  shadow scene   : rad = 10 * gain_med * (DN - dark_resid) / (flat * smile)   # LUT offset dropped
  no-shadow scene: rad = 10 * (gain_med * DN / (flat * smile) + offset_med)   # LUT offset kept

Everything here operates on the raw DN DataArray (band, y, x).
"""

import warnings

import numpy as np
import xarray as xr

import iirspy.utils as utils

# --- Parameters (single source for the paper's parameter table; see SPEC) --------------------
# Panchromatic band range (avoids OSF/invalid) used to beat single-band noise
_OSF, _INV = set(utils.OSF), set(utils.INVALID)
PAN_BANDS = [b for b in range(10, 251, 8) if b not in _OSF and b not in _INV]
CLEAN_BANDS = [42, 110, 111, 112, 113, 249, 250, 251]  # fewest bad pixels, for spatial-outlier detection

# Residual-dark / shadow detection (location-agnostic: shadow is found wherever it is, if at all)
DARK_PCT = 0.5  # darkest percentile of rows used to estimate the shadow floor + noise
DARK_K = 8.0  # row shadow threshold = floor + DARK_K * sigma
DARK_SIGMA_FLOOR = 0.15  # min shadow-cluster standard deviation [DN]
DARK_MAX_FRAC = 0.3  # darkest rows count as shadow only if floor < this * scene median
MIN_DARK_ROWS = 100  # need at least this many contiguous shadow rows to derive a residual dark
SHADOW_SNR = 2.0  # recommended default: broadband SNR below this = shadow (user-applied, not internal)
NOISE_FLOOR = 0.5  # min noise [DN] to avoid divide-by-zero in the SNR

# Flat-region detection (per-row roughness -> variable-length runs)
XSMOOTH = 25  # cross-track running-mean window isolating local scene features
YSMOOTH = 51  # along-track smoothing of the per-row roughness profile
EDGE_MARGIN = 15  # extra rows trimmed off each run end after the raw-roughness trim
MINLEN = 200  # minimum flat-window length (rows)
TOL_FAC = 1.4  # accept rows with roughness < TOL_FAC * floor
LIT_SNR = 10.0  # lit rows sit > this many dark-noise standard deviations above the dark floor (flat/smile input)
DN_BINS = ((50, 100), (100, 150), (150, 200), (200, np.inf))  # flat sampled across DN levels

# Flat / smile
SMOOTH_X = 31  # cross-track lowpass window separating sensor (high-freq) from smile (smooth)
SMILE_STRIDE = 8  # along-track subsample stride for the smile estimate
MIN_LIT_ROWS = 50  # min lit rows to estimate a smile (else fall back to smile=1)
FLAT_SMILE_MIN = 0.2  # clip flat*smile away from 0 before dividing (avoids blow-ups)


def lowpass_x(da, w=SMOOTH_X):
    """Smooth a (..., x) DataArray across x with a centered rolling median."""
    return da.rolling(x=w, center=True, min_periods=1).median()


def robust_z(resid, axis):
    """Nan-safe robust (MAD-based) z-score of resid along axis."""
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid, axis=axis, keepdims=True)), axis=axis, keepdims=True)
    robust_std = 1.4826 * mad
    return np.divide(resid, robust_std, out=np.full_like(resid, np.nan), where=robust_std > 0)


def avail(img, bands):
    """Intersect `bands` with the bands the cube carries, so band subsets calibrate. Empty raises."""
    have = set(np.asarray(img.band.values).tolist())
    sel = [b for b in bands if b in have]
    if not sel:
        raise ValueError(f"cube carries none of bands {bands[:5]}...; cannot derive the empirical correction")
    return sel


def panchromatic(img):
    """Band-averaged (y, x) brightness over PAN_BANDS (eager)."""
    return img.sel(band=avail(img, PAN_BANDS)).mean("band").compute()


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


def dark_floor(row_bright):
    """Return (floor, sigma): the dark level and its robust standard deviation from the darkest DARK_PCT rows."""
    lo = row_bright[row_bright <= np.nanpercentile(row_bright, DARK_PCT)]
    floor = np.nanmedian(lo)
    sigma = max(np.nanmedian(np.abs(lo - floor)) * 1.4826, DARK_SIGMA_FLOOR)  # MAD->sigma
    return floor, sigma


def detect_dark_rows(P):
    """
    Return (dark_mask_y, threshold, row_bright). Shadow rows via a location-agnostic threshold.

    Estimate the dark floor + noise from the darkest DARK_PCT of rows (wherever they are - no
    assumption that the scene is a pole crossing with leading shadow) and cut DARK_K standard
    deviations above it. The darkest rows only count as shadow if they are well below the scene's overall
    brightness (floor < DARK_MAX_FRAC * scene median); otherwise the scene has no usable shadow
    (e.g. an equatorial pass) and the dark step is skipped.
    """
    row_bright = np.nanmedian(P.values, axis=1)
    floor, sigma = dark_floor(row_bright)
    if floor >= DARK_MAX_FRAC * np.nanmedian(row_bright):
        return np.zeros_like(row_bright, dtype=bool), floor, row_bright
    thresh = floor + DARK_K * sigma
    return row_bright < thresh, thresh, row_bright


def broadband_snr(P, d0, d1):
    """
    Return the per-pixel broadband SNR field (y, x) float32 above the dark level of rows [d0, d1).

    Uses the panchromatic (27-band-averaged, high-SNR) brightness so it is meaningful even where
    individual long-wavelength bands are noisy. Downstream users can threshold to desired SNR.
    """
    ref = P.isel(y=slice(d0, d1))
    pan_dark = ref.median("y")  # (x,) broadband dark level
    pan_noise = ref.std("y").clip(min=NOISE_FLOOR)  # (x,) broadband noise
    return ((P - pan_dark) / pan_noise).astype("float32")


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


def row_roughness(P, floor, sigma):
    """
    Per-row scene roughness rs(y) (smoothed), raw roughness r(y), and the lit mask.

    N = P / global_prof removes the fixed detector stripe/vignette (dead columns dropped). Each
    row's cross-track high-pass isolates local scene features (e.g. craters); the row's tail-sensitive
    (5-95 pctile) range relative to its median is its roughness. A row is lit (usable for flat /
    smile) when its broadband brightness is > LIT_SNR dark-noise standard deviations above the dark floor.
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
    lit = (row_bright - floor) / sigma > LIT_SNR
    return rs, r, lit


def find_flat_runs(rs, r_raw, lit, minlen=MINLEN, tol_fac=TOL_FAC, min_runs=1):
    """
    Adaptive-threshold contiguous flat runs (variable length >= minlen), flattest-first.

    The acceptance threshold grows from tol_fac * floor until at least min_runs distinct lit
    runs (edge-trimmed) reach minlen. Returns [] when no lit rows exist.
    """
    if not lit.any():
        return []
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


def spatial_outlier_mask(fsub):
    """
    (y, x) bool mask, True at pixels that deviate strongly from the flat clean-band scene plane.

    Normalizes each clean band by its spatial mean, median-stacks to a (y, x) structure map, and
    flags pixels > 4 robust-z from the plane median (both bright and dark). These are scene
    structure - crater walls/shadows, bright rims, residual bad pixels - excluded so the flat's
    median over y isn't biased by real spatial features rather than sensor response.
    """
    cb = fsub.sel(band=avail(fsub, CLEAN_BANDS))
    struct = (cb / cb.mean(("y", "x"))).median("band").values
    z = robust_z(struct - np.nanmedian(struct), axis=(0, 1))
    return np.abs(z) > 4.0


def derive_sensor_flat(img, dark, a, b):
    """Sensor flat (band, x) = R / lowpass_x(R) from rows [a, b): high-frequency response only."""
    fsub = (img.isel(y=slice(a, b)).astype("float32") - dark).compute()
    mask = spatial_outlier_mask(fsub)
    R = fsub.where(~_xr_yx(mask, fsub)).median("y")
    return (R / lowpass_x(R)).astype("float32")


def _flat_rms(flat, ref):
    """Per-detector-element RMS deviation of a per-scene flat from the packaged reference flat.

    float64: squaring the float32 difference overflows to inf on the handful of elements where
    lowpass_x(R) rounds to ~0, which turned the whole diagnostic into inf. Non-finite elements are
    dropped rather than propagated - this is a provenance metric, not part of the correction.
    """
    d = (flat - ref).values.astype("float64")
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d**2))) if d.size else float("nan")


def build_flat(img, dark, rs, r_raw, lit, row_bright, ref_flat=None):
    """
    Median-composite sensor flat (band, x) over the flattest region of each DN bin.

    Falls back to the packaged reference flat (with a UserWarning) when no in-scene flat region
    qualifies - rough/shadowed south-polar scenes, the primary target, most often hit this. Never
    raises. Returns (flat, emp_notes) where emp_notes records the runs used, whether the fallback
    fired, and the per-scene-vs-reference RMS deviation (the cross-scene flat stability metric).
    """
    bad = utils.load_bad_pixel_mask()
    flats, runs_used = [], {}
    for lo, hi in DN_BINS:
        run = flattest_run_in_bin(rs, r_raw, lit, row_bright, lo, hi)
        runs_used[f"{lo}-{hi}"] = list(run) if run is not None else None
        if run is not None:
            flats.append(derive_sensor_flat(img, dark, *run))
    emp_notes = {"flat_runs": runs_used, "flat_fallback": False, "flat_ref_rms": None}

    if flats:
        flat = xr.concat(flats, dim="region").median("region").where(~bad).astype("float32")
        if ref_flat is not None:
            emp_notes["flat_ref_rms"] = _flat_rms(flat, ref_flat)
    elif ref_flat is not None:
        warnings.warn("no in-scene flat region; falling back to packaged reference flat", UserWarning, stacklevel=2)
        flat = ref_flat.where(~bad).astype("float32")
        emp_notes["flat_fallback"] = True
    else:
        warnings.warn("no in-scene flat region and no packaged reference flat; using flat=1", UserWarning, stacklevel=2)
        flat = xr.where(bad, np.nan, 1.0).astype("float32")
        emp_notes["flat_fallback"] = True
    return flat, emp_notes


def estimate_smile(img, dark, flat, lit, stride=SMILE_STRIDE):
    """
    Band-relative smooth cross-track field (band, x): only spectrally-varying structure.

    From the dark+flat-corrected along-track median M(band, x):
        ref(x)        = median over PAN_BANDS of lowpass_x(M)   # broadband scene profile
        smile(band,x) = lowpass_x(M(band,x)) / ref(x)
        smile        /= median_x(smile)                          # per band: preserve band scale
    Dividing by smile removes only spectrally-varying smooth cross-track structure; the broadband
    scene gradient ref(x) cancels and survives in the product. Falls back to smile=1 (with a
    warning) when too few lit rows exist. Returns (smile, emp_notes).
    """
    ys = np.flatnonzero(lit)
    if ys.size < MIN_LIT_ROWS:
        warnings.warn("too few lit rows for a smile estimate; using smile=1", UserWarning, stacklevel=2)
        return xr.ones_like(flat), {"smile_fallback": True}
    # Contiguous slice + stride over the lit span (dask-friendlier than fancy indexing)
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    sub = ((img.isel(y=slice(y0, y1, stride)).astype("float32") - dark) / flat).compute()
    lp = lowpass_x(sub.median("y"))  # (band, x) smooth cross-track
    ref = lp.sel(band=avail(lp, PAN_BANDS)).median("band")  # (x,) broadband scene profile
    smile = lp / ref
    smile = smile / smile.median("x")
    # Signal-free columns (ref ~ 0) give non-finite smile; leave those uncorrected (smile=1)
    smile = smile.where(np.isfinite(smile), 1.0)
    return smile.astype("float32"), {"smile_fallback": False}


def empirical_frames(img, ref_flat=None, dark_yrange=None, flat_yrange=None, apply_smile=False):
    """
    Derive the empirical correction frames from a raw DN cube (band, y, x).

    Returns (dark, flat, smile, snr, emp_notes):
      dark  : (band, x) residual-dark median over shadow rows, or 0 when no shadow was found.
      flat  : (band, x) sensor high-frequency response (per-scene or packaged fallback).
      smile : (band, x) band-relative smooth cross-track field (1 when it can't be estimated or
              when apply_smile is False, in which case estimate_smile is skipped entirely).
      snr   : (y, x) float32 broadband SNR field above the dark level, or None when no shadow rows.
              snr is not None <=> a residual dark was derived <=> the scene anchors its own zero
              point (calibrate_to_rad then drops the LUT offset).
      emp_notes : JSON-serializable provenance dict (dark rows/source/threshold, flat runs,
              fallbacks fired, per-scene-vs-reference flat RMS) stored in the product's attrs.

    ref_flat is the packaged reference flat (band, x) used as the flat fallback and the RMS
    diagnostic baseline; pass None to disable both. dark_yrange / flat_yrange are optional
    (ylow, yhigh) positional row ranges into the (possibly extent-cropped) cube that override the
    auto dark-row / flat-region detection with user-picked rows. apply_smile (default False) toggles
    the smile estimate; when False the expensive estimate_smile pass is skipped and smile is 1.
    """
    P = panchromatic(img)  # computed once; reused by dark detection, roughness, smile ref
    row_bright = np.nanmedian(P.values, axis=1)
    floor, sigma = dark_floor(row_bright)
    emp_notes: dict = {}

    if dark_yrange is not None:
        d0, d1 = int(dark_yrange[0]), int(dark_yrange[1])
        dark = img.isel(y=slice(d0, d1)).median("y").compute().astype("float32")
        snr = broadband_snr(P, d0, d1)
        emp_notes.update(has_shadow=True, dark_rows=[d0, d1], dark_threshold=None, dark_source="user")
    else:
        dark_mask, thresh, _ = detect_dark_rows(P)
        d0, d1 = longest_run(dark_mask)
        if d1 - d0 >= MIN_DARK_ROWS:
            dark = img.isel(y=slice(d0, d1)).median("y").compute().astype("float32")
            snr = broadband_snr(P, d0, d1)
            emp_notes.update(has_shadow=True, dark_rows=[d0, d1], dark_threshold=float(thresh), dark_source="auto")
        else:
            warnings.warn("no shadow rows found; skipping empirical dark subtraction", UserWarning, stacklevel=2)
            dark = xr.zeros_like(P.isel(y=0), dtype="float32").drop_vars("y")
            snr = None
            emp_notes.update(has_shadow=False, dark_rows=None, dark_threshold=None, dark_source="auto")

    rs, r_raw, lit = row_roughness(P, floor, sigma)
    if flat_yrange is not None:
        a, b = int(flat_yrange[0]), int(flat_yrange[1])
        bad = utils.load_bad_pixel_mask()
        flat = derive_sensor_flat(img, dark, a, b).where(~bad).astype("float32")
        emp_notes.update(
            flat_runs={"user": [a, b]},
            flat_fallback=False,
            flat_ref_rms=_flat_rms(flat, ref_flat) if ref_flat is not None else None,
        )
    else:
        flat, flat_notes = build_flat(img, dark, rs, r_raw, lit, row_bright, ref_flat)
        emp_notes.update(flat_notes)

    if apply_smile:
        smile, smile_notes = estimate_smile(img, dark, flat, lit)
    else:
        smile, smile_notes = xr.ones_like(flat), {"smile_applied": False}
    emp_notes.update(smile_notes)
    return dark, flat, smile, snr, emp_notes
