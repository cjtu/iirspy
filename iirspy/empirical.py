"""
On-orbit empirical residual-dark / flat / smile correction for IIRS L0->L1 calibration.

The IIRS gain/offset LUT is applied per detector element (band, x), but its cross-track
structure is poorly correlated with the true on-orbit detector response, injecting striping
and speckle. This module derives three corrections from the scene itself and lets the L0->L1
step use the LUT only for the per-band absolute scale (see iirs.L0.calibrate_to_rad):

  dark_resid(band, x) : median DN over the scene's shadow rows. L0 counts are already
                        dark-subtracted onboard (pre-imaging night-side dark), so this is a
                        *residual* dark - drift + hot/warm-pixel residuals + stray light. Its
                        real job is per-element hot-pixel refinement. 0 when no shadow is found.
  flat(band, x)       : sensor high-frequency response, R / lowpass_x(R), median-composited over
                        the flattest rows of each brightness quantile bin. A hardware property,
                        so it is transferable; falls back to a packaged reference flat when the
                        scene has no qualifying flat region.
  smile(band, x)      : band-relative smooth cross-track field. Only spectrally-varying smooth
                        structure is divided out; the broadband scene gradient is preserved.

Shadow and lit rows are both cut on the broadband SNR field (SHADOW_SNR, LIT_SNR), whose noise
unit is the per-pixel scatter measured in the scene's own shadow.

Correction model (shadow anchors the zero point; see SPEC / calibrate_to_rad):

  shadow scene   : rad = 10 * gain_med * (DN - dark_resid) / (flat * smile)   # LUT offset dropped
  no-shadow scene: rad = 10 * (gain_med * DN / (flat * smile) + offset_med)   # LUT offset kept

Everything here operates on the raw DN DataArray (band, y, x).
"""

import warnings
from functools import cache

import numpy as np
import xarray as xr

import iirspy.utils as utils

# --- Parameters (single source: the functions below hold no bare numbers) ---------------------
_OSF, _INV = set(utils.OSF), set(utils.INVALID)

# Band selections
PAN_BANDS = [b for b in range(10, 251, 8) if b not in _OSF and b not in _INV]  # broadband, beats single-band noise
N_CLEAN_BANDS = 8  # how many least-bad bands clean_bands() picks for spatial-outlier detection

# Residual-dark / shadow detection (location-agnostic: shadow is found wherever it is, if at all)
DARK_PCT = 0.5  # darkest percentile of rows used to estimate the shadow floor + noise
DARK_K = 8.0  # row shadow threshold = floor + DARK_K * sigma
DARK_SIGMA_FLOOR = 0.15  # min row-median scatter [DN], keeps the bootstrap threshold off zero
DARK_SMOOTH_FRAC = 0.25  # shadow rows carry under this share of the scene's typical cross-track structure
DARK_PEDESTAL_DN = 25.0  # a dark frame above this did not have its onboard subtraction applied
NOISE_FLOOR = 0.5  # min noise [DN] to avoid divide-by-zero in the SNR
SHADOW_SNR = 2.0  # broadband SNR below this is shadow (per-pixel dark noise, as LIT_SNR)

# Rows needed for any robust median over y: the dark block, a flat window, the lit span for smile.
MIN_ROWS = 200

# Flat-region detection (per-row roughness -> flattest fixed-length window per brightness bin)
CROSS_WIN = 31  # cross-track window: high-pass for roughness, lowpass for the sensor/smile split
YSMOOTH = 51  # along-track smoothing of the per-row roughness profile
ROUGH_PCTL = (5, 95)  # tail-sensitive percentile range taken as a row's roughness
DEAD_COL_FRAC = 0.2  # columns below this fraction of the median profile are dead, dropped
LIT_SNR = 10.0  # lit rows sit above this broadband SNR (per-pixel dark noise, as SHADOW_SNR)
N_DN_BINS = 4  # flat sampled over this many quantile bins of the scene's own lit brightness
OUTLIER_Z = 4.0  # robust-z beyond which a pixel is scene structure, not sensor response

# Compute guards: cost and runaway limits, not calibration choices
SMILE_STRIDE = 8  # along-track subsample stride for the smile estimate

# Constant: MAD -> standard deviation for a normal distribution, 1 / inverse_cdf(3/4)
MAD_TO_SIGMA = 1.4826


def lowpass_x(da, w=CROSS_WIN):
    """Smooth a (..., x) DataArray across x with a centered rolling median."""
    return da.rolling(x=w, center=True, min_periods=1).median()


def robust_z(resid, axis):
    """Nan-safe robust (MAD-based) z-score of resid along axis."""
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid, axis=axis, keepdims=True)), axis=axis, keepdims=True)
    robust_std = MAD_TO_SIGMA * mad
    return np.divide(resid, robust_std, out=np.full_like(resid, np.nan), where=robust_std > 0)


def avail(img, bands):
    """Intersect `bands` with the bands the cube carries, so band subsets calibrate. Empty raises."""
    have = set(np.asarray(img.band.values).tolist())
    sel = [b for b in bands if b in have]
    if not sel:
        raise ValueError(f"cube carries none of bands {bands[:5]}...; cannot derive the empirical correction")
    return sel


@cache
def clean_bands():
    """The N_CLEAN_BANDS bands with the fewest bad pixels (ties by band number), OSF/invalid excluded.

    These are the bands used to read scene structure rather than sensor response, so the fewer bad
    pixels they carry the better. Derived from the packaged mask so it tracks mask updates.
    """
    mask = utils.load_bad_pixel_mask()
    bands, counts = mask.band.values, mask.sum("x").values
    ok = np.array([b not in _OSF and b not in _INV for b in bands])
    order = np.lexsort((bands[ok], counts[ok]))
    return sorted(int(b) for b in bands[ok][order[:N_CLEAN_BANDS]])


def on_bands(frame, img):
    """Restrict a packaged (band, x) frame to the bands `img` carries, so band subsets line up."""
    return frame.sel(band=avail(img, list(np.asarray(frame.band.values))))


def panchromatic(img):
    """Band-averaged (y, x) brightness over PAN_BANDS (eager)."""
    return img.sel(band=avail(img, PAN_BANDS)).mean("band").compute()


def _xr_yx(mask_yx, like):
    """Wrap a (y, x) numpy bool mask as a DataArray aligned to like's y, x coords."""
    return xr.DataArray(mask_yx, coords={"y": like.y, "x": like.x}, dims=("y", "x"))


def runs(mask):
    """Contiguous True runs of a 1-D bool array, as [(start, stop), ...]."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    return [(int(r[0]), int(r[-1]) + 1) for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)]


def longest_run(mask):
    """Return (start, stop) of the longest contiguous True run in a 1-D bool array."""
    return max(runs(mask), key=lambda r: r[1] - r[0], default=(0, 0))


def live_cols(P):
    """Columns carrying real response: those above DEAD_COL_FRAC of the median column profile."""
    prof = np.nanmedian(P.values, axis=0)
    return prof > DEAD_COL_FRAC * np.nanmedian(prof)


def cross_track_hf(P, win=CROSS_WIN):
    """Per-row scatter [DN] left by a cross-track high-pass: terrain contrast, independent of level.

    The one shadow signature that survives a missing dark subtraction. A shadowed row holds no
    albedo structure, so neighbouring samples differ only by detector noise whatever DN the row
    sits at, while a merely dim row keeps its terrain contrast. Measured on 20201203T1859574285
    (e2g2, dark pedestal): shadow rows 1.7 DN against a scene median of 16.3, i.e. 0.6% of level
    against lit terrain's 3.0%.
    """
    from scipy.ndimage import uniform_filter1d

    v = P.values[:, live_cols(P)]
    hp = v - uniform_filter1d(v, win, axis=1, mode="nearest")
    return np.nanmedian(np.abs(hp - np.nanmedian(hp, axis=1, keepdims=True)), axis=1) * MAD_TO_SIGMA


def is_shadow(P, block, hf=None):
    """True when `block` holds no cross-track scene structure - shadow, not merely dim terrain.

    Level cannot decide this. A scene whose onboard dark subtraction never happened puts its
    shadow at hundreds of DN (20201203T1859574285: 291-347, against 0-10 across every e1g2 scene
    here), which any absolute or scene-relative level cut reads as "too bright to be shadow" and
    throws the dark frame away. Structure decides it at any pedestal.
    """
    hf = cross_track_hf(P) if hf is None else hf
    a, b = block
    return bool(np.nanmedian(hf[a:b]) < DARK_SMOOTH_FRAC * np.nanmedian(hf[np.isfinite(hf)]))


def dark_floor(row_bright):
    """Return (floor, sigma): dark level and robust scatter of the darkest DARK_PCT of rows."""
    lo = row_bright[row_bright <= np.nanpercentile(row_bright, DARK_PCT)]
    floor = np.nanmedian(lo)
    sigma = max(np.nanmedian(np.abs(lo - floor)) * MAD_TO_SIGMA, DARK_SIGMA_FLOOR)
    return floor, sigma


def detect_dark_rows(P):
    """
    Return (dark_mask_y, threshold, row_bright): a first pass at the shadow rows, from raw DN.

    Cuts DARK_K robust standard deviations above the dark floor, wherever in the scene the darkest
    rows sit (no assumption of a leading shadow). This is a bootstrap on level alone and says
    nothing about whether the scene holds a shadow at all - `is_shadow` rules on that, after
    refine_dark_block sharpens the edge on the SNR scale.
    """
    row_bright = np.nanmedian(P.values, axis=1)
    floor, sigma = dark_floor(row_bright)
    thresh = floor + DARK_K * sigma
    return row_bright < thresh, thresh, row_bright


def broadband_snr(P, d0, d1):
    """
    Return the per-pixel broadband SNR field (y, x) float32 above the dark level of rows [d0, d1).

    Signal is P minus the per-column dark level; noise is the per-column scatter within [d0, d1).
    Working on the panchromatic average keeps it meaningful where single long-wavelength bands are
    noisy. SHADOW_SNR and LIT_SNR cut this field; it also ships with the product for downstream use.
    """
    ref = P.isel(y=slice(d0, d1))
    pan_dark = ref.median("y")  # (x,) broadband dark level
    pan_noise = ref.std("y").clip(min=NOISE_FLOOR)  # (x,) broadband noise
    return ((P - pan_dark) / pan_noise).astype("float32")


def refine_dark_block(P, block):
    """Grow `block` to the full run of rows with median broadband SNR below SHADOW_SNR.

    `block` supplies the noise scale, so the shadow edge lands on the same footing as LIT_SNR.
    Only runs touching `block` are eligible - a dim patch elsewhere in the scene can also read
    below SHADOW_SNR, and taking it would put lit pixels in the dark frame. Falls back to `block`
    when nothing touching it is long enough for a stable median.
    """
    shadow = np.nanmedian(broadband_snr(P, *block).values, axis=1) < SHADOW_SNR
    a, b = block
    touching = [r for r in runs(shadow) if r[0] < b and r[1] > a]
    d0, d1 = max(touching, key=lambda r: r[1] - r[0], default=block)
    return (d0, d1) if d1 - d0 >= MIN_ROWS else block


def lit_rows(P, dark_block):
    """Rows bright enough to derive a flat or smile from: median broadband SNR above LIT_SNR.

    Without a dark block there is no in-scene noise to measure, and no shadow either, so every row
    with a finite brightness counts as lit.
    """
    if dark_block is None:
        return np.isfinite(np.nanmedian(P.values, axis=1))
    return np.nanmedian(broadband_snr(P, *dark_block).values, axis=1) > LIT_SNR


def row_roughness(P):
    """
    Per-row scene roughness rs(y), smoothed over YSMOOTH rows.

    N = P / global_prof removes the fixed detector stripe/vignette (dead columns dropped). A
    cross-track high-pass leaves local scene features (crater walls, rims); a row's roughness is
    the ROUGH_PCTL range of that, relative to the row median. rs is r smoothed over YSMOOTH rows.
    """
    from scipy.ndimage import uniform_filter1d

    vals = P.values
    global_prof = np.nanmedian(vals, axis=0)
    good = live_cols(P)
    N = vals[:, good] / global_prof[good]
    hp = N - uniform_filter1d(N, CROSS_WIN, axis=1, mode="nearest")
    plo, phi = np.nanpercentile(hp, ROUGH_PCTL, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (phi - plo) / np.nanmedian(N, axis=1)
    # Signal-free rows (median 0) have non-finite roughness. uniform_filter1d carries a running
    # sum, so smoothing them in place would spread that across the whole profile: smooth a filled
    # copy, then mark those rows infinitely rough so no flat region can include them.
    bad = ~np.isfinite(r)
    rs = uniform_filter1d(np.where(bad, np.nanmedian(r[~bad]) if (~bad).any() else 0.0, r), YSMOOTH, mode="nearest")
    rs[bad] = np.inf
    return rs


def flattest_rows(rs, mask, nrows=MIN_ROWS):
    """Indices of the `nrows` lowest-roughness rows inside `mask`; None if that many don't exist.

    Rows need not be contiguous: the flat is a per-column median over y, so row order carries no
    information, and scattered rows sample the sensor over more independent terrain. Fixed count
    rather than an adaptive roughness tolerance - taking more rows adds to the median but not the
    accuracy, and one count needs no tuning.
    """
    cand = np.flatnonzero(np.isfinite(rs) & mask)
    if cand.size < nrows:
        return None
    return np.sort(cand[np.argpartition(rs[cand], nrows - 1)[:nrows]])


def spatial_outlier_mask(fsub):
    """
    (y, x) bool mask, True at pixels that deviate strongly from the flat clean-band scene plane.

    Normalizes each clean band by its spatial mean, median-stacks to a (y, x) structure map, and
    flags pixels > 4 robust-z from the plane median (both bright and dark). These are scene
    structure - crater walls/shadows, bright rims, residual bad pixels - excluded so the flat's
    median over y isn't biased by real spatial features rather than sensor response.
    """
    cb = fsub.sel(band=avail(fsub, clean_bands()))
    struct = (cb / cb.mean(("y", "x"))).median("band").values
    z = robust_z(struct - np.nanmedian(struct), axis=(0, 1))
    return np.abs(z) > OUTLIER_Z


def derive_sensor_flat(img, dark, rows):
    """Sensor flat (band, x) = R / lowpass_x(R) from `rows`: high-frequency response only."""
    fsub = (img.isel(y=rows).astype("float32") - dark).compute()
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


def dn_bins(row_bright, lit, nbins=N_DN_BINS):
    """`nbins` equal-count brightness bins over the lit rows, as [(lo, hi), ...], last hi = inf.

    Quantiles rather than fixed DN cuts, so the flat is sampled across whatever brightness range
    the scene actually spans - polar scenes are far darker than equatorial ones.
    """
    # Equal-count bins, so cap nbins at what leaves every bin MIN_ROWS rows to draw a flat from.
    nbins = min(nbins, int(lit.sum()) // MIN_ROWS)
    if nbins < 1:
        return []
    edges = np.nanpercentile(row_bright[lit], np.linspace(0, 100, nbins + 1))
    return [(float(lo), float(hi)) for lo, hi in zip(edges[:-1], [*edges[1:-1], np.inf], strict=True)]


def build_flat(img, dark, rs, lit, row_bright, ref_flat=None):
    """
    Median-composite sensor flat (band, x) over the flattest region of each brightness bin.

    Falls back to the packaged reference flat (with a UserWarning) when no in-scene region
    qualifies. Never raises. Returns (flat, emp_notes) recording the rows used, whether the
    fallback fired, and the per-scene-vs-reference RMS deviation (the flat stability metric).

    emp_notes["flat_runs"][bin] is [first, last + 1] of the rows sampled in that bin; the rows
    are scattered within that span, not a solid block.
    """
    bad = on_bands(utils.load_bad_pixel_mask(), img)
    ref_flat = None if ref_flat is None else on_bands(ref_flat, img)
    flats, runs_used = [], {}
    for lo, hi in dn_bins(row_bright, lit):
        rows = flattest_rows(rs, lit & (row_bright >= lo) & (row_bright < hi))
        runs_used[f"{lo:.0f}-{hi:.0f}"] = None if rows is None else [int(rows[0]), int(rows[-1]) + 1]
        if rows is not None:
            flats.append(derive_sensor_flat(img, dark, rows))
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
    if ys.size < MIN_ROWS:
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
    emp_notes: dict = {}
    dark_block = None

    if dark_yrange is not None:
        d0, d1 = int(dark_yrange[0]), int(dark_yrange[1])
        dark = img.isel(y=slice(d0, d1)).median("y").compute().astype("float32")
        snr = broadband_snr(P, d0, d1)
        dark_block = (d0, d1)
        emp_notes.update(has_shadow=True, dark_rows=[d0, d1], dark_threshold=None, dark_source="user")
    else:
        dark_mask, thresh, _ = detect_dark_rows(P)
        d0, d1 = longest_run(dark_mask)
        if d1 - d0 >= MIN_ROWS:
            d0, d1 = refine_dark_block(P, (d0, d1))
        if d1 - d0 >= MIN_ROWS and is_shadow(P, (d0, d1)):
            dark = img.isel(y=slice(d0, d1)).median("y").compute().astype("float32")
            snr = broadband_snr(P, d0, d1)
            dark_block = (d0, d1)
            level = float(np.nanmedian(row_bright[d0:d1]))
            emp_notes.update(
                has_shadow=True,
                dark_rows=[d0, d1],
                dark_threshold=float(thresh),
                dark_source="auto",
                dark_level=round(level, 3),
            )
            if level > DARK_PEDESTAL_DN:
                # The shadow anchors the zero point either way, so the product is still calibrated
                # - but a pedestal this large drifts with the detector, and the single frame
                # derived here only holds near the rows it came from.
                warnings.warn(
                    f"shadow rows sit at {level:.0f} DN, not ~0: the onboard dark subtraction looks "
                    f"absent for this scene, so the empirical dark carries the full pedestal",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn("no shadow rows found; skipping empirical dark subtraction", UserWarning, stacklevel=2)
            dark = xr.zeros_like(P.isel(y=0), dtype="float32").drop_vars("y")
            snr = None
            emp_notes.update(has_shadow=False, dark_rows=None, dark_threshold=None, dark_source="auto")

    lit = lit_rows(P, dark_block)
    rs = row_roughness(P)
    if flat_yrange is not None:
        a, b = int(flat_yrange[0]), int(flat_yrange[1])
        bad = on_bands(utils.load_bad_pixel_mask(), img)
        flat = derive_sensor_flat(img, dark, np.arange(a, b)).where(~bad).astype("float32")
        emp_notes.update(
            flat_runs={"user": [a, b]},
            flat_fallback=False,
            flat_ref_rms=_flat_rms(flat, on_bands(ref_flat, img)) if ref_flat is not None else None,
        )
    else:
        flat, flat_notes = build_flat(img, dark, rs, lit, row_bright, ref_flat)
        emp_notes.update(flat_notes)

    if apply_smile:
        smile, smile_notes = estimate_smile(img, dark, flat, lit)
    else:
        smile, smile_notes = xr.ones_like(flat), {"smile_applied": False}
    emp_notes.update(smile_notes)
    return dark, flat, smile, snr, emp_notes
