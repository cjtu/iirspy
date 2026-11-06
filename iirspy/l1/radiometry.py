"""Dark, flat and radiometric scale, the L1 steps from raw DN to radiance [W/m^2/sr/um].

    DN -> (DN - dark) / flat -> 10 * gain * DN       dark in DN, flat relative response with median 1,
                                                       gain the band-median LUT gain in mW/cm^2/sr/um per DN

`subtract_dark` and `divide_flat` take a source name. "issdc_lut" is the packaged ISSDC LUT (`get_gain_offset`: the dark
is its per-element offset in DN, the flat its per-element gain relative to the band median); "issdc_lut_median" is the
dark from its band-median offset, the setup `flatfield_v1` was derived on. "issdc_derived" is the
table fitted to ISSDC's own nci L1 against raw DN (e1g2, 2019-2026; `data/issdc_derived_{gain,offset}_e1g2.csv`):
its gain is the same in every scene and differs from the packaged LUT by 6-8% rms column to column. Its offset is the
median over the fit scenes; ISSDC's per-scene offset varies by ~1 nci unit (0.3% of a typical L1). "flatfield_v1" is
`data/flatfield_v1.zip`, a quarterly static flat; "empirical" is the scene's own frames from `l1.empirical`.

Only the e1g2 LUT (1 ms) is a real calibration; every scene uses it with the gain scaled by 1 ms / exposure, which
is what ISSDC's own nci L1 does (see `get_gain_offset`). `radiometry` can instead read a calib LUT `.npz`
(e.g. `iirspy_recal_e1g2_lut_coeff.npz`) of per-epoch gains fitted on orbit, which replaces dark, flat and scale.
"""

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from iirspy import instrument
from iirspy.instrument import E1_EXPOSURE_MS, PKG_DATA
from iirspy.l1.config import CALIB
from iirspy.l1.empirical import FLAT_SMILE_MIN

KEYSTONE_CSV = "keystone_shift_e1g2.csv"


def subtract_dark(cube, qub, source, *, has_shadow=False, calib_dir=PKG_DATA):
    """`cube` less the (band, x) dark of `source`, in DN: the LUT offset ("issdc_lut"/"issdc_derived") or the scene's own.

    "empirical" with no shadow rows in the scene falls back to "issdc_lut_median".
    """
    if source == "empirical" and has_shadow:
        return cube - cube.empirical_dark
    source = "issdc_lut_median" if source == "empirical" else source
    gain, offset = get_gain_offset(qub, calib_dir=calib_dir, derived=source == "issdc_derived")
    if source == "issdc_lut_median":
        gain, offset = gain.median("x"), offset.median("x")
    return cube + offset / gain


def divide_flat(cube, qub, source, *, calib_dir=PKG_DATA):
    """`cube` divided by the (band, x) flat of `source`: the table's relative response, median 1 over x."""
    if source == "empirical":
        flat = (cube.empirical_flat * cube.empirical_smile).clip(min=FLAT_SMILE_MIN)
    elif source == "flatfield_v1":
        flat = flatfield_v1(instrument.scene_start(qub), calib_dir)
    else:
        gain, _ = get_gain_offset(qub, calib_dir=calib_dir, derived=source == "issdc_derived")
        flat = gain.median("x") / gain
    return cube / flat


def flatfield_v1(t, calib_dir=PKG_DATA):
    """The flat (band, x) of the quarter containing `t`, from `data/flatfield_v1.zip` (`flatfield_v1_<year>_q<N>.csv`).

    Dates outside 2019-2026 use the nearest quarter.
    """
    t = pd.Timestamp(t)
    year, quarter = (2019, 1) if t.year < 2019 else (2026, 4) if t.year > 2026 else (t.year, t.quarter)
    flat = instrument.read_table("flatfield_v1.zip", f"flatfield_v1_{year}_q{quarter}.csv", root=calib_dir)
    return xr.DataArray(flat.astype("float32"), coords={"band": 1 + np.arange(256), "x": 0.5 + np.arange(250)})


def radiometry(dn, qub, calib="lut", *, calib_dir=PKG_DATA):
    """`dn` -> radiance [W/m^2/sr/um]: the band-median gain of the "lut" or "issdc_derived" table, or a calib LUT `.npz` path.

    A calib LUT's provenance rides along as attrs and its bad-element mask as the `calib_bad` (band, x) coord.
    """
    if calib in CALIB:
        return 10 * dn * get_gain_offset(qub, calib_dir=calib_dir, derived=calib == "issdc_derived")[0].median("x")
    gain, offset, bad, attrs = _read_calib_lut(calib, instrument.scene_start(qub))
    rad = 10 * (dn * gain * (E1_EXPOSURE_MS / instrument.get_exposure_duration(qub)) + offset)  # e1 units, as the LUT
    rad.attrs.update(attrs)
    return rad if bad is None else rad.assign_coords(calib_bad=(("band", "x"), bad.values))


def keystone(rad, *, shift, bad_pixel_mask=True, calib_dir=PKG_DATA):
    """Cross-track keystone shift of radiance (band, y, x), as ISSDC's nci L1 has it.

    Each band is shifted by q/32 px, q from `data/keystone_shift_e1g2.csv`: 4-tap bicubic (a = -0.75, OpenCV's
    INTER_CUBIC) with replicated edge columns, out(x) = sum_k w_k in(x + floor(d) + k), k = -1..2. `shift` "simple" is
    that. "weighted" drops the flagged elements (`bad_pixel_mask` as in `instrument.mask_bad`, plus `calib_bad`) from
    the sum and renormalises, so they never leak into neighbours, and returns the flags moved to the output as a
    `keystone_bad` (band, x) coord (flagged where the nearest tap was bad).
    """
    out = np.array(rad.values, dtype="float32")
    q = instrument.read_table(KEYSTONE_CSV, root=calib_dir)[rad.band.values - 1, 0] / 32
    weighted = shift == "weighted"
    bad = instrument.load_bad_pixel_mask(calib_dir) if bad_pixel_mask is True else bad_pixel_mask
    if "calib_bad" in rad.coords:
        bad = rad.calib_bad if bad is False else bad | rad.calib_bad
    obad = np.zeros(rad.shape[::2], bool)
    for i, d in enumerate(q):
        i0, f = int(np.floor(d)), d - np.floor(d)
        w = (
            -0.75 * (f**3 - 2 * f**2 + f),
            1.25 * f**3 - 2.25 * f**2 + 1,
            -1.25 * f**3 + 1.5 * f**2 + 0.75 * f,
            0.75 * (f**3 - f**2),
        )
        tap = lambda a, k, i0=i0: a[..., 1 + i0 + k : 1 + i0 + k + 250]
        pad = np.pad(rad.values[i], ((0, 0), (2, 2)), mode="edge").astype("float64")
        if not weighted:
            out[i] = sum(wk * tap(pad, k) for k, wk in enumerate(w))
            continue
        gpad = np.pad(~np.asarray(bad.values[i] if bad is not False else np.zeros(250, bool)), (2, 2), mode="edge")
        num = sum(wk * tap(np.where(gpad, pad, 0.0), k) for k, wk in enumerate(w))
        den = sum(wk * tap(gpad.astype(float), k) for k, wk in enumerate(w))
        ok = den > 0.3
        out[i] = np.where(ok, num / np.where(ok, den, 1), np.nan)
        obad[i] = ~tap(gpad, 1 + int(f >= 0.5))  # nearest input tap
        obad[i] |= ~ok
    out = rad.copy(data=out)
    return out.assign_coords(keystone_bad=(("band", "x"), obad)) if weighted else out


def edge_copy(cube):
    """`cube` with columns 0-1 overwritten by column 2 and 248-249 by 247, as ISSDC does (discards those readings)."""
    out = cube.copy(deep=True)
    out[..., :2], out[..., 248:] = out[..., 2:3].values, out[..., 247:248].values
    if "keystone_bad" in out.coords:
        kb = out.keystone_bad.values.copy()
        kb[:, :2], kb[:, 248:] = kb[:, 2:3], kb[:, 247:248]
        out = out.assign_coords(keystone_bad=(("band", "x"), kb))
    return out


def get_gain_offset(fimg, denoise=False, gain_z=None, offset_z=None, calib_dir=PKG_DATA, derived=False):
    """
    Return the IIRS gain and offset for fimg as DataArrays, in [mW/cm^2/sr/um] per DN.

    Only ch2_iirs_cal_e1g2_lut_coeff.csv is a real calibration. The e2g2/e3g2/e4g2 tables are
    placeholders: gain ~ 1.0 at every (band, x) with no spectral structure at all, ~360x flatter
    across band than the true response. Used as shipped they put radiance ~1000x high, every pixel
    trips the saturation cut in `mask`, and the whole cube is nulled.

    ISSDC's own L1 product does not use them either. Differencing the e2g2 scene
    20201203T1859574285 against its nci L1 gives L1 = 333.6 +/- 3.8 * gain_e1g2 * DN over bands
    5-20 (a flat ratio, so the e1g2 spectral AND cross-track shape is what ISSDC applied), i.e.
    exactly gain_e1g2 / 3 in this module's units - and that scene's label reads
    isda:exposure_duration = 3 ms against e1's 1 ms. So this function always loads the e1g2 table
    and scales gain by E1_EXPOSURE_MS / exposure_duration - a no-op for e1g2 itself (1 ms).

    Note the e2g2/e3g2/e4g2 saturations_radiance tables sit ~3x above the resulting full-scale
    radiance, so the saturation cut cannot fire for those scenes (safe, but not a real check).

    `denoise` replaces speckle (robust z above `gain_z` / `offset_z` in a 5-column window) with NaN. `derived` loads the
    `data/issdc_derived_*_e1g2.csv` tables instead of the packaged LUT.
    """
    gain_scale = E1_EXPOSURE_MS / instrument.get_exposure_duration(fimg)
    if gain_z is None:
        gain_z = 0.01  # Empirically selected z-thresholds for gain, offset
    if offset_z is None:
        offset_z = 0.5
    if derived:
        lut = np.stack(
            [instrument.read_table(f"issdc_derived_{k}_e1g2.csv", root=calib_dir) for k in ("gain", "offset")], -1
        ).astype("float32")
    else:
        lut = (
            instrument
            .read_table("iir/calibration/ch2_iirs_cal_e1g2_lut_coeff.csv", root=calib_dir)
            .reshape((256, 250, 2))
            .astype("float32")
        )
    lut[:, :, 0] *= gain_scale
    coords = {"band": 1 + np.arange(0, 256), "x": 0.5 + np.arange(250)}
    gain = xr.DataArray(lut[:, :, 0], coords=coords, name="gain")
    off = xr.DataArray(lut[:, :, 1], coords=coords, name="offset")
    if denoise:
        # Outlier detection with moving window and Z-score threshold
        gwindow = gain.rolling({"band": 1, "x": 5}, center=True, min_periods=1)
        gresid = gain - gwindow.median()
        gzscore = abs(gresid / gresid.std())
        gain = gain.where(gzscore < gain_z)

        owindow = off.rolling({"band": 1, "x": 5}, center=True, min_periods=1)
        oresid = off - owindow.median()
        ozscore = abs(oresid / oresid.std())
        off = off.where(ozscore < offset_z)
    return gain, off


def _read_calib_lut(fnpz, t):
    """Gain, offset, bad-pixel mask (or None) and provenance attrs of the calib LUT epoch covering `t`.

    The npz holds `gain` (n_epoch, 256, 250) and `offset` (n_epoch, 256) or (256,), both in the LUT's
    units [mW/cm^2/sr/um per DN] for e1 (1 ms; scaled by exposure like the LUT), `epoch_start` (n_epoch,) ISO
    dates, optional `mask` (n_epoch, 256, 250) bool (True = bad) and `version`. The epoch is the latest
    `epoch_start` <= `t`. A `t` more than one median epoch spacing past the last epoch warns.
    """
    z = np.load(fnpz)
    starts = pd.DatetimeIndex(z["epoch_start"])
    t = pd.Timestamp(t).tz_localize(None) if pd.Timestamp(t).tzinfo else pd.Timestamp(t)  # label times end in Z
    e = int(starts.searchsorted(t, side="right")) - 1
    if not starts.is_monotonic_increasing:
        raise ValueError(f"{fnpz}: epoch_start must be ascending")
    if e < 0:
        raise ValueError(f"{fnpz}: scene start {t} precedes the first epoch {starts[0]}")
    if len(starts) > 1 and t > starts[-1] + starts.to_series().diff().median():
        warnings.warn(f"{fnpz}: scene start {t} is past the last epoch start {starts[-1]} (open-ended)", stacklevel=2)
    if z["gain"].shape != (len(starts), 256, 250):
        raise ValueError(f"{fnpz}: gain {z['gain'].shape} is not (n_epoch, 256, 250)")
    coords = {"band": 1 + np.arange(256), "x": 0.5 + np.arange(250)}
    gain = xr.DataArray(z["gain"][e].astype("float32"), coords=coords)
    off = np.broadcast_to(z["offset"][e] if z["offset"].ndim == 2 else z["offset"], (256,))
    offset = xr.DataArray(off.astype("float32"), coords={"band": coords["band"]})
    if "mask" in z and z["mask"].shape != (len(starts), 256, 250):
        raise ValueError(f"{fnpz}: mask {z['mask'].shape} is not (n_epoch, 256, 250)")
    mask = xr.DataArray(z["mask"][e].astype(bool), coords=coords) if "mask" in z else None
    attrs = {
        "calib_lut": str(fnpz),
        "calib_lut_sha256": hashlib.sha256(Path(fnpz).read_bytes()).hexdigest(),
        "calib_lut_version": str(z["version"]) if "version" in z else "",
        "calib_lut_epoch": str(starts[e].date()),
    }
    return gain, offset, mask, attrs
