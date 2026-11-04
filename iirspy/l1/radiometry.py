"""Radiometric scale, the L1 step from (dark/flat-corrected) DN to radiance [W/m^2/sr/um].

    rad = 10 * (gain * DN + offset)     gain, offset in mW/cm^2/sr/um per DN; x10 converts to W/m^2/sr/um

`calib` picks gain and offset: the ISSDC per-element LUT ("lut"), its per-band median ("lut_median", no
cross-track structure), the median on an empirically corrected cube ("empirical", offset dropped when the scene's
own dark was subtracted), or a G1 table `.npz` of per-epoch gains fitted on orbit.

Only the e1g2 LUT (1 ms) is a real calibration; every scene uses it with the gain scaled by 1 ms / exposure, which
is what ISSDC's own nci L1 does (see `get_gain_offset`).
"""

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pdr
import xarray as xr

from iirspy import instrument
from iirspy.instrument import DCALIB, E1_EXPOSURE_MS
from iirspy.l1.config import CALIB


def radiometry(dn, qub, calib="lut", *, calib_dir=DCALIB, has_shadow=False):
    """`dn` -> radiance [W/m^2/sr/um] by `calib`: "lut", "lut_median", "empirical" or a G1 table path.

    `has_shadow` (empirical only) says the scene's residual dark was subtracted, so the LUT offset is dropped. A G1
    table's provenance rides along as attrs and its bad-element mask as the `calib_bad` (band, x) coord.
    """
    if calib in (*CALIB, "empirical"):
        gain, offset = get_gain_offset(qub, calib_dir=calib_dir)
        if calib != "lut":
            gain, offset = gain.median("x"), offset.median("x")
        if calib == "empirical" and has_shadow:
            offset = 0.0
        return 10 * (dn * gain + offset)
    gain, offset, bad, attrs = _read_g1_table(calib, pd.Timestamp(pdr.open(qub).metaget("start_date_time")))
    rad = 10 * (dn * gain * (E1_EXPOSURE_MS / instrument.get_exposure_duration(qub)) + offset)  # e1 units, as the LUT
    rad.attrs.update(attrs)
    return rad if bad is None else rad.assign_coords(calib_bad=(("band", "x"), bad.values))


def load_reference_flat(fimg, calib_dir=DCALIB):
    """
    Return the packaged sensor flat (band, x) for fimg's exposure/gain, or None if not given.

    Prebuilt flat across multiple IIRS scenes (see iirspy.l1.empirical.build_flat),
    used as the fallback when a scene has no qualifying flat region of its own.
    Named ch2_iirs_flat_<expgain>.csv
    """
    exp_gain = instrument.get_exposure_gain(fimg)  # e.g. "e1g2"
    fflat = Path(calib_dir) / f"ch2_iirs_flat_{exp_gain}.csv"
    if not fflat.exists():
        return None
    vals = np.loadtxt(fflat, delimiter=",").astype("float32")
    coords = {"band": 1 + np.arange(0, 256), "x": 0.5 + np.arange(250)}
    return xr.DataArray(vals, coords=coords, name="reference_flat")


def get_gain_offset(fimg, denoise=False, gain_z=None, offset_z=None, calib_dir=DCALIB):
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

    `denoise` replaces speckle (robust z above `gain_z` / `offset_z` in a 5-column window) with NaN.
    """
    flut = str(Path(calib_dir) / "ch2_iirs_cal_e1g2_lut_coeff.csv")
    gain_scale = E1_EXPOSURE_MS / instrument.get_exposure_duration(fimg)
    if gain_z is None:
        gain_z = 0.01  # Empirically selected z-thresholds for gain, offset
    if offset_z is None:
        offset_z = 0.5
    lut = np.loadtxt(flut, delimiter=",").reshape((256, 250, 2)).astype("float32")
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


def _read_g1_table(fnpz, t):
    """Gain, offset, bad-pixel mask (or None) and provenance attrs of the G1 table epoch covering `t`.

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
        "calib_table": str(fnpz),
        "calib_table_sha256": hashlib.sha256(Path(fnpz).read_bytes()).hexdigest(),
        "calib_table_version": str(z["version"]) if "version" in z else "",
        "calib_table_epoch": str(starts[e].date()),
    }
    return gain, offset, mask, attrs
