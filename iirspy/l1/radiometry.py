from pathlib import Path

import numpy as np
import xarray as xr

from iirspy import instrument
from iirspy.instrument import DCALIB, E1_EXPOSURE_MS


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
    trips the saturation cut in calibrate_to_rad, and the whole cube is nulled.

    ISSDC's own L1 product does not use them either. Differencing the e2g2 scene
    20201203T1859574285 against its nci L1 gives L1 = 333.6 +/- 3.8 * gain_e1g2 * DN over bands
    5-20 (a flat ratio, so the e1g2 spectral AND cross-track shape is what ISSDC applied), i.e.
    exactly gain_e1g2 / 3 in this module's units - and that scene's label reads
    isda:exposure_duration = 3 ms against e1's 1 ms. So this function always loads the e1g2 table
    and scales gain by E1_EXPOSURE_MS / exposure_duration - a no-op for e1g2 itself (1 ms).

    Note the e2g2/e3g2/e4g2 saturations_radiance tables sit ~3x above the resulting full-scale
    radiance, so the saturation cut cannot fire for those scenes (safe, but not a real check).

    Parameters
    ----------
    flut : str
        Path to the lookup table file.
    denoise: bool
        Replace speckly noise from gain / offset with NaN.
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
