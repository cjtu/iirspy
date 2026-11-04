"""IIRS instrument constants and packaged tables: band wavelengths, bad bands and elements, exposure/gain, LUTs.

Everything here is a fixed property of the sensor or of one scene's label, read by every level. Calibration files are
named by exposure + gain, e.g. `ch2_iirs_cal_e1g2_lut_coeff.csv`, under `DCALIB`; read packaged files through
`PKG_DATA` (importlib.resources), never by path.
"""

from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pdr
import xarray as xr

PKG_DATA = files("iirspy").joinpath("data")
DCALIB = PKG_DATA.joinpath("iir/calibration")
FBADBANDS = str(PKG_DATA.joinpath("iirs_bad_bands.csv"))
FBADPIXELS = str(PKG_DATA.joinpath("bad_pixel_mask.csv"))
FWAVELENGTHS = str(PKG_DATA.joinpath("iir/miscellaneous/ch2_iirs_wavelength.csv"))
OSF = (*range(29, 35), *range(69, 76), *range(162, 172))  # Order sorting filters
INVALID = (*range(1, 7), *range(252, 257))  # Invalid band list
PAN_BANDS = [b for b in range(10, 251, 8) if b not in OSF and b not in INVALID]  # broadband, beats single-band noise

# IIRS L1 radiance is stored in [1000 mW/cm^2/sr/um]; multiply to get SI units [W/m^2/sr/um].
RAD_NATIVE_SCALE = 0.01  # [1000 mW/cm^2/sr/um] -> [W/m^2/sr/um]
E1_EXPOSURE_MS = 1.0  # exposure duration the e1g2 gain LUT was measured at (see get_gain_offset)


def get_wls(fwavelengths=FWAVELENGTHS):
    """Return the wavelength [nm] for each band from IIRS wavelengths file."""
    df = pd.read_csv(fwavelengths, header=0, names=["band", "wl"], usecols=[0, 1])
    return df["wl"].values


def get_wls_xml(fname):
    """Return the wavelength [nm] for each band from image metadata."""
    img = pdr.open(fname)
    wl_dict = list(img.metaget("Band_Bin_Set").values())
    return np.array([float(wl["center_wavelength"]) for wl in wl_dict])


def get_wl_labels(fname):
    """Return labels for wavelegnths in str format, e.g. for an ENVI header."""
    wls = get_wls(fname)
    return [f"{wl} nm" for wl in wls]


def load_bad_bands(fbad_bands=FBADBANDS):
    """Load bad bands from a CSV file."""
    # Negates the df since provided bad bands file is 1 for good, 0 for bad
    return ~pd.read_csv(fbad_bands, index_col=0).astype(bool)


def load_bad_pixel_mask(fbad_pixels=FBADPIXELS):
    """Return the per-detector-element bad pixel mask as a (band, x) DataArray, True where bad."""
    mask = pd.read_csv(fbad_pixels, header=None).values.astype(bool)
    coords = {"band": 1 + np.arange(0, 256), "x": 0.5 + np.arange(250)}
    return xr.DataArray(mask, coords=coords, name="bad_pixel")


def get_exposure_gain(fimg):
    """Return exposure (E1-E4) and gain (G2) as eXgY string."""
    img = pdr.open(fimg)
    exposure = img.metaget("isda:exposure")
    gain = img.metaget("isda:gain")
    return f"{exposure}{gain}".lower()


def get_bad_bands(fimg, buffer=0):
    """
    Return bad bands for the given image (depends on exposure and gain).

    Set buffer number of bands to left and right of each bad band as bad.
    """
    exp_gain = get_exposure_gain(fimg)  # e.g., "e1g2"
    bad_bands = load_bad_bands()[exp_gain].values  # array of True / False

    # Buffer - use kernel convolution to bump n adjacent bands, flag these as also bad
    # Ex. [0, 1, 1, 1, 0, 1, 1] with buffer 1 => [0, 0, 1, 0, 0, 0, 1]
    kernel_size = 2 * buffer + 1
    kernel = np.ones(kernel_size)
    bad_buffered = np.convolve((bad_bands).astype(int), kernel, mode="same").astype(bool)

    return bad_buffered  # array of True where band is bad


def get_lut_file(fimg, lut_type="lut_coeff", calib_dir=DCALIB):
    """
    Locate and return the correct LUT file for the given image.

    Parameters
    ----------
    fimg : str
        Path to the IIRS image file.
    lut_type : str
        Type of LUT to find (default: "lut_coeff").
        Options: "lut_coeff", "saturations_radiance"
    calib_dir : str or Path
        Directory containing calibration LUT files.
    ext : str
        File extension to look for (default: "csv").
        Options: "csv", "xml"

    Returns
    -------
    Path to the LUT file as a string.

    Raises
    ------
    FileNotFoundError if the LUT file is not found.
    """
    exp_gain = get_exposure_gain(fimg)  # e.g., "e1g2"
    pattern = f"ch2_iirs_cal_{exp_gain}_{lut_type}.csv"
    flut = Path(calib_dir) / pattern
    if flut.exists():
        return str(flut)
    raise FileNotFoundError(f"IIRS calibration file {flut} not found.")


def get_exposure_duration(fimg):
    """Return the commanded exposure duration [ms] from the IIRS label (e1 -> 1, e2 -> 3, ...).

    Not to be confused with isda:line_exposure_duration (the 53.06 ms line period, identical
    across exposure settings).
    """
    return float(pdr.open(fimg).metaget("isda:exposure_duration"))


def get_saturation_radiance(fimg, calib_dir=DCALIB):
    """Return IIRS saturation radiance file as 1D DataArray along band."""
    flut = get_lut_file(fimg, "saturations_radiance", calib_dir)
    lut = np.loadtxt(flut, delimiter=",", usecols=2).astype("float32")  # band, wl, saturation [1000 mW/cm^2/sr/um]
    return RAD_NATIVE_SCALE * xr.DataArray(lut, coords={"band": np.arange(1, 257)}, name="saturation [W/m^2/sr/um]")
