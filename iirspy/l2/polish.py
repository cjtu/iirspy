"""Spectral polish and smoothing, the late L2 steps that clean up the reflectance spectrum.

`spectral_polish` is a per-band divisor for residual band-to-band calibration structure: the Verma et al. (2022)
table, or one fitted to ISSDC's published L2 so `L2Config.preset("issdc")` reproduces it. `moving_average` is the
3-band smoothing ISSDC applies to its L2. `apply_smoothing` (boxcar, Gaussian, Savitzky-Golay) defines other spectral
smoothing operations.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter
from scipy.stats import norm

from iirspy.instrument import PKG_DATA

FPOLISH = str(PKG_DATA.joinpath("spectral_polish_verma2022.csv"))
# Per-band polish inferred from published ISSDC L2 (NDI) products; bands without a fit are 1.0
FPOLISH_ISSDC_EMPIRICAL = str(PKG_DATA.joinpath("spectral_polish_issdc_empirical.csv"))
POLISH = {"verma2022": FPOLISH, "issdc_empirical": FPOLISH_ISSDC_EMPIRICAL}


def spectral_polish(band, fpolish=FPOLISH):
    """Per-band polish divisor on `band` (by band number; 1.0 where the table has none).

    `fpolish` is a `POLISH` name or a csv with `band` and `polish` columns.
    """
    df = pd.read_csv(POLISH.get(fpolish, fpolish))
    band = np.asarray(band)
    coeff = pd.Series(df["polish"].to_numpy(), index=df["band"].to_numpy()).reindex(band).fillna(1.0)
    return xr.DataArray(coeff.to_numpy(), coords={"band": band})


def moving_average(da, width=3, edges="pair"):
    """Centred `width`-band moving average along band.

    `edges` sets the end bands: "pair" (width 3 only, the Verma et al. 2022 processing): the first
    band is the mean of bands 2-3 and the last the mean of the last two; "nan" leaves them NaN;
    "reflect" pads by reflection.
    """
    if width <= 1:
        return da
    if edges == "reflect":
        n = width // 2
        padded = da.pad(band=n, mode="reflect").rolling(band=width, center=True).mean()
        return padded.isel(band=slice(n, -n)).assign_coords(da.band.coords)
    out = da.rolling(band=width, center=True).mean()
    if edges == "pair":
        if width != 3:
            raise ValueError('edges="pair" is only defined for width 3')
        b = da.band
        first, last = da.isel(band=[1, 2]).mean("band"), da.isel(band=[-2, -1]).mean("band")
        out = xr.where(b == b[0], first, xr.where(b == b[-1], last, out)).transpose(*da.dims)
    return out


def apply_smoothing(da, smoothing, swindow):
    """Smooth the spectra over `swindow` bands: "boxcar", "gaussian" (odd window, NaN-preserving) or "savgol"."""
    if smoothing.lower() == "boxcar":
        da = da.rolling({"band": swindow}, center=True, min_periods=swindow / 2).mean(["band"])
    elif smoothing.lower() == "gaussian":
        # Interpolate NaN so spectra smooth up to a NaN band, then put the NaNs back.
        if not swindow % 2:
            raise ValueError("Gaussian swindow must be odd.")
        gaussian = norm(loc=0, scale=1).pdf(np.arange(swindow) - swindow // 2)
        weights = xr.DataArray(gaussian / gaussian.sum(), dims=["window"])
        da = (
            da
            .rolling({"band": swindow}, center=True, min_periods=1)
            .construct("window")
            .interpolate_na("band")
            .dot(weights)
            .where(~da.isnull())
        )
    elif smoothing.lower() == "savgol":  # polyorder 2; spectra must have no NaNs
        da = xr.apply_ufunc(
            savgol_filter,
            da,
            kwargs={"window_length": swindow, "polyorder": 2, "axis": -1},
            input_core_dims=[["band"]],
            output_core_dims=[["band"]],
            dask="parallelized",
            output_dtypes=[float],
        )
    return da
