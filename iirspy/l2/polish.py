import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter
from scipy.stats import norm

from iirspy.instrument import PKG_DATA

FPOLISH = str(PKG_DATA.joinpath("spectral_polish_verma2022.csv"))


# Per-band polish inferred from published ISSDC L2 (NDI) products; bands without a fit are 1.0
FPOLISH_ISSDC_EMPIRICAL = str(PKG_DATA.joinpath("spectral_polish_issdc_empirical.csv"))


def spectral_polish(band, fpolish=FPOLISH):
    """Per-band polish divisor from csv `fpolish` (`band`, `polish` columns), mapped onto `band` by
    band number; bands the table lacks get 1.0."""
    df = pd.read_csv(fpolish)
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
    """Smooth the spectra using boxcar or Gaussian smoothing."""
    if smoothing.lower() == "boxcar":
        # Moving average over wavelength (test this since .mean on a rolling array may not handle NaNs correctly)
        da = da.rolling({"band": swindow}, center=True, min_periods=swindow / 2).mean(["band"])
    elif smoothing.lower() == "gaussian":
        # make a gaussian length of swindow, normalize, apply to rolling window as dot product
        # note: this interpolates NaN to allow spectra to be smooth up to a NaN band
        #  we then need to reapply the NaNs at the end. More steps and more expensive than moving avg
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
    elif smoothing.lower() == "savgol":
        da = smooth_savgol(da, swindow)  # Assumes polyorder=2
    return da


def smooth_savgol(data: xr.DataArray, savgol_window: int = 9, savgol_polyorder: int = 2) -> xr.DataArray:
    """
    Savitzky-Golay filtering for spectral smoothing. Must have no NaNs.
    """

    def filter_spectrum(spectrum):
        """Apply Savitzky-Golay to a single spectrum."""
        # needed for dask
        spectrum = np.array(spectrum, copy=True)
        # TODO (optional): Can remove spikes here using method from https://doi.org/10.1029/2024JE008842

        # Savitzky-Golay
        spectrum = savgol_filter(spectrum, window_length=savgol_window, polyorder=savgol_polyorder)
        return spectrum

    # Apply combined filter
    result: xr.DataArray = xr.apply_ufunc(
        filter_spectrum,
        data,
        input_core_dims=[["band"]],
        output_core_dims=[["band"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    return result
