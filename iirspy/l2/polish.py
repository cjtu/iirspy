"""Spectral polish and smoothing, the L2 steps that clean up the reflectance spectrum.

`issdc_polish` is ISSDC's polish (SIS Appendix L, step 3). `spectral_polish` is a per-band divisor, the Verma et al.
(2022) table. `moving_average` is the 3-band smoothing of the Verma et al. (2022) plugin. `apply_smoothing` (boxcar,
Gaussian, Savitzky-Golay) defines other spectral smoothing operations.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter
from scipy.stats import norm

from iirspy.instrument import PKG_DATA

FPOLISH = str(PKG_DATA.joinpath("spectral_polish_verma2022.csv"))
POLISH = {"verma2022": FPOLISH}
# ISSDC's polish gain, band 1-256 x column 0-249 of the north-up, east-right NDI image
FGSP = str(PKG_DATA.joinpath("issdc_gsp.csv"))


def issdc_polish(iof, lon):
    """ISSDC's spectral polish of I/F (band, y, x), all 256 bands and 250 columns; `lon` is the scene's longitude
    with x last.

    1. Multiply by the `FGSP` gain. ISSDC indexes it in its east-right image, so it is mirrored when longitude
       falls along x.
    2. With d_b = R_b - (R_b-1 + R_b+1) / 2, replace R_b by (R_b-1 + R_b+1) / 2 where
       |d_b| > 0.9 * rms(d_b-1, d_b, d_b+1), using the unreplaced neighbours.
    """
    if lon is None or iof.sizes["band"] != 256 or iof.sizes["x"] != 250:
        raise ValueError(f"issdc polish needs the scene longitude, 256 bands and 250 columns, got {dict(iof.sizes)}")
    lon = np.atleast_2d(lon)
    row = lon[len(lon) // 2]
    g = pd.read_csv(FGSP, index_col="band").to_numpy()
    if (row[-1] - row[0] + 180) % 360 - 180 < 0:
        g = g[:, ::-1]
    r = iof * xr.DataArray(g, dims=("band", "x"), coords={"band": iof.band, "x": iof.x})
    mean = (r.shift(band=1) + r.shift(band=-1)) / 2
    d = r - mean
    dm, dp = d.shift(band=1).fillna(d), d.shift(band=-1).fillna(d)
    spike = abs(d) > 0.9 * np.sqrt((dm**2 + d**2 + dp**2) / 3)
    return xr.where(spike, mean, r).transpose(*iof.dims)


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
