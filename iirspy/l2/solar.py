"""Solar flux division, the second L2 step: thermal-free radiance -> radiance factor (I/F).

Each band is divided by the solar spectral radiance at the scene's Sun distance d [AU],

    I/F = rad / (F_sun(band) / (pi d^2))

`flux` picks F_sun, all convolved to IIRS bands at 1 AU: the ISSDC file (default), HSRS v2, or NOAA SSI CDR v03
(2010-2019 mean or the scene's month). `sdist` is "spice" (SPICE at the scene's mid-line epoch), "none" (1 AU)
or a value in AU. The flux files are in mW/cm^2/um; x10 converts to W/m^2/um.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pdr
import xarray as xr

from iirspy import io
from iirspy.georef import ancillary
from iirspy.instrument import PKG_DATA

# NOAA SSI CDR v03 monthly irradiance convolved to IIRS bands [mW/cm^2/um at 1 AU]: Coddington, O., Lean,
# J. L., Lindholm, C., Pilewskie, P. (2024). NOAA Climate Data Record (CDR) of NASA NOAA LASP Spectral
# Solar Irradiance (NNLSSI), Version 3. Monthly SSI, 2010-01 to 2025-12. NOAA National Centers for
# Environmental Information. https://doi.org/10.25921/esjz-1w61. Accessed 2026-10-09.
FNOAA = str(PKG_DATA.joinpath("solar_flux_noaa_cdr_v03.csv"))
FSOLAR = str(PKG_DATA.joinpath("iir/miscellaneous/ch2_iirs_solar_flux.txt"))
FSOLAR_HSRS = str(PKG_DATA.joinpath("ch2_iirs_solar_flux_hsrs.txt"))
AU_KM = 1.495978707e8


def divide_flux(rad, flux="issdc", sdist=1.0, qub=None):
    """`rad` over the solar flux at `sdist` AU; `flux` is an `L2Config.flux` value, `qub` the label (monthly CDR)."""
    out = rad / solar_flux(flux, sdist, qub)
    out.attrs["solar_distance_au"] = sdist
    return out


def solar_flux(flux="issdc", sdist=1.0, qub=None):
    """Solar flux F_sun / (pi d^2) [W/m^2/sr/um] by band for an `L2Config.flux` value (see `divide_flux`)."""
    if isinstance(flux, tuple):
        n = len(flux)
        f = flux[0] if n == 1 else xr.DataArray(np.asarray(flux), coords={"band": np.arange(1, n + 1)})
    elif flux == "noaa_cdr_mean":
        f = get_noaa_flux(sdist)
    elif flux == "noaa_cdr_monthly":
        f = get_noaa_flux(sdist, pd.Timestamp(pdr.open(qub).metaget("start_date_time")).strftime("%Y-%m"))
    else:
        f = get_solar_flux(sdist, {"issdc": FSOLAR, "hsrs": FSOLAR_HSRS}.get(flux, flux))
    return f


def sun_distance(sdist, qub=None):
    """Sun distance [AU] for an `L2Config.sdist` value; "spice" without a label `qub` is 1 AU."""
    if sdist == "spice":
        return get_solar_distance(qub) if qub else 1.0
    return 1.0 if sdist == "none" else float(sdist)


def get_noaa_flux(sdist=1.0, month=None, fflux=FNOAA):
    """NOAA SSI CDR v03 solar flux [W/m^2/sr/um] at `sdist` AU on band 1..256 (see `FNOAA`).

    `month` is "YYYY-MM" for that month's irradiance; None uses the 2010-2019 mean.
    """
    df = pd.read_csv(fflux)
    col = "mean_2010_2019" if month is None else month
    if col not in df:
        raise ValueError(f"{fflux} has no {col} column")
    flux = df[col].to_numpy() * 10 / (np.pi * sdist**2)
    return xr.DataArray(flux, coords={"band": df["band"].to_numpy()}, name="Solar flux [W/m^2/sr/um]")


def get_solar_flux(sdist=1.0, fflux=FSOLAR):
    """IIRS solar flux [W/m^2/sr/um] at `sdist` AU along band, from a file in ISSDC's format."""
    L = np.loadtxt(fflux, delimiter="\t", usecols=1)
    flux = L * 10 / (np.pi * sdist**2)
    return xr.DataArray(flux, coords={"band": np.arange(1, 257)}, name="Solar flux [W/m^2/sr/um]")


def get_solar_distance(fimg, kernels=None):
    """Sun-Moon distance [AU] at the scene's mid-line epoch, from SPICE.

    `kernels=None` resolves the kernel set via `io.spice_kernels(day)`. Warns and returns 1.0 AU if
    the label or kernels can't be found.
    """
    import spiceypy as sp

    fimg = Path(fimg)
    try:
        line_times = ancillary.get_line_times(fimg)
        t = pd.Timestamp(line_times[len(line_times) // 2], unit="s")
        ks = kernels if kernels is not None else io.spice_kernels(t.strftime("%Y%m%d"))
        for k in ks:
            sp.furnsh(str(k))
        et = sp.str2et(t.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        v, _ = sp.spkpos("SUN", et, "IAU_MOON", "LT+S", "MOON")
    except Exception as e:
        warnings.warn(f"get_solar_distance: could not compute from SPICE for {fimg} ({e}); using 1.0 AU", stacklevel=2)
        return 1.0
    return float(np.linalg.norm(v) / AU_KM)
