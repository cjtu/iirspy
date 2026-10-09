"""L1 radiance -> L2 reflectance as a set of swappable steps.

`L2Config` names one implementation per step; `PRESETS` are named sets of field values. Steps run
in this order (see `L2._compute_reflectance`):

    F     = solar_flux(cfg, sdist(cfg, qub), qub)       W/m^2/sr/um at the scene's distance
    trad  = thermal_radiance(cfg, rad)                   0, or eps * B(T) with T per pixel
    refl  = clip(rad - trad) / (photometric * F * polish)
    refl  = smooth(cfg, refl)

>>> L2Config.preset("issdc").smoothing
3
>>> L2Config.preset("issdc", polish=None).polish is None
True
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from iirspy import photometry, utils

FPOLISH_ISSDC_NDI = str(utils.PKG_DATA.joinpath("spectral_polish_issdc_ndi.csv"))
FNOAA = str(utils.PKG_DATA.joinpath("solar_flux_noaa_cdr_v03.csv"))

POLISH = {"verma2022": utils.FPOLISH, "issdc_ndi": FPOLISH_ISSDC_NDI}
FLUX = ("pradan", "noaa_cdr_mean", "noaa_cdr_monthly")
SDIST = ("spice", "none")
THERMAL = ("none", "isothermal")
SMOOTHING_EDGES = ("pair", "nan", "reflect")


@dataclass(frozen=True)
class L2Config:
    """One choice per L2 step. Field values are registry keys, so each step swaps independently.

    Attributes:
        flux: Solar spectrum. "pradan" (the PRADAN flux file), "noaa_cdr_mean" (NOAA SSI CDR v03,
            2010-2019 mean), "noaa_cdr_monthly" (the CDR month of the scene's start time), a path
            to a file in PRADAN's format (tab-separated nm, mW/cm^2/um at 1 AU), or a per-band
            sequence in W/m^2/sr/um for bands 1..N, used as is (no distance scaling).
        sdist: Sun distance for scaling the flux: "spice", "none" (1 AU) or a float in AU.
        thermal: Thermal tail model, "none" or "isothermal" (one temperature per pixel).
        thermal_eps: Emissivity, used both to estimate T and to remove eps * B(T).
        thermal_bands: (first, last) IIRS band numbers, 1-indexed and inclusive, whose mean
            brightness temperature is T.
        clip_negative: Zero negative thermal-removed radiance before polish.
        polish: Per-band divisor: None, "verma2022", "issdc_ndi", or a csv path with `band` and
            `polish` columns. Mapped by band number; bands the table lacks get 1.0.
        photom: Disk function, a `photometry.MODELS` key or any f(mu0, mu, g) callable.
        topo: Use the caller's topo product when one is given; False ignores it.
        phase: Rescale onto Besse's 30 deg phase reference (`photometry.besse_phase`).
        mu_min: Floor on mu0 and mu before the disk function (0 = none).
        min_lit: Null pixels with less than this lit fraction (needs topo).
        smoothing: Spectral moving-average width in bands (0 = off).
        smoothing_edges: End bands of the moving average: "pair" (the CH2IIRS plugin's rule,
            width 3 only: first = mean of bands 2-3, last = mean of the last two), "nan", or
            "reflect".
    """

    flux: Any = "pradan"
    sdist: str | float = "spice"
    thermal: str = "none"
    thermal_eps: float = 1.0
    thermal_bands: tuple[int, int] = (227, 247)
    clip_negative: bool = False
    polish: str | None = None
    photom: str | Callable = "lambert"
    topo: bool = True
    phase: bool = False
    mu_min: float = 0.0
    min_lit: float = 0.9
    smoothing: int = 0
    smoothing_edges: str = "pair"

    def __post_init__(self):
        if isinstance(self.flux, Path):
            object.__setattr__(self, "flux", str(self.flux))
        elif not isinstance(self.flux, str):
            object.__setattr__(self, "flux", tuple(float(f) for f in np.atleast_1d(self.flux)))
        elif self.flux not in FLUX and not Path(self.flux).is_file():
            raise ValueError(f"flux {self.flux!r}: not one of {FLUX} or an existing file")
        if isinstance(self.polish, Path):
            object.__setattr__(self, "polish", str(self.polish))
        if self.polish is not None and self.polish not in POLISH and not Path(self.polish).is_file():
            raise ValueError(f"polish {self.polish!r}: not one of {tuple(POLISH)} or an existing file")
        if isinstance(self.sdist, str) and self.sdist not in SDIST:
            raise ValueError(f"sdist {self.sdist!r}: not one of {SDIST} or a float")
        if self.thermal not in THERMAL:
            raise ValueError(f"thermal {self.thermal!r}: not one of {THERMAL}")
        if self.smoothing_edges not in SMOOTHING_EDGES:
            raise ValueError(f"smoothing_edges {self.smoothing_edges!r}: not one of {SMOOTHING_EDGES}")
        if self.smoothing > 1 and self.smoothing_edges == "pair" and self.smoothing != 3:
            raise ValueError('smoothing_edges="pair" is only defined for smoothing=3')
        object.__setattr__(self, "thermal_bands", tuple(int(b) for b in self.thermal_bands))
        photometry.get_model(self.photom)

    @classmethod
    def preset(cls, name: str, **overrides) -> L2Config:
        """The named preset with `overrides` applied. Raises on an unknown preset or field."""
        if name not in PRESETS:
            raise ValueError(f"unknown L2 preset {name!r}; choose from {tuple(PRESETS)}")
        return cls(**{**PRESETS[name], **overrides})

    def to_attrs(self) -> dict:
        """Every field as JSON-safe values; `L2Config(**cfg.to_attrs())` rebuilds the config."""
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if callable(v):
                v = getattr(v, "__name__", "custom")
            out[f.name] = list(v) if isinstance(v, tuple) else v
        return out


PRESETS: dict[str, dict] = {
    "iirspy": {"thermal": "isothermal", "photom": "lambert"},
    "issdc": {
        "sdist": "none",
        "thermal": "isothermal",
        "thermal_eps": 1.0,
        "clip_negative": True,
        "polish": "issdc_ndi",
        "topo": False,
        "smoothing": 3,
    },
    "verma_plugin": {
        "sdist": "none",
        "thermal": "isothermal",
        "thermal_eps": 0.95,
        "clip_negative": True,
        "polish": "verma2022",
        "topo": False,
        "smoothing": 3,
    },
}


def as_config(config: str | L2Config = "iirspy", **overrides) -> L2Config:
    """A preset name or an `L2Config`, with `overrides` applied."""
    if isinstance(config, L2Config):
        return dataclasses.replace(config, **overrides) if overrides else config
    return L2Config.preset(config, **overrides)


def sdist(cfg: L2Config, qub=None) -> float:
    """Sun distance [AU] to scale the flux by. "spice" without a label falls back to 1 AU."""
    if cfg.sdist == "none":
        return 1.0
    if cfg.sdist == "spice":
        return utils.get_solar_distance(qub) if qub else 1.0
    return float(cfg.sdist)


def _scene_month(qub) -> str:
    import pdr

    return str(pd.Timestamp(pdr.open(qub).metaget("start_date_time")).strftime("%Y-%m"))


def solar_flux(cfg: L2Config, d: float = 1.0, qub=None):
    """Per-band solar flux [W/m^2/sr/um] at `d` AU, as a DataArray on band 1..256 (or the
    given sequence as is)."""
    if isinstance(cfg.flux, tuple):
        if len(cfg.flux) == 1:
            return cfg.flux[0]
        return xr.DataArray(np.asarray(cfg.flux), coords={"band": np.arange(1, len(cfg.flux) + 1)})
    if cfg.flux in ("noaa_cdr_mean", "noaa_cdr_monthly"):
        df = pd.read_csv(FNOAA)
        col = "mean_2010_2019" if cfg.flux == "noaa_cdr_mean" else _scene_month(qub)
        if col not in df:
            raise ValueError(f"{FNOAA} has no {col} column")
        flux = df[col].to_numpy() * 10 / (np.pi * d**2)
        return xr.DataArray(flux, coords={"band": df["band"].to_numpy()}, name="Solar flux [W/m^2/sr/um]")
    return utils.get_solar_flux(d, utils.FSOLAR if cfg.flux == "pradan" else cfg.flux)


def thermal_radiance(cfg: L2Config, rad):
    """(trad, T): emitted radiance to subtract and the per-pixel temperature [K], or (0.0, None).

    T is the mean brightness temperature over `cfg.thermal_bands` (by band number) at emissivity
    `cfg.thermal_eps`; trad = eps * B(T) at every band.
    """
    if cfg.thermal == "none":
        return 0.0, None
    c, h, k = 3e8, 6.626e-34, 1.38e-23
    eps, wl = cfg.thermal_eps, rad.wl * 1e-9
    q = np.log(eps * 2 * h * c**2 * 1e-6 / (rad * wl**5) + 1)
    bt = h * c / (wl * k * q)
    t = bt.sel(band=slice(*cfg.thermal_bands)).mean("band")
    bbr = 1e-6 * (2 * h * c * c / wl**5) * 1 / (np.exp(h * c / (wl * k * t)) - 1)
    return eps * bbr, t


def polish(cfg: L2Config, band):
    """Per-band polish divisor on `band`, 1.0 where the table has none, or None when off."""
    if cfg.polish is None:
        return None
    df = pd.read_csv(POLISH.get(cfg.polish, cfg.polish))
    coeff = pd.Series(df["polish"].to_numpy(), index=df["band"].to_numpy())
    band = np.asarray(band)
    return xr.DataArray(coeff.reindex(band).fillna(1.0).to_numpy(), coords={"band": band})


def smooth(cfg: L2Config, refl):
    """Centred `cfg.smoothing`-band moving average, end bands per `cfg.smoothing_edges`."""
    w = cfg.smoothing
    if w <= 1:
        return refl
    if cfg.smoothing_edges == "reflect":
        n = w // 2
        padded = refl.pad(band=n, mode="reflect").rolling(band=w, center=True).mean()
        return padded.isel(band=slice(n, -n)).assign_coords(refl.band.coords)
    out = refl.rolling(band=w, center=True).mean()
    if cfg.smoothing_edges == "pair":
        b = refl.band
        first = refl.isel(band=[1, 2]).mean("band")
        last = refl.isel(band=[-2, -1]).mean("band")
        out = xr.where(b == b[0], first, xr.where(b == b[-1], last, out)).transpose(*refl.dims)
    return out


def stamp(refl, cfg: L2Config, **diagnostics):
    """Record `cfg` (json) and per-scene diagnostics on `refl.attrs`."""
    refl.attrs["l2_config"] = json.dumps(cfg.to_attrs())
    refl.attrs.update(diagnostics)
    return refl
