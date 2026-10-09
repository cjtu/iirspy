"""Calibration configs: one frozen dataclass per level, each field naming one method choice.

`L1Config` drives L0 -> L1 radiance (`L0.calibrate`), `L2Config` drives L1 -> L2 reflectance
(`L1.calibrate`). Presets are named sets of field values; overrides replace single fields.

>>> L2Config.preset("issdc").smoothing
3
>>> L2Config.preset("issdc", polish=None).polish is None
True
>>> L1Config.preset("iirspy").empirical
True
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from iirspy import photometry, utils

POLISH = {"verma2022": utils.FPOLISH, "issdc_empirical": utils.FPOLISH_ISSDC_EMPIRICAL}
FLUX = ("pradan", "noaa_cdr_mean", "noaa_cdr_monthly")
SDIST = ("spice", "none")
THERMAL = ("none", "isothermal")
SMOOTHING_EDGES = ("pair", "nan", "reflect")


class _Config:
    """Preset lookup and JSON round trip shared by the level configs."""

    PRESETS: ClassVar[dict[str, dict]] = {}

    @classmethod
    def preset(cls, name: str, **overrides):
        """The named preset with `overrides` applied. Raises on an unknown preset or field."""
        if name not in cls.PRESETS:
            raise ValueError(f"unknown {cls.__name__} preset {name!r}; choose from {tuple(cls.PRESETS)}")
        return cls(**{**cls.PRESETS[name], **overrides})

    @classmethod
    def resolve(cls, config="iirspy", **overrides):
        """A preset name or an instance, with `overrides` applied."""
        if isinstance(config, cls):
            return dataclasses.replace(config, **overrides) if overrides else config  # type: ignore[type-var]
        return cls.preset(config, **overrides)

    def to_attrs(self) -> dict:
        """Every field as JSON-safe values; `type(cfg)(**cfg.to_attrs())` rebuilds the config."""
        out = {}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            v = getattr(self, f.name)
            if callable(v):
                v = getattr(v, "__name__", "custom")
            out[f.name] = [list(e) if isinstance(e, tuple) else e for e in v] if isinstance(v, tuple) else v
        return out


@dataclass(frozen=True)
class L1Config(_Config):
    """L0 digital number -> L1 radiance choices (see `L0.calibrate_to_rad`).

    Attributes:
        empirical: Per-scene empirical dark/flat (+ smile) with the LUT's band-median scale,
            instead of the per-element LUT gain/offset.
        interp_bands: Fill masked bands by interpolating across band (e.g. "linear"), or None.
        interp_spatial: Fill single-column all-band NaN gaps across x.
        bad_pixel_mask: Null the packaged bad detector elements.
        exclude_wl: Extra (lo, hi) wavelength ranges [nm] to null, or None.
        attach_snr: Attach the empirical broadband SNR as an `snr` coord (empirical scenes with shadow).
        attach_empirical_frames: Attach the empirical dark/flat/smile (band, x) frames as coords.
    """

    empirical: bool = False
    interp_bands: str | None = None
    interp_spatial: bool = False
    bad_pixel_mask: bool = True
    exclude_wl: tuple[tuple[float, float], ...] | None = None
    attach_snr: bool = True
    attach_empirical_frames: bool = False

    def __post_init__(self):
        if self.exclude_wl is not None:
            object.__setattr__(self, "exclude_wl", tuple((float(lo), float(hi)) for lo, hi in self.exclude_wl))


@dataclass(frozen=True)
class L2Config(_Config):
    """L1 radiance -> L2 reflectance choices (see `L2._compute_reflectance`).

    Steps run in order: solar flux, thermal removal, clip, polish, photometric normalization,
    Besse phase, spectral smoothing.

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
        polish: Per-band divisor: None, "verma2022", "issdc_empirical", or a csv path with `band`
            and `polish` columns. Mapped by band number; bands the table lacks get 1.0.
        photom: Disk function, a `photometry.MODELS` key or any f(cos_inc, cos_view, g) callable.
        use_slope_az: Use the caller's `slope_az` terrain product when one is given.
        phase: Rescale onto Besse's 30 deg phase reference (`photometry.besse_phase`).
        cos_min: Floor on cos_inc and cos_view before the disk function (0 = none).
        min_lit: Null pixels with less than this lit fraction (needs `slope_az`).
        smoothing: Spectral moving-average width in bands (0 = off).
        smoothing_edges: End bands of the moving average (`utils.moving_average`).
    """

    flux: Any = "pradan"
    sdist: str | float = "spice"
    thermal: str = "none"
    thermal_eps: float = 1.0
    thermal_bands: tuple[int, int] = (227, 247)
    clip_negative: bool = False
    polish: str | None = None
    photom: str | Callable = "lambert"
    use_slope_az: bool = True
    phase: bool = False
    cos_min: float = 0.0
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


L1Config.PRESETS = {
    "iirspy": {"empirical": True, "interp_bands": "linear", "interp_spatial": True},
    "issdc": {},
}

L2Config.PRESETS = {
    "iirspy": {"thermal": "isothermal", "photom": "lambert"},
    "issdc": {
        "sdist": "none",
        "thermal": "isothermal",
        "thermal_eps": 1.0,
        "clip_negative": True,
        "polish": "issdc_empirical",
        "use_slope_az": False,
        "smoothing": 3,
    },
    "verma2022": {
        "sdist": "none",
        "thermal": "isothermal",
        "thermal_eps": 0.95,
        "clip_negative": True,
        "polish": "verma2022",
        "use_slope_az": False,
        "smoothing": 3,
    },
}
