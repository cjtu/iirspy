"""L1 -> L2 reflectance config: one frozen dataclass whose fields each name one method choice.

`L2Config` drives `L1.calibrate`. Presets are named sets of field values; overrides replace single fields.

>>> L2Config.preset("issdc").smoothing
3
>>> L2Config.preset("issdc", polish=None).polish is None
True
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from iirspy.config import Config
from iirspy.l2 import photometry, polish

POLISH = polish.POLISH
FLUX = ("pradan", "hsrs", "noaa_cdr_mean", "noaa_cdr_monthly")
SDIST = ("spice", "none")
THERMAL = ("none", "isothermal")
SMOOTHING_EDGES = ("pair", "nan", "reflect")


@dataclass(frozen=True)
class L2Config(Config):
    """L1 radiance -> L2 reflectance choices (see `iirspy.l2.calibrate`).

    Steps run in order: solar flux, thermal removal, polish, photometric normalization,
    Besse phase, spectral smoothing, clip.

    Attributes:
        flux: Solar spectrum. "pradan" (the PRADAN flux file), "noaa_cdr_mean" (NOAA SSI CDR v03,
            2010-2019 mean), "noaa_cdr_monthly" (the CDR month of the scene's start time), "hsrs"
            (HSRS v2 through the IIRS SRFs, packaged in PRADAN's format), a path to a file in PRADAN's
            format (tab-separated nm, mW/cm^2/um at 1 AU; scaled by 1/sdist^2), or a per-band
            sequence in W/m^2/sr/um for bands 1..N, used as is (NO distance scaling).
        sdist: Sun distance for scaling the flux: "spice", "none" (1 AU) or a float in AU.
        thermal: Thermal tail model, "none" or "isothermal" (one temperature per pixel).
        thermal_eps: Emissivity, used both to estimate T and to remove eps * B(T).
        thermal_bands: (first, last) IIRS band numbers, 1-indexed and inclusive, whose mean
            brightness temperature is T.
        thermal_t_min: T [K] at or below which (or NaN) a pixel's T falls back to the flat-surface
            equilibrium temperature at its incidence (`thermal.flat_surface_temperature`).
        clip_negative: Zero negative reflectance after smoothing.
        fill_nan: Value written over every NaN as the last step (ISSDC's ndi marks all nulls with 0), or None to keep NaN.
        polish: Per-band divisor: None, "verma2022", "issdc_empirical", or a csv path with `band`
            and `polish` columns. Mapped by band number; bands the table lacks get 1.0.
        photom: Disk function, a `photometry.MODELS` key or any f(cos_inc, cos_view, g) callable.
        use_slope_az: Use the caller's `slope_az` terrain product when one is given.
        phase: Rescale onto Besse's 30 deg phase reference (`photometry.besse_phase`).
        cos_min: Floor on cos_inc and cos_view before the disk function (0 = none). Clamps.
        min_lit: Null pixels with less than this lit fraction (needs `slope_az`).
        smoothing: Spectral moving-average width in bands (0 = off).
        smoothing_edges: End bands of the moving average (`polish.moving_average`).
    """

    flux: Any = "pradan"
    sdist: str | float = "spice"
    thermal: str = "none"
    thermal_eps: float = 1.0
    thermal_bands: tuple[int, int] = (227, 247)
    thermal_t_min: float = 0.0
    clip_negative: bool = False
    fill_nan: float | None = None
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


L2Config.PRESETS = {
    "iirspy": {"thermal": "isothermal", "photom": "lambert"},
    "l2d": {
        "flux": "hsrs",
        "sdist": "spice",
        "thermal": "isothermal",
        "photom": "lambert",
        "phase": False,
        "polish": None,
        "smoothing": 0,
        "clip_negative": False,
        "fill_nan": None,
        "use_slope_az": True,
        "min_lit": 0.0,
        "cos_min": 0.0,
    },
    "issdc": {
        "sdist": "none",
        "thermal": "isothermal",
        "thermal_eps": 1.0,
        "clip_negative": True,
        "fill_nan": 0.0,
        "polish": "issdc_empirical",
        "use_slope_az": False,
        "smoothing": 3,
    },
    "verma2022": {
        "sdist": "none",
        "thermal": "isothermal",
        "thermal_eps": 0.95,
        "thermal_bands": (225, 246),
        "clip_negative": True,
        "polish": "verma2022",
        "use_slope_az": False,
        "smoothing": 3,
    },
}
# Same L2 params as "issdc"; `iirs-compute-refl` reads the ISSDC nci L1 for this name instead of calibrating nri.
L2Config.PRESETS["issdc-nci"] = L2Config.PRESETS["issdc"]
