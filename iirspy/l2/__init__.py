"""L1 -> L2 pipeline: radiance [W/m^2/sr/um] to reflectance. `L1.calibrate` calls it.

Each step is switched by `L2Config` fields and appends {"step", "method"} to the JSON list `attrs["provenance"]`,
continuing L1's. The step modules take plain arguments; only this module reads the config. In order:

1. ISSDC polish (`polish="issdc"` only): `polish.issdc_polish`
2. Thermal removal: `thermal.subtract`
3. Solar flux division, radiance to I/F: `solar.divide_flux`
4. Photometric normalization to the standard geometry: `photometry.photometric`
5. Spectral polish (any `polish` but "issdc"): `polish.spectral_polish`
6. Smoothing: `polish.moving_average`
7. Clip negatives and fill NaN: here in `calibrate`
"""

import json

import xarray as xr

from iirspy import io
from iirspy.l2 import photometry, polish, solar, thermal


@io.xarray_keep_attrs
def calibrate(rad, cfg, *, qub=None, geom=None):
    """L1 radiance cube (band, y, x) [W/m^2/sr/um] -> L2 reflectance, by `cfg` (an `L2Config`).

    `qub` is the scene's label, for the solar distance and monthly flux (None means 1 AU). `geom` is the
    scene's `Geometry`: with topo and `cfg.use_slope_az` the photometric step uses its facet terms and the
    incidence is the cube's per-line `solar_inc`; without topo the incidence is `geom.obs.sun_zenith`; with no `geom`
    at all the cube's per-line `solar_inc`. `polish="issdc"` needs `geom` for the scene's longitude.
    """
    if geom is not None and geom.shape != rad.shape[1:]:
        raise ValueError(f"geometry {geom.shape} does not match the cube {rad.shape[1:]}")
    topo = geom is not None and cfg.use_slope_az and geom.has_topo
    inc = rad.solar_inc if geom is None or topo else xr.DataArray(geom.obs.sun_zenith.values, dims=("y", "x"))
    inc = inc * xr.ones_like(rad.isel(x=0, band=0, drop=True))
    sdist = solar.sun_distance(cfg.sdist, qub)
    steps = json.loads(rad.attrs.get("provenance", "[]"))

    def log(step, method, **inputs):
        steps.append({"step": step, "method": str(method), **inputs})

    cube = rad.copy()
    cube.attrs = {}
    if cfg.polish == "issdc":
        f = solar.solar_flux(cfg.flux, sdist, qub)
        cube = polish.issdc_polish(cube / f, geom and geom.loc.lon.values) * f
        log("polish", "issdc")
    if cfg.thermal != "none":
        cube = thermal.subtract(
            cube,
            cfg.thermal_eps,
            cfg.thermal_bands,
            inc=inc,
            sdist=sdist,
            t_min=cfg.thermal_t_min,
            model=cfg.thermal,
        )
        if "T" in cube.coords:
            log("thermal", cfg.thermal)
    cube = solar.divide_flux(cube, cfg.flux, sdist, qub)
    log("solar", cfg.flux)
    cube = photometry.photometric(
        cube, inc, geom if topo else None, photom=cfg.photom, cos_min=cfg.cos_min, min_lit=cfg.min_lit
    )
    log("photometric", getattr(cfg.photom, "__name__", cfg.photom), geom=geom and geom.method)
    if cfg.polish not in (None, "issdc"):
        cube = cube / polish.spectral_polish(cube.band, cfg.polish)
        log("polish", cfg.polish)
    if cfg.smoothing > 1:
        cube = polish.moving_average(cube, cfg.smoothing, cfg.smoothing_edges)
        log("smoothing", "moving_average")
    if cfg.clip_negative:
        cube = cube.clip(min=0)
        log("clip", "negative")
    if cfg.fill_nan is not None:
        cube = cube.fillna(cfg.fill_nan)
        log("fill_nan", cfg.fill_nan)

    cube.name = "Reflectance"
    cube.attrs |= {
        "name": "Reflectance",
        "units": "",
        "l2_config": json.dumps(cfg.to_attrs()),
        "provenance": json.dumps(steps, default=str),
    }
    return cube
