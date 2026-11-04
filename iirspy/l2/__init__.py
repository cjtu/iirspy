import json

import xarray as xr

from iirspy.l2 import photometry, polish, solar, thermal


def calibrate(rad, cfg, *, qub=None, geom=None):
    """L1 radiance cube (band, y, x) [W/m^2/sr/um] -> L2 reflectance, one step per `cfg` field group.

    `qub` is the scene's label, for the solar distance and monthly flux (None means 1 AU). `geom` is the
    scene's `Geometry`: with topo and `cfg.use_slope_az` the photometric step uses its facet terms and the
    incidence is the cube's per-line `solar_inc`; without topo the incidence is `geom.obs.sun_zenith`; with no `geom`
    at all the cube's per-line `solar_inc`.
    """
    if geom is not None and geom.shape != rad.shape[1:]:
        raise ValueError(f"geometry {geom.shape} does not match the cube {rad.shape[1:]}")
    topo = geom is not None and cfg.use_slope_az and geom.has_topo
    inc = rad.solar_inc if geom is None or topo else xr.DataArray(geom.obs.sun_zenith.values, dims=("y", "x"))
    inc = inc * xr.ones_like(rad.isel(x=0, band=0, drop=True))
    kw = {"qub": qub, "sdist": solar.sun_distance(cfg, qub), "inc": inc, "geom": geom}
    cube = rad.copy()
    cube.attrs = {"provenance": rad.attrs["provenance"]} if "provenance" in rad.attrs else {}
    cube = thermal.subtract(cube, cfg, **kw)
    cube = solar.divide_flux(cube, cfg, **kw)
    cube = photometry.photometric(cube, cfg, **kw)
    cube = polish.polish(cube, cfg, **kw)
    cube = photometry.phase(cube, cfg, **kw)
    cube = polish.smooth(cube, cfg, **kw)
    cube = polish.clip_fill(cube, cfg, **kw)

    cube.name = "Reflectance"
    cube.attrs |= {"name": "Reflectance", "units": "", "l2_config": json.dumps(cfg.to_attrs())}
    return cube
