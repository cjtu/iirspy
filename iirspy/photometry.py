"""
Photometric models and topographic normalization for the IIRS L1 -> L2 step.

A model is a disk function ``f(mu0, mu, g)``: how the surface's reflectance varies with
illumination (``mu0 = cos i``), emission (``mu = cos e``) and phase (``g``, degrees). L2 divides
the measured radiance factor by it,

    I/F = (rad - trad) / (F_solar * f(mu0, mu, g))

so the output is normalized to the model's own reference geometry (f = 1: overhead sun, nadir
view, zero phase). Without topography mu0 is the scene's per-line solar cosine; hand it
slope/aspect and mu0 and mu become the *local* cosines about the tilted facet.

Swap models by name (:data:`MODELS`) or by passing any ``f(mu0, mu, g)`` callable as ``photom``:

    >>> l2 = l1.calibrate(topo="ch2_iir_<sid>_topo.tif", photom="lunar_lambert")  # doctest: +SKIP

Slope, aspect and the cast-shadow lit fraction come from the DEM sampled into camera space
(``iirspy.georef.camera_topo``). Angle conventions match that renderer and the IIRS spm:

    slope   degrees from the local horizontal (tangent plane), 0 = flat
    aspect  degrees clockwise from grid north, direction the facet faces (downhill)
    azimuth degrees clockwise from grid north; elevation degrees above the horizon
"""

import numpy as np
import xarray as xr


def lambert(mu0, mu, g):
    """Lambert disk function: mu0."""
    return mu0


def lommel_seeliger(mu0, mu, g):
    """Lommel-Seeliger (single-scattering) disk function, normalized to 1 at mu0 = mu = 1."""
    return 2 * mu0 / (mu0 + mu)


def lunar_lambert(mu0, mu, g, coeffs=(1.0, -0.019, 2.42e-4, -1.46e-6)):
    """Lunar-Lambert disk function with McEwen's phase-dependent L(g) (McEwen 1991).

    L(g) blends Lommel-Seeliger (L=1, limb-darkened) into Lambert (L=0). The default cubic is the
    one ISIS ships; pass `coeffs` for a band- or site-fitted L(g).
    """
    # Horner by hand: np.polyval drops xarray dims, and g is per-line (y,) while mu0/mu are (y, x).
    L = np.clip(sum(c * g**i for i, c in enumerate(coeffs)), 0.0, 1.0)
    return L * lommel_seeliger(mu0, mu, g) + (1 - L) * mu0


MODELS = {"lambert": lambert, "lommel_seeliger": lommel_seeliger, "lunar_lambert": lunar_lambert}


def get_model(photom):
    """Resolve `photom` (a MODELS key or any f(mu0, mu, g) callable) to the model function.

    >>> get_model("lambert") is lambert
    True
    >>> get_model(lambda mu0, mu, g: mu0**0.7)(1.0, 1.0, 0.0)
    1.0
    """
    if callable(photom):
        return photom
    try:
        return MODELS[photom]
    except KeyError:
        raise ValueError(f"unknown photometric model {photom!r}; have {sorted(MODELS)}") from None


def cos_angle(slope, aspect, az, elev):
    """Cosine of the angle between the facet normal and a direction (az, elev), all degrees.

    Flat ground gives sin(elev) - i.e. cos of the angle from vertical - and a facet tilted into
    the direction gives more:

    >>> float(np.round(cos_angle(0, 0, 30, 45), 4))
    0.7071
    >>> float(np.round(cos_angle(20, 30, 30, 45), 4))
    0.9063
    """
    s, a = np.radians(slope), np.radians(aspect)
    e, z = np.radians(elev), np.radians(az)
    return np.cos(s) * np.sin(e) + np.sin(s) * np.cos(e) * np.cos(z - a)


def phase_angle(sun_az, sun_elev, view_az=0.0, view_elev=90.0):
    """Sun-target-observer phase angle [deg]. Nadir view reduces it to the solar incidence.

    >>> float(np.round(phase_angle(120.0, 60.0), 4))
    30.0
    """
    cg = np.sin(np.radians(sun_elev)) * np.sin(np.radians(view_elev)) + np.cos(np.radians(sun_elev)) * np.cos(
        np.radians(view_elev)
    ) * np.cos(np.radians(sun_az - view_az))
    return np.degrees(np.arccos(np.clip(cg, -1.0, 1.0)))


def topo_angles(slope, aspect, sun_az, sun_elev, view_az=0.0, view_elev=90.0):
    """(mu0, mu, g) for every pixel from local topography and the sun/view directions.

    slope/aspect are (y, x) camera-space fields; sun_az/sun_elev may be per-line (y) arrays, as
    the spm supplies them. The view defaults to nadir, which is IIRS to within its 0.8 deg FOV;
    pass view_az/view_elev when an off-nadir pointing matters.
    """
    mu0 = cos_angle(slope, aspect, sun_az, sun_elev)
    mu = cos_angle(slope, aspect, view_az, view_elev)
    return mu0, mu, phase_angle(sun_az, sun_elev, view_az, view_elev)


def _sun_tags(attrs):
    """(azimuth, elevation) in the product's own frame from its tags, or None if it carries none."""
    try:
        return float(attrs["az_grid"]), float(attrs["elev"])
    except (KeyError, TypeError, ValueError):
        return None


def load_topo(topo, like=None):
    """(slope, aspect, lit, sun) from a camera-space topo product; `lit` is 1.0 when it has none.

    `topo` is a raster written by iirspy.georef.save_topo (bands slope/aspect[/lit]), a tuple of
    those DataArrays, or a Dataset carrying them as variables. `lit` is the fraction of the solar
    disk the terrain leaves visible. `sun` is the (azimuth, elevation) the product's tags record,
    in the same tangent-plane frame as its slope and aspect, or None.

    When `like` is given the fields are matched to its y/x: a shape check and a coordinate
    transplant, never a resample.
    """
    sun = None
    if isinstance(topo, tuple):
        # (slope, aspect[, lit[, sun]]) -- what this function returns, so it round-trips
        slope, aspect, *rest = topo
        lit = rest[0] if rest else None
        sun = rest[1] if len(rest) > 1 else None
    elif isinstance(topo, xr.Dataset):
        slope, aspect = topo["slope"], topo["aspect"]
        lit = topo.get("lit")
        sun = _sun_tags(topo.attrs)
    else:
        da = topo if isinstance(topo, xr.DataArray) else xr.open_dataarray(topo, engine="rasterio")
        names = [str(n) for n in np.atleast_1d(da.attrs.get("long_name", ["slope", "aspect", "lit"]))]
        pick = lambda n: da.isel(band=names.index(n), drop=True) if n in names else None
        slope, aspect, lit = pick("slope"), pick("aspect"), pick("lit")
        if slope is None or aspect is None:
            raise ValueError(f"topo product has bands {names}; need at least slope and aspect")
        sun_az, sun_elev = pick("sun_az"), pick("sun_elev")
        sun = (sun_az, sun_elev) if sun_az is not None and sun_elev is not None else _sun_tags(da.attrs)
    if like is not None:
        if slope.shape != like.shape:
            raise ValueError(f"topo {slope.shape} does not match the image {like.shape}; regenerate it for this crop")
        coords = {d: like[d] for d in like.dims if d in like.coords}
        conv = lambda v: None if v is None else xr.DataArray(np.asarray(v), coords=coords, dims=like.dims)
        slope, aspect, lit = conv(slope), conv(aspect), conv(lit)
        if isinstance(sun, tuple) and hasattr(sun[0], "shape"):  # per-row arrays, not an (az, elev) tag pair
            sun = (conv(sun[0]), conv(sun[1]))
    return slope, aspect, 1.0 if lit is None else lit, sun
