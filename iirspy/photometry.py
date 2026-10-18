"""
Photometric models and terrain normalization for the IIRS L1 -> L2 step.

A model is a disk function ``f(cos_inc, cos_view, g)``: how the surface's reflectance varies with
the cosines of incidence and emission and with phase ``g`` [deg]. L2 divides the measured
radiance factor by it,

    I/F = (rad - trad) / (F_solar * f(cos_inc, cos_view, g))

so the output is normalized to the model's own reference geometry (f = 1: overhead sun, nadir
view, zero phase). Without terrain, cos_inc is the scene's solar cosine and the view is nadir;
given a camera-space slope/aspect (`slope_az`), both become local cosines about the tilted facet.

Swap models by name (:data:`MODELS`) or pass any ``f(cos_inc, cos_view, g)`` callable as ``photom``.

Angle conventions match the IIRS spm:

    slope   degrees from the local horizontal (tangent plane), 0 = flat
    aspect  degrees clockwise from grid north, direction the facet faces (downhill)
    azimuth degrees clockwise from grid north; elevation degrees above the horizon
"""

import functools

import numpy as np
import xarray as xr

from iirspy.utils import PKG_DATA

M3_DATA = PKG_DATA.joinpath("m3")


def lambert(cos_inc, cos_view, g):
    """Lambert disk function: cos_inc."""
    return cos_inc


def lommel_seeliger(cos_inc, cos_view, g):
    """Lommel-Seeliger (single-scattering) disk function, normalized to 1 at cos_inc = cos_view = 1."""
    return 2 * cos_inc / (cos_inc + cos_view)


def lunar_lambert(cos_inc, cos_view, g, coeffs=(1.0, -0.019, 2.42e-4, -1.46e-6)):
    """Lunar-Lambert disk function with McEwen's phase-dependent L(g) (McEwen 1991).

    L(g) blends Lommel-Seeliger (L=1, limb-darkened) into Lambert (L=0). The default cubic is the
    one ISIS ships; pass `coeffs` for a band- or site-fitted L(g).
    """
    # Horner by hand: np.polyval drops xarray dims, and g may be per-line (y,) against (y, x) cosines.
    L = np.clip(sum(c * g**i for i, c in enumerate(coeffs)), 0.0, 1.0)
    return L * lommel_seeliger(cos_inc, cos_view, g) + (1 - L) * cos_inc


MODELS = {"lambert": lambert, "lommel_seeliger": lommel_seeliger, "lunar_lambert": lunar_lambert}


def get_model(photom):
    """Resolve `photom` (a MODELS key or any f(cos_inc, cos_view, g) callable) to the model function.

    >>> get_model("lambert") is lambert
    True
    >>> get_model(lambda cos_inc, cos_view, g: cos_inc**0.7)(1.0, 1.0, 0.0)
    1.0
    """
    if callable(photom):
        return photom
    try:
        return MODELS[photom]
    except KeyError:
        raise ValueError(f"unknown photometric model {photom!r}; have {sorted(MODELS)}") from None


@functools.lru_cache(maxsize=1)
def _m3_phase_table():
    """(alpha_deg, wl_nm, f) from the vendored Besse et al. (2013) M3 highland F_ALPHA table."""
    tab = np.loadtxt(M3_DATA.joinpath("M3G20111109_RFL_F_ALPHA_HIL.TAB").open(), skiprows=1)
    alpha, f = tab[:86, 0], tab[:86, 1:]
    wl = np.loadtxt(M3_DATA.joinpath("M3G20110224_RFL_SOLAR_SPEC.TAB").open())[:, 0]
    if f.shape[1] != wl.size:
        raise ValueError(f"M3 phase table has {f.shape[1]} channels but solar spectrum has {wl.size}")
    return alpha, wl, f


def besse_phase(g, wl_nm):
    """Besse et al. (2013) M3 highland phase-function ratio f(30)/f(g), interpolated to wl_nm [nm].

    >>> round(float(besse_phase(75.0, 950.06)), 4)
    1.5474
    >>> round(float(besse_phase(30.0, 950.06)), 4)
    1.0
    >>> bool(besse_phase(200.0, 950.06) == besse_phase(85.0, 950.06))  # clamped to the domain edge
    True
    """
    alpha, wl, f = _m3_phase_table()
    wl_arr = np.atleast_1d(np.asarray(wl_nm, dtype=float))
    f_wl = np.vstack([np.interp(np.clip(wl_arr, wl.min(), wl.max()), wl, row) for row in f])  # (86, n_wl)
    g_arr = np.clip(np.asarray(g, dtype=float), 0.0, 85.0)
    ratio = np.stack([f_wl[30, j] / np.interp(g_arr, alpha, f_wl[:, j]) for j in range(wl_arr.size)], axis=0)
    return ratio if np.ndim(wl_nm) else ratio[0]


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


def local_angles(slope, aspect, sun_az, sun_elev, view_az=0.0, view_elev=90.0):
    """(cos_inc, cos_view, g) for every pixel from local slope/aspect and the sun/view directions.

    slope/aspect are (y, x) camera-space fields; sun_az/sun_elev may be per-line (y) arrays, as
    the spm supplies them. The view defaults to nadir, which is IIRS to within its 0.8 deg FOV.
    """
    cos_inc = cos_angle(slope, aspect, sun_az, sun_elev)
    cos_view = cos_angle(slope, aspect, view_az, view_elev)
    return cos_inc, cos_view, phase_angle(sun_az, sun_elev, view_az, view_elev)


def _sun_tags(attrs):
    """(azimuth, elevation) in the product's own frame from its tags, or None if it carries none."""
    try:
        return float(attrs["az_grid"]), float(attrs["elev"])
    except (KeyError, TypeError, ValueError):
        return None


def load_slope_az(slope_az, like=None):
    """(slope, aspect, lit, sun) from a camera-space slope/aspect product; `lit` is 1.0 when it has none.

    `slope_az` is a raster with bands slope/aspect[/lit_frac], a tuple of
    those DataArrays, or a Dataset carrying them as variables. `lit` is the fraction of the solar
    disk the terrain leaves visible. `sun` is the (azimuth, elevation) the product's tags record,
    in the same tangent-plane frame as its slope and aspect, or None.

    When `like` is given the fields are matched to its y/x: a shape check and a coordinate
    transplant, never a resample.
    """
    sun = None
    if isinstance(slope_az, tuple):
        # (slope, aspect[, lit[, sun]]) -- what this function returns, so it round-trips
        slope, aspect, *rest = slope_az
        lit = rest[0] if rest else None
        sun = rest[1] if len(rest) > 1 else None
    elif isinstance(slope_az, xr.Dataset):
        slope, aspect = slope_az["slope"], slope_az["aspect"]
        lit = slope_az.get("lit_frac")
        sun = _sun_tags(slope_az.attrs)
    else:
        da = slope_az if isinstance(slope_az, xr.DataArray) else xr.open_dataarray(slope_az, engine="rasterio")
        names = [str(n) for n in np.atleast_1d(da.attrs.get("long_name", ["slope", "aspect", "lit_frac"]))]
        pick = lambda n: da.isel(band=names.index(n), drop=True) if n in names else None
        slope, aspect, lit = pick("slope"), pick("aspect"), pick("lit_frac")
        if slope is None or aspect is None:
            raise ValueError(f"slope_az product has bands {names}; need at least slope and aspect")
        sun_az, sun_elev = pick("sun_az"), pick("sun_elev")
        sun = (sun_az, sun_elev) if sun_az is not None and sun_elev is not None else _sun_tags(da.attrs)
    if like is not None:
        if slope.shape != like.shape:
            raise ValueError(
                f"slope_az {slope.shape} does not match the image {like.shape}; regenerate it for this crop"
            )
        coords = {d: like[d] for d in like.dims if d in like.coords}
        conv = lambda v: None if v is None else xr.DataArray(np.asarray(v), coords=coords, dims=like.dims)
        slope, aspect, lit = conv(slope), conv(aspect), conv(lit)
        if isinstance(sun, tuple) and hasattr(sun[0], "shape"):  # per-row arrays, not an (az, elev) tag pair
            sun = (conv(sun[0]), conv(sun[1]))
    return slope, aspect, 1.0 if lit is None else lit, sun


def slope_az_from_dem(dem):
    """(slope, aspect) [deg] from a DEM [m] already on the camera grid, as `load_slope_az` takes.

    `dem` is a (y, x) DataArray whose x/y coords are in metres, x east and y north (rows may run
    either way); gradients are Sobel, aspect is the downhill direction clockwise from grid north.
    """
    from scipy.ndimage import sobel

    z = np.asarray(dem, dtype=float)
    dx = float(dem.x[1] - dem.x[0])
    dy = float(dem.y[1] - dem.y[0])
    dz_dx = sobel(z, axis=1, mode="nearest") / (8 * dx)
    dz_dy = sobel(z, axis=0, mode="nearest") / (8 * dy)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    aspect = np.degrees(np.arctan2(-dz_dx, -dz_dy)) % 360
    return xr.DataArray(slope, coords=dem.coords, dims=dem.dims), xr.DataArray(aspect, coords=dem.coords, dims=dem.dims)


def photometric_factor(
    solar_inc, sun_az, slope_az=None, like=None, photom="lambert", cos_min=0.0, min_lit=0.9, inc_max=90.0
):
    """(factor, g, floor fractions): the disk function times the lit fraction, NaN where unlit.

    Without `slope_az` the surface is the flat local horizontal at incidence `solar_inc` [deg], nadir
    view, phase = incidence. With it, cosines are local to each facet, using the product's own sun when
    it records one, else (`sun_az`, 90 - `solar_inc`). `cos_min` floors cos_inc and cos_view
    before the model (clamped, not nulled); the fraction of pixels below it is returned as
    {"cos_inc_floor_frac", "cos_view_floor_frac"}. Pixels lit less than `min_lit` (from the
    product's lit fraction) are nulled, as are those with local incidence above `inc_max` [deg]
    (fraction returned as "inc_max_null_frac").
    """
    lit = 1.0
    if slope_az is not None:
        slope, aspect, lit, sun = load_slope_az(slope_az, like=like)
        # slope/aspect are referenced to the DEM's tangent plane, the spm's angles to each pixel's
        # local horizontal, so the product's own sun is used whenever it carries one.
        az, elev = sun if sun is not None else (sun_az, 90 - solar_inc)
        cos_inc, cos_view, g = local_angles(slope, aspect, az, elev)
    else:
        cos_inc = np.cos(np.radians(solar_inc))
        cos_view, g = xr.ones_like(cos_inc), phase_angle(sun_az, 90 - solar_inc)
    floors = {"cos_inc_floor_frac": 0.0, "cos_view_floor_frac": 0.0, "inc_max_null_frac": 0.0}
    keep = cos_inc >= np.cos(np.radians(inc_max))  # before any cos_min clamp
    if inc_max < 90.0:
        floors["inc_max_null_frac"] = float(1 - np.asarray(keep).mean())
    if cos_min > 0.0:
        floors["cos_inc_floor_frac"] = float(np.asarray(cos_inc < cos_min).mean())
        cos_inc = cos_inc.clip(min=cos_min) if hasattr(cos_inc, "clip") else np.clip(cos_inc, cos_min, None)
        floors["cos_view_floor_frac"] = float(np.asarray(cos_view < cos_min).mean())
        cos_view = cos_view.clip(min=cos_min) if hasattr(cos_view, "clip") else np.clip(cos_view, cos_min, None)
    # Facets the sun does not reach have no direct beam to normalize by, so they are nulled.
    factor = get_model(photom)(cos_inc, cos_view, g) * lit
    if min_lit > 0.0:
        factor = factor.where(np.asarray(lit) >= min_lit) if hasattr(factor, "where") else factor
    if inc_max < 90.0 and hasattr(factor, "where"):
        factor = factor.where(keep)
    factor = factor.where(factor > 0) if hasattr(factor, "where") else factor
    return factor, g, floors


def besse_rescale(refl, photom, g, wl):
    """Rescale `refl` from `photom`'s own f=1 normalization onto Besse's RADF(30, 0, 30) convention."""
    xl_ref = get_model(photom)(np.cos(np.radians(30.0)), 1.0, 30.0)
    g_dims = getattr(g, "dims", ("y",))
    f_ratio = xr.DataArray(besse_phase(np.asarray(g), wl.values), dims=("band", *g_dims))
    return refl * xl_ref * f_ratio
