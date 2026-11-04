"""
Photometric normalization, the L2 step after the solar flux: I/F -> reflectance at the standard geometry.

I/F depends on the illumination and view as well as on the surface. A photometric model r(i, e, g) describes that
dependence, and this step moves every pixel onto i = 30, e = 0, g = 30 (`STANDARD_GEOMETRY`, the RELAB/M3 convention):

    reflectance = I/F * r(30, 0, 30) / r(i, e, g),    I/F = (rad - thermal) / (F_sun / π)

A model takes any of `cos_inc`, `cos_view`, `g` [deg] and `wl` [nm], passed by parameter name:

    none()                                        1: reflectance = I/F
    lambert(cos_inc)                              cos_inc
    lommel_seeliger(cos_inc, cos_view)            2 cos_inc / (cos_inc + cos_view)
    lunar_lambert(cos_inc, cos_view, g)           McEwen's L(g) blend of the two
    besse_m3_highland(cos_inc, cos_view, g, wl)   lunar_lambert x Besse et al. (2013) M3 highland f(g, wl)

A model that also takes `iof` returns the reflectance itself. That covers models nonlinear in albedo, such as Hapke
(invert w per pixel and band, then evaluate at the standard geometry), and `issdc`, which reproduces ISSDC's
I/F / cos_inc.

Without terrain the surface is the flat local horizontal: cos_inc = cos(solar_inc), nadir view, g = incidence.
With a `Geometry` that has topo, both cosines are taken about each DEM facet and r is scaled by the lit fraction.

Angle conventions match the IIRS spm:

    slope   degrees from the local horizontal (tangent plane), 0 = flat
    aspect  degrees clockwise from grid north, direction the facet faces (downhill)
    azimuth degrees clockwise from grid north; elevation degrees above the horizon

`photom` is a :data:`MODELS` name or any such callable, recorded by function name.
"""

import inspect

import numpy as np
import xarray as xr

from iirspy.instrument import PKG_DATA

M3_DATA = PKG_DATA.joinpath("m3")
STANDARD_GEOMETRY = (30.0, 0.0, 30.0)  # (i, e, g) [deg]


def photometric(iof, inc, geom=None, *, photom="lambert", cos_min=0.0, min_lit=0.9):
    """`iof` at `STANDARD_GEOMETRY` by the model `photom` (see the module docstring).

    `inc` is the solar incidence [deg] on `iof`'s (y, x). `geom` is a `Geometry` with topo whose facet slope, aspect,
    lit fraction and sun replace the flat local horizontal, or None. `cos_min` floors cos_inc and cos_view (clamped,
    not nulled; the floored fractions go to attrs). Pixels lit less than `min_lit` or where r is not positive are NaN.
    """
    like = iof.isel(band=0, drop=True)
    lit = 1.0
    if geom is None:
        cos_inc = np.cos(np.radians(inc))
        cos_view, g = xr.ones_like(cos_inc), phase_angle(iof.solar_az if "solar_az" in iof.coords else 0.0, 90 - inc)
    else:
        o = geom.obs
        sun = (o.sun_azimuth.values, 90 - o.sun_zenith.values)
        slope, aspect, lit, sun = load_slope_az((o.facet_slope, o.facet_aspect, o.lit_frac, sun), like=like)
        # slope/aspect are referenced to the DEM's tangent plane, so the sun must be in that same frame
        cos_inc, cos_view, g = local_angles(slope, aspect, *sun)
    floors = {"cos_inc_floor_frac": 0.0, "cos_view_floor_frac": 0.0}
    if cos_min > 0.0:
        floors = {
            "cos_inc_floor_frac": float((cos_inc < cos_min).mean()),
            "cos_view_floor_frac": float((cos_view < cos_min).mean()),
        }
        cos_inc, cos_view = cos_inc.clip(min=cos_min), cos_view.clip(min=cos_min)

    model = get_model(photom)
    params = inspect.signature(model).parameters
    call = lambda **kw: model(**{k: v for k, v in kw.items() if k in params})
    if "iof" in params:
        out = call(iof=iof / lit, cos_inc=cos_inc, cos_view=cos_view, g=g, wl=iof.wl)
    else:
        i, e, g_std = STANDARD_GEOMETRY
        r_std = call(cos_inc=np.cos(np.radians(i)), cos_view=np.cos(np.radians(e)), g=g_std, wl=iof.wl)
        r = xr.ones_like(like) * call(cos_inc=cos_inc, cos_view=cos_view, g=g, wl=iof.wl) * lit
        # r is numpy-backed (topo); unchunked, dividing by it makes an eager (y, x, band) float64 cube
        if iof.chunks is not None:
            r = r.chunk({"y": iof.chunksizes["y"]})
        out = iof * r_std / r.where(r > 0)  # facets the sun does not reach have no direct beam to normalize by
    if min_lit > 0.0:
        out = out.where(lit >= min_lit)
    out.attrs |= {
        **floors,
        "slope_az_used": geom is not None,
        "inc_min_deg": float(inc.min()),
        "inc_max_deg": float(inc.max()),
        "inc_mean_deg": float(inc.mean()),
    }
    return out


def no_correction():
    """No photometric correction: reflectance = I/F."""
    return 1.0


def lambert(cos_inc):
    """Lambert: cos_inc."""
    return cos_inc


def lommel_seeliger(cos_inc, cos_view):
    """Lommel-Seeliger (single scattering), normalized to 1 at cos_inc = cos_view = 1."""
    return 2 * cos_inc / (cos_inc + cos_view)


def lunar_lambert(cos_inc, cos_view, g, coeffs=(1.0, -0.019, 2.42e-4, -1.46e-6)):
    """Lunar-Lambert with McEwen's phase-dependent L(g) (McEwen 1991).

    L(g) blends Lommel-Seeliger (L=1, limb-darkened) into Lambert (L=0). The default cubic is the
    one ISIS ships; pass `coeffs` for a band- or site-fitted L(g).
    """
    # Horner by hand: np.polyval drops xarray dims, and g may be per-line (y,) against (y, x) cosines.
    L = np.clip(sum(c * g**i for i, c in enumerate(coeffs)), 0.0, 1.0)
    return L * lommel_seeliger(cos_inc, cos_view) + (1 - L) * cos_inc


def besse_m3_highland(cos_inc, cos_view, g, wl):
    """`lunar_lambert` times the Besse et al. (2013) M3 highland phase function f(g, wl) (`besse_phase`).

    Fit to M3 highland spectra: a reference for comparing with M3 products, not an IIRS photometric model.
    """
    f = xr.DataArray(besse_phase(np.asarray(g), np.asarray(wl)), dims=("band", *getattr(g, "dims", ())))
    return lunar_lambert(cos_inc, cos_view, g) * f


def issdc(iof, cos_inc):
    """ISSDC's L2 normalization, I/F / cos_inc (Lambert to overhead sun, not the standard geometry)."""
    return iof / cos_inc.where(cos_inc > 0)


MODELS = {
    "none": no_correction,
    "lambert": lambert,
    "lommel_seeliger": lommel_seeliger,
    "lunar_lambert": lunar_lambert,
    "besse_m3_highland": besse_m3_highland,
    "issdc": issdc,
}


def get_model(photom):
    """The model function for `photom`, a `MODELS` name or any callable (see the module docstring)."""
    if callable(photom):
        return photom
    try:
        return MODELS[photom]
    except KeyError:
        raise ValueError(f"unknown photometric model {photom!r}; have {sorted(MODELS)}") from None


def besse_phase(g, wl_nm):
    """Besse et al. (2013) M3 highland phase function f(g) at wl_nm [nm]: (n_wl, *g.shape), g clamped to 0-85 deg."""
    tab = np.loadtxt(M3_DATA.joinpath("M3G20111109_RFL_F_ALPHA_HIL.TAB").open(), skiprows=1)
    alpha, f = tab[:86, 0], tab[:86, 1:]
    wl = np.loadtxt(M3_DATA.joinpath("M3G20110224_RFL_SOLAR_SPEC.TAB").open())[:, 0]
    if f.shape[1] != wl.size:
        raise ValueError(f"M3 phase table has {f.shape[1]} channels but solar spectrum has {wl.size}")
    wl_arr = np.atleast_1d(np.asarray(wl_nm, dtype=float))
    f_wl = np.vstack([np.interp(np.clip(wl_arr, wl.min(), wl.max()), wl, row) for row in f])  # (86, n_wl)
    g_arr = np.clip(np.asarray(g, dtype=float), 0.0, 85.0)
    vals = np.stack([np.interp(g_arr, alpha, f_wl[:, j]) for j in range(wl_arr.size)], axis=0)
    return vals if np.ndim(wl_nm) else vals[0]


def cos_angle(slope, aspect, az, elev):
    """Cosine of the angle between the facet normal and a direction (az, elev), all degrees.

    Flat ground gives sin(elev), the cosine of the angle from vertical; a facet tilted toward the direction gives more.
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


def load_slope_az(slope_az, like=None):
    """(slope, aspect, lit, sun) from a `(slope, aspect[, lit[, sun]])` tuple; `lit` is 1.0 when absent.

    `lit` is the fraction of the solar disk the terrain leaves visible. `sun` is the (azimuth, elevation)
    in the same tangent-plane frame as slope and aspect, or None.

    When `like` is given the fields are matched to its y/x: a shape check and a coordinate
    transplant, never a resample.
    """
    slope, aspect, *rest = slope_az
    lit = rest[0] if rest else None
    sun = rest[1] if len(rest) > 1 else None
    if like is not None:
        if slope.shape != like.shape:
            raise ValueError(
                f"slope_az {slope.shape} does not match the image {like.shape}; regenerate it for this crop"
            )
        coords = {d: like[d] for d in like.dims if d in like.coords}
        conv = lambda v: None if v is None else xr.DataArray(np.asarray(v), coords=coords, dims=like.dims)
        slope, aspect, lit = conv(slope), conv(aspect), conv(lit)
        if isinstance(sun, tuple) and hasattr(sun[0], "shape"):
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
