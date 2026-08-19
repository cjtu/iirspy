"""
Co-register an IIRS strip to LOLA topography and hand back GCPs + a geotransform.

The supplied geometry csv is off by 4.8-8.8 km at the lunar south pole, and the error does not
share a sign between scenes, so every scene needs its own solve. This module runs that solve and
returns the result as ground control points in *camera* space, so the pipeline can produce a
registered L1/L2 with a single resample of the raw band -- never a resample of a resample.

    reference   two-tier tangent-plane hillshade rendered from LOLA at the scene's SPICE sun
    coarse      unbounded masked phase correlation, injection-gated  (kills the 5-9 km bulk error)
    fine        AROSICS tie points, iterated at a shrinking search radius
    output      GCP lattice whose targets carry coarse + every tie-point field, warped GCP_TPS

Typical use:

    reg = register("ch2_iir_<sid>_l1_polar.tif", fgeom, fspm)
    arr = warp(cube, reg)                  # (band, y, x) on reg.transform / reg.crs

`register` needs `arosics`, which needs `osgeo.gdal`, so it resolves only in a conda env.
Everything above the matcher -- DEM, hillshade, horizons, GCP lattice, warp -- runs without it.

Method and measurements: local-workspace/arosics/REPORT.md.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.control import GroundControlPoint
from rasterio.warp import Resampling, reproject
from scipy.interpolate import RBFInterpolator, RectBivariateSpline
from scipy.ndimage import binary_erosion, gaussian_filter, map_coordinates, rotate, uniform_filter
from scipy.spatial import cKDTree

import iirspy.utils as utils

MOON_RADIUS_M = 1737400.0
SOLAR_RADIUS_DEG = 0.266  # nominal; the true value per epoch comes from SPICE
LONLAT = CRS.from_authority("IAU", "30100")
STEREO = CRS.from_wkt(utils.IIRS_PROJ_DICT["polarstereographicsouthpole"])
TO_STEREO = Transformer.from_crs(LONLAT, STEREO, always_xy=True)
TO_LONLAT = Transformer.from_crs(STEREO, LONLAT, always_xy=True)


@dataclass
class GeorefConfig:
    """Everything tunable in one place."""

    # --- output grid
    aoi: tuple = (0.0, 0.0, 0.0, 0.0)  # xmin, ymin, xmax, ymax in stereo m; set by aoi_around()
    ps: float = 40.0  # m/px, between LOLA (20) and IIRS (76.5) so neither is badly aliased
    band: int = 54  # 1605.5 nm: strongest reflected-solar contrast, clear of the thermal tail

    # --- reference
    dem_near: str = ""  # LDEM_80S_20M .LBL
    dem_far: str = ""  # LDEM_60S_240M_JP2 .LBL; "" renders the near tier alone
    near_m: float = 20_000.0  # near-tier up-sun reach, and the far tier's min range
    far_m: float = 200_000.0  # past this no lunar relief is tall enough to cast
    margin_m: float = 2_000.0
    iirs_gsd_m: float = 76.5  # reconstructed from the IK+SPK; blur the reference to this

    # --- coarse search
    coarse_dec: int = 4  # decimation; 160 m/px keeps the unbounded search fast
    inject_px: tuple = ((30, -20), (-45, 25))  # injected control shifts, at the decimated scale
    inject_tol_px: int = 3

    # --- tie points
    niter: int = 5
    p95_stop_m: float = 20.0  # 1/2 output px; the matcher floor is 18.6-19.2 m
    win: int = 128  # px = 5.1 km matching window
    grid_res: int = 40  # px = 1.6 km tie-point spacing
    max_shift: dict = field(default_factory=lambda: {1: 50, 2: 15, 3: 6, 4: 4})  # px, then floor
    min_reliability: int = 30
    grad: bool = True  # match on gradient magnitude, not radiance
    grad_sigma: float = 1.0
    mad_from_iter: int = 3  # global MAD cull, only once the field is flat
    mad_k: float = 3.0

    # --- displacement field
    decay_m: float = 5_000.0  # blend to the bulk median outside tie-point support
    smooth_cv: bool = True  # pick TPS regularisation by k-fold CV
    smooth_fixed: float = 1.0  # used when smooth_cv is False

    # --- GCP lattice
    ncol: int = 13  # cross-track GCP columns; 6 undersamples the smile
    # Along-track GCP spacing in image rows; 25 rows is ~2 km, fine enough to carry the
    # tie-point field (1.6 km spacing) without resampling it away.
    row_step: int = 25

    def max_shift_for(self, k):
        """Search radius for iteration k; past the table, hold the tightest and keep refining."""
        return self.max_shift.get(k, min(self.max_shift.values()))


def aoi_around(lon, lat, half_m=80_000.0):
    """Square AOI in stereo metres centred on a lon/lat -- e.g. Haworth / the LRM landing site."""
    x, y = TO_STEREO.transform(lon, lat)
    return (x - half_m, y - half_m, x + half_m, y + half_m)


def grid_of(cfg):
    """(xs, ys, transform, profile) for the AOI output grid, north-up."""
    xs = np.arange(cfg.aoi[0], cfg.aoi[2], cfg.ps)
    ys = np.arange(cfg.aoi[3], cfg.aoi[1], -cfg.ps)
    tr = rasterio.transform.from_origin(xs[0] - cfg.ps / 2, ys[0] + cfg.ps / 2, cfg.ps, cfg.ps)
    prof = {
        "driver": "GTiff",
        "height": len(ys),
        "width": len(xs),
        "count": 1,
        "dtype": "float32",
        "crs": STEREO,
        "transform": tr,
        "nodata": np.nan,
        "compress": "LZW",
    }
    return xs, ys, tr, prof


# ------------------------------------------------------------------------------------------
# 1. DEM
# ------------------------------------------------------------------------------------------
def load_lola_elev(path, bounds=None):
    """
    Elevation [m] above the 1737.4 km sphere, for either LOLA polar GDR packaging.

    The two packagings do not read alike and GDAL's reported scales do not tell them apart: the
    JP2 products report scale 1.0 / offset 0.0 yet hold raw int16 counts needing the label's 0.5 m
    scaling, while the .IMG ones carry the radius offset. So read the raw band and apply the
    label's own SCALING_FACTOR/OFFSET.

    Returns (elev float64, rasterio transform, pixel_size_m).
    """
    import re

    txt = Path(path).read_text(errors="ignore")

    def lbl(key, default):
        m = re.search(rf"^\s*{key}\s*=\s*([-\d.]+)", txt, re.M)
        return float(m.group(1)) if m else default

    scale, offset = lbl("SCALING_FACTOR", 0.5), lbl("OFFSET", MOON_RADIUS_M)
    with rasterio.open(path) as src:
        win = None
        if bounds is not None:
            win = rasterio.windows.from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        raw = src.read(1, window=win).astype("float64")
        tr = src.window_transform(win) if win is not None else src.transform
        ps = abs(src.res[0])
    return raw * scale + offset - MOON_RADIUS_M, tr, ps


def tangent_z(dem, pixel_size, origin=None, radius=MOON_RADIUS_M):
    """
    Re-reference a DEM from "metres above the sphere" to height in the tangent plane at `origin`.

    At a 1-4 deg polar sun the local horizontal tilts ~5.8 deg across a 175 km window -- 1.9-2.2x
    the sun elevation itself -- so a flat-plane shadow sweep treats terrain the Moon has already
    curved below the horizon as a blocker. After this, one sun vector is correct everywhere.
    """
    ny, nx = np.shape(dem)
    r0, c0 = ((ny - 1) / 2, (nx - 1) / 2) if origin is None else origin
    dy = (np.arange(ny) - r0) * pixel_size
    dx = (np.arange(nx) - c0) * pixel_size
    return np.asarray(dem, dtype="float64") - (dx[None, :] ** 2 + dy[:, None] ** 2) / (2 * radius)


# ------------------------------------------------------------------------------------------
# 2. Horizons and cast shadows
# ------------------------------------------------------------------------------------------
def horizon_ladder(lo_deg=-2.0, hi_deg=14.0, step_deg=0.0625):
    """Ascending trial elevation angles the horizon is quantised onto (1/8 of the solar radius)."""
    return np.arange(lo_deg, hi_deg + step_deg, step_deg)


def penumbra_ladder(sun_elev, solar_radius_deg=SOLAR_RADIUS_DEG, n=9):
    """A ladder that spans just this sun's penumbra -- far cheaper than the full sweep."""
    return np.linspace(sun_elev - solar_radius_deg, sun_elev + solar_radius_deg, n)


def _rotate_upsun(z, az_deg):
    """Rotate so the look direction toward `az_deg` lands on +col. Returns (rot, valid, ang)."""
    # rows increase southward on a north-up raster; sign convention pinned against all 8 azimuths
    ang = np.degrees(np.arctan2(np.cos(np.radians(az_deg)), np.sin(np.radians(az_deg))))
    fill = np.nanmin(z) - 1e4
    rot = rotate(np.nan_to_num(z, nan=fill), -ang, reshape=True, order=1, cval=fill, mode="constant")
    return rot, rot > fill + 1.0, ang


def _unrotate(a, ang, shape, cval=0.0):
    """Rotate a field back into the original frame and centre-crop to `shape`."""
    back = rotate(a.astype("float32"), ang, reshape=True, order=1, cval=cval, mode="constant")
    r0 = (back.shape[0] - shape[0]) // 2
    c0 = (back.shape[1] - shape[1]) // 2
    return back[r0 : r0 + shape[0], c0 : c0 + shape[1]]


def _accumulate_blocked(rot, x, ladder, valid, skip=1):
    """Count ladder rungs below the skyline per cell, sweeping toward +col.

    The skyline exceeds angle t iff some upsun cell lies above the ray of slope tan(t) through the
    cell, i.e. iff the suffix-max of z - x*tan(t) (starting `skip` cells ahead) beats this cell.
    blocked(t) is monotone in t, so the count *is* the horizon's index in an ascending ladder.
    """
    acc = np.zeros(rot.shape, dtype="float32")
    zz = np.where(valid, rot, -np.inf)
    for t in ladder:
        g = zz - x[None, :] * np.tan(np.radians(t))
        # suffix max strictly ahead by `skip`
        sm = np.full_like(g, -np.inf)
        if g.shape[1] > skip:
            sm[:, :-skip] = np.maximum.accumulate(g[:, ::-1], axis=1)[:, ::-1][:, skip:]
        acc += (sm > g + 1e-9).astype("float32")
    return acc


def horizon_1az(z, pixel_size, az_deg, ladder=None, min_range_m=0.0):
    """Quantised horizon elevation toward `az_deg`, as an index into `ladder`.

    `min_range_m` ignores blockers nearer than that, which is how the near and far tiers compose:
    the near tier's own extent caps its range, the far tier skips the near tier's reach, and
    np.minimum of the two lit fractions is the combined answer.
    """
    ladder = horizon_ladder() if ladder is None else ladder
    rot, valid, ang = _rotate_upsun(np.asarray(z, dtype="float64"), az_deg)
    skip = max(1, round(min_range_m / pixel_size))
    acc = _accumulate_blocked(rot, np.arange(rot.shape[1]) * pixel_size, ladder, valid, skip)
    idx = np.clip(_unrotate(acc, ang, np.shape(z)), 0, len(ladder) - 1)
    return np.rint(idx).astype("uint16")


def lit_fraction(horizon_deg, sun_elev_deg, solar_radius_deg=SOLAR_RADIUS_DEG):
    """Fraction of the solar disk above the horizon, exact circular-segment area, in [0, 1]."""
    c = (np.asarray(horizon_deg, dtype="float64") - sun_elev_deg) / solar_radius_deg
    cc = np.clip(c, -1.0, 1.0)
    f = (np.arccos(cc) - cc * np.sqrt(np.maximum(0.0, 1.0 - cc * cc))) / np.pi
    return np.where(c >= 1.0, 0.0, np.where(c <= -1.0, 1.0, f))


def lambert_shade(dem, pixel_size, sun_az, sun_elev):
    """Lambertian cos(incidence) in [0, 1] from the metric surface normal. No cast shadows."""
    a, e = np.radians(sun_az), np.radians(sun_elev)
    s = np.array([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)])
    dzdy_row, dzdx = np.gradient(np.asarray(dem, dtype="float64"), pixel_size)
    nrm = np.stack([-dzdx, dzdy_row, np.ones_like(dem)])  # d/d(north) = -d/d(row)
    nrm /= np.linalg.norm(nrm, axis=0)
    return np.clip(nrm[0] * s[0] + nrm[1] * s[1] + nrm[2] * s[2], 0, 1)


# ------------------------------------------------------------------------------------------
# 3. Reference hillshade
# ------------------------------------------------------------------------------------------
def _enu(lon, lat):
    """East, north, up unit vectors at a body-fixed lon/lat (degrees)."""
    lo, la = np.radians(lon), np.radians(lat)
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    up = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return east, north, up


def sun_geometry(fgeom, fspm, cfg, kernels=None):
    """
    (az_grid, elev, solar_radius_deg) for the epoch this strip crosses the AOI, from SPICE.

    Three corrections a mean-spm hillshade misses, worth +0.4 to +0.9 correlation together:
    the sun azimuth is measured from *local* north but the grid's north differs by lon - lon_0
    (-5 to -50 deg along these strips); the tangent-plane elevation is not the spm's local one;
    and the solar angular radius is 0.2616-0.2700 deg here, not the hardcoded 0.25.
    """
    import spiceypy as sp

    for k in kernels or []:
        sp.furnsh(str(k))

    df = pd.read_csv(fgeom)
    lon = np.where(df.Longitude > 180, df.Longitude - 360, df.Longitude)
    gx, gy = TO_STEREO.transform(lon, df.Latitude.values)
    ins = (gx > cfg.aoi[0]) & (gx < cfg.aoi[2]) & (gy > cfg.aoi[1]) & (gy < cfg.aoi[3])
    spm = utils.load_iirs_spm(fspm)
    sub = spm[(spm.row >= df.Scan.values[ins].min()) & (spm.row <= df.Scan.values[ins].max())] if ins.any() else spm
    t = pd.Timestamp(sub.datetime.iloc[len(sub) // 2] if len(sub) else spm.datetime.iloc[0])
    et = sp.str2et(t.strftime("%Y-%m-%dT%H:%M:%S.%f"))

    v, _ = sp.spkpos("SUN", et, "IAU_MOON", "LT+S", "MOON")
    u = np.asarray(v) / np.linalg.norm(v)
    cx, cy = (cfg.aoi[0] + cfg.aoi[2]) / 2, (cfg.aoi[1] + cfg.aoi[3]) / 2
    clon, clat = TO_LONLAT.transform(cx, cy)
    east, north, up = _enu(clon, clat)
    lon0 = STEREO.to_dict().get("lon_0", 0.0)
    az_grid = (np.degrees(np.arctan2(u @ east, u @ north)) + (clon - lon0)) % 360
    el = np.degrees(np.arcsin(u @ up))
    r_sun = np.degrees(np.arcsin(695_700_000.0 / (np.linalg.norm(v) * 1000.0)))
    return float(az_grid), float(el), float(r_sun)


def _upsun_box(aoi, az_deg, reach_m, margin):
    """AOI grown by `reach_m` along the sun azimuth, both ways, plus a margin.

    Both ways so the horizon sweep has the blockers on the sun's side of every edge pixel.
    """
    dx, dy = abs(np.sin(np.radians(az_deg))) * reach_m, abs(np.cos(np.radians(az_deg))) * reach_m
    return (aoi[0] - dx - margin, aoi[1] - dy - margin, aoi[2] + dx + margin, aoi[3] + dy + margin)


def _onto(arr, tr_src, tr_dst, shape_dst):
    """Bilinear-resample `arr` from its own affine grid onto another."""
    r = np.arange(shape_dst[0])[:, None] * tr_dst.e + tr_dst.f
    c = np.arange(shape_dst[1])[None, :] * tr_dst.a + tr_dst.c
    return map_coordinates(
        arr,
        [np.broadcast_to((r - tr_src.f) / tr_src.e, shape_dst), np.broadcast_to((c - tr_src.c) / tr_src.a, shape_dst)],
        order=1,
        mode="nearest",
    )


def _tier(path, cfg, az, el, r_sun, reach_m, min_range_m):
    """Lit fraction for one DEM tier, on that DEM's own grid."""
    cx, cy = (cfg.aoi[0] + cfg.aoi[2]) / 2, (cfg.aoi[1] + cfg.aoi[3]) / 2
    z_raw, tr, ps = load_lola_elev(path, bounds=_upsun_box(cfg.aoi, az, reach_m, cfg.margin_m))
    z = tangent_z(z_raw, ps, origin=((cy - tr.f) / tr.e, (cx - tr.c) / tr.a))
    lad = penumbra_ladder(el, r_sun)
    lit = lit_fraction(lad[horizon_1az(z, ps, az, lad, min_range_m)], el, r_sun)
    return lit, tr, ps, z


def _to_gsd(a, ps_src, cfg):
    """Area-average a fine-grid field to the output cell, then blur it to the IIRS ground sample."""
    a = uniform_filter(a, size=max(1, round(cfg.ps / ps_src)), mode="nearest")
    return gaussian_filter(a, (cfg.iirs_gsd_m / 2.355) / ps_src)


def render_reference(fgeom, fspm, cfg, kernels=None):
    """
    Two-tier hillshade on the AOI grid: near field at native 20 m, far field at 240 m.

    A blocker at distance d occludes a sun at elevation e iff its height exceeds d*tan(e) +
    d^2/2R, so at 1-4 deg the horizon is set 100-160 km up-sun and a single 60 km window is blind
    to it -- always toward too much light. Returns (shade, info).
    """
    az, el, r_sun = sun_geometry(fgeom, fspm, cfg, kernels)
    lit_n, tr_n, ps_n, z_n = _tier(cfg.dem_near, cfg, az, el, r_sun, cfg.near_m, 0.0)
    lit = lit_n
    if cfg.dem_far:
        lit_f, tr_f, _, _ = _tier(cfg.dem_far, cfg, az, el, r_sun, cfg.far_m, cfg.near_m)
        lit = np.minimum(lit_n, _onto(lit_f, tr_f, tr_n, lit_n.shape))
    fine = lambert_shade(z_n, ps_n, az, el) * lit

    # Shadowing is nonlinear in topography, so render fine then average: area-average to the output
    # cell, then blur to the IIRS ground sample, removing structure the instrument cannot resolve.
    f = _to_gsd(fine, ps_n, cfg)
    xs, ys, _, _ = grid_of(cfg)
    mx, my = np.meshgrid(xs, ys)
    ci = np.clip(((mx - tr_n.c) / tr_n.a).astype(int), 0, f.shape[1] - 1)
    ri = np.clip(((my - tr_n.f) / tr_n.e).astype(int), 0, f.shape[0] - 1)
    return f[ri, ci].astype("float32"), {"az_grid": az, "elev": el, "r_sun": r_sun, "lit_frac": float(lit.mean())}


def slope_aspect(gx, gy):
    """(slope, aspect) in degrees from east/north surface gradients.

    Slope is measured from the local horizontal (the tangent plane, so the planet's curvature is
    already out of it) and aspect is the *downhill* azimuth, clockwise from grid north -- the
    direction the facet faces, which is the convention iirspy.photometry.cos_angle expects.
    """
    return np.degrees(np.arctan(np.hypot(gx, gy))), np.degrees(np.arctan2(-gx, -gy)) % 360.0


def render_topo(fgeom, fspm, cfg, kernels=None):
    """
    The hillshade taken apart into the terms a photometric model needs, on the near DEM grid.

    Same tangent-plane DEM, two-tier cast-shadow solve and smoothing chain as
    :func:`render_reference`, but returning the terms a model needs rather than one shading:
    gradients (-> slope/aspect) and `lit`, the fraction of the solar disk the horizon leaves
    visible.

    Gradients are smoothed, not slope/aspect, which are a nonlinear -- and for aspect, circular --
    function of the surface; the smoothed gradient is the gradient of the smoothed surface, the
    facet IIRS sees across its 76.5 m ground sample.

    Returns (gx, gy, lit, transform, info) on the near tier's own grid; feed it to
    :func:`camera_topo` to land in camera space.
    """
    az, el, r_sun = sun_geometry(fgeom, fspm, cfg, kernels)
    lit, tr_n, ps_n, z_n = _tier(cfg.dem_near, cfg, az, el, r_sun, cfg.near_m, 0.0)
    if cfg.dem_far:
        lit_f, tr_f, _, _ = _tier(cfg.dem_far, cfg, az, el, r_sun, cfg.far_m, cfg.near_m)
        lit = np.minimum(lit, _onto(lit_f, tr_f, tr_n, lit.shape))
    dzdy_row, dzdx = np.gradient(z_n, ps_n)
    gx, gy = _to_gsd(dzdx, ps_n, cfg), _to_gsd(-dzdy_row, ps_n, cfg)  # east, north
    return gx, gy, _to_gsd(lit, ps_n, cfg), tr_n, {"az_grid": az, "elev": el, "r_sun": r_sun}


def camera_xy(gcps, shape):
    """Map (x, y) at every camera pixel of `shape`, interpolated from a registration's GCP lattice.

    The lattice is regular in camera space (`row_step` rows by `ncol` columns), so a bicubic
    spline through it reproduces the warp between nodes: exact at the nodes, well under a 40 m
    output pixel between them.
    """
    rows = np.array(sorted({g.row for g in gcps}))
    cols = np.array(sorted({g.col for g in gcps}))
    ri = {v: i for i, v in enumerate(rows)}
    ci = {v: i for i, v in enumerate(cols)}
    xy = np.full((2, len(rows), len(cols)), np.nan)
    for g in gcps:
        xy[:, ri[g.row], ci[g.col]] = (g.x, g.y)
    if not np.isfinite(xy).all():
        raise ValueError("GCPs do not form a full lattice; camera_xy needs the lattice register() builds")
    ys, xs = np.arange(shape[0]), np.arange(shape[1])
    kx, ky = min(3, len(rows) - 1), min(3, len(cols) - 1)
    return tuple(RectBivariateSpline(rows, cols, a, kx=kx, ky=ky)(ys, xs) for a in xy)


def camera_topo(gcps, shape, fgeom, fspm, cfg, kernels=None):
    """(slope, aspect, lit) as (y, x) camera-space arrays, for the L2 photometric correction.

    Camera space, not the map grid: L2 is computed there, where the per-line solar geometry lives.
    Each camera pixel is sampled bilinearly at its registered map position, so its slope, aspect
    and lit fraction come from the DEM neighbourhood that produced the hillshade it matched.

    NaN outside `cfg.aoi`: the hillshade covers only that box, so only there are the GCPs measured
    rather than extrapolated. On a strip several times longer than the AOI that is most of the
    scene.
    """
    gx, gy, lit, tr, info = render_topo(fgeom, fspm, cfg, kernels)
    x, y = camera_xy(gcps, shape)
    rc = [(y - tr.f) / tr.e, (x - tr.c) / tr.a]
    # Sample the gradients, then convert: aspect is circular, so interpolating the angle would
    # average 359 and 1 degrees into 180.
    sx, sy, slit = (map_coordinates(a, rc, order=1, mode="constant", cval=np.nan) for a in (gx, gy, lit))
    out = (x < cfg.aoi[0]) | (x > cfg.aoi[2]) | (y < cfg.aoi[1]) | (y > cfg.aoi[3])
    sx, sy, slit = (np.where(out, np.nan, a) for a in (sx, sy, slit))
    slope, aspect = slope_aspect(sx, sy)
    return slope.astype("float32"), aspect.astype("float32"), slit.astype("float32"), info


def save_topo(fout, slope, aspect, lit, tags=None):
    """Write the camera-space topo product: a 3-band float32 GeoTIFF, bands named for L2.

    No CRS or transform -- this is camera space, and iirspy.photometry.load_topo matches it to the
    L1 cube by shape. Small enough (a few MB) to ship beside every L1.
    """
    with rasterio.open(
        fout,
        "w",
        driver="GTiff",
        height=slope.shape[0],
        width=slope.shape[1],
        count=3,
        dtype="float32",
        compress="LZW",
        tiled=True,
    ) as dst:
        for i, (name, a) in enumerate([("slope", slope), ("aspect", aspect), ("lit", lit)], start=1):
            dst.write(np.asarray(a, dtype="float32"), i)
            dst.set_band_description(i, name)
        dst.update_tags(**{k: str(v) for k, v in (tags or {}).items()})
    return fout


# ------------------------------------------------------------------------------------------
# 4. Image access and matching primitives
# ------------------------------------------------------------------------------------------
def read_band(ftif, band):
    """The (y, x) plane for IIRS band *number* `band`, wherever it sits in the file.

    A full 256-band product is positionally indexed; a streamed band subset is not, and assuming
    it is reads a different wavelength with no error at all -- so honour the `band_numbers` tag.
    """
    with rasterio.open(ftif) as src:
        nums = src.tags().get("band_numbers")
        idx = [int(b) for b in nums.split(",")].index(band) + 1 if nums else band
        return src.read(idx).astype("float32")


def _standardize(a, m, nodata):
    return np.where(m, (a - a[m].mean()) / (a[m].std() + 1e-9), nodata)


def grad_mag(a, m, sigma):
    """Gradient magnitude of a masked field, for illumination-invariant matching.

    IIRS radiance against a Lambertian hillshade is cross-modal with an unknown nonlinear transfer;
    gradient magnitude discards the level and keeps the structure. The data boundary is masked, not
    differenced -- a nodata edge is the strongest gradient in the scene.
    """
    fill = float(a[m].mean()) if m.any() else 0.0
    sm = gaussian_filter(np.where(m, a, fill), sigma)
    gy, gx = np.gradient(sm)
    return np.hypot(gy, gx), binary_erosion(m, iterations=int(np.ceil(3 * sigma)) + 1)


def coarse_shift(img, ref, cfg):
    """
    Unbounded masked phase correlation, with an injected-shift control gate.

    The geometry error is 5-9 km, far past any bounded search, so this searches all shifts at
    `coarse_dec` decimation. Every number is gated: inject a known shift, re-measure, and require
    the answer to move by that much. Returns (dx_m, dy_m, info).
    """
    from skimage.registration import phase_cross_correlation

    d = cfg.coarse_dec
    a, b = img[::d, ::d], ref[::d, ::d]

    def measure(x, y):
        mx, my = np.isfinite(x), np.isfinite(y)
        xs = np.where(mx, (x - x[mx].mean()) / (x[mx].std() + 1e-9), 0)
        ys = np.where(my, (y - y[my].mean()) / (y[my].std() + 1e-9), 0)
        s, _, _ = phase_cross_correlation(ys, xs, reference_mask=my, moving_mask=mx, overlap_ratio=0.15)
        return np.asarray(s, dtype=float)

    base = measure(a, b)
    controls, ok = [], []
    for dy, dx in cfg.inject_px:
        got = measure(np.roll(np.roll(a, dy, 0), dx, 1), b) - base
        passed = bool(abs(got[0] + dy) < cfg.inject_tol_px and abs(got[1] + dx) < cfg.inject_tol_px)
        controls.append({"inject": [dy, dx], "change": [round(got[0], 1), round(got[1], 1)], "passed": passed})
        ok.append(passed)
    ps = cfg.ps * d
    return (
        float(base[1] * ps),
        float(-base[0] * ps),
        {
            "shift_px": [round(base[0], 1), round(base[1], 1)],
            "shift_km": [round(base[1] * ps / 1e3, 2), round(-base[0] * ps / 1e3, 2)],
            "controls": controls,
            "trustworthy": all(ok),
        },
    )


def tie_points(ref, tgt, transform, cfg, max_shift_px):
    """One AROSICS COREG_LOCAL pass on two arrays already sharing a grid, filtered to usable points.

    Both are standardised (and optionally reduced to gradient magnitude) before matching; nodata is
    a sentinel because arosics does not take NaN.
    """
    from arosics import COREG_LOCAL
    from geoarray import GeoArray

    nodata = -9999.0
    rm, tm = np.isfinite(ref), np.isfinite(tgt)
    r, t = ref, tgt
    if cfg.grad:
        r, rm = grad_mag(r, rm, cfg.grad_sigma)
        t, tm = grad_mag(t, tm, cfg.grad_sigma)
    gt = (transform.c, transform.a, 0.0, transform.f, 0.0, transform.e)
    mk = lambda arr, m: GeoArray(
        _standardize(arr, m, nodata), geotransform=gt, projection=STEREO.to_wkt(), nodata=nodata
    )

    crl = COREG_LOCAL(
        mk(r, rm),
        mk(t, tm),
        grid_res=cfg.grid_res,
        window_size=(cfg.win, cfg.win),
        max_shift=max_shift_px,
        nodata=(nodata, nodata),
        q=True,
        progress=False,
        min_reliability=cfg.min_reliability,
        tieP_filter_level=3,
    )
    tp = crl.CoRegPoints_table
    return tp[(nodata != tp.ABS_SHIFT) & tp.X_SHIFT_PX.notna() & (cfg.min_reliability <= tp.RELIABILITY)]


def _reject_shift_cap(tp, cap_px):
    """Drop mis-locks past the search radius; a TPS interpolates *through* its control points."""
    return tp[np.hypot(tp.Y_SHIFT_PX, tp.X_SHIFT_PX) <= cap_px]


def _reject_mad(tp, k, ps):
    """Global MAD cut on |shift|. Safe only once the bulk field is flat -- hence mad_from_iter.

    At iteration 1 the field genuinely carries 300-500 m of coherent terrain-scale structure and
    this rejects 43% of points, deleting exactly what is being fitted. By iteration 3 the field is
    flat and the same rule trims isolated disagreement and converges an iteration earlier.
    """
    d = np.hypot(tp.Y_SHIFT_PX, tp.X_SHIFT_PX).to_numpy(float)
    if len(d) < 8:
        return tp, 0, None
    med = float(np.median(d))
    lim = med + k * 1.4826 * float(np.median(np.abs(d - med)))
    return tp[d <= lim], int((d > lim).sum()), round(lim * ps, 1)


# ------------------------------------------------------------------------------------------
# 5. Displacement field
# ------------------------------------------------------------------------------------------
SMOOTH_GRID = np.geomspace(1e-2, 1e9, 25)


def choose_smoothing(pts, disp, k=5, seed=0):
    """Pick the TPS regularisation by k-fold CV on the tie points themselves.

    An interpolating spline would reproduce matcher noise and extrapolate it outward, so the
    regularisation is fitted rather than fixed. Returns
    (smoothing, held_out_rms_m, bulk_median_rms_m).
    """
    fold = np.random.default_rng(seed).permutation(len(pts)) % k
    base = []
    for f in range(k):
        tr, te = fold != f, fold == f
        if tr.sum() >= 1 and te.any():
            base.append(np.linalg.norm(np.median(disp[tr], axis=0) - disp[te], axis=1))
    baseline = float(np.sqrt(np.mean(np.concatenate(base) ** 2))) if base else np.inf

    best, best_err = float(SMOOTH_GRID[-1]), np.inf
    for s in SMOOTH_GRID:
        err = []
        for f in range(k):
            tr, te = fold != f, fold == f
            if tr.sum() < 4 or not te.any():
                continue
            try:
                m = RBFInterpolator(pts[tr], disp[tr], kernel="thin_plate_spline", smoothing=float(s))
                err.append(np.linalg.norm(m(pts[te]) - disp[te], axis=1))
            except Exception:  # singular at tiny smoothing -- that candidate is out
                err.clear()
                break
        if not err:
            continue
        e = float(np.sqrt(np.mean(np.concatenate(err) ** 2)))
        if e < best_err:
            best, best_err = float(s), e
    return best, best_err, baseline


def displacement_field(tp, cfg):
    """
    Tie-point displacement field q -> (dx, dy) metres, decaying to the bulk median off-support.

    The GCP lattice spans the whole strip but tie points exist only where it crosses the AOI. A TPS
    diverges outside its control-point hull, so beyond the support the field blends to the bulk
    median with a Gaussian in distance-to-nearest.
    """
    pts = tp[["X_MAP", "Y_MAP"]].to_numpy(float)
    disp = np.c_[tp.X_SHIFT_PX.to_numpy(float) * cfg.ps, -tp.Y_SHIFT_PX.to_numpy(float) * cfg.ps]
    bulk = np.median(disp, axis=0)
    info = {"n": len(pts), "bulk_m": [round(float(bulk[0]), 1), round(float(bulk[1]), 1)]}
    if len(tp) < 3:  # too few for a spline; the bulk shift is the trustworthy part anyway
        return (lambda q: np.repeat(bulk[None], len(q), axis=0)), info

    smooth = cfg.smooth_fixed
    if cfg.smooth_cv:
        smooth, cv_rms, base_rms = choose_smoothing(pts, disp)
        info |= {
            "smoothing": smooth,
            "cv_rms_m": round(cv_rms, 1),
            "bulk_median_rms_m": round(base_rms, 1),
            "beats_bulk": bool(cv_rms < base_rms),
        }
    rbf = RBFInterpolator(pts, disp, kernel="thin_plate_spline", smoothing=smooth)
    tree = cKDTree(pts)

    def fieldfn(q):
        w = np.exp(-((tree.query(q)[0] / cfg.decay_m) ** 2))[:, None]
        return w * rbf(q) + (1 - w) * np.repeat(bulk[None], len(q), axis=0)

    return fieldfn, info


# ------------------------------------------------------------------------------------------
# 6. GCP lattice and warp
# ------------------------------------------------------------------------------------------
@dataclass
class Registration:
    """The solve: GCPs in camera space plus the grid they target."""

    gcps: list
    crs: CRS
    transform: rasterio.Affine
    shape: tuple  # (height, width) of the output grid
    stats: dict

    @property
    def converged(self):
        return bool(self.stats.get("converged"))


def gcp_lattice(fgeom, ny, nx, cfg):
    """Camera (row, col) -> map (x, y) lattice from the supplied geometry csv, uncorrected."""
    glon, glat, _ = utils.geom2grid(fgeom, (-180, 180, -90, -80.0))
    glon, glat = glon[:ny], glat[:ny]
    # The last row must be a GCP row: arange(0, ny, row_step) stops short, leaving the final ~1 km
    # beyond every GCP where GDAL's TPS extrapolates freely -- i.e. the along-track image edge.
    rows = np.unique(np.r_[np.arange(0, ny, cfg.row_step), ny - 1])
    cols = np.unique(np.linspace(0, nx - 1, cfg.ncol).astype(int))
    jj, ii = np.meshgrid(rows, cols, indexing="ij")
    x, y = TO_STEREO.transform(glon[jj, ii], glat[jj, ii])
    return jj, ii, x, y


def _to_gcps(jj, ii, x, y):
    return [
        GroundControlPoint(row=float(jj[a, b]), col=float(ii[a, b]), x=float(x[a, b]), y=float(y[a, b]))
        for a in range(jj.shape[0])
        for b in range(jj.shape[1])
    ]


def project(band, gcps, cfg, resampling=Resampling.bilinear):
    """One reproject of a camera-space band onto the AOI grid through `gcps`.

    METHOD=GCP_TPS: a plain GCP list makes GDAL fit one global polynomial, which smooths the
    correction away; TPS interpolates the GCPs instead.
    """
    xs, ys, tr, _ = grid_of(cfg)
    out = np.full((len(ys), len(xs)), np.nan, "float32")
    reproject(
        source=band,
        destination=out,
        src_crs=STEREO,
        gcps=gcps,
        dst_transform=tr,
        dst_crs=STEREO,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        METHOD="GCP_TPS",
    )
    return out


def register(ftif, fgeom, fspm, cfg, kernels=None, reference=None, verbose=False):
    """
    Solve a scene's registration against LOLA. Returns a :class:`Registration`.

    The correction is composed into the GCP *targets*, one iteration at a time -- each tie-point
    field maps a current map position to the correction needed there, so applying field k at the
    position field k-1 produced is the right order, and the image is only ever resampled once.
    """
    ref = reference
    info = {}
    if ref is None:
        ref, hs_info = render_reference(fgeom, fspm, cfg, kernels)
        info["reference"] = hs_info

    band = read_band(ftif, cfg.band)
    ny, nx = band.shape
    jj, ii, x, y = gcp_lattice(fgeom, ny, nx, cfg)
    _, _, tr, _ = grid_of(cfg)

    # base: supplied geometry alone -- what the coarse search is measured on
    img = project(band, _to_gcps(jj, ii, x, y), cfg)
    dx, dy, cinfo = coarse_shift(img, ref, cfg)
    info["coarse"] = cinfo
    x, y = x + dx, y + dy

    iters, converged = [], False
    for k in range(1, cfg.niter + 1):
        img = project(band, _to_gcps(jj, ii, x, y), cfg)
        tp = tie_points(ref, img, tr, cfg, cfg.max_shift_for(k))
        tp = _reject_shift_cap(tp, cfg.max_shift_for(k))
        n_mad, mad_lim = 0, None
        if cfg.mad_from_iter and k >= cfg.mad_from_iter:
            tp, n_mad, mad_lim = _reject_mad(tp, cfg.mad_k, cfg.ps)
        if not len(tp):
            iters.append({"iter": k, "n_kept": 0, "note": "no tie points"})
            break

        d = np.hypot(tp.Y_SHIFT_PX, tp.X_SHIFT_PX).to_numpy(float) * cfg.ps
        p95 = float(np.percentile(d, 95))
        fieldfn, finfo = displacement_field(tp, cfg)
        s = fieldfn(np.c_[x.ravel(), y.ravel()]).reshape(*x.shape, 2)
        x, y = x + s[..., 0], y + s[..., 1]
        iters.append({
            "iter": k,
            "max_shift_px": cfg.max_shift_for(k),
            "n_kept": len(tp),
            "n_mad_rejected": n_mad,
            "mad_limit_m": mad_lim,
            "shift_m": {"median": round(float(np.median(d)), 1), "p95": round(p95, 1)},
            "applied_med_m": round(float(np.median(np.hypot(s[..., 0], s[..., 1]))), 1),
            "field": finfo,
        })
        if verbose:
            print(iters[-1], flush=True)
        # p95 of iteration k is the residual measured on product k-1; under a pixel means the
        # product this iteration just produced is converged.
        if p95 < cfg.p95_stop_m:
            converged = True
            break

    xs, ys, tr, _ = grid_of(cfg)
    return Registration(
        gcps=_to_gcps(jj, ii, x, y),
        crs=STEREO,
        transform=tr,
        shape=(len(ys), len(xs)),
        stats=info | {"iters": iters, "converged": converged, "band": cfg.band},
    )


def warp(cube, reg, cfg, resampling=Resampling.bilinear):
    """
    Apply a :class:`Registration` to a cube or single band. Returns a DataArray on the AOI grid.

    `cube` may be a path to an IIRS product or a (band, y, x) DataArray in camera space. Every
    band is resampled once, straight from camera space.
    """
    if isinstance(cube, str | Path):
        da = xr.open_dataarray(cube, engine="rasterio")
        from iirspy.iirs import _band_numbers

        da = da.assign_coords(band=_band_numbers(da))
    else:
        da = cube
    if "band" not in da.dims:
        da = da.expand_dims("band")

    xs, ys, tr, _ = grid_of(cfg)
    out = np.stack([
        project(da.isel(band=i).values.astype("float32"), reg.gcps, cfg, resampling) for i in range(da.sizes["band"])
    ])
    res = xr.DataArray(
        out,
        dims=("band", "y", "x"),
        coords={"band": da.band.values, "y": ys, "x": xs},
        attrs=dict(da.attrs) | {"georef_converged": str(reg.converged)},
    )
    if "wl" in da.coords:
        res = res.assign_coords(wl=("band", da.wl.values))
    return res.rio.write_crs(reg.crs).rio.write_transform(tr)
