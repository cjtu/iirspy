"""
Co-register an IIRS strip to LOLA topography and hand back GCPs + a geotransform.

The supplied geometry csv is off by up to 10 km, particularly at the poles. Use image feature
matching to produce ground control points in camera (x,y) space, so the pipeline can produce a
registered L1/L2 directly from the GCPs.

    reference   two-tier tangent-plane hillshade rendered from LOLA at the scene's SPICE sun
    coarse      unbounded masked phase correlation, injection-gated  (kills the 5-9 km bulk error)
    fine        AROSICS tie points, iterated at a shrinking search radius
    output      GCP lattice whose targets carry coarse + every tie-point field, warped GCP_TPS

Typical use:

    reg = register("ch2_iir_<sid>_l1_polar.tif", fgeom, fspm, cfg)
    arr = warp(cube, reg, cfg)             # (band, y, x) on reg.transform / reg.crs

The matcher needs `arosics`, which needs `osgeo.gdal`, which needs conda. `register` handles
that (if arosics not importable it runs the solve in a conda env through :mod:`iirspy.coreg`).
"""

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.control import GroundControlPoint
from rasterio.warp import Resampling, reproject
from scipy.interpolate import RBFInterpolator, RectBivariateSpline
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    map_coordinates,
    rotate,
    uniform_filter,
)
from scipy.spatial import cKDTree

import iirspy.utils as utils

AROSICS_ENV = os.environ.get("IIRSPY_AROSICS_ENV", "arosics")  # conda env iirspy.coreg runs in
MAX_RAM_BYTES = float(os.environ.get("IIRSPY_MAX_RAM_BYTES", 4e9))  # past this, project() spills to disk

MOON_RADIUS_M = 1737400.0
SOLAR_RADIUS_DEG = 0.266  # nominal; the true value per epoch comes from SPICE
LONLAT = CRS.from_authority("IAU", "30100")
STEREO = CRS.from_wkt(utils.IIRS_PROJ_DICT["polarstereographicsouthpole"])
TO_STEREO = Transformer.from_crs(LONLAT, STEREO, always_xy=True)
TO_LONLAT = Transformer.from_crs(STEREO, LONLAT, always_xy=True)


@lru_cache
def stereo_crs(pole: str = "south") -> CRS:
    """The IIRS projected CRS for `pole` ("south", "north", or "equatorial" equidistant cylindrical)."""
    key = "equatorial" if pole == "equatorial" else f"polarstereographic{pole}pole"
    return CRS.from_wkt(utils.IIRS_PROJ_DICT[key])


@lru_cache
def to_stereo(pole: str = "south") -> Transformer:
    return Transformer.from_crs(LONLAT, stereo_crs(pole), always_xy=True)


@lru_cache
def to_lonlat(pole: str = "south") -> Transformer:
    return Transformer.from_crs(stereo_crs(pole), LONLAT, always_xy=True)


@dataclass
class GeorefConfig:
    """Everything tunable in one place."""

    # --- output grid
    pole: str = "south"  # "south", "north", or "equatorial" -- selects the projected CRS
    aoi: tuple = (0.0, 0.0, 0.0, 0.0)  # xmin, ymin, xmax, ymax in stereo m; set by aoi_around()
    ps: float = 40.0  # m/px, between LOLA (20) and IIRS (76.5) so neither is badly aliased
    band: int = 54  # 1605.5 nm: strongest reflected-solar contrast, clear of the thermal tail

    # --- reference
    dem_near: str = ""  # LDEM_80S_20M .LBL
    dem_far: str = ""  # LDEM_60S_240M_JP2 .LBL; "" renders the near tier alone
    near_m: float = 20_000.0  # near-tier up-sun reach, and the far tier's min range
    far_m: float = 206_000.0  # hard ceiling on the up-sun search; see `shadow_reach_m`
    max_relief_m: float = 12_180.0  # tallest peak above a point, within reach; see `shadow_reach_m`
    far_dec: int = 4  # far-tier read decimation; horizon-blocker search doesn't need native res,
    # and this lets dem_far point at the same product as dem_near -- reads its overview pyramid
    # (JP2 ships one) instead of needing a separate, coarser DEM product for reach alone
    margin_m: float = 2_000.0
    iirs_gsd_m: float = 76.5  # reconstructed from the IK+SPK; blur the reference to this

    # --- coarse search
    coarse_dec: int = 4  # decimation; 160 m/px keeps the unbounded search fast
    inject_px: tuple = ((30, -20), (-45, 25))  # injected control shifts, at the decimated scale
    inject_tol_px: int = 3
    # How far the initial shift can look (capped at ~2x the largest geometry error, ~9 km)
    coarse_max_m: float = 20_000.0

    # --- quality gates
    # Fraction of tie-point candidates that must survive reliability check for chunk to be registered
    min_match_frac: float = 0.10  # Registered chunks keep 0.68-0.86, broken ones keep 0.003-0.009
    min_kept: int = 50  # a p95 over fewer points than this is not evidence of convergence

    # --- tie points
    niter: int = 6  # Max number of iters before run stops with converged=False
    p95_stop_m: float = 0.0  # Stop if the p95 error is below this; 0 disables
    p95_plateau_frac: float = 0.02  # Stop if fit improved by less than this fraction per iter (0 = off)
    min_iter: int = 0  # Minimum number of iterations (0 = off)
    win: int = 128  # px = 5.1 km matching window
    grid_res: int = 40  # px = 1.6 km tie-point spacing
    max_shift: dict = field(default_factory=lambda: {1: 50, 2: 15, 3: 6, 4: 4})  # px, then floor
    min_reliability: int = 30
    edge_reject_m: float = 1000  # 0 = off; drop tie points within this dist of the swath's nodata (edge or shadow)
    shadow_max: float = 0.75  # 1 = off; drop points whose reference window is more shadowed than this
    # Second pass at half `grid_res` within this far of the swath edge, `win` unchanged: the grid is
    # anchored to the array origin, so the outermost point can sit a full grid_res inboard of where
    # window quality runs out. Widening the margin this way beats shrinking `win`, which carries the
    # terrain the match needs.
    edge_dense_m: float = 0.0  # 0 = off
    grad: bool = True  # match on gradient magnitude, not radiance
    grad_sigma: float = 1.0
    mad_from_iter: int = 3  # global MAD cull, only once the field is flat
    mad_k: float = 3.0

    # --- displacement field
    # how to treat pixels far from identified tie points
    decay_m: float = 5_000.0  # distance at which the field falls back on overall median (or local plane)
    # Edge fit substitutes a local plan for the bulk median so swath margin trusts local tie points over global
    edge_fit_k: int = 8  # 0 = off, use the bulk median
    smooth_cv: bool = True  # pick TPS regularisation by k-fold CV
    smooth_fixed: float = 1.0  # used when smooth_cv is False

    # --- GCP lattice
    ncol: int = 13  # cross-track GCP columns; 6 undersamples the smile
    row_step: int = 25  # Along-track GCP spacing (rows); 25 is ~2km, so minimal resampling at 1.6 km tie-point spacing
    # The latitude crop of current cube (needed to compute offset from raw to computed GCP grid)
    lat_band: tuple | None = None
    # The chunk rows this solve cares about (rows outside chunk mostly irrelevant after small margin)
    gcp_rows: tuple | None = None
    gcp_row_margin: int = 250  # px, margin around gcp_rows to keep in the lattice; -1 = keep all rows

    def __post_init__(self):
        # json round-trips (iirspy.coreg) stringify the keys of max_shift
        self.max_shift = {int(k): v for k, v in self.max_shift.items()}

    def max_shift_for(self, k):
        """Search radius for iteration k; past the table, hold the tightest and keep refining."""
        return self.max_shift.get(k, min(self.max_shift.values()))


def aoi_around(lon, lat, half_m=80_000.0, pole="south"):
    """Square AOI in stereo metres centred on a lon/lat -- e.g. Haworth / the LRM landing site."""
    x, y = to_stereo(pole).transform(lon, lat)
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
        "crs": stereo_crs(cfg.pole),
        "transform": tr,
        "nodata": np.nan,
        "compress": "LZW",
    }
    return xs, ys, tr, prof


# ------------------------------------------------------------------------------------------
# 1. DEM
# ------------------------------------------------------------------------------------------
def load_lola_elev(path, bounds=None, dec=1):
    """
    Elevation [m] above the 1737.4 km sphere, for either LOLA polar GDR packaging.

    The two packagings do not read alike and GDAL's reported scales do not tell them apart: the
    JP2 products report scale 1.0 / offset 0.0 yet hold raw int16 counts needing the label's 0.5 m
    scaling, while the .IMG ones carry the radius offset. So read the raw band and apply the
    label's own SCALING_FACTOR/OFFSET.

    `dec` > 1 decimates the read via `out_shape`, which pulls from the file's own overview
    pyramid instead of decoding at native resolution -- cheap on a JP2, which ships one.

    Returns (elev float64, rasterio transform, pixel_size_m).
    """
    import re

    if "GLD100" in Path(path).name.upper():
        # WAC GLD100 is not a LOLA GDR: its DN *is* metres above the 1737.4 km sphere (unity scale,
        # no radius offset -- see make_wac_gld100_cog.sh). Applying the LOLA 0.5/1737400 trick here
        # halves every height and is silent about it.
        scale, offset = 1.0, MOON_RADIUS_M
    elif Path(path).suffix.lower() in (".tif", ".tiff"):
        # COG rebuild of a GDR (local-workspace/make_dem_cogs.py) -- the raw int16 counts are
        # copied through untouched, and every LOLA GDR label carries these same two constants,
        # so they are inlined rather than shipped in a sidecar. GDAL's own scale/offset tags stay
        # untrusted here for the same reason as above.
        scale, offset = 0.5, MOON_RADIUS_M
    else:
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
        win_w, win_h = (win.width, win.height) if win is not None else (src.width, src.height)
        out_shape = (max(1, round(win_h / dec)), max(1, round(win_w / dec))) if dec > 1 else None
        raw = src.read(1, window=win, out_shape=out_shape).astype("float64")
        nodata = src.nodata
        tr = src.window_transform(win) if win is not None else src.transform
        if out_shape is not None:
            tr = tr * tr.scale(win_w / out_shape[1], win_h / out_shape[0])
        ps = abs(src.res[0]) * (win_w / out_shape[1] if out_shape is not None else 1)
    elev = raw * scale + offset - MOON_RADIUS_M
    # A DEM that declares nodata (GLD100 stops at ~+/-79 and outside its inscribed circle) would
    # otherwise carry its fill DN straight through as a -32 km pit that shadows the whole render.
    # Flat fill at the window's own median: no relief to cast with, no cliff at the coverage edge.
    if nodata is not None:
        gap = raw == nodata
        if gap.any():
            elev[gap] = np.median(elev[~gap]) if (~gap).any() else 0.0
    return elev, tr, ps


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
    gx, gy = to_stereo(cfg.pole).transform(lon, df.Latitude.values)
    ins = (gx > cfg.aoi[0]) & (gx < cfg.aoi[2]) & (gy > cfg.aoi[1]) & (gy < cfg.aoi[3])
    spm = utils.load_iirs_spm(fspm)
    sub = spm[(spm.row >= df.Scan.values[ins].min()) & (spm.row <= df.Scan.values[ins].max())] if ins.any() else spm
    t = pd.Timestamp(sub.datetime.iloc[len(sub) // 2] if len(sub) else spm.datetime.iloc[0])
    et = sp.str2et(t.strftime("%Y-%m-%dT%H:%M:%S.%f"))

    v, _ = sp.spkpos("SUN", et, "IAU_MOON", "LT+S", "MOON")
    u = np.asarray(v) / np.linalg.norm(v)
    cx, cy = (cfg.aoi[0] + cfg.aoi[2]) / 2, (cfg.aoi[1] + cfg.aoi[3]) / 2
    clon, clat = to_lonlat(cfg.pole).transform(cx, cy)
    east, north, up = _enu(clon, clat)
    lon0 = stereo_crs(cfg.pole).to_dict().get("lon_0", 0.0)
    # True-north bearing -> grid bearing needs (clon - lon0) added at the south pole but SUBTRACTED
    # at the north pole: true north points radially *outward* from a south-pole-centred map but
    # radially *inward* on a north-pole-centred one, which flips the handedness of the correction.
    # Measured directly against both CRS's (verified against a finite-difference "which way does
    # increasing latitude move on the map" probe, exact to 1e-9 deg at 8 test azimuths per pole).
    # The equatorial equidistant-cylindrical CRS has no convergence at all -- meridians are straight
    # vertical lines on that map, so grid north equals true north everywhere and the term drops out.
    grid_sign = 0.0 if cfg.pole == "equatorial" else (-1.0 if cfg.pole == "north" else 1.0)
    az_grid = (np.degrees(np.arctan2(u @ east, u @ north)) + grid_sign * (clon - lon0)) % 360
    el = np.degrees(np.arcsin(u @ up))
    r_sun = np.degrees(np.arcsin(695_700_000.0 / (np.linalg.norm(v) * 1000.0)))
    return float(az_grid), float(el), float(r_sun)


def shadow_reach_m(elev_deg, relief_m=12_180.0, radius=MOON_RADIUS_M):
    """Farthest a blocker of height `relief_m` can still cast, at sun elevation `elev_deg` [m].

    A blocker at range d clears the line of sight when its height exceeds ``d*tan(e)`` plus the
    ``d^2/2R`` the surface has already curved away, so

        d = R * (-tan(e) + sqrt(tan(e)^2 + 2h/R))

    At e -> 0 this is the curvature horizon ``sqrt(2*R*h)``, and no sun angle can reach past it.

    `relief_m` defaults to the tallest peak standing above any point within reach, measured off the
    LOLA polar tiles themselves (12.180 km, LDEM_45S_100M; LDEM_75S_30M gives 11.87 km). That is
    deliberately *peak-above-the-shadowed-point*, not tile-wide peak-to-trough (14.7-15.7 km) --
    the latter pairs a summit with a basin floor up to twice the reach apart, which cannot shadow
    each other, and would oversize the read by ~17%.

    >>> round(shadow_reach_m(0.0))          # curvature horizon, sqrt(2Rh)
    205726
    >>> round(shadow_reach_m(3.16))         # deep polar chunk, sun barely up
    131069
    >>> round(shadow_reach_m(34.28))        # far end of a polar strip
    17736
    >>> shadow_reach_m(-5.0) == shadow_reach_m(0.0)   # sun below the horizon: no less reach
    True
    """
    t = np.tan(np.radians(max(float(elev_deg), 0.0)))
    return float(radius * (-t + np.sqrt(t * t + 2 * relief_m / radius)))


def _far_reach(cfg, elev_deg):
    """Per-render up-sun distance for the far tier, rounded UP to the next kilometre.

    Missing a blocker leaves the reference too bright, which is a correctness error in the thing
    the tie points match against; carrying a slightly oversized DEM window costs ~1% of a chunk's
    wall clock (the near tier dominates the read). So every rounding here goes outward.
    """
    d = min(float(cfg.far_m), shadow_reach_m(elev_deg, cfg.max_relief_m))
    return float(np.ceil(d / 1000.0) * 1000.0)


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


def _tier(path, cfg, az, el, r_sun, reach_m, min_range_m, dec=1):
    """Lit fraction for one DEM tier, on that DEM's own grid."""
    cx, cy = (cfg.aoi[0] + cfg.aoi[2]) / 2, (cfg.aoi[1] + cfg.aoi[3]) / 2
    z_raw, tr, ps = load_lola_elev(path, bounds=_upsun_box(cfg.aoi, az, reach_m, cfg.margin_m), dec=dec)
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
        lit_f, tr_f, _, _ = _tier(cfg.dem_far, cfg, az, el, r_sun, _far_reach(cfg, el), cfg.near_m, dec=cfg.far_dec)
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
        lit_f, tr_f, _, _ = _tier(cfg.dem_far, cfg, az, el, r_sun, _far_reach(cfg, el), cfg.near_m, dec=cfg.far_dec)
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


def _bounded_peak(xcorr, shape, cap_px):
    """Peak of a full-mode masked cross-correlation, searched only within `cap_px` of no shift.

    `cross_correlate_masked(moving, reference, ..., mode="full")` holds shift `s` at index
    `reference.shape - 1 - s`, so the admissible shifts are one contiguous window of `xcorr`.

    >>> x = np.zeros((9, 9)); x[4, 4] = 1.0; x[0, 0] = 5.0  # zero shift, and a taller peak at +4
    >>> [float(v) for v in _bounded_peak(x, (5, 5), 4)]
    [4.0, 4.0]
    >>> [float(v) for v in _bounded_peak(x, (5, 5), 2)]
    [0.0, 0.0]
    """
    ny, nx = shape
    r0, c0 = max(ny - 1 - cap_px, 0), max(nx - 1 - cap_px, 0)
    win = xcorr[r0 : min(ny + cap_px, xcorr.shape[0]), c0 : min(nx + cap_px, xcorr.shape[1])]
    i, j = np.unravel_index(np.argmax(win), win.shape)
    return np.array([ny - 1 - (i + r0), nx - 1 - (j + c0)], dtype=float)


def coarse_shift(img, ref, cfg):
    """
    Masked phase correlation bounded to `coarse_max_m`, with an injected-shift control gate.

    The geometry error is 5-9 km, so the search runs at `coarse_dec` decimation and the peak is
    taken only within `coarse_max_m`; a low-contrast pair otherwise peaks on a nearly-disjoint
    overlap and returns a shift that puts the swath outside its own AOI.

    Every number is gated: inject a known shift, re-measure, and require the answer to move by that
    much. Returns (dx_m, dy_m, info). A shift whose controls fail is reported but not applied, and
    `_quality` marks the chunk rejected.
    """
    from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked

    d = cfg.coarse_dec
    a, b = img[::d, ::d], ref[::d, ::d]
    cap_px = round(cfg.coarse_max_m / (cfg.ps * d))

    def measure(x, y):
        mx, my = np.isfinite(x), np.isfinite(y)
        xs = np.where(mx, (x - x[mx].mean()) / (x[mx].std() + 1e-9), 0)
        ys = np.where(my, (y - y[my].mean()) / (y[my].std() + 1e-9), 0)
        xcorr = cross_correlate_masked(xs, ys, mx, my, axes=(0, 1), mode="full", overlap_ratio=0.15)
        return _bounded_peak(xcorr, ys.shape, cap_px)

    base = measure(a, b)
    controls, ok = [], []
    for dy, dx in cfg.inject_px:
        got = measure(np.roll(np.roll(a, dy, 0), dx, 1), b) - base
        passed = bool(abs(got[0] + dy) < cfg.inject_tol_px and abs(got[1] + dx) < cfg.inject_tol_px)
        controls.append({"inject": [dy, dx], "change": [round(got[0], 1), round(got[1], 1)], "passed": passed})
        ok.append(passed)
    ps = cfg.ps * d
    dx_m, dy_m = float(base[1] * ps), float(-base[0] * ps)
    accepted = bool(all(ok))
    return (
        dx_m if accepted else 0.0,
        dy_m if accepted else 0.0,
        {
            "shift_px": [round(base[0], 1), round(base[1], 1)],
            "shift_km": [round(dx_m / 1e3, 2), round(dy_m / 1e3, 2)],
            "search_cap_px": cap_px,
            "controls": controls,
            "trustworthy": all(ok),
            "accepted": accepted,
        },
    )


def _rowcol(tp, transform, shape):
    """(row, col) of each tie point in an array on `transform`, clipped into `shape`."""
    row = ((tp.Y_MAP - transform.f) / transform.e).round().astype(int).clip(0, shape[0] - 1)
    col = ((tp.X_MAP - transform.c) / transform.a).round().astype(int).clip(0, shape[1] - 1)
    return row.to_numpy(), col.to_numpy()


def edge_dist_m(mask, ps):
    """Metres from every pixel to the nearest False in `mask` -- i.e. to the swath's own data edge.

    The target's finite footprint, not the AOI box: the hillshade reference covers the whole box,
    so the swath boundary is the only edge a match window can fall off. Shadow is finite data, not
    nodata, so it does not count as an edge (same distinction as the shadow gate).
    """
    return distance_transform_edt(mask) * ps


def _window_shadow(ref, row, col, win):
    """Shadowed fraction of each point's `win`-square reference window, via a summed-area table."""
    ny, nx = ref.shape
    ii = np.cumsum(np.cumsum(np.pad((ref <= 0).astype("float64"), ((1, 0), (1, 0))), 0), 1)
    h = win // 2
    r0, r1 = np.clip(row - h, 0, ny), np.clip(row + h, 0, ny)
    c0, c1 = np.clip(col - h, 0, nx), np.clip(col + h, 0, nx)
    area = ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]
    return area / np.maximum((r1 - r0) * (c1 - c0), 1)


def _coreg_pass(gref, gtgt, cfg, max_shift_px, grid_res, bad_tgt=None):
    """One COREG_LOCAL pass, optionally with the target's interior masked off so only the grid
    points near the swath edge are scored (`bad_tgt` True = skip)."""
    from arosics import COREG_LOCAL
    from geoarray import GeoArray

    kw = {}
    if bad_tgt is not None:
        kw["mask_baddata_tgt"] = GeoArray(bad_tgt.astype("uint8"), geotransform=gtgt.gt, projection=gtgt.projection)
    crl = COREG_LOCAL(
        gref,
        gtgt,
        grid_res=grid_res,
        window_size=(cfg.win, cfg.win),
        max_shift=max_shift_px,
        nodata=(gref.nodata, gtgt.nodata),
        q=True,
        progress=False,
        min_reliability=cfg.min_reliability,
        tieP_filter_level=3,
        # arosics defaults `CPUs` to `multiprocessing.cpu_count()`, which reports the machine's
        # cores and not the cgroup's -- inside a Slurm allocation that is the whole node (192 on
        # Nibi) no matter how few `--cpus-per-task` were granted, so the default oversubscribes by
        # ~50x and thrashes. `sched_getaffinity` is the count this process may actually use.
        CPUs=len(os.sched_getaffinity(0)),
        **kw,
    )
    tp = crl.CoRegPoints_table
    kept = tp[(gtgt.nodata != tp.ABS_SHIFT) & tp.X_SHIFT_PX.notna() & (cfg.min_reliability <= tp.RELIABILITY)]
    return kept, len(tp)  # unfiltered length is the denominator of `match_frac`


def tie_points(ref, tgt, transform, cfg, max_shift_px):
    """One AROSICS COREG_LOCAL pass on two arrays already sharing a grid, filtered to usable points.

    Both are standardised (and optionally reduced to gradient magnitude) before matching; nodata is
    a sentinel because arosics does not take NaN.

    `cfg.edge_dense_m` adds a second, denser pass confined to the swath margin, and
    `cfg.edge_reject_m` / `cfg.shadow_max` drop the two classes of confidently-wrong point that
    `min_reliability` cannot see (see :class:`GeorefConfig`). All three are off by default.
    """
    from geoarray import GeoArray

    nodata = -9999.0
    rm, tm = np.isfinite(ref), np.isfinite(tgt)
    r, t = ref, tgt
    if cfg.grad:
        r, rm = grad_mag(r, rm, cfg.grad_sigma)
        t, tm = grad_mag(t, tm, cfg.grad_sigma)
    gt = (transform.c, transform.a, 0.0, transform.f, 0.0, transform.e)
    proj = stereo_crs(cfg.pole).to_wkt()
    mk = lambda arr, m: GeoArray(_standardize(arr, m, nodata), geotransform=gt, projection=proj, nodata=nodata)
    gref, gtgt = mk(r, rm), mk(t, tm)

    tp, n_cand = _coreg_pass(gref, gtgt, cfg, max_shift_px, cfg.grid_res)
    dist = edge_dist_m(tm, abs(transform.a))
    if cfg.edge_dense_m and cfg.grid_res > 1:
        # Split rather than concatenate: the half-spacing grid is a superset of the coarse one
        # (both anchored at the array origin), so taking the coarse pass outside the margin and
        # the dense pass inside it covers every point exactly once.
        margin = dist <= cfg.edge_dense_m
        dense, n_dense = _coreg_pass(gref, gtgt, cfg, max_shift_px, cfg.grid_res // 2, bad_tgt=~margin)
        keep = [tp[~margin[_rowcol(tp, transform, tm.shape)]], dense[margin[_rowcol(dense, transform, tm.shape)]]]
        tp = pd.concat(keep, ignore_index=True)
        n_cand += n_dense

    row, col = _rowcol(tp, transform, tm.shape)
    if cfg.edge_reject_m:
        tp = tp[dist[row, col] >= cfg.edge_reject_m]
        row, col = _rowcol(tp, transform, tm.shape)
    if cfg.shadow_max < 1.0:
        tp = tp[_window_shadow(ref, row, col, cfg.win) <= cfg.shadow_max]
    return tp, n_cand


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


def _local_plane(pts, disp, tree, k):
    """Off-support fallback: the plane fitted to the `k` tie points nearest each query point.

    Returns f(q) -> (dx, dy) in metres, the plane evaluated *at* q and clipped component-wise to
    the range of those k displacements, so a one-sided margin cannot extrapolate without bound.
    """
    k = min(k, len(pts))

    def far(q):
        _, idx = tree.query(q, k=k)
        idx = idx.reshape(len(q), k)
        # Centre the neighbours on the query so the fitted intercept *is* the value at q.
        a = np.concatenate([np.ones((len(q), k, 1)), pts[idx] - np.asarray(q)[:, None, :]], axis=-1)
        d = disp[idx]  # (n, k, 2)
        coef = np.linalg.pinv(a) @ d  # pinv, not solve: k collinear neighbours are rank-deficient
        return np.clip(coef[:, 0, :], d.min(axis=1), d.max(axis=1))

    return far


def displacement_field(tp, cfg):
    """
    Tie-point displacement field q -> (dx, dy) metres, decaying to the bulk median off-support.

    The GCP lattice spans the whole strip but tie points exist only where it crosses the AOI. A TPS
    diverges outside its control-point hull, so beyond the support the field blends to the bulk
    median with a Gaussian in distance-to-nearest.

    A low-texture chunk can leave too few, or too degenerately placed, tie points to fit a spline
    at all; those iterations fall back to the bulk median shift and say so in `info["degenerate"]`.
    They cannot reach `converged`, which needs `cfg.min_kept` points.
    """
    pts = tp[["X_MAP", "Y_MAP"]].to_numpy(float)
    disp = np.c_[tp.X_SHIFT_PX.to_numpy(float) * cfg.ps, -tp.Y_SHIFT_PX.to_numpy(float) * cfg.ps]
    bulk = np.median(disp, axis=0)
    info = {"n": len(pts), "bulk_m": [round(float(bulk[0]), 1), round(float(bulk[1]), 1)]}

    smooth = cfg.smooth_fixed
    if cfg.smooth_cv:
        t0 = time.time()
        smooth, cv_rms, base_rms = choose_smoothing(pts, disp)
        info |= {
            "smoothing": smooth,
            "cv_rms_m": round(cv_rms, 1),
            "bulk_median_rms_m": round(base_rms, 1),
            "beats_bulk": bool(cv_rms < base_rms),
            "smooth_s": round(time.time() - t0, 2),  # 125 dense O(n^3) fits; grows fast with `n`
        }
    try:
        rbf = RBFInterpolator(pts, disp, kernel="thin_plate_spline", smoothing=smooth)
    except (np.linalg.LinAlgError, ValueError) as e:
        # The TPS trend term (1, x, y) needs 3 points spanning a plane: scipy raises ValueError
        # below 3 and LinAlgError when they are collinear. Return bulk shift where spline fails.
        info["degenerate"] = f"tps_unfittable: {e}".rstrip(". ")
        return (lambda q: np.repeat(bulk[None], len(q), axis=0)), info
    tree = cKDTree(pts)
    far = _local_plane(pts, disp, tree, cfg.edge_fit_k) if cfg.edge_fit_k else None
    info["off_support"] = f"local_plane_k{cfg.edge_fit_k}" if far else "bulk_median"

    def fieldfn(q):
        w = np.exp(-((tree.query(q)[0] / cfg.decay_m) ** 2))[:, None]
        out = np.repeat(bulk[None], len(q), axis=0) if far is None else far(q)
        return w * rbf(q) + (1 - w) * out

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


POLE_LAT_BAND: dict[str, tuple[float, float]] = {
    "south": (-90.0, -80.0),
    "north": (80.0, 90.0),
    "equatorial": (-60.0, 60.0),
}
for _pole in POLE_LAT_BAND:
    _override = os.environ.get(f"IIRSPY_POLE_LAT_BAND_{_pole.upper()}")
    if _override:
        # Env var, not a runtime dict assignment, so a subprocess that re-imports this module
        # (register()'s arosics-conda path) also picks up the override.
        _lo, _hi = (float(v) for v in _override.split(","))
        POLE_LAT_BAND[_pole] = (_lo, _hi)


def _lattice_row_bounds(ny: int, cfg) -> tuple[int, int]:
    """Inclusive first and last cube row the GCP lattice spans.

    The whole cube unless `gcp_rows` names a sub-range and `gcp_row_margin` is non-negative, in
    which case the lattice is clipped to those rows plus that margin either side.

    >>> from types import SimpleNamespace
    >>> _lattice_row_bounds(14400, SimpleNamespace(gcp_rows=(4350, 6100), gcp_row_margin=-1))
    (0, 14399)
    >>> _lattice_row_bounds(14400, SimpleNamespace(gcp_rows=(4350, 6100), gcp_row_margin=0))
    (4350, 6100)
    >>> _lattice_row_bounds(14400, SimpleNamespace(gcp_rows=(4350, 6100), gcp_row_margin=250))
    (4100, 6350)
    >>> _lattice_row_bounds(14400, SimpleNamespace(gcp_rows=(4350, 6100), gcp_row_margin=1000))
    (3350, 7100)
    >>> _lattice_row_bounds(14400, SimpleNamespace(gcp_rows=None, gcp_row_margin=250))
    (0, 14399)
    >>> _lattice_row_bounds(500, SimpleNamespace(gcp_rows=(100, 400), gcp_row_margin=9999))
    (0, 499)
    """
    if cfg.gcp_rows is None or cfg.gcp_row_margin < 0:
        return 0, ny - 1
    row0, row1 = cfg.gcp_rows
    return max(0, row0 - cfg.gcp_row_margin), min(ny - 1, row1 + cfg.gcp_row_margin)


def gcp_lattice(fgeom, ny, nx, cfg):
    """Camera (row, col) -> map (x, y) lattice from the supplied geometry csv, uncorrected.

    The lat band must reproduce exactly the crop `run_l1_polar.run_one` applied when it built the
    cube (`POLE_LAT_BAND` mirrors its per-pole `extent`), or GCP `row` stops lining up with the
    cube's own rows.
    """
    lat_band = cfg.lat_band or POLE_LAT_BAND[cfg.pole]
    extent = (-180, 180, *lat_band)
    # `lat_band` must be the crop the cube was built with to align GCPs with original rows
    _, xyext = utils.parse_geom(fgeom, extent)
    scan0, n_geom_rows = xyext[2], xyext[3] - xyext[2] + 1
    if n_geom_rows < ny:
        raise ValueError(
            f"lat_band {lat_band} spans {n_geom_rows} geometry rows but the cube has {ny} -- "
            "pass the crop the cube was built with as GeorefConfig.lat_band"
        )
    r0, r1 = _lattice_row_bounds(ny, cfg)
    # The bounding rows must themselves be GCP rows: arange stops short of r1, leaving the last
    # ~1 km beyond every GCP where GDAL's TPS extrapolates freely.
    rows = np.unique(np.r_[np.arange(r0, r1 + 1, cfg.row_step), r1])
    cols = np.unique(np.linspace(0, nx - 1, cfg.ncol).astype(int))
    # Cube row r is geometry Scan scan0 + r.
    glon, glat, _ = utils.geom2grid(fgeom, extent, xs=cols, ys=scan0 + rows)
    jj, ii = np.meshgrid(rows, cols, indexing="ij")
    x, y = to_stereo(cfg.pole).transform(glon, glat)
    return jj, ii, x, y


def _to_gcps(jj, ii, x, y):
    return [
        GroundControlPoint(row=float(jj[a, b]), col=float(ii[a, b]), x=float(x[a, b]), y=float(y[a, b]))
        for a in range(jj.shape[0])
        for b in range(jj.shape[1])
    ]


def window_of(cfg, window=None):
    """(shape, transform) of the AOI grid, or of `window` = (r0, r1, c0, c1) cut out of it."""
    xs, ys, tr, _ = grid_of(cfg)
    if window is None:
        return (len(ys), len(xs)), tr
    r0, r1, c0, c1 = window
    return (r1 - r0, c1 - c0), tr * rasterio.Affine.translation(c0, r0)


def unify_nodata(cube):
    """Stack a cube with its own hole indicator, both under one shared nodata mask.

    Returns a (2 * nband, y, x) array: the data bands, then a 0/1 plane per band marking where
    that band had nodata. Warp it in one :func:`project` call and drop every output pixel whose
    indicator came back above zero -- exactly the set whose resampling footprint touched a hole in
    *that* band. `unstack_nodata` does that.

    Why not just warp the cube. GDAL's multi-band warp does not mask bands independently: a band
    with a smaller valid footprint is pulled down toward the others (measured on an L2 cube, band
    27 beside band 251: 165,783 px kept against 171,428 alone -- 3.3% lost, values bit-identical).
    Band-at-a-time is correct but pays the ~1700-point TPS solve 256 times.

    So remove the disagreement rather than the sharing. Nodata becomes the pixels where *no* band
    has data, which every band then shares, and a per-band hole inside that footprint becomes 0 in
    the data half and 1 in the indicator half. A fill value alone cannot mark those holes:
    resampling blends it into a continuum, so any cut-off leaves surviving contamination of its own
    magnitude (measured: -999.185 survived a -1000 cut-off).

    Stacked rather than warped separately because GDAL transforms each destination pixel once per
    *call*, not once per band -- the per-band cost is only the resampling (measured: 5 bands cost
    94.6 s one at a time against 19.6 s in one call). So carrying the indicator alongside is
    nearly free, where a second call would pay the whole transform again.
    """
    nan = np.isnan(cube)
    dead = nan.all(axis=0)
    out = np.empty((2 * len(cube), *cube.shape[1:]), "float32")
    out[: len(cube)] = np.where(nan, np.float32(0), cube)
    out[len(cube) :] = nan
    out[:, dead] = np.nan
    return out


def unstack_nodata(warped, band):
    """One band out of a warped :func:`unify_nodata` stack, its touched-a-hole pixels set NaN."""
    n = len(warped) // 2
    return np.where(warped[n + band] > 0, np.nan, warped[band])


def data_window(arr, pad=8):
    """(r0, r1, c0, c1) bounding the finite pixels of a projected band, padded by `pad`."""
    r, c = np.where(np.isfinite(arr))
    h, w = arr.shape[-2:]
    return (
        max(int(r.min()) - pad, 0),
        min(int(r.max()) + 1 + pad, h),
        max(int(c.min()) - pad, 0),
        min(int(c.max()) + 1 + pad, w),
    )


def project(band, gcps, cfg, resampling=Resampling.bilinear, window=None):
    """One reproject of a camera-space band -- or a whole (band, y, x) cube -- onto the AOI grid.

    Pass the cube, not one band at a time: GDAL solves the warp once per call.
    - `window` crops the output to (r0, r1, c0, c1) of the AOI grid (down from several GB strip).
    - SRC_METHOD=GCP_TPS: thin-plate-spline interpolation.
    - tolerance=0 is the exact transformer (rasterio's default 0.125 replaces TPS with a
    polynomial fitted within an eighth of a pixel which added 5 m of noise at 40 m/px)
    """
    shape, tr = window_of(cfg, window)
    shape = np.shape(band)[:-2] + shape
    # Every band must stay in one call to avoid solving the warp repeatedly, but full AOI cube is
    # multiple GB, so spill to disk if it exceeds MAX_RAM_BYTES.
    out: np.ndarray
    if np.prod(shape) * 4 > MAX_RAM_BYTES:
        tmp = tempfile.NamedTemporaryFile(suffix=".f32", delete=False)  # noqa: SIM115
        out = np.memmap(tmp.name, dtype="float32", mode="w+", shape=shape)
        out[:] = np.nan
        Path(tmp.name).unlink()  # unlinked but held open: the pages die with the array
    else:
        out = np.full(shape, np.nan, "float32")
    reproject(
        source=band,
        destination=out,
        src_crs=stereo_crs(cfg.pole),
        gcps=gcps,
        dst_transform=tr,
        dst_crs=stereo_crs(cfg.pole),
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        tolerance=0.0,
        num_threads=len(os.sched_getaffinity(0)),  # parallize warp kernel over available cores
        SRC_METHOD="GCP_TPS",
    )
    return out


def save_grid(fout, arr, cfg, nodata=np.nan):
    """Write a single AOI-grid band (hillshade, projected product) as a georeferenced float32 tif."""
    _, _, tr, _ = grid_of(cfg)
    with rasterio.open(
        fout,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=stereo_crs(cfg.pole),
        transform=tr,
        nodata=nodata,
        compress="LZW",
    ) as dst:
        dst.write(np.asarray(arr, dtype="float32"), 1)
    return fout


def _has_arosics():
    """True only for a real, importable `arosics` -- not a same-named directory on `sys.path`.

    A directory called `arosics/` with no `__init__.py` anywhere on the path makes `find_spec`
    succeed as a *namespace* package, whose `loader` is None and which of course has no
    `COREG_LOCAL`. `local-workspace/arosics/` in this repo is exactly that, so the plain
    `find_spec(...) is None` test used to report the package present and send the solve down the
    in-process branch, dying with "cannot import name 'COREG_LOCAL' from 'arosics' (unknown
    location)" instead of subprocessing to the conda environment that actually has it.
    """
    spec = importlib.util.find_spec("arosics")
    return spec is not None and spec.loader is not None


def arosics_python():
    """Interpreter of the arosics conda environment. Raises if that environment is missing."""
    exe = os.environ.get("IIRSPY_AROSICS_PYTHON")
    if exe and Path(exe).exists():
        return Path(exe)
    conda = shutil.which("conda") or shutil.which("mamba") or os.environ.get("CONDA_EXE", "")
    if conda:
        out = subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True)  # noqa: S603
        for e in json.loads(out.stdout or "{}").get("envs", []):
            if Path(e).name == AROSICS_ENV and Path(e, "bin", "python").exists():
                return Path(e, "bin", "python")
    raise RuntimeError(
        f"co-registration needs the '{AROSICS_ENV}' conda environment: arosics needs osgeo.gdal, "
        "which PyPI does not ship. Create it from the repository root with\n"
        "    mamba env create -f arosics-environment.yml\n"
        "or set IIRSPY_AROSICS_PYTHON to an interpreter that has arosics."
    )


def _register_subprocess(ftif, fgeom, fspm, cfg, kernels, reference, verbose):
    """Run the solve through iirspy.coreg in the arosics environment; rebuild it from the json."""
    py = arosics_python()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        ref = reference
        if ref is not None and not isinstance(ref, str | Path):
            ref = save_grid(Path(tmp, "reference.tif"), ref, cfg)
        job = Path(tmp, "job.json")
        out = Path(tmp, "gcps.json")
        job.write_text(
            json.dumps({
                "ftif": str(ftif),
                "fgeom": str(fgeom),
                "fspm": str(fspm),
                "cfg": asdict(cfg),
                "kernels": [str(k) for k in kernels or []],
                "reference": None if ref is None else str(ref),
                "verbose": verbose,
                "out": str(out),
            })
        )
        subprocess.run([str(py), "-m", "iirspy.coreg", str(job)], check=True, cwd=str(root))  # noqa: S603
        d = json.loads(out.read_text())
    xs, ys, tr, _ = grid_of(cfg)
    return Registration(
        gcps=[GroundControlPoint(row=r, col=c, x=x, y=y) for r, c, x, y in d["gcps"]],
        crs=stereo_crs(cfg.pole),
        transform=tr,
        shape=(len(ys), len(xs)),
        stats=d["stats"],
    )


def _stop_iterating(k: int, p95: float, prev_p95: float, cfg: GeorefConfig, n_kept: int) -> tuple[bool, str | None]:
    """Whether iteration `k`'s residual is good enough to stop, and why.

    Two independent exits, both gated by `min_iter` (0 = no floor): `p95_stop_m` is a tight
    "this solve is already excellent" fast exit; `p95_plateau_frac` is the default day-to-day
    exit -- stop once p95 is no longer improving by more than that fraction of the previous
    iteration, regardless of its absolute value.

    Neither exit fires on fewer than `min_kept` tie points: a p95 only describes the points it was
    measured on, so a handful of them can show an excellent residual on a mis-locked chunk.

    >>> from dataclasses import replace
    >>> cfg = replace(GeorefConfig(), p95_stop_m=18.0, p95_plateau_frac=0.02, min_iter=0)
    >>> _stop_iterating(1, 15.0, float("inf"), cfg, 800)  # under the floor -- fast exit
    (True, 'threshold')
    >>> _stop_iterating(2, 19.0, 19.1, cfg, 800)  # barely moved, still above the floor
    (True, 'plateau')
    >>> _stop_iterating(2, 19.0, 25.0, cfg, 800)  # still improving fast -- keep going
    (False, None)
    >>> _stop_iterating(1, 10.0, float("inf"), replace(cfg, min_iter=2), 800)  # min_iter floor
    (False, None)
    >>> _stop_iterating(1, 15.0, float("inf"), cfg, 4)  # excellent p95, but over 4 points
    (False, None)
    """
    if k < cfg.min_iter or n_kept < cfg.min_kept:
        return False, None
    if p95 < cfg.p95_stop_m:
        return True, "threshold"
    if cfg.p95_plateau_frac and (prev_p95 - p95) < cfg.p95_plateau_frac * prev_p95:
        return True, "plateau"
    return False, None


def _quality(iters, cinfo, cfg):
    """Whether this solve registered at all, judged on tie points rather than on `corr`.

    `corr` measures the projected band against the hillshade, which albedo dominates wherever the
    terrain is mare/highland, so it cannot separate a registered chunk from a mis-locked one. The
    fraction of tie-point candidates kept can.

    >>> from dataclasses import replace
    >>> cfg = replace(GeorefConfig(), min_match_frac=0.10, min_kept=50)
    >>> ok = {"accepted": True}
    >>> _quality([{"n_kept": 1080, "match_frac": 0.68}], ok, cfg)["rejected"]
    False
    >>> _quality([{"n_kept": 12, "match_frac": 0.009}], ok, cfg)["reason"]
    'match_frac 0.009 < 0.1'
    >>> _quality([{"n_kept": 1080, "match_frac": 0.68}], {"accepted": False}, cfg)["reason"]
    'coarse shift refused'
    >>> _quality([], ok, cfg)["reason"]
    'no iterations'
    """
    last = iters[-1] if iters else {}
    frac, n_kept = last.get("match_frac"), last.get("n_kept", 0)
    if not iters:
        reason = "no iterations"
    elif not cinfo.get("accepted", True):
        reason = "coarse shift refused"
    elif frac is not None and frac < cfg.min_match_frac:
        reason = f"match_frac {frac} < {cfg.min_match_frac}"
    elif n_kept < cfg.min_kept:
        reason = f"n_kept {n_kept} < {cfg.min_kept}"
    else:
        reason = None
    return {
        "match_frac": frac,
        "n_kept": n_kept,
        "coarse_accepted": cinfo.get("accepted"),
        "rejected": reason is not None,
        "reason": reason,
    }


def _best_iterate(best_p95, best_xy, last_p95, x, y):
    """The best-p95 lattice seen this solve if it beats the last iterate, else `(x, y)` as-is."""
    if best_xy is not None and best_p95 < last_p95:
        return (*best_xy, "best_iterate_fallback")
    return x, y, None


def register(ftif, fgeom, fspm, cfg, kernels=None, reference=None, verbose=False):
    """
    Solve a scene's registration against LOLA. Returns a :class:`Registration`.

    The correction is composed into the GCP *targets*, one iteration at a time -- each tie-point
    field maps a current map position to the correction needed there, so applying field k at the
    position field k-1 produced is the right order, and the image is only ever resampled once.

    `reference` is the hillshade to match against, as an array or the path to one; without it the
    scene's own is rendered. When arosics is not importable the solve runs in the arosics conda
    environment through :mod:`iirspy.coreg`.
    """
    if not _has_arosics():
        return _register_subprocess(ftif, fgeom, fspm, cfg, kernels, reference, verbose)

    ref = reference
    info = {}
    if isinstance(ref, str | Path):
        with rasterio.open(ref) as src:
            a = src.read(1, masked=True).filled(np.nan).astype("float32")
        ref = np.where(a == -9999.0, np.nan, a)
    if ref is None:
        ref, hs_info = render_reference(fgeom, fspm, cfg, kernels)
        info["reference"] = hs_info

    t0 = time.time()
    band = read_band(ftif, cfg.band)
    ny, nx = band.shape
    jj, ii, x, y = gcp_lattice(fgeom, ny, nx, cfg)
    _, _, tr, _ = grid_of(cfg)
    info["setup_s"] = round(time.time() - t0, 2)

    # base: supplied geometry alone -- what the coarse search is measured on
    t0 = time.time()
    img = project(band, _to_gcps(jj, ii, x, y), cfg)
    info["project_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    dx, dy, cinfo = coarse_shift(img, ref, cfg)
    info["coarse"] = cinfo
    info["coarse_s"] = round(time.time() - t0, 2)
    x, y = x + dx, y + dy

    iters, converged, converged_reason, prev_p95 = [], False, None, np.inf
    best_p95, best_xy = np.inf, None
    for k in range(1, cfg.niter + 1):
        t_iter = time.time()
        img = project(band, _to_gcps(jj, ii, x, y), cfg)
        project_s = time.time() - t_iter
        tp, n_cand = tie_points(ref, img, tr, cfg, cfg.max_shift_for(k))
        tp = _reject_shift_cap(tp, cfg.max_shift_for(k))
        n_mad, mad_lim = 0, None
        if cfg.mad_from_iter and k >= cfg.mad_from_iter:
            tp, n_mad, mad_lim = _reject_mad(tp, cfg.mad_k, cfg.ps)
        if not len(tp):
            iters.append({
                "iter": k,
                "n_kept": 0,
                "n_candidates": n_cand,
                "note": "no tie points",
                "iter_s": round(time.time() - t_iter, 2),
            })
            break

        d = np.hypot(tp.Y_SHIFT_PX, tp.X_SHIFT_PX).to_numpy(float) * cfg.ps
        p95 = float(np.percentile(d, 95))
        fieldfn, finfo = displacement_field(tp, cfg)
        s = fieldfn(np.c_[x.ravel(), y.ravel()]).reshape(*x.shape, 2)
        x, y = x + s[..., 0], y + s[..., 1]
        # Same `min_kept` floor as `_stop_iterating`: a p95 over a handful of points can be the
        # smallest of the run without the lattice being any good, and this one is handed back.
        if len(tp) >= cfg.min_kept and p95 < best_p95:
            best_p95, best_xy = p95, (x.copy(), y.copy())
        iters.append({
            "iter": k,
            "max_shift_px": cfg.max_shift_for(k),
            "n_kept": len(tp),
            "n_candidates": n_cand,
            "match_frac": round(len(tp) / n_cand, 3) if n_cand else None,
            "n_mad_rejected": n_mad,
            "mad_limit_m": mad_lim,
            "shift_m": {"median": round(float(np.median(d)), 1), "p95": round(p95, 1)},
            "applied_med_m": round(float(np.median(np.hypot(s[..., 0], s[..., 1]))), 1),
            "field": finfo,
            # The matcher's own share is iter_s - project_s - field.smooth_s.
            "project_s": round(project_s, 2),
            "iter_s": round(time.time() - t_iter, 2),
        })
        if verbose:
            print(iters[-1], flush=True)
        # p95 of iteration k is the residual measured on product k-1; under a pixel means the
        # product this iteration just produced is converged.
        converged, converged_reason = _stop_iterating(k, p95, prev_p95, cfg, len(tp))
        if converged:
            break
        prev_p95 = p95
    else:
        # Exhausted `niter` without converging -- use the best iterate seen, not blindly the
        # last, in case a late MAD cull or reweight made p95 worse right at the end.
        x, y, converged_reason = _best_iterate(best_p95, best_xy, p95, x, y)

    xs, ys, tr, _ = grid_of(cfg)
    return Registration(
        gcps=_to_gcps(jj, ii, x, y),
        crs=stereo_crs(cfg.pole),
        transform=tr,
        shape=(len(ys), len(xs)),
        stats=info
        | {
            "iters": iters,
            "converged": converged,
            "converged_reason": converged_reason,
            "quality": _quality(iters, cinfo, cfg),
            "band": cfg.band,
        },
    )


def warp(cube, reg, cfg, resampling=Resampling.bilinear, window=None):
    """
    Apply a :class:`Registration` to a cube or single band. Returns a DataArray on the AOI grid.

    `cube` may be a path to an IIRS product or a (band, y, x) DataArray in camera space. Every
    band is resampled once, straight from camera space. A strip fills ~10% of the AOI, so pass
    `window` (see :func:`data_window`) unless the whole 160 km box is wanted -- a 256-band cube on
    the full grid is tens of GB.
    """
    if isinstance(cube, str | Path):
        da = xr.open_dataarray(cube, engine="rasterio")
        from iirspy.iirs import _band_numbers

        da = da.assign_coords(band=_band_numbers(da))
    else:
        da = cube
    if "band" not in da.dims:
        da = da.expand_dims("band")

    xs, ys, _, _ = grid_of(cfg)
    (ny, nx), tr = window_of(cfg, window)
    r0, c0 = (window[0], window[2]) if window else (0, 0)
    ys, xs = ys[r0 : r0 + ny], xs[c0 : c0 + nx]
    out = project(da.values.astype("float32"), reg.gcps, cfg, resampling, window)
    res = xr.DataArray(
        out,
        dims=("band", "y", "x"),
        coords={"band": da.band.values, "y": ys, "x": xs},
        attrs=dict(da.attrs) | {"georef_converged": str(reg.converged)},
    )
    if "wl" in da.coords:
        res = res.assign_coords(wl=("band", da.wl.values))
    return res.rio.write_crs(reg.crs).rio.write_transform(tr)
