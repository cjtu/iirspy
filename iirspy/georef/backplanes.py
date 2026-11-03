"""Camera-space backplanes (M3-style LOC/OBS/QA) for one solve group, absolute-scan-row keyed."""

import json
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay, cKDTree

from iirspy import io
from iirspy.georef import ancillary
from iirspy.georef import chunks as ck
from iirspy.georef.georef import (
    MOON_RADIUS_M,
    GeorefConfig,
    enu,
    load_lola_elev,
    lonlat_to_xyz,
    read_band,
    sc_positions_km,
    stereo_crs,
    xy_to_lonlat,
)
from iirspy.instrument import PAN_BANDS
from iirspy.l1.empirical import (
    MIN_ROWS,
    SHADOW_SNR,
    broadband_snr,
    detect_dark_rows,
    longest_run,
    refine_dark_block,
)

IIRS_GSD_M = GeorefConfig().iirs_gsd_m
DEM_MARGIN_M = 2_000.0  # bbox pad for a DEM window read around one row block's ground footprint
AU_M = 1.495978707e11
GEOM_SHADOW_LIT = 0.5  # lit fraction below this: sun centre below the local LOLA horizon


def chunk_overlaps(chunks: list[dict]) -> list[tuple[int, int]]:
    """Crop-relative (row0, row1) overlaps between consecutive chunks of a solve summary's `chunks` plan."""
    rows = sorted((int(c["row0"]), int(c["row1"])) for c in chunks)
    return [(max(a0, b0), min(a1, b1)) for (a0, a1), (b0, b1) in pairwise(rows) if max(a0, b0) < min(a1, b1)]


# ------------------------------------------------------------------------------------------
# LOC: lon/lat/radius at every camera pixel
# ------------------------------------------------------------------------------------------
def _dem_band(group: str, lat: float, dem_roots: tuple[Path, ...]) -> dict:
    """The `chunks.bands()` entry covering `lat` within `group`; clamped to the nearest band's
    `lat_range` if `lat` falls outside every one (LOC extrapolating past the solved edge)."""
    resolved = ck.bands(dem_roots)
    cands = [b for b in resolved.values() if b["group"] == group]
    inside = [b for b in cands if b["lat_range"][0] <= lat <= b["lat_range"][1]]
    if inside:
        return inside[0]
    return min(cands, key=lambda b: min(abs(lat - b["lat_range"][0]), abs(lat - b["lat_range"][1])))


def _sample_radius(
    x: np.ndarray, y: np.ndarray, lat: np.ndarray, group: str, dem_roots: tuple[Path, ...]
) -> np.ndarray:
    """LOLA radius [m from Moon centre] at every (x, y) in `group`'s stereo CRS.

    DEM band picked once from this block's median latitude, not per pixel: a block only crosses
    a band boundary near the transition itself, and `_dem_band` already falls back to the nearest
    band there, so the height error at that seam is bounded by the DEM tiers' own overlap.
    """
    band = _dem_band(group, float(np.nanmedian(lat)), dem_roots)
    with rasterio.open(band["dem_near"]) as src:
        b, px = src.bounds, abs(src.res[0])
    # Clamped into the DEM so a block wholly past its edge (an equatorial strip's first rows south of
    # the +-60 deg DEM) reads a 1-px edge strip; `mode="nearest"` then extends it, as it already does
    # for a block that only straddles the edge.
    bounds = (
        float(np.clip(np.nanmin(x) - DEM_MARGIN_M, b.left, b.right - px)),
        float(np.clip(np.nanmin(y) - DEM_MARGIN_M, b.bottom, b.top - px)),
        float(np.clip(np.nanmax(x) + DEM_MARGIN_M, b.left + px, b.right)),
        float(np.clip(np.nanmax(y) + DEM_MARGIN_M, b.bottom + px, b.top)),
    )
    elev, tr, _ = load_lola_elev(band["dem_near"], bounds=bounds)
    rr = (y - tr.f) / tr.e
    cc = (x - tr.c) / tr.a
    h = map_coordinates(elev, [rr, cc], order=1, mode="nearest")
    return np.asarray(MOON_RADIUS_M + h)


def group_loc_core(
    gcps: dict,
    group: str,
    dem_roots: tuple[Path, ...],
    nrow: int | None = None,
    ncol: int | None = None,
    row_block: int = 500,
):
    """(lon, lat, radius) at every camera pixel centre of one group's GCP lattice, crop-relative rows.

    x/y are an interpolating bicubic spline on the merged (row, col) lattice, which the solve always
    leaves complete and rectilinear. Not GDAL's TPS: on these ~5-22k point lattices its solve is
    ill-conditioned and returns pockets of 1e20+ m near some GCPs, and costs O(n_gcps) per pixel.
    Pixel centres sit at (r + 0.5, c + 0.5) in GDAL's corner-origin GCP convention, the same one
    `georef.project`/`scene_glt` warp with. Rows/cols off the lattice (the solve can leave the first
    few hundred rows without GCPs, and `nrow`/`ncol` may exceed the lattice extent) extrapolate
    linearly from the lattice edge -- a cubic end polynomial runs away by ~100 km over 450 rows.
    """
    ur = np.unique([r for r, _c in gcps])
    uc = np.unique([c for _r, c in gcps])
    if len(ur) * len(uc) != len(gcps):
        raise ValueError(f"GCP lattice is incomplete: {len(gcps)} points on {len(ur)} rows x {len(uc)} cols")
    ir = {v: i for i, v in enumerate(ur)}
    ic = {v: i for i, v in enumerate(uc)}
    gx = np.empty((len(ur), len(uc)))
    gy = np.empty_like(gx)
    for (r, c), (x, y) in gcps.items():
        gx[ir[r], ic[c]], gy[ir[r], ic[c]] = x, y
    nrow = int(ur[-1]) + 1 if nrow is None else nrow
    ncol = int(uc[-1]) + 1 if ncol is None else ncol
    kx, ky = min(3, len(ur) - 1), min(3, len(uc) - 1)
    splines = [RectBivariateSpline(ur, uc, g, kx=kx, ky=ky, s=0) for g in (gx, gy)]
    cc = np.arange(ncol) + 0.5
    ce = np.clip(cc, uc[0], uc[-1])

    def at(rr):
        re = np.clip(rr, ur[0], ur[-1])
        return [
            sp(re, ce) + sp(re, ce, dx=1) * (rr - re)[:, None] + sp(re, ce, dy=1) * (cc - ce)[None] for sp in splines
        ]

    lon = np.full((nrow, ncol), np.nan)
    lat = np.full((nrow, ncol), np.nan)
    radius = np.full((nrow, ncol), np.nan)
    for r0 in range(0, nrow, row_block):
        r1 = min(r0 + row_block, nrow)
        x, y = at(np.arange(r0, r1) + 0.5)
        lo, la = xy_to_lonlat(x, y, pole=group)
        lon[r0:r1], lat[r0:r1] = lo, la
        radius[r0:r1] = _sample_radius(x, y, la, group, dem_roots)
    return lon, lat, radius


def _tie_point_residual(fits: list[dict]) -> dict[str, float] | None:
    """Aggregate p50/p95 tie-point residual [m] across chunk `fits` (a solve summary's `per_chunk_fit`),
    or `None` if none carries one (an older solve, or a synthetic test)."""
    med, p95 = [], []
    for fit in fits:
        iters = fit.get("stats", {}).get("iters", [])
        if iters and "shift_m" in iters[-1]:
            med.append(iters[-1]["shift_m"]["median"])
            p95.append(iters[-1]["shift_m"]["p95"])
    if not med:
        return None
    return {"p50_m": float(np.median(med)), "p95_m": float(np.max(p95))}


def _tags_to_attrs(tags: dict) -> dict:
    """`tags` ready for `da.attrs`: dict/list values json.dumps'ed so `io.write_camera_product`/
    `write_envi_hdr` (which only round-trip scalar attrs) both carry every tag."""
    return {k: (json.dumps(v) if isinstance(v, dict | list) else v) for k, v in tags.items()}


def _backplane_da(bands: list, names: list[str], units: list[str], row0: int, tags: dict) -> xr.DataArray:
    """(band, y, x) DataArray for a camera-space backplane: `bands` stacked in order, `names`/
    `units` as non-dimension coords on `band` (`io.write_camera_product`'s band-label branch for a product
    with no `wl`), `y`/`x` absolute pixel-centre coords (`row0`=absolute scan row of row 0),
    matching the convention an L1 cube's own `y`/`x` use (see `IIRSData._cube_scan0`)."""
    arr = np.stack(bands)
    ny, nx = arr.shape[1:]
    da = xr.DataArray(
        arr,
        dims=("band", "y", "x"),
        coords={
            "band": np.arange(1, len(bands) + 1),
            "band_name": ("band", names),
            "units": ("band", units),
            "y": row0 + np.arange(ny) + 0.5,
            "x": np.arange(nx) + 0.5,
        },
    )
    da.attrs = _tags_to_attrs(tags)
    return da


def write_loc(
    fout: Path,
    loc: dict,
    sid: str,
    compress: str = "ZSTD",
    predictor: int = 3,
    dtype: str = "float32",
    row0: int = 0,
    fits: list[dict] | None = None,
    dem_roots: tuple[Path, ...] = (),
    tags: dict | None = None,
) -> Path:
    """Write `<sid>_loc.tif`/`.img`: 3 bands lon [deg, 0-360 E], lat [deg, planetocentric], radius
    [m from Moon centre], matching the M3 LOC band order/units. Full strip rows x 250, camera
    space: no CRS or transform. `dtype` defaults f32 (ULP <=0.9 m lon, 0.23 m lat, 0.125 m radius,
    well under the ~80 m pixel this backs); computed in f64 and cast only here -- note the shared
    GeoTIFF writer (`io.write_camera_product`) always writes float dtypes as float32, so `dtype="float64"`
    only affects the pre-write cast, not the file actually written. `compress`/`predictor`/`dtype`
    are exposed, not frozen. `row0` is this array's absolute scan row 0 (0 for the current
    full-strip callers). `fits` are the solve's per-chunk fits, for the tie-point residual tags.
    """
    fout = Path(fout)
    fout.parent.mkdir(parents=True, exist_ok=True)
    lon, lat, radius = loc["lon"], loc["lat"], loc["radius"]
    tp = _tie_point_residual(fits or [])
    tags = {
        "IIRS_PRODUCT": "LOC",
        "IIRS_FORMAT_VERSION": "1",
        "IIRS_SID": sid,
        "GEOMETRY_BASIS": (
            "Registration-derived: camera pixels registered to LOLA hillshade per solve group, bicubic spline "
            "on the GCP lattice (pixel-centre offset); radius is LOLA sampled at the "
            "registered point; not a sensor-model ray/DEM intersection. No fill value defined by "
            "the M3 SIS; NaN outside every solved group's rows."
        ),
        "GROUPS": {g: list(r) for g, r in loc["groups"].items()},
        "PRECISION": f"{dtype}; ULP up to 0.9 m lon, 0.23 m lat, 0.125 m radius" if dtype == "float32" else dtype,
        "TIE_POINT_RESIDUAL_P50_M": tp["p50_m"] if tp else "unavailable",
        "TIE_POINT_RESIDUAL_P95_M": tp["p95_m"] if tp else "unavailable",
        "PROVENANCE_IIRSPY_VERSION": io.IIRSPY_VERSION,
        "PROVENANCE_DEM_ROOTS": ":".join(str(p) for p in dem_roots),
    } | (tags or {})
    da = _backplane_da(
        [np.asarray(lon, dtype=dtype), np.asarray(lat, dtype=dtype), np.asarray(radius, dtype=dtype)],
        ["longitude", "latitude", "radius"],
        ["deg", "deg", "m"],
        row0,
        tags,
    )
    return Path(io.write_camera_product(da, fout, compress=compress, predictor=predictor))


def _read_backplane(path: Path) -> tuple[xr.DataArray, list[str], dict]:
    """This backplane's data (band, y, x) with absolute `y`/`x` coords, its per-band names (from
    GDAL band descriptions -- generic over format and band count), and its scalar tags.

    ENVI's own extra header keys land in GDAL's "ENVI" tag namespace, not the default one `.tif`
    tags use -- merged here so a scalar provenance attr round-trips from either format. GDAL's ENVI
    header parser silently drops any value containing "=" (e.g. an inequality in free text), a
    format limitation, not something this merge can recover.
    """
    da = io.apply_envi_start(xr.open_dataarray(path, engine="rasterio"))
    with rasterio.open(path) as src:
        names = [src.descriptions[i] or f"band{i + 1}" for i in range(src.count)]
        tags = {**src.tags(ns="ENVI"), **src.tags()}
    return da, names, tags


def read_loc(path: Path) -> xr.Dataset:
    """`<sid>_loc.tif`/`.img` back as an `xr.Dataset` (lon, lat, radius), dims `("y", "x")` matching
    how `IIRSData`/`L1` name a camera-space cube's own dims (`iirs.py`'s `y`=scan row, `x`=camera
    col). `y`/`x` coords are the absolute scan row and camera column, not a pixel index restarting
    at 0 -- via `io.apply_envi_start`, the same convention `L1.from_file` uses. Also reads ISSDC's
    ndi `_d_loc_` backplane (Longitude/Latitude/Radius/Height).
    """
    da, names, tags = _read_backplane(path)
    name_map = {"longitude": "lon", "latitude": "lat"}
    return xr.Dataset(
        {name_map.get(n.lower(), n.lower()): (("y", "x"), da.values[i]) for i, n in enumerate(names)},
        coords={"y": da.y.values, "x": da.x.values},
        attrs=dict(tags),
    )


def read_obs(path: Path) -> xr.Dataset:
    """`<sid>_obs.tif`/`.img` back as an `xr.Dataset`, one variable per band named from its GDAL
    band description. Generic over however many bands the OBS product carries, same `("y", "x")`
    absolute-coordinate convention as `read_loc`."""
    da, names, tags = _read_backplane(path)
    return xr.Dataset(
        {name: (("y", "x"), da.values[i]) for i, name in enumerate(names)},
        coords={"y": da.y.values, "x": da.x.values},
        attrs=dict(tags),
    )


# ------------------------------------------------------------------------------------------
# OBS: ISSDC ndi _d_obs_ bands (the values they promise) + extras, at every camera pixel
# ------------------------------------------------------------------------------------------
def _local_az_zenith(
    u: np.ndarray, east: np.ndarray, north: np.ndarray, up: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(azimuth, zenith) [deg] of a unit look vector `u`, in the local topocentric frame
    (`east`/`north`/`up`, each same-shape as `u`): azimuth clockwise from local north 0-360,
    zenith from the local vertical (M3 OBS convention, DPSIS Sec 2.5.3.3).

    >>> [round(float(v), 3) for v in _local_az_zenith(*[np.array(v) for v in
    ...     ((0.5, 0.0, 0.8660254), (0, 1, 0), (0, 0, 1), (1, 0, 0))])]
    [0.0, 60.0]
    >>> [round(float(v), 3) for v in _local_az_zenith(*[np.array(v) for v in
    ...     ((1.0, 0.0, 0.0), (0, 1, 0), (0, 0, 1), (1, 0, 0))])]
    [0.0, 0.0]
    """
    az = np.degrees(np.arctan2((u * east).sum(0), (u * north).sum(0))) % 360
    zen = 90.0 - np.degrees(np.arcsin((u * up).sum(0)))
    return az, zen


def _row_geometry(
    sid: str,
    lon: np.ndarray,
    lat: np.ndarray,
    radius: np.ndarray,
    scan0: int,
    kernels: list[Path],
    paths: io.Paths,
    stage: Path | None = None,
):
    """Per-pixel sun/sensor terms (IIRS OBS bands 1-7) for one row block, from its own loc (lon, lat,
    radius): sun and sensor azimuth/zenith [deg], sun distance [AU], sensor range [m].

    Row times come from the nri label (`ancillary.scan_utc`), never from .spm row numbers. The sun needs only
    the generic kernels; the spacecraft comes from `georef.sc_positions_km` (CH2 SPK, else the .spm). Whatever
    cannot be computed -- no label, no kernels, no spacecraft state -- is left NaN.

    Azimuth/zenith are in the local topocentric (east, north, up) frame at each pixel (`georef.enu`),
    azimuth clockwise from local north. Distances are true per-pixel (Sun/spacecraft minus this pixel's
    body-fixed XYZ), so they carry the small topocentric parallax across the scene.
    """
    import spiceypy as sp

    ny, nx = lon.shape
    sun_az, sun_zen, sun_dist_au, sens_az, sens_zen, sens_dist_m = (np.full((ny, nx), np.nan) for _ in range(6))
    sc_xyz_km = np.full((3, ny), np.nan)
    flabel = ck.label(sid, paths, stage)
    if flabel is None or not kernels:
        return sun_az, sun_zen, sun_dist_au, sens_az, sens_zen, sens_dist_m, "none", sc_xyz_km

    for k in kernels:
        sp.furnsh(str(k))
    scans = scan0 + np.arange(ny)
    ets = [sp.str2et(u) for u in ancillary.scan_utc(flabel, scans)]
    sc_xyz_km, sc_source = sc_positions_km(flabel, ck.spm(sid, paths, stage), scans)
    xyz_km = np.stack(lonlat_to_xyz(lon, lat, radius)) / 1000.0  # (3, ny, nx)
    for r, et in enumerate(ets):
        if not np.isfinite(lon[r]).any():
            continue
        east, north, up = enu(lon[r], lat[r])  # each (3, nx)
        p = xyz_km[:, r, :]
        d_sun = np.asarray(sp.spkpos("SUN", et, "IAU_MOON", "LT+S", "MOON")[0])[:, None] - p
        dist_sun = np.linalg.norm(d_sun, axis=0)
        sun_az[r], sun_zen[r] = _local_az_zenith(d_sun / dist_sun, east, north, up)
        sun_dist_au[r] = dist_sun * 1000.0 / AU_M
        d_sc = sc_xyz_km[:, r, None] - p
        dist_sc = np.linalg.norm(d_sc, axis=0)
        sens_az[r], sens_zen[r] = _local_az_zenith(d_sc / dist_sc, east, north, up)
        sens_dist_m[r] = dist_sc * 1000.0
    return sun_az, sun_zen, sun_dist_au, sens_az, sens_zen, sens_dist_m, sc_source, sc_xyz_km


def _phase_angle(sun_zen, sens_zen, sun_az, sens_az):
    """Phase angle [deg] between the to-sun and to-sensor rays, from the spherical law of cosines
    on their zenith/azimuth pair (OBS band 5); acute where ISSDC's own band 5 is 180 minus this."""
    z1, z2 = np.radians(sun_zen), np.radians(sens_zen)
    daz = np.radians(sun_az - sens_az)
    c = np.cos(z1) * np.cos(z2) + np.sin(z1) * np.sin(z2) * np.cos(daz)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _facet_cos_i(sun_zen, sun_az, facet_slope, facet_aspect):
    """cos(topographic incidence), M3 OBS band 10 — the SIS's own `i_topo` formula, evaluated
    directly rather than via acos(...) then cos(...)."""
    z, s = np.radians(sun_zen), np.radians(facet_slope)
    daz = np.radians(sun_az - facet_aspect)
    return np.clip(np.cos(z) * np.cos(s) + np.sin(z) * np.sin(s) * np.cos(daz), -1.0, 1.0)


def _read_topo(f: Path) -> dict | None:
    """A `georef.save_topo` product's bands by name, or `None` if `f` is not on disk."""
    if not Path(f).exists():
        return None
    with rasterio.open(f) as src:
        return {src.descriptions[i]: src.read(i + 1) for i in range(src.count)}


def _l1_snr(f: Path) -> np.ndarray | None:
    """Broadband SNR over `PAN_BANDS` of the L1 cube `f`, or `None` if it is not on disk."""
    if not Path(f).exists():
        return None
    bands = [read_band(f, b) for b in PAN_BANDS]
    P = xr.DataArray(np.stack(bands), dims=("band", "y", "x"), coords={"band": PAN_BANDS}).mean("band")
    dark_mask, _thresh, _row_bright = detect_dark_rows(P)
    d0, d1 = longest_run(dark_mask)
    if d1 - d0 < 2:
        return np.full(P.shape, np.nan, "float32")
    if d1 - d0 >= MIN_ROWS:
        d0, d1 = refine_dark_block(P, (d0, d1))
    return np.asarray(broadband_snr(P, d0, d1).values, dtype="float32")


def pan_incomplete(f: Path) -> np.ndarray:
    """Any-PAN-band-NaN mask of the L1 cube `f`."""
    return np.isnan(np.stack([read_band(f, b) for b in PAN_BANDS])).any(0)


def _kernels_or_none(day: str, paths: io.Paths) -> list[Path]:
    """`ck.kernels(day, paths)`, or [] when there is no SPICE tree: `_row_geometry` then leaves sun/sensor NaN."""
    try:
        return ck.kernels(day, paths)
    except FileNotFoundError:
        return []


OBS_NAMES = [
    "sun_azimuth",
    "sun_zenith",
    "sensor_azimuth",
    "sensor_zenith",
    "phase",
    "sun_distance",
    "sensor_distance",
    "facet_slope",
    "facet_aspect",
    "facet_cos_i",
    "lit_frac",
    "sky_view",
    "snr",
    "tie_dist",
]


def _group_obs(sid, group, lon, lat, radius, scan0, topo, snr, paths, stage_dir) -> tuple[dict, str, np.ndarray]:
    """One group's OBS bands (all but `tie_dist`) over its own rows, which start at absolute Scan `scan0`:
    ({name: (ny, nx)}, spacecraft source, (3, ny) spacecraft km). `topo`/`snr` may be `None` (-> NaN)."""
    ny, nx = lon.shape
    s_az, s_zen, s_dist, e_az, e_zen, e_dist, sc_source, sc_xyz_km = _row_geometry(
        sid, lon, lat, radius, scan0, _kernels_or_none(sid[:8], paths), paths, stage_dir
    )
    nan = np.full((ny, nx), np.nan, "float32")
    slope = aspect = lit_frac = cos_i = nan
    if topo is not None:
        m = min(ny, topo["slope"].shape[0])
        slope, aspect, lit_frac = nan.copy(), nan.copy(), nan.copy()
        slope[:m], lit_frac[:m] = topo["slope"][:m], topo["lit_frac"][:m]
        # the topo's aspect is map-grid referenced; OBS wants local true north
        lon0 = stereo_crs(group).to_dict().get("lon_0", 0.0)
        grid_sign = 0.0 if group == "equatorial" else (-1.0 if group == "north" else 1.0)
        aspect[:m] = (topo["aspect"][:m] - grid_sign * (lon[:m] - lon0)) % 360.0
        cos_i = _facet_cos_i(s_zen, s_az, slope, aspect)
    bands = {
        "sun_azimuth": s_az,
        "sun_zenith": s_zen,
        "sensor_azimuth": e_az,
        "sensor_zenith": e_zen,
        "phase": _phase_angle(s_zen, e_zen, s_az, e_az),
        "sun_distance": s_dist,
        "sensor_distance": e_dist,
        "facet_slope": slope,
        "facet_aspect": aspect,
        "facet_cos_i": cos_i,
        "lit_frac": lit_frac,
        "sky_view": nan,
        "snr": nan if snr is None else snr[:ny],
    }
    return bands, sc_source, sc_xyz_km


def obs_from_loc(
    sid: str, group: str, loc: dict, scan0: int, gcps: dict, paths: io.Paths, ftopo=None, fl1=None, stage_dir=None
) -> dict:
    """All OBS bands over a group's `loc` (`lon`/`lat`/`radius`, crop-relative rows starting at absolute Scan
    `scan0`): sun/sensor terms from SPICE, facet terms from the topo product `ftopo`, SNR from the L1 cube `fl1`,
    tie distance from the merged `gcps` lattice."""
    lon, lat, radius = loc["lon"], loc["lat"], loc["radius"]
    ny, nx = lon.shape
    bands, sc_source, sc_xyz_km = _group_obs(
        sid,
        group,
        lon,
        lat,
        radius,
        scan0,
        _read_topo(ftopo) if ftopo else None,
        _l1_snr(fl1) if fl1 else None,
        paths,
        stage_dir,
    )
    bands = {n: np.asarray(a, "float32") for n, a in bands.items()}
    tree = cKDTree(np.array(list(gcps), dtype="float64"))
    rr, cc = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    d, _ = tree.query(np.stack([rr.ravel(), cc.ravel()], axis=1))
    bands["tie_dist"] = (d * IIRS_GSD_M).reshape(ny, nx).astype("float32")
    return {
        "bands": bands,
        "groups": {group: (scan0, scan0 + ny - 1)},
        "sensor_source": sc_source,
        "sc_xyz_m": sc_xyz_km * 1000.0,
    }


ISSDC_OBS_FIXES = (
    "vs ISSDC ndi _d_obs_: band 5 is the phase angle (ISSDC writes 180 - phase); band 6 is the absolute "
    "Sun-pixel distance in AU (ISSDC: km, but labelled au minus a constant); band 7 "
    "is the true pixel-IIRS range in m (ISSDC: near-constant ~1735.6 km); bands 8-9 are real facet "
    "slope/aspect in deg (ISSDC: ~89.9 deg slope): slope is the tilt of the LOLA facet normal from the local "
    "vertical, i.e. of the facet from the local horizontal (tangent) plane; aspect is the facet's downhill "
    "direction projected onto that tangent plane, clockwise from local true north, 0-360; both from the DEM "
    "gradient smoothed to the IIRS ground sample"
)


def write_obs(
    fout: Path, obs: dict, sid: str, compress: str = "ZSTD", predictor: int = 3, row0: int = 0, tags: dict | None = None
) -> Path:
    """Write `<sid>_obs.tif`/`.img`: 14-band f32, camera space (no CRS). Bands 1-9 are ISSDC's ndi `_d_obs_`
    layout, with the values it promises (see `ISSDC_OBS_FIXES`); bands 10-14 are extras. Lossless, tiled,
    `INTERLEAVE=BAND`."""
    fout = Path(fout)
    fout.parent.mkdir(parents=True, exist_ok=True)
    names_units = [
        ("sun_azimuth", "deg"),
        ("sun_zenith", "deg"),
        ("sensor_azimuth", "deg"),
        ("sensor_zenith", "deg"),
        ("phase", "deg"),
        ("sun_distance", "AU"),
        ("sensor_distance", "m"),
        ("facet_slope", "deg"),
        ("facet_aspect", "deg"),
        ("facet_cos_i", "1"),
        ("lit_frac", "1"),
        ("sky_view", "1"),
        ("snr", "1"),
        ("tie_dist", "m"),
    ]
    bands = obs["bands"]
    tags = {
        "IIRS_PRODUCT": "OBS",
        "IIRS_FORMAT_VERSION": "1",
        "IIRS_SID": sid,
        "CONVENTION": "Bands 1-9 follow the ISSDC ndi _d_obs_ band order; bands 10+ are extensions",
        "ISSDC_OBS_FIXES": ISSDC_OBS_FIXES,
        "GEOMETRY_BASIS": (
            "Registration-derived: sun/sensor terms at the loc's registered XYZ, row times from the nri "
            "label; facet terms from the LOLA DEM normal, not a sensor-model ray/DEM intersection."
        ),
        "SENSOR_GEOMETRY_SOURCE": (
            f"{obs['sensor_source']} (spk: CH2 SPK; spm: the nri .spm state vector, identical to the SPK; "
            "none: bands 3-5 and 7 NaN)"
        ),
        "BANDS": "; ".join(f"{i} {n} [{u}]" for i, (n, u) in enumerate(names_units, start=1)),
        "ANGLE_CONVENTION": (
            "Azimuth clockwise from local topocentric north, 0-360 deg. Zenith from the local "
            "vertical (sphere normal for sun/sensor bands, DEM facet normal for facet_cos_i). "
            "Facet aspect clockwise from local north."
        ),
        "FACET_SCALE_M": str(IIRS_GSD_M),
        "PAN_BANDS": ",".join(str(b) for b in PAN_BANDS),
        "SHADOW_SNR": str(SHADOW_SNR),
        "QA_FILE": f"{sid}_qa.tif",
        "SKY_VIEW_STATUS": "reserved, NaN: the lit_frac horizon sweep only samples toward the sun's own azimuth",
        "TIE_DIST_DEFINITION": "distance to the nearest solved GCP lattice node, not the raw tie-point detection",
        "PROVENANCE_IIRSPY_VERSION": io.IIRSPY_VERSION,
        "GROUPS": {g: list(r) for g, r in obs["groups"].items()},
    } | (tags or {})
    names = [n for n, _u in names_units]
    units = [u for _n, u in names_units]
    da = _backplane_da([bands[n] for n in names], names, units, row0, tags)
    return Path(io.write_camera_product(da, fout, compress=compress, predictor=predictor))


# ------------------------------------------------------------------------------------------
# QA: uint16 bit flags
# ------------------------------------------------------------------------------------------
QA_BITS = {
    "no_geometry": (0, "LOC undefined (row outside every solved group)"),
    "gcp_extrapolated": (1, "outside the control lattice hull, LOC extrapolating"),
    "in_gcp_chunk_overlap": (2, "overlap of two chunks of one group, geometry cross-faded"),
    "in_gcp_region_overlap": (3, "overlap of two solve groups, geometry seam-blended"),
    "in_geometric_shadow": (4, f"lit_frac < {GEOM_SHADOW_LIT} (sun centre below local LOLA horizon)"),
    "is_lit": (5, f"broadband PAN snr >= {SHADOW_SNR}"),
    "is_saturated": (6, "any PAN band at/above saturation radiance"),
    "pan_incomplete": (7, "any PAN band NaN at this pixel"),
    "no_topo": (8, "no DEM facet terms: facet_cos_i is the sphere cosine, slope/aspect/lit_frac NaN"),
}


def qa_mask(name: str) -> int:
    return 1 << QA_BITS[name][0]


def _extrapolated_mask(gcps: dict, ny: int, nx: int) -> np.ndarray:
    """True where a pixel centre lies outside the convex hull of the GCP lattice (row, col), i.e. LOC extrapolates."""
    pts = np.array(list(gcps), dtype="float64")
    rr, cc = np.meshgrid(np.arange(ny) + 0.5, np.arange(nx) + 0.5, indexing="ij")
    inside = Delaunay(pts).find_simplex(np.stack([rr.ravel(), cc.ravel()], axis=1)) >= 0
    return np.asarray(~inside.reshape(ny, nx))


def build_qa(
    loc: dict,
    obs: dict,
    gcps: dict | None = None,
    overlaps: Sequence[tuple[int, int]] = (),
    pan_bad: np.ndarray | None = None,
    saturated: np.ndarray | None = None,
) -> np.ndarray:
    """The uint16 QA bit field for one group, from `loc`, `obs`, the merged `gcps` lattice (extrapolation hull),
    crop-relative chunk `overlaps`, and the optional per-pixel PAN-incomplete / saturated masks (bit left 0 when
    the mask is `None`, since `SATURATION_CHECK` records whether it was evaluated)."""
    ny, nx = loc["lon"].shape
    qa = np.zeros((ny, nx), dtype="uint16")
    qa[np.isnan(loc["lon"])] |= qa_mask("no_geometry")
    if gcps:
        qa[_extrapolated_mask(gcps, ny, nx)] |= qa_mask("gcp_extrapolated")
    for lo, hi in overlaps:
        qa[lo:hi] |= qa_mask("in_gcp_chunk_overlap")
    qa[np.nan_to_num(obs["bands"]["lit_frac"], nan=1.0) < GEOM_SHADOW_LIT] |= qa_mask("in_geometric_shadow")
    qa[np.nan_to_num(obs["bands"]["snr"], nan=-np.inf) >= SHADOW_SNR] |= qa_mask("is_lit")
    for mask, name in ((pan_bad, "pan_incomplete"), (saturated, "is_saturated")):
        if mask is not None:
            qa[mask[:ny]] |= qa_mask(name)
    return qa


def write_qa(
    fout: Path,
    qa: np.ndarray,
    sid: str,
    groups: dict[str, tuple[int, int]],
    saturation_status: dict,
    row0: int = 0,
    tags: dict | None = None,
) -> Path:
    """Write `<sid>_qa.tif`/`.img`: uint16 bit flags, lossless."""
    fout = Path(fout)
    fout.parent.mkdir(parents=True, exist_ok=True)
    tags = {
        "IIRS_PRODUCT": "QA",
        "IIRS_FORMAT_VERSION": "1",
        "IIRS_SID": sid,
        **{f"QA_BIT_{b}": f"{n}: {d}" for n, (b, d) in QA_BITS.items()},
        "QA_RESERVED_BITS": f"{len(QA_BITS)}-15",
        "PAN_BANDS": ",".join(str(b) for b in PAN_BANDS),
        "GEOM_SHADOW_LIT": str(GEOM_SHADOW_LIT),
        "SHADOW_SNR": str(SHADOW_SNR),
        "SATURATION_CHECK": saturation_status,
        "GROUPS": {g: list(r) for g, r in groups.items()},
        "PROVENANCE_IIRSPY_VERSION": io.IIRSPY_VERSION,
    } | (tags or {})
    da = _backplane_da([qa], ["qa_bits"], [""], row0, tags)
    return Path(io.write_camera_product(da, fout, compress="ZSTD", predictor=2))
