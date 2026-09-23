"""Camera-space backplanes (M3-style LOC/OBS/QA) for one solve group, absolute-scan-row keyed."""

import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.control import GroundControlPoint
from rasterio.transform import GCPTransformer
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay, cKDTree

import iirspy.utils as utils
from iirspy import chunks as ck
from iirspy import solve
from iirspy.empirical import (
    MIN_ROWS,
    PAN_BANDS,
    SHADOW_SNR,
    broadband_snr,
    detect_dark_rows,
    longest_run,
    refine_dark_block,
)
from iirspy.georef import (
    MOON_RADIUS_M,
    GeorefConfig,
    _enu,
    load_lola_elev,
    lonlat_to_xyz,
    read_band,
    stereo_crs,
    xy_to_lonlat,
)
from iirspy.iirs import _apply_envi_start, _write_camera_product
from iirspy.utils import IIRSPY_VERSION

IIRS_GSD_M = GeorefConfig().iirs_gsd_m
DEM_MARGIN_M = 2_000.0  # bbox pad for a DEM window read around one row block's ground footprint
AU_M = 1.495978707e11
GEOM_SHADOW_LIT = 0.5  # lit fraction below this: sun centre below the local LOLA horizon


@dataclass
class GroupInfo:
    group: str
    scan0: int
    row_range: tuple[int, int]  # absolute scan rows, inclusive, from the group's own .gcps extent
    chunk_rows: list[tuple[int, int, int]]  # (chunk i, absolute row0, absolute row1)


def group_info(sid: str, group: str) -> GroupInfo:
    """Absolute-scan row range and per-chunk row ranges for one solved group.

    The solve summary is authoritative: `chunks.json` holds the same pre-solve `chunks` plan, but only
    the summary carries `scan0` and is written once the merge actually succeeds. The gcps file's
    own row extent (not the chunk plan) sets `row_range`, since a partial merge can fall short of
    the last planned chunk; `chunk_rows` is the plan's chunks clipped to that same extent.
    """
    d = ck.recal_dir(sid, group)
    f = d / f"georef_solve_summary_{ck.GROUP_SHORT[group]}.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: {sid}/{group} was not merged (partial --chunks run?)")
    data = json.loads(f.read_text())
    scan0 = int(data["scan0"])
    gcps = solve._load_gcps(solve.merged_gcps_path(sid, group))
    rows = [int(r) for r, _c in gcps]
    row_range = (scan0 + min(rows), scan0 + max(rows))
    chunk_rows = [
        (int(c["i"]), scan0 + int(c["row0"]), scan0 + int(c["row1"])) for c in data["chunks"] if c["row0"] <= max(rows)
    ]
    return GroupInfo(group, scan0, row_range, chunk_rows)


def chunk_overlaps(gi: GroupInfo) -> list[tuple[int, int, int, int]]:
    """(row0, row1, chunk_i, chunk_i) absolute-row overlaps between consecutive chunks of one group."""

    out = []
    rows = sorted(gi.chunk_rows, key=lambda c: c[1])
    for (ia, a0, a1), (ib, b0, b1) in pairwise(rows):
        lo, hi = max(a0, b0), min(a1, b1)
        if lo < hi:
            out.append((lo, hi, ia, ib))
    return out


# ------------------------------------------------------------------------------------------
# LOC: lon/lat/radius at every camera pixel
# ------------------------------------------------------------------------------------------
def _dem_band(group: str, lat: float) -> dict:
    """The `chunks.bands()` entry covering `lat` within `group`; clamped to the nearest band's
    `lat_range` if `lat` falls outside every one (TPS extrapolating past the solved edge)."""
    resolved = ck.bands()
    cands = [b for b in resolved.values() if b["group"] == group]
    inside = [b for b in cands if b["lat_range"][0] <= lat <= b["lat_range"][1]]
    if inside:
        return inside[0]
    return min(cands, key=lambda b: min(abs(lat - b["lat_range"][0]), abs(lat - b["lat_range"][1])))


def _sample_radius(x: np.ndarray, y: np.ndarray, lat: np.ndarray, group: str) -> np.ndarray:
    """LOLA radius [m from Moon centre] at every (x, y) in `group`'s stereo CRS.

    DEM band picked once from this block's median latitude, not per pixel: a block only crosses
    a band boundary near the transition itself, and `_dem_band` already falls back to the nearest
    band there, so the height error at that seam is bounded by the DEM tiers' own overlap.
    """
    band = _dem_band(group, float(np.nanmedian(lat)))
    bounds = (
        float(np.nanmin(x)) - DEM_MARGIN_M,
        float(np.nanmin(y)) - DEM_MARGIN_M,
        float(np.nanmax(x)) + DEM_MARGIN_M,
        float(np.nanmax(y)) + DEM_MARGIN_M,
    )
    elev, tr, _ = load_lola_elev(band["dem_near"], bounds=bounds)
    rr = (y - tr.f) / tr.e
    cc = (x - tr.c) / tr.a
    h = map_coordinates(elev, [rr, cc], order=1, mode="nearest")
    return np.asarray(MOON_RADIUS_M + h)


def _group_loc_core(gcps: dict, group: str, nrow: int | None = None, ncol: int | None = None, row_block: int = 500):
    """(lon, lat, radius) at every camera pixel centre of one group's GCP TPS, crop-relative rows.

    `offset="center"`: GDAL's GCP pixel/line convention puts integer row/col on pixel *corners*, so
    this reproduces the `georef.project`/`scene_glt` warp; `offset="ul"` would be half a pixel off.
    `nrow`/`ncol` default to the lattice extent; larger values extrapolate the TPS past the last GCP.
    The transformer is built once (its O(n_gcps^3) solve dominates) and evaluated in `row_block` rows.
    """
    rows_g = [r for r, _c in gcps]
    cols_g = [c for _r, c in gcps]
    nrow = int(max(rows_g)) + 1 if nrow is None else nrow
    ncol = int(max(cols_g)) + 1 if ncol is None else ncol
    gcp_list = [GroundControlPoint(row=float(r), col=float(c), x=x, y=y) for (r, c), (x, y) in gcps.items()]

    lon = np.full((nrow, ncol), np.nan)
    lat = np.full((nrow, ncol), np.nan)
    radius = np.full((nrow, ncol), np.nan)
    cols = np.arange(ncol)
    with GCPTransformer(gcp_list, tps=True) as t:
        for r0 in range(0, nrow, row_block):
            r1 = min(r0 + row_block, nrow)
            rr, cc = np.meshgrid(np.arange(r0, r1), cols, indexing="ij")
            xf, yf = t.xy(rr.ravel().tolist(), cc.ravel().tolist(), offset="center")
            x = np.asarray(xf).reshape(rr.shape)
            y = np.asarray(yf).reshape(rr.shape)
            lo, la = xy_to_lonlat(x, y, pole=group)
            lon[r0:r1], lat[r0:r1] = lo, la
            radius[r0:r1] = _sample_radius(x, y, la, group)
    return lon, lat, radius


def _tie_point_residual(sid: str, groups: list[str], fit_dir: Path | None = None) -> dict[str, float] | None:
    """Aggregate p50/p95 tie-point residual [m] across every chunk fit json under `groups`' own
    solve dirs, or `None` if no `chunk*_fit.json` is found (an older solve, or a synthetic test)."""
    med, p95 = [], []
    for g in groups:
        d = fit_dir if fit_dir is not None else ck.recal_dir(sid, g)
        for f in sorted(Path(d).glob("chunk*_fit.json")):
            iters = json.loads(f.read_text()).get("stats", {}).get("iters", [])
            if iters and "shift_m" in iters[-1]:
                med.append(iters[-1]["shift_m"]["median"])
                p95.append(iters[-1]["shift_m"]["p95"])
    if not med:
        return None
    return {"p50_m": float(np.median(med)), "p95_m": float(np.max(p95))}


def _tags_to_attrs(tags: dict) -> dict:
    """`tags` ready for `da.attrs`: dict/list values json.dumps'ed so `_save_geotiff`/
    `write_envi_hdr` (which only round-trip scalar attrs) both carry every tag."""
    return {k: (json.dumps(v) if isinstance(v, dict | list) else v) for k, v in tags.items()}


def _backplane_da(bands: list, names: list[str], units: list[str], row0: int, tags: dict) -> xr.DataArray:
    """(band, y, x) DataArray for a camera-space backplane: `bands` stacked in order, `names`/
    `units` as non-dimension coords on `band` (`_save_geotiff`'s band-label branch for a product
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
    fit_dir: Path | None = None,
) -> Path:
    """Write `<sid>_loc.tif`/`.img`: 3 bands lon [deg, 0-360 E], lat [deg, planetocentric], radius
    [m from Moon centre], matching the M3 LOC band order/units. Full strip rows x 250, camera
    space: no CRS or transform. `dtype` defaults f32 (ULP <=0.9 m lon, 0.23 m lat, 0.125 m radius,
    well under the ~80 m pixel this backs); computed in f64 and cast only here -- note the shared
    GeoTIFF writer (`iirs._save_geotiff`) always writes float dtypes as float32, so `dtype="float64"`
    only affects the pre-write cast, not the file actually written. `compress`/`predictor`/`dtype`
    are exposed, not frozen. `row0` is this array's absolute scan row 0 (0 for the current
    full-strip callers).
    """
    fout = Path(fout)
    fout.parent.mkdir(parents=True, exist_ok=True)
    lon, lat, radius = loc["lon"], loc["lat"], loc["radius"]
    tp = _tie_point_residual(sid, list(loc["groups"]), fit_dir=fit_dir)
    tags = {
        "IIRS_PRODUCT": "LOC",
        "IIRS_FORMAT_VERSION": "1",
        "IIRS_SID": sid,
        "GEOMETRY_BASIS": (
            "Registration-derived: camera pixels TPS-fitted to LOLA hillshade per solve group "
            "(pixel-centre offset); radius is LOLA sampled at the "
            "registered point; not a sensor-model ray/DEM intersection. No fill value defined by "
            "the M3 SIS; NaN outside every solved group's rows."
        ),
        "GROUPS": {g: list(r) for g, r in loc["groups"].items()},
        "PRECISION": f"{dtype}; ULP up to 0.9 m lon, 0.23 m lat, 0.125 m radius" if dtype == "float32" else dtype,
        "TIE_POINT_RESIDUAL_P50_M": tp["p50_m"] if tp else "unavailable",
        "TIE_POINT_RESIDUAL_P95_M": tp["p95_m"] if tp else "unavailable",
        "PROVENANCE_IIRSPY_VERSION": IIRSPY_VERSION,
        "PROVENANCE_DEM_ROOTS": ":".join(str(p) for p in ck._DEM_ROOTS),
    }
    da = _backplane_da(
        [np.asarray(lon, dtype=dtype), np.asarray(lat, dtype=dtype), np.asarray(radius, dtype=dtype)],
        ["longitude", "latitude", "radius"],
        ["deg", "deg", "m"],
        row0,
        tags,
    )
    return Path(_write_camera_product(da, fout, compress=compress, predictor=predictor))


def _read_backplane(path: Path) -> tuple[xr.DataArray, list[str], dict]:
    """This backplane's data (band, y, x) with absolute `y`/`x` coords, its per-band names (from
    GDAL band descriptions -- generic over format and band count), and its scalar tags.

    ENVI's own extra header keys land in GDAL's "ENVI" tag namespace, not the default one `.tif`
    tags use -- merged here so a scalar provenance attr round-trips from either format. GDAL's ENVI
    header parser silently drops any value containing "=" (e.g. an inequality in free text), a
    format limitation, not something this merge can recover.
    """
    da = _apply_envi_start(xr.open_dataarray(path, engine="rasterio"))
    with rasterio.open(path) as src:
        names = [src.descriptions[i] or f"band{i + 1}" for i in range(src.count)]
        tags = {**src.tags(ns="ENVI"), **src.tags()}
    return da, names, tags


def read_loc(path: Path) -> xr.Dataset:
    """`<sid>_loc.tif`/`.img` back as an `xr.Dataset` (lon, lat, radius), dims `("y", "x")` matching
    how `IIRSData`/`L1` name a camera-space cube's own dims (`iirs.py`'s `y`=scan row, `x`=camera
    col). `y`/`x` coords are the absolute scan row and camera column, not a pixel index restarting
    at 0 -- via `iirs._apply_envi_start`, the same convention `L1.from_file` uses.
    """
    da, names, tags = _read_backplane(path)
    name_map = {"longitude": "lon", "latitude": "lat", "radius": "radius"}
    return xr.Dataset(
        {name_map[n]: (("y", "x"), da.values[i]) for i, n in enumerate(names)},
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
# OBS: M3 sun/sensor/facet angles + IIRS extras, at every camera pixel
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


def _row_geometry(sid: str, lon: np.ndarray, lat: np.ndarray, radius: np.ndarray, scan0: int, kernels: list[Path]):
    """Per-pixel M3 OBS bands 1-7 (to-sun/to-sensor azimuth/zenith/phase/path length) for one
    row block, from SPICE plus this block's own loc (lon, lat, radius).

    Azimuth/zenith come straight from the local topocentric (east, north, up) frame at each pixel
    (`georef._enu`), which is already what M3's "local north" convention means — unlike
    `georef.sun_geometry_rows`, no grid-north correction is needed here, since that correction only
    exists to feed the hillshade renderer, which works in map-projection space.

    Path length is the true per-pixel distance (Sun/spacecraft position minus this pixel's body-fixed
    XYZ), not a scene-constant range, so it carries the same small topocentric parallax the M3 SIS
    describes as "deviations from the scene mean". Sensor terms are NaN (and `sc_ok=False`) if the
    kernel set has no Chandrayaan-2 orbit SPK for this epoch.
    """
    import spiceypy as sp

    for k in kernels:
        sp.furnsh(str(k))

    ny, nx = lon.shape
    fspm = ck.ancillary(sid)["miscellaneous/raw"]
    if fspm is None:
        raise FileNotFoundError(f"{sid}: no .spm for row timing")
    spm = utils.load_iirs_spm(fspm)
    spm_row = spm.row.to_numpy()
    spm_ns = spm.datetime.to_numpy().astype("datetime64[ns]").astype("int64")
    row_ns = np.interp(scan0 + np.arange(ny), spm_row, spm_ns).astype("int64")

    xyz_km = np.stack(lonlat_to_xyz(lon, lat, radius)) / 1000.0  # (3, ny, nx)
    sun_az = np.full((ny, nx), np.nan)
    sun_zen = np.full((ny, nx), np.nan)
    sun_dist_au = np.full((ny, nx), np.nan)
    sens_az = np.full((ny, nx), np.nan)
    sens_zen = np.full((ny, nx), np.nan)
    sens_dist_m = np.full((ny, nx), np.nan)
    sc_xyz_km = np.full((3, ny), np.nan)

    sc_ok = True
    for r in range(ny):
        if not np.isfinite(lon[r]).any():
            continue
        et = sp.str2et(pd.Timestamp(row_ns[r]).strftime("%Y-%m-%dT%H:%M:%S.%f"))
        east, north, up = _enu(lon[r], lat[r])  # each (3, nx)
        p = xyz_km[:, r, :]

        v_sun, _ = sp.spkpos("SUN", et, "IAU_MOON", "LT+S", "MOON")
        d_sun = np.asarray(v_sun)[:, None] - p
        dist_sun = np.linalg.norm(d_sun, axis=0)
        sun_az[r], sun_zen[r] = _local_az_zenith(d_sun / dist_sun, east, north, up)
        sun_dist_au[r] = dist_sun * 1000.0 / AU_M

        if sc_ok:
            try:
                v_sc, _ = sp.spkpos("CHANDRAYAAN-2", et, "IAU_MOON", "LT+S", "MOON")
            except Exception:
                sc_ok = False
                continue
            d_sc = np.asarray(v_sc)[:, None] - p
            dist_sc = np.linalg.norm(d_sc, axis=0)
            sens_az[r], sens_zen[r] = _local_az_zenith(d_sc / dist_sc, east, north, up)
            sens_dist_m[r] = dist_sc * 1000.0
            sc_xyz_km[:, r] = np.asarray(v_sc)

    return sun_az, sun_zen, sun_dist_au, sens_az, sens_zen, sens_dist_m, sc_ok, sc_xyz_km


def _phase_angle(sun_zen, sens_zen, sun_az, sens_az):
    """Phase angle [deg] between the to-sun and to-sensor rays, from the spherical law of cosines
    on their zenith/azimuth pair (M3 OBS band 5)."""
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


def _read_group_topo(sid: str, group: str) -> dict | None:
    """`<sid>_<group>_topo.tif` (slope, aspect, lit_frac), or `None` if not on disk for this group.

    `aspect` there is map-grid-referenced (the DEM gradient's own projected rows/cols), not the
    local-true-north the M3 OBS convention needs — `build_obs` corrects it with the same
    grid/true-north offset `georef.sun_geometry` derives for the hillshade renderer.
    """
    f = ck.recal_dir(sid, group) / f"{sid}_{group}_topo.tif"
    if not f.exists():
        return None
    with rasterio.open(f) as src:
        bands = {src.descriptions[i]: src.read(i + 1) for i in range(src.count)}
    return bands


def _l1_rad_path(sid: str, group: str) -> Path:
    return ck.RECAL_ROOT / "data" / "recalibrated" / sid[:8] / f"{sid}_{group}" / f"{sid}_{group}_l1_rad.tif"


def _group_snr(sid: str, group: str) -> np.ndarray | None:
    """Broadband SNR over `empirical.PAN_BANDS`, from the group's own recalibrated L1 cube, or
    `None` if that cube is not on disk (a group processed only through geometry)."""
    f = _l1_rad_path(sid, group)
    if not f.exists():
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


def _pan_incomplete(sid: str, group: str) -> np.ndarray | None:
    """Any-PAN-band-NaN mask from the group's L1 cube, or `None` if it is not on disk."""
    f = _l1_rad_path(sid, group)
    if not f.exists():
        return None
    bands = [read_band(f, b) for b in PAN_BANDS]
    return np.isnan(np.stack(bands)).any(0)


def _pan_saturation_mask(sid: str, group: str, stage_dir: Path) -> np.ndarray | None:
    """True where any `empirical.PAN_BANDS` band's raw DN, run through the per-scene per-element
    gain/offset LUT, reaches its saturation radiance — the comparison `IIRSData.calibrate_to_rad`
    makes just before it NaNs the pixel out (that pre-NaN radiance is never itself exposed, and the
    empirical correction's own dark/flat/smile terms are near-unity broadband scaling, so the
    per-element LUT path used here is the same threshold, not a materially different one).

    Needs the nri raw cube, staged here (ancillary is already staged for other products; the cube
    itself is not) via the same ranged `issdc-iirs --bands` path `iirspy.solve._stage_inputs` uses,
    restricted to `PAN_BANDS`. `None` if the nri bundle is not archived for `sid`.
    """
    fzip = ck.nri_zip(sid)
    fgeom = ck.ancillary(sid)["geometry/calibrated"]
    if fzip is None or fgeom is None:
        return None
    from iirspy import L0

    day = sid[:8]
    stage_dir = Path(stage_dir)
    solve.SID, solve.GROUP, solve.ZIP, solve.STAGE = sid, group, fzip, stage_dir
    solve._stage_inputs(day, PAN_BANDS)
    lat_range = ck.l1_lat_range(group)
    scan0 = solve._scene_scan0(fgeom, lat_range)
    gi = group_info(sid, group)
    l0 = L0(sid, str(stage_dir), chunk={"band": -1, "y": 1024, "x": -1})
    # `y` is the native absolute Scan number, same crop `build_l1` makes off the geometry csv.
    l0.img = l0.img.sel(y=slice(scan0, gi.row_range[1] - gi.row_range[0] + scan0), band=PAN_BANDS)
    gain, offset = utils.get_gain_offset(l0.qub, calib_dir=utils.DCALIB)
    sat = utils.get_saturation_radiance(l0.qub, calib_dir=utils.DCALIB)
    rad = 10 * (l0.img * gain.sel(band=PAN_BANDS) + offset.sel(band=PAN_BANDS))
    return np.asarray((rad >= sat.sel(band=PAN_BANDS)).any("band").compute().values)


def build_obs(sid: str, loc: dict, stage_dir: Path | None = None) -> dict:
    """Every OBS band, assembled group by group over `loc` (`lon`/`lat`/`radius`/`groups`).
    Overlap rows between two groups take the earlier group's own value (first-owner-wins).
    """
    groups = list(loc["groups"])
    ny, nx = loc["lon"].shape
    names = [
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
    out = {n: np.full((ny, nx), np.nan, "float32") for n in names}
    owner = np.full(ny, "", dtype=object)
    sc_ok_any = False
    sun_dist_all = []
    sc_xyz_km_full = np.full((3, ny), np.nan)

    for g in groups:
        gi = group_info(sid, g)
        r0, r1 = gi.row_range[0], gi.row_range[1] + 1
        write = ~(owner[r0:r1] != "")
        owner[r0:r1] = np.where(write, g, owner[r0:r1])
        if not write.any():
            continue

        lon_g, lat_g, rad_g = loc["lon"][r0:r1], loc["lat"][r0:r1], loc["radius"][r0:r1]
        s_az, s_zen, s_dist, e_az, e_zen, e_dist, sc_ok, sc_xyz_km = _row_geometry(
            sid, lon_g, lat_g, rad_g, r0, ck.kernels(sid[:8])
        )
        sc_ok_any = sc_ok_any or sc_ok
        sc_xyz_km_full[:, r0:r1] = np.where(write[None, :], sc_xyz_km, sc_xyz_km_full[:, r0:r1])
        phase = _phase_angle(s_zen, e_zen, s_az, e_az)

        topo = _read_group_topo(sid, g)
        lon0_val = stereo_crs(g).to_dict().get("lon_0", 0.0)
        grid_sign = 0.0 if g == "equatorial" else (-1.0 if g == "north" else 1.0)
        nblk = r1 - r0
        if topo is not None:
            m = min(nblk, topo["slope"].shape[0])
            slope = np.full((nblk, nx), np.nan, "float32")
            aspect = np.full((nblk, nx), np.nan, "float32")
            lit_frac = np.full((nblk, nx), np.nan, "float32")
            slope[:m] = topo["slope"][:m]
            aspect_grid = topo["aspect"][:m]
            lit_frac[:m] = topo["lit_frac"][:m]
            aspect[:m] = (aspect_grid - grid_sign * (lon_g[:m] - lon0_val)) % 360.0
            cos_i = _facet_cos_i(s_zen, s_az, slope, aspect)
        else:
            slope = aspect = lit_frac = cos_i = np.full((nblk, nx), np.nan, "float32")

        snr = _group_snr(sid, g)
        n = r1 - r0
        for name, full in (
            ("sun_azimuth", s_az),
            ("sun_zenith", s_zen),
            ("sensor_azimuth", e_az),
            ("sensor_zenith", e_zen),
            ("phase", phase),
            ("sun_distance", s_dist),
            ("sensor_distance", e_dist),
            ("facet_slope", slope),
            ("facet_aspect", aspect),
            ("facet_cos_i", cos_i),
            ("lit_frac", lit_frac),
        ):
            out[name][r0:r1] = np.where(write[:, None], full[:n], out[name][r0:r1])
        if snr is not None:
            out["snr"][r0:r1] = np.where(write[:, None], snr[:n], out["snr"][r0:r1])
        sun_dist_all.append(s_dist[np.isfinite(s_dist)])

    sun_mean_au = float(np.concatenate(sun_dist_all).mean()) if sun_dist_all else float("nan")
    out["sun_distance"] = out["sun_distance"] - sun_mean_au
    out["tie_dist"] = _tie_dist(sid, groups, ny, nx)

    return {
        "bands": out,
        "groups": loc["groups"],
        "sun_distance_mean_au": sun_mean_au,
        "sensor_available": sc_ok_any,
        "sc_xyz_m": sc_xyz_km_full * 1000.0,
    }


def _tie_dist(sid: str, groups: list[str], ny: int, nx: int) -> np.ndarray:
    """Camera-space Euclidean distance [m] from every pixel to the nearest GCP lattice node of its
    owning group, `IIRS_GSD_M` per row/col unit. The solve does not persist its raw AROSICS tie-point
    detections separately from the lattice it fits through them, so this is distance to the nearest
    *solved lattice node*, not the original tie-point pixel — stated in the OBS `DEFINITION` tag.
    """
    out = np.full((ny, nx), np.nan, "float32")
    for g in groups:
        gi = group_info(sid, g)
        gcps = solve._load_gcps(solve.merged_gcps_path(sid, g))
        pts = np.array([(gi.scan0 + r, c) for r, c in gcps], dtype="float64")
        if len(pts) == 0:
            continue
        tree = cKDTree(pts)
        r0, r1 = gi.row_range[0], gi.row_range[1] + 1
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(nx), indexing="ij")
        d, _ = tree.query(np.stack([rr.ravel(), cc.ravel()], axis=1))
        out[r0:r1] = np.where(np.isnan(out[r0:r1]), (d * IIRS_GSD_M).reshape(rr.shape).astype("float32"), out[r0:r1])
    return out


def write_obs(fout: Path, obs: dict, sid: str, compress: str = "ZSTD", predictor: int = 3, row0: int = 0) -> Path:
    """Write `<sid>_obs.tif`/`.img`: 14-band f32, camera space (no CRS), bands 1-10 in M3 order/
    units, bands 11-14 IIRS extras. Lossless, tiled, `INTERLEAVE=BAND`."""
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
        "CONVENTION": "Bands 1-10 follow the Chandrayaan-1 M3 L1B OBS layout and units; bands 11+ are IIRS extensions",
        "GEOMETRY_BASIS": (
            "Registration-derived: sun/sensor terms from SPICE at the loc's TPS-registered XYZ; "
            "facet terms from the LOLA DEM normal, not a sensor-model ray/DEM intersection."
        ),
        "BANDS": "; ".join(f"{i} {n} [{u}]" for i, (n, u) in enumerate(names_units, start=1)),
        "ANGLE_CONVENTION": (
            "Azimuth clockwise from local topocentric north, 0-360 deg. Zenith from the local "
            "vertical (sphere normal for sun/sensor bands, DEM facet normal for facet_cos_i). "
            "Facet aspect clockwise from local north (M3 DPSIS Sec 2.5.3.3/3.2.1.7-8)."
        ),
        "FACET_SCALE_M": str(IIRS_GSD_M),
        "PAN_BANDS": ",".join(str(b) for b in PAN_BANDS),
        "SHADOW_SNR": str(SHADOW_SNR),
        "SUN_DISTANCE_MEAN_AU": str(obs["sun_distance_mean_au"]),
        "SENSOR_GEOMETRY_AVAILABLE": str(obs["sensor_available"]),
        "QA_FILE": f"{sid}_qa.tif",
        "SKY_VIEW_STATUS": "reserved, NaN: the lit_frac horizon sweep only samples toward the sun's own azimuth",
        "TIE_DIST_DEFINITION": "distance to the nearest solved GCP lattice node, not the raw tie-point detection",
        "PROVENANCE_IIRSPY_VERSION": IIRSPY_VERSION,
        "GROUPS": {g: list(r) for g, r in obs["groups"].items()},
    }
    names = [n for n, _u in names_units]
    units = [u for _n, u in names_units]
    da = _backplane_da([bands[n] for n in names], names, units, row0, tags)
    return Path(_write_camera_product(da, fout, compress=compress, predictor=predictor))


# ------------------------------------------------------------------------------------------
# QA: uint16 bit flags
# ------------------------------------------------------------------------------------------
QA_BITS = {
    "no_geometry": 0,
    "gcp_extrapolated": 1,
    "in_gcp_chunk_overlap": 2,
    "in_gcp_region_overlap": 3,
    "in_geometric_shadow": 4,
    "is_lit": 5,
    "is_saturated": 6,
    "pan_incomplete": 7,
}


def _extrapolated_mask(sid: str, groups: dict[str, tuple[int, int]], ny: int, nx: int) -> np.ndarray:
    """True where a pixel lies outside the convex hull of its owning group's own GCP lattice
    (row, col), i.e. the TPS is extrapolating rather than interpolating."""
    out = np.zeros((ny, nx), dtype=bool)
    for g in groups:
        gi = group_info(sid, g)
        gcps = solve._load_gcps(solve.merged_gcps_path(sid, g))
        pts = np.array([(r, c) for r, c in gcps], dtype="float64")
        if len(pts) < 4:
            continue
        tri = Delaunay(pts)
        r0, r1 = gi.row_range[0], gi.row_range[1] + 1
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(nx), indexing="ij")
        inside = tri.find_simplex(np.stack([rr.ravel(), cc.ravel()], axis=1)) >= 0
        out[r0:r1] |= ~inside.reshape(rr.shape)
    return out


def build_qa(sid: str, loc: dict, obs: dict, saturation: dict[str, np.ndarray | None] | None = None) -> np.ndarray:
    """The uint16 QA bit field, from `loc`, `obs`, and per-group PAN saturation
    masks (`saturation`, keyed by group; `None`/missing group -> bit 6 left 0 for that group's rows,
    since `SATURATION_CHECK` records whether it was actually evaluated)."""
    ny, nx = loc["lon"].shape
    qa = np.zeros((ny, nx), dtype="uint16")
    groups = loc["groups"]

    qa[np.isnan(loc["lon"])] |= 1 << QA_BITS["no_geometry"]
    qa[_extrapolated_mask(sid, groups, ny, nx)] |= 1 << QA_BITS["gcp_extrapolated"]

    for g in groups:
        gi = group_info(sid, g)
        for lo, hi, _ia, _ib in chunk_overlaps(gi):
            qa[lo:hi] |= 1 << QA_BITS["in_gcp_chunk_overlap"]
    for seam in loc.get("seams", ()):
        lo, hi = seam["row_range"]
        qa[lo:hi] |= 1 << QA_BITS["in_gcp_region_overlap"]

    lit_frac = obs["bands"]["lit_frac"]
    qa[np.nan_to_num(lit_frac, nan=1.0) < GEOM_SHADOW_LIT] |= 1 << QA_BITS["in_geometric_shadow"]
    snr = obs["bands"]["snr"]
    qa[np.nan_to_num(snr, nan=-np.inf) >= SHADOW_SNR] |= 1 << QA_BITS["is_lit"]

    for g in groups:
        gi = group_info(sid, g)
        r0, r1 = gi.row_range[0], gi.row_range[1] + 1
        pan_bad = _pan_incomplete(sid, g)
        if pan_bad is not None:
            n = min(r1 - r0, pan_bad.shape[0])
            bit = 1 << QA_BITS["pan_incomplete"]
            qa[r0 : r0 + n] = np.where(pan_bad[:n], qa[r0 : r0 + n] | bit, qa[r0 : r0 + n])
        sat = saturation.get(g) if saturation else None
        if sat is not None:
            n = min(r1 - r0, sat.shape[0])
            bit = 1 << QA_BITS["is_saturated"]
            qa[r0 : r0 + n] = np.where(sat[:n], qa[r0 : r0 + n] | bit, qa[r0 : r0 + n])

    return qa


def write_qa(
    fout: Path, qa: np.ndarray, sid: str, groups: dict[str, tuple[int, int]], saturation_status: dict, row0: int = 0
) -> Path:
    """Write `<sid>_qa.tif`/`.img`: uint16 bit flags, lossless."""
    fout = Path(fout)
    fout.parent.mkdir(parents=True, exist_ok=True)
    tags = {
        "IIRS_PRODUCT": "QA",
        "IIRS_FORMAT_VERSION": "1",
        "IIRS_SID": sid,
        "QA_BIT_0": "no_geometry: LOC undefined (row outside every solved group)",
        "QA_BIT_1": "gcp_extrapolated: outside the GCP lattice hull, TPS extrapolating",
        "QA_BIT_2": "in_gcp_chunk_overlap: overlap of two chunks of one group, geometry cross-faded",
        "QA_BIT_3": "in_gcp_region_overlap: overlap of two solve groups, geometry seam-blended",
        "QA_BIT_4": f"in_geometric_shadow: lit_frac < {GEOM_SHADOW_LIT} (sun centre below local LOLA horizon)",
        "QA_BIT_5": f"is_lit: broadband PAN snr >= {SHADOW_SNR}",
        "QA_BIT_6": "is_saturated: any PAN band at/above saturation radiance",
        "QA_BIT_7": "pan_incomplete: any PAN band NaN at this pixel",
        "QA_RESERVED_BITS": "8-15",
        "PAN_BANDS": ",".join(str(b) for b in PAN_BANDS),
        "GEOM_SHADOW_LIT": str(GEOM_SHADOW_LIT),
        "SHADOW_SNR": str(SHADOW_SNR),
        "SATURATION_CHECK": saturation_status,
        "GROUPS": {g: list(r) for g, r in groups.items()},
        "PROVENANCE_IIRSPY_VERSION": IIRSPY_VERSION,
    }
    da = _backplane_da([qa], ["qa_bits"], [""], row0, tags)
    return Path(_write_camera_product(da, fout, compress="ZSTD", predictor=2))
