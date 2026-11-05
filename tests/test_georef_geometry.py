"""Geometry, backplane (LOC/OBS/QA) and topography helpers against synthetic fixtures with known answers."""

import json

import numpy as np
import pytest
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import from_origin

from iirspy import iirs, instrument, io
from iirspy.georef import arosics, georef
from iirspy.georef import backplanes as bp
from iirspy.georef import chunks as ck
from iirspy.georef.geometry import Geometry, quality


def test_xyz_round_trips_group_crs_for_every_pole():
    for pole in ck.GROUPS:
        x, y = (12_345.6, -7_890.1) if pole != "equatorial" else (400_000.0, 150_000.0)
        lon, lat = georef.xy_to_lonlat(x, y, pole)
        xyz = georef.lonlat_to_xyz(lon, lat, georef.MOON_RADIUS_M)
        lon2, lat2, r2 = georef.xyz_to_lonlat(*xyz)
        x2, y2 = georef.to_stereo(pole).transform(lon2, lat2)
        assert abs(x2 - x) < 1e-6
        assert abs(y2 - y) < 1e-6
        assert abs(r2 - georef.MOON_RADIUS_M) < 1e-6


def _write_group(paths, sid, group, scan0, rows, cols):
    d = ck.recal_dir(sid, paths)
    d.mkdir(parents=True, exist_ok=True)
    gcps = {(r, c): (1000.0 * c, -1000.0 * r + 10.0 * group.__hash__() % 7) for r in rows for c in cols}
    arosics.save_gcps(ck.merged_gcps_path(sid, group, paths), gcps)
    chunks = [{"i": 0, "row0": min(rows), "row1": max(rows)}]
    (d / f"{sid}_{group}_solve_summary.json").write_text(json.dumps({"scan0": scan0, "chunks": chunks}))
    return gcps


def _paths(tmp_path, **kw):
    archive = tmp_path / "archive"
    return io.Paths(**{
        "archive": archive,
        "anc_roots": (archive,),
        "dem_roots": (),
        "spice": tmp_path / "spice",
        "recal_root": tmp_path / "recal",
        **kw,
    })


# ------------------------------------------------------------------------------------------
# LOC
# ------------------------------------------------------------------------------------------
def _write_plain_group(paths, sid, group, scan0, rows, cols):
    """Like `_write_group` but with plain x=1000*col, y=-1000*row, no per-group hash offset, so
    two groups can share the same ground truth for a seam-blend test."""
    d = ck.recal_dir(sid, paths)
    d.mkdir(parents=True, exist_ok=True)
    gcps = {(r, c): (1000.0 * c, -1000.0 * r) for r in rows for c in cols}
    arosics.save_gcps(ck.merged_gcps_path(sid, group, paths), gcps)
    chunks = [{"i": 0, "row0": min(rows), "row1": max(rows)}]
    (d / f"{sid}_{group}_solve_summary.json").write_text(json.dumps({"scan0": scan0, "chunks": chunks}))
    return gcps


def _write_dem_tif(path, elev_m, x_range, y_range, ps=200.0):
    """Constant-elevation synthetic LOLA COG: `load_lola_elev`'s `.tif` branch assumes DN counts
    with scale=0.5, offset=MOON_RADIUS_M (see its docstring), so DN = elev_m / 0.5 gives back
    exactly `elev_m` everywhere; no `load_lola_elev` code path is special-cased for tests."""
    xmin, xmax = x_range
    ymin, ymax = y_range
    w, h = max(1, int((xmax - xmin) / ps)), max(1, int((ymax - ymin) / ps))
    tr = rasterio.transform.from_origin(xmin, ymax, ps, ps)
    dn = np.full((h, w), round(elev_m / 0.5), dtype="int16")
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype="int16", transform=tr) as dst:
        dst.write(dn, 1)
    return path


def test_group_loc_samples_constant_dem_elevation(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    sid = "20210101T0000000000"
    rows, cols = range(0, 21, 10), range(0, 3)
    _write_plain_group(paths, sid, "south", scan0=100, rows=rows, cols=cols)

    dem = _write_dem_tif(tmp_path / "dem.tif", elev_m=500.0, x_range=(-2000, 4000), y_range=(-22000, 2000))
    monkeypatch.setattr(
        ck, "bands", lambda dem_roots: {"south": {"group": "south", "lat_range": (-90.0, 90.0), "dem_near": str(dem)}}
    )

    gcps_file = arosics.load_gcps(ck.merged_gcps_path(sid, "south", paths))
    lon, lat, radius = bp.group_loc_core(gcps_file, "south", paths.dem_roots, row_block=8)
    assert lon.shape == (21, 3)
    assert np.isfinite(lon).all() and np.isfinite(lat).all()
    np.testing.assert_allclose(radius, georef.MOON_RADIUS_M + 500.0, atol=1.0)
    # the lattice is linear in (row, col), so pixel centres (r + 0.5, c + 0.5) are exact
    rr, cc = np.meshgrid(np.arange(21) + 0.5, np.arange(3) + 0.5, indexing="ij")
    lon0, lat0 = georef.xy_to_lonlat(1000.0 * cc, -1000.0 * rr, "south")
    np.testing.assert_allclose(lon, lon0, atol=1e-9)
    np.testing.assert_allclose(lat, lat0, atol=1e-9)


def test_write_read_loc_round_trip(tmp_path):
    ny, nx = 4, 3
    lon = np.tile(np.linspace(10.0, 13.0, nx), (ny, 1))
    lat = np.tile(np.linspace(-80.0, -77.0, ny)[:, None], (1, nx))
    radius = np.full((ny, nx), georef.MOON_RADIUS_M + 100.0)
    loc = {"lon": lon, "lat": lat, "radius": radius, "groups": {"south": (0, 3)}}

    f = bp.write_loc(tmp_path / "sid_loc.tif", loc, "20210101T0000000000")
    with rasterio.open(f) as src:
        assert src.dtypes[0] == "float32"
        assert src.tags()["IIRS_PRODUCT"] == "LOC"
        assert "PRECISION" in src.tags()
        assert [src.descriptions[i] for i in range(3)] == ["longitude", "latitude", "radius"]

    ds = bp.read_loc(f)
    assert set(ds.data_vars) == {"lon", "lat", "radius"}
    assert ds.lon.dims == ("y", "x")
    np.testing.assert_allclose(ds.lon.values, lon, atol=1e-3)
    np.testing.assert_allclose(ds.radius.values, radius, atol=1e-3)
    assert ds.attrs["IIRS_SID"] == "20210101T0000000000"


def test_write_read_loc_and_qa_round_trip_tif_and_img_absolute_rows(tmp_path):
    """Both formats (`.tif` GeoTIFF and `.img` ENVI BIL) must round-trip identically for a
    non-zero `row0`: absolute `y`, band names/units/tags preserved, QA dtype stays uint16."""
    ny, nx = 4, 3
    row0 = 137
    lon = np.tile(np.linspace(10.0, 13.0, nx), (ny, 1))
    lat = np.tile(np.linspace(-80.0, -77.0, ny)[:, None], (1, nx))
    radius = np.full((ny, nx), georef.MOON_RADIUS_M + 100.0)
    loc = {"lon": lon, "lat": lat, "radius": radius, "groups": {"south": (row0, row0 + ny - 1)}}
    qa = np.arange(ny * nx, dtype="uint16").reshape(ny, nx)

    for ext in (".tif", ".img"):
        floc = bp.write_loc(tmp_path / f"sid_loc{ext}", loc, "20210101T0000000000", row0=row0)
        fqa = bp.write_qa(
            tmp_path / f"sid_qa{ext}", qa, "20210101T0000000000", {"south": (row0, row0 + ny - 1)}, {}, row0=row0
        )

        ds = bp.read_loc(floc)
        np.testing.assert_allclose(ds.lon.values, lon, atol=1e-3)
        np.testing.assert_allclose(ds.lat.values, lat, atol=1e-3)
        np.testing.assert_allclose(ds.radius.values, radius, atol=1e-3)
        assert list(ds.y.values.astype(int)) == list(row0 + np.arange(ny))
        assert ds.attrs["IIRS_SID"] == "20210101T0000000000"

        with rasterio.open(fqa) as src:
            assert src.dtypes[0] == "uint16"  # dtype survives the shared writer for both formats
            np.testing.assert_array_equal(src.read(1), qa)
            assert src.descriptions[0] == "qa_bits"
            # ENVI's own extra header keys land in GDAL's "ENVI" tag namespace, not the default
            # one `.tif` uses; merge both like `backplanes._read_backplane` does.
            tags = {**src.tags(ns="ENVI"), **src.tags()}
            assert tags["IIRS_PRODUCT"] == "QA"


def test_glt_from_loc_no_holes_when_camera_spacing_coarser_than_output_grid():
    from dataclasses import replace

    import xarray as xr

    from iirspy.georef.georef import GeorefConfig, glt_from_loc

    # camera points 2000 m apart; output grid at 500 m, 4x finer than the camera spacing. A cutoff
    # of one output pixel (the old bug) misses most cells inside the footprint.
    ny, nx = 4, 4
    rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    x, y = 2000.0 * cols, -2000.0 * rows
    lon, lat = georef.xy_to_lonlat(x, y, "south")
    ds = xr.Dataset(
        {"lon": (("y", "x"), lon), "lat": (("y", "x"), lat)},
        coords={"y": np.arange(ny), "x": np.arange(nx)},
    )
    # aoi kept inside the camera footprint (0-6000, -6000-0) by one camera-pixel margin, so no cell
    # queried here is an extrapolation past the outermost camera point.
    cfg = replace(GeorefConfig(), pole="south", aoi=(500.0, -5500.0, 5500.0, -500.0), ps=500.0)
    glt, _tr = glt_from_loc(ds, "south", cfg)
    assert (glt[0] >= 0).all()
    assert (glt[1] >= 0).all()


# ------------------------------------------------------------------------------------------
# OBS angle conventions, QA bit packing, tags
# ------------------------------------------------------------------------------------------
def test_local_az_zenith_sun_due_north_30deg_elevation():
    east, north, up = (np.array(v, dtype=float) for v in ((0, 1, 0), (0, 0, 1), (1, 0, 0)))
    u_sun = np.array([np.sin(np.radians(30)), 0.0, np.cos(np.radians(30))])
    az, zen = bp._local_az_zenith(u_sun, east, north, up)
    assert abs(float(az) - 0.0) < 1e-6
    assert abs(float(zen) - 60.0) < 1e-6


def test_local_az_zenith_nadir_sensor_is_zero_zenith():
    east, north, up = (np.array(v, dtype=float) for v in ((0, 1, 0), (0, 0, 1), (1, 0, 0)))
    _az, zen = bp._local_az_zenith(up.copy(), east, north, up)
    assert abs(float(zen) - 0.0) < 1e-6


def test_phase_angle_between_coincident_rays_is_zero():
    phase = bp._phase_angle(
        sun_zen=np.array([60.0]), sens_zen=np.array([60.0]), sun_az=np.array([0.0]), sens_az=np.array([0.0])
    )
    np.testing.assert_allclose(phase, 0.0, atol=1e-6)


def test_phase_angle_matches_known_separation():
    # sun straight up (zenith 0); sensor at zenith 60 -> phase equals the sensor's own zenith
    phase = bp._phase_angle(
        sun_zen=np.array([0.0]), sens_zen=np.array([60.0]), sun_az=np.array([0.0]), sens_az=np.array([123.0])
    )
    np.testing.assert_allclose(phase, 60.0, atol=1e-6)


def test_facet_cos_i_flat_facet_matches_sun_zenith_cosine():
    sun_zen = np.array([30.0, 45.0])
    cos_i = bp._facet_cos_i(sun_zen, sun_az=np.array([10.0, 200.0]), facet_slope=np.zeros(2), facet_aspect=np.zeros(2))
    np.testing.assert_allclose(cos_i, np.cos(np.radians(sun_zen)), atol=1e-6)


def test_facet_cos_i_facet_facing_the_sun_is_brighter_than_flat():
    # slope tilted toward the sun (aspect == sun azimuth) increases cos(i) above the flat case
    sun_zen = np.array([60.0])
    flat = bp._facet_cos_i(sun_zen, sun_az=np.array([90.0]), facet_slope=np.zeros(1), facet_aspect=np.zeros(1))
    tilted = bp._facet_cos_i(
        sun_zen, sun_az=np.array([90.0]), facet_slope=np.array([20.0]), facet_aspect=np.array([90.0])
    )
    assert tilted[0] > flat[0]


def test_build_qa_bit_packing_no_geometry_and_shadow_lit_disagreement():
    ny, nx = 3, 2
    loc = {
        "lon": np.array([[np.nan, np.nan], [10.0, 10.0], [10.0, 10.0]]),
        "lat": np.zeros((ny, nx)),
        "radius": np.full((ny, nx), georef.MOON_RADIUS_M),
        "groups": {},
    }
    lit = np.array([[1.0, 1.0], [0.1, 0.1], [0.9, 0.9]])  # row1 in geometric shadow, row2 lit
    snr = np.array([[0.0, 0.0], [5.0, 5.0], [0.1, 0.1]])  # row1 also reads as lit -> bit4&5 disagree
    obs = {"bands": {"lit_frac": lit, "snr": snr}, "groups": {}}
    qa = bp.build_qa(loc, obs)

    no_geom = bp.qa_mask("no_geometry")
    shadow = bp.qa_mask("in_geometric_shadow")
    is_lit = bp.qa_mask("is_lit")
    assert (qa[0] & no_geom).all()
    assert not (qa[1] & no_geom).any()
    assert (qa[1] & shadow).all() and (qa[1] & is_lit).all()  # the disagreement case
    assert not (qa[2] & shadow).any() and not (qa[2] & is_lit).any()


def test_write_obs_tags_include_bands_pan_bands_and_shadow_snr(tmp_path):
    ny, nx = 3, 2
    bands = {
        n: np.full((ny, nx), 1.0, "float32")
        for n in (
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
        )
    }
    bands["sky_view"][:] = np.nan
    obs = {"bands": bands, "groups": {"south": (0, ny - 1)}, "sensor_source": "spm"}
    f = bp.write_obs(tmp_path / "sid_obs.tif", obs, "20210101T0000000000")
    with rasterio.open(f) as src:
        assert src.count == 14
        tags = src.tags()
        assert tags["IIRS_PRODUCT"] == "OBS"
        assert "sun_azimuth" in tags["BANDS"] and "tie_dist" in tags["BANDS"]
        assert tags["PAN_BANDS"] == ",".join(str(b) for b in instrument.PAN_BANDS)
        assert tags["SHADOW_SNR"] == str(bp.SHADOW_SNR)
        assert src.descriptions[0] == "sun_azimuth"
        assert np.isnan(src.read(12)).all()  # sky_view reserved


def test_spm_archive_then_stage_then_zip(tmp_path):
    import zipfile

    sid, rel = "20231203T0022175440", "miscellaneous/raw/20231203/ch2_iir_nri_20231203T0022175440_d_img_d18.spm"
    archive, stage = tmp_path / "archive", tmp_path / "stage"
    (archive / "zips").mkdir(parents=True)
    with zipfile.ZipFile(archive / "zips" / f"ch2_iir_nri_{sid}_d_img_d18.zip", "w") as zf:
        zf.writestr(rel, "spm")
    paths = _paths(tmp_path)

    assert ck.spm(sid, paths) is None  # no stage: never extracts
    assert ck.spm(sid, paths, stage) == stage / rel  # extracted from the zip
    assert (stage / rel).read_text() == "spm"
    (archive / rel).parent.mkdir(parents=True)
    (archive / rel).write_text("archived")
    assert ck.spm(sid, paths, stage) == archive / rel  # archive wins


def test_label_comes_from_the_nri_zip_like_the_spm(tmp_path):
    import zipfile

    sid, rel = "20231203T0022175440", "data/raw/20231203/ch2_iir_nri_20231203T0022175440_d_img_d18.xml"
    archive, stage = tmp_path / "archive", tmp_path / "stage"
    (archive / "zips").mkdir(parents=True)
    with zipfile.ZipFile(archive / "zips" / f"ch2_iir_nri_{sid}_d_img_d18.zip", "w") as zf:
        zf.writestr(rel, "<label/>")
        zf.writestr("data/raw/20231203/x.qub", "cube")
    paths = _paths(tmp_path)

    assert ck.label(sid, paths, stage) == stage / rel
    assert not (stage / "data/raw/20231203/x.qub").exists()  # the label only, never the cube


def test_geometry_save_open_round_trip_is_lazy_and_exact(tmp_path):
    ny, nx, row0 = 6, 4, 137
    lon = np.tile(np.linspace(10.0, 13.0, nx), (ny, 1))
    lat = np.tile(np.linspace(-80.0, -77.0, ny)[:, None], (1, nx))
    radius = np.full((ny, nx), 1737400.0)
    obs = {"sun_zenith": np.full((ny, nx), 61.5), "facet_cos_i": np.full((ny, nx), 0.48)}
    qa = np.full((ny, nx), bp.qa_mask("no_geometry") | bp.qa_mask("no_topo"), "uint16")
    geo = Geometry.from_arrays(
        "20210101T0000000000",
        "south",
        row0,
        "issdc",
        {"lon": lon, "lat": lat, "radius": radius},
        obs,
        qa,
        quality("issdc", "warn"),
        tags={"SENSOR_SOURCE": "nadir"},
    )
    geo.save(tmp_path)

    back = Geometry.open(tmp_path, "20210101T0000000000", "south")
    assert back._loc is None and back._obs is None and back._qa is None
    assert (back.method, back.row0, back.quality["grade"]) == ("issdc", row0, "warn")
    np.testing.assert_allclose(back.loc.lon.values, lon, atol=1e-3)
    np.testing.assert_allclose(back.obs.sun_zenith.values, 61.5, atol=1e-5)
    assert np.isnan(back.obs.snr.values).all()
    np.testing.assert_array_equal(back.qa.values, qa)
    assert list(back.loc.y.values.astype(int)) == list(row0 + np.arange(ny))


# ------------------------------------------------------------------------------------------
# Topography, warping and GCP lattice
# ------------------------------------------------------------------------------------------
def test_slope_aspect_on_a_known_plane():
    """A plane rising 1 m per 1 m eastward is a 45 deg slope facing west, aspect being downhill."""
    slope, aspect = georef.slope_aspect(gx=1.0, gy=0.0)
    assert slope == pytest.approx(45.0)
    assert aspect == pytest.approx(270.0)
    # ... and one dropping toward the north faces 0 deg, the wrap point, not 360
    assert georef.slope_aspect(0.0, -np.tan(np.radians(10)))[1] == pytest.approx(0.0)
    assert georef.slope_aspect(0.0, -np.tan(np.radians(10)))[0] == pytest.approx(10.0)


def test_a_windowed_warp_is_the_same_pixels_as_the_full_one():
    """project() must be extent-independent: rasterio's default 0.125 px tolerance refits the TPS
    per destination chunk, so the chunking -- and the answer -- follows the output window."""
    cfg = georef.GeorefConfig(aoi=(0.0, 0.0, 4000.0, 4000.0), ps=40.0)
    ny, nx = 60, 20
    rows, cols = np.arange(0, ny, 5), np.arange(0, nx, 2)
    # Targets bent off the affine, or a per-chunk polynomial would reproduce them exactly and the
    # approximate transformer would agree with the exact one by accident. Gently: fold the warp and
    # the source window per destination chunk starts to matter on its own.
    gcps = [
        GroundControlPoint(
            row=float(r),
            col=float(c),
            x=500.0 + 30.0 * c + 20.0 * np.sin(r / 9.0),
            y=3500.0 - 50.0 * r + 20.0 * np.cos(c / 3.0),
        )
        for r in rows
        for c in cols
    ]
    band = np.arange(ny * nx, dtype="float32").reshape(ny, nx)
    win = (10, 60, 5, 40)
    full = georef.project(band, gcps, cfg)
    assert full.shape == (100, 100)
    windowed = georef.project(band, gcps, cfg, window=win)
    assert windowed.shape == (50, 35)
    np.testing.assert_array_equal(windowed, full[10:60, 5:40])

    # ... and a cube goes through in one call, band for band
    cube = np.stack([band, band * 2.0])
    np.testing.assert_array_equal(georef.project(cube, gcps, cfg)[1], full * 2.0)


def test_local_plane_extrapolates_the_local_gradient_and_stays_bounded():
    """`_local_plane` is the off-support fallback: local level + local slope, clipped."""
    from scipy.spatial import cKDTree

    from iirspy.georef.georef import _local_plane

    pts = np.stack(np.meshgrid(np.arange(5) * 1000.0, np.arange(5) * 1000.0), -1).reshape(-1, 2)
    disp = np.c_[1e-3 * pts[:, 0], np.full(len(pts), 7.0)]  # dx ramps 0->4 m, dy constant
    far = _local_plane(pts, disp, cKDTree(pts), 8)

    inside = far(np.array([[2000.0, 2000.0]]))
    assert np.allclose(inside, [2.0, 7.0], atol=1e-6)  # recovers the ramp where support surrounds it

    out = far(np.array([[8000.0, 2000.0]]))  # 4 km past the last column, support one-sided
    assert out[0, 1] == 7.0  # the constant component stays constant
    assert 3.0 < out[0, 0] <= 4.0  # extrapolates outward, clipped to the neighbours' own range


def test_load_lola_elev_scales_gld100_unscaled_and_fills_its_nodata(tmp_path):
    """GLD100 DN is metres above the sphere already, and its declared nodata must not survive."""
    a = np.array([[100, 200], [-32768, 400]], dtype="int16")
    p = tmp_path / "WAC_GLD100_P900N0000_100M.tif"
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=from_origin(0.0, 0.0, 100.0, 100.0),
        nodata=-32768,
    ) as dst:
        dst.write(a, 1)

    z, _, ps = georef.load_lola_elev(p)
    assert ps == 100.0
    assert z[0, 0] == 100.0  # unity scale, no radius offset -- not the LOLA 0.5/1737400 rule
    assert z[1, 1] == 400.0
    assert z[1, 0] == pytest.approx(np.median([100.0, 200.0, 400.0]))  # flat fill, not a -32 km pit


def test_displacement_field_falls_back_to_the_bulk_median_on_collinear_tie_points():
    """A low-texture chunk can leave tie points on a line; the TPS trend term is then rank-deficient.

    scipy raises `LinAlgError` rather than returning a degenerate fit, so an uncaught one aborts the
    whole scene solve.
    """
    import pandas as pd

    from iirspy.georef.georef import GeorefConfig, displacement_field

    cfg = GeorefConfig()
    q = np.array([[-4e5, -8.5e5], [-3e5, -7e5]])  # on and far off the support
    # n=1 and n=2 raise ValueError (too few for the trend term), 10 collinear raise LinAlgError;
    # every one of them must degrade to the bulk median rather than propagate.
    for n in (1, 2, 10):
        tp = pd.DataFrame({
            "X_MAP": np.full(n, -4e5),  # one column of points: no independent x spread
            "Y_MAP": np.linspace(-9e5, -8e5, n),
            "X_SHIFT_PX": np.linspace(-1.0, 1.0, n),
            "Y_SHIFT_PX": np.full(n, 0.5),
        })
        fieldfn, info = displacement_field(tp, cfg)

        assert "tps_unfittable" in info["degenerate"], n
        assert np.allclose(fieldfn(q), [np.median(tp.X_SHIFT_PX) * cfg.ps, -0.5 * cfg.ps]), n


def test_row_bands_falls_back_to_midlat_near_the_polar_seam():
    """A camera row is 'south' only if it and every row within `row_step` of it are fully inside -76.5..-90."""
    from dataclasses import replace

    cfg = replace(georef.GeorefConfig(), pole="south", row_step=25)
    assert ck.BANDS["south"]["lat_range"] == (-90.0, -76.5)
    lat = np.full((160, 3), -85.0)
    lat[80:] = -70.0

    band = georef._row_bands(lat, cfg)

    assert set(band[:55]) == {"south"}
    assert set(band[55:]) == {"south_midlat"}


def test_scene_topo_assembles_pieces_with_no_blending_and_exact_per_row_sun(tmp_path, monkeypatch):
    """Pieces just concatenate (no seam blending); the sun bands are `sun_geometry_rows`'s own
    exact per-pixel values passed straight through, not a piece-centroid interpolation."""
    from dataclasses import replace

    ny, nx = 3500, 4  # > _split_run's max_rows -> at least 2 pieces, one band (equatorial, no split)
    jj, ii = np.mgrid[0:ny, 0:nx]
    lon, lat = georef.to_lonlat("equatorial").transform(1000.0 * ii, 100.0 * jj)
    cfg = replace(georef.GeorefConfig(), pole="equatorial", margin_m=5000.0, ps=2000.0)
    # bands() resolves real DEM paths on disk for every band; scene_topo only needs the names, and
    # render_topo (mocked below) never opens dem_near/dem_far, so a placeholder dict skips the DEM tree.
    monkeypatch.setattr(
        ck, "bands", lambda dem_roots: {"equatorial": {**ck.BANDS["equatorial"], "dem_near": "x", "dem_far": "x"}}
    )

    calls = []

    def fake_render_topo(fgeom, flabel, cfg, kernels=None):
        xs, ys, tr, _ = georef.grid_of(cfg)
        val = float(cfg.aoi[1])  # unique per piece: pieces don't overlap in y
        gx = np.full((len(ys), len(xs)), val, "float32")
        gy = np.zeros_like(gx)
        lit = np.ones_like(gx)
        calls.append(val)
        return gx, gy, lit, tr, {"az_grid": 10.0, "elev": val / 1e5, "r_sun": 0.27}

    monkeypatch.setattr(georef, "render_topo", fake_render_topo)

    # Distinct from render_topo's own per-piece (az, elev), and varying across x too (not just y),
    # so the test fails if scene_topo either falls back to piece info or collapses the per-pixel
    # (not just per-row) field save_topo is now expected to carry through untouched.
    want_elev = np.linspace(-1.0, 1.0, ny)[:, None] + np.linspace(0.0, 0.01, nx)[None, :]
    monkeypatch.setattr(
        georef,
        "sun_geometry_rows",
        lambda lon, lat, flabel, pole, scan0, kernels=None: (
            np.full(lon.shape, 20.0),
            want_elev,
        ),
    )

    f = georef.scene_topo("sid", "equatorial", lon, lat, "fgeom", "flabel", cfg, [], tmp_path, ())

    assert len(calls) >= 2  # the strip really did split into multiple pieces

    with rasterio.open(f) as src:
        names = [src.descriptions[i] for i in range(src.count)]
        slope = src.read(names.index("slope") + 1)
        sun_elev = src.read(names.index("sun_elev") + 1)

    # Recompute the expected piece boundaries the same way scene_topo does, and check every row's
    # slope matches its own piece's constant gradient -- concatenation, no cross-piece blending.
    band_of_row = georef._row_bands(lat, cfg)
    pieces = [p for a, z in ck.true_runs(band_of_row == "equatorial") for p in georef._split_run(a, z)]
    assert len(pieces) == len(calls)
    for (a, z), val in zip(pieces, calls, strict=True):
        want = np.degrees(np.arctan(abs(val)))
        np.testing.assert_allclose(slope[a:z], want, atol=1e-3)

    # Every pixel's sun elevation is sun_geometry_rows's own per-pixel value, not an interpolation
    # of render_topo's piece-level (az, elev) and not collapsed back down to per-row.
    np.testing.assert_allclose(sun_elev, want_elev, atol=1e-5)


def test_coarse_shift_falls_through_to_a_lower_ranked_base_peak_when_rank0_fails_controls(monkeypatch):
    """A probe's rank-0 correlation peak can be a decoy (e.g. a shadow edge); coarse_shift must
    fall through to a lower-ranked base peak whose controls actually agree."""
    from dataclasses import replace

    import skimage.registration._masked_phase_cross_correlation as mpc

    from iirspy.georef.georef import GeorefConfig, coarse_shift

    def peaks_at(*entries):
        x = np.zeros((9, 9))
        for i, j, h in entries:
            x[i, j] = h
        return x

    base_xcorr = peaks_at((6, 6, 5.0), (4, 4, 1.0))  # rank0 decoy at row/col (-2,-2), rank1 at (0,0)
    ctrl_xcorr = peaks_at((2, 2, 5.0), (5, 5, 1.0))  # rank0 decoy at (2,2), rank1 real match at (-1,-1)
    calls = iter([base_xcorr, ctrl_xcorr])
    monkeypatch.setattr(mpc, "cross_correlate_masked", lambda *a, **k: next(calls))

    cfg = replace(
        GeorefConfig(),
        coarse_dec=1,
        ps=1.0,
        coarse_max_m=2.0,
        coarse_topk=2,
        coarse_peak_sep_px=1,
        inject_px=((1, 1),),
        inject_tol_px=1,
    )
    dx_m, dy_m, info = coarse_shift(np.zeros((5, 5)), np.zeros((5, 5)), cfg)

    assert info["base_rank"] == 1  # rank0 (-2,-2) fails the control; rank1 (0,0) is used instead
    assert info["accepted"] is True
    assert info["shift_px"] == [0.0, 0.0]
    assert (dx_m, dy_m) == (0.0, 0.0)


def test_gcp_lattice_samples_the_geometry_spline_at_the_right_rows(tmp_path):
    """The lattice evaluates the geometry TPS only where GCPs go, so row->Scan must line up.

    Lattice nodes sit on csv nodes (every Scan, the same 6 pixels), where the interpolating spline
    is exact, so a row/Scan off-by-one fails hard instead of shifting the product slightly.
    """
    from dataclasses import replace

    import pandas as pd

    scan0, ny, nx = 500, 200, 250  # cube row r is geometry Scan scan0 + r
    scans = np.arange(scan0, scan0 + ny)
    pixels = np.linspace(0, nx - 1, 6).astype(int)  # 6-point cross-track sampling, = the lattice cols
    pg, sg = np.meshgrid(pixels, scans)
    lon = 0.01 * pg + 0.002 * sg
    lat = -60.0 + 0.001 * sg
    fgeom = tmp_path / "geom.csv"
    pd.DataFrame({
        "Pixel": pg.ravel(),
        "Scan": sg.ravel(),
        "Longitude": lon.ravel(),
        "Latitude": lat.ravel(),
    }).to_csv(fgeom, index=False)

    cfg = replace(georef.GeorefConfig(), pole="equatorial", lat_band=(lat.min(), lat.max()), row_step=25, ncol=6)
    jj, ii, x, y = georef.gcp_lattice(fgeom, ny, nx, cfg)

    want_lon = 0.01 * ii + 0.002 * (scan0 + jj)
    want_lat = -60.0 + 0.001 * (scan0 + jj)
    wx, wy = georef.to_stereo(cfg.pole).transform(want_lon, want_lat)
    assert np.allclose(x, wx, atol=1e-3) and np.allclose(y, wy, atol=1e-3)  # sub-mm, on metres

    with pytest.raises(ValueError, match="pass the crop the cube was built with"):
        georef.gcp_lattice(fgeom, ny + 50, nx, cfg)  # geometry shorter than the cube it is given

    # No default crop to fall back on: guessing one mis-indexes every GCP by thousands of rows.
    with pytest.raises(ValueError, match="lat_band"):
        georef.gcp_lattice(fgeom, ny, nx, replace(cfg, lat_band=None))

    gcps, s0, cam = georef.geom_csv_gcps(fgeom, cfg)  # the whole csv: crop-relative rows, scan0 back out
    assert (s0, cam) == (scan0, (ny, nx))
    assert min(g.row for g in gcps) == 0 and max(g.row for g in gcps) == ny - 1


@pytest.mark.parametrize(("ul", "flip"), [((0, 0), (False, False)), ((-1, 0), (True, False)), ((0, -1), (False, True))])
def test_read_issdc_loc_puts_the_labels_refined_upper_left_at_scan0_pixel0(tmp_path, monkeypatch, ul, flip):
    """ndi is stored north-up/west-left; the label's Refined upper-left is the nci (Scan 0, Pixel 0)."""
    ny, nx = 6, 4
    lat, lon = np.meshgrid(np.linspace(5, 0, ny), np.linspace(300, 301, nx), indexing="ij")  # north-up, west-left
    floc = tmp_path / "ch2_iir_ndi_X_d_loc_hw1_ard.img"
    with rasterio.open(floc, "w", driver="ENVI", height=ny, width=nx, count=2, dtype="float64") as dst:
        dst.write(np.stack([lon, lat]))
        dst.set_band_description(1, "Longitude")
        dst.set_band_description(2, "Latitude")
    label = {"isda:upper_left_longitude": lon[ul], "isda:upper_left_latitude": lat[ul]}
    monkeypatch.setattr(iirs.pdr, "open", lambda f: type("L", (), {"metaget": lambda self, k: label})())

    loc, got = iirs.read_issdc_loc(tmp_path / "ch2_iir_ndi_X_d_rfl_hw1_srd.qub")
    assert got == flip
    assert (float(loc.lon[0, 0]), float(loc.lat[0, 0])) == (lon[ul], lat[ul])
    assert np.array_equal(loc.y.values, np.sort(loc.y.values))  # coords stay ascending scan/pixel
