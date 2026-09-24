"""Backplane LOC/OBS/QA helpers against synthetic fixtures with known answers."""

import json

import numpy as np
import rasterio

from iirspy import backplanes as bp
from iirspy import chunks as ck
from iirspy import georef, solve


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


def _write_group(root, sid, group, scan0, rows, cols):
    d = ck.recal_dir(sid, group)
    d.mkdir(parents=True, exist_ok=True)
    gcps = {(r, c): (1000.0 * c, -1000.0 * r + 10.0 * group.__hash__() % 7) for r in rows for c in cols}
    solve._save_gcps(solve.merged_gcps_path(sid, group), gcps)
    chunks = [{"i": 0, "row0": min(rows), "row1": max(rows)}]
    (d / f"georef_solve_summary_{ck.GROUP_SHORT[group]}.json").write_text(
        json.dumps({"scan0": scan0, "chunks": chunks})
    )
    return gcps


def test_group_info_row_range_uses_gcps_extent_not_the_full_chunk_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(ck, "RECAL_ROOT", tmp_path / "recal")
    sid = "20210101T0000000000"
    _write_group(tmp_path, sid, "south", scan0=500, rows=range(0, 11, 5), cols=range(0, 3))
    gi = bp.group_info(sid, "south")
    assert gi.scan0 == 500
    assert gi.row_range == (500, 510)
    assert gi.chunk_rows == [(0, 500, 510)]


# ------------------------------------------------------------------------------------------
# LOC
# ------------------------------------------------------------------------------------------
def _write_plain_group(sid, group, scan0, rows, cols):
    """Like `_write_group` but with plain x=1000*col, y=-1000*row, no per-group hash offset, so
    two groups can share the same ground truth for a seam-blend test."""
    d = ck.recal_dir(sid, group)
    d.mkdir(parents=True, exist_ok=True)
    gcps = {(r, c): (1000.0 * c, -1000.0 * r) for r in rows for c in cols}
    solve._save_gcps(solve.merged_gcps_path(sid, group), gcps)
    chunks = [{"i": 0, "row0": min(rows), "row1": max(rows)}]
    (d / f"georef_solve_summary_{ck.GROUP_SHORT[group]}.json").write_text(
        json.dumps({"scan0": scan0, "chunks": chunks})
    )
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
    monkeypatch.setattr(ck, "RECAL_ROOT", tmp_path / "recal")
    sid = "20210101T0000000000"
    rows, cols = range(0, 21, 10), range(0, 3)
    _write_plain_group(sid, "south", scan0=100, rows=rows, cols=cols)

    dem = _write_dem_tif(tmp_path / "dem.tif", elev_m=500.0, x_range=(-2000, 4000), y_range=(-22000, 2000))
    monkeypatch.setattr(
        ck, "bands", lambda: {"south": {"group": "south", "lat_range": (-90.0, 90.0), "dem_near": str(dem)}}
    )

    gcps_file = solve._load_gcps(solve.merged_gcps_path(sid, "south"))
    lon, lat, radius = bp._group_loc_core(gcps_file, "south", row_block=8)
    assert lon.shape == (21, 3)
    assert np.isfinite(lon).all() and np.isfinite(lat).all()
    np.testing.assert_allclose(radius, georef.MOON_RADIUS_M + 500.0, atol=1.0)
    # cross-checked against the same GCP TPS evaluated independently via GCPTransformer
    from rasterio.control import GroundControlPoint
    from rasterio.transform import GCPTransformer

    gcps = {(r, c): (1000.0 * c, -1000.0 * r) for r in rows for c in cols}
    gcp_list = [GroundControlPoint(row=float(r), col=float(c), x=x, y=y) for (r, c), (x, y) in gcps.items()]
    with GCPTransformer(gcp_list, tps=True) as t:
        x0, y0 = t.xy([0], [0], offset="center")
    lon0, lat0 = georef.xy_to_lonlat(x0[0], y0[0], "south")
    assert abs(lon[0, 0] - lon0) < 1e-6 and abs(lat[0, 0] - lat0) < 1e-6


def test_write_read_loc_round_trip(tmp_path):
    ny, nx = 4, 3
    lon = np.tile(np.linspace(10.0, 13.0, nx), (ny, 1))
    lat = np.tile(np.linspace(-80.0, -77.0, ny)[:, None], (1, nx))
    radius = np.full((ny, nx), georef.MOON_RADIUS_M + 100.0)
    loc = {"lon": lon, "lat": lat, "radius": radius, "groups": {"south": (0, 3)}}

    f = bp.write_loc(tmp_path / "sid_loc.tif", loc, "20210101T0000000000")
    with rasterio.open(f) as src:
        assert src.dtypes[0] == "float32"  # maintainer decision: f32 default, not f64
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


def test_glt_from_loc_recovers_known_camera_pixels(monkeypatch):
    from dataclasses import replace

    import xarray as xr

    from iirspy.georef import GeorefConfig, glt_from_loc

    # camera and output grids share the same 2000 m spacing, chosen so every camera point sits
    # exactly on an output cell centre: no ties for "nearest" to break either way.
    ny, nx = 5, 4
    rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    x, y = 2000.0 * cols, -2000.0 * rows
    lon, lat = georef.xy_to_lonlat(x, y, "south")
    ds = xr.Dataset(
        {"lon": (("y", "x"), lon), "lat": (("y", "x"), lat)},
        coords={"y": np.arange(ny), "x": np.arange(nx)},
    )
    cfg = replace(GeorefConfig(), pole="south", aoi=(0.0, -10_000.0, 8_000.0, 0.0), ps=2000.0)
    glt, _tr = glt_from_loc(ds, "south", cfg)
    assert glt.shape == (2, ny, nx)
    assert (glt[0] >= 0).all()
    # every output cell must recover the exact camera (row, col) it was built from
    for r in range(ny):
        for c in range(nx):
            gx = int((x[r, c] - _tr.c) / _tr.a)
            gy = int((y[r, c] - _tr.f) / _tr.e)
            assert glt[0, gy, gx] == c
            assert glt[1, gy, gx] == r


def test_glt_from_loc_no_holes_when_camera_spacing_coarser_than_output_grid():
    from dataclasses import replace

    import xarray as xr

    from iirspy.georef import GeorefConfig, glt_from_loc

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
    qa = bp.build_qa("dummy_sid", loc, obs, saturation=None)

    no_geom = 1 << bp.QA_BITS["no_geometry"]
    shadow = 1 << bp.QA_BITS["in_geometric_shadow"]
    is_lit = 1 << bp.QA_BITS["is_lit"]
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
    obs = {"bands": bands, "groups": {"south": (0, ny - 1)}, "sun_distance_mean_au": 1.234, "sensor_available": True}
    f = bp.write_obs(tmp_path / "sid_obs.tif", obs, "20210101T0000000000")
    with rasterio.open(f) as src:
        assert src.count == 14
        tags = src.tags()
        assert tags["IIRS_PRODUCT"] == "OBS"
        assert "sun_azimuth" in tags["BANDS"] and "tie_dist" in tags["BANDS"]
        assert tags["PAN_BANDS"] == ",".join(str(b) for b in bp.PAN_BANDS)
        assert tags["SHADOW_SNR"] == str(bp.SHADOW_SNR)
        assert src.descriptions[0] == "sun_azimuth"
        assert np.isnan(src.read(12)).all()  # sky_view reserved


def test_write_qa_tags_name_every_bit(tmp_path):
    qa = np.zeros((2, 2), dtype="uint16")
    f = bp.write_qa(tmp_path / "sid_qa.tif", qa, "20210101T0000000000", {"south": (0, 1)}, {"south": "evaluated"})
    with rasterio.open(f) as src:
        tags = src.tags()
        assert tags["IIRS_PRODUCT"] == "QA"
        for bit in range(8):
            assert f"QA_BIT_{bit}" in tags
        assert tags["GEOM_SHADOW_LIT"] == str(bp.GEOM_SHADOW_LIT)


def test_spm_archive_then_stage_then_zip(tmp_path, monkeypatch):
    import zipfile

    sid, rel = "20231203T0022175440", "miscellaneous/raw/20231203/ch2_iir_nri_20231203T0022175440_d_img_d18.spm"
    archive, stage = tmp_path / "archive", tmp_path / "stage"
    (archive / "zips").mkdir(parents=True)
    with zipfile.ZipFile(archive / "zips" / f"ch2_iir_nri_{sid}_d_img_d18.zip", "w") as zf:
        zf.writestr(rel, "spm")
    monkeypatch.setattr(ck, "ARCHIVE", archive)
    monkeypatch.setattr(ck, "ANC_ROOTS", [archive])

    assert ck.spm(sid) is None  # no stage: never extracts
    assert ck.spm(sid, stage) == stage / rel  # extracted from the zip
    assert (stage / rel).read_text() == "spm"
    (archive / rel).parent.mkdir(parents=True)
    (archive / rel).write_text("archived")
    assert ck.spm(sid, stage) == archive / rel  # archive wins
