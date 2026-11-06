"""A GLT must be the map GDAL itself would have used.

`make_glt` warps two camera-space index bands through the same `project` call as everything else,
so indexing a cube through the result has to reproduce a nearest-neighbour warp of that cube. The
rendering tests below place cube pixels on the map through a Geometry's GLT.
"""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.control import GroundControlPoint
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, transform_geom

from iirspy.georef import georef
from iirspy.georef.geometry import Geometry, quality
from iirspy.iirs import IIRSData, _loc_extent

CFG = georef.GeorefConfig(aoi=(0.0, 0.0, 4000.0, 4000.0), ps=40.0)
NY, NX = 60, 20


def _gcps():
    """A lattice bent off the affine, so the warp is a real TPS and not an accidental polynomial."""
    return [
        GroundControlPoint(
            row=float(r),
            col=float(c),
            x=500.0 + 30.0 * c + 20.0 * np.sin(r / 9.0),
            y=3500.0 - 50.0 * r + 20.0 * np.cos(c / 3.0),
        )
        for r in np.arange(0, NY, 5)
        for c in np.arange(0, NX, 2)
    ]


def _cube():
    """(3, NY, NX) with a per-band hole and a shared dead margin."""
    a = np.arange(NY * NX, dtype="float32").reshape(NY, NX)
    holed = a.copy()
    holed[20:30, 5:12] = np.nan
    cube = np.stack([a, holed, a * -3.0])
    cube[:, :5, :] = np.nan
    return cube


def test_indexing_through_the_glt_is_a_nearest_warp(tmp_path):
    gcps = _gcps()
    cube = _cube()
    g = georef.make_glt(gcps, CFG, (NY, NX))
    want = georef.project(cube, gcps, CFG, resampling=Resampling.nearest)
    np.testing.assert_array_equal(georef.apply_glt(cube, g), want)


def test_glt_and_projection_are_unchanged_when_spilled_to_disk(tmp_path, monkeypatch):
    """The memmap destination is uninitialised, so this also proves GDAL fills nodata itself."""
    gcps, cube = _gcps(), _cube()
    g_ram = georef.make_glt(gcps, CFG, (NY, NX))
    p_ram = georef.project(cube, gcps, CFG)
    monkeypatch.setattr(georef, "MAX_RAM_BYTES", 0)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    g_disk = georef.make_glt(gcps, CFG, (NY, NX))
    p_disk = georef.project(cube, gcps, CFG)
    assert isinstance(g_disk, np.memmap) and isinstance(p_disk, np.memmap)
    np.testing.assert_array_equal(g_disk, g_ram)
    np.testing.assert_array_equal(p_disk, p_ram)
    assert not list(tmp_path.iterdir())  # spill files are unlinked, not left behind


def test_glt_marks_only_the_camera_footprint(tmp_path):
    """Nodata means "no camera pixel here", never "that pixel had no data" -- so one GLT serves L1,
    L2 and anything built later, each bringing its own NaN pattern."""
    g = georef.make_glt(_gcps(), CFG, (NY, NX))
    valid = g[0] >= 0
    assert valid.any() and not valid.all()
    np.testing.assert_array_equal(valid, g[1] >= 0)  # both bands agree on the footprint
    assert g[0][valid].max() < NX
    assert g[1][valid].max() < NY
    assert g[0][valid].min() >= 0

    # ... and a cube whose own holes fall inside that footprint still reports them
    out = georef.apply_glt(_cube(), g)
    assert np.isnan(out[1]).sum() > np.isnan(out[0]).sum()


def test_glt_round_trips_through_a_cog_with_the_tags_that_make_it_usable(tmp_path):
    """Everything needed to apply the table has to travel inside it: which scene and group, what
    frame band 2 is in, and the camera extent the indices are valid over."""
    gcps = _gcps()
    g = georef.make_glt(gcps, CFG, (NY, NX))
    f = georef.save_glt(
        tmp_path / "glt.tif",
        g,
        CFG,
        sid="20210103T1829495344",
        group="south",
        scan0=0,
        lat_range=(-90.0, -59.0),
        camera_shape=(NY, NX),
    )
    with rasterio.open(f) as src:
        assert src.count == 2
        assert src.dtypes == ("int32", "int32")
        assert src.nodata == -1
        assert src.block_shapes[0] == (256, 256)
        # Against a product this pipeline already writes, not against `stereo_crs` directly: GDAL
        # normalises the WKT on write (drops the default scale_factor, adds the unit authority), so
        # the invariant worth asserting is that a GLT lands on the same grid as the warped rasters.
        with rasterio.open(georef.save_grid(tmp_path / "ref.tif", np.zeros((4, 4), "float32"), CFG)) as ref:
            assert src.crs == ref.crs
        assert src.transform == georef.window_of(CFG)[1]

    back, tags = georef.read_glt(f)
    np.testing.assert_array_equal(back, g)
    assert tags["sid"] == "20210103T1829495344"
    assert tags["group"] == "south"
    assert tags["scan0"] == 0
    assert tags["row_frame"] == "absolute_scan"
    assert tags["lat_range"] == [-90.0, -59.0]
    assert tags["camera_shape"] == [NY, NX]


def test_apply_glt_windows_without_changing_the_answer(tmp_path):
    g = georef.make_glt(_gcps(), CFG, (NY, NX))
    cube = _cube()
    full = georef.apply_glt(cube, g)
    win = (10, 60, 5, 40)
    np.testing.assert_array_equal(georef.apply_glt(cube, g, window=win), full[:, 10:60, 5:40])


def test_apply_glt_rejects_a_cube_the_glt_was_not_built_for():
    g = georef.make_glt(_gcps(), CFG, (NY, NX))
    with pytest.raises(ValueError, match="camera"):
        georef.apply_glt(_cube()[:, : NY - 3], g)


def test_glt_rows_are_absolute_scans_not_crop_rows():
    """The GCP lattice is crop-relative, so a GLT that stored raw lattice rows would only ever
    apply to the exact latitude cut the solve used -- and silently mis-index any other. Storing the
    absolute Scan means the caller states what their own cube starts at, and nothing else."""
    gcps = _gcps()
    cube = _cube()
    plain = georef.make_glt(gcps, CFG, (NY, NX))
    shifted = georef.make_glt(gcps, CFG, (NY, NX), scan0=11050)

    on = plain[0] >= 0
    np.testing.assert_array_equal(shifted[0], plain[0])  # columns never move
    np.testing.assert_array_equal(shifted[1][on], plain[1][on] + 11050)

    # ... and the same cube read back through either table, told where it starts, is the same data
    np.testing.assert_array_equal(
        georef.apply_glt(cube, shifted, cube_scan0=11050), georef.apply_glt(cube, plain, cube_scan0=0)
    )


def test_apply_glt_rejects_the_wrong_scan_origin():
    g = georef.make_glt(_gcps(), CFG, (NY, NX), scan0=11050)
    with pytest.raises(ValueError, match="cube_scan0"):
        georef.apply_glt(_cube(), g, cube_scan0=0)


def test_scene_glt_reuses_a_table_but_a_fresh_solve_replaces_it(tmp_path):
    """A re-solve changes the GCPs, so the table on disk is stale -- returning it would silently
    project every later product through the old geometry."""
    from dataclasses import replace

    cfg = replace(CFG, lat_band=(-90.0, -59.0))
    a = georef.scene_glt("sid", "south", _gcps(), cfg, 0, tmp_path)
    first = a.read_bytes()

    moved = [GroundControlPoint(row=g.row, col=g.col, x=g.x + 400.0, y=g.y) for g in _gcps()]
    assert georef.scene_glt("sid", "south", moved, cfg, 0, tmp_path).read_bytes() == first
    assert georef.scene_glt("sid", "south", moved, cfg, 0, tmp_path, overwrite=True).read_bytes() != first


def _synthetic_loc(ny, nx, x0=500.0, dx=30.0, y0=3500.0, dy=-50.0, pole="south"):
    rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    x, y = x0 + dx * cols, y0 + dy * rows
    lon, lat = georef.xy_to_lonlat(x, y, pole)
    loc = xr.Dataset(
        {"lon": (("y", "x"), lon), "lat": (("y", "x"), lat)},
        coords={"y": np.arange(ny), "x": np.arange(nx)},
    )
    return loc, x, y


def _geom(loc):
    shape = loc.lon.shape
    return Geometry.from_arrays(
        "sid",
        "south",
        0,
        "arosics",
        {"lon": loc.lon.values, "lat": loc.lat.values, "radius": np.full(shape, 1737400.0)},
        {},
        np.zeros(shape, "uint16"),
        quality("arosics", "pass"),
    )


def _bound_inst(img):
    inst = SimpleNamespace(img=img)
    inst.to_geotiff = MethodType(IIRSData.to_geotiff, inst)
    inst._cube_scan0 = MethodType(IIRSData._cube_scan0, inst)
    inst._render = MethodType(IIRSData._render, inst)
    return inst


def test_glt_recovers_known_camera_pixels_from_a_planar_loc():
    ny, nx = 6, 5
    loc, x, y = _synthetic_loc(ny, nx, dx=2000.0, dy=-2000.0)
    geom = _geom(loc)
    table, tr = geom.glt(geom.grid(2000.0))
    assert table.shape == (2, ny, nx)
    assert (table[0] >= 0).all()
    for r in range(ny):
        for c in range(nx):
            gx = int((x[r, c] - tr.c) / tr.a)
            gy = int((y[r, c] - tr.f) / tr.e)
            assert table[0, gy, gx] == c
            assert table[1, gy, gx] == r


def test_to_geotiff_places_pixel_values_at_the_expected_map_cell(tmp_path):
    ny, nx = 6, 5
    loc, x, y = _synthetic_loc(ny, nx, dx=2000.0, dy=-2000.0)
    cube = np.arange(ny * nx, dtype="float32").reshape(1, ny, nx)
    img = xr.DataArray(cube, dims=("band", "y", "x"), coords={"band": [1]})
    geom = _geom(loc)
    fout = tmp_path / "render.tif"
    _bound_inst(img).to_geotiff(str(fout), geom=geom, grid=geom.grid(2000.0), full_bands=False)

    with rasterio.open(fout) as src:
        data = src.read(1)
        tr = src.transform
    r, c = 2, 3
    gx = int((x[r, c] - tr.c) / tr.a)
    gy = int((y[r, c] - tr.f) / tr.e)
    assert data[gy, gx] == cube[0, r, c]


def test_to_geotiff_from_a_saved_glt_renders_only_the_cropped_cubes_scans(tmp_path):
    ny, nx = 6, 5
    loc, x, y = _synthetic_loc(ny, nx, dx=2000.0, dy=-2000.0)
    crs = georef.stereo_crs("south")
    geom = _geom(loc)
    table, tr = geom.glt(geom.grid(2000.0))
    fglt = tmp_path / "glt.tif"
    with rasterio.open(
        fglt, "w", driver="GTiff", height=ny, width=nx, count=2, dtype="int32", crs=crs, transform=tr
    ) as dst:
        dst.write(table)

    cube = np.arange(ny * nx, dtype="float32").reshape(1, ny, nx)
    full = xr.DataArray(cube, dims=("band", "y", "x"), coords={"band": [1], "y": np.arange(ny), "x": np.arange(nx)})
    inst = _bound_inst(full.isel(y=slice(2, 5)))  # scans 2-4 only, as a lat crop leaves it
    fout = tmp_path / "render.tif"
    inst.to_geotiff(str(fout), glt=fglt, full_bands=False, row_block=1)

    with rasterio.open(fout) as src:
        data, otr = src.read(1), src.transform
    assert data.shape == (3, nx)
    assert set(np.unique(data[np.isfinite(data)])) == set(cube[0, 2:5].ravel())
    r, c = 3, 1
    assert data[int((y[r, c] - otr.f) / otr.e), int((x[r, c] - otr.c) / otr.a)] == cube[0, r, c]


def test_geoloc_vrt_warps_to_the_same_placement_as_glt(tmp_path):
    from rasterio.vrt import WarpedVRT

    ny, nx = 6, 5
    loc, x, y = _synthetic_loc(ny, nx, dx=2000.0, dy=-2000.0)
    cube = np.arange(ny * nx, dtype="float32").reshape(1, ny, nx)

    fcube = tmp_path / "cube.tif"
    with rasterio.open(fcube, "w", driver="GTiff", height=ny, width=nx, count=1, dtype="float32") as dst:
        dst.write(cube[0], 1)

    crs = georef.stereo_crs("south")
    fvrt = georef.geoloc_vrt(fcube, loc, crs, tmp_path)

    with rasterio.open(fvrt) as src, WarpedVRT(src, src_crs=crs, crs=crs, src_method="GEOLOC_ARRAY") as vrt:
        warped = vrt.read(1)
        tr = vrt.transform

    r, c = 2, 3
    gx = int((x[r, c] - tr.c) / tr.a)
    gy = int((y[r, c] - tr.f) / tr.e)
    assert 0 <= gy < warped.shape[0] and 0 <= gx < warped.shape[1]
    window = warped[max(gy - 1, 0) : gy + 2, max(gx - 1, 0) : gx + 2]
    assert cube[0, r, c] in window


def test_clip_aoi_warps_onto_the_map_grid_and_matches_a_polygon_masked_apply_glt():
    ny, nx = 10, 8
    rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    x, y = 500.0 + 30.0 * cols, 3500.0 - 50.0 * rows
    lon, lat = georef.xy_to_lonlat(x, y, "south")
    geom = Geometry.from_arrays(
        "sid",
        "south",
        0,
        "arosics",
        {"lon": lon, "lat": lat, "radius": np.full((ny, nx), 1737400.0)},
        {},
        np.zeros((ny, nx), "uint16"),
        quality("arosics", "pass"),
    )

    cube = np.arange(ny * nx, dtype="float32").reshape(1, ny, nx)
    img = xr.DataArray(cube, dims=("band", "y", "x"), coords={"band": [1]})
    inst = SimpleNamespace(img=img, basename="sid")
    inst._cube_scan0 = MethodType(IIRSData._cube_scan0, inst)
    inst.clip_aoi = MethodType(IIRSData.clip_aoi, inst)
    inst._render = MethodType(IIRSData._render, inst)

    crs = georef.stereo_crs("south")
    x0, y0, x1, y1 = 500.0, 3000.0, 800.0, 3400.0
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    geom_lonlat = {
        "type": "Polygon",
        "coordinates": [[georef.to_lonlat("south").transform(px, py) for px, py in corners]],
    }

    out = inst.clip_aoi(geom_lonlat, geom=geom, res=40.0)

    want_geom = transform_geom(georef.LONLAT, crs, geom_lonlat)
    gxs, gys = zip(*want_geom["coordinates"][0], strict=False)
    table, tr = geom.glt(geom.grid(40.0, (min(gxs), min(gys), max(gxs), max(gys))))
    poly_mask = geometry_mask([want_geom], out_shape=table.shape[1:], transform=tr, invert=True)
    table[0][~poly_mask] = georef.NODATA
    want = georef.apply_glt(cube, table, cube_scan0=0)

    np.testing.assert_array_equal(out.values, want)
    from pyproj import Transformer

    out_x, out_y = Transformer.from_crs(georef.LONLAT, out.rio.crs, always_xy=True).transform(45.0, -80.0)
    want_x, want_y = georef.to_stereo("south").transform(45.0, -80.0)
    assert out_x == pytest.approx(want_x, abs=1e-6)
    assert out_y == pytest.approx(want_y, abs=1e-6)
    assert out.rio.transform() == tr


def test_loc_extent_crops_rows_by_latitude_from_the_ndi_loc_backplane():
    ny, nx = 20, 5
    lat = np.repeat(np.linspace(-10, 9, ny)[:, None], nx, axis=1)  # 1 deg per row
    lon = np.full((ny, nx), 300.0)  # 0-360 east, i.e. -60
    loc = xr.Dataset({"lon": (("y", "x"), lon), "lat": (("y", "x"), lat)})

    assert _loc_extent(loc, (-180, 180, -2, 3)) == (0, nx - 1, 8, 14)  # rows at lat -2..3
    assert _loc_extent(loc, (-61, -59, None, None)) == (0, nx - 1, 0, ny)
    with pytest.raises(ValueError):
        _loc_extent(loc, (0, 10, None, None))
