def test_import_classes():
    pass


def test_output_bands_window_matches_full_interpolation():
    """`calibrate_to_rad(output_bands=...)` narrows *before* `interpolate_na("band", max_gap=6)`.

    That's only safe because a band outside +-6 of the target could never have fed its
    interpolated value anyway (that's what max_gap=6 means) -- so narrowing first must give the
    same answer, for any target, as computing on the full band axis and slicing after. Exercise
    every gap width up to max_gap so a fix that only widens/narrows the reachable window would
    still be caught.
    """
    import numpy as np
    import xarray as xr

    rng = np.random.default_rng(0)
    bands = list(range(1, 41))  # PAN_BANDS-sized axis, arbitrary values
    target = 20
    window = sorted(range(target - 6, target + 7))

    da = xr.DataArray(
        rng.normal(size=(len(bands), 5, 5)).astype("float32"),
        dims=("band", "y", "x"),
        coords={"band": bands},
    )
    for gap in range(0, 7):  # 0 = target itself masked; up to max_gap=6 either side
        masked = da.copy()
        masked.loc[{"band": [target - i for i in range(gap + 1) if target - i in bands]}] = np.nan

        full_then_slice = masked.interpolate_na("band", max_gap=6, method="linear").sel(band=[target])
        window_first = masked.sel(band=window).interpolate_na("band", max_gap=6, method="linear").sel(band=[target])

        np.testing.assert_array_equal(full_then_slice.values, window_first.values)


def test_geotiff_write_resumes_from_progress_sidecar(tmp_path):
    """A killed block-wise write picks up at the recorded row instead of redoing the cube.

    Simulates the cluster's timeout: half the blocks are on disk, the `.progress` sidecar says so,
    and the rerun must fill only the rest and still produce the full, correct raster.
    """
    import json

    import numpy as np
    import rasterio
    import xarray as xr

    from iirspy.iirs import _save_geotiff

    ny, block = 12, 4
    data = np.arange(2 * ny * 3, dtype="float32").reshape(2, ny, 3)
    da = xr.DataArray(data, dims=("band", "y", "x"), coords={"band": [10, 20]}).rio.write_crs("EPSG:4326")
    fout = tmp_path / "cube.tif"

    _save_geotiff(str(fout), block, da)
    fprog = tmp_path / "cube.tif.progress"
    assert not fprog.exists()  # a finished write leaves no sidecar behind
    with rasterio.open(fout) as src:
        np.testing.assert_array_equal(src.read(), data)

    # Roll the file back to "two of three blocks written, then killed", and additionally poison the
    # first block: a resume must leave it alone, so the poison surviving is what proves rows before
    # `rows_done` were skipped rather than silently rewritten.
    with rasterio.open(fout, "r+") as dst:
        dst.write(np.zeros((2, block, 3), "float32"), window=rasterio.windows.Window(0, 0, 3, block))
        dst.write(np.zeros((2, block, 3), "float32"), window=rasterio.windows.Window(0, 2 * block, 3, block))
    fprog.write_text(json.dumps({"shape": [2, ny, 3], "row_block": block, "rows_done": 2 * block}))

    _save_geotiff(str(fout), block, da)
    with rasterio.open(fout) as src:
        got = src.read()
    assert not got[:, :block].any()  # block 0 skipped, still poisoned
    np.testing.assert_array_equal(got[:, block:], data[:, block:])  # blocks 1-2: kept, then rewritten
    assert not fprog.exists()

    # A sidecar that disagrees with the cube being written is not a resume point
    fprog.write_text(json.dumps({"shape": [2, ny, 3], "row_block": block * 2, "rows_done": 2 * block}))
    from iirspy.iirs import _resume_row

    assert _resume_row(fout, fprog, {"shape": [2, ny, 3], "row_block": block}) == 0


def test_empirical_sidecar_roundtrips_dark_flat_smile(tmp_path):
    """`attach_empirical_frames` coords -> `_save_empirical_sidecar` -> one .npz with all three."""
    import numpy as np
    import xarray as xr

    from iirspy.iirs import IIRSData

    bands, x = [10, 20, 30], np.arange(4.0)
    img = xr.DataArray(
        np.zeros((3, 2, 4), "float32"),
        dims=("band", "y", "x"),
        coords={
            "band": bands,
            "x": x,
            "empirical_dark": (("band", "x"), np.full((3, 4), 1.0, "float32")),
            "empirical_flat": (("band", "x"), np.full((3, 4), 2.0, "float32")),
            "empirical_smile": (("band", "x"), np.full((3, 4), 3.0, "float32")),
        },
    )
    from types import SimpleNamespace

    fout = tmp_path / "cube.tif"
    fnpz = IIRSData._save_empirical_sidecar(SimpleNamespace(img=img), fout)

    assert fnpz == str(tmp_path / "cube_empirical.npz")
    d = np.load(fnpz)
    np.testing.assert_array_equal(d["dark"], 1.0)
    np.testing.assert_array_equal(d["flat"], 2.0)
    np.testing.assert_array_equal(d["smile"], 3.0)
    np.testing.assert_array_equal(d["band"], bands)


def test_clip_aoi_warps_onto_the_map_grid_and_matches_a_polygon_masked_apply_glt(tmp_path):
    from types import MethodType, SimpleNamespace

    import numpy as np
    import pytest
    import xarray as xr
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_geom

    from iirspy import georef
    from iirspy.iirs import IIRSData

    ny, nx = 10, 8
    rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    x, y = 500.0 + 30.0 * cols, 3500.0 - 50.0 * rows
    lon, lat = georef.xy_to_lonlat(x, y, "south")
    loc = xr.Dataset(
        {"lon": (("y", "x"), lon), "lat": (("y", "x"), lat)},
        coords={"y": np.arange(ny), "x": np.arange(nx)},
    )

    cube = np.arange(ny * nx, dtype="float32").reshape(1, ny, nx)
    img = xr.DataArray(cube, dims=("band", "y", "x"), coords={"band": [1]})
    inst = SimpleNamespace(img=img, _loc_cache=loc)
    inst.glt = MethodType(IIRSData.glt, inst)
    inst._cube_scan0 = MethodType(IIRSData._cube_scan0, inst)
    inst._read_loc = MethodType(IIRSData._read_loc, inst)
    inst.clip_aoi = MethodType(IIRSData.clip_aoi, inst)

    crs = georef.stereo_crs("south")
    x0, y0, x1, y1 = 500.0, 3000.0, 800.0, 3400.0
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    geom_lonlat = {
        "type": "Polygon",
        "coordinates": [[georef.to_lonlat("south").transform(px, py) for px, py in corners]],
    }

    out = inst.clip_aoi(geom_lonlat, crs, res=40.0)

    want_geom = transform_geom(georef.LONLAT, crs, geom_lonlat)
    gxs, gys = zip(*want_geom["coordinates"][0], strict=False)
    table, tr = inst.glt(crs, 40.0, bounds=(min(gxs), min(gys), max(gxs), max(gys)))
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
