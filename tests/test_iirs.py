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

    from iirspy.iirs import IIRSData

    ny, block = 12, 4
    data = np.arange(2 * ny * 3, dtype="float32").reshape(2, ny, 3)
    da = xr.DataArray(data, dims=("band", "y", "x"), coords={"band": [10, 20]}).rio.write_crs("EPSG:4326")
    fout = tmp_path / "cube.tif"

    IIRSData._save_geotiff(None, str(fout), block, da)
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

    IIRSData._save_geotiff(None, str(fout), block, da)
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
    """`clip_aoi` must equal `apply_glt` run on a GLT window whose out-of-polygon cells were
    nulled first, and come back with its own crs/transform -- no GCP sidecar needed."""
    from types import SimpleNamespace

    import numpy as np
    import rasterio
    import xarray as xr
    from rasterio.control import GroundControlPoint
    from rasterio.features import geometry_mask, geometry_window
    from rasterio.warp import transform_geom

    from iirspy import georef
    from iirspy.iirs import IIRSData

    cfg = georef.GeorefConfig(aoi=(0.0, 0.0, 4000.0, 4000.0), ps=40.0)
    ny, nx = 60, 20
    gcps = [
        GroundControlPoint(row=float(r), col=float(c), x=500.0 + 30.0 * c, y=3500.0 - 50.0 * r)
        for r in np.arange(0, ny, 5)
        for c in np.arange(0, nx, 2)
    ]
    glt = georef.make_glt(gcps, cfg, (ny, nx))
    fglt = georef.save_glt(
        tmp_path / "glt.tif",
        glt,
        cfg,
        sid="sid",
        group="south",
        scan0=0,
        lat_range=(-90.0, -60.0),
        camera_shape=(ny, nx),
    )

    cube = np.arange(ny * nx, dtype="float32").reshape(1, ny, nx)
    img = xr.DataArray(cube, dims=("band", "y", "x"), coords={"band": [1]})
    inst = SimpleNamespace(img=img)

    with rasterio.open(fglt) as src:
        crs_wkt = src.crs.to_wkt()

    x0, y0, x1, y1 = 900.0, 2900.0, 1400.0, 3400.0
    geom = {"type": "Polygon", "coordinates": [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]}

    out = IIRSData.clip_aoi(inst, fglt, geom, crs=crs_wkt)

    with rasterio.open(fglt) as src:
        want_geom = transform_geom(crs_wkt, src.crs, geom)
        window = geometry_window(src, [want_geom])
        sub_glt = src.read(window=window)
        crop_transform = src.window_transform(window)
    poly_mask = geometry_mask([want_geom], out_shape=sub_glt.shape[1:], transform=crop_transform, invert=True)
    sub_glt[0][~poly_mask] = georef.NODATA
    want = georef.apply_glt(cube, sub_glt, cube_scan0=0)

    np.testing.assert_array_equal(out.values, want)
    assert out.rio.crs == rasterio.CRS.from_wkt(crs_wkt)
    assert out.rio.transform() == crop_transform
