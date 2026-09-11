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
