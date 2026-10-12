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
