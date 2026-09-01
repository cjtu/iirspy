"""A GLT must be the map GDAL itself would have used.

`make_glt` warps two camera-space index bands through the same `project` call as everything else,
so indexing a cube through the result has to reproduce a nearest-neighbour warp of that cube.
"""

import numpy as np
import pytest
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.warp import Resampling

from iirspy import georef, glt

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
    g = glt.make_glt(gcps, CFG, (NY, NX))
    want = georef.project(cube, gcps, CFG, resampling=Resampling.nearest)
    np.testing.assert_array_equal(glt.apply_glt(cube, g), want)


def test_glt_marks_only_the_camera_footprint(tmp_path):
    """Nodata means "no camera pixel here", never "that pixel had no data" -- so one GLT serves L1,
    L2 and anything built later, each bringing its own NaN pattern."""
    g = glt.make_glt(_gcps(), CFG, (NY, NX))
    valid = g[0] >= 0
    assert valid.any() and not valid.all()
    np.testing.assert_array_equal(valid, g[1] >= 0)  # both bands agree on the footprint
    assert g[0][valid].max() < NX
    assert g[1][valid].max() < NY
    assert g[0][valid].min() >= 0

    # ... and a cube whose own holes fall inside that footprint still reports them
    out = glt.apply_glt(_cube(), g)
    assert np.isnan(out[1]).sum() > np.isnan(out[0]).sum()


def test_glt_round_trips_through_a_cog_with_the_tags_that_make_it_usable(tmp_path):
    """Everything needed to apply the table has to travel inside it: which scene and group, what
    frame band 2 is in, and the camera extent the indices are valid over."""
    gcps = _gcps()
    g = glt.make_glt(gcps, CFG, (NY, NX))
    f = glt.save_glt(
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

    back, tags = glt.read_glt(f)
    np.testing.assert_array_equal(back, g)
    assert tags["sid"] == "20210103T1829495344"
    assert tags["group"] == "south"
    assert tags["scan0"] == 0
    assert tags["row_frame"] == "absolute_scan"
    assert tags["lat_range"] == [-90.0, -59.0]
    assert tags["camera_shape"] == [NY, NX]


def test_apply_glt_windows_without_changing_the_answer(tmp_path):
    g = glt.make_glt(_gcps(), CFG, (NY, NX))
    cube = _cube()
    full = glt.apply_glt(cube, g)
    win = (10, 60, 5, 40)
    np.testing.assert_array_equal(glt.apply_glt(cube, g, window=win), full[:, 10:60, 5:40])


def test_apply_glt_rejects_a_cube_the_glt_was_not_built_for():
    g = glt.make_glt(_gcps(), CFG, (NY, NX))
    with pytest.raises(ValueError, match="camera"):
        glt.apply_glt(_cube()[:, : NY - 3], g)


def test_glt_rows_are_absolute_scans_not_crop_rows():
    """The GCP lattice is crop-relative, so a GLT that stored raw lattice rows would only ever
    apply to the exact latitude cut the solve used -- and silently mis-index any other. Storing the
    absolute Scan means the caller states what their own cube starts at, and nothing else."""
    gcps = _gcps()
    cube = _cube()
    plain = glt.make_glt(gcps, CFG, (NY, NX))
    shifted = glt.make_glt(gcps, CFG, (NY, NX), scan0=11050)

    on = plain[0] >= 0
    np.testing.assert_array_equal(shifted[0], plain[0])  # columns never move
    np.testing.assert_array_equal(shifted[1][on], plain[1][on] + 11050)

    # ... and the same cube read back through either table, told where it starts, is the same data
    np.testing.assert_array_equal(
        glt.apply_glt(cube, shifted, cube_scan0=11050), glt.apply_glt(cube, plain, cube_scan0=0)
    )


def test_apply_glt_rejects_the_wrong_scan_origin():
    g = glt.make_glt(_gcps(), CFG, (NY, NX), scan0=11050)
    with pytest.raises(ValueError, match="cube_scan0"):
        glt.apply_glt(_cube(), g, cube_scan0=0)


def test_scene_glt_reuses_a_table_but_a_fresh_solve_replaces_it(tmp_path):
    """A re-solve changes the GCPs, so the table on disk is stale -- returning it would silently
    project every later product through the old geometry."""
    from dataclasses import replace

    cfg = replace(CFG, lat_band=(-90.0, -59.0))
    a = glt.scene_glt("sid", "south", _gcps(), cfg, 0, tmp_path)
    first = a.read_bytes()

    moved = [GroundControlPoint(row=g.row, col=g.col, x=g.x + 400.0, y=g.y) for g in _gcps()]
    assert glt.scene_glt("sid", "south", moved, cfg, 0, tmp_path).read_bytes() == first
    assert glt.scene_glt("sid", "south", moved, cfg, 0, tmp_path, overwrite=True).read_bytes() != first
