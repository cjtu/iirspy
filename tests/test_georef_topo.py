import numpy as np
import pytest
from rasterio.control import GroundControlPoint
from rasterio.transform import from_origin

from iirspy import georef


def test_slope_aspect_on_a_known_plane():
    """A plane rising 1 m per 1 m eastward is a 45 deg slope facing west, aspect being downhill."""
    slope, aspect = georef.slope_aspect(gx=1.0, gy=0.0)
    assert slope == pytest.approx(45.0)
    assert aspect == pytest.approx(270.0)
    # ... and one dropping toward the north faces 0 deg, the wrap point, not 360
    assert georef.slope_aspect(0.0, -np.tan(np.radians(10)))[1] == pytest.approx(0.0)
    assert georef.slope_aspect(0.0, -np.tan(np.radians(10)))[0] == pytest.approx(10.0)


def test_camera_xy_recovers_the_affine_its_lattice_came_from():
    """Sampled between the GCP nodes, the spline must reproduce the map coords they encode."""
    ny, nx = 120, 40
    tr = from_origin(-1000.0, 5000.0, 40.0, 40.0)  # arbitrary map grid
    rows, cols = np.r_[np.arange(0, ny, 25), ny - 1], np.linspace(0, nx - 1, 13).astype(int)
    gcps = [
        GroundControlPoint(row=float(r), col=float(c), x=tr.c + c * tr.a, y=tr.f + r * tr.e)
        for r in np.unique(rows)
        for c in np.unique(cols)
    ]
    x, y = georef.camera_xy(gcps, (ny, nx))
    jj, ii = np.mgrid[0:ny, 0:nx]
    np.testing.assert_allclose(x, tr.c + ii * tr.a, atol=1e-6)
    np.testing.assert_allclose(y, tr.f + jj * tr.e, atol=1e-6)


def test_camera_xy_rejects_a_partial_lattice():
    gcps = [GroundControlPoint(row=0.0, col=0.0, x=0.0, y=0.0), GroundControlPoint(row=1.0, col=1.0, x=1.0, y=1.0)]
    with pytest.raises(ValueError, match="lattice"):
        georef.camera_xy(gcps, (2, 2))


def test_topo_product_round_trips_into_the_l2_reader(tmp_path):
    """save_topo -> photometry.load_topo must come back band-for-band, names and all."""
    import iirspy.photometry as photometry

    slope = np.linspace(0, 30, 12).reshape(4, 3).astype("float32")
    aspect = np.linspace(0, 359, 12).reshape(4, 3).astype("float32")
    lit = np.linspace(0, 1, 12).reshape(4, 3).astype("float32")
    f = georef.save_topo(tmp_path / "topo.tif", slope, aspect, lit, tags={"az_grid": 22.0, "elev": 3.6})
    s, a, li, sun = photometry.load_topo(f)
    assert sun == (22.0, 3.6)  # the frame slope/aspect live in must survive the write
    np.testing.assert_allclose(s.values, slope)
    np.testing.assert_allclose(a.values, aspect)
    np.testing.assert_allclose(li.values, lit)


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


def test_a_bands_warp_does_not_depend_on_which_bands_share_the_call():
    """GDAL masks a narrow-footprint band down toward its neighbours in a multi-band warp, so a
    bare cube's nodata depends on its own band list. unify_nodata removes that dependence."""
    cfg = georef.GeorefConfig(aoi=(0.0, 0.0, 4000.0, 4000.0), ps=40.0)
    ny, nx = 60, 20
    gcps = [
        GroundControlPoint(row=float(r), col=float(c), x=500.0 + 30.0 * c, y=3500.0 - 50.0 * r)
        for r in np.arange(0, ny, 5)
        for c in np.arange(0, nx, 2)
    ]
    holed = np.arange(ny * nx, dtype="float32").reshape(ny, nx)
    holed[20:30, 5:12] = np.nan  # a hole this band has and the others do not
    full = np.full((ny, nx), 7.0, "float32")
    both_nan = np.s_[:5, :]  # nodata in every band, so it stays the shared mask
    holed[both_nan] = np.nan
    full[both_nan] = np.nan

    def warp(cube):
        return georef.unstack_nodata(georef.project(georef.unify_nodata(cube), gcps, cfg), 0)

    # Every companion set must give band 0 the same answer. (A one-band cube is deliberately not
    # in this set: with nothing to disagree with, its hole IS the shared mask, so GDAL renormalises
    # over it instead of marking it -- the documented "warped alone" difference.)
    with_one = warp(np.stack([holed, full]))
    with_two = warp(np.stack([holed, full, full * 2.0]))
    with_other = warp(np.stack([holed, full * 3.0]))
    np.testing.assert_array_equal(with_one, with_two)
    np.testing.assert_array_equal(with_one, with_other)

    # ... nothing that touched the hole survives: the fill is 0, below every real value here
    kept = with_two[np.isfinite(with_two)]
    assert kept.size > 100
    assert kept.min() > 0.0

    # The contamination this guards against needs real per-band footprints to reproduce and does
    # not show on a constant companion; it was measured on an L2 cube (band 27: 165,783 px beside
    # band 251 against 171,428 alone). What this test pins is the invariant, not the bug.
