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


def test_local_plane_extrapolates_the_local_gradient_and_stays_bounded():
    """`_local_plane` is the off-support fallback: local level + local slope, clipped."""
    from scipy.spatial import cKDTree

    from iirspy.georef import _local_plane

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
    import rasterio

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

    from iirspy.georef import GeorefConfig, displacement_field

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


def test_gcp_lattice_samples_the_geometry_spline_at_the_right_rows(tmp_path):
    """The lattice evaluates the geometry TPS only where GCPs go, so row->Scan must line up.

    A linear geometry is reproduced exactly by a thin-plate spline, so the lattice's lon/lat are
    analytic and a row/Scan off-by-one fails hard instead of shifting the product slightly.
    """
    from dataclasses import replace

    import pandas as pd

    scan0, ny, nx = 500, 200, 250  # cube row r is geometry Scan scan0 + r
    scans = np.arange(scan0, scan0 + ny)
    pixels = np.array([0, 50, 100, 150, 200, nx - 1])  # the 6-point cross-track sampling
    pg, sg = np.meshgrid(pixels, scans)
    lon = 0.01 * pg + 0.002 * sg  # linear: reproduced exactly by a thin-plate spline
    lat = -60.0 + 0.001 * sg
    fgeom = tmp_path / "geom.csv"
    pd.DataFrame({
        "Pixel": pg.ravel(),
        "Scan": sg.ravel(),
        "Longitude": lon.ravel(),
        "Latitude": lat.ravel(),
    }).to_csv(fgeom, index=False)

    cfg = replace(georef.GeorefConfig(), pole="equatorial", lat_band=(lat.min(), lat.max()), row_step=25, ncol=13)
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
