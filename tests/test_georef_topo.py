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


def test_camera_topo_is_camera_gradients_then_slope_aspect(monkeypatch):
    """camera_topo must be exactly the split: camera_gradients's (sx, sy) fed through slope_aspect."""
    from dataclasses import replace

    def fake_render_topo(fgeom, fspm, cfg, kernels=None):
        tr = from_origin(0.0, 40.0, 40.0, 40.0)
        gx = np.linspace(-1.0, 1.0, 4 * 4).reshape(4, 4)
        gy = np.linspace(1.0, -1.0, 4 * 4).reshape(4, 4)
        lit = np.full((4, 4), 0.8)
        return gx, gy, lit, tr, {"az_grid": 12.0, "elev": 5.0, "r_sun": 0.27}

    monkeypatch.setattr(georef, "render_topo", fake_render_topo)
    cfg = replace(georef.GeorefConfig(), aoi=(-1000.0, -1000.0, 1000.0, 1000.0))
    gcps = [
        GroundControlPoint(row=float(r), col=float(c), x=20.0 + 40.0 * c, y=20.0 + 40.0 * r)
        for r in (0, 3)
        for c in (0, 3)
    ]
    shape = (4, 4)

    slope, aspect, lit, info = georef.camera_topo(gcps, shape, "fgeom", "fspm", cfg)
    sx, sy, lit2, info2 = georef.camera_gradients(gcps, shape, "fgeom", "fspm", cfg)
    want_slope, want_aspect = georef.slope_aspect(sx, sy)

    np.testing.assert_allclose(slope, want_slope)
    np.testing.assert_allclose(aspect, want_aspect)
    np.testing.assert_allclose(lit, lit2)
    assert info == info2


def test_row_bands_falls_back_to_midlat_across_the_polar_seam():
    """A camera row is 'south' only if both bracketing lattice rows are fully inside -76.5..-90."""
    from dataclasses import replace

    from iirspy import chunks as ck

    cfg = replace(georef.GeorefConfig(), pole="south")
    assert ck.BANDS["south"]["lat_range"] == (-90.0, -76.5)
    # Lattice rows 0, 50, 100, 150: latitudes straddle the seam so row 50 is the last fully-polar
    # lattice row and row 100 straddles it (one column polar, one not).
    lattice_lat = {0: -85.0, 50: -80.0, 100: -76.0, 150: -70.0}
    cols = (0, 10)
    gcps = []
    for r, lat in lattice_lat.items():
        for c in cols:
            # row 100 straddles the seam: col 0 stays polar, col 10 crosses into midlat territory
            this_lat = lat if not (r == 100 and c == 10) else -77.0
            x, y = georef.to_stereo("south").transform(0.0, this_lat)
            gcps.append(GroundControlPoint(row=float(r), col=float(c), x=x, y=y))

    band = georef._row_bands(gcps, ny=160, cfg=cfg)

    # rows [0, 50): both bracketing rows (0, 50) are fully polar -> south
    assert set(band[0:50]) == {"south"}
    # rows [50, 100): row 100 is NOT fully polar (straddles) -> conservative fallback to midlat
    assert set(band[50:100]) == {"south_midlat"}
    # rows [100, 150) and beyond: neither bracket is fully polar -> midlat
    assert set(band[100:150]) == {"south_midlat"}


def test_scene_topo_assembles_pieces_with_no_blending_and_monotone_sun(tmp_path, monkeypatch):
    """Pieces just concatenate (no seam blending), and per-row sun interpolates monotonically."""
    from dataclasses import replace

    from iirspy import chunks as ck

    ny, nx = 3500, 4  # > _split_run's max_rows -> at least 2 pieces, one band (equatorial, no split)
    rows = np.unique(np.r_[np.arange(0, ny, 25), ny - 1])
    cols = (0, nx - 1)
    gcps = [GroundControlPoint(row=float(r), col=float(c), x=1000.0 * c, y=100.0 * r) for r in rows for c in cols]
    cfg = replace(georef.GeorefConfig(), pole="equatorial", margin_m=5000.0, ps=2000.0)
    # bands() resolves real DEM paths on disk for every band; scene_topo only needs the names, and
    # render_topo (mocked below) never opens dem_near/dem_far, so a placeholder dict skips the DEM tree.
    monkeypatch.setattr(
        ck, "bands", lambda: {"equatorial": {**ck.BANDS["equatorial"], "dem_near": "x", "dem_far": "x"}}
    )

    calls = []

    def fake_render_topo(fgeom, fspm, cfg, kernels=None):
        xs, ys, tr, _ = georef.grid_of(cfg)
        val = float(cfg.aoi[1])  # unique per piece: pieces don't overlap in y
        gx = np.full((len(ys), len(xs)), val, "float32")
        gy = np.zeros_like(gx)
        lit = np.ones_like(gx)
        elev = val / 1e5
        calls.append(val)
        return gx, gy, lit, tr, {"az_grid": 10.0, "elev": elev, "r_sun": 0.27}

    monkeypatch.setattr(georef, "render_topo", fake_render_topo)
    f = georef.scene_topo("sid", "equatorial", gcps, (ny, nx), "fgeom", "fspm", cfg, [], tmp_path)

    assert len(calls) >= 2  # the strip really did split into multiple pieces

    import rasterio

    with rasterio.open(f) as src:
        names = [src.descriptions[i] for i in range(src.count)]
        slope = src.read(names.index("slope") + 1)
        sun_elev = src.read(names.index("sun_elev") + 1)

    # Recompute the expected piece boundaries the same way scene_topo does, and check every row's
    # slope matches its own piece's constant gradient -- concatenation, no cross-piece blending.
    from iirspy import chunks as ck

    band_of_row = georef._row_bands(gcps, ny, cfg)
    pieces = [p for a, z in ck._runs(band_of_row == "equatorial") for p in georef._split_run(a, z)]
    assert len(pieces) == len(calls)
    for (a, z), val in zip(pieces, calls, strict=True):
        want = np.degrees(np.arctan(abs(val)))
        np.testing.assert_allclose(slope[a:z], want, atol=1e-3)

    # Per-row sun elevation must be monotone between piece centres (np.interp of an increasing
    # sequence), matching the along-track sun geometry each piece reported.
    assert np.all(np.diff(sun_elev) >= -1e-6)


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
