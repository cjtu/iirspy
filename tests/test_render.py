from types import MethodType, SimpleNamespace

import numpy as np
import rasterio
import xarray as xr

from iirspy import georef, utils
from iirspy.iirs import IIRSData


def _synthetic_loc(ny, nx, x0=500.0, dx=30.0, y0=3500.0, dy=-50.0, pole="south"):
    rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    x, y = x0 + dx * cols, y0 + dy * rows
    lon, lat = georef.xy_to_lonlat(x, y, pole)
    loc = xr.Dataset(
        {"lon": (("y", "x"), lon), "lat": (("y", "x"), lat)},
        coords={"y": np.arange(ny), "x": np.arange(nx)},
    )
    return loc, x, y


def _bound_inst(img, loc):
    inst = SimpleNamespace(img=img, _loc_cache=loc)
    inst.glt = MethodType(IIRSData.glt, inst)
    inst.to_geotiff = MethodType(IIRSData.to_geotiff, inst)
    inst._cube_scan0 = MethodType(IIRSData._cube_scan0, inst)
    inst._read_loc = MethodType(IIRSData._read_loc, inst)
    inst._render = MethodType(IIRSData._render, inst)
    return inst


def test_glt_recovers_known_camera_pixels_from_a_planar_loc():
    ny, nx = 6, 5
    loc, x, y = _synthetic_loc(ny, nx, dx=2000.0, dy=-2000.0)
    inst = _bound_inst(img=xr.DataArray(np.zeros((1, ny, nx), "float32"), dims=("band", "y", "x")), loc=loc)

    crs = georef.stereo_crs("south")
    table, tr = inst.glt(crs, res=2000.0)
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
    inst = _bound_inst(img, loc)

    crs = georef.stereo_crs("south")
    fout = tmp_path / "render.tif"
    inst.to_geotiff(str(fout), crs, res=2000.0, full_bands=False)

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
    table, tr = _bound_inst(xr.DataArray(np.zeros((1, ny, nx), "float32"), dims=("band", "y", "x")), loc).glt(
        crs, res=2000.0
    )
    fglt = tmp_path / "glt.tif"
    with rasterio.open(
        fglt, "w", driver="GTiff", height=ny, width=nx, count=2, dtype="int32", crs=crs, transform=tr
    ) as dst:
        dst.write(table)

    cube = np.arange(ny * nx, dtype="float32").reshape(1, ny, nx)
    full = xr.DataArray(cube, dims=("band", "y", "x"), coords={"band": [1], "y": np.arange(ny), "x": np.arange(nx)})
    inst = _bound_inst(full.isel(y=slice(2, 5)), loc=None)  # scans 2-4 only, as a lat crop leaves it
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


def test_get_iirs_paths_finds_reprocessed_products(tmp_path):
    sid = "20210101T0000000000"
    geo = tmp_path / "geometry" / "recalibrated" / sid[:8] / sid
    dat = tmp_path / "data" / "recalibrated" / sid[:8] / sid
    geo.mkdir(parents=True)
    dat.mkdir(parents=True)
    (geo / f"{sid}.gcps").write_text("row,col,x,y,group\n")
    (geo / f"{sid}_loc.tif").write_bytes(b"")
    (geo / f"{sid}_obs.tif").write_bytes(b"")
    (dat / f"{sid}_l1_rad.tif").write_bytes(b"")

    paths = utils.get_iirs_paths(tmp_path, basenames=[sid], exts=("gcps", "loc", "obs", "tif"), level=1)
    assert paths["gcps"][sid] == geo / f"{sid}.gcps"
    assert paths["loc"][sid] == geo / f"{sid}_loc.tif"
    assert paths["obs"][sid] == geo / f"{sid}_obs.tif"
    assert paths["tif"][sid] == dat / f"{sid}_l1_rad.tif"

    paths2 = utils.get_iirs_paths(tmp_path, basenames=[sid], exts=("tif",), level=2)
    assert "tif" not in paths2
