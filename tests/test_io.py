"""Round-trip tests: a saved cube must read back with the same data, band numbers and wavelengths.

Band numbers matter because streamed band subsets are not 1-indexed (e.g. {10, 18, 54}); a reader
that assumes 1..N silently attaches the wrong wavelength to every plane.
"""

import numpy as np
import pytest
import xarray as xr

from iirspy import L0, L1, L2, utils
from iirspy.iirs import _apply_envi_start, _band_numbers, _row_block

BASENAME = "20201202T1527488942"
FULL = [1, 2, 3]
SUBSET = [10, 18, 54]  # as fetch_nci_ancillary.py streams them


Y0 = 12000  # absolute first line, as a polar crop has: keeps the transform off Affine.identity


def cube(bands):
    """Small (band, y, x) float32 cube with IIRS band numbers, true wavelengths and absolute lines."""
    ny, nx = 6, 5
    data = np.arange(len(bands) * ny * nx, dtype="float32").reshape(len(bands), ny, nx)
    return xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={
            "band": bands,
            "y": np.arange(ny) + Y0 + 0.5,
            "x": np.arange(nx) + 0.5,
            "wl": ("band", utils.get_wls()[np.array(bands) - 1]),
        },
    )


def make(cls, bands, directory):
    """An instance of cls wrapping cube(bands), without needing bundle data on disk."""
    if cls is L0:
        inst = L0.__new__(L0)
        inst.basename, inst.directory, inst.level = BASENAME, directory, 0
        inst.img = cube(bands)
        inst.shape = inst.img.shape
        inst.nband, inst.ny, inst.nx = inst.shape
        return inst
    return cls.from_xarray(cube(bands), BASENAME, directory)


# ENVI BIL carries no map info by design; the y-coord assertion below guards the GeoTIFF transform.
@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
@pytest.mark.parametrize("cls", [L0, L1, L2], ids=["L0", "L1", "L2"])
@pytest.mark.parametrize("ext", [".tif", ".img"])
@pytest.mark.parametrize("bands", [FULL, SUBSET], ids=["full", "subset"])
def test_roundtrip_preserves_bands_and_data(cls, ext, bands, tmp_path):
    inst = make(cls, bands, tmp_path)
    fout = tmp_path / f"roundtrip{ext}"
    inst.save(fout, snr_sidecar=False)

    back = xr.open_dataarray(fout, engine="rasterio")
    np.testing.assert_allclose(back.values, inst.img.values, rtol=1e-6)
    assert list(_band_numbers(back)) == bands
    if ext == ".tif":  # ENVI BIL carries no transform, so absolute lines survive only in GeoTIFF
        np.testing.assert_allclose(back.y.values, inst.img.y.values)


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
@pytest.mark.parametrize("ext", [".tif", ".img"])
@pytest.mark.parametrize("bands", [FULL, SUBSET], ids=["full", "subset"])
def test_from_file_preserves_bands_wls_and_absolute_lines(bands, ext, tmp_path):
    inst = make(L1, bands, tmp_path)
    fout = tmp_path / f"roundtrip{ext}"
    inst.save(fout, snr_sidecar=False)

    back = L1.from_file(fout, BASENAME, tmp_path)
    assert list(back.img.band.values) == bands
    np.testing.assert_allclose(back.img.wl.values, utils.get_wls()[np.array(bands) - 1])
    np.testing.assert_allclose(back.img.values, inst.img.values * utils.RAD_NATIVE_SCALE, rtol=1e-6)
    # Absolute line/sample offset survives: GeoTIFF via the transform, ENVI via x start / y start
    np.testing.assert_allclose(back.img.y.values, inst.img.y.values)
    np.testing.assert_allclose(back.img.x.values, inst.img.x.values)
    assert back.img.y.values[0] < back.img.y.values[-1], "lines must stay in acquisition order"


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
@pytest.mark.parametrize("ext", [".tif", ".img"])
def test_snr_sidecar_is_written_by_the_same_driver_as_its_cube(ext, tmp_path):
    """A sidecar written through a different driver than the cube it rides with cannot record where
    it starts -- and a camera-space product that cannot say its first scan is not addressable by a
    GLT. Derived products (mineral parameters, OHIBD) go out the same way, so the dispatch is
    shared rather than reimplemented per product."""
    inst = make(L1, FULL, tmp_path)
    ny, nx = inst.img.sizes["y"], inst.img.sizes["x"]
    snr = np.arange(ny * nx, dtype="float32").reshape(ny, nx)
    inst.img = inst.img.assign_coords(snr=(("y", "x"), snr))

    inst.save(tmp_path / f"cube{ext}")
    fsnr = tmp_path / f"cube_snr{ext}"
    assert fsnr.exists(), "the sidecar must take the cube's own format, not always ENVI"

    back = _apply_envi_start(xr.open_dataarray(fsnr, engine="rasterio").sortby("y").sortby("x"))
    np.testing.assert_allclose(back.y.values, inst.img.y.values)
    np.testing.assert_allclose(back.x.values, inst.img.x.values)
    np.testing.assert_allclose(np.asarray(back.values).reshape(ny, nx), snr)


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
@pytest.mark.parametrize("chunk", [True, False, {"band": -1, "y": 2, "x": -1}], ids=["auto", "off", "explicit"])
def test_write_block_tracks_chunking_and_pixels_do_not(chunk, tmp_path):
    """One y-block policy, not two: a write step is the cube's own dask chunk unless overridden.

    Unaligned, every write block straddles two compute blocks and re-reads them. The pixels must
    not care either way -- the file bytes can differ, since a differently-windowed write packs the
    GeoTIFF's internal tiles differently.
    """
    inst = make(L1, FULL, tmp_path)
    fout = tmp_path / "cube.tif"
    inst.save(fout, snr_sidecar=False)

    back = L1.from_file(fout, BASENAME, tmp_path, chunk=chunk)
    ychunks = back.img.chunks[back.img.dims.index("y")] if back.img.chunks else None
    assert _row_block(back.img) == (max(ychunks) if ychunks else _row_block(back.img.compute()))
    assert _row_block(back.img, 7) == 7, "an explicit sub_rows still wins"

    fout2 = tmp_path / "cube2.tif"
    back.save(fout2, snr_sidecar=False)
    reread = xr.open_dataarray(fout2, engine="rasterio")
    np.testing.assert_allclose(reread.values, back.img.values, rtol=1e-6)
