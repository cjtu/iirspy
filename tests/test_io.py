"""Round-trip tests: a saved cube must read back with the same data, band numbers and wavelengths.

Band numbers matter because streamed band subsets are not 1-indexed (e.g. {10, 18, 54}); a reader
that assumes 1..N silently attaches the wrong wavelength to every plane.
"""

import numpy as np
import pytest
import xarray as xr

from iirspy import L0, L1, L2, utils
from iirspy.iirs import _band_numbers

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
