"""crop -> save -> from_file is the same at L0, L1 and L2, and a cropped L0 still calibrates."""

import numpy as np
import pytest
import xarray as xr

from iirspy import L0, L1, L2, utils

BASENAME = "20201202T1527488942"
BANDS = [10, 18, 54]
Y0 = 12000  # absolute first line, as a polar crop has


def cube():
    ny, nx = 8, 5
    data = np.arange(len(BANDS) * ny * nx, dtype="float32").reshape(len(BANDS), ny, nx) + 1
    return xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={
            "band": BANDS,
            "y": np.arange(ny) + Y0 + 0.5,
            "x": np.arange(nx) + 0.5,
            "wl": ("band", utils.get_wls()[np.array(BANDS) - 1]),
            "solar_inc": ("y", np.arange(ny, dtype="float64")),
        },
    )


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
@pytest.mark.parametrize("cls", [L0, L1, L2], ids=["L0", "L1", "L2"])
def test_crop_save_from_file_roundtrip(cls, tmp_path):
    full = cls.from_xarray(cube(), BASENAME, tmp_path)
    sub = full.crop(y=(Y0 + 2.5, Y0 + 5.5), x=(1.5, 3.5), bands=[BANDS[2], BANDS[0]])
    assert type(sub) is cls and full.ny == 8, "crop returns a new object, original untouched"
    assert (sub.nband, sub.ny, sub.nx) == (2, 4, 3) and sub.extent == (1, 3, Y0 + 2, Y0 + 5)
    np.testing.assert_array_equal(sub.img.solar_inc.values, [2, 3, 4, 5])

    fout = sub.save(tmp_path / "crop.tif", snr_sidecar=False, full_bands=False)
    back = cls.from_file(fout, BASENAME, tmp_path)
    assert type(back) is cls
    assert list(back.img.band.values) == [BANDS[2], BANDS[0]]
    np.testing.assert_allclose(back.img.wl.values, sub.img.wl.values)
    np.testing.assert_allclose(back.img.y.values, sub.img.y.values)
    np.testing.assert_allclose(back.img.x.values, sub.img.x.values)
    scale = utils.RAD_NATIVE_SCALE if cls is L1 else 1.0  # the L1 file is in native units
    np.testing.assert_allclose(back.img.values, sub.img.values * scale, rtol=1e-6)


def test_crop_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        L0.from_xarray(cube(), BASENAME, tmp_path).crop(y=(0, 1))


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning", "ignore:L1 files not found")
def test_cropped_l0_from_file_calibrates(tmp_path, monkeypatch):
    """gain 2, offset 1 (no label needed) -> rad = 10 * (2 DN + 1)."""
    coords = {"band": 1 + np.arange(256), "x": 0.5 + np.arange(250)}
    gain = xr.DataArray(np.full((256, 250), 2.0), coords=coords)
    sat = xr.DataArray(np.full(256, 1e9), coords={"band": coords["band"]})
    monkeypatch.setattr(utils, "get_gain_offset", lambda *a, **k: (gain, gain * 0 + 1))
    monkeypatch.setattr(utils, "get_saturation_radiance", lambda *a, **k: sat)
    sub = L0.from_xarray(cube(), BASENAME, tmp_path).crop(y=(Y0 + 2.5, Y0 + 5.5), bands=[BANDS[0], BANDS[2]])
    l0 = L0.from_file(sub.save(tmp_path / "dn.tif", full_bands=False), BASENAME, tmp_path)
    kw = {
        "empirical": False,
        "bad_pixel_mask": False,
        "fix_frames": False,
        "fill_frames": False,
        "interp_bands": None,
        "interp_spatial": False,
    }
    rad = l0.calibrate_to_rad(**kw)
    keep = ~rad.band.isin((*utils.OSF, *utils.INVALID))
    expected = 10 * (2 * l0.img.where(keep, drop=True).values + 1)
    np.testing.assert_allclose(rad.where(keep, drop=True).values, expected, rtol=1e-6)
    assert isinstance(l0.calibrate(**kw), L1)
