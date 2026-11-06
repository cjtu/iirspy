"""Round-trip tests: a saved cube must read back with the same data, band numbers and wavelengths.

Band numbers matter because streamed band subsets are not 1-indexed (e.g. {10, 18, 54}); a reader
that assumes 1..N silently attaches the wrong wavelength to every plane. Also covers crop -> save ->
from_file at every level, and the on-disk product path lookup.
"""

import numpy as np
import pytest
import xarray as xr

from iirspy import L0, L1, L2, instrument, io
from iirspy.io import apply_envi_start, band_numbers
from iirspy.l1 import radiometry

BASENAME = "20201202T1527488942"
FULL = [1, 2, 3]
SUBSET = [10, 18, 54]


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
            "wl": ("band", instrument.get_wls()[np.array(bands) - 1]),
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
    inst.save(fout, snr_sidecar=False, full_bands=False)

    back = xr.open_dataarray(fout, engine="rasterio")
    np.testing.assert_allclose(back.values, inst.img.values, rtol=1e-6)
    assert list(band_numbers(back)) == bands
    if ext == ".tif":  # ENVI BIL carries no transform, so absolute lines survive only in GeoTIFF
        np.testing.assert_allclose(back.y.values, inst.img.y.values)


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
@pytest.mark.parametrize("ext", [".tif", ".img"])
@pytest.mark.parametrize("bands", [FULL, SUBSET], ids=["full", "subset"])
def test_from_file_preserves_bands_wls_and_absolute_lines(bands, ext, tmp_path):
    inst = make(L1, bands, tmp_path)
    fout = tmp_path / f"roundtrip{ext}"
    inst.save(fout, snr_sidecar=False, full_bands=False)

    back = L1.from_file(fout, BASENAME, tmp_path)
    assert list(back.img.band.values) == bands
    np.testing.assert_allclose(back.img.wl.values, instrument.get_wls()[np.array(bands) - 1])
    np.testing.assert_allclose(back.img.values, inst.img.values * instrument.RAD_NATIVE_SCALE, rtol=1e-6)
    # Absolute line/sample offset survives: GeoTIFF via the transform, ENVI via x start / y start
    np.testing.assert_allclose(back.img.y.values, inst.img.y.values)
    np.testing.assert_allclose(back.img.x.values, inst.img.x.values)
    assert back.img.y.values[0] < back.img.y.values[-1], "lines must stay in acquisition order"


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
@pytest.mark.parametrize("ext", [".tif", ".img"])
def test_snr_sidecar_is_written_by_the_same_driver_as_its_cube(ext, tmp_path):
    """A sidecar written through a different driver than the cube it rides with cannot record where
    it starts -- and a camera-space product that cannot say its first scan is not addressable by a
    GLT."""
    inst = make(L1, FULL, tmp_path)
    ny, nx = inst.img.sizes["y"], inst.img.sizes["x"]
    snr = np.arange(ny * nx, dtype="float32").reshape(ny, nx)
    inst.img = inst.img.assign_coords(snr=(("y", "x"), snr))

    inst.save(tmp_path / f"cube{ext}")
    fsnr = tmp_path / f"cube_snr{ext}"
    assert fsnr.exists(), "the sidecar must take the cube's own format, not always ENVI"

    back = apply_envi_start(xr.open_dataarray(fsnr, engine="rasterio").sortby("y").sortby("x"))
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
    if ychunks:
        assert io.rows_per_block(back.img) == max(ychunks), "the cube's own y-chunk is the write block"
    assert io.rows_per_block(back.img, 7) == 7, "an explicit sub_rows still wins"

    fout2 = tmp_path / "cube2.tif"
    back.save(fout2, snr_sidecar=False)
    reread = xr.open_dataarray(fout2, engine="rasterio")
    np.testing.assert_allclose(reread.values, back.img.values, rtol=1e-6)


def _crop_cube():
    ny, nx = 8, 5
    data = np.arange(len(SUBSET) * ny * nx, dtype="float32").reshape(len(SUBSET), ny, nx) + 1
    return xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={
            "band": SUBSET,
            "y": np.arange(ny) + Y0 + 0.5,
            "x": np.arange(nx) + 0.5,
            "wl": ("band", instrument.get_wls()[np.array(SUBSET) - 1]),
            "solar_inc": ("y", np.arange(ny, dtype="float64")),
        },
    )


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
@pytest.mark.parametrize("cls", [L0, L1, L2], ids=["L0", "L1", "L2"])
def test_crop_save_from_file_roundtrip(cls, tmp_path):
    full = cls.from_xarray(_crop_cube(), BASENAME, tmp_path)
    sub = full.crop(y=(Y0 + 2.5, Y0 + 5.5), x=(1.5, 3.5), bands=[SUBSET[2], SUBSET[0]])
    assert type(sub) is cls and full.ny == 8, "crop returns a new object, original untouched"
    assert (sub.nband, sub.ny, sub.nx) == (2, 4, 3) and sub.extent == (1, 3, Y0 + 2, Y0 + 5)
    np.testing.assert_array_equal(sub.img.solar_inc.values, [2, 3, 4, 5])

    fout = sub.save(tmp_path / "crop.tif", snr_sidecar=False, full_bands=False)
    back = cls.from_file(fout, BASENAME, tmp_path)
    assert type(back) is cls
    assert list(back.img.band.values) == [SUBSET[2], SUBSET[0]]
    np.testing.assert_allclose(back.img.wl.values, sub.img.wl.values)
    np.testing.assert_allclose(back.img.y.values, sub.img.y.values)
    np.testing.assert_allclose(back.img.x.values, sub.img.x.values)
    scale = instrument.RAD_NATIVE_SCALE if cls is L1 else 1.0  # the L1 file is in native units
    np.testing.assert_allclose(back.img.values, sub.img.values * scale, rtol=1e-6)


def test_crop_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        L0.from_xarray(_crop_cube(), BASENAME, tmp_path).crop(y=(0, 1))


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning", "ignore:L1 files not found")
def test_cropped_l0_from_file_calibrates(tmp_path, monkeypatch):
    """gain 2, offset 1 (no label needed) -> rad = 10 * (2 DN + 1)."""
    coords = {"band": 1 + np.arange(256), "x": 0.5 + np.arange(250)}
    gain = xr.DataArray(np.full((256, 250), 2.0), coords=coords)
    sat = xr.DataArray(np.full(256, 1e9), coords={"band": coords["band"]})
    monkeypatch.setattr(radiometry, "get_gain_offset", lambda *a, **k: (gain, gain * 0 + 1))
    monkeypatch.setattr(instrument, "get_saturation_radiance", lambda *a, **k: sat)
    sub = L0.from_xarray(_crop_cube(), BASENAME, tmp_path).crop(y=(Y0 + 2.5, Y0 + 5.5), bands=[SUBSET[0], SUBSET[2]])
    l0 = L0.from_file(sub.save(tmp_path / "dn.tif", full_bands=False), BASENAME, tmp_path)
    kw = {
        "dark": "issdc_lut_median",
        "flat": None,
        "bad_pixel_mask": False,
        "fix_frames": False,
        "fill_frames": False,
        "interp_bands": None,
        "interp_spatial": False,
    }
    l1 = l0.calibrate(**kw)
    rad = l1.img
    keep = ~rad.band.isin((*instrument.OSF, *instrument.INVALID))
    expected = 10 * (2 * l0.img.where(keep, drop=True).values + 1)
    np.testing.assert_allclose(rad.where(keep, drop=True).values, expected, rtol=1e-6)
    assert isinstance(l1, L1)


def test_get_iirs_paths_finds_reprocessed_products(tmp_path):
    sid = "20210101T0000000000"
    geo = tmp_path / "geometry" / "recalibrated" / sid[:8]
    dat = tmp_path / "data" / "recalibrated" / sid[:8]
    der = tmp_path / "data" / "rederived" / sid[:8]
    for d in (geo, dat, der):
        d.mkdir(parents=True)
    p = f"{sid}_south"
    (geo / f"{p}.gcps").write_text("row,col,x,y\n")
    (geo / f"{p}_loc.tif").write_bytes(b"")
    (der / f"{p}_obs.tif").write_bytes(b"")
    (dat / f"{p}_l1_rad.tif").write_bytes(b"")

    paths = io.get_iirs_paths(tmp_path, basenames=[sid], exts=("gcps", "loc", "obs", "tif"), level=1)
    assert paths["gcps"][sid] == geo / f"{p}.gcps"
    assert paths["loc"][sid] == geo / f"{p}_loc.tif"
    assert paths["obs"][sid] == der / f"{p}_obs.tif"
    assert paths["tif"][sid] == dat / f"{p}_l1_rad.tif"

    paths2 = io.get_iirs_paths(tmp_path, basenames=[sid], exts=("tif",), level=2)
    assert "tif" not in paths2

    # a second group makes group=None ambiguous; naming the group resolves it
    (geo / f"{sid}_equatorial_loc.tif").write_bytes(b"")
    with pytest.raises(ValueError, match="pass group="):
        io.get_iirs_paths(tmp_path, basenames=[sid], exts=("loc",))
    paths3 = io.get_iirs_paths(tmp_path, basenames=[sid], exts=("loc",), group="equatorial")
    assert paths3["loc"][sid] == geo / f"{sid}_equatorial_loc.tif"
