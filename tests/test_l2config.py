import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from iirspy import l2config, utils
from iirspy.l2config import L2Config

L2_REPRO = Path(os.environ.get("IIRS_L2_REPRO", Path.home() / "data/moon/ch2_iirs/l2_repro"))


def test_polish_maps_by_band_number_not_position():
    """A cube with bands dropped must still get each band's own coefficient."""
    table = l2config.polish(L2Config(polish="verma2022"), np.arange(1, 257))
    kept = np.setdiff1d(np.arange(1, 257), [*utils.OSF, *utils.INVALID, 100, 101])
    sub = l2config.polish(L2Config(polish="verma2022"), kept)
    np.testing.assert_array_equal(sub.values, table.sel(band=kept).values)
    assert float(table.sel(band=1)) == 1.0  # verma2022 starts at band 8


@pytest.mark.parametrize("name", sorted(l2config.PRESETS))
def test_every_field_survives_to_attrs(name):
    cfg = L2Config.preset(name, flux=np.linspace(1, 2, 256), sdist=0.99)
    assert L2Config(**cfg.to_attrs()) == cfg
    assert L2Config.preset("iirspy", **cfg.to_attrs()) == cfg


@pytest.mark.parametrize(
    "make",
    [
        lambda: L2Config.preset("isdc"),
        lambda: L2Config.preset("issdc", polsh=None),
        lambda: L2Config(thermal="verma"),
        lambda: L2Config(flux="noaa"),
        lambda: L2Config(polish="issdc"),
        lambda: L2Config(sdist="spcie"),
        lambda: L2Config(photom="hapke"),
        lambda: L2Config(smoothing_edges="pairs"),
    ],
)
def test_unknown_preset_field_or_key_raises(make):
    with pytest.raises((ValueError, TypeError)):
        make()


def _align(ndi, cands, pad, n, band=48):
    """The candidate window and line offset that best correlates with ndi's core lines."""
    ref = ndi[band, pad : pad + n, ::5].ravel()
    best = (-2.0, None)
    for rad in cands:
        for d in range(-pad, pad + 1):
            x = rad[band, pad + d : pad + d + n, ::5].ravel()
            ok = np.isfinite(ref) & np.isfinite(x)
            c = np.corrcoef(ref[ok], x[ok])[0, 1] if ok.sum() > 100 else -1.0
            if c > best[0]:
                best = (c, rad[:, pad + d : pad + d + n])
    return best[1]


@pytest.mark.skipif(not L2_REPRO.is_dir(), reason="needs the local l2_repro NCI/NDI window cache")
def test_issdc_preset_reproduces_ndi():
    from iirspy.iirs import L2

    pad, n = 8, 256
    wl = utils.get_wls()
    short = (wl > 750) & (wl < 2300)
    meds = []
    for d in sorted(p for p in L2_REPRO.iterdir() if (p / "nci.npz").exists() and (p / "ndi.npz").exists()):
        nci, ndi = np.load(d / "nci.npz"), np.load(d / "ndi.npz")
        rfl, sunzen = ndi["rfl"], ndi["obs"][:, 1]  # each npz key access re-reads the whole array
        rads = [nci[k] for k in ("rad", "rad_flip") if k in nci]
        for w in range(len(rfl)):
            ref = rfl[w][:, pad : pad + n]
            rad = _align(rfl[w], [r[w] for r in rads], pad, n)
            ny, nx = ref.shape[1:]
            coords = {"band": np.arange(1, 257), "wl": ("band", wl), "y": np.arange(ny), "x": np.arange(nx)}
            da = xr.DataArray(rad * utils.RAD_NATIVE_SCALE, dims=("band", "y", "x"), coords=coords)
            da = da.assign_coords(solar_inc=("y", np.zeros(ny)), solar_az=("y", np.zeros(ny)))
            inc = xr.DataArray(sunzen[w, pad : pad + n], dims=("y", "x"))
            refl = L2.__new__(L2)._compute_reflectance(da, None, None, L2Config.preset("issdc"), inc=inc).values
            ok = np.isfinite(refl) & np.isfinite(ref) & (ref > 0.01)
            ratio = np.where(ok, refl / np.where(ok, ref, 1), np.nan)
            meds.append(np.nanmedian(np.nanmedian(ratio.reshape(256, -1), 1)[short]))
    assert meds, f"no scene in {L2_REPRO} has both nci.npz and ndi.npz"
    np.testing.assert_allclose(meds, 1, atol=0.01)  # each window, median over 750-2300 nm
