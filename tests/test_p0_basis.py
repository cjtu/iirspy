"""P0 basis: lut_median and G1-table calibration, HSRS flux with distance scaling, refl CLI configs."""

import numpy as np
import pytest
import xarray as xr

from iirspy import L0, iirs, refl, utils
from iirspy.config import L1Config

BASENAME = "20201202T1527488942"
BANDS = [10, 18, 54]
P0_KW = {"bad_pixel_mask": False, "fix_frames": False, "attach_empirical_frames": False}


@pytest.fixture
def l0(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    coords = {"band": 1 + np.arange(256), "x": 0.5 + np.arange(250)}
    gain = xr.DataArray(rng.uniform(1, 2, (256, 250)), coords=coords)
    offset = xr.DataArray(rng.uniform(-1, 1, (256, 250)), coords=coords)
    sat = xr.DataArray(np.full(256, 1e9), coords={"band": coords["band"]})
    monkeypatch.setattr(utils, "get_gain_offset", lambda *a, **k: (gain, offset))
    monkeypatch.setattr(utils, "get_saturation_radiance", lambda *a, **k: sat)
    monkeypatch.setattr(L0, "_start_time", lambda self: "2020-12-02T15:27:48")
    dn = xr.DataArray(
        rng.uniform(1, 100, (3, 6, 250)).astype("float32"),
        dims=("band", "y", "x"),
        coords={
            "band": BANDS,
            "y": np.arange(6) + 0.5,
            "x": coords["x"],
            "wl": ("band", utils.get_wls()[np.array(BANDS) - 1]),
        },
    )
    return L0.from_xarray(dn, BASENAME, tmp_path), gain, offset


def _cal(l0, **kw):
    return l0.calibrate_to_rad(**{**P0_KW, "empirical": False, "interp_bands": None, "interp_spatial": False, **kw})


def test_p0_and_identity_g1_table(l0, tmp_path):
    obj, gain, offset = l0
    rad = _cal(obj, calib="lut_median")
    gm, om = gain.median("x"), offset.median("x")
    np.testing.assert_allclose(rad.values, (10 * (obj.img * gm + om)).values, rtol=1e-6)

    ngain = np.stack([np.broadcast_to(gm.values[:, None], (256, 250))] * 2)
    ft = tmp_path / "g1.npz"
    np.savez(
        ft, gain=ngain * 5, offset=np.stack([om.values] * 2) * 5, epoch_start=["2019-01-01", "2020-12-03"], version="t"
    )  # epoch 2 starts after the scene: must not be picked
    np.savez(ft.with_name("g1b.npz"), gain=ngain[:1], offset=om.values, epoch_start=["2019-01-01"], version="t")
    rad_t = _cal(obj, calib=str(ft.with_name("g1b.npz")))
    np.testing.assert_allclose(rad_t.values, rad.values, rtol=1e-6)
    assert rad_t.attrs["calib_table_version"] == "t" and len(rad_t.attrs["calib_table_sha256"]) == 64
    with pytest.raises(ValueError, match="precedes"):
        np.savez(ft, gain=ngain[:1], offset=om.values, epoch_start=["2021-01-01"])
        _cal(obj, calib=str(ft))


def test_attach_empirical_frames_does_not_apply(l0, monkeypatch):
    obj, *_ = l0
    ones = xr.DataArray(np.full((3, 250), 0.5), coords={"band": BANDS, "x": obj.img.x.values})
    monkeypatch.setattr(utils, "load_reference_flat", lambda *a, **k: None)
    monkeypatch.setattr(iirs, "empirical_frames", lambda *a, **k: (ones * 0, ones, ones, None, {"note": 1}))
    base = _cal(obj, calib="lut_median")
    att = _cal(obj, calib="lut_median", attach_empirical_frames=True)
    assert {"empirical_dark", "empirical_flat", "empirical_smile"} <= set(att.coords) and "empirical_notes" in att.attrs
    np.testing.assert_array_equal(att.values, base.values)


def test_hsrs_flux_distance_scaling():
    d = 1.3
    one, far = utils.get_solar_flux(1.0, utils.FSOLAR_HSRS), utils.get_solar_flux(d, utils.FSOLAR_HSRS)
    assert one.size == 256
    np.testing.assert_allclose(far.values, one.values / d**2)


def test_refl_cli_resolves_configs(tmp_path):
    args = refl._parser().parse_args([
        "sid",
        "--group",
        "equatorial",
        "--l1",
        "p0",
        "--l1-calib",
        "lut_median",
        "--l2",
        "l2d",
    ])
    c1, c2 = refl.l1_config(args), refl.l2_config(args)
    assert c1 == L1Config.preset("p0") and c1.attach_empirical_frames and not c1.empirical
    assert c2.flux == "hsrs" and c2.thermal == "isothermal" and not c2.clip_negative
    with pytest.raises(ValueError, match="empirical"):
        L1Config(empirical=True, calib="lut_median")
