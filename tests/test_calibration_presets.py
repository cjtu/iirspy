"""Calibration presets: lut_median and calib LUT calibration, HSRS flux with distance scaling, refl CLI configs."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from iirspy import L0, instrument
from iirspy.cli import refl
from iirspy.l1 import calibrate, empirical, radiometry
from iirspy.l1.config import L1Config
from iirspy.l2 import solar

BASENAME = "20201202T1527488942"
BANDS = [10, 18, 54]
LUT_ONLY_KW = {"bad_pixel_mask": False, "fix_frames": False, "attach_empirical_frames": False}


@pytest.fixture
def l0(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    coords = {"band": 1 + np.arange(256), "x": 0.5 + np.arange(250)}
    gain = xr.DataArray(rng.uniform(1, 2, (256, 250)), coords=coords)
    offset = xr.DataArray(rng.uniform(-1, 1, (256, 250)), coords=coords)
    sat = xr.DataArray(np.full(256, 1e9), coords={"band": coords["band"]})
    monkeypatch.setattr(radiometry, "get_gain_offset", lambda *a, **k: (gain, offset))
    monkeypatch.setattr(instrument, "get_saturation_radiance", lambda *a, **k: sat)
    monkeypatch.setattr(radiometry.pdr, "open", lambda f: SimpleNamespace(metaget=lambda k: "2020-12-02T15:27:48"))
    monkeypatch.setattr(instrument, "get_exposure_duration", lambda *a: 1.0)
    dn = xr.DataArray(
        rng.uniform(1, 100, (3, 6, 250)).astype("float32"),
        dims=("band", "y", "x"),
        coords={
            "band": BANDS,
            "y": np.arange(6) + 0.5,
            "x": coords["x"],
            "wl": ("band", instrument.get_wls()[np.array(BANDS) - 1]),
        },
    )
    return L0.from_xarray(dn, BASENAME, tmp_path), gain, offset


def _cal(l0, **kw):
    cfg = L1Config.resolve(
        "iirspy", **{**LUT_ONLY_KW, "empirical": False, "interp_bands": None, "interp_spatial": False, **kw}
    )
    return calibrate(l0.img, cfg, qub=l0.qub)


def test_lut_median_and_identity_calib_lut(l0, tmp_path):
    obj, gain, offset = l0
    rad = _cal(obj, calib="lut_median")
    gm, om = gain.median("x"), offset.median("x")
    np.testing.assert_allclose(rad.values, (10 * (obj.img * gm + om)).values, rtol=1e-6)

    ngain = np.stack([np.broadcast_to(gm.values[:, None], (256, 250))] * 2)
    ft = tmp_path / "iirspy_recal_e1g2_lut_coeff.npz"
    np.savez(
        ft, gain=ngain * 5, offset=np.stack([om.values] * 2) * 5, epoch_start=["2019-01-01", "2020-12-03"], version="t"
    )  # epoch 2 starts after the scene: must not be picked
    np.savez(
        ft.with_name("iirspy_recal_e1g2_lut_coeff_b.npz"),
        gain=ngain[:1],
        offset=om.values,
        epoch_start=["2019-01-01"],
        version="t",
    )
    rad_t = _cal(obj, calib=str(ft.with_name("iirspy_recal_e1g2_lut_coeff_b.npz")))
    np.testing.assert_allclose(rad_t.values, rad.values, rtol=1e-6)
    assert rad_t.attrs["calib_lut_version"] == "t" and len(rad_t.attrs["calib_lut_sha256"]) == 64
    assert radiometry._read_calib_lut(ft, pd.Timestamp("2020-12-02T15:27:48Z"))[3]["calib_lut_epoch"] == "2019-01-01"
    with pytest.MonkeyPatch.context() as mp:  # table gain is in e1 (1 ms) units, scaled by exposure like the LUT
        mp.setattr(instrument, "get_exposure_duration", lambda *a: 10.0)
        rad_e3 = _cal(obj, calib=str(ft.with_name("iirspy_recal_e1g2_lut_coeff_b.npz")))
    np.testing.assert_allclose(rad_e3.values, (10 * (obj.img * gm / 10 + om)).values, rtol=1e-6)
    with pytest.raises(ValueError, match="precedes"):
        np.savez(ft, gain=ngain[:1], offset=om.values, epoch_start=["2021-01-01"])
        _cal(obj, calib=str(ft))


def test_calib_lut_mask_union_and_open_ended_warning(tmp_path):
    m = np.zeros((3, 256, 250), bool)
    m[:, 5, 7] = True
    ft = tmp_path / "iirspy_recal_e1g2_lut_coeff_mask.npz"
    np.savez(
        ft,
        gain=np.ones((3, 256, 250)),
        offset=np.zeros(256),
        mask=m,
        epoch_start=["2019-01-01", "2019-07-01", "2020-01-01"],
    )
    tm = radiometry._read_calib_lut(ft, pd.Timestamp("2020-01-02"))[2]
    assert bool((instrument.load_bad_pixel_mask() | tm)[5, 7])
    with pytest.warns(UserWarning, match="past the last epoch"):
        radiometry._read_calib_lut(ft, pd.Timestamp("2021-01-01"))


def test_attach_empirical_frames_does_not_apply(l0, monkeypatch):
    obj, *_ = l0
    ones = xr.DataArray(np.full((3, 250), 0.5), coords={"band": BANDS, "x": obj.img.x.values})
    monkeypatch.setattr(empirical, "load_reference_flat", lambda *a, **k: None)
    monkeypatch.setattr(
        empirical, "empirical_frames", lambda *a, **k: (ones * 0, ones, ones, None, {"has_shadow": False})
    )
    base = _cal(obj, calib="lut_median")
    att = _cal(obj, calib="lut_median", attach_empirical_frames=True)
    assert {"empirical_dark", "empirical_flat", "empirical_smile"} <= set(att.coords) and "empirical_notes" in att.attrs
    np.testing.assert_array_equal(att.values, base.values)


def test_hsrs_flux_distance_scaling():
    d = 1.3
    one, far = solar.get_solar_flux(1.0, solar.FSOLAR_HSRS), solar.get_solar_flux(d, solar.FSOLAR_HSRS)
    assert one.size == 256
    np.testing.assert_allclose(far.values, one.values / d**2)


def test_refl_cli_resolves_configs(tmp_path):
    args = refl._parser().parse_args([
        "sid",
        "--group",
        "equatorial",
        "--l1",
        "lut-only",
        "--l1-calib",
        "lut_median",
        "--l2",
        "l2d",
    ])
    c1, c2 = refl.l1_config(args), refl.l2_config(args)
    assert c1 == L1Config.preset("lut-only") and c1.attach_empirical_frames and not c1.empirical
    assert c2.flux == "hsrs" and c2.thermal == "isothermal" and not c2.clip_negative
    with pytest.raises(ValueError, match="empirical"):
        L1Config(empirical=True, calib="lut_median")
