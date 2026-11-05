"""L0 -> L1 -> L2 calibration: presets and LUTs, exposure scaling, pipeline provenance, config round-trips."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from iirspy import L0, instrument
from iirspy.l1 import calibrate, empirical, radiometry
from iirspy.l1.config import L1Config
from iirspy.l2 import calibrate as calibrate_l2
from iirspy.l2 import polish
from iirspy.l2.config import L2Config

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


def test_output_bands_window_matches_full_interpolation():
    """`l1.calibrate(output_bands=...)` narrows *before* `interpolate_na("band", max_gap=6)`.

    That's only safe because a band outside +-6 of the target could never have fed its
    interpolated value anyway (that's what max_gap=6 means) -- so narrowing first must give the
    same answer, for any target, as computing on the full band axis and slicing after. Exercise
    every gap width up to max_gap so a fix that only widens/narrows the reachable window would
    still be caught.
    """
    rng = np.random.default_rng(0)
    bands = list(range(1, 41))  # PAN_BANDS-sized axis, arbitrary values
    target = 20
    window = sorted(range(target - 6, target + 7))

    da = xr.DataArray(
        rng.normal(size=(len(bands), 5, 5)).astype("float32"),
        dims=("band", "y", "x"),
        coords={"band": bands},
    )
    for gap in range(0, 7):  # 0 = target itself masked; up to max_gap=6 either side
        masked = da.copy()
        masked.loc[{"band": [target - i for i in range(gap + 1) if target - i in bands]}] = np.nan

        full_then_slice = masked.interpolate_na("band", max_gap=6, method="linear").sel(band=[target])
        window_first = masked.sel(band=window).interpolate_na("band", max_gap=6, method="linear").sel(band=[target])

        np.testing.assert_array_equal(full_then_slice.values, window_first.values)


@pytest.mark.parametrize("texp", [1.0, 3.0, 6.0, 12.0])
def test_gain_is_e1g2_scaled_by_exposure(monkeypatch, texp):
    """Gain is always the e1g2 LUT scaled by E1_EXPOSURE_MS / exposure_duration (no-op at 1 ms).

    The e2g2/e3g2/e4g2 lut_coeff tables are flat ~1.0 placeholders with no spectral structure;
    used as shipped they put radiance ~1000x high and null whole cubes at the saturation cut.
    Verified against the ISSDC nci L1 of e2g2 scene 20201203T1859574285 (ratio 333.6 = 1/3).
    """
    monkeypatch.setattr(instrument, "get_exposure_duration", lambda _f: 1.0)
    ref, _ = radiometry.get_gain_offset("dummy")

    monkeypatch.setattr(instrument, "get_exposure_duration", lambda _f: texp)
    gain, _ = radiometry.get_gain_offset("dummy")

    np.testing.assert_allclose(gain.values, ref.values * (instrument.E1_EXPOSURE_MS / texp), rtol=1e-6)
    # The real e1g2 spectral response must survive; the placeholder tables are flat across band
    band_med = gain.median("x")
    assert float(band_med.max() / band_med.min()) > 100


def test_offset_is_not_exposure_scaled(monkeypatch):
    """Offset is a post-dark-subtraction bias, not accumulated signal, so exposure must not scale it."""
    monkeypatch.setattr(instrument, "get_exposure_duration", lambda _f: 1.0)
    _, ref = radiometry.get_gain_offset("dummy")

    monkeypatch.setattr(instrument, "get_exposure_duration", lambda _f: 3.0)
    _, off = radiometry.get_gain_offset("dummy")

    assert off.equals(ref)


def test_get_exposure_duration_reads_commanded_not_line(monkeypatch):
    """isda:exposure_duration (1/3/6/12 ms), not isda:line_exposure_duration (the 53.06 ms period)."""
    meta = {"isda:exposure_duration": "3", "isda:line_exposure_duration": "53.060"}
    monkeypatch.setattr(
        instrument,
        "pdr",
        type("_pdr", (), {"open": staticmethod(lambda _f: type("_img", (), {"metaget": staticmethod(meta.get)})())}),
    )

    assert instrument.get_exposure_duration("dummy") == 3.0


@pytest.fixture
def dn(monkeypatch):
    rng = np.random.default_rng(0)
    coords = {"band": 1 + np.arange(256), "x": 0.5 + np.arange(250)}
    gain = xr.DataArray(rng.uniform(1, 2, (256, 250)), coords=coords)
    sat = xr.DataArray(np.full(256, 1e9), coords={"band": coords["band"]})
    monkeypatch.setattr(radiometry, "get_gain_offset", lambda *a, **k: (gain, gain * 0))
    monkeypatch.setattr(instrument, "get_saturation_radiance", lambda *a, **k: sat)
    return xr.DataArray(
        rng.uniform(1, 100, (3, 6, 250)).astype("float32"),
        dims=("band", "y", "x"),
        coords={
            "band": BANDS,
            "y": np.arange(6) + 0.5,
            "x": coords["x"],
            "wl": ("band", instrument.get_wls()[np.array(BANDS) - 1]),
            "solar_inc": ("y", np.full(6, 60.0)),
        },
    )


def _steps(da):
    return [e["step"] for e in json.loads(da.attrs["provenance"])]


def test_every_active_step_leaves_one_provenance_entry_in_pipeline_order(dn):
    l1 = calibrate(dn, L1Config(calib="lut_median", bad_pixel_mask=False, interp_bands="linear"), qub="")
    assert _steps(l1) == ["radiometry", "mask", "fill"]
    l2 = calibrate_l2(l1, L2Config(flux=1.0, smoothing=3, smoothing_edges="nan", clip_negative=True))
    assert _steps(l2) == ["radiometry", "mask", "fill", "solar", "photometric", "smoothing", "clip"]


def test_thermal_without_thermal_bands_is_a_no_op(dn):
    l1 = calibrate(dn, L1Config(calib="lut_median", bad_pixel_mask=False), qub="")
    with pytest.warns(UserWarning, match="no thermal bands"):
        l2 = calibrate_l2(l1, L2Config(flux=1.0, thermal="isothermal"))
    assert "T" not in l2.coords and "thermal" not in _steps(l2)


def test_polish_maps_by_band_number_not_position():
    """A cube with bands dropped must still get each band's own coefficient."""
    table = polish.spectral_polish(np.arange(1, 257))
    kept = np.setdiff1d(np.arange(1, 257), [*instrument.OSF, *instrument.INVALID, 100, 101])
    np.testing.assert_array_equal(polish.spectral_polish(kept).values, table.sel(band=kept).values)
    assert float(table.sel(band=1)) == 1.0  # verma2022 starts at band 8


@pytest.mark.parametrize(
    "cfg",
    [
        *(L2Config.preset(n, flux=np.linspace(1, 2, 256), sdist=0.99) for n in L2Config.PRESETS),
        *(L1Config.preset(n, exclude_wl=[(1800, 1900)]) for n in L1Config.PRESETS),
    ],
)
def test_every_field_survives_to_attrs(cfg):
    assert type(cfg)(**cfg.to_attrs()) == cfg
    assert type(cfg).preset("iirspy", **cfg.to_attrs()) == cfg


@pytest.mark.parametrize(
    "make",
    [
        lambda: L2Config.preset("isdc"),
        lambda: L2Config.preset("issdc", polsh=None),
        lambda: L2Config(thermal="verma"),
        lambda: L2Config(flux="noaa"),
        lambda: L2Config(polish="issdc_empirical"),
        lambda: L2Config(sdist="spcie"),
        lambda: L2Config(photom="hapke"),
        lambda: L2Config(smoothing_edges="pairs"),
        lambda: L1Config.preset("issdc", empircal=True),
        lambda: L1Config.preset("nci"),
    ],
)
def test_unknown_preset_field_or_key_raises(make):
    with pytest.raises((ValueError, TypeError)):
        make()
