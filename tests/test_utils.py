import numpy as np
import pytest

from iirspy import utils


def test_import_utils():
    assert True


@pytest.mark.parametrize("texp", [1.0, 3.0, 6.0, 12.0])
def test_gain_is_e1g2_scaled_by_exposure(monkeypatch, texp):
    """Gain is always the e1g2 LUT scaled by E1_EXPOSURE_MS / exposure_duration (no-op at 1 ms).

    The e2g2/e3g2/e4g2 lut_coeff tables are flat ~1.0 placeholders with no spectral structure;
    used as shipped they put radiance ~1000x high and null whole cubes at the saturation cut.
    Verified against the ISSDC nci L1 of e2g2 scene 20201203T1859574285 (ratio 333.6 = 1/3).
    """
    monkeypatch.setattr(utils, "get_exposure_duration", lambda _f: 1.0)
    ref, _ = utils.get_gain_offset("dummy")

    monkeypatch.setattr(utils, "get_exposure_duration", lambda _f: texp)
    gain, _ = utils.get_gain_offset("dummy")

    np.testing.assert_allclose(gain.values, ref.values * (utils.E1_EXPOSURE_MS / texp), rtol=1e-6)
    # The real e1g2 spectral response must survive; the placeholder tables are flat across band
    band_med = gain.median("x")
    assert float(band_med.max() / band_med.min()) > 100


def test_offset_is_not_exposure_scaled(monkeypatch):
    """Offset is a post-dark-subtraction bias, not accumulated signal, so exposure must not scale it."""
    monkeypatch.setattr(utils, "get_exposure_duration", lambda _f: 1.0)
    _, ref = utils.get_gain_offset("dummy")

    monkeypatch.setattr(utils, "get_exposure_duration", lambda _f: 3.0)
    _, off = utils.get_gain_offset("dummy")

    assert off.equals(ref)


def test_get_exposure_duration_reads_commanded_not_line(monkeypatch):
    """isda:exposure_duration (1/3/6/12 ms), not isda:line_exposure_duration (the 53.06 ms period)."""
    meta = {"isda:exposure_duration": "3", "isda:line_exposure_duration": "53.060"}
    monkeypatch.setattr(
        utils,
        "pdr",
        type("_pdr", (), {"open": staticmethod(lambda _f: type("_img", (), {"metaget": staticmethod(meta.get)})())}),
    )

    assert utils.get_exposure_duration("dummy") == 3.0
