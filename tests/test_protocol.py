import json

import numpy as np
import pytest
import xarray as xr

from iirspy import instrument
from iirspy.l1 import calibrate as calibrate_l1
from iirspy.l1 import radiometry
from iirspy.l1.config import L1Config
from iirspy.l2 import calibrate as calibrate_l2
from iirspy.l2.config import L2Config

BANDS = [10, 18, 54]


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
    l1 = calibrate_l1(dn, L1Config(calib="lut_median", bad_pixel_mask=False, interp_bands="linear"), qub="")
    assert _steps(l1) == ["radiometry", "mask", "fill"]
    l2 = calibrate_l2(l1, L2Config(flux=1.0, smoothing=3, smoothing_edges="nan", clip_negative=True))
    assert _steps(l2) == ["radiometry", "mask", "fill", "solar", "photometric", "smoothing", "clip"]


def test_thermal_without_thermal_bands_is_a_no_op(dn):
    l1 = calibrate_l1(dn, L1Config(calib="lut_median", bad_pixel_mask=False), qub="")
    with pytest.warns(UserWarning, match="no thermal bands"):
        l2 = calibrate_l2(l1, L2Config(flux=1.0, thermal="isothermal"))
    assert "T" not in l2.coords and "thermal" not in _steps(l2)
