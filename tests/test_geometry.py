import numpy as np

from iirspy.georef import backplanes as bp
from iirspy.georef.geometry import Geometry, quality


def test_geometry_save_open_round_trip_is_lazy_and_exact(tmp_path):
    ny, nx, row0 = 6, 4, 137
    lon = np.tile(np.linspace(10.0, 13.0, nx), (ny, 1))
    lat = np.tile(np.linspace(-80.0, -77.0, ny)[:, None], (1, nx))
    radius = np.full((ny, nx), 1737400.0)
    obs = {"sun_zenith": np.full((ny, nx), 61.5), "facet_cos_i": np.full((ny, nx), 0.48)}
    qa = np.full((ny, nx), bp.qa_mask("no_geometry") | bp.qa_mask("no_topo"), "uint16")
    geo = Geometry.from_arrays(
        "20210101T0000000000",
        "south",
        row0,
        "issdc",
        {"lon": lon, "lat": lat, "radius": radius},
        obs,
        qa,
        quality("issdc", "warn"),
        tags={"SENSOR_SOURCE": "nadir"},
    )
    geo.save(tmp_path)

    back = Geometry.open(tmp_path, "20210101T0000000000", "south")
    assert back._loc is None and back._obs is None and back._qa is None
    assert (back.method, back.row0, back.quality["grade"]) == ("issdc", row0, "warn")
    np.testing.assert_allclose(back.loc.lon.values, lon, atol=1e-3)
    np.testing.assert_allclose(back.obs.sun_zenith.values, 61.5, atol=1e-5)
    assert np.isnan(back.obs.snr.values).all()
    np.testing.assert_array_equal(back.qa.values, qa)
    assert list(back.loc.y.values.astype(int)) == list(row0 + np.arange(ny))
