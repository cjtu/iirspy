import numpy as np
import pytest
import xarray as xr

from iirspy import photometry as ph


def test_flat_topo_reproduces_cos_inc():
    """Zero slope must give back the per-line solar cosine, so `topo` only ever adds topography."""
    inc = np.array([60.0, 85.0, 89.0])
    mu0, mu, g = ph.topo_angles(0.0, 0.0, sun_az=137.0, sun_elev=90 - inc)
    np.testing.assert_allclose(mu0, np.cos(np.radians(inc)), atol=1e-12)
    np.testing.assert_allclose(mu, 1.0, atol=1e-12)  # nadir view over flat ground
    np.testing.assert_allclose(g, inc, atol=1e-9)  # nadir view: phase == incidence


def test_slope_toward_sun_gains_light_and_away_loses_it():
    sun_az, sun_elev = 90.0, 5.0  # low polar sun out of the east
    lit = ph.cos_angle(slope=10.0, aspect=90.0, az=sun_az, elev=sun_elev)  # facing the sun
    flat = ph.cos_angle(slope=0.0, aspect=0.0, az=sun_az, elev=sun_elev)
    away = ph.cos_angle(slope=10.0, aspect=270.0, az=sun_az, elev=sun_elev)  # facing away
    assert lit > flat > away
    assert away < 0  # a 10 deg back-slope under a 5 deg sun is unlit, not dim


@pytest.mark.parametrize("name", sorted(ph.MODELS))
def test_models_are_unity_at_reference_geometry(name):
    """Every disk function is normalized to 1 at overhead sun, nadir view, zero phase."""
    assert ph.get_model(name)(1.0, 1.0, 0.0) == pytest.approx(1.0)


def test_lunar_lambert_between_its_two_limits():
    mu0, mu, g = 0.2, 0.9, 60.0
    f = ph.lunar_lambert(mu0, mu, g)
    assert min(ph.lambert(mu0, mu, g), ph.lommel_seeliger(mu0, mu, g)) <= f
    assert f <= max(ph.lambert(mu0, mu, g), ph.lommel_seeliger(mu0, mu, g))


def test_get_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown photometric model"):
        ph.get_model("hapke")


def test_load_topo_checks_the_crop():
    like = xr.DataArray(np.zeros((4, 3)), dims=("y", "x"), coords={"y": np.arange(4), "x": np.arange(3)})
    slope, aspect, lit, sun = ph.load_topo((xr.zeros_like(like), xr.full_like(like, 90.0)), like=like)
    assert slope.shape == like.shape and float(aspect[0, 0]) == 90.0
    assert lit == 1.0  # a product without a lit band leaves the direct beam unattenuated
    assert sun is None  # a bare slope/aspect pair carries no sun frame, so L2 falls back to spm
    with pytest.raises(ValueError, match="does not match the image"):
        ph.load_topo((xr.zeros_like(like.isel(x=slice(0, 2))), xr.zeros_like(like.isel(x=slice(0, 2)))), like=like)


def _synthetic_l1(ny=4, nx=3):
    """A tiny radiance cube carrying the per-line solar geometry L2 reads off its coords."""
    rad = xr.DataArray(
        np.ones((2, ny, nx), dtype="float32"),
        dims=("band", "y", "x"),
        coords={
            "band": [10, 20],
            "wl": ("band", np.array([1000.0, 2000.0])),
            "y": np.arange(ny) + 100.0,  # absolute line offset, as a cropped cube has
            "x": np.arange(nx) + 0.5,
            "solar_inc": ("y", np.full(ny, 80.0)),
            "solar_az": ("y", np.full(ny, 210.0)),
        },
    )
    return rad


def _refl(rad, **kw):
    from iirspy.iirs import L2

    return L2.__new__(L2)._compute_reflectance(rad, None, None, solar_flux=1.0, **kw)


def test_flat_topo_matches_the_no_topo_reflectance():
    """The topo path with zero slope and no shadowing must reproduce the flat-sphere I/F."""
    rad = _synthetic_l1()
    flat = xr.zeros_like(rad.isel(band=0, drop=True))
    both = (_refl(rad), _refl(rad, topo=(flat, flat, xr.ones_like(flat))))
    np.testing.assert_allclose(both[0].values, both[1].values, rtol=1e-6)


def test_shadowed_and_backfacing_pixels_go_nan_not_huge():
    rad = _synthetic_l1()
    flat = xr.zeros_like(rad.isel(band=0, drop=True))
    lit = xr.ones_like(flat)
    lit[0, 0] = 0.0  # fully shadowed pixel
    slope = xr.zeros_like(flat)
    slope[1, 1] = 45.0  # steep facet turned away from a 10 deg sun
    aspect = xr.full_like(flat, 30.0)  # sun_az is 210, so this faces directly away
    refl = _refl(rad, topo=(slope, aspect, lit))
    assert np.isnan(refl.values[:, 0, 0]).all()
    assert np.isnan(refl.values[:, 1, 1]).all()
    assert np.isfinite(refl.values[:, 2, 2]).all()


def test_photom_model_changes_the_answer_and_takes_a_callable():
    rad = _synthetic_l1()
    flat = xr.zeros_like(rad.isel(band=0, drop=True))
    topo = (xr.full_like(flat, 20.0), xr.full_like(flat, 210.0), xr.ones_like(flat))
    lam = _refl(rad, topo=topo, photom="lambert")
    ll = _refl(rad, topo=topo, photom="lunar_lambert")
    assert not np.allclose(lam.values, ll.values)
    same = _refl(rad, topo=topo, photom=ph.lambert)
    np.testing.assert_allclose(lam.values, same.values)


def test_penumbra_is_dropped_by_the_min_lit_default():
    """min_lit defaults to 0.9: a half-blocked solar disk is nulled, and the cut is opt-outable."""
    rad = _synthetic_l1()
    flat = xr.zeros_like(rad.isel(band=0, drop=True))
    lit = xr.ones_like(flat)
    lit[0, 1] = 0.5
    refl = _refl(rad, topo=(flat, flat, lit))
    assert np.isnan(refl.values[:, 0, 1]).all()
    assert np.isfinite(refl.values[:, 2, 2]).all()
    assert np.isfinite(_refl(rad, topo=(flat, flat, lit), min_lit=0.0).values[:, 0, 1]).all()
