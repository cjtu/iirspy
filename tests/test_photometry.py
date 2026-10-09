import json

import numpy as np
import pytest
import xarray as xr

from iirspy import photometry as ph


def test_flat_slope_az_reproduces_cos_inc():
    """Zero slope must give back the per-line solar cosine, so `slope_az` only ever adds topography."""
    inc = np.array([60.0, 85.0, 89.0])
    cos_inc, cos_view, g = ph.local_angles(0.0, 0.0, sun_az=137.0, sun_elev=90 - inc)
    np.testing.assert_allclose(cos_inc, np.cos(np.radians(inc)), atol=1e-12)
    np.testing.assert_allclose(cos_view, 1.0, atol=1e-12)  # nadir view over flat ground
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
    cos_inc, cos_view, g = 0.2, 0.9, 60.0
    f = ph.lunar_lambert(cos_inc, cos_view, g)
    assert min(ph.lambert(cos_inc, cos_view, g), ph.lommel_seeliger(cos_inc, cos_view, g)) <= f
    assert f <= max(ph.lambert(cos_inc, cos_view, g), ph.lommel_seeliger(cos_inc, cos_view, g))


def test_get_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown photometric model"):
        ph.get_model("hapke")


def test_load_slope_az_checks_the_crop():
    like = xr.DataArray(np.zeros((4, 3)), dims=("y", "x"), coords={"y": np.arange(4), "x": np.arange(3)})
    slope, aspect, lit, sun = ph.load_slope_az((xr.zeros_like(like), xr.full_like(like, 90.0)), like=like)
    assert slope.shape == like.shape and float(aspect[0, 0]) == 90.0
    assert lit == 1.0  # a product without a lit band leaves the direct beam unattenuated
    assert sun is None  # a bare slope/aspect pair carries no sun frame, so L2 falls back to spm
    with pytest.raises(ValueError, match="does not match the image"):
        ph.load_slope_az((xr.zeros_like(like.isel(x=slice(0, 2))), xr.zeros_like(like.isel(x=slice(0, 2)))), like=like)


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


def _refl(rad, slope_az=None, **kw):
    from iirspy.config import L2Config
    from iirspy.iirs import L2

    return L2.__new__(L2)._compute_reflectance(rad, None, L2Config(flux=1.0, **kw), slope_az=slope_az)


def test_flat_slope_az_matches_the_no_slope_az_reflectance():
    """The slope_az path with zero slope and no shadowing must reproduce the flat-sphere I/F."""
    rad = _synthetic_l1()
    flat = xr.zeros_like(rad.isel(band=0, drop=True))
    both = (_refl(rad), _refl(rad, slope_az=(flat, flat, xr.ones_like(flat))))
    np.testing.assert_allclose(both[0].values, both[1].values, rtol=1e-6)


def test_shadowed_and_backfacing_pixels_go_nan_not_huge():
    rad = _synthetic_l1()
    flat = xr.zeros_like(rad.isel(band=0, drop=True))
    lit = xr.ones_like(flat)
    lit[0, 0] = 0.0  # fully shadowed pixel
    slope = xr.zeros_like(flat)
    slope[1, 1] = 45.0  # steep facet turned away from a 10 deg sun
    aspect = xr.full_like(flat, 30.0)  # sun_az is 210, so this faces directly away
    refl = _refl(rad, slope_az=(slope, aspect, lit))
    assert np.isnan(refl.values[:, 0, 0]).all()
    assert np.isnan(refl.values[:, 1, 1]).all()
    assert np.isfinite(refl.values[:, 2, 2]).all()


def test_photom_model_changes_the_answer_and_takes_a_callable():
    rad = _synthetic_l1()
    flat = xr.zeros_like(rad.isel(band=0, drop=True))
    slope_az = (xr.full_like(flat, 20.0), xr.full_like(flat, 210.0), xr.ones_like(flat))
    lam = _refl(rad, slope_az=slope_az, photom="lambert")
    ll = _refl(rad, slope_az=slope_az, photom="lunar_lambert")
    assert not np.allclose(lam.values, ll.values)
    same = _refl(rad, slope_az=slope_az, photom=ph.lambert)
    np.testing.assert_allclose(lam.values, same.values)


COS_85 = np.cos(np.radians(85.0))


def test_cos_min_floors_grazing_incidence_without_changing_default():
    """cos_min=0 is a no-op; a nonzero floor caps the I/F of a near-terminator pixel."""
    rad = _synthetic_l1()  # solar_inc=80 deg everywhere -> cos_inc ~ 0.17, well above any floor here
    flat = xr.zeros_like(rad.isel(band=0, drop=True))
    baseline = _refl(rad, min_lit=0.0)
    np.testing.assert_allclose(baseline.values, _refl(rad, min_lit=0.0, cos_min=0.0).values, rtol=0, atol=0)

    grazing = rad.copy()
    grazing["solar_inc"] = ("y", np.full(grazing.sizes["y"], 89.9))  # cos_inc ~ 0.0017, near-terminator
    unclamped = _refl(grazing, min_lit=0.0)
    clamped = _refl(grazing, min_lit=0.0, cos_min=0.05)
    assert np.all(np.abs(clamped.values) <= np.abs(unclamped.values) + 1e-9)
    assert np.abs(clamped.values).max() <= 1.0 / 0.05 + 1e-6
    assert json.loads(clamped.attrs["l2_config"])["cos_min"] == 0.05
    assert clamped.attrs["cos_inc_floor_frac"] == 1.0  # every pixel here is below the floor
    assert clamped.attrs["cos_view_floor_frac"] == 0.0  # nadir view (no slope_az view tilt) here
    assert (
        _refl(rad, min_lit=0.0, slope_az=(flat, flat, xr.ones_like(flat)), cos_min=0.0).attrs["cos_inc_floor_frac"]
        == 0.0
    )


def _cos_view_dependent_photom(cos_inc, cos_view, g):
    """cos_inc*cos_view -- unlike lambert/lommel_seeliger/lunar_lambert this actually diverges as cos_view -> 0,
    so it exercises what an emission floor is for even though none of the built-in models do."""
    return cos_inc * cos_view


def test_cos_min_floors_grazing_emission_independently_of_incidence():
    """A steep facet drives cos_view (emission cosine) toward 0 under nadir view; cos_min clamps it too."""
    rad = _synthetic_l1()
    rad["solar_inc"] = ("y", np.full(rad.sizes["y"], 10.0))  # near-overhead sun: cos_inc stays high
    flat = xr.zeros_like(rad.isel(band=0, drop=True))
    steep = xr.full_like(flat, 89.0)  # cos_view = cos(89deg) ~= 0.0175 under nadir view, near-grazing
    aspect = xr.full_like(flat, 210.0)  # face the sun so cos_inc is not also collapsed by the tilt
    lit = xr.ones_like(flat)

    unclamped = _refl(rad, min_lit=0.0, slope_az=(steep, aspect, lit), photom=_cos_view_dependent_photom)
    clamped = _refl(rad, min_lit=0.0, slope_az=(steep, aspect, lit), photom=_cos_view_dependent_photom, cos_min=COS_85)
    assert np.all(np.abs(clamped.values) < np.abs(unclamped.values))
    assert clamped.attrs["cos_view_floor_frac"] == 1.0  # every pixel here has cos_view below the floor
    assert clamped.attrs["cos_inc_floor_frac"] == 0.0  # incidence itself is nowhere near the floor


def test_grazing_incidence_and_emission_together_stay_bounded():
    """Besse's rule: clamping incidence and emission at 85 deg keeps I/F from blowing up either way."""
    rad = _synthetic_l1()
    rad["solar_inc"] = ("y", np.full(rad.sizes["y"], 89.5))  # grazing incidence
    flat = xr.zeros_like(rad.isel(band=0, drop=True))
    steep = xr.full_like(flat, 89.0)  # and grazing emission
    # aspect cross-slope to the sun (sun_az=210, aspect=300) so the 89deg tilt doesn't itself
    # correct the incidence angle back toward normal -- it stays grazing, same as flat ground.
    aspect = xr.full_like(flat, 300.0)
    lit = xr.ones_like(flat)

    unclamped = _refl(rad, min_lit=0.0, slope_az=(steep, aspect, lit), photom=_cos_view_dependent_photom)
    clamped = _refl(rad, min_lit=0.0, slope_az=(steep, aspect, lit), photom=_cos_view_dependent_photom, cos_min=COS_85)
    assert np.all(np.isfinite(clamped.values))
    assert np.all(np.abs(clamped.values) < np.abs(unclamped.values))
    assert np.abs(clamped.values).max() <= 1.0 / COS_85**2 + 1e-3
    assert clamped.attrs["cos_inc_floor_frac"] == 1.0
    assert clamped.attrs["cos_view_floor_frac"] == 1.0


def test_besse_phase_table_parses_known_value():
    """Spot-check against Besse's table: ch19/950.06 nm, f(75)/f(30) ~ 0.6462 ."""
    _, wl, f = ph._m3_phase_table()
    i = int(np.argmin(np.abs(wl - 950.06)))
    assert f[30, i] == pytest.approx(0.255755, abs=1e-6)
    assert (f[75, i] / f[30, i]) == pytest.approx(0.6462, abs=1e-4)


def test_besse_phase_ratio_matches_spot_check_and_is_identity_at_30():
    assert ph.besse_phase(30.0, 950.06) == pytest.approx(1.0)
    assert ph.besse_phase(75.0, 950.06) == pytest.approx(1.548, abs=2e-3)


def test_besse_phase_clamps_beyond_the_85deg_domain():
    """Past 85 deg the table's quartic fit rolls over; usage must clamp there."""
    at_edge = ph.besse_phase(85.0, 950.06)
    assert ph.besse_phase(90.0, 950.06) == pytest.approx(at_edge)
    assert ph.besse_phase(150.0, 950.06) == pytest.approx(at_edge)


def test_besse_phase_broadcasts_over_wavelength_and_geometry():
    g = np.array([10.0, 30.0, 75.0])
    out = ph.besse_phase(g, np.array([950.06, 1508.99]))
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[:, 1], 1.0)  # alpha=30 is the identity for every band


def test_phase_wiring_reproduces_a_scene_at_the_reference_geometry():
    """At phase=30 with lommel_seeliger, besse_phase is 1 everywhere, so `phase` only rescales by
    the fixed XL(30,0,30) reference constant -- the ratio to the no-phase run is that constant."""
    rad = _synthetic_l1()
    rad["solar_inc"] = ("y", np.full(rad.sizes["y"], 30.0))  # nadir view: phase == incidence
    baseline = _refl(rad, photom="lommel_seeliger", min_lit=0.0)
    with_phase = _refl(rad, photom="lommel_seeliger", phase=True, min_lit=0.0)
    xl_ref = ph.lommel_seeliger(np.cos(np.radians(30.0)), 1.0, 30.0)
    np.testing.assert_allclose(with_phase.values, baseline.values * xl_ref, rtol=1e-6)


def test_penumbra_is_dropped_by_the_min_lit_default():
    """min_lit defaults to 0.9: a half-blocked solar disk is nulled, and the cut is opt-outable."""
    rad = _synthetic_l1()
    flat = xr.zeros_like(rad.isel(band=0, drop=True))
    lit = xr.ones_like(flat)
    lit[0, 1] = 0.5
    refl = _refl(rad, slope_az=(flat, flat, lit))
    assert np.isnan(refl.values[:, 0, 1]).all()
    assert np.isfinite(refl.values[:, 2, 2]).all()
    assert np.isfinite(_refl(rad, slope_az=(flat, flat, lit), min_lit=0.0).values[:, 0, 1]).all()


@pytest.mark.parametrize("ydir", [1, -1])
def test_slope_az_from_dem_faces_downhill_whichever_way_rows_run(ydir):
    """A plane dropping 1 m per 10 m eastward is a 5.71 deg slope facing east (aspect 90)."""
    x, y = np.arange(0.0, 100.0, 10.0), ydir * np.arange(0.0, 80.0, 10.0)
    dem = xr.DataArray(-0.1 * x[None, :] + 0.0 * y[:, None], coords={"y": y, "x": x}, dims=("y", "x"))
    slope, aspect = ph.slope_az_from_dem(dem)
    inner = (slice(1, -1), slice(1, -1))  # edge pixels see a one-sided gradient
    np.testing.assert_allclose(slope.values[inner], np.degrees(np.arctan(0.1)), atol=1e-9)
    np.testing.assert_allclose(aspect.values[inner], 90.0, atol=1e-9)
    north_up = xr.DataArray(-0.1 * y[:, None] + 0.0 * x[None, :], coords={"y": y, "x": x}, dims=("y", "x"))
    np.testing.assert_allclose(ph.slope_az_from_dem(north_up)[1].values[inner], 0.0, atol=1e-9)  # drops north
