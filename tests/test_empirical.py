"""Tests for the empirical dark / flat / smile correction.

Most tests here are smoke tests on a synthetic cube: they check that each stage runs, returns the
declared shapes, and takes its documented fallback. The acceptance test at the bottom is the real
one - it injects a known residual dark, per-column gain and smile into a real L0 subset and checks
each stage recovers what was injected.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from iirspy import empirical as emp
from iirspy.iirs import _band_numbers

FIXTURE = Path(__file__).parent / "data" / "iirs_l0_subset.tif"
FIXTURE_BANDS = [10, 18, 26, 42, 110, 111, 112]
PAN = [b for b in FIXTURE_BANDS if b in emp.PAN_BANDS]  # [10, 18, 26, 42]


def synth_cube(nband=7, ny=600, nx=250, bands=None, dark_rows=200, seed=0):
    """Synthetic (band, y, x) DN cube: `dark_rows` dark rows then a lit, gently textured scene."""
    rng = np.random.default_rng(seed)
    bands = FIXTURE_BANDS[:nband] if bands is None else bands
    data = np.full((len(bands), ny, nx), 15.0, dtype="float32")  # dark level
    scene = 300 + 20 * np.sin(np.linspace(0, 6, nx))[None, :] + rng.normal(0, 3, (ny - dark_rows, nx))
    data[:, dark_rows:, :] += scene[None].astype("float32")
    data += rng.normal(0, 1.0, data.shape).astype("float32")
    return xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": bands, "y": np.arange(ny) + 0.5, "x": np.arange(nx) + 0.5},
    )


@pytest.fixture(scope="module")
def real_cube():
    """The committed 7-band x 800-line x 250-sample raw-DN crop of a real IIRS scene."""
    da = xr.open_dataarray(FIXTURE, engine="rasterio")
    return da.assign_coords(band=_band_numbers(da)).astype("float32")


# --- unit / smoke -----------------------------------------------------------------------------


def test_fixture_carries_pan_and_clean_bands(real_cube):
    assert list(real_cube.band.values) == FIXTURE_BANDS
    assert emp.avail(real_cube, emp.PAN_BANDS) == PAN
    assert emp.avail(real_cube, emp.clean_bands()) == [42, 110, 111, 112]


def test_avail_raises_when_no_band_present(real_cube):
    with pytest.raises(ValueError, match="carries none of bands"):
        emp.avail(real_cube, [200, 201])


def test_longest_run():
    assert emp.longest_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)) == (4, 7)
    assert emp.longest_run(np.zeros(5, dtype=bool)) == (0, 0)


def test_robust_z_flags_an_outlier():
    a = np.random.default_rng(0).normal(0, 1, (20, 20))
    a[5, 5] = 30.0
    z = emp.robust_z(a - np.median(a), axis=(0, 1))
    assert z[5, 5] > 20
    assert np.nanmax(np.abs(np.delete(z, 5 * 20 + 5))) < 5  # everything else stays within noise


def test_dark_floor_and_detect_dark_rows_find_the_dark_block():
    cube = synth_cube()
    P = emp.panchromatic(cube)
    floor, sigma = emp.dark_floor(np.nanmedian(P.values, axis=1))
    assert 10 < floor < 20 and sigma >= emp.DARK_SIGMA_FLOOR
    mask, _, row_bright = emp.detect_dark_rows(P)
    assert emp.longest_run(mask) == (0, 200)
    assert row_bright.shape == (cube.sizes["y"],)


def test_detect_dark_rows_returns_nothing_when_scene_is_uniformly_lit():
    cube = synth_cube(dark_rows=0)
    mask, _, _ = emp.detect_dark_rows(emp.panchromatic(cube))
    assert not mask.any()


def test_lit_rows_uses_per_pixel_snr_not_row_median_scatter():
    """A shadowed scene's row-median scatter collapses onto DARK_SIGMA_FLOOR; per-pixel noise does not."""
    cube = synth_cube()
    cube[:, :200, :] = 0.0  # true shadow: row medians are exactly 0, their scatter is ~0
    P = emp.panchromatic(cube)
    lit = emp.lit_rows(P, (0, 200))
    assert not lit[:200].any(), "shadow rows must not be lit"
    assert lit[200:].all(), "scene rows must be lit"
    # the old rule would admit rows a hair above 0; the SNR rule needs real signal
    floor, sigma = emp.dark_floor(np.nanmedian(P.values, axis=1))
    assert floor == 0.0 and sigma == emp.DARK_SIGMA_FLOOR
    assert emp.LIT_SNR * sigma < 2.0  # <- the 1.5 DN cut the old rule produced


def test_lit_rows_without_a_dark_block_treats_every_row_as_lit(real_cube):
    assert emp.lit_rows(emp.panchromatic(real_cube), None).all()


def test_broadband_snr_is_high_in_lit_rows_and_low_in_dark():
    cube = synth_cube()
    P = emp.panchromatic(cube)
    snr = emp.broadband_snr(P, 0, 200)
    assert snr.dtype == np.float32
    assert float(snr.isel(y=slice(0, 200)).median()) < emp.SHADOW_SNR
    assert float(snr.isel(y=slice(200, None)).median()) > 50


def test_row_roughness_and_flattest_window(real_cube):
    P = emp.panchromatic(real_cube)
    rs = emp.row_roughness(P)
    lit = emp.lit_rows(P, None)  # fixture is a fully lit crop: no dark block to measure noise in
    assert rs.shape == lit.shape == (real_cube.sizes["y"],)
    a, b = emp.flattest_window(rs, lit)
    assert b - a == emp.MIN_ROWS
    best = float(np.nanmean(rs[a:b]))  # no other window of that length is flatter
    others = [float(np.nanmean(rs[i : i + emp.MIN_ROWS])) for i in range(0, len(rs) - emp.MIN_ROWS, 25)]
    assert best <= min(others) + 1e-9


def test_signal_free_rows_do_not_poison_the_roughness_profile():
    """Signal-free rows have infinite roughness; smoothing must not spread that down the profile."""
    cube = synth_cube()
    cube[:, 300:340, :] = 0.0
    P = emp.panchromatic(cube)
    rs = emp.row_roughness(P)
    lit = emp.lit_rows(P, (0, 200))
    assert np.isinf(rs[300:340]).all(), "signal-free rows must be maximally rough"
    assert np.isfinite(rs[400:]).all(), "the rest of the roughness profile must survive"
    win = emp.flattest_window(rs, lit)
    assert win is not None and not (win[0] < 340 and win[1] > 300), "must not straddle the dead rows"


def test_flattest_window_without_enough_rows_returns_none():
    assert emp.flattest_window(np.zeros(500), np.zeros(500, dtype=bool)) is None
    assert emp.flattest_window(np.zeros(50), np.ones(50, dtype=bool)) is None


def test_spatial_outlier_mask_flags_injected_spikes(real_cube):
    sub = real_cube.isel(y=slice(300, 500)).copy()
    sub[:, 10, 10] = 10000.0
    mask = emp.spatial_outlier_mask(sub)
    assert mask.shape == (200, real_cube.sizes["x"])
    assert mask[10, 10]


def test_empirical_frames_runs_end_to_end(real_cube):
    dark, flat, smile, snr, notes = emp.empirical_frames(real_cube, apply_smile=True)
    nband, nx = real_cube.sizes["band"], real_cube.sizes["x"]
    assert flat.shape == smile.shape == (nband, nx)
    assert flat.dtype == smile.dtype == np.float32
    assert set(notes) >= {"has_shadow", "dark_rows", "flat_runs", "flat_fallback", "smile_fallback"}
    if notes["has_shadow"]:
        assert dark.shape == (nband, nx) and snr.shape == real_cube.shape[1:]


def test_no_shadow_scene_skips_dark_and_reports_it():
    cube = synth_cube(dark_rows=0)
    with pytest.warns(UserWarning, match="no shadow rows"):
        dark, _, _, snr, notes = emp.empirical_frames(cube)
    assert notes["has_shadow"] is False and snr is None
    assert float(abs(dark).max()) == 0.0


def test_smile_falls_back_to_ones_without_enough_lit_rows():
    cube = synth_cube(ny=260, dark_rows=250)  # only 10 lit rows
    flat = xr.ones_like(cube.isel(y=0, drop=True))
    with pytest.warns(UserWarning, match="too few lit rows"):
        smile, notes = emp.estimate_smile(cube, 0.0, flat, np.arange(cube.sizes["y"]) >= 250)
    assert notes["smile_fallback"] is True
    assert float(abs(smile - 1).max()) == 0.0


def test_flat_falls_back_to_reference_when_no_region_qualifies(real_cube):
    ref = xr.ones_like(real_cube.isel(y=0, drop=True)) * 1.5
    nolit = np.zeros(real_cube.sizes["y"], dtype=bool)
    rs = np.zeros(real_cube.sizes["y"])
    with pytest.warns(UserWarning, match="falling back to packaged reference flat"):
        flat, notes = emp.build_flat(real_cube, 0.0, rs, nolit, np.zeros(real_cube.sizes["y"]), ref)
    assert notes["flat_fallback"] is True
    assert float(flat.where(np.isfinite(flat)).median()) == pytest.approx(1.5)


# --- acceptance: inject known artifacts into a real scene and recover them ----------------------

DARK_ROWS = 250
FLAT_RUN = (350, 750)


def inject(cube, seed=1, colgain_amp=0.04, smile_amp=0.10):
    """Return (cube_with_artifacts, dark0, colgain, smile_true) built from a real lit scene.

    Model  DN = dark0(band, x) + scene(band, y, x) * colgain(band, x) * smile(band, x),
    with scene forced to 0 over the first DARK_ROWS rows so the dark finder has a shadow to
    locate. colgain is white across x (what a per-detector-element response looks like, and what
    lowpass_x leaves behind); smile is smooth across x and built so its median over the PAN bands
    is 1 for every x - the broadband profile the smile step deliberately preserves.
    """
    rng = np.random.default_rng(seed)
    nband, _, nx = cube.shape
    scene = cube.copy()
    scene[:, :DARK_ROWS, :] = 0.0

    dark0 = 12.0 + 4.0 * rng.random((nband, nx)).astype("float32")
    colgain = (1.0 + colgain_amp * rng.standard_normal((nband, nx))).astype("float32")

    # PAN bands get amplitudes that median to 0; the rest carry the recoverable spectral smile
    amps = np.zeros(nband, dtype="float32")
    pan_idx = [i for i, b in enumerate(cube.band.values) if b in PAN]
    amps[pan_idx] = smile_amp * np.array([-1.0, -1 / 3, 1 / 3, 1.0], dtype="float32")
    rest = [i for i in range(nband) if i not in pan_idx]
    amps[rest] = smile_amp * np.array([0.6, -0.8, 0.4], dtype="float32")[: len(rest)]
    shape_x = np.cos(np.pi * np.arange(nx) / nx).astype("float32")
    smile_true = 1.0 + amps[:, None] * shape_x[None, :]
    smile_true /= np.median(smile_true, axis=1, keepdims=True)

    def frame(a):
        return xr.DataArray(a, dims=("band", "x"), coords={"band": cube.band, "x": cube.x})

    clean = (frame(dark0) + scene).transpose("band", "y", "x").astype("float32")
    dirty = (frame(dark0) + scene * frame(colgain) * frame(smile_true)).transpose("band", "y", "x")
    return clean, dirty.astype("float32"), frame(dark0), frame(colgain), frame(smile_true)


def test_acceptance_recovers_injected_dark_colgain_and_smile(real_cube):
    clean_cube, inj_cube, dark0, colgain, smile_true = inject(real_cube)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = emp.empirical_frames(clean_cube, flat_yrange=FLAT_RUN, apply_smile=True)
        got = emp.empirical_frames(inj_cube, flat_yrange=FLAT_RUN, apply_smile=True)
    _, b_flat, b_smile, _, _ = base
    dark, flat, smile, snr, notes = got

    # 1. the dark finder locates the contrived shadow on its own, and the dark frame matches
    assert notes["has_shadow"] is True and notes["dark_source"] == "auto"
    d0, d1 = notes["dark_rows"]
    assert d0 == 0 and d1 == pytest.approx(DARK_ROWS, abs=5)
    assert float(abs(dark - dark0).median()) < 0.5

    # 2. flat recovers the injected per-column gain (relative to the scene's own flat).
    # Measured: injected sigma 0.039, median |err| 0.009, corr 0.95 - the residual is the scene's
    # own cross-track high-frequency content, which no scene-derived flat can separate out.
    rec_gain = (flat / b_flat).values
    err = rec_gain - colgain.values
    ok = np.isfinite(err)
    assert np.nanmedian(np.abs(err)) < 0.015
    assert np.corrcoef(rec_gain[ok], colgain.values[ok])[0, 1] > 0.9

    # 3. smile recovers the injected spectral cross-track field (measured median |err| 0.007)
    rec_smile = (smile / b_smile).values
    assert np.nanmedian(np.abs(rec_smile - smile_true.values)) < 0.01

    # 4. SNR field separates the contrived shadow from the lit scene
    assert float(snr.isel(y=slice(0, DARK_ROWS)).median()) < emp.SHADOW_SNR
    assert float(snr.isel(y=slice(DARK_ROWS, None)).median()) > emp.SHADOW_SNR
