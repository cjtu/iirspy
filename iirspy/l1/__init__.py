"""L0 -> L1 pipeline: raw digital numbers to radiance [W/m^2/sr/um]. `L0.calibrate` calls it.

    DN -> residual dark -> / flat * smile -> radiometry -> re-time frames -> mask -> fill

Each step is switched by `L1Config` fields and appends {"step", "method"} to the JSON list `attrs["provenance"]`,
which L2 continues. The step modules take plain arguments; only this module reads the config.
"""

import json

import xarray as xr

from iirspy import instrument, io
from iirspy.l1 import empirical, fill, frames, mask, radiometry
from iirspy.l1.config import CALIB


def calibrate(dn, cfg, *, qub, calib_dir=instrument.DCALIB, empirical_kws=None, frame_offsets=None, output_bands=None):
    """L0 digital number cube (band, y, x) -> L1 radiance [W/m^2/sr/um], by `cfg` (an `L1Config`).

    `qub` is the scene's raw label. `empirical_kws` go to `empirical.empirical_frames`; `frame_offsets` is the
    strip's frame table (`frames.build_table`); `output_bands` narrows the bands after radiometry, so the
    empirical frames still see every band of `dn`.
    """
    steps = []

    def log(step, method):
        steps.append({"step": step, "method": method})

    with xr.set_options(keep_attrs=True):  # older xarray drops attrs in arithmetic
        cube, has_shadow = dn, False
        if cfg.empirical or cfg.attach_empirical_frames:
            cube, notes = empirical.residual_dark(
                dn, qub, subtract=cfg.empirical, attach_snr=cfg.attach_snr, calib_dir=calib_dir, **(empirical_kws or {})
            )
            has_shadow = notes["has_shadow"]
            log("residual_dark", "scene" if cfg.empirical and has_shadow else "none")
        if cfg.empirical:
            cube = cube / (cube.empirical_flat * cube.empirical_smile).clip(min=empirical.FLAT_SMILE_MIN)
            log("flat_smile", "empirical")
        calib = "empirical" if cfg.empirical else cfg.calib
        cube = radiometry.radiometry(cube, qub, calib, calib_dir=calib_dir, has_shadow=has_shadow)
        log("radiometry", calib if calib in (*CALIB, "empirical") else "g1")
        cube = cube if output_bands is None else cube.sel(band=output_bands)
        if cfg.fix_frames:
            cube = frames.retime(cube, qub, frame_offsets)
            if "lost_frames" in cube.coords:
                log("retime", "frame_table")
        cube = mask.mask(cube, qub, bad_pixel_mask=cfg.bad_pixel_mask, exclude_wl=cfg.exclude_wl, calib_dir=calib_dir)
        log("mask", "bad_elements")
        if cfg.fill_frames or cfg.interp_bands or cfg.interp_spatial:
            cube = fill.fill(
                cube, fill_frames=cfg.fill_frames, interp_bands=cfg.interp_bands, interp_spatial=cfg.interp_spatial
            )
            log("fill", "interpolate")

    drop = ["lost_frames"] + [
        c for c in cube.coords if str(c).startswith("empirical_") and not cfg.attach_empirical_frames
    ]
    out = cube.drop_vars(drop, errors="ignore").astype("float32")
    out.name = "Radiance [W/m^2/sr/µm]"
    out.attrs |= {
        "name": "Radiance",
        "units": "W/m^2/sr/um",
        "calibration_source": "empirical" if cfg.empirical else "user",
        "iirspy_version": io.IIRSPY_VERSION,
        "l1_config": json.dumps(cfg.to_attrs()),
        "calib": cfg.calib if cfg.calib in CALIB else "g1_table",
        "provenance": json.dumps(steps),
    }
    return out
