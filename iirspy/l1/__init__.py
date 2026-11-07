"""L0 -> L1 pipeline: raw digital numbers to radiance [W/m^2/sr/um]. `L0.calibrate` calls it.

Each step is configured by `L1Config` fields and appends {"step", "method"} to the JSON `attrs["provenance"]`.

Steps, in order:

0. Empirical frames, when asked for: `empirical.derive_frames` derives the scene's dark, flat and smile and applies nothing
1. Dark subtraction, in DN: `radiometry.subtract_dark`
2. Flat division (and smile), in DN: `radiometry.divide_flat`
3. Radiometry, DN to radiance: `radiometry.radiometry`
4. Keystone shift: `radiometry.keystone`
5. ISSDC edge-column copy: `radiometry.edge_copy`
6. Re-time frame defects: `retime`
7. Mask bad bands, elements and saturation: `instrument.mask_bad`
8. Fill frame defects and short gaps: `fill_gaps`

Takes a DN cube (band, y, x) and returns a radiance cube of the same shape, both `xarray.DataArray`.
"""

import json

from iirspy import instrument, io
from iirspy.l1 import empirical, radiometry
from iirspy.l1.config import CALIB
from iirspy.l1.frame_defects import fill_gaps, retime


@io.xarray_keep_attrs
def calibrate(
    dn, cfg, *, qub, calib_dir=instrument.PKG_DATA, empirical_kws=None, frame_defects=None, output_bands=None
):
    """L0 digital number cube (band, y, x) -> L1 radiance [W/m^2/sr/um], by `cfg` (an `L1Config`).

    `qub` is the scene's raw label. `empirical_kws` go to `empirical.empirical_frames`; `frame_defects` is the
    strip's frame-defect table (`frame_defects.build_table`); `output_bands` narrows the bands after radiometry, so the
    empirical frames still see every band of `dn`.
    """
    steps = []

    def log(step, method):
        steps.append({"step": step, "method": method})

    cube, has_shadow = dn, False
    if "empirical" in (cfg.dark, cfg.flat) or cfg.attach_empirical_frames:
        fallback = radiometry.flatfield_v1(instrument.scene_start(qub), calib_dir)
        cube, notes = empirical.derive_frames(
            dn, fallback, attach_snr=cfg.attach_snr and cfg.dark == "empirical", **(empirical_kws or {})
        )
        has_shadow = notes["has_shadow"]
        log("empirical_frames", "scene" if has_shadow else "no shadow")
    if cfg.dark:
        cube = radiometry.subtract_dark(cube, qub, cfg.dark, has_shadow=has_shadow, calib_dir=calib_dir)
        log("dark", cfg.dark)
    if cfg.flat:
        cube = radiometry.divide_flat(cube, qub, cfg.flat, calib_dir=calib_dir)
        log("flat", cfg.flat)
    cube = radiometry.radiometry(cube, qub, cfg.calib, calib_dir=calib_dir)
    log("radiometry", cfg.calib if cfg.calib in CALIB else "calib_lut")
    cube = cube if output_bands is None else cube.sel(band=output_bands)
    if cfg.keystone:
        cube = radiometry.keystone(cube, shift=cfg.keystone, bad_pixel_mask=cfg.bad_pixel_mask, calib_dir=calib_dir)
        log("keystone", cfg.keystone)
    if cfg.edge_copy:
        cube = radiometry.edge_copy(cube)
        log("edge_copy", "issdc")
    if cfg.fix_frame_defects:
        cube = retime(cube, qub, frame_defects)
        if "lost_frames" in cube.coords:
            log("retime", "frame_table")
    cube = instrument.mask_bad(
        cube, qub, bad_pixel_mask=cfg.bad_pixel_mask, exclude_wl=cfg.exclude_wl, calib_dir=calib_dir
    )
    log("mask", "bad_elements")
    if cfg.fill_frame_defects or cfg.interp_bands or cfg.interp_spatial:
        cube = fill_gaps(
            cube,
            fill_frame_defects=cfg.fill_frame_defects,
            interp_bands=cfg.interp_bands,
            interp_spatial=cfg.interp_spatial,
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
        "calibration_source": "empirical" if "empirical" in (cfg.dark, cfg.flat) else "user",
        "iirspy_version": io.IIRSPY_VERSION,
        "l1_config": json.dumps(cfg.to_attrs()),
        "calib": cfg.calib if cfg.calib in CALIB else "calib_lut",
        "provenance": json.dumps(steps),
    }
    return out
