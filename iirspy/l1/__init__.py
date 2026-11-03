import json

from iirspy import instrument, io
from iirspy.l1 import empirical, fill, frames, mask, radiometry
from iirspy.l1.config import CALIB


def calibrate(
    dn, cfg, *, ws=None, qub, calib_dir=instrument.DCALIB, empirical_kws=None, frame_offsets=None, output_bands=None
):
    """L0 digital number cube (band, y, x) -> L1 radiance [W/m^2/sr/um], one step per `cfg` field group.

    `qub` is the scene's raw label. `empirical_kws` go to `empirical.empirical_frames`; `frame_offsets` is the
    strip's frame table (`frames.build_table`); `output_bands` narrows the bands after radiometry, so the
    empirical frames still see every band of `dn`.
    """
    kw = {"qub": qub, "calib_dir": calib_dir, "empirical_kws": empirical_kws, "frame_offsets": frame_offsets}
    rad = empirical.residual_dark(dn, cfg, ws=ws, **kw)
    rad = empirical.flat_smile(rad, cfg, ws=ws, **kw)
    rad = radiometry.radiometry(rad, cfg, ws=ws, **kw)
    rad = rad if output_bands is None else rad.sel(band=output_bands)
    rad = frames.retime(rad, cfg, ws=ws, **kw)
    rad = mask.mask(rad, cfg, ws=ws, **kw)
    rad = fill.fill(rad, cfg, ws=ws, **kw)

    drop = [c for c in rad.coords if str(c).startswith("empirical_") and not cfg.attach_empirical_frames]
    out = rad.drop_vars(drop).astype("float32")
    out.name = "Radiance [W/m^2/sr/µm]"
    out.attrs |= {
        "name": "Radiance",
        "units": "W/m^2/sr/um",
        "calibration_source": "empirical" if cfg.empirical else "user",
        "iirspy_version": io.IIRSPY_VERSION,
        "l1_config": json.dumps(cfg.to_attrs()),
        "calib": cfg.calib if cfg.calib in CALIB else "g1_table",
    }
    return out
