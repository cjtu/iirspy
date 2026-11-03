import time

import xarray as xr

from iirspy import instrument, io


def _in_wl_ranges(wl, ranges):
    """True where `wl` falls in any (lo, hi) of `ranges` (inclusive); all False if `ranges` is falsy."""
    hit = xr.zeros_like(wl, dtype=bool)
    for lo, hi in ranges or ():
        hit = hit | ((wl >= lo) & (wl <= hi))
    return hit


def mask(rad, cfg, *, ws=None, qub, calib_dir=instrument.DCALIB, **_):
    """Null the OSF and invalid bands, `exclude_wl`, bad detector elements and saturated pixels."""
    t0 = time.monotonic()
    out = rad.where(~(rad.band.isin((*instrument.OSF, *instrument.INVALID)) | _in_wl_ranges(rad.wl, cfg.exclude_wl)))
    if cfg.bad_pixel_mask is True:
        bad = instrument.load_bad_pixel_mask()
        if "calib_bad" in out.coords:
            bad = bad | out.calib_bad
    else:
        bad = cfg.bad_pixel_mask
    if bad is not False:
        out = out.where(~bad)
    out = out.drop_vars("calib_bad", errors="ignore")
    out = out.where(out < instrument.get_saturation_radiance(qub, calib_dir))
    return io.record_step(out, rad, "mask", "bad_elements", t0, cfg, ("bad_pixel_mask", "exclude_wl"), qub=str(qub))
