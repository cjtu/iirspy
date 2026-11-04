"""Masking, the L1 step after radiometry: null what no correction can recover.

Nulls the order-sorting-filter seam and invalid bands, any `exclude_wl` ranges, the bad detector elements (packaged
mask, plus a G1 table's own) and pixels at or above the label's saturation radiance. Nulls are NaN, which `fill`
may later bridge.
"""

from iirspy import instrument


def mask(rad, qub, *, bad_pixel_mask=True, exclude_wl=None, calib_dir=instrument.DCALIB):
    """`rad` with the OSF and invalid bands, `exclude_wl` (lo, hi) [nm] ranges, bad elements and saturation nulled.

    `bad_pixel_mask` True is the packaged mask, unioned with a G1 table's `calib_bad` coord when present; False is
    none; a (band, x) bool array is used as given.
    """
    bad_band = rad.band.isin((*instrument.OSF, *instrument.INVALID))
    for lo, hi in exclude_wl or ():
        bad_band = bad_band | ((rad.wl >= lo) & (rad.wl <= hi))
    out = rad.where(~bad_band)
    if bad_pixel_mask is True:
        bad = instrument.load_bad_pixel_mask()
        if "calib_bad" in out.coords:
            bad = bad | out.calib_bad
    else:
        bad = bad_pixel_mask
    if bad is not False:
        out = out.where(~bad)
    out = out.drop_vars("calib_bad", errors="ignore")
    return out.where(out < instrument.get_saturation_radiance(qub, calib_dir))
