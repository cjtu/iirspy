import time

import dask.array as da
import numpy as np
import xarray as xr

from iirspy import io
from iirspy.l1 import frames

BAND_GAP = 9  # widest spectral hole `interpolate_na` bridges: two dropped groups (8 bands) in a row
ROW_GAP = 5  # likewise along y: up to 4 lost frames in a row


def _interp_rows(rad, max_gap=ROW_GAP):
    """Linear fill along y of NaN runs shorter than `max_gap` rows, chunk by chunk (overlapped so a run on a chunk edge fills).

    A 2-row run straddling the chunk edge fills; a run of `max_gap` rows or more stays NaN:

    >>> a = np.ones((1, 12, 1)); a[0, 5:7] = np.nan; a[0, 8:] = np.nan
    >>> r = xr.DataArray(da.from_array(a, chunks=(1, 6, 1)), dims=("band", "y", "x"))
    >>> np.isnan(_interp_rows(r, 4).values[0, :, 0]).astype(int).tolist()
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    """
    ax = rad.get_axis_num("y")
    data = rad.data if isinstance(rad.data, da.Array) else da.from_array(rad.data)
    data = data.rechunk({a: -1 for a in range(rad.ndim) if a != ax})

    def block(b):
        dims = [f"d{i}" for i in range(b.ndim)]
        dims[ax] = "y"
        return (
            xr.DataArray(b, dims=dims, coords={"y": np.arange(b.shape[ax])}).interpolate_na("y", max_gap=max_gap).values
        )

    return rad.copy(data=data.map_overlap(block, depth={ax: max_gap}, boundary="none", dtype=data.dtype))


def fill(rad, cfg, **_):
    """Rebuild the lost frames from each pixel's own spectrum, then interpolate the remaining band and spatial gaps."""
    lost = rad.lost_frames.values if "lost_frames" in rad.coords else None
    out = rad.drop_vars("lost_frames", errors="ignore")
    if not (cfg.fill_frames or cfg.interp_bands or cfg.interp_spatial):
        return out
    t0 = time.monotonic()
    if cfg.fill_frames and lost is not None:
        out = frames.fill(out, lost)
    if cfg.interp_bands is not None:
        out = out.interpolate_na("band", max_gap=BAND_GAP, keep_attrs=True, method=cfg.interp_bands)
    if cfg.interp_spatial:
        # max_gap=2 in 0.5-based x coords spans exactly one missing column
        out = out.interpolate_na("x", max_gap=2, keep_attrs=True, method="linear")
        out = _interp_rows(out)
    return io.get_step_log(out, rad, "fill", "interpolate", t0, cfg, ("fill_frames", "interp_bands", "interp_spatial"))
