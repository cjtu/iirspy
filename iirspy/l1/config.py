"""L0 -> L1 radiance config: one frozen dataclass whose fields each name one method choice.

`L1Config` drives `L0.calibrate`. Presets are named sets of field values; overrides replace single fields.

>>> L1Config.preset("iirspy").empirical
True
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from iirspy.config import Config

CALIB = ("lut", "lut_median")


@dataclass(frozen=True)
class L1Config(Config):
    """L0 digital number -> L1 radiance choices (see `L0.calibrate_to_rad`).

    Attributes:
        empirical: Per-scene empirical dark/flat (+ smile) with the LUT's band-median scale,
            instead of the per-element LUT gain/offset.
        interp_bands: Fill masked bands by interpolating across band (e.g. "linear"), or None.
        interp_spatial: Fill single-column all-band NaN gaps across x.
        fix_frames: Move each 4-band group onto true frames and null its lost frames (needs the
            scene's frame table, see `iirspy.l1.frames`).
        fill_frames: Rebuild the nulled frames from each pixel's own spectrum (`iirspy.l1.frames.fill`).
        bad_pixel_mask: Null the packaged bad detector elements.
        exclude_wl: Extra (lo, hi) wavelength ranges [nm] to null, or None.
        attach_snr: Attach the empirical broadband SNR as an `snr` coord (empirical scenes with shadow).
        attach_empirical_frames: Attach the empirical dark/flat/smile (band, x) frames as coords
            (diagnostic only when `empirical` is off: they are saved, not applied).
        calib: Radiometric scale when not `empirical`. "lut" is the per-element LUT gain/offset,
            "lut_median" the LUT's per-band median gain/offset (no cross-track structure), or a path
            to a G1 table `.npz` (see `L0.calibrate_to_rad`). `empirical` already uses the median, so
            it needs "lut".
    """

    empirical: bool = False
    interp_bands: str | None = None
    interp_spatial: bool = False
    fix_frames: bool = False
    fill_frames: bool = False
    bad_pixel_mask: bool = True
    exclude_wl: tuple[tuple[float, float], ...] | None = None
    attach_snr: bool = True
    attach_empirical_frames: bool = False
    calib: str = "lut"

    def __post_init__(self):
        if isinstance(self.calib, Path):
            object.__setattr__(self, "calib", str(self.calib))
        if self.calib not in CALIB and not Path(self.calib).is_file():
            raise ValueError(f"calib {self.calib!r}: not one of {CALIB} or an existing G1 table")
        if self.empirical and self.calib != "lut":
            raise ValueError(
                f"empirical already uses the band-median LUT scale; calib must be 'lut', not {self.calib!r}"
            )
        if self.exclude_wl is not None:
            object.__setattr__(self, "exclude_wl", tuple((float(lo), float(hi)) for lo, hi in self.exclude_wl))


L1Config.PRESETS = {
    "iirspy": {
        "empirical": True,
        "interp_bands": "linear",
        "interp_spatial": True,
        "fix_frames": True,
        "fill_frames": True,
    },
    "issdc": {},
    "p0": {
        "calib": "lut_median",
        "empirical": False,
        "fix_frames": True,
        "fill_frames": False,
        "bad_pixel_mask": False,
        "interp_bands": None,
        "interp_spatial": False,
        "attach_empirical_frames": True,
    },
}
