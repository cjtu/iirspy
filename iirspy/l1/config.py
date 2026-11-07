"""L0 -> L1 radiance config: one frozen dataclass whose fields each name one method choice.

`L1Config` drives `L0.calibrate`. Presets are named sets of field values; overrides replace single fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from iirspy.config import Config

CALIB = ("lut", "issdc_derived")
DARK = (None, "issdc_lut", "issdc_lut_median", "issdc_derived", "empirical")
FLAT = (None, "issdc_lut", "issdc_derived", "flatfield_v1", "empirical")


@dataclass(frozen=True)
class L1Config(Config):
    """L0 digital number -> L1 radiance choices (see `iirspy.l1.calibrate`).

    Attributes:
        dark: DN dark subtracted first (`radiometry.subtract_dark`): None, "issdc_lut" (the whole packaged LUT, one
            offset per detector element), "issdc_lut_median" (its band-median offset, as `flatfield_v1` was derived on),
            "issdc_derived" (offset fitted to ISSDC's nci L1) or "empirical" (the scene's own shadow-row dark;
            "issdc_lut_median" when it has none).
        flat: Flat divided out next (`radiometry.divide_flat`): None, "issdc_lut" or "issdc_derived" (per-element gains
            relative to the band median), "flatfield_v1" (`data/flatfield_v1.zip`) or "empirical" (the scene's own flat
            and smile). The LUT's band median alone is a flat of 1, so there is no "issdc_lut_median" flat.
        interp_bands: Fill masked bands by interpolating across band (e.g. "linear"), or None.
        interp_spatial: Fill single-column all-band NaN gaps across x.
        fix_frame_defects: Move each 4-band group onto true frames and null its lost frames (needs the
            scene's frame-defect table, see `iirspy.l1.frame_defects`).
        fill_frame_defects: Rebuild the nulled frames from each pixel's own spectrum (`iirspy.l1.frame_defects.fill`).
        bad_pixel_mask: Null the packaged bad detector elements.
        exclude_wl: Extra (lo, hi) wavelength ranges [nm] to null, or None.
        attach_snr: Attach the empirical broadband SNR as an `snr` coord (empirical scenes with shadow).
        attach_empirical_frames: Attach the empirical dark/flat/smile (band, x) frames as coords
            (diagnostic only when `empirical` is off: they are saved, not applied).
        keystone: Cross-track keystone shift after radiometry (`radiometry.keystone`), any `calib`: None, "simple"
            (ISSDC's bicubic) or "weighted" (bicubic but flagged bad elements excluded from the interpolation).
        edge_copy: Copy column 2 into 0-1 and 247 into 248-249 as ISSDC does (discards those readings).
        calib: Radiometric scale: the band-median gain of the packaged LUT ("lut") or of the table fitted to ISSDC's
            nci L1 ("issdc_derived"), or a path to a calib LUT `.npz` (see `iirspy.l1.radiometry`) that replaces dark,
            flat and scale (both must then be None).
    """

    dark: str | None = "issdc_lut"
    flat: str | None = "issdc_lut"
    interp_bands: str | None = None
    interp_spatial: bool = False
    fix_frame_defects: bool = False
    fill_frame_defects: bool = False
    bad_pixel_mask: bool = True
    exclude_wl: tuple[tuple[float, float], ...] | None = None
    attach_snr: bool = True
    attach_empirical_frames: bool = False
    keystone: str | None = None
    edge_copy: bool = False
    calib: str = "lut"

    def __post_init__(self):
        if isinstance(self.calib, Path):
            object.__setattr__(self, "calib", str(self.calib))
        if self.calib not in CALIB and not Path(self.calib).is_file():
            raise ValueError(f"calib {self.calib!r}: not one of {CALIB} or an existing calib LUT")
        if self.dark not in DARK or self.flat not in FLAT:
            raise ValueError(f"dark {self.dark!r} not in {DARK}, or flat {self.flat!r} not in {FLAT}")
        if self.calib not in CALIB and (self.dark or self.flat):
            raise ValueError("a calib LUT replaces dark and flat: set both to None")
        if self.keystone not in (None, "simple", "weighted"):
            raise ValueError(f"keystone must be None, 'simple' or 'weighted', not {self.keystone!r}")
        if self.exclude_wl is not None:
            object.__setattr__(self, "exclude_wl", tuple((float(lo), float(hi)) for lo, hi in self.exclude_wl))


L1Config.PRESETS = {
    "iirspy": {
        "dark": "issdc_lut_median",
        "flat": "flatfield_v1",
        "interp_bands": "linear",
        "interp_spatial": True,
        "fix_frame_defects": True,
        "fill_frame_defects": True,
    },
    "issdc": {
        "calib": "issdc_derived",
        "dark": "issdc_derived",
        "flat": "issdc_derived",
        "keystone": "simple",
        "edge_copy": True,
    },
    "lut-only": {
        "dark": "issdc_lut_median",
        "flat": None,
        "fix_frame_defects": True,
        "fill_frame_defects": False,
        "bad_pixel_mask": False,
        "interp_bands": None,
        "interp_spatial": False,
        "attach_empirical_frames": True,
    },
}
