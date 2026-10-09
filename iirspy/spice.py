"""SPICE kernel lookup for IIRS observations.

Paths come from `IIRS_ARCHIVE` and `IIRS_SPICE`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ARCHIVE = Path(os.environ.get("IIRS_ARCHIVE", "/mnt/d/ddata/moon/ch2_iirs"))

# Kernel tree holding lsk/, pck/, sclk/, fk/ and spk/.
SPICE = Path(os.environ.get("IIRS_SPICE", str(ARCHIVE / "spice")))


def kernels(day: str) -> list[Path]:
    """Generic SPICE kernels plus the orbit SPKs covering `day`, a "YYYYMMDD" string.

    Raises if the tree or a covering SPK is missing: `sun_geometry` furnishes whatever it is given,
    so an incomplete set yields a wrong sun vector rather than an error.
    """
    import re

    if not SPICE.is_dir():
        raise FileNotFoundError(f"SPICE kernel tree not found at {SPICE} -- set IIRS_SPICE")
    ks = [SPICE / p for p in ("lsk/naif0012.tls", "pck/pck00010.tpc", "sclk/ch2_sclk_v1.tsc", "fk/ch2_v01.tf")]
    ks = [p for p in ks if p.exists()]
    d = pd.to_datetime(day, format="%Y%m%d")
    for f in sorted(SPICE.glob("spk/*.bsp")):
        m = re.findall(r"(\d{2}[A-Za-z]{3}\d{4})", f.name)
        if (
            len(m) == 2 and pd.to_datetime(m[0], format="%d%b%Y") <= d <= pd.to_datetime(m[1], format="%d%b%Y")
        ) or f.name.startswith("de4"):
            ks.append(f)
    if not any(f.suffix == ".bsp" for f in ks):
        raise FileNotFoundError(f"no SPK in {SPICE / 'spk'} covers {day}")
    return ks
