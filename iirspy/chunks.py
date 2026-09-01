"""Chunk planning for strip registration.

Determines which latitude band a row falls into, which LOLA DEM to use as reference, where to cut
the strip into chunks (`plan_chunks`), which SPICE kernels cover the observation (`kernels`), and
what `GeorefConfig` a chunk starts from (`chunk_cfg`).

Stateless: no scene is read and nothing is written. Paths come from `IIRS_ARCHIVE`,
`IIRS_DEM_ROOTS`, `IIRS_SPICE` and `IIRS_ANC_ROOTS`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import numpy as np

from iirspy.georef import MOON_RADIUS_M, GeorefConfig, to_stereo

ARCHIVE = Path(os.environ.get("IIRS_ARCHIVE", "/mnt/d/ddata/moon/ch2_iirs"))
ANC_ROOTS = [Path(p) for p in os.environ["IIRS_ANC_ROOTS"].split(":")] if "IIRS_ANC_ROOTS" in os.environ else [ARCHIVE]

# Minimum correlation against the reference for a solve to count as registered. `converged` alone
# does not imply it.
GOOD_CORR = 0.85

# Searched in order, first hit wins; COG rebuilds shadow the original GDR JP2s.
_DEM_ROOTS = [
    Path(p) for p in os.environ.get("IIRS_DEM_ROOTS", "/mnt/d/ddata/moon/lro_lola:/mnt/d/ddata/moon/lro_wac").split(":")
]

# Kernel tree holding lsk/, pck/, sclk/, fk/ and spk/.
SPICE = Path(os.environ.get("IIRS_SPICE", str(ARCHIVE / "spice")))

# Our own re-solved products, laid out like the archive's own `<category>/<level>/<day>/` tree so
# the two read the same way -- but not under IIRS_ARCHIVE, since these are derived, not PDS-shipped.
RECAL_ROOT = Path(os.environ.get("IIRS_RECAL_ROOT", str(Path.home() / "data" / "ch2_iirs")))


def _dem(sub: str, name: str) -> str:
    """Path to a LOLA product by its bare product name, preferring a COG over the GDR JP2.

    Matched case-insensitively, returning the directory's real entry: PDS ships these labels as
    both `_JP2.LBL` and `_jp2.lbl`, a difference a case-insensitive drvfs mount hides and ext4
    does not.
    """
    for root in _DEM_ROOTS:
        d = root / sub
        if not d.is_dir():
            continue
        entries = {f.name.lower(): f for f in d.iterdir()}
        for leaf in (f"{name}.tif", f"{name}_jp2.lbl"):
            hit = entries.get(leaf.lower())
            if hit is not None:
                return str(hit)
    raise FileNotFoundError(f"{name} not found under any of {[str(r) for r in _DEM_ROOTS]}")


_POLAR = "LOLA_GDR/POLAR"
_SLDEM = "SLDEM2015_512_60S_60N_000_360"


def _pole_radius_m(crs_pole: str, lat: float) -> float:
    """Stereographic distance from `crs_pole`'s pole to `lat`."""
    x, y = to_stereo(crs_pole).transform(0.0, lat)
    return float(np.hypot(x, y))


def _equatorward(lat_range: tuple[float, float]) -> float:
    """Whichever bound of `lat_range` sits closer to the equator."""
    return min(lat_range, key=abs)


# One entry per latitude band. `group` is the CRS family a band's GCPs merge within: south and
# south_midlat share the south polar stereographic CRS, north likewise, and equatorial stands alone
# in an equidistant cylindrical CRS.
#
# `dem_range` is what `dem_near` covers, read from its own label. `lat_range` is what the band is
# assigned, and sits strictly inside `dem_range` so a chunk's buffered AOI and up-sun horizon
# window still land on real elevation at a seam.
#
# Coverage:
#   LDEM_75{N,S}_30M    +/-76.5 to +/-90   (1.5° buffer for long hillshade shadows)
#   LDEM_45{N,S}_100M  +/-60 to +/-76.5  (link to equatorial - lower accuracy than 75N/S or SLDEM)
#   SLDEM2015_512       -60 to 60  (equatorial LOLA-Kaguya merged, more accurate than LDEM equatorial)
#
# `near`/`far` name the products; `bands()` resolves them to paths. Keeping the names here lets
# this module import with no DEM tree present.

BANDS: dict[str, dict] = {
    "south": {
        "lat_range": (-90.0, -76.5),
        "dem_range": (-90.0, -75.0),
        "group": "south",
        "near": (f"{_POLAR}/SOUTH_POLE", "LDEM_75S_30M"),
        "far": (f"{_POLAR}/SOUTH_POLE", "LDEM_45S_100M"),
        "p95_stop_m": 16.5,
    },
    "south_midlat": {
        "lat_range": (-76.5, -60.0),
        "dem_range": (-90.0, -45.0),
        "group": "south",
        "near": (f"{_POLAR}/SOUTH_POLE", "LDEM_45S_100M"),
        "far": (f"{_POLAR}/SOUTH_POLE", "LDEM_45S_100M"),
        "p95_stop_m": 20.0,
    },
    "equatorial": {
        "lat_range": (-60.0, 60.0),
        "dem_range": (-60.0, 60.0),
        "group": "equatorial",
        "near": ("SLDEM", _SLDEM),
        "far": ("SLDEM", _SLDEM),
        "xy_half": (5_458_000.0, 1_819_000.0),  # SLDEM's own bounds; not stereographic
        "p95_stop_m": 16.5,
    },
    "north_midlat": {
        "lat_range": (60.0, 76.5),
        "dem_range": (45.0, 90.0),
        "group": "north",
        "near": (f"{_POLAR}/NORTH_POLE", "LDEM_45N_100M"),
        "far": (f"{_POLAR}/NORTH_POLE", "LDEM_45N_100M"),
        "p95_stop_m": 16.5,
    },
    "north": {
        "lat_range": (76.5, 90.0),
        "dem_range": (75.0, 90.0),
        "group": "north",
        "near": (f"{_POLAR}/NORTH_POLE", "LDEM_75N_30M"),
        "far": (f"{_POLAR}/NORTH_POLE", "LDEM_45N_100M"),
        "p95_stop_m": 16.5,
    },
}
for _name, _e in BANDS.items():
    _lo, _hi = _e["lat_range"]
    _dlo, _dhi = _e["dem_range"]
    if not _dlo <= _lo < _hi <= _dhi:
        raise ValueError(f"{_name}: lat_range {(_lo, _hi)} escapes dem_range {(_dlo, _dhi)}")
    if "xy_half" not in _e:
        _r = _pole_radius_m(_e["group"], _equatorward(_e["dem_range"]))
        _e["xy_half"] = (_r, _r)
# Bands sharing a group must tile, since a group merges into one GCP set. Bands in different
# groups may abut but never overlap.
for _a, _ba in BANDS.items():
    for _z, _bz in BANDS.items():
        if _a >= _z or _ba["group"] != _bz["group"]:
            continue
        if not (_ba["lat_range"][1] <= _bz["lat_range"][0] or _bz["lat_range"][1] <= _ba["lat_range"][0]):
            raise ValueError(f"{_a} and {_z} share a group and overlap in lat_range")

GROUPS = ("south", "north", "equatorial")
GROUP_PRIMARY_BAND = {"south": "south", "north": "north", "equatorial": "equatorial"}
AOI_BUFFER_M = 15_000.0  # covers coarse_shift's bulk error plus the tie-point window half-width
ROW_STEP = 25  # GCP lattice row spacing
# Extra latitude kept in the L1 crop beyond the solved bands, so a swath shifted equatorward by the
# registration still has rows under every chunk's buffered AOI. Crop only; chunks stop at the band
# edge.
L1_BUFFER_DEG = 1.0


@lru_cache
def bands() -> dict[str, dict]:
    """`BANDS` with each band's `near`/`far` product names resolved to `dem_near`/`dem_far` paths.

    Raises `FileNotFoundError` if a LOLA product is missing.
    """
    return {
        name: {
            **{k: v for k, v in e.items() if k not in ("near", "far")},
            "dem_near": _dem(*e["near"]),
            "dem_far": _dem(*e["far"]),
        }
        for name, e in BANDS.items()
    }


def group_lat_range(group: str) -> tuple[float, float]:
    """Latitudes `group` solves: the union of its bands' `lat_range`.

    >>> group_lat_range("south")
    (-90.0, -60.0)
    >>> group_lat_range("equatorial")
    (-60.0, 60.0)
    """
    lo = min(b["lat_range"][0] for b in BANDS.values() if b["group"] == group)
    hi = max(b["lat_range"][1] for b in BANDS.values() if b["group"] == group)
    return (lo, hi)


def l1_lat_range(group: str) -> tuple[float, float]:
    """`group_lat_range` widened by `L1_BUFFER_DEG` at each equatorward end, clipped to the poles.

    What the L1 crop covers, as opposed to what gets solved.

    >>> l1_lat_range("south")
    (-90.0, -59.0)
    >>> l1_lat_range("equatorial")
    (-61.0, 61.0)
    """
    lo, hi = group_lat_range(group)
    return (max(-90.0, lo - L1_BUFFER_DEG if lo > -90.0 else lo), min(90.0, hi + L1_BUFFER_DEG if hi < 90.0 else hi))


def group_xy_half(group: str) -> tuple[float, float]:
    """The widest `xy_half` among `group`'s bands, set by whichever sits closest to the equator."""
    halves: list[tuple[float, float]] = [b["xy_half"] for b in BANDS.values() if b["group"] == group]
    return max(halves, key=lambda h: h[0])


def chunk_cfg(group: str) -> GeorefConfig:
    """`group`'s base `GeorefConfig`. Callers replace `aoi`, `dem_near` and `dem_far` per chunk."""
    primary = bands()[GROUP_PRIMARY_BAND[group]]
    return GeorefConfig(
        pole=group,
        dem_near=primary["dem_near"],
        dem_far=primary["dem_far"],
        row_step=ROW_STEP,
    )


def kernels(day: str) -> list[Path]:
    """Generic SPICE kernels plus the orbit SPKs covering `day`, a "YYYYMMDD" string.

    Raises if the tree or a covering SPK is missing: `sun_geometry` furnishes whatever it is given,
    so an incomplete set yields a wrong sun vector rather than an error.
    """
    import re

    import pandas as pd

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


def _one(pattern: str) -> Path | None:
    """First `ANC_ROOTS` match for `pattern`.

    Globbed, never rebuilt by hand: the detector suffix is not always `d32` (`20231222T0751377198`
    is `d18`).
    """
    for root in ANC_ROOTS:
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    return None


def recal_dir(sid: str, group: str) -> Path:
    """Where `sid`/`group`'s recalibrated geometry lives: `geometry/recalibrated/<day>/<sid>_<group>`.

    One dir holds everything a solve produces for that scene/group -- merged GCPs, chunk fits,
    run log, and the GLT -- mirroring `geometry/calibrated`'s own `<day>` nesting.
    """
    return RECAL_ROOT / "geometry" / "recalibrated" / sid[:8] / f"{sid}_{group}"


def ancillary(sid: str) -> dict[str, Path | None]:
    """The geometry csv and spm for `sid`, plus this pipeline's own recalibrated GCPs/GLT per
    group -- each `None` if not on disk yet."""
    day = sid[:8]
    out: dict[str, Path | None] = {
        "geometry/calibrated": _one(f"geometry/calibrated/{day}/ch2_iir_nci_{sid}_g_grd_*.csv"),
        "miscellaneous/raw": _one(f"miscellaneous/raw/{day}/ch2_iir_nri_{sid}_d_img_*.spm"),
    }
    for group in GROUPS:
        d = recal_dir(sid, group)
        gcps = d / f"{sid}_{group}.gcps"
        glt = d / f"{sid}_{group}_glt.tif"
        out[f"geometry/recalibrated/gcps_{group}"] = gcps if gcps.is_file() and gcps.stat().st_size > 0 else None
        out[f"geometry/recalibrated/glt_{group}"] = glt if glt.is_file() else None
    return out


def nri_zip(sid: str) -> Path | None:
    """The nri bundle for `sid`, searched recursively since `zips/` nests some by region."""
    hits = sorted((ARCHIVE / "zips").rglob(f"ch2_iir_nri_{sid}_d_img_*.zip"))
    return hits[0] if hits else None


def strip_backbone(fgeom: Path, lat_min: float = -90.0, lat_max: float = 90.0):
    """One (lon, lat) per geometry `Scan`, median across `Pixel`: the strip's along-track backbone.

    Left in lon/lat rather than projected, since a strip can cross groups with different CRSs and
    arc length is measured on the sphere by `_haversine_m`.
    """
    import pandas as pd

    df = pd.read_csv(fgeom)
    df["Longitude"] = (df["Longitude"] + 180) % 360 - 180
    df = df[(df.Latitude >= lat_min) & (df.Latitude <= lat_max)]
    g = df.groupby("Scan")[["Longitude", "Latitude"]].median().sort_index()
    return g.index.to_numpy(), g.Longitude.to_numpy(), g.Latitude.to_numpy()


def _haversine_m(lon1, lat1, lon2, lat2, radius=MOON_RADIUS_M):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def _even_chunks(length_m: float, width_range_km: tuple[float, float], overlap_frac: float):
    """Split `length_m` into equal chunks within `width_range_km`, each overlapping the next by
    `overlap_frac` of its own width. Returns [(s0, s1), ...] with s0=0 at the region's start.

    >>> _even_chunks(120_000.0, (100.0, 150.0), 0.10)
    [(0.0, 120000.0)]
    >>> [(round(a), round(z)) for a, z in _even_chunks(400_000.0, (100.0, 150.0), 0.10)]
    [(0, 142857), (128571, 271429), (257143, 400000)]
    """
    lo_m, hi_m = width_range_km[0] * 1000, width_range_km[1] * 1000
    if length_m <= hi_m:
        return [(0.0, length_m)]
    n = 2
    while True:
        w = length_m / (1 + (n - 1) * (1 - overlap_frac))
        if lo_m <= w <= hi_m:
            break
        n += 1
    step = w * (1 - overlap_frac)
    return [(i * step, min(i * step + w, length_m)) for i in range(n)]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(start, stop), ...] per contiguous run of True in a 1D bool array, stop exclusive.

    >>> _runs(np.array([False, True, True, False, True]))
    [(1, 3), (4, 5)]
    >>> _runs(np.array([False, False]))
    []
    """
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


# Rows forced to overlap across a band seam within a group, so the merge cross-fades the seam like
# any other chunk boundary instead of butting the two runs edge to edge. Always extended from the
# band with the wider `dem_range`, so it cannot run off the narrower tile.
SEAM_OVERLAP_M = 15_000.0


@dataclass
class _Run:
    """One contiguous stretch of track inside a single band, in along-track metres."""

    s_lo: float
    s_hi: float
    band: str


def _band_runs(lat: np.ndarray, s: np.ndarray) -> list[_Run]:
    """One `_Run` per contiguous stretch of each band the track enters, ordered along it.

    Same-group seams are widened into a real overlap from the wider-`dem_range` side, so the merge
    cross-fades them like any other chunk boundary.
    """
    runs = []
    for name, b in BANDS.items():
        lo, hi = b["lat_range"]
        for a, z in _runs((lat >= lo) & (lat <= hi)):
            runs.append(_Run(float(s[a]), float(s[z - 1]), name))
    runs.sort(key=lambda r: (r.s_lo, r.s_hi))

    def dem_span(band_name: str) -> float:
        lo, hi = BANDS[band_name]["dem_range"]
        return float(hi) - float(lo)

    for lower, upper in pairwise(runs):
        same_group = BANDS[lower.band]["group"] == BANDS[upper.band]["group"]
        if not same_group or not (0 <= upper.s_lo - lower.s_hi < 5_000.0):
            continue
        if dem_span(lower.band) >= dem_span(upper.band):
            lower.s_hi = min(lower.s_hi + SEAM_OVERLAP_M, float(s[-1]))
        else:
            upper.s_lo = max(upper.s_lo - SEAM_OVERLAP_M, 0.0)
    return runs


def _chunk_aoi(df, scans_in_chunk, band: dict) -> tuple[float, float, float, float]:
    """Bbox of a chunk's track points in its band's CRS, buffered and clamped to `xy_half`."""
    pts = df[df.Scan.isin(scans_in_chunk)]
    lon = np.where(pts.Longitude > 180, pts.Longitude - 360, pts.Longitude)
    x, y = to_stereo(band["group"]).transform(lon, pts.Latitude.values)
    xh, yh = band["xy_half"]
    x0, x1 = np.clip([x.min() - AOI_BUFFER_M, x.max() + AOI_BUFFER_M], -xh, xh)
    y0, y1 = np.clip([y.min() - AOI_BUFFER_M, y.max() + AOI_BUFFER_M], -yh, yh)
    return (float(x0), float(y0), float(x1), float(y1))


def plan_chunks(fgeom: Path, width_range_km: tuple[float, float] = (100.0, 150.0), overlap_frac: float = 0.10):
    """Chunk plan for a strip: one `_even_chunks` pass per contiguous run of each band it crosses,
    so no chunk straddles a band edge and none needs more than one DEM tier.

    Returns chunks ordered along the strip, each with: `i`, `band`, `group`, `dem_near`/`dem_far`,
    `s0`/`s1` (great-circle metres from the strip's southernmost sample), `scan_lo`/`scan_hi` (the
    geometry csv's own `Scan` bounds, for row membership) and `aoi` (the chunk's bbox in its band's
    CRS, buffered by `AOI_BUFFER_M` and clamped to `xy_half`).

    Callers filter by `group` and only merge within one group.
    """
    import pandas as pd

    scans, lon, lat = strip_backbone(fgeom)
    if len(scans) < 2:
        return []
    s = np.concatenate([[0.0], np.cumsum(_haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:]))])
    if lat[0] > lat[-1]:  # keep s increasing from the southernmost end
        s = s.max() - s
        order = np.argsort(s)
        scans, lon, lat, s = scans[order], lon[order], lat[order], s[order]

    df = pd.read_csv(fgeom)
    df["Longitude"] = (df["Longitude"] + 180) % 360 - 180

    resolved = bands()
    chunks: list[dict] = []
    for run in _band_runs(lat, s):
        b = resolved[run.band]
        for cs0, cs1 in _even_chunks(run.s_hi - run.s_lo, width_range_km, overlap_frac):
            gs0, gs1 = run.s_lo + cs0, run.s_lo + cs1
            in_chunk = scans[(s >= gs0) & (s <= gs1)]
            chunks.append({
                "i": len(chunks),
                "band": run.band,
                "group": b["group"],
                "dem_near": b["dem_near"],
                "dem_far": b["dem_far"],
                "p95_stop_m": b["p95_stop_m"],
                "s0": float(gs0),
                "s1": float(gs1),
                "scan_lo": int(in_chunk.min()),
                "scan_hi": int(in_chunk.max()),
                "aoi": _chunk_aoi(df, in_chunk, b),
            })

    # Split any residual row gap between same-group neighbours at its midpoint, so no row is left
    # unowned. Seam widening makes an overlap the common case, which is left alone.
    for a, z in pairwise(chunks):
        if BANDS[a["band"]]["group"] != BANDS[z["band"]]["group"]:
            continue
        gap = z["scan_lo"] - a["scan_hi"] - 1
        if gap > 0:
            mid = a["scan_hi"] + (gap + 1) // 2
            a["scan_hi"], z["scan_lo"] = mid, mid + 1
    return chunks
