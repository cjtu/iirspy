"""File I/O and data locations: find ISSDC bundle members and reprocessed products, read labels, write ENVI/GeoTIFF.

`Paths` resolves the data trees from `IIRS_*` env vars on first use. `get_iirs_paths` finds each scene's files by
extension, in ISSDC's `<subdir>/<YYYYMMDD>/` layout first, then flat, then by a recursive walk. Cube writers
stream row blocks, so peak memory is one block whatever the scene size.
"""

import functools
import hashlib
import os
import re
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pdr
import rasterio
import xarray as xr

CHUNKSIZE = 128 * 2**20  # [B] dask y-chunk target (dask's array.chunk-size default; profiled fastest + lowest RAM)


LVL2MTC = {0: "nri", 1: "nci", 2: "ndi"}  # data level -> ISSDC "Mission-Type-Camera" identifier
LVL2DIR = {0: "raw", 1: "calibrated", 2: "derived"}  # data level -> ISSDC bundle subdir
# ext -> subdir of an ISSDC bundle ({lvl} = LVL2DIR[level]); "xml-csv" is the csv's xml label
ISSDC_SUBDIR = {
    **dict.fromkeys(("png", "xml-png"), "browse/{lvl}"),
    **dict.fromkeys(("hdr", "qub", "xml"), "data/{lvl}"),
    **dict.fromkeys(("csv", "xml-csv"), "geometry/calibrated"),
    **dict.fromkeys(("lbr", "oat", "oath", "spm"), "miscellaneous/{lvl}"),
}
# ext -> (subdir, suffix) of iirspy's reprocessed products, one per solve group; "tif" depends on the level
REPROCESSED = {
    "gcps": ("geometry/recalibrated", ".gcps"),
    "loc": ("geometry/recalibrated", "_loc.tif"),
    "obs": ("data/rederived", "_obs.tif"),
}
ENVI_DTYPE_CODES = {"uint8": 1, "int16": 2, "int32": 3, "float32": 4, "float64": 5, "uint16": 12}


try:
    IIRSPY_VERSION = metadata.version("iirspy")
except metadata.PackageNotFoundError:  # running from a source tree not pip-installed
    IIRSPY_VERSION = "0+source"


class Paths:
    """Data locations. A field not passed falls back to its `IIRS_*` env var when first read, so a run
    only needs the ones it touches: `IIRS_ARCHIVE`, `IIRS_DEM_ROOTS`, `IIRS_RECAL_ROOT`, and optionally
    `IIRS_ANC_ROOTS` (default: the archive) and `IIRS_SPICE` (default: `<archive>/spice`).
    """

    def __init__(self, archive=None, anc_roots=None, dem_roots=None, spice=None, recal_root=None):
        self._given: dict[str, Any] = {
            "archive": archive,
            "anc_roots": anc_roots,
            "dem_roots": dem_roots,
            "spice": spice,
            "recal_root": recal_root,
        }

    @staticmethod
    def _env(var: str) -> str:
        if var not in os.environ:
            raise RuntimeError(f"set {var} (see README)")
        return os.environ[var]

    @property
    def archive(self) -> Path:
        return self._given["archive"] or Path(self._env("IIRS_ARCHIVE"))

    @property
    def anc_roots(self) -> tuple[Path, ...]:
        if self._given["anc_roots"]:
            return tuple(self._given["anc_roots"])
        return (
            tuple(Path(p) for p in os.environ["IIRS_ANC_ROOTS"].split(":"))
            if "IIRS_ANC_ROOTS" in os.environ
            else (self.archive,)
        )

    @property
    def dem_roots(self) -> tuple[Path, ...]:
        return (
            self._given["dem_roots"]
            if self._given["dem_roots"] is not None
            else tuple(Path(p) for p in self._env("IIRS_DEM_ROOTS").split(":"))
        )

    @property
    def spice(self) -> Path:
        return self._given["spice"] or Path(os.environ.get("IIRS_SPICE", self.archive / "spice"))

    @property
    def recal_root(self) -> Path:
        return self._given["recal_root"] or Path(self._env("IIRS_RECAL_ROOT"))


def band_numbers(img):
    """Original IIRS band numbers, from the GeoTIFF band_numbers tag or ENVI band names, else 1..N."""
    for attr in ("band_numbers", "band_names"):
        val = img.attrs.get(attr, "")
        if isinstance(val, str) and all(b.strip().isdigit() for b in val.split(",")):
            return [int(b) for b in val.split(",")]
    return img.band.values


def apply_envi_start(da):
    """Shift x/y by the ENVI `x start`/`y start` offset, which GDAL reports but does not apply."""
    for dim, attr in (("x", "x_start"), ("y", "y_start")):
        start = da.attrs.get(attr)
        # Only when the file carried no transform (coords still 0-based); else it is already applied
        if start is not None and float(da[dim][0]) < 1:
            da = da.assign_coords({dim: da[dim].values + int(start) - 1})
    return da


def extract(fzip, out_dir, **kw):
    """`issdc_iirs.fetch` on one local bundle (`kw`: include/exclude/bands), md5-verified against its
    PDS4 labels. Raises on failure, where `fetch` itself only prints and returns an empty list."""
    # Local import: this module is also imported in envs without issdc.
    from issdc_iirs import fetch

    paths = fetch([str(fzip)], str(out_dir), **kw)[str(fzip)]
    if not paths:
        raise RuntimeError(f"issdc_iirs extracted nothing from {fzip} (see its output above)")
    return paths


def unzip_iirs(ddir, basename, level):
    """Find an ISSDC bundle for `basename` under `ddir` and extract all of it, qub included, beside itself."""
    mtc = LVL2MTC[level]
    f = next((f for f in Path(ddir).glob(f"**/*{mtc}*.zip") if basename in f.stem), None)
    if f is None:
        raise FileNotFoundError(f"Image {basename} not found in {ddir}. Please download from ISSDC or check file path.")
    extract(f, f.parent, exclude=())
    paths = get_iirs_paths(f.parent, level=level, basenames=[basename])
    if "qub" not in paths:
        raise RuntimeError("Unzip failed.")
    return paths


def iirsbasename(input_str):
    """Return the image basename from str_in (e.g., 20201226T1745264921)"""
    pattern = r"\d{8}T\d{10}"
    match = re.search(pattern, input_str)
    if match is None:
        raise ValueError(f"Can't parse basename: {input_str}")
    return match.group()


def _iirs_basename(img_path_obj):
    """Return IIRS basename e.g. 20210122T0920157625 from pathlib path, or None if the filename
    doesn't follow the IIRS naming convention."""
    parts = img_path_obj.stem.split("_")
    return parts[3] if len(parts) > 3 else None


def _find_direct(ddir, subdir, ext, bnames):
    """Fast lookup under IIRS archive path: `<ddir>/<subdir>/<sid[:8]>/*.<ext>`."""
    found: dict[str, Path] = {}
    for sid in bnames:
        date_dir = Path(ddir) / subdir / sid[:8]
        if date_dir.is_dir():
            for f in date_dir.glob(f"*.{ext}"):
                if _iirs_basename(f) == sid:
                    found[sid] = f
    return found


def _find_flat(ddir, ext, wanted):
    """Fast lookup for files dropped directly in `ddir` instead of the normal tree."""
    return {b: f for f in Path(ddir).glob(f"*.{ext}") if f.is_file() and (b := _iirs_basename(f)) in wanted}


def _find_recursive(ddir, subdir, ext, wanted):
    """Slow full recursive walk search for missing files (last resort)."""
    return {b: f for f in Path(ddir).glob(f"**/{subdir}/**/*.{ext}") if (b := _iirs_basename(f)) in wanted}


def _find_reprocessed(ddir, subdir, suffix, bnames, group=None):
    """Reprocessed products at `<ddir>/<subdir>/<sid[:8]>/<sid>_<group><suffix>`, the flat day-dir
    layout of ISSDC's own trees. `group=None` matches any group but raises if a sid has several."""
    found: dict[str, Path] = {}
    for sid in bnames:
        hits = sorted((Path(ddir) / subdir / sid[:8]).glob(f"{sid}_{group or '*'}{suffix}"))
        if len(hits) > 1:
            raise ValueError(f"{sid} has one {suffix} per group, pass group=: {[f.name for f in hits]}")
        if hits:
            found[sid] = hits[0]
    return found


def get_iirs_paths(
    ddir,
    basenames,
    exts=("qub", "hdr", "xml", "csv", "xml-csv", "lbr", "oat", "oath", "spm", "png", "xml-png"),
    level=1,
    group=None,
):
    """Return a list of paths to IIRS image, geom, and misc files for the given sids.

    `group` picks one solve group's reprocessed products (gcps/loc/obs/tif); see `_find_reprocessed`.
    """
    out: dict[str, Any] = {}
    bnames = [basenames] if isinstance(basenames, str) else list(basenames)
    for ext in exts:
        if ext == "tif":
            subdir, suffix = ("data/recalibrated", "_l1_rad.tif") if level == 1 else ("data/rederived", "_l2_refl.tif")
            found = _find_reprocessed(ddir, subdir, suffix, bnames, group)
        elif ext in REPROCESSED:
            found = _find_reprocessed(ddir, *REPROCESSED[ext], bnames, group)
        elif ext in ISSDC_SUBDIR:
            # cheapest lookup first, escalating only for the sids the previous tier missed
            subdir, e = ISSDC_SUBDIR[ext].format(lvl=LVL2DIR[level]), ext.split("-")[0]
            found = _find_direct(ddir, subdir, e, bnames)
            if missing := [sid for sid in bnames if sid not in found]:
                found.update(_find_flat(ddir, e, missing))
            if missing := [sid for sid in bnames if sid not in found]:
                found.update(_find_recursive(ddir, subdir, e, missing))
        else:
            raise ValueError(f"Unknown IIRS file extension: {ext}")
        if found:
            out[ext] = found
    # Add list of basenames to dict
    if "qub" in out:
        out["imgs"] = list(out["qub"].keys())
    return out


def get_iirs_shape_meta(fimg):
    """Return the shape of fimg from metadata (bands, lines, samples)."""
    axs = pdr.open(fimg).metaget("Array_3D_Spectrum").getall("Axis_Array")
    bands, lines, samples = (int(axs[i]["elements"]) for i in range(3))
    return bands, lines, samples


def provenance(**extra) -> dict:
    """Package version, Python, UTC time and argv of this run, plus `extra`."""
    import datetime
    import platform
    import sys

    return {
        "iirspy_version": IIRSPY_VERSION,
        "python": platform.python_version(),
        "utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        **extra,
    }


def xarray_keep_attrs(fn):
    """Decorator: run `fn` with xarray's `keep_attrs=True` (older xarray drops attrs in arithmetic)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with xr.set_options(keep_attrs=True):
            return fn(*args, **kwargs)

    return wrapper


def ncpu() -> int:
    """Cores this process may use. `os.cpu_count()` reports the whole machine, not the cpuset."""
    return len(os.sched_getaffinity(0))


def spice_kernels(day, spice_dir=None) -> list[Path]:
    """Generic SPICE kernels plus the orbit SPKs covering `day`, a "YYYYMMDD" string.

    `spice_dir` (default env `IIRS_SPICE`) holds lsk/, pck/, sclk/, fk/ and spk/. Raises if it or a
    covering SPK is missing.
    """
    import os

    spice_dir = spice_dir or os.environ.get("IIRS_SPICE")
    if not spice_dir or not Path(spice_dir).is_dir():
        raise FileNotFoundError(f"SPICE kernel tree not found at {spice_dir!r} -- set IIRS_SPICE")
    spice_dir = Path(spice_dir)
    ks = [spice_dir / p for p in ("lsk/naif0012.tls", "pck/pck00010.tpc", "sclk/ch2_sclk_v1.tsc", "fk/ch2_v01.tf")]
    ks = [p for p in ks if p.exists()]
    d = pd.to_datetime(day, format="%Y%m%d")
    for f in sorted(spice_dir.glob("spk/*.bsp")):
        m = re.findall(r"(\d{2}[A-Za-z]{3}\d{4})", f.name)
        if (
            len(m) == 2 and pd.to_datetime(m[0], format="%d%b%Y") <= d <= pd.to_datetime(m[1], format="%d%b%Y")
        ) or f.name.startswith("de4"):
            ks.append(f)
    if not any(f.suffix == ".bsp" for f in ks):
        raise FileNotFoundError(f"no SPK in {spice_dir / 'spk'} covers {day}")
    return ks


def write_envi_hdr(
    fhdr,
    nx,
    ny,
    nband,
    wls,
    description="IIRS",
    bands=None,
    x_start=1,
    y_start=1,
    dtype="float32",
    band_names=None,
    extra_tags=None,
):
    """Write an ENVI header for a BIL cube.

    `bands` are the IIRS band numbers of the planes (default 1..nband); they go in `band names` so
    a band subset reads back as itself rather than 1..N, unless `band_names` (arbitrary strings,
    e.g. non-spectral per-band names) overrides them -- that also drops the wavelength lines,
    since a non-wavelength product has none. `x_start`/`y_start` are the 1-indexed sample/line of
    this crop in the parent scene: ENVI carries no transform, so they are the only place a crop's
    absolute position survives (GeoTIFF uses the geotransform instead). `extra_tags` (scalar
    provenance attrs) are written as extra header keys; GDAL surfaces unknown ENVI header keys as
    dataset metadata.
    """
    dtype_code = ENVI_DTYPE_CODES[str(np.dtype(dtype))]
    if band_names is not None:
        names = ", ".join(str(n) for n in band_names)
        wl_lines = ""
    else:
        bands = range(1, nband + 1) if bands is None else bands
        names = ", ".join(str(int(b)) for b in bands)
        if wls is None:
            wl_lines = ""
        else:
            wl_str = ", ".join(f"{float(w):.4f}" for w in wls)
            wl_lines = f"wavelength units = Nanometers\nwavelength = {{{wl_str}}}\n"
    extra_lines = "".join(f"{k} = {v}\n" for k, v in (extra_tags or {}).items())
    Path(fhdr).write_text(
        "ENVI\n"
        f"description = {{ {description} }}\n"
        f"samples = {nx}\nlines = {ny}\nbands = {nband}\n"
        "header offset = 0\nfile type = ENVI Standard\n"
        f"data type = {dtype_code}\ninterleave = bil\nbyte order = 0\n"
        f"x start = {int(x_start)}\ny start = {int(y_start)}\n"
        f"band names = {{{names}}}\n"
        f"{wl_lines}{extra_lines}"
    )


def write_envi_bil(da, fout, sub_rows, description="IIRS"):
    """Sequentially stream a (band, y, x) DataArray to an ENVI BIL file (bounded memory).

    Integer dtypes are written as-is (e.g. QA's uint16); everything else is written float32.
    `sub_rows` (rows per write step) is required.
    """
    fout = str(fout)
    nband, ny, nx = da.shape
    dtype = str(da.dtype) if np.issubdtype(da.dtype, np.integer) else "float32"
    with open(fout, "wb") as f:
        for a in range(0, ny, sub_rows):
            blk = np.asarray(da.isel(y=slice(a, a + sub_rows)).values, dtype=dtype)  # (band, rows, x)
            blk.transpose(1, 0, 2).tofile(f)  # C-order of (row, band, x) == ENVI BIL
    has_wl = "wl" in da.coords
    wls = da.wl.values if has_wl else None
    band_names = None if has_wl else da.coords.get("band_name")
    if band_names is not None:
        band_names = [str(n) for n in band_names.values]
    # Pixel centres (n + 0.5) -> ENVI's 1-indexed sample/line of the crop's upper-left pixel
    x_start, y_start = (int(np.floor(float(da[d].min()))) + 1 for d in ("x", "y"))
    extra_tags = {
        k: v
        for k, v in da.attrs.items()
        if isinstance(v, str | int | float | bool) and k not in ("bands", "band_names", "description")
    }
    write_envi_hdr(
        Path(fout).with_suffix(".hdr"),
        nx,
        ny,
        nband,
        wls,
        description,
        da.band.values,
        x_start,
        y_start,
        dtype=dtype,
        band_names=band_names,
        extra_tags=extra_tags,
    )
    return fout


class ChecksumError(Exception):
    """Exception raised for checksum mismatches."""

    def __init__(self, fname, actual, expected):
        self.message = f"Checksum failed for {fname}. Actual: {actual}. Expected: {expected}"
        super().__init__(self.message)


def get_md5(fname):
    """Return md5 hash of file."""
    hash_md5 = hashlib.md5()  # noqa: S324
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def checksum(fname):
    """Raise an exception if the MD5 checksum of file does not match label."""
    actual = get_md5(fname)
    expected = pdr.open(fname).metaget("md5_checksum")
    if actual != expected:
        raise ChecksumError(fname, actual, expected)


def rows_per_block(da, sub_rows=None):
    """Rows per write step. Defaults to the array's own y-chunk, so a block is read exactly once.

    An explicit `sub_rows` wins; unchunked, `CHUNKSIZE` worth of rows.
    """
    if sub_rows:
        return sub_rows
    chunks = getattr(da, "chunks", None)
    if chunks:
        return max(chunks[da.dims.index("y")])
    nband, _, nx = da.shape
    return max(1, int(CHUNKSIZE / (nband * nx * 4)))


def write_camera_product(da, fout, row_block=None, compress="ZSTD", predictor=None):
    """Format dispatch for any camera-space DataArray: `.tif` -> GeoTIFF, else ENVI BIL."""
    fout = str(fout)
    row_block = rows_per_block(da, row_block)
    if fout.lower().endswith(".tif"):
        return save_geotiff(fout, row_block, da, compress=compress, predictor=predictor)
    return write_envi_bil(da, fout, row_block, f"IIRS {da.attrs.get('name', '')}".strip())


def save_geotiff(fout, row_block, da, compress="ZSTD", predictor=None):
    """Write a BigTIFF sequentially in windowed blocks (bounded memory).

    Integer `da` (e.g. QA's uint16) keeps its own dtype and nodata from `da.attrs["nodata"]` (or
    none); everything else writes float32 with NaN nodata. `compress`/`predictor` default to ZSTD
    with predictor 2 (int) or 3 (float).
    """
    from dask.diagnostics import ProgressBar
    from rasterio.windows import Window

    nband, ny, nx = da.shape
    is_int = np.issubdtype(da.dtype, np.integer)
    dtype = str(da.dtype) if is_int else "float32"
    predictor = predictor if predictor is not None else (2 if is_int else 3)
    nodata = da.attrs.get("nodata") if is_int else np.nan
    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": nband,
        "dtype": dtype,
        "nodata": nodata,
        "crs": da.rio.crs if da.rio.crs else None,
        "transform": da.rio.transform(),
        "compress": compress,
        "predictor": predictor,
        "tiled": True,
        "interleave": "band",  # per-band planes: viewers read one band without decompressing all 256
        "BIGTIFF": "YES",
    }
    wls = [f"{float(w):.2f}" for w in da.wl.values] if "wl" in da.coords else None
    band_names = None if wls or "band_name" not in da.coords else da.coords["band_name"].values
    band_units = None if band_names is None else da.coords["units"].values
    # Provenance: scalar attrs (incl. empirical_notes JSON) round-trip as GDAL metadata tags.
    # Skip stale bands/band_names attrs; band_numbers and per-band descriptions below are authoritative.
    tags = {
        k: str(v)
        for k, v in da.attrs.items()
        if isinstance(v, str | int | float | bool) and k not in ("bands", "band_names")
    }
    tags["PROVENANCE_IIRSPY_VERSION"] = IIRSPY_VERSION
    if wls:
        # Which IIRS bands these planes are: 1..256 on a full cube, arbitrary on a subset
        tags["band_numbers"] = ",".join(str(int(b)) for b in da.band.values)
    with rasterio.open(fout, "w", **profile) as dst, ProgressBar():
        if tags:
            dst.update_tags(**tags)
        for y0 in range(0, ny, row_block):
            y1 = min(y0 + row_block, ny)
            block = da.isel(y=slice(y0, y1)).values.astype(dtype, copy=False)
            dst.write(block, window=Window(0, y0, nx, y1 - y0))
        if wls:
            for i, wl in enumerate(wls):
                dst.set_band_description(i + 1, wl)
                # Per-band "wavelength"/"wavelength_units" is the generic hyperspectral metadata
                # GDAL/QGIS read for spectral band labels; description alone isn't.
                dst.update_tags(i + 1, wavelength=wl, wavelength_units="Nanometers")
        elif band_names is not None and band_units is not None:
            for i, (name, unit) in enumerate(zip(band_names, band_units, strict=True)):
                dst.set_band_description(i + 1, str(name))
                dst.update_tags(i + 1, UNITS=str(unit), DEFINITION=f"{name} [{unit}]")
    return fout
