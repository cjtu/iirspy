import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pdr
import rasterio

CHUNKSIZE = 128 * 2**20  # [B] dask y-chunk target (dask's array.chunk-size default; profiled fastest + lowest RAM)


# Convert data level to "Mission-Type-Camera" identifier
LVL2MTC = {0: "nri", 1: "nci", 2: "ndi"}


try:
    IIRSPY_VERSION = metadata.version("iirspy")
except metadata.PackageNotFoundError:  # running from a source tree not pip-installed
    IIRSPY_VERSION = "0+source"


@dataclass(frozen=True)
class Paths:
    archive: Path
    anc_roots: tuple[Path, ...]
    dem_roots: tuple[Path, ...]
    spice: Path
    recal_root: Path


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
    """Find a PRADAN bundle for `basename` under `ddir` and extract all of it, qub included, beside itself."""
    mtc = LVL2MTC[level]
    f = next((f for f in Path(ddir).glob(f"**/*{mtc}*.zip") if basename in f.stem), None)
    if f is None:
        raise FileNotFoundError(
            f"Image {basename} not found in {ddir}. Please download from PRADAN or check file path."
        )
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


def _find_reprocessed(ddir, subdir, bnames, suffix, group=None):
    """Reprocessed products at `<ddir>/<subdir>/<sid[:8]>/<sid>_<group><suffix>`, the flat day-dir
    layout of PRADAN's own trees. `group=None` matches any group but raises if a sid has several."""
    found: dict[str, Path] = {}
    for sid in bnames:
        hits = sorted((Path(ddir) / subdir / sid[:8]).glob(f"{sid}_{group or '*'}{suffix}"))
        if len(hits) > 1:
            raise ValueError(f"{sid} has one {suffix} per group, pass group=: {[f.name for f in hits]}")
        if hits:
            found[sid] = hits[0]
    return found


def _find_by_basename(ddir, subdir, ext, bnames):
    """Look up each sid in `bnames`, cheapest strategy first (see `_find_direct`/`_find_flat`/
    `_find_recursive`), only escalating for sids the previous tier didn't find."""
    found = _find_direct(ddir, subdir, ext, bnames)
    missing = [sid for sid in bnames if sid not in found]
    if missing:
        found.update(_find_flat(ddir, ext, missing))
        missing = [sid for sid in bnames if sid not in found]
    if missing:
        found.update(_find_recursive(ddir, subdir, ext, missing))
    return found


def _resolve_ext(ddir, ext, level, bnames, group=None):
    """Every path `get_iirs_paths` knows how to locate for one `ext` at one `level`."""
    LVL2DIR = {0: "raw", 1: "calibrated", 2: "derived"}
    if ext == "gcps":
        return _find_reprocessed(ddir, "geometry/recalibrated", bnames, ".gcps", group)
    if ext == "loc":
        return _find_reprocessed(ddir, "geometry/recalibrated", bnames, "_loc.tif", group)
    if ext == "obs":
        return _find_reprocessed(ddir, "data/rederived", bnames, "_obs.tif", group)
    if ext == "tif":
        if level == 1:
            return _find_reprocessed(ddir, "data/recalibrated", bnames, "_l1_rad.tif", group)
        return _find_reprocessed(ddir, "data/rederived", bnames, "_l2_refl.tif", group)
    if ext in ("png", "xml-png"):
        subdir = "browse/" + LVL2DIR[level]
    elif ext in ("hdr", "qub", "xml"):
        subdir = "data/" + LVL2DIR[level]
    elif ext in ("csv", "xml-csv"):
        subdir = "geometry/calibrated"
    elif ext in ("lbr", "oat", "oath", "spm"):
        subdir = "miscellaneous/" + LVL2DIR[level]
    else:
        raise ValueError(f"Unknown IIRS file extension: {ext}")
    return _find_by_basename(ddir, subdir, ext.split("-")[0], bnames)


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
        found = _resolve_ext(ddir, ext, level, bnames, group)
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


def get_step_log(out, prev, step: str, method: str, t0: float, cfg=None, fields=(), **inputs):
    """`out` carrying `prev`'s attrs and its provenance list plus one entry for this step.

    `t0` is the step's `time.monotonic()` start; `fields` names the `cfg` fields the step read.
    """
    entry = {
        "step": step,
        "method": method,
        "cfg": {f: getattr(cfg, f) for f in fields},
        "version": IIRSPY_VERSION,
        "inputs": inputs,
        "wall_s": round(time.monotonic() - t0, 3),
    }
    prov = [*json.loads(prev.attrs.get("provenance", "[]")), entry]
    out.attrs = {**prev.attrs, **out.attrs, "provenance": json.dumps(prov, default=str)}
    return out


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


ENVI_DTYPE_CODES = {"uint8": 1, "int16": 2, "int32": 3, "float32": 4, "float64": 5, "uint16": 12}


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


def _write_bil_rows(f, da, i0, i1, sub_rows, dtype="float32"):
    """Stream rows [i0:i1) of a (band, y, x) DataArray to open file f as ENVI BIL."""
    for a in range(i0, i1, sub_rows):
        b = min(a + sub_rows, i1)
        blk = np.asarray(da.isel(y=slice(a, b)).values, dtype=dtype)  # (band, rows, x)
        blk.transpose(1, 0, 2).tofile(f)  # C-order of (row, band, x) == ENVI BIL


def write_envi_bil(da, fout, sub_rows, description="IIRS"):
    """Sequentially stream a (band, y, x) DataArray to an ENVI BIL file (bounded memory).

    Integer dtypes are written as-is (e.g. QA's uint16); everything else is written float32.
    `sub_rows` (rows per write step) is required.
    """
    fout = str(fout)
    nband, ny, nx = da.shape
    dtype = str(da.dtype) if np.issubdtype(da.dtype, np.integer) else "float32"
    with open(fout, "wb") as f:
        _write_bil_rows(f, da, 0, ny, sub_rows, dtype)
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
