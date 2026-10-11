import json
import warnings
from abc import ABC, abstractmethod
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import pdr
import rasterio
import xarray as xr
from rioxarray.exceptions import NoDataInBounds

import iirspy.frames as frames
import iirspy.photometry as photometry
import iirspy.utils as utils
from iirspy.config import POLISH, L1Config, L2Config
from iirspy.empirical import empirical_frames

# Skip div 0 and 0/0 warnings
np.seterr(divide="ignore", invalid="ignore")


def bounds2extent(bounds):
    """Convert bounds to extent tuple."""
    minx, miny, maxx, maxy = bounds
    return (minx, maxx, miny, maxy)


def _try_load_next_level_metadata(basename, target_level, directory, xyextent):
    """Attempt to load ancillary files for target_level. Return dict with found files or None."""
    directory = Path(directory).expanduser().absolute()
    result = {"geometry": None, "spm": None, "geometry_df": None}

    try:
        paths = utils.get_iirs_paths(directory, level=target_level, basenames=[basename])
        if target_level >= 1:
            csv_path = paths.get("csv", {}).get(basename)
            if csv_path and csv_path.exists():
                result["geometry"] = csv_path
                df, _ = utils.parse_geom(csv_path, xyextent=xyextent)
                result["geometry_df"] = df

            spm_path = paths.get("spm", {}).get(basename)
            if spm_path and spm_path.exists():
                result["spm"] = spm_path
    except (FileNotFoundError, KeyError, ValueError):
        pass

    return result


def _write_json_atomic(path, obj):
    """Write `obj` as json via a temp file + rename, so a kill mid-write can't leave a torn file."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def _resume_row(fout, fprog, key):
    """First row `_save_geotiff` still has to write, from a previous run's progress sidecar.

    0 unless `fout` and its sidecar both exist, the sidecar agrees with `key` (same cube shape and
    block size -- a different one is a different product, not a resume), and `fout` still opens for
    update. A file killed mid-flush may not reopen at all, so that case restarts from 0 too.
    """
    if not (Path(fout).exists() and Path(fprog).exists()):
        return 0
    try:
        prog = json.loads(Path(fprog).read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    if {k: prog.get(k) for k in key} != key or not prog.get("rows_done"):
        return 0
    try:
        with rasterio.open(fout, "r+"):
            pass
    except Exception:
        return 0
    return int(prog["rows_done"])


def _band_numbers(img):
    """Original IIRS band numbers, from the GeoTIFF band_numbers tag or ENVI band names, else 1..N."""
    for attr in ("band_numbers", "band_names"):
        val = img.attrs.get(attr, "")
        if isinstance(val, str) and all(b.strip().isdigit() for b in val.split(",")):
            return [int(b) for b in val.split(",")]
    return img.band.values


def _apply_envi_start(da):
    """Shift x/y by the ENVI `x start`/`y start` offset, which GDAL reports but does not apply."""
    for dim, attr in (("x", "x_start"), ("y", "y_start")):
        start = da.attrs.get(attr)
        # Only when the file carried no transform (coords still 0-based); else it is already applied
        if start is not None and float(da[dim][0]) < 1:
            da = da.assign_coords({dim: da[dim].values + int(start) - 1})
    return da


def _chunks(nband, ny, nx, chunk=True):
    """dask chunk dict for a (band, y, x) cube, or None to leave it unchunked.

    `chunk` is a dict to use as-is, or True to split along y at `utils.CHUNKSIZE`. A cube already
    under one chunk's worth stays whole.
    """
    if not chunk:
        return None
    if isinstance(chunk, dict):
        return chunk
    if nband * ny * nx * 4 <= utils.CHUNKSIZE:
        return None
    return {"band": nband, "y": int(utils.CHUNKSIZE / (nband * nx * 4)), "x": nx}


def _row_block(da, sub_rows=None):
    """Rows per write step. Defaults to the array's own y-chunk, so a block is read exactly once.

    An explicit `sub_rows` wins; unchunked, `utils.CHUNKSIZE` worth of rows.
    """
    if sub_rows:
        return sub_rows
    chunks = getattr(da, "chunks", None)
    if chunks:
        return max(chunks[da.dims.index("y")])
    nband, _, nx = da.shape
    return max(1, int(utils.CHUNKSIZE / (nband * nx * 4)))


FLAT_SMILE_MIN = 0.2  # clip flat*smile away from 0 before dividing (avoids blow-ups)


def _in_wl_ranges(wl, ranges):
    """True where `wl` falls in any (lo, hi) of `ranges` (inclusive); all False if `ranges` is falsy."""
    hit = xr.zeros_like(wl, dtype=bool)
    for lo, hi in ranges or ():
        hit = hit | ((wl >= lo) & (wl <= hi))
    return hit


def _narrow_bands(rad, output_bands):
    """`rad` restricted to `output_bands`, or `rad` unchanged if `output_bands` is None."""
    return rad if output_bands is None else rad.sel(band=output_bands)


def _expand_to_full_bands(da):
    """Reindex `da`'s band dim onto the full 1..256 IIRS range, NaN-filling any band not present.

    A cube narrowed to a band subset writes only its own bands contiguously, this restores the full
    1:1 mapping so the dropped/OSF/invalid bands become all-NaN, consistent with IIRS images.
    """
    all_bands = np.arange(1, len(utils.get_wls()) + 1)
    if da.sizes["band"] == len(all_bands) and np.array_equal(da.band.values, all_bands):
        return da
    da = da.reindex(band=all_bands)
    return da.assign_coords(wl=("band", utils.get_wls()))


def _retime(rad, fix, frame_offsets, qub):
    """(`rad` on true frames, (band, y) lost-frame mask) via `iirspy.frames.apply`, or (`rad`, None) when off.

    No `frame_offsets` builds the table from the full cube `qub` and caches it beside it as
    `<qub stem>_frames_offsets.csv`.
    """
    if not fix:
        return rad, None
    if frame_offsets is None:
        frame_offsets = Path(qub).with_name(f"{Path(qub).stem}_frames_offsets.csv")
        if not frame_offsets.exists():
            try:
                frames.build_table(qub, frame_offsets)
            except ValueError as e:
                warnings.warn(f"fix_frames skipped: {e}", UserWarning, stacklevel=3)
                return rad, None
    return frames.apply(rad, frame_offsets)


def _attach_empirical_frames(rad, dark, flat, smile, attach):
    """Attach dark/flat/smile as (band, x) coords on `rad` when `attach`, for save()'s sidecar.

    dark is (x,)-only zeros when no shadow was found; broadcast to flat's (band, x) so all three
    sidecar frames share one shape.
    """
    if not attach:
        return rad
    return rad.assign_coords(
        empirical_dark=(("band", "x"), dark.broadcast_like(flat).values),
        empirical_flat=(("band", "x"), flat.values),
        empirical_smile=(("band", "x"), smile.values),
    )


def _write_camera_product(da, fout, row_block=None, compress="ZSTD", predictor=None):
    """Format dispatch for any camera-space DataArray: `.tif` -> GeoTIFF, else ENVI BIL."""
    fout = str(fout)
    row_block = _row_block(da, row_block)
    if fout.lower().endswith(".tif"):
        return _save_geotiff(fout, row_block, da, compress=compress, predictor=predictor)
    return utils.write_envi_bil(da, fout, row_block, f"IIRS {da.attrs.get('name', '')}".strip())


def _save_geotiff(fout, row_block, da, compress="ZSTD", predictor=None):
    """Write a BigTIFF sequentially in windowed blocks (bounded memory).

    Restartable: each finished block records `rows_done` in a `<fout>.progress` sidecar, and a
    rerun of the same shape reopens the file and picks up there.

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
    tags["PROVENANCE_IIRSPY_VERSION"] = utils.IIRSPY_VERSION
    if wls:
        # Which IIRS bands these planes are: 1..256 on a full cube, arbitrary on a subset
        tags["band_numbers"] = ",".join(str(int(b)) for b in da.band.values)
    fprog = Path(str(fout) + ".progress")
    key = {"shape": [nband, ny, nx], "row_block": int(row_block)}
    y_start = _resume_row(fout, fprog, key)
    mode = "r+" if y_start else "w"
    with rasterio.open(fout, mode, **({} if y_start else profile)) as dst, ProgressBar():
        if tags:
            dst.update_tags(**tags)
        for y0 in range(y_start, ny, row_block):
            y1 = min(y0 + row_block, ny)
            block = da.isel(y=slice(y0, y1)).values.astype(dtype, copy=False)
            dst.write(block, window=Window(0, y0, nx, y1 - y0))
            _write_json_atomic(fprog, {**key, "rows_done": y1})
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
    fprog.unlink(missing_ok=True)
    return fout


class IIRSData(ABC):
    """Abstract base class for IIRS data products."""

    def __init__(self, basename, directory=".", extent=(None, None, None, None), chunk=True, level=1):
        """
        Initialize the IIRS data class.

        Parameters
        ----------
        basename : str
            Basename of the image to read (e.g. 20201214T0844306700).
        directory : str
            Path to the directory containing IIRS data files.
        extent : tuple
            Extent in (minx, maxx, miny, maxy) format.
        chunk : bool or dict
            Chunk image automatically (default: True). Or supply dict of x,y,band chunk sizes (see dask).
        """
        self.basename = utils.iirsbasename(basename)
        self.directory = Path(directory).expanduser().absolute()
        self.extent = extent
        self.level = level

        # Get paths to relevant files
        paths = utils.get_iirs_paths(self.directory, level=self.level, basenames=[self.basename])
        if "qub" not in paths:
            paths = utils.unzip_iirs(self.directory, self.basename, self.level)
        if "qub" not in paths:
            raise FileNotFoundError(f"{self.basename} not found at {self.directory}.")
        # An ndi (L2) bundle ships no lbr/oat/spm/csv, so every ancillary is optional here
        self.qub = paths["qub"].get(self.basename, "")
        self.hdr = paths.get("hdr", {}).get(self.basename, "")
        self.xml = paths.get("xml", {}).get(self.basename, "")
        self.lbr = paths.get("lbr", {}).get(self.basename, "")
        self.oat = paths.get("oat", {}).get(self.basename, "")
        self.oath = paths.get("oath", {}).get(self.basename, "")
        self.spm = paths.get("spm", {}).get(self.basename, "")
        if self.level == 1:
            self.csv = paths.get("csv", {}).get(self.basename, "")
            self.xml_csv = paths.get("xml-csv", {}).get(self.basename, "")

        # Store metadata from the qub file
        self.metadata = self._extract_metadata()

        # Read image
        self.img = xr.open_dataarray(self.qub, engine="rasterio")
        # Band subsets are not 1-indexed; without their true numbers every sel(band=N) is off
        self.img = self.img.assign_coords(band=_band_numbers(self.img))
        self.shape = self.img.shape
        self.nband, self.ny, self.nx = self.shape

        # Wavelength [nm], indexed by band number so subsets get their own wls, not the first N
        self.img = self.img.assign_coords(wl=("band", utils.get_wls()[self.img.band.values - 1]))

        # Chunk with dask if needed
        chunk = _chunks(self.nband, self.ny, self.nx, chunk)
        if chunk:
            self.img = self.img.chunk(chunk)

    def _extract_metadata(self):
        """Extract relevant metadata from the given qub file."""
        img = pdr.open(self.qub)
        product_meta = img.metaget("isda:Product_Parameters")
        metadata = {k[5:]: v for k, v in product_meta.items()}
        extra_keys = [
            "logical_identifier",
            "modification_date",
            "start_date_time",
            "stop_date_time",
            "file_size",
            "md5_checksum",
        ]
        for key in extra_keys:
            metadata[key] = img.metaget(key)
        return metadata

    def metaget(self, key):
        """Search metadata for key."""
        return pdr.open(self.qub).metaget(key)

    def checksum(self):
        """Verify all data was downloaded correctly."""
        utils.checksum(self.qub.as_posix())
        if self.level == 1:
            utils.checksum(self.csv.as_posix())

    def mask_bad_bands(self):
        """Mask invalid bands (OSF, bad bands)."""
        self.img = self.img.where(~self.img.band.isin((*utils.OSF, *utils.INVALID)))

    def save(self, fout, sub_rows=None, snr_sidecar=True, empirical_sidecar=True, full_bands=True):
        """
        Stream the image cube to disk with bounded memory.

        ENVI (.img, default): float32 BIL, streamed sub_rows scanlines at a time. The dask
        threaded scheduler computes each block across all cores; peak memory stays ~one block
        regardless of image or machine size. GeoTIFF (.tif): float32 BigTIFF, windowed blocks.

        When the cube carries an empirical `snr` coordinate (from calibrate_to_rad with attach_snr),
        a float32 sidecar `<basename>_snr<ext>` is written alongside it in the same format (ENVI
        coords don't survive the BIL write), unless snr_sidecar is False.

        When the cube carries `empirical_dark`/`empirical_flat`/`empirical_smile` coordinates (from
        calibrate_to_rad with attach_empirical_frames), they're written to a single
        `<basename>_empirical.npz` (keys dark/flat/smile/band/x) beside fout, unless
        empirical_sidecar is False -- these are (band, x) diagnostic frames, not a raster, so they
        skip the cube writer entirely.

        Parameters
        ----------
        fout : str or Path
            Output path. Extension selects the format (.tif -> GeoTIFF, else ENVI).
        sub_rows : int, optional
            Scanlines computed/written per step (bounds peak memory). Default None: the cube's own
            dask y-chunk, so a write block and a compute block are the same rows.
        snr_sidecar : bool
            Write the broadband `snr` field (if present) to a float32 sidecar next to fout.
        empirical_sidecar : bool
            Write the empirical dark/flat/smile frames (if present) to an `.npz` sidecar next to fout.
        full_bands : bool
            Reindex the cube onto the full 1..256 IIRS band range before writing, NaN-filling any
            band a subset dropped (e.g. `output_bands`, OSF/invalid), so band position always
            matches the real IIRS band number and wavelength. Default True.

        Returns
        -------
        str : the path written.
        """
        fout = str(fout)
        if self.level in (1, 2):
            self.img.attrs.setdefault("IIRS_PRODUCT", "L1" if self.level == 1 else "L2")
            self.img.attrs.setdefault("IIRS_FORMAT_VERSION", "1")
            self.img.attrs.setdefault("IIRS_SID", self.basename)
        if snr_sidecar and "snr" in self.img.coords:
            self._save_snr_sidecar(fout, sub_rows)
        if empirical_sidecar and "empirical_flat" in self.img.coords:
            self._save_empirical_sidecar(fout)
        img = _expand_to_full_bands(self.img) if full_bands else self.img
        return self._write(img, fout, sub_rows)

    def _write(self, da, fout, row_block=None, compress="ZSTD", predictor=None):
        """Format dispatch for any camera-space product: `.tif` -> GeoTIFF, else ENVI BIL."""
        return _write_camera_product(da, fout, row_block, compress=compress, predictor=predictor)

    def _save_snr_sidecar(self, fout, row_block=None):
        """Write the (y, x) empirical broadband SNR field beside fout, in fout's own format."""
        snr = self.img.snr.reset_coords(drop=True).expand_dims(band=[1]).astype("float32")
        snr.attrs = {"name": "SNR", "units": ""}
        f = Path(fout)
        return self._write(snr, f.with_name(f.stem + "_snr" + f.suffix), row_block)

    def _save_empirical_sidecar(self, fout):
        """Write the (band, x) empirical dark/flat/smile frames beside fout as one .npz."""
        f = Path(fout)
        fnpz = f.with_name(f.stem + "_empirical.npz")
        np.savez_compressed(
            fnpz,
            dark=self.img.empirical_dark.values.astype("float32"),
            flat=self.img.empirical_flat.values.astype("float32"),
            smile=self.img.empirical_smile.values.astype("float32"),
            band=self.img.band.values,
            x=self.img.x.values,
        )
        return str(fnpz)

    @abstractmethod
    def plot(self, band=12, yrange=(None, None), xrange=(None, None), **kwargs):
        """
        Plot image at band and x, y indices. Lowers resolution along y if large.
        """
        data = self.img.sel(band=band, y=slice(*yrange), x=slice(*xrange))
        # Defaults
        size = kwargs.pop("size", 5)
        vmin = kwargs.pop("vmin", 0)
        title = kwargs.pop("title", f"{self.basename}")
        cmap = kwargs.pop("cmap", "gray")
        cbarlabel = kwargs.pop("cbarlabel", "")
        if "ax" not in kwargs:
            kwargs["size"] = size  # Supply size only if ax not specified

        # Coarsen data for quicker plot
        if len(data.y) > 2000:
            data = data.sel(y=slice(None, None, len(data.y) // 1000))

        # Xarray plots with lat increasing up, but when it's raw DN (y coords) origin should be upper left
        yincrease = "y" in kwargs and kwargs["y"].lower() in ["lat", "latitude"]
        # Plot
        p = data.plot(vmin=vmin, cmap=cmap, yincrease=yincrease, **kwargs)
        ax = p.axes
        ax.set_title(title)
        ax.set_aspect("equal")

        if kwargs.get("add_colorbar", True) and cbarlabel:
            p.colorbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        return p, ax

    @abstractmethod
    def plot_spectra(self, bands=(None, None), yrange=(None, None), xrange=(None, None), **kwargs):
        """
        Plot image at band and x, y indices. Lowers resolution along y if large.
        """
        data = self.img.sel(band=slice(*bands), y=slice(*yrange), x=slice(*xrange))
        ax = utils.plot_spectra_with_sigma(data, **kwargs)
        ylabel = self.img.attrs.get("name", "")
        ylabel += f" [${self.img.attrs.get('units', '')}$]" if self.img.attrs.get("units", "") else ""
        ax.set_ylabel(ylabel)
        return ax

    def detect_stripes(self, sigma_threshold):
        """Detect stripes in image. See utils.detect_stripes()."""
        return utils.detect_stripes(self.img, sigma_threshold)

    def apply_smoothing(self, **kwargs):
        return utils.apply_smoothing(**kwargs)


class L0(IIRSData):
    """Class for reading and handling L0 IIRS data (digital numbers)."""

    def __init__(self, basename, directory=".", extent=(None, None, None, None), chunk=True):
        """
        Initialize the IIRS L0 data class.

        Parameters
        ----------
        basename : str
            Basename of the image to read (e.g. 20201214T0844306700).
        directory : str
            Path to the directory containing IIRS data files.
        extent : tuple
            Extent in (minx, maxx, miny, maxy) format.
        chunk : bool or dict
            Chunk image automatically (default: True). Or supply dict of x,y,band chunk sizes (see dask).
        """
        super().__init__(basename, directory, extent, chunk, level=0)
        self.img = self.img.sel(y=slice(*self.extent[-2:]), x=slice(*self.extent[:2]))
        try:
            self.bounds = self.img.rio.bounds()
        except NoDataInBounds as e:
            raise ValueError(f"No data found within extent {self.extent} for {self.basename}.") from e
        self.extent = tuple(int(e) for e in bounds2extent(self.bounds))
        self.img.attrs["name"] = "Digital Number"
        self.img.attrs["units"] = ""

    def plot(self, band=12, yrange=(None, None), xrange=(None, None), flip_xy=False, **kwargs):
        """Plot image at band and x, y indices if supplied."""
        if "cbarlabel" not in kwargs:
            kwargs["cbarlabel"] = "Digital Number"
        _, ax = super().plot(band, yrange, xrange, **kwargs)

        # Flip image if flip_xy (can plot a descending orbit with north up)
        if flip_xy:
            ax.yaxis.set_inverted(True)
            ax.xaxis.set_inverted(True)
        return ax

    def plot_spectra(self, bands=(None, None), yrange=(None, None), xrange=(None, None), **kwargs):
        return super().plot_spectra(bands, yrange, xrange, **kwargs)

    def calibrate_to_rad(
        self,
        config="iirspy",
        *,
        empirical_kws=None,
        calib_dir=utils.DCALIB,
        output_bands=None,
        frame_offsets=None,
        **overrides,
    ):
        """
        IIRS L0 digital number -> L1 radiance [W/m^2/sr/um], with method choices from `config`.

          L1_Radiance [mW/cm^2/sr/um] = Gain * L0_DN + Offset   (x10 to [W/m^2/sr/um])

        Raw counts are already dark subtracted; the gain/offset LUT is picked by the scene's gain
        and exposure settings. The LUT is per detector element (band, x), but its cross-track
        structure is poorly correlated with the on-orbit response and injects striping/speckle.
        With `empirical`, a per-scene residual dark + sensor flat + band-relative smile (see
        iirspy.empirical) replaces it and the LUT gives only the per-band absolute scale; in-scene
        shadow, when present, anchors the zero point in place of the LUT offset:

          scene with dark rows   : rad = 10 * gain_med * (DN - dark_resid) / (flat * smile)
          scene without dark rows: rad = 10 * (gain_med * DN / (flat * smile) + offset_med)

        With `fix_frames`, each 4-band group is moved onto true frames by the `frame_offsets` table
        and its lost frames nulled (see iirspy.frames); `fill_frames` rebuilds them. Then OSF/invalid
        bands, `exclude_wl`, bad pixels and saturated pixels are nulled, and
        `interp_bands`/`interp_spatial` fill gaps.

        Parameters
        ----------
        config : str or L1Config
            Preset name or config (see `iirspy.config.L1Config`); `overrides` replace single fields.
        empirical_kws : dict or None
            Extra keyword arguments for empirical.empirical_frames, e.g. dark_yrange/flat_yrange
            (positional row ranges into this cube) or apply_smile=True.
        calib_dir : path
            Calibration LUT directory.
        output_bands : list of int or None
            Band numbers to mask/interpolate/return. The empirical frames always see every band
            `self.img` holds; only narrow past what `interp_bands`' max_gap=6 can reach.
        frame_offsets : path or None
            Frame table from `iirspy.frames.build_table`, for `fix_frames`. None builds it from this
            cube (all 256 bands, ~1 min) and caches it beside the .qub; a band-subset cube can't be
            scanned, so its rows stay in raw order, with a warning.

        Returns
        -------
        xarray.DataArray
            Radiance [W/m^2/sr/um], with the config as json in `attrs["l1_config"]`.
        """
        cfg = L1Config.resolve(config, **overrides)
        empirical, interp_bands, interp_spatial = cfg.empirical, cfg.interp_bands, cfg.interp_spatial
        attach_snr, attach_empirical_frames = cfg.attach_snr, cfg.attach_empirical_frames
        bad_pixel_mask, exclude_wl = cfg.bad_pixel_mask, cfg.exclude_wl
        gain, offset = utils.get_gain_offset(self.qub, calib_dir=calib_dir)

        if empirical:
            # Empirical correction: LUT gives per-band absolute scale, flat/smile handle cross-track
            ref_flat = utils.load_reference_flat(self.qub, calib_dir=calib_dir)
            dark, flat, smile, snr, emp_notes = empirical_frames(self.img, ref_flat, **(empirical_kws or {}))
            gain_med, offset_med = gain.median("x"), offset.median("x")
            fs = (flat * smile).clip(min=FLAT_SMILE_MIN)  # avoid divide by 0
            if snr is not None:
                # Shadow scene: the in-scene zero anchors the zero point, so drop the LUT offset
                rad = 10 * gain_med * (self.img - dark) / fs
                if attach_snr:
                    rad = rad.assign_coords(snr=(("y", "x"), snr.values))
            else:
                # No zero reference: the lab offset is the only zero-point info, so keep it
                rad = 10 * (gain_med * self.img / fs + offset_med)
            rad = _attach_empirical_frames(rad, dark, flat, smile, attach_empirical_frames)
        else:
            # Apply per-element gain and offset to convert DN -> Radiance
            rad = 10 * (self.img * gain + offset)  # [mW/cm^2/sr/μm] -> [W/m^2/sr/um]

        # Narrow to the requested output bands before the mask/interpolate stage below -- the
        # empirical frames above already saw every band `self.img` holds, so this can't affect them.
        rad = _narrow_bands(rad, output_bands)

        # Re-time each 4-band group onto true frames; its lost frames go NaN until `fill_frames`
        rad, lost = _retime(rad, cfg.fix_frames, frame_offsets, self.qub)

        # Drop OSF and invalid bands, plus any caller-supplied wavelength ranges. interp if specified
        rad = rad.where(~(rad.band.isin((*utils.OSF, *utils.INVALID)) | _in_wl_ranges(rad.wl, exclude_wl)))

        # Drop known bad detector elements (x, bands)
        if bad_pixel_mask is True:  # Load default bad pixel mask
            bad_pixel_mask = utils.load_bad_pixel_mask()
        if bad_pixel_mask is not False:  # Apply user-supplied mask
            rad = rad.where(~bad_pixel_mask)

        # Drop saturated pixels
        rad = rad.where(rad < utils.get_saturation_radiance(self.qub, calib_dir))

        # Rebuild lost frames from each pixel's own spectrum before any generic band interpolation
        rad = frames.fill(rad, lost) if lost is not None and cfg.fill_frames else rad

        # Interpolate across bands
        if interp_bands is not None:
            rad = rad.interpolate_na("band", max_gap=6, keep_attrs=True, method=interp_bands)

        # Fill single-column all-band NaN gaps across x (max_gap=2 in 0.5-based x coords spans
        # exactly one missing column; wider seams and edge columns have no bounded gap to fill)
        if interp_spatial:
            rad = rad.interpolate_na("x", max_gap=2, keep_attrs=True, method="linear")

        # Format and output Radiance DataArray (float32: gain/offset LUTs are float32, keep graph float32)
        out = rad.astype("float32")
        out.name = "Radiance [W/m^2/sr/µm]"
        out.attrs["name"] = "Radiance"
        out.attrs["units"] = "W/m^2/sr/um"
        out.attrs["calibration_source"] = "empirical" if empirical else "user"
        try:
            out.attrs["iirspy_version"] = metadata.version("iirspy")
        except metadata.PackageNotFoundError:  # not pip-installed (source tree)
            out.attrs["iirspy_version"] = "0+source"
        out.attrs["l1_config"] = json.dumps(cfg.to_attrs())
        if empirical:
            out.attrs["empirical_notes"] = json.dumps(emp_notes)  # provenance: how the product was made

        return out

    def calibrate(self, config="iirspy", **kwargs):
        """Calibrate to an L1 object (see `calibrate_to_rad`), with ISSDC L1 metadata if it exists."""
        img = self.calibrate_to_rad(config, **kwargs)
        try:
            out = L1(self.basename, str(self.directory), self.extent)
            out.img = img
        except FileNotFoundError:
            warnings.warn(
                f"L1 files not found. Expected prefix: ch2_iir_nci_{self.basename} "
                f"at {self.directory}. Continuing without metadata.",
                UserWarning,
                stacklevel=2,
            )
            out = L1.from_xarray(img, self.basename, str(self.directory))
        return out


class L1(IIRSData):
    """Class for reading, handling, and processing L1 IIRS data to L2 reflectance."""

    @classmethod
    def from_xarray(
        cls,
        data,
        basename,
        directory=".",
        latlonextent=None,
    ):
        """Create L1 instance from user-calibrated xarray.DataArray.

        Opportunistically loads L1 metadata if available.

        Parameters
        ----------
        data : xarray.DataArray
            Radiance data with wavelength coordinate.
        basename : str
            Image basename (e.g. 20201214T0844306700).
        directory : str
            Directory to search for L1 ancillary files.
        latlonextent : tuple, optional
            Approximate image corners (minlon, maxlon, minlat, maxlat).

        Returns
        -------
        L1
            Instance with attached metadata where available.
        """
        instance = cls.__new__(cls)
        instance.basename = utils.iirsbasename(basename)
        instance.directory = Path(directory).expanduser().absolute()
        instance.level = 1
        instance.img = data
        instance.shape = data.shape
        instance.nband, instance.ny, instance.nx = instance.shape
        instance.extent = (int(data.x.min()), int(data.x.max()), int(data.y.min()), int(data.y.max()))
        instance.bounds = (instance.extent[0], instance.extent[2], instance.extent[1], instance.extent[3])

        instance.img.attrs["name"] = "Radiance"
        instance.img.attrs["units"] = "W/m^2/sr/um"
        if "calibration_source" not in data.attrs:
            instance.img.attrs["calibration_source"] = "user"

        metadata = _try_load_next_level_metadata(instance.basename, 1, directory, instance.extent)
        instance.csv = metadata["geometry"]
        instance.geomdf = metadata["geometry_df"]
        instance.spm = metadata["spm"]
        instance.metadata = {}

        if latlonextent and instance.geomdf is not None:
            instance.geomdf, xy_extent = utils.parse_geom(instance.csv, latlonextent, center=False)
            instance.extent = xy_extent
            lon, lat = utils.geom2latlon_coords(instance.geomdf, xy_extent, instance.nx, instance.ny)
            instance.img = instance.img.assign_coords({"lon": ("x", lon), "lat": ("y", lat)})

        if instance.spm is not None:
            # Solar incidence/azimuth need the scene's per-line clock times, which come from the
            # L1 qub/xml label metadata (start time, exposure, line count, orbit direction). Locate
            # the label for this basename; the tiny .xml suffices if the bulky .qub was cleaned up.
            paths = utils.get_iirs_paths(
                instance.directory, exts=("qub", "xml"), level=1, basenames=[instance.basename]
            )
            fimg = paths.get("qub", {}).get(instance.basename) or paths.get("xml", {}).get(instance.basename)
            if fimg is not None:
                instance.qub = fimg
                # yrange as contiguous absolute line indices matching this cube's y, so the
                # incidence array lines up exactly (extent min/max int-truncation can be off by 1).
                y0 = round(float(instance.img.y.min()) - 0.5)
                yr = (y0, y0 + instance.img.sizes["y"])
                inc, iaz = utils.get_iirs_inc_az(fimg, instance.spm, yr)
                instance.img = instance.img.assign_coords({
                    "solar_inc": ("y", inc),
                    "solar_az": ("y", iaz),
                })

        return instance

    @classmethod
    def from_file(cls, path, basename, directory=".", chunk=True):
        """Load a cropped empirical-L1 GeoTIFF or ENVI cube (native [1000 mW/cm^2/sr/um]) as an L1.

        Restores the band numbers, wavelengths and absolute line/sample offset, rescales radiance to
        physical [W/m^2/sr/um] (utils.RAD_NATIVE_SCALE) and attaches geometry from the ancillary
        spm/csv in `directory`, so the result feeds L2 identically to a downloaded L1.

        Parameters
        ----------
        path : str or Path
            Cropped L1 raster written by the polar pipeline (radiance in [1000 mW/cm^2/sr/um]).
        basename : str
            Image basename (e.g. 20210723T1445053074), used to find spm/geometry in `directory`.
        directory : str
            Directory holding the IIRS bundle (nci ancillary: spm + geometry csv).
        chunk : bool or dict
            Chunk image automatically (default: True). Or supply dict of x,y,band chunk sizes (see dask).
        """
        da = _apply_envi_start(xr.open_dataarray(path, engine="rasterio").sortby("y").sortby("x"))
        nband, ny, nx = da.shape
        chunk = _chunks(nband, ny, nx, chunk)
        if chunk:
            da = da.chunk(chunk)
        bands = np.asarray(_band_numbers(da), dtype=int)
        da = da.assign_coords(band=bands, wl=("band", utils.get_wls()[bands - 1]))
        da = da * utils.RAD_NATIVE_SCALE  # [1000 mW/cm^2/sr/um] -> [W/m^2/sr/um]
        da.attrs["name"] = "Radiance"
        da.attrs["units"] = "W/m^2/sr/um"
        da.attrs["calibration_source"] = "empirical"

        instance = cls.from_xarray(da, basename, directory)
        # from_xarray only attaches lat/lon when given a latlonextent; derive them for the actual
        # absolute crop extent instead (mirrors L1.__init__), so polar analysis has lat/lon coords.
        if instance.geomdf is not None:
            lon, lat = utils.geom2latlon_coords(instance.geomdf, instance.extent, instance.nx, instance.ny)
            instance.img = instance.img.assign_coords({"lon": ("x", lon), "lat": ("y", lat)})
        return instance

    def __init__(
        self,
        basename,
        directory=".",
        xyextent=(None, None, None, None),
        lonlatextent=(None, None, None, None),
        chunk=True,
    ):
        """
        Initialize the IIRS L1 data class.

        Parameters
        ----------
        basename : str
            Basename of the image to read (e.g. 20201214T0844306700).
        directory : str
            Path to the directory containing IIRS data files.
        xyextent : tuple
            Extent in (minx, maxx, miny, maxy) format. Only one of xyextent and lonlatextent can be given.
        lonlatextent : tuple
            Extent in (minlon, maxlon, minlat, maxlat) format.
        chunk : bool or dict
            Chunk image automatically (default: True). Or supply dict of x,y,band chunk sizes (see dask).
        """
        if any(e is not None for e in xyextent) and any(e is not None for e in lonlatextent):
            raise ValueError("Only one of lonlatextent and xyextent can be given.")
        super().__init__(basename, directory, xyextent, chunk, level=1)

        # Parse geometry, store gcps and extent in x, y
        self.geomdf, xy_extent = utils.parse_geom(self.csv, lonlatextent, xyextent, center=False)
        # replace any None in extent with values from xy_extent
        self.extent = tuple(xy if ex is None else ex for ex, xy in zip(self.extent, xy_extent, strict=False))
        # ...but the geometry csv spans the whole strip and the cube may be a crop: with no extent
        # given, take the line range from the image (`_apply_envi_start` put y in strip numbering).
        if all(e is None for e in xyextent) and all(e is None for e in lonlatextent):
            # floor/ceil, not int(): rasterio hands back pixel-CENTRE coords (0.5 .. n-0.5), so
            # truncating both ends drops a row and a column off the crop.
            self.extent = (
                int(np.floor(self.img.x.values[0])),
                int(np.ceil(self.img.x.values[-1])),
                int(np.floor(self.img.y.values[0])),
                int(np.ceil(self.img.y.values[-1])),
            )
        # x extent is an inclusive last-pixel index (249 on a 250-wide cube) against pixel-centre
        # coords, so +1 keeps that last column; y stays half-open.
        self.img = self.img.sel(y=slice(*self.extent[-2:]), x=slice(self.extent[0], self.extent[1] + 1))
        try:
            self.bounds = self.img.rio.bounds()
        except NoDataInBounds as e:
            raise ValueError(f"No data found within extent {self.extent} for {self.basename}.") from e

        self.shape = self.img.shape  # [nband, ny, nx]

        # Fix units
        self.img *= utils.RAD_NATIVE_SCALE  # [1000 mW/cm^2/sr/um] -> [W/m^2/sr/um]
        self.img.attrs["name"] = "Radiance"
        self.img.attrs["units"] = "W/m^2/sr/um"
        self.img.attrs["calibration_source"] = "issdc"

        # Assign lat, lon coordinates
        # `self.extent` (the cube's crop), not `xy_extent` (the whole strip).
        lon, lat = utils.geom2latlon_coords(self.geomdf, self.extent, self.shape[2], self.shape[1])

        # Get solar incidence for each line of image
        inc, iaz = utils.get_iirs_inc_az(self.qub, self.spm, self.extent[2:])

        # Assign new coords
        self.img = self.img.assign_coords({
            "lon": ("x", lon),
            "lat": ("y", lat),
            "solar_inc": ("y", inc),
            "solar_az": ("y", iaz),
        })

    def plot(self, band=12, yrange=(None, None), xrange=(None, None), **kwargs):
        """Plot image at band and x, y indices if supplied."""
        if "cbarlabel" not in kwargs:
            kwargs["cbarlabel"] = "Radiance [$W/m^2/sr/um$]"
        x, y = ("lon", "lat") if "lon" in self.img.coords and "lat" in self.img.coords else ("x", "y")
        _, ax = super().plot(band, yrange, xrange, x=x, y=y, **kwargs)
        return ax

    def plot_spectra(self, bands=(None, None), yrange=(None, None), xrange=(None, None), **kwargs):
        return super().plot_spectra(bands, yrange, xrange, **kwargs)

    def calibrate(self, config="iirspy", *, inc=None, slope_az=None, sun_az_offset=0.0, **overrides):
        """Calibrate to an L2 reflectance object (requires the SPM file for solar angles).

        `config` is an `L2Config` or a preset name (see `iirspy.config.L2Config`); `overrides`
        replace individual fields, e.g. `calibrate("issdc", polish=None)`. The keywords are
        per-scene inputs: `inc` (solar incidence [deg], per line or per pixel, default the spm's),
        `slope_az` (camera-space slope/aspect[/lit], see `photometry.load_slope_az`; make one from
        a DEM with `photometry.slope_az_from_dem`) and `sun_az_offset` (deg added to the spm
        azimuth to reach slope_az's grid-north frame).
        """
        if self.spm is None:
            raise FileNotFoundError(
                f"SPM file required for reflectance calibration. Expected: "
                f"{self.directory}/miscellaneous/calibrated/ch2_iir_nci_{self.basename}_spm.dat"
            )
        cfg = L2Config.resolve(config, **overrides)
        return L2._from_l1(self, cfg, inc=inc, slope_az=slope_az, sun_az_offset=sun_az_offset)


def _sun_distance(cfg, qub):
    """Sun distance [AU] for `cfg.sdist`; "spice" without a label is 1 AU."""
    if cfg.sdist == "none":
        return 1.0
    if cfg.sdist == "spice":
        return utils.get_solar_distance(qub) if qub else 1.0
    return float(cfg.sdist)


def _solar_flux(cfg, sdist, qub):
    """Per-band solar flux [W/m^2/sr/um] at `sdist` for `cfg.flux`."""
    if isinstance(cfg.flux, tuple):
        n = len(cfg.flux)
        return cfg.flux[0] if n == 1 else xr.DataArray(np.asarray(cfg.flux), coords={"band": np.arange(1, n + 1)})
    if cfg.flux == "noaa_cdr_mean":
        return utils.get_noaa_flux(sdist)
    if cfg.flux == "noaa_cdr_monthly":
        return utils.get_noaa_flux(sdist, pd.Timestamp(pdr.open(qub).metaget("start_date_time")).strftime("%Y-%m"))
    return utils.get_solar_flux(sdist, utils.FSOLAR if cfg.flux == "pradan" else cfg.flux)


class L2(IIRSData):
    """Class for IIRS L2 reflectance data."""

    # Per-scene attrs `_compute_reflectance` records beside `l2_config`
    DIAGNOSTICS = (
        "solar_distance_au",
        "inc_min_deg",
        "inc_max_deg",
        "inc_mean_deg",
        "cos_inc_floor_frac",
        "cos_view_floor_frac",
        "slope_az_used",
        "sun_az_offset",
    )

    @classmethod
    def from_xarray(
        cls,
        data,
        basename,
        directory=".",
        latlonextent=None,
    ):
        """Create L2 instance from user-calibrated xarray.DataArray.

        Opportunistically loads L2 metadata if available.

        Parameters
        ----------
        data : xarray.DataArray
            Reflectance data with wavelength coordinate.
        basename : str
            Image basename (e.g. 20201214T0844306700).
        directory : str
            Directory to search for L2 ancillary files.
        latlonextent : tuple, optional
            Approximate image corners (minlon, maxlon, minlat, maxlat).

        Returns
        -------
        L2
            Instance with attached metadata where available.
        """
        instance = cls.__new__(cls)
        instance.basename = utils.iirsbasename(basename)
        instance.directory = Path(directory).expanduser().absolute()
        instance.level = 2
        instance.img = data
        instance.shape = data.shape
        instance.nband, instance.ny, instance.nx = instance.shape
        instance.extent = (int(data.x.min()), int(data.x.max()), int(data.y.min()), int(data.y.max()))
        instance.bounds = (instance.extent[0], instance.extent[2], instance.extent[1], instance.extent[3])
        instance.metadata = {}

        instance.img.attrs["name"] = "Reflectance"
        instance.img.attrs["units"] = ""
        if "calibration_source" not in data.attrs:
            instance.img.attrs["calibration_source"] = "user"

        metadata = _try_load_next_level_metadata(instance.basename, 2, directory, instance.extent)
        instance.csv = metadata["geometry"]
        instance.geomdf = metadata["geometry_df"]
        instance.spm = metadata["spm"]

        if latlonextent and instance.geomdf is not None:
            instance.geomdf, xy_extent = utils.parse_geom(instance.csv, latlonextent, center=False)
            instance.extent = xy_extent
            lon, lat = utils.geom2latlon_coords(instance.geomdf, xy_extent, instance.nx, instance.ny)
            instance.img = instance.img.assign_coords({"lon": ("x", lon), "lat": ("y", lat)})

        return instance

    @classmethod
    def _from_l1(cls, l1_instance, cfg, inc=None, slope_az=None, sun_az_offset=0.0):
        """Internal method to create L2 from L1 instance."""
        instance = cls.__new__(cls)
        instance.basename = l1_instance.basename
        instance.directory = l1_instance.directory
        instance.level = 2
        instance.extent = l1_instance.extent
        instance.bounds = l1_instance.bounds
        instance.shape = l1_instance.shape
        instance.nband, instance.ny, instance.nx = instance.shape
        instance.csv = l1_instance.csv
        instance.geomdf = l1_instance.geomdf
        instance.spm = l1_instance.spm
        instance.metadata = l1_instance.metadata.copy()

        instance.img = instance._compute_reflectance(
            l1_instance.img,
            l1_instance.qub if hasattr(l1_instance, "qub") else None,
            cfg,
            inc=inc,
            slope_az=slope_az,
            sun_az_offset=sun_az_offset,
        )
        instance.img.attrs["calibration_source"] = "user"

        return instance

    def _compute_reflectance(self, rad, qub_path, cfg, inc=None, slope_az=None, sun_az_offset=0.0):
        """I/F reflectance from radiance, with each step chosen by `cfg` (see `iirspy.config.L2Config`).

        Steps, in order: solar flux at the scene's distance, thermal removal, clip, polish,
        photometric normalization, Besse phase, spectral smoothing. Clip and polish are folded into
        one division with the flux and photometric terms; all divisors are positive.

        Parameters
        ----------
        rad : xarray.DataArray
            (band, y, x) radiance [W/m^2/sr/um] with `wl` and per-line `solar_inc`/`solar_az` coords.
        qub_path : str or None
            Label for the scene's epoch (solar distance, monthly flux). None means 1 AU.
        cfg : L2Config
            Method choices.
        inc : float or xarray.DataArray, optional
            Solar incidence [deg], per line or per pixel. Default `rad.solar_inc`.
        slope_az : str, Path, xarray object or tuple, optional
            Camera-space slope/aspect[/lit] on this cube's grid (`photometry.load_slope_az`). Used
            only when `cfg.use_slope_az`.
        sun_az_offset : float
            Degrees added to the spm solar azimuth (local north) to reach slope_az's grid north.

        Returns
        -------
        xarray.DataArray
            Reflectance, with `cfg` as json in `attrs["l2_config"]`, the per-scene `L2.DIAGNOSTICS` in
            attrs, and the thermal temperature [K] as coord `T` when removed.
        """
        slope_az = slope_az if cfg.use_slope_az else None
        cos_inc = np.cos(np.radians(inc)) if inc is not None else np.cos(np.radians(rad.solar_inc))
        cos_inc = cos_inc * xr.ones_like(rad.isel(x=0, band=0, drop=True))

        sdist = _sun_distance(cfg, qub_path)
        solar_flux = _solar_flux(cfg, sdist, qub_path)

        trad, temp = (0.0, None)
        if cfg.thermal == "isothermal":
            trad, temp = utils.thermal_radiance(rad, cfg.thermal_eps, cfg.thermal_bands)

        sun_az = (rad.solar_az if "solar_az" in rad.coords else 0.0) + sun_az_offset
        photfn, g, floors = photometry.photometric_factor(
            cos_inc,
            rad.solar_inc,
            sun_az,
            slope_az,
            like=rad.isel(band=0, drop=True),
            photom=cfg.photom,
            cos_min=cfg.cos_min,
            min_lit=cfg.min_lit,
        )

        num = rad - trad
        if cfg.clip_negative:
            num = num.clip(min=0)
        den = photfn * solar_flux
        if cfg.polish is not None:
            den = den * utils.spectral_polish(rad.band, POLISH.get(cfg.polish, cfg.polish))
        refl = num / den
        refl = photometry.besse_rescale(refl, cfg.photom, g, rad.wl) if cfg.phase else refl
        refl = utils.moving_average(refl, cfg.smoothing, cfg.smoothing_edges)
        if "wl" not in refl.coords:
            refl = refl.assign_coords(wl=rad.wl)
        if temp is not None:
            refl = refl.assign_coords(T=temp)
        refl.name = "Reflectance"
        refl.attrs["name"] = "Reflectance"
        refl.attrs["units"] = ""

        inc_deg = np.degrees(np.arccos(np.clip(cos_inc, -1.0, 1.0)))
        refl.attrs["l2_config"] = json.dumps(cfg.to_attrs())
        refl.attrs.update(
            **floors,
            sun_az_offset=sun_az_offset,
            slope_az_used=slope_az is not None,
            solar_distance_au=sdist,
            inc_min_deg=float(inc_deg.min()),
            inc_max_deg=float(inc_deg.max()),
            inc_mean_deg=float(inc_deg.mean()),
        )
        return refl

    def __init__(
        self,
        basename,
        directory=".",
        xyextent=(None, None, None, None),
        lonlatextent=(None, None, None, None),
        chunk=True,
    ):
        """Initialize L2 from ISSDC files (if available)."""
        if any(e is not None for e in xyextent) and any(e is not None for e in lonlatextent):
            raise ValueError("Only one of lonlatextent and xyextent can be given.")
        super().__init__(basename, directory, xyextent, chunk, level=2)

        metadata = _try_load_next_level_metadata(self.basename, 2, directory, self.extent)
        self.csv = metadata["geometry"]
        self.geomdf = metadata["geometry_df"]
        self.spm = metadata["spm"]

        if self.geomdf is not None:
            self.geomdf, xy_extent = utils.parse_geom(self.csv, lonlatextent, center=False)
            self.extent = tuple(xy if ex is None else ex for ex, xy in zip(self.extent, xy_extent, strict=False))
        elif any(e is not None for e in lonlatextent):
            raise FileNotFoundError(f"no geometry csv beside {self.qub} to crop by lon/lat")

        # x extent is an inclusive last-pixel index (249 on a 250-wide cube) against pixel-centre
        # coords, so +1 keeps that last column; y stays half-open.
        self.img = self.img.sel(y=slice(*self.extent[-2:]), x=slice(self.extent[0], self.extent[1] + 1))
        try:
            self.bounds = self.img.rio.bounds()
        except NoDataInBounds as e:
            raise ValueError(f"No data found within extent {self.extent} for {self.basename}.") from e

        self.shape = self.img.shape
        self.img = self.img.where(self.img != 0)  # ISSDC ndi fill is exact 0, not a declared nodata

        if self.geomdf is not None:
            lon, lat = utils.geom2latlon_coords(self.geomdf, self.extent, self.shape[2], self.shape[1])
            self.img = self.img.assign_coords({"lon": ("x", lon), "lat": ("y", lat)})

        if self.spm is not None:
            inc, iaz = utils.get_iirs_inc_az(self.qub, self.spm, self.extent[2:])
            self.img = self.img.assign_coords({"solar_inc": ("y", inc), "solar_az": ("y", iaz)})

    def plot(self, band=12, yrange=(None, None), xrange=(None, None), north_up=True, **kwargs):
        """Plot reflectance image."""
        if "cbarlabel" not in kwargs:
            kwargs["cbarlabel"] = "Reflectance"
        x, y = ("lon", "lat") if "lon" in self.img.coords and "lat" in self.img.coords else ("x", "y")
        _, ax = super().plot(band, yrange, xrange, x=x, y=y, **kwargs)
        return ax

    def plot_spectra(self, bands=(None, None), yrange=(None, None), xrange=(None, None), **kwargs):
        return super().plot_spectra(bands, yrange, xrange, **kwargs)
