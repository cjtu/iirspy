"""`L0`, `L1` and `L2`: one IIRS scene at one processing level, wrapping its (band, y, x) cube and ancillary files.

    L0(sid, dir).calibrate() -> L1   # raw DN -> radiance (`iirspy.l1.calibrate`)
    L1.georef(method, ...)   -> Geometry
    L1.calibrate(geom=...)   -> L2   # radiance -> reflectance (`iirspy.l2.calibrate`)

Each class reads the ISSDC product (or a saved iirspy crop) lazily with dask; `crop` -> `save` -> `from_file` keeps
absolute y/x and band numbers, so a crop stays tied to its parent scene.
"""

import copy
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pdr
import rasterio
import xarray as xr
from rioxarray.exceptions import NoDataInBounds

from iirspy import instrument, io, l1, l2
from iirspy.georef import GEOREF, ancillary, georef
from iirspy.l1 import empirical, frames
from iirspy.l1.config import L1Config
from iirspy.l2 import polish
from iirspy.l2.config import L2Config

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
        paths = io.get_iirs_paths(directory, level=target_level, basenames=[basename])
        if target_level >= 1:
            csv_path = paths.get("csv", {}).get(basename)
            if csv_path and csv_path.exists():
                result["geometry"] = csv_path
                df, _ = ancillary.parse_geom(csv_path, xyextent=xyextent)
                result["geometry_df"] = df

            spm_path = paths.get("spm", {}).get(basename)
            if spm_path and spm_path.exists():
                result["spm"] = spm_path
    except (FileNotFoundError, KeyError, ValueError):
        pass

    return result


def _flip(da, flip: tuple[bool, bool]):
    """Reverse the pixel order of `da` along y and/or x, leaving the coordinate labels ascending.

    `flip` is `(flip_y, flip_x)`: True reverses that dimension's data, False leaves it alone.
    """
    flip_y, flip_x = flip
    dims = [d for d, do in (("y", flip_y), ("x", flip_x)) if do]
    out = da.isel({d: slice(None, None, -1) for d in dims})
    return out.assign_coords({d: da[d].values for d in dims})


def read_issdc_loc(fqub):
    """ISSDC's ndi `_d_loc_` beside `fqub` as a `backplanes.read_loc` dataset in the nci camera frame
    (y = Scan, x = Pixel, as the geometry csv and our GLTs index it), and the (flip_y, flip_x) that
    takes the ndi cube to the same frame, or None if there is no `_d_loc_`.

    ndi products are stored north-up/west-left, not in acquisition order: a descending pass comes
    column-mirrored, an ascending one row-reversed. The label's Refined upper-left corner is nci
    Scan 0/Pixel 0, so whichever loc corner sits on it is the origin.
    """
    from iirspy.georef.backplanes import read_loc

    floc = next(Path(fqub).parent.glob(f"{Path(fqub).name.split('_d_')[0]}_d_loc_*.img"), None)
    if floc is None:
        return None
    loc = read_loc(floc)
    ul = pdr.open(fqub).metaget("isda:Refined_Corner_Coordinates")
    lon0, lat0 = float(ul["isda:upper_left_longitude"]), float(ul["isda:upper_left_latitude"])
    lon, lat = loc.lon.values[:: -1 or None], loc.lat.values
    corners = [(i, j) for i in (0, -1) for j in (0, -1)]
    dlon = [((lon[c] - lon0 + 180) % 360 - 180) * np.cos(np.radians(lat0)) for c in corners]
    d = np.hypot(dlon, [lat[c] - lat0 for c in corners])
    flip_y, flip_x = corners[int(np.argmin(d))]
    flip = (bool(flip_y), bool(flip_x))
    return _flip(loc, flip), flip


def _loc_extent(loc, lonlatextent):
    """(minx, maxx, miny, maxy) of the rows whose pixels fall in `lonlatextent`, from an ndi bundle's
    `read_issdc_loc` (it has no geometry csv). Same convention as `ancillary.parse_geom`: x inclusive,
    y half-open."""
    lon, lat = (loc.lon.values + 180) % 360 - 180, loc.lat.values
    lon0, lon1, lat0, lat1 = (d if e is None else e for e, d in zip(lonlatextent, (-180, 180, -90, 90), strict=True))
    rows = np.flatnonzero(((lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)).any(axis=1))
    if not rows.size:
        raise ValueError(f"no pixels of the _d_loc_ inside {lonlatextent}")
    return (0, lon.shape[1] - 1, int(rows[0]), int(rows[-1]) + 1)


def _chunks(nband, ny, nx, chunk=True):
    """dask chunk dict for a (band, y, x) cube, or None to leave it unchunked.

    `chunk` is a dict to use as-is, or True to split along y at `io.CHUNKSIZE`. A cube already
    under one chunk's worth stays whole.
    """
    if not chunk:
        return None
    if isinstance(chunk, dict):
        return chunk
    if nband * ny * nx * 4 <= io.CHUNKSIZE:
        return None
    return {"band": nband, "y": int(io.CHUNKSIZE / (nband * nx * 4)), "x": nx}


def _expand_to_full_bands(da):
    """Reindex `da`'s band dim onto the full 1..256 IIRS range, NaN-filling any band not present.

    A cube narrowed to a band subset writes only its own bands contiguously, this restores the full
    1:1 mapping so the dropped/OSF/invalid bands become all-NaN, consistent with IIRS images.
    """
    all_bands = np.arange(1, len(instrument.get_wls()) + 1)
    if da.sizes["band"] == len(all_bands) and np.array_equal(da.band.values, all_bands):
        return da
    da = da.reindex(band=all_bands)
    return da.assign_coords(wl=("band", instrument.get_wls()))


class IIRSData(ABC):
    """Abstract base class for IIRS data products.

    Cropped images work the same at every level: `crop` -> `save` -> `from_file` round-trips values,
    absolute y/x and band numbers. With `fix_frames`, crop with a few rows of margin.
    """

    _NATIVE_SCALE = 1.0  # file units -> product units, applied by `from_file`
    _SOURCE: str | None = None  # `calibration_source` attr `from_file` stamps, if any
    bounds: tuple

    @classmethod
    @abstractmethod
    def from_xarray(cls, data, basename, directory="."): ...

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
        self.basename = io.iirsbasename(basename)
        self.directory = Path(directory).expanduser().absolute()
        self.extent = extent
        self.level = level

        # Get paths to relevant files
        paths = io.get_iirs_paths(self.directory, level=self.level, basenames=[self.basename])
        if "qub" not in paths:
            paths = io.unzip_iirs(self.directory, self.basename, self.level)
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
        self.img = self.img.assign_coords(band=io.band_numbers(self.img))
        self.shape = self.img.shape
        self.nband, self.ny, self.nx = self.shape

        # Wavelength [nm], indexed by band number so subsets get their own wls, not the first N
        self.img = self.img.assign_coords(wl=("band", instrument.get_wls()[self.img.band.values - 1]))

        # Chunk with dask if needed
        chunk = _chunks(self.nband, self.ny, self.nx, chunk)
        if chunk:
            self.img = self.img.chunk(chunk)

    @classmethod
    def from_file(cls, path, basename, directory=".", chunk=True):
        """Load a cropped GeoTIFF or ENVI cube written by `save` as this level.

        Restores band numbers, wavelengths and absolute line/sample offset, and rescales from the
        file's native units (`_NATIVE_SCALE`). Ancillary files are looked up in `directory`.

        Parameters
        ----------
        path : str or Path
            Cropped raster.
        basename : str
            Image basename (e.g. 20210723T1445053074), used to find ancillary files in `directory`.
        directory : str
            Directory holding the IIRS bundle.
        chunk : bool or dict
            Chunk image automatically (default: True). Or supply dict of x,y,band chunk sizes (see dask).
        """
        da = io.apply_envi_start(xr.open_dataarray(path, engine="rasterio").sortby("y").sortby("x"))
        nband, ny, nx = da.shape
        chunk = _chunks(nband, ny, nx, chunk)
        if chunk:
            da = da.chunk(chunk)
        bands = np.asarray(io.band_numbers(da), dtype=int)
        da = da.assign_coords(band=bands, wl=("band", instrument.get_wls()[bands - 1]))
        if cls._NATIVE_SCALE != 1.0:
            da = da * cls._NATIVE_SCALE
        if cls._SOURCE:
            da.attrs["calibration_source"] = cls._SOURCE
        return cls.from_xarray(da, basename, directory)

    def crop(self, y=None, x=None, bands=None):
        """New object cut to `y`/`x` = (lo, hi) inclusive in this cube's absolute y/x labels and to `bands` (IIRS band numbers).

        None leaves that axis whole; y/x-indexed coords (lat, lon, solar_inc, snr, ...) are carried along.
        """
        img = self.img if bands is None else self.img.sel(band=list(bands))
        img = img.sel(y=slice(*(y or (None, None))), x=slice(*(x or (None, None))))
        if 0 in img.shape:
            raise ValueError(f"crop(y={y}, x={x}, bands={bands}) is empty for {self.basename}")
        out = copy.copy(self)
        out.img = img
        out.shape = img.shape
        out.nband, out.ny, out.nx = out.shape
        out.extent = (int(img.x.min()), int(img.x.max()), int(img.y.min()), int(img.y.max()))
        out.bounds = (out.extent[0], out.extent[2], out.extent[1], out.extent[3])
        return out

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
        io.checksum(self.qub.as_posix())
        if self.level == 1:
            io.checksum(self.csv.as_posix())

    def mask_bad_bands(self):
        """Mask invalid bands (OSF, bad bands)."""
        self.img = self.img.where(~self.img.band.isin((*instrument.OSF, *instrument.INVALID)))

    def save(self, fout, sub_rows=None, snr_sidecar=True, empirical_sidecar=True, full_bands=True):
        """
        Stream the image cube to disk with bounded memory.

        ENVI (.img, default): float32 BIL, streamed sub_rows scanlines at a time. The dask
        threaded scheduler computes each block across all cores; peak memory stays ~one block
        regardless of image or machine size. GeoTIFF (.tif): float32 BigTIFF, windowed blocks.

        When the cube carries an empirical `snr` coordinate (from `L0.calibrate` with attach_snr),
        a float32 sidecar `<basename>_snr<ext>` is written alongside it in the same format (ENVI
        coords don't survive the BIL write), unless snr_sidecar is False.

        When the cube carries `empirical_dark`/`empirical_flat`/`empirical_smile` coordinates (from
        `L0.calibrate` with attach_empirical_frames), they're written to a single
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
        return io.write_camera_product(img, fout, sub_rows)

    def fix_frames(self, fcsv, fill=False):
        """Move each 4-band group onto true frames and NaN its lost ones, or with `fill` rebuild them.

        `fcsv` is the strip's frame table (`frames.build_table`), derived from the nri or nci cube, never an ndi.
        """
        if self.img.attrs.get("frames_fixed"):
            raise ValueError(f"{self.basename}: frames already fixed")
        if not Path(fcsv).exists():
            raise FileNotFoundError(f"no frame table {fcsv}: build it from the nri (L0) or nci (L1) cube first")
        attrs = dict(self.img.attrs)
        img, lost = frames.apply(self.img, fcsv)
        self.img = frames.fill(img, lost) if fill else img
        self.img.attrs.update(attrs, frames_fixed=1, frames_filled=int(fill))

    def _save_snr_sidecar(self, fout, row_block=None):
        """Write the (y, x) empirical broadband SNR field beside fout, in fout's own format."""
        snr = self.img.snr.reset_coords(drop=True).expand_dims(band=[1]).astype("float32")
        snr.attrs = {"name": "SNR", "units": ""}
        f = Path(fout)
        return io.write_camera_product(snr, f.with_name(f.stem + "_snr" + f.suffix), row_block)

    def _save_empirical_sidecar(self, fout):
        """Write the (band, x) empirical frames (dark, flat, smile, flat_unmasked, flat_bin<i>) beside fout as one .npz."""
        f = Path(fout)
        fnpz = f.with_name(f.stem + "_empirical.npz")
        frames = {
            str(k).removeprefix("empirical_"): v.values.astype("float32")
            for k, v in self.img.coords.items()
            if str(k).startswith("empirical_")
        }
        np.savez_compressed(fnpz, **frames, band=self.img.band.values, x=self.img.x.values)
        return str(fnpz)

    def _cube_scan0(self):
        """Absolute scan row of this cube's first line.

        `self.img`'s `y` coord is never reset by cropping (`.sel`, not `.isel`), so the first
        label is already the absolute scan row.
        """
        return int(self.img.y.values[0])

    def _render(self, img, table, tr, crs, row_block=None):
        """`img` pulled through GLT `table` as a lazy (band, y, x) DataArray, `row_block` map rows
        per dask chunk, so a write streams it instead of holding the whole map grid."""
        import dask.array as dsa

        cube, scan0 = img.values, self._cube_scan0()
        rows = row_block or max(1, int(io.CHUNKSIZE / (cube.shape[0] * table.shape[2] * 4)))
        t = dsa.from_array(table, chunks=(2, rows, -1))
        warped = t.map_blocks(
            lambda b: georef.apply_glt(cube, b, cube_scan0=scan0),
            chunks=((cube.shape[0],), *t.chunks[1:]),
            dtype="float32",
        )
        da = xr.DataArray(warped, dims=("band", "y", "x"), coords={"band": img.band.values})
        if "wl" in img.coords:
            da = da.assign_coords(wl=("band", img.wl.values))
        da.attrs = dict(self.img.attrs)
        return da.rio.write_crs(crs).rio.write_transform(tr)

    def to_geotiff(self, fout, *, geom=None, grid=None, glt=None, bands=None, row_block=None, full_bands=True):
        """Render `self.img` onto the map grid `grid` through `geom`'s GLT and write it.

        `grid` is a `GeorefConfig` (see `Geometry.grid`). `glt` is a saved GLT (`georef.save_glt`, e.g. the solve's
        `<sid>_<group>_glt.tif`) used instead of `geom`/`grid`; its own grid and CRS win and it is cropped to the scans
        this cube holds. `bands` narrows which band numbers render; `full_bands` reindexes the output to the full
        1..256 IIRS band range before writing.
        """
        if glt is not None:
            with rasterio.open(glt) as src:
                table, tr, crs = src.read(), src.transform, src.crs
            table, tr = georef.crop_glt(table, tr, self._cube_scan0(), self.img.sizes["y"])
        else:
            if geom is None or grid is None:
                raise ValueError("to_geotiff needs geom and grid, or a saved glt")
            table, tr = geom.glt(grid)
            crs = georef.stereo_crs(grid.pole)
        img = self.img if bands is None else self.img.sel(band=list(bands))
        da = self._render(img, table, tr, crs, row_block)
        return io.write_camera_product(_expand_to_full_bands(da) if full_bands else da, fout, row_block)

    def clip_aoi(self, polygon, *, geom, res, geom_crs=None, fout=None, row_block=None, full_bands=True):
        """Render `self.img` clipped to `polygon` through `geom`'s GLT at `res` m/px.

        The polygon's bounding box in the group's CRS becomes the GLT's bounds to keep the render small, then
        any GLT cell outside the true polygon (not just its bbox) is nulled before `apply_glt`
        resamples. Returns the clipped `DataArray` if `fout` is None, else writes it and returns
        the written path.
        """
        from rasterio.features import geometry_mask
        from rasterio.warp import transform_geom

        crs = georef.stereo_crs(geom.group)
        poly = transform_geom(georef.LONLAT if geom_crs is None else geom_crs, crs, polygon)
        xs, ys = zip(*poly["coordinates"][0], strict=False)
        table, tr = geom.glt(geom.grid(res, (min(xs), min(ys), max(xs), max(ys))))
        poly_mask = geometry_mask([poly], out_shape=table.shape[1:], transform=tr, invert=True)
        table = table.copy()
        table[0][~poly_mask] = georef.NODATA  # outside the true polygon, not just its bbox
        if not (table[0] >= 0).any():
            raise ValueError(f"no camera pixels of {self.basename} fall within the given polygon")

        da = self._render(self.img, table, tr, crs, row_block)
        if fout is None:
            return da
        return io.write_camera_product(_expand_to_full_bands(da) if full_bands else da, fout, row_block)

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

        # Camera space: scan 0 at the top
        p = data.plot(vmin=vmin, cmap=cmap, yincrease=False, **kwargs)
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
        ax = plot_spectra_with_sigma(data, **kwargs)
        ylabel = self.img.attrs.get("name", "")
        ylabel += f" [${self.img.attrs.get('units', '')}$]" if self.img.attrs.get("units", "") else ""
        ax.set_ylabel(ylabel)
        return ax

    def detect_stripes(self, sigma_threshold):
        """Detect stripes in image. See empirical.detect_stripes()."""
        return empirical.detect_stripes(self.img, sigma_threshold)

    def apply_smoothing(self, **kwargs):
        return polish.apply_smoothing(self.img, **kwargs)


class L0(IIRSData):
    """Class for reading and handling L0 IIRS data (digital numbers)."""

    @classmethod
    def from_xarray(cls, data, basename, directory="."):
        """Create L0 instance from a (cropped) digital-number xarray.DataArray.

        `qub` is the scene's raw .qub in `directory`, else its .xml label, which is all `calibrate` reads.
        """
        instance = cls.__new__(cls)
        instance.basename = io.iirsbasename(basename)
        instance.directory = Path(directory).expanduser().absolute()
        instance.level = 0
        instance.img = data
        instance.shape = data.shape
        instance.nband, instance.ny, instance.nx = instance.shape
        instance.extent = (int(data.x.min()), int(data.x.max()), int(data.y.min()), int(data.y.max()))
        instance.bounds = (instance.extent[0], instance.extent[2], instance.extent[1], instance.extent[3])
        instance.img.attrs["name"] = "Digital Number"
        instance.img.attrs["units"] = ""

        exts = ("qub", "hdr", "xml", "lbr", "oat", "oath", "spm")
        paths = io.get_iirs_paths(instance.directory, exts=exts, level=0, basenames=[instance.basename])
        for ext in exts:
            setattr(instance, ext, paths.get(ext, {}).get(instance.basename, ""))
        instance.qub = instance.qub or instance.xml
        instance.metadata = instance._extract_metadata() if instance.qub else {}
        return instance

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

    def calibrate(
        self,
        config="iirspy",
        *,
        empirical_kws=None,
        calib_dir=instrument.PKG_DATA,
        output_bands=None,
        frame_offsets=None,
        **overrides,
    ):
        """Calibrate to an L1 object, with ISSDC L1 metadata if it exists (see `iirspy.l1.calibrate`).

        `config` is an `L1Config` or a preset name; `overrides` replace single fields. `empirical_kws` go to
        `empirical.empirical_frames`, e.g. dark_yrange/flat_yrange (positional row ranges into this cube) or
        apply_smile=True. `output_bands` narrows the returned bands. `frame_offsets` is the strip's frame table
        (`frames.build_table`) for `fix_frames`; None builds it from this cube (all 256 bands, ~1 min) and caches it
        beside the .qub.
        """
        img = l1.calibrate(
            self.img,
            L1Config.resolve(config, **overrides),
            qub=self.qub,
            calib_dir=calib_dir,
            empirical_kws=empirical_kws,
            frame_offsets=frame_offsets,
            output_bands=output_bands,
        )
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

    _NATIVE_SCALE = instrument.RAD_NATIVE_SCALE  # [1000 mW/cm^2/sr/um] -> [W/m^2/sr/um]
    _SOURCE = "empirical"

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
        instance.basename = io.iirsbasename(basename)
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
            instance.geomdf, xy_extent = ancillary.parse_geom(instance.csv, latlonextent, center=False)
            instance.extent = xy_extent

        if instance.spm is not None:
            # Solar incidence/azimuth need the scene's per-line clock times, which come from the
            # L1 qub/xml label metadata (start time, exposure, line count, orbit direction). Locate
            # the label for this basename; the tiny .xml suffices if the bulky .qub was cleaned up.
            paths = io.get_iirs_paths(instance.directory, exts=("qub", "xml"), level=1, basenames=[instance.basename])
            fimg = paths.get("qub", {}).get(instance.basename) or paths.get("xml", {}).get(instance.basename)
            if fimg is not None:
                instance.qub = fimg
                # yrange as contiguous absolute line indices matching this cube's y, so the
                # incidence array lines up exactly (extent min/max int-truncation can be off by 1).
                y0 = round(float(instance.img.y.min()) - 0.5)
                yr = (y0, y0 + instance.img.sizes["y"])
                inc, iaz = ancillary.get_iirs_inc_az(fimg, instance.spm, yr)
                instance.img = instance.img.assign_coords({
                    "solar_inc": ("y", inc),
                    "solar_az": ("y", iaz),
                })

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
        self.geomdf, xy_extent = ancillary.parse_geom(self.csv, lonlatextent, xyextent, center=False)
        # replace any None in extent with values from xy_extent
        self.extent = tuple(xy if ex is None else ex for ex, xy in zip(self.extent, xy_extent, strict=False))
        # ...but the geometry csv spans the whole strip and the cube may be a crop: with no extent
        # given, take the line range from the image (`io.apply_envi_start` put y in strip numbering).
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
        x1 = None if self.extent[1] is None else self.extent[1] + 1
        self.img = self.img.sel(y=slice(*self.extent[-2:]), x=slice(self.extent[0], x1))
        try:
            self.bounds = self.img.rio.bounds()
        except NoDataInBounds as e:
            raise ValueError(f"No data found within extent {self.extent} for {self.basename}.") from e

        self.shape = self.img.shape  # [nband, ny, nx]

        # Fix units
        self.img *= instrument.RAD_NATIVE_SCALE  # [1000 mW/cm^2/sr/um] -> [W/m^2/sr/um]
        self.img.attrs["name"] = "Radiance"
        self.img.attrs["units"] = "W/m^2/sr/um"
        self.img.attrs["calibration_source"] = "issdc"

        # Get solar incidence for each line of image
        inc, iaz = ancillary.get_iirs_inc_az(self.qub, self.spm, self.extent[2:])

        # Assign new coords
        self.img = self.img.assign_coords({"solar_inc": ("y", inc), "solar_az": ("y", iaz)})

    def plot(self, band=12, yrange=(None, None), xrange=(None, None), **kwargs):
        """Plot image at band and x, y indices if supplied."""
        if "cbarlabel" not in kwargs:
            kwargs["cbarlabel"] = "Radiance [$W/m^2/sr/um$]"
        _, ax = super().plot(band, yrange, xrange, **kwargs)
        return ax

    def plot_spectra(self, bands=(None, None), yrange=(None, None), xrange=(None, None), **kwargs):
        return super().plot_spectra(bands, yrange, xrange, **kwargs)

    def georef(self, method, *, work=None, **inputs):
        """The scene's `Geometry` by `method` (a `GEOREF` key); `inputs` are that method's own (see `iirspy.georef`)."""
        return GEOREF[method](
            work=work, sid=self.basename, shape=self.img.shape[-2:], row0=self._cube_scan0(), **inputs
        )

    def calibrate(self, config="iirspy", *, geom=None, **overrides):
        """Calibrate to an L2 reflectance object (see `iirspy.l2.calibrate`; needs the SPM file for solar angles).

        `config` is an `L2Config` or a preset name; `overrides` replace individual fields, e.g.
        `calibrate("issdc", polish=None)`. `geom` is the scene's `Geometry` (from `georef`) for the photometric step.
        """
        if self.spm is None:
            raise FileNotFoundError(
                f"SPM file required for reflectance calibration. Expected: "
                f"{self.directory}/miscellaneous/calibrated/ch2_iir_nci_{self.basename}_spm.dat"
            )
        return L2._from_l1(self, L2Config.resolve(config, **overrides), geom)


class L2(IIRSData):
    """Class for IIRS L2 reflectance data."""

    # Per-scene attrs `iirspy.l2.calibrate` records beside `l2_config`
    DIAGNOSTICS = (
        "solar_distance_au",
        "inc_min_deg",
        "inc_max_deg",
        "inc_mean_deg",
        "cos_inc_floor_frac",
        "cos_view_floor_frac",
        "slope_az_used",
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
        instance.basename = io.iirsbasename(basename)
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
            instance.geomdf, xy_extent = ancillary.parse_geom(instance.csv, latlonextent, center=False)
            instance.extent = xy_extent

        return instance

    @classmethod
    def _from_l1(cls, l1_instance, cfg, geom=None):
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

        instance.img = l2.calibrate(
            l1_instance.img, cfg, qub=l1_instance.qub if hasattr(l1_instance, "qub") else None, geom=geom
        )
        instance.img.attrs["calibration_source"] = "user"

        return instance

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

        issdc = None if self.geomdf is not None else read_issdc_loc(self.qub)
        if issdc is not None:
            self.img = _flip(self.img, issdc[1])  # into the nci camera frame, like the loc
        if self.geomdf is not None:
            self.geomdf, xy_extent = ancillary.parse_geom(self.csv, lonlatextent, center=False)
            self.extent = tuple(xy if ex is None else ex for ex, xy in zip(self.extent, xy_extent, strict=False))
        elif any(e is not None for e in lonlatextent):
            if issdc is None:
                raise FileNotFoundError(f"no geometry csv or _d_loc_ backplane beside {self.qub} to crop by lon/lat")
            self.extent = _loc_extent(issdc[0], lonlatextent)

        # x extent is an inclusive last-pixel index (249 on a 250-wide cube) against pixel-centre
        # coords, so +1 keeps that last column; y stays half-open.
        x1 = None if self.extent[1] is None else self.extent[1] + 1
        self.img = self.img.sel(y=slice(*self.extent[-2:]), x=slice(self.extent[0], x1))
        try:
            self.bounds = self.img.rio.bounds()
        except NoDataInBounds as e:
            raise ValueError(f"No data found within extent {self.extent} for {self.basename}.") from e

        self.shape = self.img.shape
        self.img = self.img.where(self.img != 0)  # ISSDC ndi fill is exact 0, not a declared nodata

        if self.spm is not None:
            inc, iaz = ancillary.get_iirs_inc_az(self.qub, self.spm, self.extent[2:])
            self.img = self.img.assign_coords({"solar_inc": ("y", inc), "solar_az": ("y", iaz)})

    def plot(self, band=12, yrange=(None, None), xrange=(None, None), north_up=True, **kwargs):
        """Plot reflectance image."""
        if "cbarlabel" not in kwargs:
            kwargs["cbarlabel"] = "Reflectance"
        _, ax = super().plot(band, yrange, xrange, **kwargs)
        return ax

    def plot_spectra(self, bands=(None, None), yrange=(None, None), xrange=(None, None), **kwargs):
        return super().plot_spectra(bands, yrange, xrange, **kwargs)


def plot_spectra_with_sigma(da: xr.DataArray, ax=None, label: str = "", stdev_alpha=0.2, **kwargs):
    """
    Plot the median reflectance spectrum and ±1 sigma spread as lower alpha bands.

    Parameters
    ----------
    da : xr.DataArray
        Reflectance data with dimensions ('band', 'y', 'x').
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, uses current axis.
    label : str, optional
        Label for the median line.
    color : str, optional
        Color for the median line and fill.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    # Compute statistics along spatial dimensions
    median = da.median(dim=("x", "y")).compute()
    std = da.std(dim=("x", "y")).compute()
    wl = da.wl.values

    # Plot median
    ax.plot(wl, median, label=label, **kwargs)

    # Plot ±1 sigma as filled area
    ax.fill_between(
        wl,
        median - std,
        median + std,
        color="gray",
        alpha=stdev_alpha,
    )
    ax.set_xlabel("Wavelength")
    ax.legend(frameon=False)
    return ax
