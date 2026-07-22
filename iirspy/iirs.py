import json
import warnings
from abc import ABC, abstractmethod
from importlib import metadata
from pathlib import Path

import numpy as np
import pdr
import xarray as xr
from rioxarray.exceptions import NoDataInBounds

import iirspy.utils as utils
from iirspy.empirical import FLAT_SMILE_MIN, empirical_frames

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
        self.qub = paths["qub"].get(self.basename, "")
        self.hdr = paths["hdr"].get(self.basename, "")
        self.xml = paths["xml"].get(self.basename, "")
        self.lbr = paths["lbr"].get(self.basename, "")
        self.oat = paths["oat"].get(self.basename, "")
        self.oath = paths["oath"].get(self.basename, "")
        self.spm = paths["spm"].get(self.basename, "")
        if self.level == 1:
            self.csv = paths["csv"].get(self.basename, "")
            self.xml_csv = paths["xml-csv"].get(self.basename, "")

        # Store metadata from the qub file
        self.metadata = self._extract_metadata()

        # Read image
        self.img = xr.open_dataarray(self.qub, engine="rasterio")
        self.shape = self.img.shape
        self.nband, self.ny, self.nx = self.shape

        # Assign wavelength [nm] as coordinate
        self.img = self.img.assign_coords(wl=("band", utils.get_wls()))

        # Chunk with dask if needed
        if chunk and self.nband * self.ny * self.nx * 4 > utils.CHUNKSIZE:
            if not isinstance(chunk, dict):
                dy = int(utils.CHUNKSIZE / (self.nband * self.nx * 4))
                chunk = {"band": self.nband, "y": dy, "x": self.nx}
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

    def save(self, fout, sub_rows=1000, snr_sidecar=True):
        """
        Stream the image cube to disk with bounded memory.

        ENVI (.img, default): float32 BIL, streamed sub_rows scanlines at a time. The dask
        threaded scheduler computes each block across all cores; peak memory stays ~one block
        regardless of image or machine size. GeoTIFF (.tif): float32 BigTIFF, windowed blocks.

        When the cube carries an empirical `snr` coordinate (from calibrate_to_rad with attach_snr),
        a float32 sidecar `<basename>_snr.img` is written alongside it (ENVI coords don't survive the
        BIL write), unless snr_sidecar is False.

        Parameters
        ----------
        fout : str or Path
            Output path. Extension selects the format (.tif -> GeoTIFF, else ENVI).
        sub_rows : int
            Scanlines computed/written per step (bounds peak memory).
        snr_sidecar : bool
            Write the broadband `snr` field (if present) to a float32 sidecar next to fout.

        Returns
        -------
        str : the path written.
        """
        fout = str(fout)
        if snr_sidecar and "snr" in self.img.coords:
            self._save_snr_sidecar(fout)
        if fout.lower().endswith(".tif"):
            return self._save_geotiff(fout, sub_rows)
        description = f"IIRS {self.img.attrs.get('name', '')}".strip()
        return utils.write_envi_bil(self.img, fout, sub_rows, description)

    def _save_snr_sidecar(self, fout):
        """Write the (y, x) empirical broadband SNR field to a float32 ENVI sidecar next to fout."""
        import rasterio

        snr = self.img.snr.values.astype("float32")
        ny, nx = snr.shape
        fsnr = Path(fout).with_name(Path(fout).stem + "_snr.img")
        with rasterio.open(fsnr, "w", driver="ENVI", height=ny, width=nx, count=1, dtype="float32") as dst:
            dst.write(snr, 1)
        return str(fsnr)

    def _save_geotiff(self, fout, row_block):
        """Write a float32 BigTIFF sequentially in windowed blocks (bounded memory)."""
        import rasterio
        from dask.diagnostics import ProgressBar
        from rasterio.windows import Window

        da = self.img
        nband, ny, nx = da.shape
        profile = {
            "driver": "GTiff",
            "height": ny,
            "width": nx,
            "count": nband,
            "dtype": "float32",
            "nodata": np.nan,
            "crs": da.rio.crs if da.rio.crs else None,
            "transform": da.rio.transform(),
            "compress": "LZW",
            "tiled": True,
            "interleave": "band",  # per-band planes: viewers read one band without decompressing all 256
            "BIGTIFF": "YES",
        }
        wls = [f"{float(w):.2f}" for w in da.wl.values] if "wl" in da.coords else None
        # Provenance: scalar attrs (incl. empirical_notes JSON) round-trip as GDAL metadata tags
        tags = {k: str(v) for k, v in da.attrs.items() if isinstance(v, str | int | float | bool)}
        with rasterio.open(fout, "w", **profile) as dst, ProgressBar():
            if tags:
                dst.update_tags(**tags)
            for y0 in range(0, ny, row_block):
                y1 = min(y0 + row_block, ny)
                block = da.isel(y=slice(y0, y1)).values.astype("float32", copy=False)
                dst.write(block, window=Window(0, y0, nx, y1 - y0))
            if wls:
                for i, wl in enumerate(wls):
                    dst.set_band_description(i + 1, wl)
        return fout

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
        p, ax = super().plot(band, yrange, xrange, **kwargs)

        # Flip image if flip_xy (can plot a descending orbit with north up)
        if flip_xy:
            ax.yaxis.set_inverted(True)
            ax.xaxis.set_inverted(True)
        return ax

    def plot_spectra(self, bands=(None, None), yrange=(None, None), xrange=(None, None), **kwargs):
        return super().plot_spectra(bands, yrange, xrange, **kwargs)

    def calibrate_to_rad(
        self,
        empirical=False,
        interp_bands=None,
        interp_spatial=False,
        attach_snr=True,
        empirical_kws=None,
        bad_pixel_mask=True,
        calib_dir=utils.DCALIB,
    ):
        """
        Perform IIRS L0 digital number to L1 radiance calibration.

          L1_Radiance [mW/cm^2/sr/um] = Gain * L0_DN + Offset

        Raw data is already dark subtracted and automatically retrieves correct gain and offset
        corresponding to the instrument gain / exposure settings used during acquisition.

        Note: Returns radiance in [W/m^2/sr/µm] which is a factor of 10 greater than [mW/cm^2/sr/um].

        The IIRS gain/offset LUT is per detector element (band, x), but its cross-track structure
        is poorly correlated with the on-orbit response and injects striping/speckle. With
        empirical=True, derive a per-scene residual dark + sensor flat + band-relative smile (see
        iirspy.empirical) and use the LUT only for the per-band absolute scale. The in-scene dark,
        when present, anchors the zero point (no longer use the LUT offset):

          scene with dark rows   : rad = 10 * gain_med * (DN - dark_resid) / (flat * smile)
          scene without dark rows: rad = 10 * (gain_med * DN / (flat * smile) + offset_med)

        Steps
        -----
        1) Retrieve the per-element gain/offset LUT for the scene's gain/exposure settings.
        2) Convert DN -> radiance and scale [mW/cm^2/sr/um] -> [W/m^2/sr/µm] (factor of 10):
           - empirical=False: apply the per-element LUT directly (rad = 10 * (DN * gain + offset)).
           - empirical=True: derive per-scene residual dark + sensor flat (+ optional smile) via
             empirical_frames, take the LUT band-median gain/offset, and divide out flat*smile.
             Drop the LUT offset when in-scene dark rows anchor the zero point, else keep it.
             Attach the broadband SNR field when attach_snr and dark rows exist.
        3) Set invalid data to NaN (bad bands, order sorting filters, known bad pixels, saturated pixels).
        4) (Optional) Fill NaNs by interpolating across band using strategy in interp_bands (e.g. "linear").
        5) (NOT IMPLEMENTED) IIRS postprocessing steps (keystone correction, radiance adjustment in OSF and at edges).

        Parameters
        ----------
        empirical : bool
            Apply the on-orbit empirical dark/flat (+ optional smile) correction with LUT band-median scale.
        interp_bands : str or None
            If set, fill masked bands by interpolating across band (e.g. "linear").
        interp_spatial : bool
            After interp_bands, fill isolated single-column NaN gaps (detector columns that are
            NaN across every band, so band interpolation cannot reach them) by interpolating
            across x. Only single-column gaps are filled: wider seams (e.g. OSF, multi-column
            bad-pixel runs) and image-edge columns (no neighbour on one side) are left as NaN.
        attach_snr : bool
            When empirical and the scene has shadow, attach the per-pixel broadband SNR field (float32
            (y, x)) as an `snr` coordinate on the returned DataArray. Radiance is left as measured:
            downstream users threshold the SNR themselves (a typical cut is empirical.SHADOW_SNR) to
            null, ignore, or study low-signal shadow pixels. save() can write it to a float32 sidecar
            (ENVI coords don't survive the BIL write).
        empirical_kws : dict or None
            Extra keyword arguments forwarded to empirical.empirical_frames, e.g.
            dark_yrange=(ylow, yhigh) / flat_yrange=(ylow, yhigh) to override the auto dark-row /
            flat-region detection with user-picked positional row ranges into this cube.
            The smile correction is off by default; enable it with apply_smile=True.
        bad_pixel_mask : bool or array-like
            Mask known bad detector elements (x, band). True (default) uses the packaged mask
            (utils.load_bad_pixel_mask); False skips masking; or pass a custom boolean mask
            (True where bad) to null instead.

        Returns
        -------
        (xarray.DataArray): Radiance DataArray in [W/m^2/sr/µm].
        """
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
        else:
            # Apply per-element gain and offset to convert DN -> Radiance
            rad = 10 * (self.img * gain + offset)  # [mW/cm^2/sr/μm] -> [W/m^2/sr/um]

        # Drop OSF and invalid bands. interp if specified
        rad = rad.where(~rad.band.isin((*utils.OSF, *utils.INVALID)))

        # Drop known bad detector elements (x, bands)
        if bad_pixel_mask is True:  # Load default bad pixel mask
            bad_pixel_mask = utils.load_bad_pixel_mask()
        if bad_pixel_mask is not False:  # Apply user-supplied mask
            rad = rad.where(~bad_pixel_mask)

        # Drop saturated pixels
        rad = rad.where(rad < utils.get_saturation_radiance(self.qub, calib_dir))

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
        out.attrs["iirspy_version"] = metadata.version("iirspy")
        if empirical:
            out.attrs["empirical_notes"] = json.dumps(emp_notes)  # provenance: how the product was made

        return out

    def calibrate(self, *args, **kwargs):
        """Calibrate to L1 radiance using ISSDC calibration files if they exist."""
        img = self.calibrate_to_rad(*args, **kwargs)
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
                # yrange as contiguous absolute line indices matching this cube's y, so the
                # incidence array lines up exactly (extent min/max int-truncation can be off by 1).
                y0 = int(round(float(instance.img.y.min()) - 0.5))
                yr = (y0, y0 + instance.img.sizes["y"])
                inc, iaz = utils.get_iirs_inc_az(fimg, instance.spm, yr)
                instance.img = instance.img.assign_coords({
                    "solar_inc": ("y", inc),
                    "solar_az": ("y", iaz),
                })

        return instance

    @classmethod
    def from_geotiff(cls, path, basename, directory="."):
        """Load a cropped empirical-L1 GeoTIFF (written in native [1000 mW/cm^2/sr/um]) as an L1.

        The GeoTIFF transform carries the original absolute line/sample offset, so the reloaded
        cube keeps full-scene pixel coordinates and its geometry (solar angles, lat/lon) can be
        re-derived from the ancillary spm/csv in `directory`. Radiance is rescaled to physical
        [W/m^2/sr/um] exactly as the ISSDC L1 reader does (utils.RAD_NATIVE_SCALE), so the result
        feeds L2 identically to a downloaded L1. See L0.calibrate + run_l1_polar.py for the writer.

        Parameters
        ----------
        path : str or Path
            Cropped L1 GeoTIFF written by the polar pipeline (radiance in [1000 mW/cm^2/sr/um]).
        basename : str
            Image basename (e.g. 20210723T1445053074), used to find spm/geometry in `directory`.
        directory : str
            Directory holding the IIRS bundle (nci ancillary: spm + geometry csv).
        """
        da = xr.open_dataarray(path, engine="rasterio").sortby("y").sortby("x")
        nband = da.shape[0]
        da = da.assign_coords(band=np.arange(1, nband + 1), wl=("band", utils.get_wls()[:nband]))
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
        self.img = self.img.sel(y=slice(*self.extent[-2:]), x=slice(*self.extent[:2]))
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
        lon, lat = utils.geom2latlon_coords(self.geomdf, xy_extent, self.shape[2], self.shape[1])

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
        p, ax = super().plot(band, yrange, xrange, x=x, y=y, **kwargs)
        return ax

    def plot_spectra(self, bands=(None, None), yrange=(None, None), xrange=(None, None), **kwargs):
        return super().plot_spectra(bands, yrange, xrange, **kwargs)

    def calibrate(self, inc=None, dem=None, solar_flux=None, thermal_corr=""):
        """Calibrate to L2 reflectance object (requires SPM file for solar angles)."""
        if self.spm is None:
            raise FileNotFoundError(
                f"SPM file required for reflectance calibration. Expected: "
                f"{self.directory}/miscellaneous/calibrated/ch2_iir_nci_{self.basename}_spm.dat"
            )

        return L2._from_l1(
            self,
            inc=inc,
            dem=dem,
            solar_flux=solar_flux,
            thermal_corr=thermal_corr,
        )


class L2(IIRSData):
    """Class for IIRS L2 reflectance data."""

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
    def _from_l1(cls, l1_instance, inc=None, dem=None, solar_flux=None, thermal_corr=""):
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
            l1_instance.csv if hasattr(l1_instance, "csv") else None,
            inc=inc,
            dem=dem,
            solar_flux=solar_flux,
            thermal_corr=thermal_corr,
        )
        instance.img.attrs["calibration_source"] = "user"

        return instance

    def _compute_reflectance(self, rad, qub_path, csv_path, inc=None, dem=None, solar_flux=None, thermal_corr=""):
        """
        Compute I/f reflectance from radiance data and solar geometry.

        Optinally corrects for topographic effects and thermal correction.

        Parameters
        ----------
        rad : xarray.DataArray
            Radiance data array containing solar_inc, solar_az, and wl attributes.
        qub_path : str or None
            Path to QUB file for extracting solar distance. If None, assumes solar distance of 1.0 AU.
        csv_path : str
            Path to CSV file containing IIRS latitude/longitude information.
        inc : float or xarray.DataArray, optional
            Solar incidence angle in degrees. If None, extracted from rad coordinates.
        dem : xarray.DataArray or str, optional
            Digital elevation model as an xarray DataArray or path to DEM file.
            If provided, topographic effects are corrected using the DEM. Default is None.
        solar_flux : array-like, optional
            Solar flux values for each band. If None, computed automatically from solar distance
            and default solar flux constants. Default is None.
        thermal_corr : str, optional
            Thermal correction method. Currently supports:
            - "" (empty string): No thermal correction (default)
            - "verma": Apply Verma thermal correction method
        Returns
        -------
        xarray.DataArray
            Reflectance data array with name "Reflectance" and empty units attribute.
        Raises
        ------
        ValueError
            If thermal_corr is not "" or "verma".
        Notes
        -----
        The reflectance is calculated as:
        where trad is the thermal radiance component (if thermal correction is applied).
        """
        cos_inc = np.cos(np.radians(inc)) if inc is not None else np.cos(np.radians(rad.solar_inc))

        if dem is not None:
            if not hasattr(dem, "sel"):
                dem = xr.open_dataarray(dem, engine="rasterio")
            lat, _ = utils.get_iirs_latlon(csv_path)
            lat = lat.interp_like(rad.solar_inc)
            dem = dem.interp_like(rad.sel(band=dem.band), method="slinear")
            cos_inc = utils.get_cos_inc_dem(dem, rad.solar_inc, rad.solar_az, lat)
        cos_inc = cos_inc * xr.ones_like(rad.isel(x=0, band=0))

        sdist = utils.get_solar_distance(qub_path) if qub_path else 1.0

        if solar_flux is None:
            solar_flux = utils.get_solar_flux(sdist, utils.FSOLAR)

        trad = 0.0
        if thermal_corr == "verma":
            trad = utils.get_thermal_rad_verma(rad, rad.wl * 1e-9)
        elif thermal_corr:
            raise ValueError('thermal_corr must be "" or "verma"')

        refl = (rad - trad) / (cos_inc * solar_flux)
        # add wl array as coordinate if missing
        if "wl" not in refl.coords:
            refl = refl.assign_coords(wl=rad.wl)
        refl.name = "Reflectance"
        refl.attrs["name"] = "Reflectance"
        refl.attrs["units"] = ""

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

        self.img = self.img.sel(y=slice(*self.extent[-2:]), x=slice(*self.extent[:2]))
        try:
            self.bounds = self.img.rio.bounds()
        except NoDataInBounds as e:
            raise ValueError(f"No data found within extent {self.extent} for {self.basename}.") from e

        self.shape = self.img.shape

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
        p, ax = super().plot(band, yrange, xrange, x=x, y=y, **kwargs)
        return ax

    def plot_spectra(self, bands=(None, None), yrange=(None, None), xrange=(None, None), **kwargs):
        return super().plot_spectra(bands, yrange, xrange, **kwargs)
