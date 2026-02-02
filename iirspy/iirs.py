from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pdr
import xarray as xr
from rioxarray.exceptions import NoDataInBounds

import iirspy.utils as utils

# Skip div 0 and 0/0 warnings
np.seterr(divide="ignore", invalid="ignore")


def bounds2extent(bounds):
    """Convert bounds to extent tuple."""
    minx, miny, maxx, maxy = bounds
    return (minx, maxx, miny, maxy)


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
        self.img = self.img.assign_coords(wl=("band", utils.get_wls(self.qub)))

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
            "version_id",
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

        # Plot
        p = data.plot(vmin=vmin, cmap=cmap, **kwargs)
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
        return utils.plot_spectra_with_sigma(data, **kwargs)

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
        self.extent = bounds2extent(self.bounds)

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

    def calibrate_to_rad(self, denoise_gain=False, interp_bands=None, calib_dir=utils.DCALIB):
        """
        Perform IIRS L0 digital number to L1 radiance calibration.

          L1_Radiance [mW/cm^2/sr/um] = Gain * L0_DN + Offset

        Raw data is already dark subtracted and automatically retrieves correct gain and offset
        corresponding to the instrument gain / exposure settings used during acquisition.

        Note: Returns radiance in [W/m^2/sr/µm] which is a factor of 10 greater than [mW/cm^2/sr/um].

        Steps
        -----
        1) Retrieve and apply gain and offset to convert to radiance [W/cm^2/sr/um].
        2) Convert to [W/m^2/sr/µm].
        3) Set invalid data to NaN (bad bands, order sorting filters, saturated pixels)
        4) (Optional) Fill NaNs by interpolating across band using strategy in interp_bands (e.g. "linear")
        5) (NOT IMPLEMENTED) IIRS postprocessing steps (keystone correction, radiance adjustment in OSF and at edges).

        Returns
        -------
        (xarray.DataArray): Radiance DataArray in [W/m^2/sr/µm].
        """
        gain, offset = utils.get_gain_offset(self.qub, denoise_gain, calib_dir=calib_dir)

        # Apply gain and offset to convert DN -> Radiance
        rad = 10 * (self.img * gain + offset)  # [mW/cm^2/sr/μm] -> [W/m^2/sr/um]

        # Drop OSF and invalid bands. interp if specified
        rad = rad.where(~rad.band.isin((*utils.OSF, *utils.INVALID)))

        # Drop saturated pixels
        rad = rad.where(rad < utils.get_saturation_radiance(self.qub, calib_dir))

        # Interpolate across bands
        if interp_bands is not None:
            rad = rad.interpolate_na("band", max_gap=11, keep_attrs=True, method=interp_bands)

        # Format and output Radiance DataArray
        out = rad
        out.name = "Radiance [W/m^2/sr/µm]"

        return out

    def calibrate_to_l1(self, *args, **kwargs):
        """Calibrate to IIRS Level 1 radiance object.

        Requires L1 data to be downloaded from ISSDC to same directory as L0.
        """
        try:
            out = L1(self.basename, str(self.directory), self.extent)
        except FileNotFoundError as err:
            raise RuntimeError(
                f"Cannot find L1 metadata at {self.directory}. Download from ISSDC \
                or use calibrate_to_rad() method to return radiance without L1 metadata."
            ) from err

        # Replace l1 image with the calibrated image, keeping metadata the same
        out.img = self.calibrate_to_rad(*args, **kwargs)
        return out


class L1(IIRSData):
    """Class for reading, handling, and processing L1 IIRS data to L2 reflectance."""

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
        self.geomdf, xy_extent = utils.parse_geom(self.csv, lonlatextent, center=False)
        # replace any None in extent with values from xy_extent
        self.extent = tuple(xy if ex is None else ex for ex, xy in zip(self.extent, xy_extent))
        self.img = self.img.sel(y=slice(*self.extent[-2:]), x=slice(*self.extent[:2]))
        try:
            self.bounds = self.img.rio.bounds()
        except NoDataInBounds as e:
            raise ValueError(f"No data found within extent {self.extent} for {self.basename}.") from e

        self.shape = self.img.shape  # [nband, ny, nx]

        # Fix units
        self.img *= 0.01  # [1000 mW/cm^2/sr/um] -> [W/m^2/sr/um]

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

    def plot(self, band=12, yrange=(None, None), xrange=(None, None), north_up=True, **kwargs):
        """Plot image at band and x, y indices if supplied."""
        if "cbarlabel" not in kwargs:
            kwargs["cbarlabel"] = "Radiance [$W/m^2/sr/um$]"
        p, ax = super().plot(band, yrange, xrange, x="lon", y="lat", **kwargs)
        return ax

    def plot_spectra(self, bands=(None, None), yrange=(None, None), xrange=(None, None), **kwargs):
        return super().plot_spectra(bands, yrange, xrange, **kwargs)

    def calibrate_to_refl(self, dem=None, solar_flux=None, thermal_corr=""):
        """
        Perform IIRS radiance to I/f reflectance calibration.

        Steps
        -----


        Returns
        -------
        (xarray.DataArray): I/f reflectance [unitless].
        """
        import numpy as np

        # Compute solar inc, solar flux and thermal rad for I/f = (rad-trad) / (cos(inc) * solar_flux)
        rad = self.img

        cos_inc = np.cos(np.radians(rad.solar_inc))
        # (Optional): adjust cos_inc relative to dem
        if dem is not None:
            if not hasattr(dem, "sel"):
                dem = xr.open_dataarray(dem, engine="rasterio")
            lat, _ = utils.get_iirs_latlon(self.csv)
            lat = lat.interp_like(rad.solar_inc)
            dem = dem.interp_like(rad.sel(band=dem.band), method="slinear")
            cos_inc = utils.get_cos_inc_dem(dem, rad.solar_inc, rad.solar_az, lat)
        cos_inc = cos_inc * xr.ones_like(rad.isel(x=0, band=0))  # Add coords (along y)
        # cos_inc = cos_inc.where((cos_inc > 0.03) & (cos_inc < 0.999))

        # Retrieve solar_flux and scale by solar distance at time of this image
        sdist = utils.get_solar_distance(self.qub)
        if solar_flux is None:
            solar_flux = utils.get_solar_flux(sdist, utils.FSOLAR)

        # Compute thermal radiance
        trad = 0.0
        if thermal_corr == "verma":
            trad = utils.get_thermal_rad_verma(rad, rad.wl * 1e-9)  # wl [nm] -> wl [m]
        elif thermal_corr:
            raise ValueError('Invalid thermal_corr. Options: ("", "verma")')

        # Convert to I/F reflectance
        refl = (rad - trad) / (cos_inc * solar_flux)

        # if polish:
        #     da = utils.apply_spectral_polish(da, fpolish)

        # Format and output Reflectance DataArray
        refl.name = "Reflectance"
        refl = refl.assign_coords(wl=("band", utils.get_wls(self.qub)))
        return refl
