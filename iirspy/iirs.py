from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pdr
import xarray as xr

import iirspy.utils as utils

# Skip div 0 and 0/0 warnings
np.seterr(divide="ignore", invalid="ignore")


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
        self.qub = paths["qub"].get(basename, "")
        self.hdr = paths["hdr"].get(basename, "")
        self.xml = paths["xml"].get(basename, "")
        self.lbr = paths["lbr"].get(basename, "")
        self.oat = paths["oat"].get(basename, "")
        self.oath = paths["oath"].get(basename, "")
        self.spm = paths["spm"].get(basename, "")
        if self.level == 1:
            self.csv = paths["csv"].get(basename, "")
            self.xml_csv = paths["xml-csv"].get(basename, "")

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
        metadata = {
            "projection": img.metaget("isda:projection"),
            "orbit_direction": img.metaget("isda:orbit_limb_direction"),  # L1 Only
            "start_time": img.metaget("start_date_time"),
            "stop_time": img.metaget("stop_date_time"),
            "exposure": img.metaget("isda:exposure"),
            "gain": img.metaget("isda:gain"),
            "line_exposure_duration": img.metaget("isda:line_exposure_duration"),
            "md5_checksum": img.metaget("md5_checksum"),
        }
        return metadata

    def metaget(self, key):
        """Search metadata for key."""
        return pdr.open(self.qub).metaget(key)

    def checksum(self):
        """Verify all data was downloaded correctly."""
        utils.checksum(self.qub.as_posix())
        if self.level == 1:
            utils.checksum(self.csv.as_posix())

    @abstractmethod
    def plot(self, band=12, y=(None, None), x=(None, None), **kwargs):
        """
        Plot image at band and x, y indices. Lowers resolution along y if large.
        """
        data = self.img.sel(band=band, y=slice(*y), x=slice(*x))

        # Defaults
        size = kwargs.pop("size", 5)
        vmin = kwargs.pop("vmin", 0)
        title = kwargs.pop("title", f"{self.basename}")
        cmap = kwargs.pop("cmap", "inferno")
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

        return p, ax

    def detect_stripes(self, sigma_threshold):
        """Detect stripes in image. See utils.detect_stripes()."""
        return utils.detect_stripes(self.img, sigma_threshold)


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
        self.bounds = self.img.rio.bounds()

    def plot(self, band=12, y=(None, None), x=(None, None), flip_xy=False, **kwargs):
        """Plot image at band and x, y indices if supplied."""
        cbarlabel = kwargs.pop("cbarlabel", "Digital Number")
        p, ax = super().plot(band, y, x, **kwargs)
        p.colorbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Flip image if flip_xy (can plot a descending orbit with north up)
        if flip_xy:
            ax.yaxis.set_inverted(True)
            ax.xaxis.set_inverted(True)
        return ax

    def calibrate_to_rad(self, denoise_gain=False, interp_bands=None, calib_dir=utils.DCALIB):
        """
        Perform IIRS L0 digital number to L1 radiance calibration.

        Steps
        -----
        0) (Raw data is already dark subtracted).
        1) Find and apply gain and offset to convert to radiance [W/cm^2/sr/um].
        2) Convert to [W/m^2/sr/µm].
        4) Drop invalid data (bad bands, order sorting filters, saturated pixels)
        5) (Optional) Interpolate across band using the strategy in interp_bands (e.g. "linear")
        4) (IIRS steps not implemented) Postprocessing (keystone correction, radiance adjustment in OSF and at edges).
        5) Write to file as 32-bit floating point binary BSQ.

        Returns
        -------
        (xarray.DataArray): Radiance DataArray in [W/m^2/sr/µm].
        """
        gain, offset = utils.get_gain_offset(self.qub, denoise_gain, calib_dir=calib_dir)

        # Apply gain and offset to convert DN -> Radiance
        rad = 10 * self.img * gain + offset  # [W/m^2/sr/um]

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
        extent=(None, None, None, None),
        latlonextent=(None, None, None, None),
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
        extent : tuple
            Extent in (minx, maxx, miny, maxy) format. Only one of xyextent and extent can be given.
        latlonextent : tuple
            Extent in (minlon, maxlon, minlat, maxlat) format.
        chunk : bool or dict
            Chunk image automatically (default: True). Or supply dict of x,y,band chunk sizes (see dask).
        """
        super().__init__(basename, directory, extent, chunk, level=1)

        if any(e is not None for e in extent) and any(e is not None for e in latlonextent):
            raise ValueError("Only one of extent and xyextent can be given.")

        # Parse geometry, store gcps and extent in x, y
        self.gcps, xy_extent = utils.parse_geom(self.csv, latlonextent, as_gcps=True, center=True)
        if all(e is None for e in extent):
            self.extent = xy_extent
        self.img = self.img.sel(y=slice(*self.extent[-2:]), x=slice(*self.extent[:2]))
        self.bounds = self.img.rio.bounds()

        # Fix units
        self.img *= 0.01  # [1000 mW/cm^2/sr/um] -> [W/m^2/sr/um]

    def plot(self, band=12, y=(None, None), x=(None, None), north_up=True, **kwargs):
        """Plot image at band and x, y indices if supplied."""
        cbarlabel = kwargs.pop("cbarlabel", "Radiance [$W/m^2/sr/um$]")
        p, ax = super().plot(band, (None, None), (None, None), **kwargs)
        p.colorbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Flip image if collected on descending orbit
        if north_up and self.metadata.get("orbit_direction", "").lower() == "descending":
            ax.yaxis.set_inverted(True)
            ax.xaxis.set_inverted(True)
        return ax

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

        # Get solar inc
        inc, iaz = utils.get_iirs_inc_az(self.qub, self.spm, self.extent[2:])

        # (Optional): adjust cos_inc relative to dem
        if dem is None:
            cos_inc = np.cos(np.radians(inc))
        else:
            if not hasattr(dem, "sel"):
                dem = xr.open_dataarray(dem, engine="rasterio")
            lat, _ = utils.get_iirs_latlon(self.csv)
            lat = lat.interp_like(inc)
            dem = dem.interp_like(rad.sel(band=dem.band), method="slinear")
            cos_inc = utils.get_cos_inc_dem(dem, inc, iaz, lat)
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
