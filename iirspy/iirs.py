from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pdr
import xarray as xr

import iirspy.utils as utils


class IIRSData(ABC):
    """Abstract base class for IIRS data products."""

    def __init__(self, basename, directory=".", extent=(None, None, None, None), chunk=True):
        """
        Initialize the IIRS data class.

        Parameters
        ----------
        basename : str
            Basename of the image to read (e.g. 20201214T0844306700).
        directory : str
            Path to the directory containing IIRS data files.
        extent : tuple
            Extent in (minlon, maxlon, minlat, maxlat) format.
        chunk : bool or dict
            Chunk image automatically (default: True). Or supply dict of x,y,band chunk sizes (see dask).
        """
        self.basename = utils.iirsbasename(basename)
        self.directory = Path(directory).expanduser().absolute()
        self.extent = extent

        # Get paths to relevant files
        paths = utils.get_iirs_paths(self.directory)
        self.qub = paths["qub"].get(basename, "")
        self.hdr = paths["hdr"].get(basename, "")
        self.xml = paths["xml"].get(basename, "")
        self.lbr = paths["lbr"].get(basename, "")
        self.oat = paths["oat"].get(basename, "")
        self.oath = paths["oath"].get(basename, "")
        self.spm = paths["spm"].get(basename, "")

        # Store metadata from the qub file
        self.metadata = self._extract_metadata(self.qub)

        # Read image
        self.img = xr.open_dataarray(self.qub, engine="rasterio")
        self.nband, self.ny, self.nx = self.img.shape

        # Chunk with dask if needed
        if chunk and self.nband * self.ny * self.nx * 4 > utils.CHUNKSIZE:
            if not isinstance(chunk, dict):
                dy = int(utils.CHUNKSIZE / (self.nband * self.nx * 4))
                chunk = {"band": self.nband, "y": dy, "x": self.nx}
            self.img = self.img.chunk(chunk)

    def _extract_metadata(self, qub_file):
        """Extract relevant metadata from the given qub file."""
        img = pdr.open(qub_file)
        metadata = {
            "projection": img.metaget("isda:projection"),
            "orbit_direction": img.metaget("isda:orbit_limb_direction"),
            "start_time": img.metaget("start_date_time"),
            "stop_time": img.metaget("stop_date_time"),
            "exposure": img.metaget("isda:exposure"),
            "gain": img.metaget("isda:gain"),
            "line_exposure_duration": img.metaget("isda:line_exposure_duration"),
            "md5_checksum": img.metaget("md5_checksum"),
        }
        return metadata

    @abstractmethod
    def plot(self, band=12, y=(None, None), x=(None, None), **kwargs):
        """Plot image at band and x, y indices if supplied."""
        pass


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
        super().__init__(basename, directory, extent, chunk)
        self.xy_extent = self.extent
        self.img = self.img.sel(y=slice(*self.xy_extent[-2:]), x=slice(*self.xy_extent[:2]))

    def plot(self, band=12, y=(None, None), x=(None, None), **kwargs):
        """Plot image at band and x, y indices if supplied."""
        data = self.img.sel(band=band, y=slice(*y), x=slice(*x))

        # Defaults
        size = kwargs.pop("size", 5)
        vmin = kwargs.pop("vmin", 0)
        title = kwargs.pop("title", f"{self.basename}")
        cmap = kwargs.pop("cmap", "inferno")
        cbarlabel = kwargs.pop("cbarlabel", "Digital Number")

        # Coarsen data for quicker plot
        data = data.sel(y=slice(None, None, len(data.y) // 1000))

        # Plot
        ax = data.plot(vmin=vmin, size=size, cmap=cmap, **kwargs)
        ax.axes.set_title(title, fontsize=10)
        ax.axes.set_aspect("equal")
        ax.colorbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        return ax


class L1(IIRSData):
    """Class for reading, handling, and processing L1 IIRS data to L2 reflectance."""

    def __init__(self, basename, directory=".", extent=(None, None, None, None), chunk=True, flip_desc=True):
        """
        Initialize the IIRS L1 data class.

        Parameters
        ----------
        basename : str
            Basename of the image to read (e.g. 20201214T0844306700).
        directory : str
            Path to the directory containing IIRS data files.
        extent : tuple
            Extent in (minlon, maxlon, minlat, maxlat) format.
        chunk : bool or dict
            Chunk image automatically (default: True). Or supply dict of x,y,band chunk sizes (see dask).
        flip_desc : bool
            If image collected on descending orbit, flip it in y direction.
        """
        super().__init__(basename, directory, extent, chunk)

        # Get paths to relevant files
        paths = utils.get_iirs_paths(self.directory)
        self.csv = paths["csv"].get(basename, "")
        self.xml_csv = paths["xml-csv"].get(basename, "")

        # Parse geometry, store gcps and extent in x, y
        self.gcps, self.xy_extent = utils.parse_geom(self.csv, extent, as_gcps=True, center=True)
        self.img = self.img.sel(y=slice(*self.xy_extent[-2:]), x=slice(*self.xy_extent[:2]))

        # Fix units
        self.img *= 0.01  # [1000 mW/cm^2/sr/um] -> [W/m^2/sr/um]

        # Flip image if collected on descending orbit
        if flip_desc and self.metadata.get("orbit_direction", "").lower() == "descending":
            ymax = int(np.ceil(xr.open_dataarray(self.qub, engine="rasterio").y.max()))
            self.img.coords["y"] = ymax - self.img.y

    def get_extent(self, latlon=True):
        """
        Get the extent in lat/lon (degrees) of x/y (cols,rows).

        Returns
        -------
        tuple
            Tuple of (minlon, maxlon, minlat, maxlat) or (minx, maxx, miny, maxy)
        """
        return self.extent if latlon else self.xy_extent

    def plot(self, band=12, y=(None, None), x=(None, None), **kwargs):
        """Plot image at band and x, y indices if supplied."""
        data = self.img.sel(band=band, y=slice(*y), x=slice(*x))

        # Defaults
        size = kwargs.pop("size", 5)
        vmin = kwargs.pop("vmin", 0)
        title = kwargs.pop("title", f"{self.basename}")
        cmap = kwargs.pop("cmap", "inferno")
        cbarlabel = kwargs.pop("cbarlabel", "Radiance [$W/m^2/sr/um$]")

        # Coarsen data for quicker plot
        data = data.sel(y=slice(None, None, len(data.y) // 1000))

        # Plot
        ax = data.plot(vmin=vmin, size=size, cmap=cmap, **kwargs)
        ax.axes.set_title(title, fontsize=10)
        ax.axes.set_aspect("equal")
        ax.colorbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        return ax
