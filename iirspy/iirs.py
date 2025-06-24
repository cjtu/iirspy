from abc import ABC, abstractmethod
from pathlib import Path

import pdr
import xarray as xr

import iirspy.utils as utils


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
            Extent in (minlon, maxlon, minlat, maxlat) format.
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
        if level == 1:
            self.csv = paths["csv"].get(basename, "")
            self.xml_csv = paths["xml-csv"].get(basename, "")

        # Store metadata from the qub file
        self.metadata = self._extract_metadata(self.qub)

        # Read image
        self.img = xr.open_dataarray(self.qub, engine="rasterio")
        self.shape = self.img.shape
        self.nband, self.ny, self.nx = self.shape
        self.bounds = self.img.rio.bounds()

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
            "orbit_direction": img.metaget("isda:orbit_limb_direction"),  # L1 Only
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
        super().__init__(basename, directory, extent, chunk, level=0)
        self.img = self.img.sel(y=slice(*self.extent[-2:]), x=slice(*self.extent[:2]))

    def plot(self, band=12, y=(None, None), x=(None, None), north_up=True, **kwargs):
        """Plot image at band and x, y indices if supplied."""
        data = self.img.sel(band=band, y=slice(*y), x=slice(*x))

        # Defaults
        size = kwargs.pop("size", 5)
        vmin = kwargs.pop("vmin", 0)
        title = kwargs.pop("title", f"{self.basename}")
        cmap = kwargs.pop("cmap", "inferno")
        cbarlabel = kwargs.pop("cbarlabel", "Digital Number")

        # Coarsen data for quicker plot
        if len(data.y) > 2000:
            data = data.sel(y=slice(None, None, len(data.y) // 1000))

        # Plot
        ax = data.plot(vmin=vmin, size=size, cmap=cmap, **kwargs)
        ax.axes.set_title(title, fontsize=10)
        ax.axes.set_aspect("equal")
        ax.colorbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Flip image if collected on descending orbit
        if north_up and self.bounds[1] < self.bounds[3]:
            ax.axes.yaxis.set_inverted(True)
            ax.axes.xaxis.set_inverted(True)
        return ax


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

        # Fix units
        self.img *= 0.01  # [1000 mW/cm^2/sr/um] -> [W/m^2/sr/um]

    def plot(self, band=12, y=(None, None), x=(None, None), north_up=True, **kwargs):
        """Plot image at band and x, y indices if supplied."""
        data = self.img.sel(band=band, y=slice(*y), x=slice(*x))

        # Defaults
        size = kwargs.pop("size", 5)
        vmin = kwargs.pop("vmin", 0)
        title = kwargs.pop("title", f"{self.basename}")
        cmap = kwargs.pop("cmap", "inferno")
        cbarlabel = kwargs.pop("cbarlabel", "Radiance [$W/m^2/sr/um$]")

        # Coarsen data for quicker plot
        if len(data.y) > 2000:
            data = data.sel(y=slice(None, None, len(data.y) // 1000))

        # Plot
        ax = data.plot(vmin=vmin, size=size, cmap=cmap, **kwargs)
        ax.axes.set_title(title, fontsize=10)
        ax.axes.set_aspect("equal")
        ax.colorbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Flip image if collected on descending orbit
        if north_up and self.metadata.get("orbit_direction", "").lower() == "descending":
            ax.axes.yaxis.set_inverted(True)
            ax.axes.xaxis.set_inverted(True)
        return ax
