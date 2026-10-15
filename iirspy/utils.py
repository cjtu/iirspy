import hashlib
import re
import warnings
from importlib import metadata
from importlib.resources import files
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdr

# import xarray_regrid
import xarray as xr

# import xesmf as xe
from rasterio.control import GroundControlPoint
from scipy.interpolate import RBFInterpolator, make_interp_spline
from scipy.signal import savgol_filter
from scipy.stats import norm

warnings.filterwarnings("ignore", message="Dataset has no geotransform")

## Constants
PKG_DATA = files("iirspy").joinpath("data")
DCALIB = PKG_DATA.joinpath("iir/calibration")
CHUNKSIZE = 128 * 2**20  # [B] dask y-chunk target (dask's array.chunk-size default; profiled fastest + lowest RAM)
FPOLISH = str(PKG_DATA.joinpath("spectral_polish_verma2022.csv"))
# Per-band polish inferred from published ISSDC L2 (NDI) products; bands without a fit are 1.0
FPOLISH_ISSDC_EMPIRICAL = str(PKG_DATA.joinpath("spectral_polish_issdc_empirical.csv"))
# NOAA SSI CDR v03 monthly irradiance convolved to IIRS bands [mW/cm^2/um at 1 AU]: Coddington, O., Lean,
# J. L., Lindholm, C., Pilewskie, P. (2024). NOAA Climate Data Record (CDR) of NASA NOAA LASP Spectral
# Solar Irradiance (NNLSSI), Version 3. Monthly SSI, 2010-01 to 2025-12. NOAA National Centers for
# Environmental Information. https://doi.org/10.25921/esjz-1w61. Accessed 2026-10-09.
FNOAA = str(PKG_DATA.joinpath("solar_flux_noaa_cdr_v03.csv"))
FBADBANDS = str(PKG_DATA.joinpath("iirs_bad_bands.csv"))
FBADPIXELS = str(PKG_DATA.joinpath("bad_pixel_mask.csv"))
FSOLAR = str(PKG_DATA.joinpath("iir/miscellaneous/ch2_iirs_solar_flux.txt"))
FWAVELENGTHS = str(PKG_DATA.joinpath("iir/miscellaneous/ch2_iirs_wavelength.csv"))
# Projections used in the Ch2 IIRS selenoref tool https://doi.org/10.1007/s12524-024-01814-4
IIRS_PROJ_DICT = {
    "equatorial": 'PROJCS["Moon_Equidistant_Cylindrical",GEOGCS["Moon 2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Greenwich",0],UNIT["Decimal_Degree",0.0174532925199433]],PROJECTION["Equidistant_Cylindrical"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Standard_Parallel_1",0],UNIT["Meter",1]]',
    "polarstereographicsouthpole": 'PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon 2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Greenwich",0],UNIT["Decimal_Degree",0.0174532925199433]],PROJECTION["Stereographic"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Scale_Factor",1],PARAMETER["Latitude_Of_Origin",-90],UNIT["Meter",1]]',
    "polarstereographicnorthpole": 'PROJCS["Moon_North_Pole_Stereographic",GEOGCS["Moon 2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Greenwich",0],UNIT["Decimal_Degree",0.0174532925199433]],PROJECTION["Stereographic"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Scale_Factor",1],PARAMETER["Latitude_Of_Origin",90],UNIT["Meter",1]]',
}
# Convert data level to "Mission-Type-Camera" identifier
LVL2MTC = {0: "nri", 1: "nci", 2: "ndi"}
OSF = (*range(29, 35), *range(69, 76), *range(162, 172))  # Order sorting filters
INVALID = (*range(1, 7), *range(252, 257))  # Invalid band list
# IIRS L1 radiance is stored in [1000 mW/cm^2/sr/um]; multiply to get physical [W/m^2/sr/um].
RAD_NATIVE_SCALE = 0.01  # [1000 mW/cm^2/sr/um] -> [W/m^2/sr/um]
E1_EXPOSURE_MS = 1.0  # exposure duration the e1g2 gain LUT was measured at (see get_gain_offset)
AU_KM = 1.495978707e8
try:
    IIRSPY_VERSION = metadata.version("iirspy")
except metadata.PackageNotFoundError:  # running from a source tree not pip-installed
    IIRSPY_VERSION = "0+source"


## Reflectance corr
def get_noaa_flux(sdist=1.0, month=None, fflux=FNOAA):
    """NOAA SSI CDR v03 solar flux [W/m^2/sr/um] at `sdist` AU on band 1..256 (see `FNOAA`).

    `month` is "YYYY-MM" for that month's irradiance; None uses the 2010-2019 mean.
    """
    df = pd.read_csv(fflux)
    col = "mean_2010_2019" if month is None else month
    if col not in df:
        raise ValueError(f"{fflux} has no {col} column")
    flux = df[col].to_numpy() * 10 / (np.pi * sdist**2)
    return xr.DataArray(flux, coords={"band": df["band"].to_numpy()}, name="Solar flux [W/m^2/sr/um]")


def flat_surface_temperature(inc, sdist=1.0, s0=1370.0, eps=0.95, a0=0.08, a=0.045, b=0.14):
    """Radiative-equilibrium dayside temperature [K] of a flat surface at solar incidence `inc` [deg].

    T = [S (1 - A) cos(i) / (eps sigma)]^(1/4), Williams et al. (2017) Eq. 1, with the incidence-dependent
    albedo A(i) = A0 + a (i/45)^3 + b (i/90)^8 of Vasavada et al. (2012); the defaults are Williams' fit to
    the Diviner zonal-mean noon temperatures (where i = latitude). S = `s0` / `sdist`^2. 0 K where i >= 90.

    Williams, J.-P., Paige, D. A., Greenhagen, B. T., & Sefton-Nash, E. (2017). The global surface temperatures
    of the Moon as measured by the Diviner Lunar Radiometer Experiment. Icarus, 283, 300-325.
    Vasavada, A. R., et al. (2012). Lunar equatorial surface temperatures and regolith properties from the
    Diviner Lunar Radiometer Experiment. JGR Planets, 117, E00H18.

    >>> tuple(round(float(flat_surface_temperature(i)), 1) for i in (0, 60, 90))
    (391.1, 318.4, 0.0)
    """
    albedo = a0 + a * (inc / 45) ** 3 + b * (inc / 90) ** 8
    cos_i = np.clip(np.cos(np.radians(inc)), 0, None)
    return (s0 / sdist**2 * (1 - albedo) * cos_i / (eps * 5.670374e-8)) ** 0.25


def thermal_radiance(rad, eps=1.0, bands=(227, 247), inc=None, sdist=1.0, t_min=0.0):
    """(eps * B(T), T) for an isothermal surface, T [K] per pixel.

    T is the mean brightness temperature at emissivity `eps` over IIRS band numbers `bands`
    (1-indexed, inclusive), selected by band number. `rad` is (band, y, x) [W/m^2/sr/um] with a
    `wl` [nm] coord. Where T is not above `t_min` (or not finite) and `inc` [deg] is given, T falls
    back to `flat_surface_temperature(inc, sdist)`. A cube with none of `bands` has no thermal
    term: (0.0, None).
    """
    tb = rad.sel(band=slice(*bands))
    if not tb.sizes["band"]:
        warnings.warn(f"no thermal bands {bands} in the cube, thermal removal skipped", UserWarning, stacklevel=2)
        return 0.0, None
    c, h, k = 3e8, 6.626e-34, 1.38e-23
    wl_t = tb.wl * 1e-9
    q = np.log(eps * 2 * h * c**2 * 1e-6 / (tb * wl_t**5) + 1)
    t = (h * c / (wl_t * k * q)).mean("band")
    if inc is not None:
        t = t.where(t > t_min, flat_surface_temperature(inc, sdist))
    wl = rad.wl * 1e-9
    bbr = 1e-6 * (2 * h * c * c / wl**5) * 1 / (np.exp(h * c / (wl * k * t)) - 1)
    return eps * bbr, t


def spectral_polish(band, fpolish=FPOLISH):
    """Per-band polish divisor from csv `fpolish` (`band`, `polish` columns), mapped onto `band` by
    band number; bands the table lacks get 1.0."""
    df = pd.read_csv(fpolish)
    band = np.asarray(band)
    coeff = pd.Series(df["polish"].to_numpy(), index=df["band"].to_numpy()).reindex(band).fillna(1.0)
    return xr.DataArray(coeff.to_numpy(), coords={"band": band})


def moving_average(da, width=3, edges="pair"):
    """Centred `width`-band moving average along band.

    `edges` sets the end bands: "pair" (width 3 only, the Verma et al. 2022 processing): the first
    band is the mean of bands 2-3 and the last the mean of the last two; "nan" leaves them NaN;
    "reflect" pads by reflection.
    """
    if width <= 1:
        return da
    if edges == "reflect":
        n = width // 2
        padded = da.pad(band=n, mode="reflect").rolling(band=width, center=True).mean()
        return padded.isel(band=slice(n, -n)).assign_coords(da.band.coords)
    out = da.rolling(band=width, center=True).mean()
    if edges == "pair":
        if width != 3:
            raise ValueError('edges="pair" is only defined for width 3')
        b = da.band
        first, last = da.isel(band=[1, 2]).mean("band"), da.isel(band=[-2, -1]).mean("band")
        out = xr.where(b == b[0], first, xr.where(b == b[-1], last, out)).transpose(*da.dims)
    return out


def apply_smoothing(da, smoothing, swindow):
    """Smooth the spectra using boxcar or Gaussian smoothing."""
    if smoothing.lower() == "boxcar":
        # Moving average over wavelength (test this since .mean on a rolling array may not handle NaNs correctly)
        da = da.rolling({"band": swindow}, center=True, min_periods=swindow / 2).mean(["band"])
    elif smoothing.lower() == "gaussian":
        # make a gaussian length of swindow, normalize, apply to rolling window as dot product
        # note: this interpolates NaN to allow spectra to be smooth up to a NaN band
        #  we then need to reapply the NaNs at the end. More steps and more expensive than moving avg
        if not swindow % 2:
            raise ValueError("Gaussian swindow must be odd.")
        gaussian = norm(loc=0, scale=1).pdf(np.arange(swindow) - swindow // 2)
        weights = xr.DataArray(gaussian / gaussian.sum(), dims=["window"])
        da = (
            da
            .rolling({"band": swindow}, center=True, min_periods=1)
            .construct("window")
            .interpolate_na("band")
            .dot(weights)
            .where(~da.isnull())
        )
    elif smoothing.lower() == "savgol":
        da = smooth_savgol(da, swindow)  # Assumes polyorder=2
    return da


def get_iirs_inc_az(fimg, fspm, yrange):
    """
    Return incidence angle from IIRS SPM timestamped file.

    Computes spacecraft clock time for each line and interpolates from SPM.
    """
    ymin, ymax = yrange
    get_sun_elev = get_spm_interpolator(fspm, "sun_elev")
    get_sun_az = get_spm_interpolator(fspm, "sun_az")
    line_times = get_line_times(fimg)
    inc = 90 - abs(get_sun_elev(line_times)[ymin:ymax])
    az = get_sun_az(line_times)[ymin:ymax]
    return inc, az


def get_spm_interpolator(fspm, col="sun_elev"):
    """Return interpolator f(timestamp) = spm[col] for a column in fspm."""
    spm = load_iirs_spm(fspm)
    spm_times = spm["timestamp"].values
    spm_vals = spm[col].values

    # Return linear interpolator function (linear: k=1, will extrapolate)
    return make_interp_spline(spm_times, spm_vals, k=1)


def geom2grid(fgeom, extent, xs=None, ys=None):
    """Interpolate sparse geometry csv to full 2D lat and lon grids."""
    df_buf, xyext = parse_geom(fgeom, extent)

    # Add buffer to the extent to ensure edges are included in interpolator
    # NOTE: Makes almost no difference (pixel lvl offset, overall offset much larger)
    # df_buf, _ = parse_geom(fgeom, extent, buffer=0.5)

    # Create thin plate spline interpolators
    points = df_buf[["Pixel", "Scan"]].values
    lons = df_buf["Longitude"].values
    lats = df_buf["Latitude"].values
    rbf_lon = RBFInterpolator(points, lons, kernel="thin_plate_spline")
    rbf_lat = RBFInterpolator(points, lats, kernel="thin_plate_spline")

    # Create a 2D grid of all pixel x and y values in extent
    pixel_range = np.arange(xyext[0], xyext[1] + 1, 1)
    scan_range = np.arange(xyext[2], xyext[3] + 1, 1)
    if xs is not None:
        pixel_range = xs
    if ys is not None:
        scan_range = ys
    grid_pixel, grid_scan = np.meshgrid(pixel_range, scan_range)

    # Create the 2D interpolated grids of lon and lat
    gridlon = rbf_lon(np.column_stack([grid_pixel.ravel(), grid_scan.ravel()])).reshape(grid_pixel.shape)
    gridlat = rbf_lat(np.column_stack([grid_pixel.ravel(), grid_scan.ravel()])).reshape(grid_pixel.shape)
    return gridlon, gridlat, xyext


def geom2latlon_coords(df_geom, xyext, nx, ny):
    """
    Convert geometry coordinates to latitude/longitude coordinates using thin plate spline
    interpolation.

    Parameters
    ----------
    df_geom : pandas.DataFrame
        DataFrame containing geometry from parse_geom(). Has columns:
        - "Pixel": pixel coordinate values
        - "Scan": scan coordinate values
        - "Longitude": longitude values in degrees
        - "Latitude": latitude values in degrees
    xyext : array-like
        Extent specification as [x_min, x_max, y_min, y_max] defining the pixel
        and scan ranges for which to interpolate coordinates.
    nx : int
        Number of pixels in the x (pixel) direction.
    ny : int
        Number of pixels in the y (scan) direction.

    Returns
    -------
    lon_1d : numpy.ndarray
        1D array of interpolated longitude values with length nx.
    lat_1d : numpy.ndarray
        1D array of interpolated latitude values with length ny.
    """

    # Create thin plate spline interpolators
    points = df_geom[["Pixel", "Scan"]].values
    lons = df_geom["Longitude"].values
    lats = df_geom["Latitude"].values

    rbf_lon = RBFInterpolator(points, lons, kernel="thin_plate_spline")
    rbf_lat = RBFInterpolator(points, lats, kernel="thin_plate_spline")

    # Create 1D coordinate arrays
    pixel_range = np.linspace(xyext[0], xyext[1], nx)
    scan_range = np.linspace(xyext[2], xyext[3], ny)

    # Evaluate lon along the first scan line (constant y)
    lon_1d = rbf_lon(np.column_stack([pixel_range, np.full(nx, scan_range[0])]))

    # Evaluate lat along the first pixel line (constant x)
    lat_1d = rbf_lat(np.column_stack([np.full(ny, pixel_range[0]), scan_range]))

    return lon_1d, lat_1d


## File I/O


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


def read_geom_csv(fgeom) -> pd.DataFrame:
    """IIRS geometry csv, minus the (0, 0) fill rows some bundles pad the tail with.

    The fill rows have Pixel=Scan=0 and Longitude=Latitude=0.0 exactly, duplicating the real Scan=0
    row; no real sample lands on (0.0, 0.0) to full float precision.
    """
    df = pd.read_csv(fgeom)
    return df[(df.Longitude != 0.0) | (df.Latitude != 0.0)]


def get_wls(fwavelengths=FWAVELENGTHS):
    """Return the wavelength [nm] for each band from IIRS wavelengths file."""
    df = pd.read_csv(fwavelengths, header=0, names=["band", "wl"], usecols=[0, 1])
    return df["wl"].values


def get_wls_xml(fname):
    """Return the wavelength [nm] for each band from image metadata."""
    img = pdr.open(fname)
    wl_dict = list(img.metaget("Band_Bin_Set").values())
    return np.array([float(wl["center_wavelength"]) for wl in wl_dict])


def get_wl_labels(fname):
    """Return labels for wavelegnths in str format, e.g. for an ENVI header."""
    wls = get_wls(fname)
    return [f"{wl} nm" for wl in wls]


def parse_geom(
    fgeom,
    latlonextent=(None, None, None, None),
    xyextent=(None, None, None, None),
    buffer=0,
    center=False,
    as_gcps=False,
    corners_only=False,
):
    """
    Read lat,lon,x,y from iirs geometry file and return as list of GCPs or DataFrame.

    Subset to specified extent, snapping to the nearest larger bounding box of GCPs.

    Parameters
    ----------
    fgeom : str
        Path to IIRS geometry CSV file.
    latlonextent : tuple
        Extent in (minlon, maxlon, minlat, maxlat) format. Cannot be used with xyextent.
    xyextent : tuple
        Extent in (minx, maxx, miny, maxy) format (Pixel, Pixel, Scan, Scan).
        Cannot be used with latlonextent.
    buffer : int
        Additional buffer to add around extent (default: 0).
        Number of GCP grid steps to expand beyond the snapped boundary.
    center : bool
        If True, use pixel centers (adds 0.5 to Pixel and Scan coordinates).
    as_gcps : bool
        If True, return list of GroundControlPoint objects. Otherwise return DataFrame.
    corners_only : bool
        If True and as_gcps=True, return only corner GCPs.

    Returns
    -------
    out : list of GroundControlPoint or DataFrame
        Subset of GCPs/data within the extent.
    xyext : list
        Bounding box in pixel coordinates [minx, maxx, miny, maxy].

    Examples
    --------
    >>> # Filter by lat/lon, snapping to nearest GCP grid
    >>> gcps, xyext = parse_geom(fgeom, latlonextent=(-10, 10, 20, 40))  # doctest: +SKIP

    >>> # Filter by pixel coordinates, snapping to nearest GCP grid
    >>> # If xyextent is (101, 200, 499, 601) and GCPs are every 50,
    >>> # returns GCPs with x from 100 to 200 and y from 450 to 650
    >>> gcps, xyext = parse_geom(fgeom, xyextent=(101, 200, 499, 601), as_gcps=True)  # doctest: +SKIP
    """
    # Check that only one extent type is provided
    latlon_given = any(e is not None for e in latlonextent)
    xy_given = any(e is not None for e in xyextent)

    if latlon_given and xy_given:
        raise ValueError("Only one of latlonextent or xyextent can be specified, not both.")

    # Read GCPs from iirs geometry file
    df = read_geom_csv(fgeom)
    df["Longitude"] = (df["Longitude"] + 180) % 360 - 180  # lon in [-180, 180]

    # Use pixel centers if requested
    if center:
        df["Pixel"] = df["Pixel"] + 0.5
        df["Scan"] = df["Scan"] + 0.5

    # Filter by extent with snapping to nearest larger bounding box
    if latlon_given:
        minlon, maxlon, minlat, maxlat = latlonextent
        out = _snap_to_gcp_grid(
            df,
            dim1_col="Longitude",
            dim2_col="Latitude",
            dim1_min=minlon,
            dim1_max=maxlon,
            dim2_min=minlat,
            dim2_max=maxlat,
            dim1_default=(-180, 180),
            dim2_default=(-90, 90),
            buffer=buffer,
        )
    elif xy_given:
        minx, maxx, miny, maxy = xyextent
        out = _snap_to_gcp_grid(
            df,
            dim1_col="Pixel",
            dim2_col="Scan",
            dim1_min=minx,
            dim1_max=maxx,
            dim2_min=miny,
            dim2_max=maxy,
            dim1_default=(None, None),
            dim2_default=(None, None),
            buffer=buffer,
        )
    else:
        # No extent specified, use all data
        out = df

    # Get the pixel coordinate bounding box
    if len(out) == 0:
        raise ValueError("No GCPs found within the specified extent.")

    xyext = [min(out["Pixel"]), max(out["Pixel"]), min(out["Scan"]), max(out["Scan"])]

    # Filter to corners only if requested
    if corners_only:
        corner_pixels = {xyext[0], xyext[1]}
        corner_scans = {xyext[2], xyext[3]}
        out = out[out["Pixel"].isin(corner_pixels) & out["Scan"].isin(corner_scans)]

    # Convert to GCPs if requested
    if as_gcps:
        out = [
            GroundControlPoint(row["Scan"], row["Pixel"], row["Longitude"], row["Latitude"])
            for _, row in out.iterrows()
        ]

    return out, xyext


def _snap_to_gcp_grid(
    df, dim1_col, dim2_col, dim1_min, dim1_max, dim2_min, dim2_max, dim1_default, dim2_default, buffer=0
):
    """
    Helper function to snap extent to nearest larger bounding box of GCPs.

    Parameters
    ----------
    df : DataFrame
        GCP dataframe with columns for both dimensions.
    dim1_col, dim2_col : str
        Column names for the two dimensions (e.g., "Pixel"/"Scan" or "Longitude"/"Latitude").
    dim1_min, dim1_max, dim2_min, dim2_max : float or None
        Min/max values for filtering.
    dim1_default, dim2_default : tuple
        Default (min, max) values if None is provided.
    buffer : int
        Number of GCP grid steps to expand beyond the snapped boundary.

    Returns
    -------
    DataFrame
        Filtered dataframe with GCPs in the snapped bounding box.
    """
    # Get unique values for both dimensions
    dim1_vals = sorted(df[dim1_col].unique())
    dim2_vals = sorted(df[dim2_col].unique())

    # Helper function to find snapped boundary
    def snap_min(val, vals, default):
        if val is None:
            return default if default is not None else min(vals)
        return max((v for v in vals if v <= val), default=min(vals))

    def snap_max(val, vals, default):
        if val is None:
            return default if default is not None else max(vals)
        return min((v for v in vals if v >= val), default=max(vals))

    # Snap to nearest GCP boundaries
    dim1_min_snap = snap_min(dim1_min, dim1_vals, dim1_default[0])
    dim1_max_snap = snap_max(dim1_max, dim1_vals, dim1_default[1])
    dim2_min_snap = snap_min(dim2_min, dim2_vals, dim2_default[0])
    dim2_max_snap = snap_max(dim2_max, dim2_vals, dim2_default[1])

    # Apply buffer if specified
    if buffer > 0:
        dim1_min_idx = dim1_vals.index(dim1_min_snap)
        dim1_max_idx = dim1_vals.index(dim1_max_snap)
        dim2_min_idx = dim2_vals.index(dim2_min_snap)
        dim2_max_idx = dim2_vals.index(dim2_max_snap)

        dim1_min_snap = dim1_vals[max(0, dim1_min_idx - buffer)]
        dim1_max_snap = dim1_vals[min(len(dim1_vals) - 1, dim1_max_idx + buffer)]
        dim2_min_snap = dim2_vals[max(0, dim2_min_idx - buffer)]
        dim2_max_snap = dim2_vals[min(len(dim2_vals) - 1, dim2_max_idx + buffer)]

    # Filter dataframe to snapped extent
    out = df[
        (df[dim1_col] >= dim1_min_snap)
        & (df[dim1_col] <= dim1_max_snap)
        & (df[dim2_col] >= dim2_min_snap)
        & (df[dim2_col] <= dim2_max_snap)
    ]

    return out


def load_iirs_spm(fspm):
    """Parse IIRS spm file. Add datetime and timestamp columns."""
    colnames = [
        "type",
        "row",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "millisecond",
        "scx",
        "scy",
        "scz",
        "scvx",
        "scvy",
        "scvz",
        "phase",
        "sun_aspect",
        "sun_az",
        "sun_elev",
    ]
    df = pd.read_csv(fspm, sep="\\s+", header=None, usecols=range(0, 19), names=colnames)
    df["year"] = df["year"].astype(str).str.slice(3, None)
    df["datetime"] = pd.to_datetime(df.iloc[:, 2:9])
    # Conversion to epoch time (like .timestamp() but faster)
    df["timestamp"] = df["datetime"].astype("datetime64[ns]").astype("int64") / 1e9
    # df['timestamp'] = df['datetime'].apply(lambda x: x.timestamp())  # Slow
    return df


def get_line_times(fimg):
    """Return start and stop times from IIRS metadata."""
    img = pdr.open(fimg)
    tstart = pd.to_datetime(img.metaget("start_date_time")).timestamp()
    _, lines, _ = get_iirs_shape_meta(fimg)
    dt = float(img.metaget("isda:line_exposure_duration")) / 1000  # [ms]->[s]

    # Note: clock not precise - sometimes nlines != (tstop - tstart) / dt
    # For this reason, don't do np.arange(tstart, tstop+dt, dt) nor linspace(tstart, tstop, nlines)
    # tstop = pd.to_datetime(img.metaget('stop_date_time')).timestamp()

    # Note: Data collection time follows spacecraft collect direction
    line_times = tstart + dt * np.arange(lines)
    return line_times


def scan_utc(flabel, scans):
    """UTC (ISO, SPICE-ready) of each absolute Scan in `scans`, from `flabel`'s start time + line exposure.

    The one scan -> time mapping: the .spm is ~40 ms ancillary records, not one per ~53 ms scan line, so
    its row number is not a Scan.
    """
    t = get_line_times(flabel)[np.asarray(scans, dtype=int)]
    return [pd.Timestamp(v, unit="s").strftime("%Y-%m-%dT%H:%M:%S.%f") for v in np.atleast_1d(t)]


def get_iirs_shape_meta(fimg):
    """Return the shape of fimg from metadata (bands, lines, samples)."""
    axs = pdr.open(fimg).metaget("Array_3D_Spectrum").getall("Axis_Array")
    bands, lines, samples = (int(axs[i]["elements"]) for i in range(3))
    return bands, lines, samples


def load_bad_bands(fbad_bands=FBADBANDS):
    """Load bad bands from a CSV file."""
    # Negates the df since provided bad bands file is 1 for good, 0 for bad
    return ~pd.read_csv(fbad_bands, index_col=0).astype(bool)


def load_bad_pixel_mask(fbad_pixels=FBADPIXELS):
    """Return the per-detector-element bad pixel mask as a (band, x) DataArray, True where bad."""
    mask = pd.read_csv(fbad_pixels, header=None).values.astype(bool)
    coords = {"band": 1 + np.arange(0, 256), "x": 0.5 + np.arange(250)}
    return xr.DataArray(mask, coords=coords, name="bad_pixel")


def load_reference_flat(fimg, calib_dir=DCALIB):
    """
    Return the packaged sensor flat (band, x) for fimg's exposure/gain, or None if not given.

    Prebuilt flat across multiple IIRS scenes (see iirspy.empirical.build_flat),
    used as the fallback when a scene has no qualifying flat region of its own.
    Named ch2_iirs_flat_<expgain>.csv
    """
    exp_gain = get_exposure_gain(fimg)  # e.g. "e1g2"
    fflat = Path(calib_dir) / f"ch2_iirs_flat_{exp_gain}.csv"
    if not fflat.exists():
        return None
    vals = np.loadtxt(fflat, delimiter=",").astype("float32")
    coords = {"band": 1 + np.arange(0, 256), "x": 0.5 + np.arange(250)}
    return xr.DataArray(vals, coords=coords, name="reference_flat")


def get_exposure_gain(fimg):
    """Return exposure (E1-E4) and gain (G2) as eXgY string."""
    img = pdr.open(fimg)
    exposure = img.metaget("isda:exposure")
    gain = img.metaget("isda:gain")
    return f"{exposure}{gain}".lower()


def get_bad_bands(fimg, buffer=0):
    """
    Return bad bands for the given image (depends on exposure and gain).

    Set buffer number of bands to left and right of each bad band as bad.
    """
    exp_gain = get_exposure_gain(fimg)  # e.g., "e1g2"
    bad_bands = load_bad_bands()[exp_gain].values  # array of True / False

    # Buffer - use kernel convolution to bump n adjacent bands, flag these as also bad
    # Ex. [0, 1, 1, 1, 0, 1, 1] with buffer 1 => [0, 0, 1, 0, 0, 0, 1]
    kernel_size = 2 * buffer + 1
    kernel = np.ones(kernel_size)
    bad_buffered = np.convolve((bad_bands).astype(int), kernel, mode="same").astype(bool)

    return bad_buffered  # array of True where band is bad


def get_lut_file(fimg, lut_type="lut_coeff", calib_dir=DCALIB):
    """
    Locate and return the correct LUT file for the given image.

    Parameters
    ----------
    fimg : str
        Path to the IIRS image file.
    lut_type : str
        Type of LUT to find (default: "lut_coeff").
        Options: "lut_coeff", "saturations_radiance"
    calib_dir : str or Path
        Directory containing calibration LUT files.
    ext : str
        File extension to look for (default: "csv").
        Options: "csv", "xml"

    Returns
    -------
    Path to the LUT file as a string.

    Raises
    ------
    FileNotFoundError if the LUT file is not found.
    """
    exp_gain = get_exposure_gain(fimg)  # e.g., "e1g2"
    pattern = f"ch2_iirs_cal_{exp_gain}_{lut_type}.csv"
    flut = Path(calib_dir) / pattern
    if flut.exists():
        return str(flut)
    raise FileNotFoundError(f"IIRS calibration file {flut} not found.")


def get_exposure_duration(fimg):
    """Return the commanded exposure duration [ms] from the IIRS label (e1 -> 1, e2 -> 3, ...).

    Not to be confused with isda:line_exposure_duration (the 53.06 ms line period, identical
    across exposure settings).
    """
    return float(pdr.open(fimg).metaget("isda:exposure_duration"))


def get_gain_offset(fimg, denoise=False, gain_z=None, offset_z=None, calib_dir=DCALIB):
    """
    Return the IIRS gain and offset for fimg as DataArrays, in [mW/cm^2/sr/um] per DN.

    Only ch2_iirs_cal_e1g2_lut_coeff.csv is a real calibration. The e2g2/e3g2/e4g2 tables are
    placeholders: gain ~ 1.0 at every (band, x) with no spectral structure at all, ~360x flatter
    across band than the true response. Used as shipped they put radiance ~1000x high, every pixel
    trips the saturation cut in calibrate_to_rad, and the whole cube is nulled.

    ISSDC's own L1 product does not use them either. Differencing the e2g2 scene
    20201203T1859574285 against its nci L1 gives L1 = 333.6 +/- 3.8 * gain_e1g2 * DN over bands
    5-20 (a flat ratio, so the e1g2 spectral AND cross-track shape is what ISSDC applied), i.e.
    exactly gain_e1g2 / 3 in this module's units - and that scene's label reads
    isda:exposure_duration = 3 ms against e1's 1 ms. So this function always loads the e1g2 table
    and scales gain by E1_EXPOSURE_MS / exposure_duration - a no-op for e1g2 itself (1 ms).

    Note the e2g2/e3g2/e4g2 saturations_radiance tables sit ~3x above the resulting full-scale
    radiance, so the saturation cut cannot fire for those scenes (safe, but not a real check).

    Parameters
    ----------
    flut : str
        Path to the lookup table file.
    denoise: bool
        Replace speckly noise from gain / offset with NaN.
    """
    flut = str(Path(calib_dir) / "ch2_iirs_cal_e1g2_lut_coeff.csv")
    gain_scale = E1_EXPOSURE_MS / get_exposure_duration(fimg)
    if gain_z is None:
        gain_z = 0.01  # Empirically selected z-thresholds for gain, offset
    if offset_z is None:
        offset_z = 0.5
    lut = np.loadtxt(flut, delimiter=",").reshape((256, 250, 2)).astype("float32")
    lut[:, :, 0] *= gain_scale
    coords = {"band": 1 + np.arange(0, 256), "x": 0.5 + np.arange(250)}
    gain = xr.DataArray(lut[:, :, 0], coords=coords, name="gain")
    off = xr.DataArray(lut[:, :, 1], coords=coords, name="offset")
    if denoise:
        # Outlier detection with moving window and Z-score threshold
        gwindow = gain.rolling({"band": 1, "x": 5}, center=True, min_periods=1)
        gresid = gain - gwindow.median()
        gzscore = abs(gresid / gresid.std())
        gain = gain.where(gzscore < gain_z)

        owindow = off.rolling({"band": 1, "x": 5}, center=True, min_periods=1)
        oresid = off - owindow.median()
        ozscore = abs(oresid / oresid.std())
        off = off.where(ozscore < offset_z)
    return gain, off


def get_saturation_radiance(fimg, calib_dir=DCALIB):
    """Return IIRS saturation radiance file as 1D DataArray along band."""
    flut = get_lut_file(fimg, "saturations_radiance", calib_dir)
    lut = np.loadtxt(flut, delimiter=",", usecols=2).astype("float32")  # band, wl, saturation [1000 mW/cm^2/sr/um]
    return RAD_NATIVE_SCALE * xr.DataArray(lut, coords={"band": np.arange(1, 257)}, name="saturation [W/m^2/sr/um]")


def get_solar_flux(sdist=1.0, fflux=FSOLAR):
    """Return IIRS solar flux as DataArray along band."""
    L = np.loadtxt(fflux, delimiter="\t", usecols=1)
    flux = L * 10 / (np.pi * sdist**2)  # [W/m^2/sr/um]
    return xr.DataArray(flux, coords={"band": np.arange(1, 257)}, name="Solar flux [W/m^2/sr/um]")


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


def get_solar_distance(fimg, kernels=None):
    """Return the Sun-Moon distance in AU at the scene's mid-line epoch, from SPICE.

    `kernels=None` resolves the kernel set via `spice_kernels(day)`. Warns and returns 1.0 AU if
    the label or kernels can't be found.
    """
    import spiceypy as sp

    fimg = Path(fimg)
    try:
        line_times = get_line_times(fimg)
        t = pd.Timestamp(line_times[len(line_times) // 2], unit="s")
        ks = kernels if kernels is not None else spice_kernels(t.strftime("%Y%m%d"))
        for k in ks:
            sp.furnsh(str(k))
        et = sp.str2et(t.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        v, _ = sp.spkpos("SUN", et, "IAU_MOON", "LT+S", "MOON")
    except Exception as e:
        warnings.warn(f"get_solar_distance: could not compute from SPICE for {fimg} ({e}); using 1.0 AU", stacklevel=2)
        return 1.0
    return float(np.linalg.norm(v) / AU_KM)


## ENVI BIL streaming writer
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


## Checksums
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


## Image smoothing
def detect_stripes(da, sigma_threshold=1):
    """
    Detect vertical stripes for each individual band in a DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array with dimensions ('band', 'y', 'x').
    sigma_threshold : float, optional
        The number of standard deviations to use as the threshold for detecting
        anomalous pixels. Higher values are more conservative (default: 1).

    Returns
    -------
    xarray.DataArray
        Bool DataArray with dimensions ('band', 'x') indicating if is a stripe.
    """
    # Fraction of anomalous pixels to be considered a stripe, in terms of sigma
    #  E.g. 68% for sigma=1, 95% for sigma=2, 99.7% for sigma=3, etc.
    frac = norm().cdf(sigma_threshold) - norm().cdf(-sigma_threshold)
    thresh = frac * len(da.y)  # threshold in num pixels

    # Apply ufunc is hard to parse - applies the stripe detection band-wise
    #   I.e. `for band in da: detect_stripes_2D(da[band])`
    #   but with some fancy vectorizaiton and dask-delayed compute magic
    out = xr.apply_ufunc(
        detect_stripes_2D,
        da,
        thresh,
        input_core_dims=[["y", "x"], []],
        output_core_dims=[["x"]],
        exclude_dims={"y"},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype],
    )
    return out


def detect_stripes_2D(data, threshold):
    """
    Detect stripe along lines (y) in a 2D image (e.g. a single band from a data cube).

    Defines a stripe as a line with more than thresh pixels lower or higher
    than both adjacent pixels (e.g. a hot or cold line).

    Parameters
    ----------
    data : 2D array
        Numpy array image [lines (y), samples (x)]
    thresh : num
        Fraction of anomalous pixels to consider a line a stripe.

    References
    ----------
    Yokoya, N., Miyamura, N., & Iwasaki, A. (2010). Preprocessing of
        hyperspectral imagery with consideration of smile and keystone
        properties. In Multispectral, Hyperspectral, and Ultraspectral Remote
        Sensing Technology, Techniques, and Applications III (Vol. 7857,
        pp. 73-81). SPIE.

    """
    # Make a left and right 2D image to compare with vector math
    #  (duplicate left and right edge for simplicity)
    data_l = np.insert(data[:, :-1], 0, data[:, 0], axis=1)
    data_r = np.insert(data[:, 1:], -1, data[:, -1], axis=1)
    # Number anomalous (larger or smaller than both neighbours)
    num_anom = np.sum(((data < data_l) & (data < data_r)) | ((data > data_l) & (data > data_r)), axis=0)
    is_stripe_x = num_anom > threshold  # 1D bool array along samples (x)
    return is_stripe_x


def fourier_filter(img, vthresh=0.8, vtilt=0.0, hthresh=0.0, htilt=0.0, get_filt_at_band=None):
    """
    Filters linear features from an img cube in ftt domain.

    Creates a mask in fft domain that interpolates vertical (v) and horizontal
    (h) features (e.g., stripes, dead pixel rows, etc).

    The vthresh and hthresh control how aggressive the filter is:
    - v_span=0.1 will slightly fade vertical features identified
    - v_span=0.9 will aggerssively fade vertical features (real signal may blur)

    The htilt and vtilt control the sublinearity of the features.
    For example:
    - htilt=0 will only target perfectly horizontal stripes
    - htilt=0.9 will capture sub-horiontal features (real signal may blur)

    In practice, balance of the span and depth for horizonal and vertical
    lines will be needed. The parameters depend on the image size and the
    nature of the artefacts.

    It can be helpful to show the shape of the noise in the fft domain.
    Use get_filt_at_band with all other params 0 to get the fft for
    that band (plot with plt.imshow). Vertical striping will appear as a
    bright ray from the center to the left/right in the fft domain, while
    horizontal striping will appear as bright rays towards the top/bottom.
    Tweak params and plot until the mask covers those bright rays.

    See Suárez-Valencia (2024) ESS (https://doi.org/10.1029/2023EA003464)

    Parameters
    ----------
    img (np.array, xr.DataArray)
        Input image (band, y, x)
    vthresh (float, 0-1)
        How aggressively vertical features are filtered (default 0.8).
    vtilt (float, 0-1)
        How vertical (0) or sub-vertical (up to 1) of features to filter (default 0).
    hthresh (float, 0-1)
        How aggressively horizontal features are filtered (default 0.8).
    htilt (float, 0-1)
        How horizontal (0) or sub-horizontal (up to 1) of features to filter (default 0).
    get_filt_at_band (int or None)
        Return the masked fft domain image at the given band number.
    """
    # Mask triangles parameters
    y, x = img.shape[1:3]
    cy, cx = (y // 2, x // 2)

    # Mask triangles. Note h/v transposed in phase space (v:left/right, h:up/down)
    h_mask_base = int(cy * (vtilt))
    h_mask_height = int(cx * (1 - vthresh))
    v_mask_base = int(cx * (htilt))
    v_mask_height = int(cy * (1 - hthresh))

    # Left triangle vertices (top left, point towards center, bottom left)
    left_triangle = np.array([[0, cy - h_mask_base], [cx - h_mask_height, cy], [0, cy + h_mask_base]])

    # Right triangle vertices (top right, point towards center, bottom right)
    right_triangle = np.array([[x, cy - h_mask_base], [cx + h_mask_height, cy], [x, cy + h_mask_base]])

    # Top triangle vertices (top left, point towards center, top right)
    top_triangle = np.array([[cx - v_mask_base, 0], [cx, cy - v_mask_height], [cx + v_mask_base, 0]])

    # Bottom triangle vertices (bottom left, point towards center, bottom right)
    bottom_triangle = np.array([[cx - v_mask_base, y], [cx, cy + v_mask_height], [cx + v_mask_base, y]])

    # Draw all triangles
    mask = np.ones((y, x, 2))
    cv2.fillPoly(mask, [left_triangle.astype(np.int32)], (0, 0))
    cv2.fillPoly(mask, [right_triangle.astype(np.int32)], (0, 0))
    cv2.fillPoly(mask, [top_triangle.astype(np.int32)], (0, 0))
    cv2.fillPoly(mask, [bottom_triangle.astype(np.int32)], (0, 0))

    if get_filt_at_band is not None:
        img = img[get_filt_at_band, :, :].data
        fft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        fft_shift = np.fft.fftshift(fft)
        masked_fft = fft_shift * mask
        # Scale for plotting
        mfft = 20 * np.log(cv2.magnitude(masked_fft[:, :, 0], masked_fft[:, :, 1]))
        return mfft

    out = img.copy()
    for band in range(img.shape[0]):
        # if np.isnan(np.sum(img[band,:,:])):
        # continue
        # Fourier transform
        fft = cv2.dft(img[band, :, :].data, flags=cv2.DFT_COMPLEX_OUTPUT)
        fft_shift = np.fft.fftshift(fft)
        # Apply mask
        masked_fft = fft_shift * mask
        # Inverse Fourier transform
        ifft_shift = np.fft.ifftshift(masked_fft)
        ifft = cv2.idft(ifft_shift) / (y * x)
        out[band, :, :] = cv2.magnitude(ifft[:, :, 0], ifft[:, :, 1])

    # Preserve original non-data regions
    imgmask = img != 0
    return out * imgmask


def smooth_savgol(data: xr.DataArray, savgol_window: int = 9, savgol_polyorder: int = 2) -> xr.DataArray:
    """
    Savitzky-Golay filtering for spectral smoothing. Must have no NaNs.
    """

    def filter_spectrum(spectrum):
        """Apply Savitzky-Golay to a single spectrum."""
        # needed for dask
        spectrum = np.array(spectrum, copy=True)
        # TODO (optional): Can remove spikes here using method from https://doi.org/10.1029/2024JE008842

        # Savitzky-Golay
        spectrum = savgol_filter(spectrum, window_length=savgol_window, polyorder=savgol_polyorder)
        return spectrum

    # Apply combined filter
    result: xr.DataArray = xr.apply_ufunc(
        filter_spectrum,
        data,
        input_core_dims=[["band"]],
        output_core_dims=[["band"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    return result


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
