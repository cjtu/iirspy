import hashlib
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pdr
import rioxarray
import xarray as xr
import xesmf as xe
from osgeo import gdal
from pyproj import CRS
from rasterio.control import GroundControlPoint
from scipy.interpolate import RBFInterpolator, make_interp_spline
from scipy.stats import norm

## Constants
# Projections used in the Ch2 IIRS selenoref tool https://doi.org/10.1007/s12524-024-01814-4
IIRS_PROJ_DICT = {
    "equatorial": 'PROJCS["Moon_Equidistant_Cylindrical",GEOGCS["Moon 2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Greenwich",0],UNIT["Decimal_Degree",0.0174532925199433]],PROJECTION["Equidistant_Cylindrical"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Standard_Parallel_1",0],UNIT["Meter",1]]',
    "polarstereographicsouthpole": 'PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon 2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Greenwich",0],UNIT["Decimal_Degree",0.0174532925199433]],PROJECTION["Stereographic"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Scale_Factor",1],PARAMETER["Latitude_Of_Origin",-90],UNIT["Meter",1]]',
    "polarstereographicnorthpole": 'PROJCS["Moon_North_Pole_Stereographic",GEOGCS["Moon 2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Greenwich",0],UNIT["Decimal_Degree",0.0174532925199433]],PROJECTION["Stereographic"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Scale_Factor",1],PARAMETER["Latitude_Of_Origin",90],UNIT["Meter",1]]',
}
# Statistical polish from Verma et al. (2022) - see https://github.com/prabhakaralok/CH2IIRS
POLISH = np.concatenate([
    [np.nan] * 7,
    [
        1.007951631,
        1.003136498,
        1.012137177,
        0.983221552,
        0.982301846,
        0.980570808,
        0.974865433,
        0.97640194,
        0.992019454,
        1.007100398,
        1.004167929,
        0.986444903,
        1.005016461,
        1.013553818,
        1.009490688,
        1.003308071,
        1.011969658,
        1.0141375,
        1.006193945,
        1.015513554,
        1.020218579,
        1.016652203,
        1.023330972,
        1.020211072,
        1.015149832,
        1.00860599,
        1.000767149,
        0.994761146,
        1.067891991,
        1.044841918,
        0.972851837,
        0.9762951,
        0.979046208,
        0.97102683,
        0.982455709,
        0.987001059,
        0.988436954,
        0.979160136,
        0.9986847,
        0.997752521,
        0.998447812,
        1.016854934,
        0.997900542,
        0.983831379,
        0.969376425,
        0.979670198,
        0.973878228,
        0.996289037,
        1.014184498,
        1.01477613,
        1.001481496,
        0.975778394,
        0.997684295,
        0.984763646,
        0.983395789,
        0.98823384,
        0.988012215,
        0.952374697,
        0.937581808,
        1.013480796,
        1.014063354,
        1.022099753,
        1.03204087,
        1.035484491,
        1.041175268,
        1.036634478,
        1.034566162,
        1.03809732,
        1.046558655,
        1.021576737,
        1.028588801,
        1.007904219,
        0.972531419,
        0.956629793,
        0.955287872,
        0.979925337,
        1.002466257,
        1.014420813,
        1.008463966,
        1.00003099,
        1.008471522,
        0.99282371,
        0.99980302,
        0.987547313,
        0.984068212,
        0.992388063,
        0.988139868,
        1.012650312,
        1.052139241,
        1.012472306,
        0.965437548,
        0.939243701,
        0.967441753,
        0.963544372,
        0.990952989,
        0.973117569,
        1.013248635,
        1.020203633,
        1.029157186,
        1.020485688,
        1.020579985,
        1.006238104,
        1.014835128,
        1.006275303,
        1.010898129,
        1.00045314,
        1.008028897,
        1.012421421,
        1.027533403,
        1.031189861,
        1.029572772,
        0.998242,
        0.988472694,
        0.964883574,
        0.965888213,
        0.949632716,
        0.959468989,
        0.953967103,
        0.97185828,
        0.975099523,
        0.99858193,
        0.998573727,
        1.021937907,
        1.019142949,
        1.038073792,
        1.031876174,
        1.041953094,
        1.029414494,
        1.031461328,
        1.008333598,
        1.0158236,
        0.997626672,
        0.997772942,
        0.951297807,
        0.98848544,
        0.983055255,
        0.973287938,
        1.020004362,
        1.007683124,
        1.002357289,
        1.009943767,
        0.967795252,
        0.999227534,
        0.97305723,
        0.984516543,
        0.996874775,
        1.05340332,
        1.037099582,
        1.015145219,
        1.02252679,
        0.987330432,
        0.896931034,
        0.854160633,
        1.141899078,
        1.108682789,
        1.023030659,
        0.939513645,
        0.914993685,
        0.994857247,
        1.264567511,
        0.759581027,
        1.008047196,
        1.012150301,
        0.962843991,
        1.019548525,
        0.989332932,
        1.049050719,
        0.980318607,
        1.018721793,
        0.965286873,
        1.027176253,
        0.963709485,
        1.010598964,
        0.971110269,
        1.076940686,
        1.020255436,
        0.929940419,
        0.996612177,
        0.958811048,
        1.059205186,
        1.002448358,
        0.973559634,
        1.002937935,
        1.013683258,
        1.0474072,
        1.000925241,
        0.985500261,
        0.982660401,
        0.938587583,
        1.046900212,
        0.952743318,
        1.026649622,
        1.115011575,
        0.887761742,
        1.005459078,
        0.95483364,
        0.967275555,
        1.13852458,
        0.986343909,
        0.965651948,
        1.101161236,
        0.988116073,
        0.853138055,
        0.974375472,
        1.177442497,
        0.927739407,
        0.932486397,
        0.962300997,
        0.985181241,
        0.829206405,
        1.493934047,
        1.059590538,
        0.71322689,
        0.875132631,
        1.121072252,
        0.938541303,
        1.139668914,
        0.897165185,
        1.117762208,
        0.961511703,
        0.528956345,
        1.711570255,
        1.20749058,
        1.024243553,
        0.713165588,
        1.06875679,
        0.570841199,
        0.932178556,
        2.546499369,
        2.664596812,
        1.224978705,
        0.567434034,
        0.726464718,
        0.745085856,
        1.21801428,
        1.140071493,
        0.828417838,
        1.063662997,
        0.953993832,
        1.084346492,
        0.826805177,
        1.166156321,
        0.873581953,
        0.906603941,
        1.403382254,
        0.898905495,
        1.006518779,
        np.nan,
        np.nan,
    ],
])
FSOLAR = "./data/ch2_iirs_solar_flux.txt"


## Reflectance corr
def iirs_refl(  # noqa: C901 `iirs_refl` is too complex
    fqub,
    fspm,
    fgeom="",
    fout="",
    fflux=FSOLAR,
    dem=None,
    extent=(None, None, None, None),
    smoothing="none",
    swindow=1,
    polish=False,
    drop_bad_bands=True,
    ychunks=4000,
    destripe=False,
    **destripe_kws,
):
    """
    Convert IIRS L1 radiance to L2 reflectance using I/f folmula:

    F = foc / (π*d^2)
    R = I / (μo * F)

    Where foc is the spectral solar irradiance convolved with the relative
    response function of IIRS from PRADAN (ch2_iirs_solar_flux.txt), d is the
    solar distance (AU), μo is the cosine of the solar incidence angle, and I
    is the L1 IIRS spectral radiance.

    Optionally reads GCPs from metadata and returns a geotif
    (note: low accuracy near the poles).
    Optionally smooths spectra with a boxcar filter along wavelength.
    Optionally applies fourier smoothing to remove vertical/horizontal stripes.
    TODO: Optionally computes a thermal correction for the NIR bands.
    TODO: Optionally computes the incidence angle relative to a given DEM.

    Parameters
    ----------
    fqub (str or path): IIRS L1 radiance ".qub" file.
    fspm (str or path): IIRS sun parameter ".spm" file.
    fgeom (str or path): IIRS geometry ".csv" file.
    fout (str): Path to write reflectance image (.img for ENVI, .tif for geotiff).
    fflux (str): Path to IIRS solar flux input file.
    extent(tuple): (xmin, xmax, ymin, ymax) in degrees if fgeom is given, otherwise in pixels.
    smoothing (int): Smooth spectra ('none', 'boxcar', 'gaussian').
    swindow (int): Window size for smoothing (must be odd for gaussian).
    polish (bool): Apply IIRS spectral polish (Verma et al. 2022)
    drop_bad_bands (bool): Set bad bands defined in the IIRS SIS to NaN.
    destripe (bool): Whether to apply fourier destriping routine.
    destripe_kws (dict): Options for destriping (see fourier_filter).
    ychunks (dict): How many lines to chunk input image into using dask.
    """
    # If fgeom is given, interpret extent as degrees, otherwise interpret as pixels
    if fgeom:
        gridlon, gridlat, xyext = geom2grid(fgeom, extent)
        xmin, xmax, ymin, ymax = xyext
    else:
        xmin, xmax, ymin, ymax = extent

    # Read and unscale IIRS L1 radiance from [1000 mW/cm^2/sr/um] -> [W/m^2/sr/um]
    da = 0.01 * xr.open_dataarray(fqub, engine="rasterio").sel(x=slice(xmin, xmax), y=slice(ymin, ymax))
    if ychunks:
        da = da.chunk(y=ychunks, x=len(da.x), band=len(da.band))

    # Polish: Apply statistical polish used in Verma et al. 2022
    if polish:
        coeff = POLISH[:, None, None]
        da /= coeff

    # Drop bad bands: Sets bad bands to nan (bad bands based on gain,exposure)
    if drop_bad_bands:
        bb = get_bad_bands(fqub)[:, None, None]
        da = da.where(bb)

    # Reflectance correction: I/F
    # Get solar spectrum
    L = pd.read_csv(fflux, sep="\t", header=None, names=["wl", "flux"])
    wl = L["wl"].values[:, None, None] * 1e-9  # wl [m]
    sdist = get_solar_distance(fqub)
    F = L["flux"].values[:, None, None] * 10 / (np.pi * sdist**2)  # Solar flux [W/m^2/sr/um]

    # Get solar incidence for each line of image
    inc, iaz = get_iirs_inc_az(fqub, fspm, yrange=(int(da.y[0]), int(da.y[-1]) + 1))
    cos_inc = np.cos(np.radians(inc[None, :, None]))
    if dem is not None:
        # Find angle between sun and pixel in DEM
        elev = 90 - inc[None, :, None]
        cos_inc = dem.dot(elev)
        print("DEM not implemented")

    # Convert to I/F reflectance
    da = da / (cos_inc * F)
    da = da.assign_coords(wl=("band", wl.squeeze() * 1e9))  # Attach wavelengths

    # Destriping: Fourier filtering for vertical / horizontal artefacts
    if destripe:
        # TODO: check for off by one error (lines shifted up 1 pixel)
        # TODO: test on dask chunked images
        da = fourier_filter(da, **destripe_kws)

    # Smoothing:
    if smoothing.lower() == "boxcar":
        # Moving average over wavelength (test this since .mean on a rolling array may not handle NaNs correctly)
        da = da.rolling({"band": swindow}, center=True, min_periods=swindow / 2).mean(["band"])
    elif smoothing.lower() == "gaussian":
        # make a gaussian length of swindow, normalize, apply to rolling window as dot product
        # note: this interpolates NaN to allow spectra to be smooth up to a NaN band
        #  we then need to reapply the NaNs at the end. More steps and more expensive than moving avg
        if not swindow % 2:
            raise ValueError("Gaussian swindow must be odd.")  # noqa: TRY003
        gaussian = norm(loc=0, scale=1).pdf(np.arange(swindow) - swindow // 2)
        weights = xr.DataArray(gaussian / gaussian.sum(), dims=["window"])
        da = (
            da.rolling({"band": swindow}, center=True, min_periods=1)
            .construct("window")
            .interpolate_na("band")
            .dot(weights)
            .where(~da.isnull())
        )

    # Remove bad values, reduce precision for writing
    da = da.where(da >= 0).astype("float32")

    # Write image
    if fout:
        if fout[-4:].lower() == ".img":
            write_envi(da, fout)
        elif fgeom and fout[-4:].lower() == ".tif":
            da = warp2grid(da, xyext, gridlon, gridlat)
            da.rio.to_raster(fout, driver="GTiff", compress="LZW")
        else:
            da.rio.to_raster(fout)
    return da


def iirs_refl_verma(da, inc=None, smoothing=3, fflux=FSOLAR):
    """
    Return IIRS L2 reflectance from L1 radiance.

    Corrects for incidence angle and thermal tail (Verma et al., 2022).
    See CH2IIRS QGIS plugin (https://github.com/prabhakaralok/CH2IIRS).

    Parameters
    ----------
    da (xr.DataArray): IIRS L1 radiance
    inc (np.array): Solar incidence angle from get_iirs_inc
    smoothing (int): Window size for boxcar average smoothing along wavelength
    fflux (str): Path to IIRS solar flux input file.
    """
    # Unscale IIRS L1 radiance from [1000 mW/cm^2/sr/um] -> [W/m^2/sr/um]
    da = da * 0.01

    # Get solar spectrum
    L = pd.read_csv(fflux, sep="\t", header=None, names=["wl", "flux"])
    wl = L["wl"].values[:, None, None] * 1e-9  # wl [m]
    ss = L["flux"].values[:, None, None] * 10 / 3.14  # Solar flux [W/m^2/sr/um]
    ss = np.round(ss, 4)

    # Get statistical polish
    coeff = POLISH[:, None, None]

    # Isothermal temperature correction
    c = 3e8
    h = 6.626e-34
    k = 1.38e-23
    ε = 0.95  # emissivity

    # Computes BT assuming ε=0.95, takes T=mean(BT), removes εbbr(T)
    # Only use wavelengths from 4500 to 4875 for T estimation
    q = np.log(ε * 2 * h * c**2 * 1e-6 / (da * wl**5) + 1)
    bt = h * c / (wl * k * q)
    btavg = bt.isel(band=slice(224, 246)).mean("band")
    bbr = 1e-6 * (2 * h * c * c / wl**5) * 1 / (np.exp(h * c / (wl * k * btavg.values)) - 1)
    da = (da - ε * bbr) / ss
    da = da.clip(0) / coeff

    # Apply incidence angle correction if inc array is given
    da = da / (np.cos(np.radians(inc[None, :, None])))

    # Moving average over wavelength
    if smoothing > 0:
        da = da.rolling({"band": smoothing}, center=True).mean(["band"])

    # Attach wls
    da = da.assign_coords(wl=("band", wl.squeeze() * 1e9))
    return da


def get_reflectance_factor(fimg, fspm):
    """Return I/f reflectance from radiance and solar irradiance."""
    # Get cinc, format along lines / y direction assuming (band, y, x)
    inc, _ = get_iirs_inc_az(fimg, fspm)
    cinc = np.cos(np.radians(inc))[None, :, None]

    # Get solar flux scaled by sdist and pi along band direction
    sdist = get_solar_distance(fimg)
    f_scaled = get_iirs_solar_flux(sdist, fimg)["flux"][:, None, None]
    return 1 / (cinc * f_scaled)


def get_iirs_inc_az(fimg, fspm, yrange=(None, None)):
    """
    Return incidence angle from IIRS SPM timestamped file.

    Computes spacecraft clock time for each line and interpolates from SPM.
    """
    get_sun_elev = get_spm_interpolator(fspm, "sun_elev")
    get_sun_az = get_spm_interpolator(fspm, "sun_az")
    line_times = get_line_times(fimg)
    inc = 90 - get_sun_elev(line_times)[yrange[0], yrange[1]]
    az = get_sun_az(line_times)[yrange[0], yrange[1]]
    return inc, az


def get_spm_interpolator(fspm, col="sun_elev"):
    """Return interpolator f(timestamp) = spm[col] for a column in fspm."""
    spm = load_iirs_spm(fspm)
    spm_times = spm["timestamp"].values
    spm_vals = spm[col].values

    # Return linear interpolator function (linear: k=1, will extrapolate)
    return make_interp_spline(spm_times, spm_vals, k=1)


# Georeferencing
def iirs2geotiff(
    fqub,
    fgeom,
    fspm=None,
    fout=None,
    extent=(None, None, None, None),
    md5checksum=False,
    chunks=None,
    reflectance=False,
):
    """
    Return iirs xarray subset to latlon and projected on grid from geom.
    Optionally write to geotiff.

    Parameters
    ----------
    fqub: str
        Path to input radiance ch2_iir_... .qub file.
    fgeom: str
        Path to input geometry ch2_iir_... .csv file.
    fout: str or None
        Path to output geotiff. If None, return xarray.
    extent: tuple
        Extent in degrees (minlon, maxlon, minlat, maxlat).
    md5checksum: bool
        Check MD5 checksum of input files against xml label.
    bav: bool
        Bands as variables. Preserves wavelength labels in geotiff. Is slow!
    """
    if md5checksum:
        checksum(fgeom)
        checksum(fqub)
    if chunks is None:
        chunks = {"y": 4000}

    src = xr.open_dataarray(fqub, engine="rasterio", chunks=chunks)
    if reflectance:
        ref_factor = get_reflectance_factor(fqub, fspm, len(src.y))
        src = src * ref_factor

    # Subset to extent
    gridlon, gridlat, ext = geom2grid(fgeom, extent)
    ds = warp2grid(src, ext, gridlon, gridlat)
    if fout:
        ds.rio.to_raster(fout)
        print(f"Wrote {fout}")

    return ds


def warp2grid(da, ext, gridlon, gridlat, method="bilinear"):
    """Warp da to supplied grid and write crs."""
    src = da.sel(x=slice(ext[0] - 0.5, ext[1] + 0.5), y=slice(ext[2] - 0.5, ext[3] + 0.5))
    src = src.assign_coords(lon=(["y", "x"], gridlon), lat=(["y", "x"], gridlat))

    # Generate output grid
    minlon, maxlon = np.min(gridlon), np.max(gridlon)
    minlat, maxlat = np.min(gridlat), np.max(gridlat)
    dlat = (maxlat - minlat) / len(src.y)
    dlon = (maxlon - minlon) / len(src.x)
    target_grid = {
        "lon": np.arange(minlon, maxlon + dlon, dlon, dtype="float32"),
        "lat": np.arange(minlat, maxlat + dlat, dlat, "float32"),
    }
    regridder = xe.Regridder(src, target_grid, method, unmapped_to_nan=True)
    out = regridder(src)

    # Write crs (Moon unprojected)
    out.rio.write_crs(CRS.from_authority("IAU", "30100"), inplace=True)
    out.rio.set_spatial_dims("lon", "lat", inplace=True)
    out.rio.write_coordinate_system(inplace=True)
    return out


def fix_metadata(fqub, fgeotiff):
    """Fix metadata for geotiff to match original qub."""
    ds = rioxarray.open_rasterio(fgeotiff, band_as_variable=True)

    ds.attrs["name"] = "radiance"  # 'reflectance'
    ds.attrs["longname"] = "radiance (µW/cm^2/sr/µm)"  # 'reflectance'
    # ds.attrs['lines'] = ds.sizes['lat']
    # ds.attrs['samples'] = ds.sizes['lon']

    # Assign wavelength labels
    wls = get_wls(fqub, as_str=True)
    ds = ds.rename({f"band_{i + 1}": wl for i, wl in enumerate(wls)})

    # for band in ds.band.values:
    #     # TODO: Compute and store band stats?
    #     ds[band].attrs['x'] = 'y'

    pass


def get_iirs_proj(fqub):
    """Return the projection as WKT from the qub xml metadata."""
    img = pdr.open(fqub)
    proj_name = img.metaget("isda:projection") or "equatorial"
    pole = img.metaget("isda:area") or ""
    proj_name = f"{proj_name}{pole}".lower().replace(" ", "")
    return IIRS_PROJ_DICT[proj_name]


def geom2grid(fgeom, extent):
    """Interpolate sparse geometry csv to full 2D lat and lon grids."""
    df_buf, xy_ext = parse_geom(fgeom, extent)

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
    pixel_range = np.arange(xy_ext[0], xy_ext[1] + 1, 1)
    scan_range = np.arange(xy_ext[2], xy_ext[3] + 1, 1)
    grid_pixel, grid_scan = np.meshgrid(pixel_range, scan_range)

    # Create the 2D interpolated grids of lon and lat
    gridlon = rbf_lon(np.column_stack([grid_pixel.ravel(), grid_scan.ravel()])).reshape(grid_pixel.shape)
    gridlat = rbf_lat(np.column_stack([grid_pixel.ravel(), grid_scan.ravel()])).reshape(grid_pixel.shape)
    return gridlon, gridlat, xy_ext


def iirs2gcps(fqub, fgeom, fout=None, extent=(None, None, None, None)):
    """
    Write iirs qub subset to latlon as geotiff with GCPs. Optionally check MD5 checksum.

    Parameters
    ----------
    extent: (minlon, maxlon, minlat, maxlat)
    """
    # Get GCPs and ext in pixel coordinates
    gcps, ext = parse_geom(fgeom, extent, as_gcps=True)

    # Read the hyperspectral data cube
    ds = xr.open_dataarray(fqub, engine="rasterio").sel(
        x=slice(ext[0] - 0.5, ext[1] + 0.5), y=slice(ext[2] - 0.5, ext[3] + 0.5)
    )

    # Set the GCPs and CRS (Moon unprojected)
    ds.rio.write_gcps(gcps, CRS.from_authority("IAU", "30100"), inplace=True)
    if fout is not None:
        ds.rio.to_raster(fout, dtype="float32")
        print(f"Wrote {fout}")
    return ds


def points2gcps(fqub, fpoints, fout, extent=(None, None, None, None)):
    """Attach tie-points from a GIS .points file to fqub."""
    # TODO
    pass


## File I/O
def unzip(fzip, ddir):
    """Unzip a file to a directory."""
    with zipfile.ZipFile(fzip, "r") as zip_ref:
        zip_ref.extractall(ddir)


def get_iirs_paths(ddir, exts=("qub", "hdr", "xml", "csv", "xml-csv", "lbr", "oat", "oath", "spm")):
    """Return a list of paths to IIRS image, geom, and misc files."""
    out = {}
    for ext in exts:
        subdir = "."
        if ext in ("hdr", "qub", "xml"):
            subdir = "data"
        elif ext in ("csv", "xml-csv"):
            subdir = "geometry"
        else:
            subdir = "miscellaneous"
        out[ext] = {f.stem.split("_")[3]: f for f in Path(ddir).glob(f'{subdir}/**/*.{ext.split("-")[0]}')}
    out["imgs"] = list(out["qub"].keys())
    return out


## Checksums
class ChecksumError(Exception):
    """Exception raised for checksum mismatches."""

    def __init__(self, fname):
        self.message = f"Checksum failed for {fname}."
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
    if pdr.open(fname).metaget("md5_checksum") != get_md5(fname):
        raise ChecksumError(fname)


def get_wls(fname, as_str=False):
    """Return the wavelength labels from the image metadata."""
    img = pdr.open(fname)
    wl_dict = list(img.metaget("Band_Bin_Set").values())
    wls = [float(wl["center_wavelength"]) for wl in wl_dict]
    if as_str:
        return [f"{wl} nm" for wl in wls]
    return wls


def parse_geom(fgeom, extent=(None, None, None, None), buffer=0, center=True, as_gcps=False):
    """
    Read lat,lon,x,y from iirs geometry file and return as list of GCPs or DataFrame.

    Subset to minlon, maxlon, minlat, maxlat, and optionally a wider buffer [deg].
    """
    minlon, maxlon, minlat, maxlat = extent

    # Read GCPs from iirs geometry file
    df = pd.read_csv(fgeom)
    df["Longitude"] = (df["Longitude"] + 180) % 360 - 180  # lon in [-180, 180]

    # Use pixel centers
    if center:
        df["Pixel"] = df["Pixel"] + 0.5
        df["Scan"] = df["Scan"] + 0.5

    # Subset to extent. None => no limit
    minlon = -180 if minlon is None else minlon
    maxlon = 180 if maxlon is None else maxlon
    minlat = -90 if minlat is None else minlat
    maxlat = 90 if maxlat is None else maxlat
    df = df[
        (df["Longitude"] >= minlon - buffer)
        & (df["Longitude"] <= maxlon + buffer)
        & (df["Latitude"] >= minlat - buffer)
        & (df["Latitude"] <= maxlat + buffer)
    ]
    # Add 0.5 to pixel and scan to get the pixel center
    ext = [min(df["Pixel"]), max(df["Pixel"]), min(df["Scan"]), max(df["Scan"])]
    if as_gcps:
        gcps = [
            GroundControlPoint(row["Scan"], row["Pixel"], row["Longitude"], row["Latitude"]) for i, row in df.iterrows()
        ]
        return gcps, ext
    return df, ext


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
    df["timestamp"] = df["datetime"].astype(int) / 1e9  # Equiv to .timestamp(), but faster
    # df['timestamp'] = df['datetime'].apply(lambda x: x.timestamp())  # Slow
    return df


def get_line_times(fimg):
    """Return start and stop times from IIRS metadata."""
    img = pdr.open(fimg)
    tstart = pd.to_datetime(img.metaget("start_date_time")).timestamp()
    _, lines, _ = get_iirs_shape_meta(fimg)
    dt = float(img.metaget("isda:line_exposure_duration")) / 1000  # [ms]->[s]
    orbit_dir = img.metaget("isda:orbit_limb_direction").lower()  # Ascending or Descending

    # Note: clock not precise - sometimes nlines != (tstop - tstart) / dt
    # For this reason, don't do np.arange(tstart, tstop+dt, dt) nor linspace(tstart, tstop, nlines)
    # tstop = pd.to_datetime(img.metaget('stop_date_time')).timestamp()

    # Data collection time is bottom up for ascending orbit, top down for descending
    line_times = tstart + dt * np.arange(lines)
    if orbit_dir == "Ascending":
        line_times = line_times[::-1]
    return line_times


def get_iirs_shape_meta(fimg):
    """Return the shape of fimg from metadata (bands, lines, samples)."""
    axs = pdr.open(fimg).metaget("Array_3D_Spectrum").getall("Axis_Array")
    bands, lines, samples = (int(axs[i]["elements"]) for i in range(3))
    return bands, lines, samples


def get_iirs_solar_flux(sdist=1, fflux="../data/moon/ch2/iirs/iir/miscellaneous/ch2_iirs_solar_flux.txt"):
    """Return the solar flux from the IIRS solar flux csv."""
    df = pd.read_csv(fflux, sep="\t", header=None, names=["wl", "flux"])
    df.loc[:, "flux"] = df.flux / 3.14 / sdist**2
    return df


def load_bad_bands(fbad_bands="../data/moon/ch2/iirs/iirs_bad_bands.csv"):
    """Load bad bands from a CSV file."""
    return pd.read_csv(fbad_bands, index_col=0).astype(bool)


def get_exposore_gain(fimg):
    """Return exposure (E1-E4) and gain (G2) as eXgY string."""
    img = pdr.open(fimg)
    exposure = img.metaget("isda:exposure")
    gain = img.metaget("isda:gain")
    return f"{exposure}{gain}"


def get_bad_bands(fimg):
    """Return bad bands for the given image (depends on exposure and gain)."""
    exp_gain = get_exposore_gain(fimg)  # e.g. 'e1g2
    bad_bands = load_bad_bands()
    return bad_bands[exp_gain].values  # np array band 1-256


def get_solar_distance(fimg):
    """Return the solar distance from the IIRS metadata."""
    # TODO: Do properly - needs spice. doesn't seem to be in the metadata
    fimg = Path(fimg)
    if "20201226T1745264921" in fimg.stem:
        return 0.9855
    elif "20210122T0920157625" in fimg.stem:
        return 0.9849
    elif "20210719T1622353775" in fimg.stem:
        return 1.0174
    elif "20210622T1850441449" in fimg.stem or "20210622T1454378054" in fimg.stem or "20210622T1256344234" in fimg.stem:
        return 1.0184
    return 1


def write_envi(da, fout):
    """Write dataarray cube to fout."""
    wls = [f"{wl:.3f} nm" for wl in da.wl]
    nz, ny, nx = da.shape
    dst = gdal.GetDriverByName("ENVI").Create(fout, nx, ny, nz, gdal.GDT_Float32)
    for i in range(nz):
        db = dst.GetRasterBand(i + 1)
        db.SetDescription(wls[i])
        db.WriteArray(da.isel(band=i).values)


## Thermal corr
# TODO


## Image smoothing
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
    cv2.fillPoly(mask, [left_triangle], 0)
    cv2.fillPoly(mask, [right_triangle], 0)
    cv2.fillPoly(mask, [top_triangle], 0)
    cv2.fillPoly(mask, [bottom_triangle], 0)

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


if __name__ == "__main__":  # pragma: no cover
    pass
