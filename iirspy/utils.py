import hashlib
import re
import warnings
import zipfile
from importlib.resources import files
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pdr
import rasterio

# import xarray_regrid
import xarray as xr

# import xesmf as xe
from pyproj import CRS
from rasterio.control import GroundControlPoint
from scipy.interpolate import RBFInterpolator, make_interp_spline
from scipy.ndimage import convolve
from scipy.stats import norm

warnings.filterwarnings("ignore", message="Dataset has no geotransform")

## Constants
PKG_DATA = files("iirspy").joinpath("data")
DCALIB = PKG_DATA.joinpath("iir/calibration")
CHUNKSIZE = 200e6  # [MB] chunk large images into this size with dask to fit in RAM (typically between 100MB-1GB)
FPOLISH = str(PKG_DATA.joinpath("spectral_polish_verma2022.csv"))
FBADBANDS = str(PKG_DATA.joinpath("iirs_bad_bands.csv"))
FSOLAR = str(PKG_DATA.joinpath("ch2_iirs_solar_flux.txt"))
# Projections used in the Ch2 IIRS selenoref tool https://doi.org/10.1007/s12524-024-01814-4
IIRS_PROJ_DICT = {
    "equatorial": 'PROJCS["Moon_Equidistant_Cylindrical",GEOGCS["Moon 2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Greenwich",0],UNIT["Decimal_Degree",0.0174532925199433]],PROJECTION["Equidistant_Cylindrical"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Standard_Parallel_1",0],UNIT["Meter",1]]',
    "polarstereographicsouthpole": 'PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon 2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Greenwich",0],UNIT["Decimal_Degree",0.0174532925199433]],PROJECTION["Stereographic"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Scale_Factor",1],PARAMETER["Latitude_Of_Origin",-90],UNIT["Meter",1]]',
    "polarstereographicnorthpole": 'PROJCS["Moon_North_Pole_Stereographic",GEOGCS["Moon 2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Greenwich",0],UNIT["Decimal_Degree",0.0174532925199433]],PROJECTION["Stereographic"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Scale_Factor",1],PARAMETER["Latitude_Of_Origin",90],UNIT["Meter",1]]',
}
# Convert data level to "Mission-Type-Camera" identifier
LVL2MTC = {0: "nri", 1: "nci", 2: "ndi"}


## Reflectance corr
def iirs_refl(
    fqub,
    fspm,
    fgeom="",
    fout="",
    ftif="",
    md5checksum=False,
    fflux=FSOLAR,
    dem=None,
    extent=(None, None, None, None),
    yrange=(None, None),
    smoothing="none",
    swindow=1,
    polish=False,
    fpolish=FPOLISH,
    drop_bad_bands=True,
    bad_bands_buffer=0,
    bad_band_mask=None,
    thermal="verma",
    ychunks=4000,
    max_refl=np.inf,
    destripe=False,
    **destripe_kws,
):
    """
    Convert IIRS L1 radiance to L2 reflectance using I/f formula.
    """
    # Step 0: Ensure data was downloaded correctly
    if md5checksum:
        checksum(fqub)
        checksum(fspm)
        if fgeom:
            checksum(fgeom)

    # Step 1: Read and preprocess the input data
    da = preprocess_input_data(fqub, fgeom, fspm, ftif, extent, yrange, ychunks)

    # Step 2: Handle bad bands
    if drop_bad_bands:
        da = handle_bad_bands(da, fqub, bad_band_mask, bad_bands_buffer)

    # Step 3: Apply destriping if needed
    if destripe:
        da = apply_destriping(da, **destripe_kws)

    # Step 4: Perform reflectance correction
    da = apply_reflectance_correction(da, fqub, fflux, dem, thermal, max_refl)

    # Step 5: Apply optional spectral polish
    if polish:
        da = apply_spectral_polish(da, fpolish)

    # Step 6: Apply smoothing
    if smoothing.lower() != "none":
        da = apply_smoothing(da, smoothing, swindow)

    # Step 7: Write output if specified
    if fout:
        write_output(da, fout, fgeom, extent, ftif)

    return da


def preprocess_input_data(fqub, fgeom, fspm, ftif, extent, yrange, ychunks):
    """Read and preprocess input data."""
    # Read in dataarray. If tif, assume it is L1 radiance that is already cropped
    if ftif:
        da = xr.open_dataarray(ftif, engine="rasterio")
    else:
        # Subset da using xy coords from extent
        if fgeom:
            # when fgeom is given, assume extent is in lat/lon, parse to get the extent in xy
            _, extent = parse_geom(fgeom, extent)
        da = xr.open_dataarray(fqub, engine="rasterio").sel(
            x=slice(extent[0], extent[1]), y=slice(extent[2], extent[3])
        )

    if ychunks:
        da = da.chunk(y=ychunks, x=len(da.x), band=len(da.band))

    # Add geometry
    orbit_dir = pdr.open(fqub).metaget("isda:orbit_limb_direction").lower()  # Ascending or Descending
    yflip = False
    if orbit_dir == "ascending":
        yflip = False
    elif orbit_dir == "descending":
        yflip = True
    else:
        print("Unknown orbit, assuming ascending")
    if fgeom:
        # Ascending
        yoff = 51
        ymin, ymax = yrange
        if yflip:  # Descending
            yoff = -86
            ymin, ymax = yrange
        ys = np.linspace(ymin, ymax, len(da.y), endpoint=False) - yoff
        lon2d, lat2d, extent = geom2grid(fgeom, extent, xs=da.x, ys=ys)
        da = da.assign_coords(lon=(("y", "x"), lon2d))
        da = da.assign_coords(lat=(("y", "x"), lat2d))

    # Get solar incidence for each line of image
    inc, iaz = get_iirs_inc_az(fqub, fspm, yrange)
    if yflip:
        iaz -= 180
    da = da.assign_coords(inc=("y", inc))
    da = da.assign_coords(iaz=("y", iaz))

    # Adjust for direction of image collection
    if yflip:
        da.coords["y"] = min(da.y) - da.y

    # IIRS L1 radiance is in [1000 mW/cm^2/sr/um]
    da = 0.01 * da  # [1000 mW/cm^2/sr/um] -> [W/m^2/sr/um]
    return da


def handle_bad_bands(da, fqub, bad_band_mask=None, bad_bands_buffer=0):
    """Handle bad bands by setting them to NaN."""
    if bad_band_mask is None:
        bad_band_mask = get_bad_bands(fqub, bad_bands_buffer)  # array where True == bad
    da = da.where(~bad_band_mask[:, None, None])
    return da


def apply_destriping(da, **destripe_kws):
    """Apply Fourier destriping to the data."""
    da = fourier_filter(da, **destripe_kws)
    return da


def apply_reflectance_correction(da, fqub, fflux, dem, thermal, max_refl):
    """Perform reflectance correction using I/F formula."""
    # Set very low radiance pixels to NaN (TODO: this is an arbitrary choice of band/min value)
    da = da.where(da.sel(band=10) > 0.5)

    # Get solar spectrum and wavelength and assign to da TODO: move to import step
    L = pd.read_csv(fflux, sep="\t", header=None, names=["wl", "flux"])
    da = da.assign_coords(wl=("band", np.round(L["wl"].values, 3)))

    sdist = get_solar_distance(fqub)
    solar_flux = L["flux"].values * 10 / (np.pi * sdist**2)  # [W/m^2/sr/um]
    da = da.assign_coords(F=("band", solar_flux))

    # (Optional): adjust cos_inc relative to dem
    if dem is not None:
        if not hasattr(dem, "sel"):
            dem = xr.open_dataarray(dem, engine="rasterio").sel(band=1)
            if dem.shape == (2822, 2789):  # TODO: fix and delete
                dem.coords["y"] = -dem.y
        # dem = dem.rio.reproject_match(da.sel(band=1))
        # dem = dem.assign_coords(x=da.x, y=da.y)
        dem = dem.interp_like(da.isel(band=1), method="slinear")
        cos_inc = get_cos_inc_dem(dem, da.inc, da.iaz, da.lat)
    else:
        cos_inc = np.cos(np.radians(da.inc))
    cos_inc = cos_inc.where((cos_inc > 0.03) & (cos_inc < 0.999))

    # Get thermal component
    trad = 0
    if thermal == "verma":
        trad = get_thermal_rad_verma(da, da.wl * 1e-9)

    # Convert to I/F reflectance
    da = (da - trad) / (cos_inc * da.F)

    # Remove bad values
    da = da.where(da <= max_refl)

    return da


def apply_spectral_polish(da, fpolish):
    """Apply spectral polish to the reflectance data."""
    spec_polish = pd.read_csv(fpolish).set_index("band").to_xarray()
    da /= spec_polish.sel(band=slice(None, 101))
    return da


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
            da.rolling({"band": swindow}, center=True, min_periods=1)
            .construct("window")
            .interpolate_na("band")
            .dot(weights)
            .where(~da.isnull())
        )
    return da


def write_output(da, fout, fgeom, extent, ftif):
    """Write the processed data to the specified output file."""
    da.rio.write_crs("IAU_2015:30135", inplace=True)
    da.rio.write_nodata(np.nan, inplace=True)
    if fout[-4:].lower() == ".img":
        write_envi(da, fout)
    elif fgeom and fout[-4:].lower() == ".tif" and not ftif:
        gridlon, gridlat, xyext = geom2grid(fgeom, extent)
        da = warp2grid(da, xyext, gridlon, gridlat)
        da = da.swap_dims({"band": "wl"})
        da.rio.to_raster(fout, driver="GTiff", compress="LZW")
    else:
        da = da.swap_dims({"band": "wl"})
        da.rio.to_raster(fout)
        fix_metadata(fout, da.wl.values)


def iirs_refl_verma(da, inc=None, smoothing=3, fflux=FSOLAR, fpolish=FPOLISH):
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
    coeff = pd.read_csv(fpolish)["polish"].values[:, None, None]

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


def get_thermal_rad_verma(da, wl, eps=0.95, tbands=(224, 246)):
    """
    Return thermal radiance assuming isothermal surface as Verma et al. (2022).
    """
    c = 3e8
    h = 6.626e-34
    k = 1.38e-23

    # Computes BT assuming emissivity and a blackbody
    q = np.log(eps * 2 * h * c**2 * 1e-6 / (da * wl**5) + 1)
    bt = h * c / (wl * k * q)

    # Remove bbr(avg_BT) assuming surface is isothermal
    # Only use wavelengths within tbands for T estimation (default 4500 to 4875)
    btavg = bt.isel(band=slice(*tbands)).mean("band")
    bbr = 1e-6 * (2 * h * c * c / wl**5) * 1 / (np.exp(h * c / (wl * k * btavg)) - 1)
    return eps * bbr


def get_cos_inc_dem(dem, inc, iaz, latitudes):
    """
    Return cos(inc) relative to a DEM using the dot product, considering local latitude.

    Parameters
    ----------
    dem : xr.DataArray
        DEM data with the same spatial resolution as the image.
    inc : np.ndarray
        Solar incidence angle (degrees) for each row of data.
    iaz : np.ndarray
        Solar azimuth angle (degrees) for each row of data.
    latitudes : np.ndarray
        1D array of latitude values corresponding to each row of the DEM.

    Returns
    -------
    cos_inc : np.ndarray
        2D array of cos(inc) values adjusted for the DEM and local latitude.
    """
    # Calculate local illumination using dot product method between surface normal and sun vector

    res_x = abs(float(dem.x[1] - dem.x[0]))
    res_y = abs(float(dem.y[1] - dem.y[0]))

    # Calculate surface gradients using kernel convolution
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8 * res_x)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / (8 * res_y)

    # Calculate derivatives
    dz_dx = xr.apply_ufunc(
        convolve,
        dem,
        input_core_dims=[["y", "x"]],
        kwargs={"weights": kernel_x, "mode": "nearest"},
        output_core_dims=[["y", "x"]],
        vectorize=True,
    )
    dz_dy = xr.apply_ufunc(
        convolve,
        dem,
        input_core_dims=[["y", "x"]],
        kwargs={"weights": kernel_y, "mode": "nearest"},
        output_core_dims=[["y", "x"]],
        vectorize=True,
    )
    # Create surface normal vectors [nx, ny, nz]
    # For each pixel: n = [-dz/dx, -dz/dy, 1] (unnormalized)
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(dz_dx)

    # Normalize the normal vectors
    norm_factor = np.sqrt(nx**2 + ny**2 + nz**2)
    nx = nx / norm_factor
    ny = ny / norm_factor
    nz = nz / norm_factor

    # Convert incidence and azimuth to radians
    inc_rad = np.radians(inc)
    iaz_rad = np.radians(iaz)

    # Calculate sun vector components [sx, sy, sz] for each row
    sx = np.sin(inc_rad) * np.sin(iaz_rad)  # East component
    sy = np.sin(inc_rad) * np.cos(iaz_rad)  # North component
    sz = np.cos(inc_rad)  # Up component

    # Adjust sun vector for local latitude
    lat_rad = np.radians(latitudes)
    sy_adj = sy * np.cos(lat_rad) - sz * np.sin(lat_rad)
    sz_adj = sy * np.sin(lat_rad) + sz * np.cos(lat_rad)

    # Replace original sy and sz with adjusted values
    sy = sy_adj
    sz = sz_adj

    # Compute dot product between normal and sun vector for each pixel
    cos_inc = nx * sx + ny * sy + nz * sz

    # Debugging: Print intermediate values for testing
    if np.any(np.isnan(cos_inc)):
        print("NaN values detected in cos_inc. Check inputs and calculations.")
    if np.min(cos_inc) < 0:
        print("Negative cos_inc values detected. Verify sun vector and surface normal calculations.")

    # Handle negative values (local shadows)
    # cos_inc = np.clip(cos_inc, 0, 1)
    return cos_inc


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
    # TODO: need to replace regridder
    # regridder = xe.Regridder(src, target_grid, method, unmapped_to_nan=True)
    # out = regridder(src)
    out = target_grid

    # Write crs (Moon unprojected)
    out.rio.write_crs(CRS.from_authority("IAU", "30100"), inplace=True)
    out.rio.set_spatial_dims("lon", "lat", inplace=True)
    out.rio.write_coordinate_system(inplace=True)
    return out


def warp2gcps(fqub, da, gcps, gcps_crs, fout, method="bilinear"):
    """Warp and project the image based on the provided GCPs."""
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    # Create a temporary file to store the warped image
    with rasterio.open(fqub) as src:
        # Create a VRT dataset with GCPs
        vrt_options = {
            "crs": da.rio.crs,
            "src_crs": gcps_crs,
            "src_transform": rasterio.transform.from_gcps(gcps),
            "src_method": "GCP_TPS",
        }
        with rasterio.vrt.WarpedVRT(src, **vrt_options) as vrt:
            transform, width, height = calculate_default_transform(
                gcps_crs, da.rio.crs, da.rio.width, da.rio.height, gcps=gcps
            )
            print(transform, "\n", width, height)
            kwargs = da.rio.profile
            kwargs.update({"crs": gcps_crs, "transform": transform, "width": width, "height": height})

            with rasterio.open(fout, "w", **kwargs) as dst:
                for i in range(1, da.rio.count + 1):
                    reproject(
                        source=rasterio.band(vrt, i),
                        destination=rasterio.band(dst, i),
                        src_transform=da.rio.transform,
                        src_crs=da.rio.crs,
                        gcps=gcps,
                        dst_transform=transform,
                        dst_crs=gcps_crs,
                        resampling=getattr(Resampling, method),
                    )

        # Read the warped image back into an xarray DataArray
        warped_da = xr.open_dataarray(fout, engine="rasterio")

    return warped_da


def fix_metadata(fout, wls):
    """Fix metadata for geotiff to match original qub."""
    with rasterio.open(fout, "r+", driver="GTiff") as dst:
        dst.descriptions = tuple([str(wl) for wl in wls])

    # ds = rioxarray.open_rasterio(fout, band_as_variable=True, engine='rasterio')
    # ds.attrs["name"] = "radiance"  # 'reflectance'
    # ds.attrs["longname"] = "radiance (µW/cm^2/sr/µm)"  # 'reflectance'
    # # ds.attrs['lines'] = ds.sizes['lat']
    # # ds.attrs['samples'] = ds.sizes['lon']

    # # Assign wavelength labels
    # ds = ds.rename({f"band_{i + 1}": wl for i, wl in enumerate(wls)})
    # ds.rio.to_raster(fout)

    # for band in ds.band.values:
    #     # TODO: Compute and store band stats?
    #     ds[band].attrs['x'] = 'y'
    return


def get_iirs_proj(fqub):
    """Return the projection as WKT from the qub xml metadata."""
    img = pdr.open(fqub)
    proj_name = img.metaget("isda:projection") or "equatorial"
    pole = img.metaget("isda:area") or ""
    proj_name = f"{proj_name}{pole}".lower().replace(" ", "")
    return IIRS_PROJ_DICT[proj_name]


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


def iirs2gcps(fqub, fgeom, fout=None, extent=(None, None, None, None), corners_only=False):
    """
    Write iirs qub subset to latlon as geotiff with GCPs. Optionally check MD5 checksum.

    Parameters
    ----------
    extent: (minlon, maxlon, minlat, maxlat)
    """
    # Get GCPs and ext in pixel coordinates
    gcps, xyext = parse_geom(fgeom, extent, as_gcps=True, corners_only=corners_only)

    # Read the hyperspectral data cube
    ds = xr.open_dataarray(fqub, engine="rasterio").sel(
        x=slice(xyext[0] - 0.5, xyext[1] + 0.5), y=slice(xyext[2] - 0.5, xyext[3] + 0.5)
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
def read_gcps(fgcps):
    """Parse gcps and CRS from a QGIS georeferencer .points file."""
    with open(fgcps) as f:
        crs = CRS.from_wkt(f.readline().lstrip("#CRS: "))
    df = pd.read_csv(fgcps, skiprows=1, header=0)
    # df['sourceY'] = len(da.y) + df.sourceY
    # df['sourceY'] =  abs(df.sourceY)
    gcps = [GroundControlPoint(row["sourceX"], row["sourceY"], row["mapX"], row["mapY"]) for _, row in df.iterrows()]
    return gcps, crs


def unzip_iirs(ddir, basename, level, md5checksum=True):
    """Find zipped IIRS image from PRADAN ISSDC, unzip and run checksum."""
    mtc = LVL2MTC[level]
    f = next((f for f in Path(ddir).glob(f"**/*{mtc}*.zip") if basename in f.stem), None)
    if f is None:
        raise ValueError(f"Image {basename} not found in {ddir}. Please download from PRADAN or check file path.")
    with zipfile.ZipFile(f, "r") as zipf:
        print(f"Extracting {basename} to {f.parent}")
        zipf.extractall(f.parent)
    paths = get_iirs_paths(f.parent, level=level, basenames=[basename])
    if "qub" not in paths:
        raise RuntimeError("Unzip failed.")
    if checksum:
        print("Verifying unzipped image...", end=" ")
        checksum(paths["qub"][basename].as_posix())
        print("Success!")
    return paths


def iirsbasename(input_str):
    """Return the image basename from str_in (e.g., 20201226T1745264921)"""
    pattern = r"\d{8}T\d{10}"
    try:
        return re.search(pattern, input_str).group()
    except AttributeError as e:
        raise ValueError(f"Can't parse basename: {input_str}") from e


def get_iirs_paths(
    ddir,
    exts=("qub", "hdr", "xml", "csv", "xml-csv", "lbr", "oat", "oath", "spm", "png", "xml-png"),
    level=1,
    basenames=None,
):
    """Return a list of paths to IIRS image, geom, and misc files."""

    def basename(img_path_obj):
        """Return IIRS basename e.g. 20210122T0920157625 from pathlib path."""
        return img_path_obj.stem.split("_")[3]

    LVL2DIR = {0: "raw", 1: "calibrated", 2: "derived"}
    out = {}
    for ext in exts:
        subdir = "."
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
        paths = Path(ddir).glob(f'**/{subdir}/**/*.{ext.split("-")[0]}')

        if basenames is not None:
            basenames = [basenames] if isinstance(basenames, str) else basenames
            out[ext] = {basename(f): f for f in paths if basename(f) in basenames}
        else:
            out[ext] = {basename(f): f for f in paths}
        # Drop this entry from dict if it is empty
        if not out[ext]:
            del out[ext]
    # Add list of basenames to dict
    if "qub" in out:
        out["imgs"] = list(out["qub"].keys())
    return out


def get_wls(fname, as_str=False):
    """Return the wavelength labels from the image metadata."""
    img = pdr.open(fname)
    wl_dict = list(img.metaget("Band_Bin_Set").values())
    wls = [float(wl["center_wavelength"]) for wl in wl_dict]
    if as_str:
        return [f"{wl} nm" for wl in wls]
    return wls


def parse_geom(fgeom, extent=(None, None, None, None), buffer=0, center=False, as_gcps=False, corners_only=False):
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
    out = df[
        (df["Longitude"] >= minlon - buffer)
        & (df["Longitude"] <= maxlon + buffer)
        & (df["Latitude"] >= minlat - buffer)
        & (df["Latitude"] <= maxlat + buffer)
    ]
    # Convert output dataframe to list of gcps or gcp corners if requested
    # TODO: better error handling for ranges returning 0-length xyext
    xyext = [min(out["Pixel"]), max(out["Pixel"]), min(out["Scan"]), max(out["Scan"])]
    if as_gcps and corners_only:
        out = [
            GroundControlPoint(row["Scan"], row["Pixel"], row["Longitude"], row["Latitude"])
            for i, row in out.iterrows()
            if row["Pixel"] in xyext and row["Scan"] in xyext
        ]
    elif as_gcps:
        out = [
            GroundControlPoint(row["Scan"], row["Pixel"], row["Longitude"], row["Latitude"])
            for i, row in out.iterrows()
        ]
    return out, xyext


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


def get_iirs_solar_flux(sdist=1, fflux=FSOLAR):
    """Return the solar flux from the IIRS solar flux csv."""
    df = pd.read_csv(fflux, sep="\t", header=None, names=["wl", "flux"])
    df.loc[:, "flux"] = df.flux / 3.14 / sdist**2
    return df


def load_bad_bands(fbad_bands=FBADBANDS):
    """Load bad bands from a CSV file."""
    # Negates the df since provided bad bands file is 1 for good, 0 for bad
    return ~pd.read_csv(fbad_bands, index_col=0).astype(bool)


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


def get_gain_offset(flut, denoise=False):
    """
    Return the IIRS gain and offset for fimg as DataArrays.

    Parameters
    ----------
    flut : str
        Path to the lookup table file.
    denoise: bool
        Replace speckly noise from gain / offset with NaN.
    """
    lut = np.loadtxt(flut, delimiter=",").reshape((256, 250, 2))
    coords = {"band": 1 + np.arange(0, 256), "x": 0.5 + np.arange(250)}
    gain = xr.DataArray(lut[:, :, 0], coords=coords, name="gain")
    off = xr.DataArray(lut[:, :, 1], coords=coords, name="offset")
    if denoise:
        # Outlier detection with moving window and local median deviations
        thresh = 40  # Empirical - seemed to filter out noise
        window = gain.rolling({"band": 3, "x": 5}, center=True, min_periods=1)
        meddev = abs(gain - window.median())
        meddev_norm = meddev / meddev.median()
        gain = gain.where(meddev_norm < thresh)

        window = off.rolling({"band": 3, "x": 5}, center=True, min_periods=1)
        meddev = abs(off - window.median())
        meddev_norm = meddev / meddev.median()
        off = off.where(meddev_norm < thresh)
    return gain, off


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
    """Write dataarray cube to fout using rasterio."""

    wls = [f"{wl:.2f}" for wl in da.wl]
    nz, ny, nx = da.shape
    transform = da.rio.transform

    with rasterio.open(
        fout,
        "w",
        driver="ENVI",
        height=ny,
        width=nx,
        count=nz,
        dtype=da.dtype,
        transform=transform,
        crs=da.rio.crs if da.rio.crs else None,
    ) as dst:
        for i in range(nz):
            dst.write(da.isel(band=i).values, i + 1)
            dst.set_band_description(i + 1, wls[i])
            # TODO fix wavelength metadata


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
    fqub = (
        "/home/cjtu/projects/lai/issdc-requester/data/corrected_python/refl/ch2_iir_20210622T1256344234_destriped.img"
    )
    fgeom = "/home/cjtu/projects/lai/data/moon/ch2/iirs/geometry/calibrated/20210622/ch2_iir_nci_20210622T1256344234_g_grd_d32.csv"
    fpoints = "/home/cjtu/projects/lai/data/moon/ch2/iirs/corrected_python/refl/ch2_iir_20210622T1256344234_destriped.img.points"

    # Read un-georeferenced data qub and add projection info
    da = xr.open_dataarray(fqub, engine="rasterio")
    gridlon, gridlat, xyext = geom2grid(fgeom, (None, None, -86, -83))
    gcps, gcps_crs = read_gcps(fpoints)

    # projected = warp2grid(da, xyext, gridlon, gridlat)
    proj_from_gcps = warp2gcps(da, gcps, gcps_crs, "./test.tif")
    pass
