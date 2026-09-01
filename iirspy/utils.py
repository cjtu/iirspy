import hashlib
import re
import warnings
import zipfile
from importlib.resources import files
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
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
from scipy.signal import savgol_filter
from scipy.stats import norm

warnings.filterwarnings("ignore", message="Dataset has no geotransform")

## Constants
PKG_DATA = files("iirspy").joinpath("data")
DCALIB = PKG_DATA.joinpath("iir/calibration")
CHUNKSIZE = 128 * 2**20  # [B] dask y-chunk target (dask's array.chunk-size default; profiled fastest + lowest RAM)
FPOLISH = str(PKG_DATA.joinpath("spectral_polish_verma2022.csv"))
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
    swindow=9,
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
        ymin, ymax = yrange
        yoff = 0
        # Ascending
        # yoff = 51
        # ymin, ymax = yrange
        # if yflip:  # Descending
        #     yoff = -86
        #     ymin, ymax = yrange
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
    da = RAD_NATIVE_SCALE * da  # [1000 mW/cm^2/sr/um] -> [W/m^2/sr/um]
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
    da = da * RAD_NATIVE_SCALE

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
        "lat": np.arange(minlat, maxlat + dlat, dlat, dtype="float32"),
    }
    # TODO: need to replace regridder
    # regridder = xe.Regridder(src, target_grid, method, unmapped_to_nan=True)
    # out = regridder(src)
    out = target_grid

    # Write crs (Moon unprojected)
    # NOTE: WIP — regridder above is disabled, so `out` is currently a dict and
    # these .rio calls will fail at runtime until the regridder is restored.
    out.rio.write_crs(CRS.from_authority("IAU", "30100"), inplace=True)  # type: ignore[attr-defined]
    out.rio.set_spatial_dims("lon", "lat", inplace=True)  # type: ignore[attr-defined]
    out.rio.write_coordinate_system(inplace=True)  # type: ignore[attr-defined]
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
        raise FileNotFoundError(
            f"Image {basename} not found in {ddir}. Please download from PRADAN or check file path."
        )
    with zipfile.ZipFile(f, "r") as zipf:
        print(f"Extracting {basename} to {f.parent}")
        zipf.extractall(f.parent)
    paths = get_iirs_paths(f.parent, level=level, basenames=[basename])
    if "qub" not in paths:
        raise RuntimeError("Unzip failed.")
    if md5checksum:
        print("Verifying unzipped image...", end=" ")
        checksum(paths["qub"][basename].as_posix())
        print("Success!")
    return paths


def iirsbasename(input_str):
    """Return the image basename from str_in (e.g., 20201226T1745264921)"""
    pattern = r"\d{8}T\d{10}"
    match = re.search(pattern, input_str)
    if match is None:
        raise ValueError(f"Can't parse basename: {input_str}")
    return match.group()


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
    out: dict[str, Any] = {}
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
        paths = Path(ddir).glob(f"**/{subdir}/**/*.{ext.split('-')[0]}")

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


def get_iirs_latlon(fgeom, center=False):
    """Return lat and lon grids from iirs geometry csv file as DataArrays.

    Note: Provided lat/lon are reported for every 50th line/sample, not every pixel.
    """
    df = pd.read_csv(fgeom).rename(columns={"Pixel": "x", "Scan": "y"})
    df["Longitude"] = (df["Longitude"] + 180) % 360 - 180  # lon in [-180, 180]
    # set coords from 1 to max(coord) unless center coords, which start at 0.5
    df["x"] = df.x + 1 - 0.5 * center
    df["y"] = df.y + 1 - 0.5 * center
    df2d = df.set_index(["y", "x"])
    lat = xr.DataArray(df2d.Latitude.unstack())
    lon = xr.DataArray(df2d.Longitude.unstack())

    return lat, lon


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
    df = pd.read_csv(fgeom)
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


def get_solar_distance(fimg, kernels=None):
    """Return the Sun-Moon distance in AU at the scene's mid-line epoch, from SPICE.

    `kernels=None` resolves the kernel set via `chunks.kernels(day)`. Warns and returns 1.0 AU if
    the label or kernels can't be found, rather than failing the whole calibration.
    """
    import spiceypy as sp

    from iirspy import chunks

    fimg = Path(fimg)
    try:
        line_times = get_line_times(fimg)
        t = pd.Timestamp(line_times[len(line_times) // 2], unit="s")
        ks = kernels if kernels is not None else chunks.kernels(t.strftime("%Y%m%d"))
        for k in ks:
            sp.furnsh(str(k))
        et = sp.str2et(t.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        v, _ = sp.spkpos("SUN", et, "IAU_MOON", "LT+S", "MOON")
    except Exception as e:
        warnings.warn(f"get_solar_distance: could not compute from SPICE for {fimg} ({e}); using 1.0 AU", stacklevel=2)
        return 1.0
    return float(np.linalg.norm(v) / AU_KM)


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


## ENVI BIL streaming writer
def write_envi_hdr(fhdr, nx, ny, nband, wls, description="IIRS", bands=None, x_start=1, y_start=1):
    """Write an ENVI header for a float32 BIL cube (data type 4, little-endian).

    `bands` are the IIRS band numbers of the planes (default 1..nband); they go in `band names` so
    a band subset reads back as itself rather than 1..N. `x_start`/`y_start` are the 1-indexed
    sample/line of this crop in the parent scene: ENVI carries no transform, so they are the only
    place a crop's absolute position survives (GeoTIFF uses the geotransform instead).
    """
    bands = range(1, nband + 1) if bands is None else bands
    band_names = ", ".join(str(int(b)) for b in bands)
    wl_str = ", ".join(f"{float(w):.4f}" for w in wls)
    Path(fhdr).write_text(
        "ENVI\n"
        f"description = {{ {description} }}\n"
        f"samples = {nx}\nlines = {ny}\nbands = {nband}\n"
        "header offset = 0\nfile type = ENVI Standard\n"
        "data type = 4\ninterleave = bil\nbyte order = 0\n"
        f"x start = {int(x_start)}\ny start = {int(y_start)}\n"
        "wavelength units = Nanometers\n"
        f"band names = {{{band_names}}}\n"
        f"wavelength = {{{wl_str}}}\n"
    )


def _write_bil_rows(f, da, i0, i1, sub_rows):
    """Stream rows [i0:i1) of a (band, y, x) DataArray to open file f as ENVI BIL float32."""
    for a in range(i0, i1, sub_rows):
        b = min(a + sub_rows, i1)
        blk = np.asarray(da.isel(y=slice(a, b)).values, dtype="float32")  # (band, rows, x)
        blk.transpose(1, 0, 2).tofile(f)  # C-order of (row, band, x) == ENVI BIL


def write_envi_bil(da, fout, sub_rows, description="IIRS"):
    """Sequentially stream a (band, y, x) DataArray to an ENVI BIL float32 file (bounded memory).

    `sub_rows` is required: it is resolved once in `IIRSData._write`, so no second default can
    drift away from the cube's dask chunking.
    """
    fout = str(fout)
    nband, ny, nx = da.shape
    with open(fout, "wb") as f:
        _write_bil_rows(f, da, 0, ny, sub_rows)
    wls = da.wl.values if "wl" in da.coords else np.arange(1, nband + 1)
    # Pixel centres (n + 0.5) -> ENVI's 1-indexed sample/line of the crop's upper-left pixel
    x_start, y_start = (int(np.floor(float(da[d].min()))) + 1 for d in ("x", "y"))
    write_envi_hdr(Path(fout).with_suffix(".hdr"), nx, ny, nband, wls, description, da.band.values, x_start, y_start)
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
