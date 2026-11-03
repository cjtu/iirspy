import numpy as np
import pandas as pd
import pdr
from rasterio.control import GroundControlPoint
from scipy.interpolate import RBFInterpolator, make_interp_spline

from iirspy import io

# Projections used in the Ch2 IIRS selenoref tool https://doi.org/10.1007/s12524-024-01814-4
IIRS_PROJ_DICT = {
    "equatorial": 'PROJCS["Moon_Equidistant_Cylindrical",GEOGCS["Moon 2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Greenwich",0],UNIT["Decimal_Degree",0.0174532925199433]],PROJECTION["Equidistant_Cylindrical"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Standard_Parallel_1",0],UNIT["Meter",1]]',
    "polarstereographicsouthpole": 'PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon 2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Greenwich",0],UNIT["Decimal_Degree",0.0174532925199433]],PROJECTION["Stereographic"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Scale_Factor",1],PARAMETER["Latitude_Of_Origin",-90],UNIT["Meter",1]]',
    "polarstereographicnorthpole": 'PROJCS["Moon_North_Pole_Stereographic",GEOGCS["Moon 2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Greenwich",0],UNIT["Decimal_Degree",0.0174532925199433]],PROJECTION["Stereographic"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Scale_Factor",1],PARAMETER["Latitude_Of_Origin",90],UNIT["Meter",1]]',
}


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


def geom2grid(fgeom, extent=(None, None, None, None), xs=None, ys=None, xyextent=(None, None, None, None)):
    """Interpolate sparse geometry csv to full 2D lat and lon grids, from the csv points in `extent` (lon/lat)
    or `xyextent` (Pixel/Scan), snapped outward to the point grid."""
    df_buf, xyext = parse_geom(fgeom, extent, xyextent)

    # Add buffer to the extent to ensure edges are included in interpolator
    # NOTE: Makes almost no difference (pixel lvl offset, overall offset much larger)
    # df_buf, _ = parse_geom(fgeom, extent, buffer=0.5)

    # TPS on unit vectors, not lon/lat: near a pole lon swings up to 180 deg within a few scans (and
    # wraps at +-180 anywhere), and a spline on lon oscillates there by tens of km.
    points = df_buf[["Pixel", "Scan"]].values
    lons = df_buf["Longitude"].values
    lon, lat = np.radians(lons), np.radians(df_buf["Latitude"].values)
    rbf = RBFInterpolator(
        points, np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], kernel="thin_plate_spline"
    )

    # Create a 2D grid of all pixel x and y values in extent
    pixel_range = np.arange(xyext[0], xyext[1] + 1, 1)
    scan_range = np.arange(xyext[2], xyext[3] + 1, 1)
    if xs is not None:
        pixel_range = xs
    if ys is not None:
        scan_range = ys
    grid_pixel, grid_scan = np.meshgrid(pixel_range, scan_range)

    v = rbf(np.column_stack([grid_pixel.ravel(), grid_scan.ravel()]))
    v /= np.linalg.norm(v, axis=1)[:, None]
    gridlon = np.degrees(np.arctan2(v[:, 1], v[:, 0])).reshape(grid_pixel.shape)
    gridlat = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1))).reshape(grid_pixel.shape)
    if (lons > 180).any():  # keep the csv's own 0-360 convention
        gridlon %= 360
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


def read_geom_csv(fgeom) -> pd.DataFrame:
    """IIRS geometry csv, minus the (0, 0) fill rows some bundles pad the tail with.

    The fill rows have Pixel=Scan=0 and Longitude=Latitude=0.0 exactly, duplicating the real Scan=0
    row; no real sample lands on (0.0, 0.0) to full float precision.
    """
    df = pd.read_csv(fgeom)
    return df[(df.Longitude != 0.0) | (df.Latitude != 0.0)]


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
    _, lines, _ = io.get_iirs_shape_meta(fimg)
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
