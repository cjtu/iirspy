from collections.abc import Callable

from iirspy.georef import arosics, issdc
from iirspy.georef.geometry import Geometry as Geometry
from iirspy.georef.georef import LONLAT as LONLAT
from iirspy.georef.georef import MOON_RADIUS_M as MOON_RADIUS_M
from iirspy.georef.georef import NODATA as NODATA
from iirspy.georef.georef import WAC_MIN_ELEV_DEG as WAC_MIN_ELEV_DEG
from iirspy.georef.georef import GeorefConfig as GeorefConfig
from iirspy.georef.georef import apply_glt as apply_glt
from iirspy.georef.georef import camera_gradients as camera_gradients
from iirspy.georef.georef import coarse_shift as coarse_shift
from iirspy.georef.georef import crop_glt as crop_glt
from iirspy.georef.georef import displacement_field as displacement_field
from iirspy.georef.georef import gcp_lattice as gcp_lattice
from iirspy.georef.georef import geoloc_vrt as geoloc_vrt
from iirspy.georef.georef import geom_csv_gcps as geom_csv_gcps
from iirspy.georef.georef import glt_from_loc as glt_from_loc
from iirspy.georef.georef import grid_of as grid_of
from iirspy.georef.georef import load_lola_elev as load_lola_elev
from iirspy.georef.georef import lonlat_to_xyz as lonlat_to_xyz
from iirspy.georef.georef import make_glt as make_glt
from iirspy.georef.georef import pole_from_crs as pole_from_crs
from iirspy.georef.georef import project as project
from iirspy.georef.georef import read_band as read_band
from iirspy.georef.georef import read_glt as read_glt
from iirspy.georef.georef import register as register
from iirspy.georef.georef import render_reference as render_reference
from iirspy.georef.georef import render_wac as render_wac
from iirspy.georef.georef import save_gcp_vrt as save_gcp_vrt
from iirspy.georef.georef import save_glt as save_glt
from iirspy.georef.georef import save_grid as save_grid
from iirspy.georef.georef import save_topo as save_topo
from iirspy.georef.georef import sc_positions_km as sc_positions_km
from iirspy.georef.georef import scan_order_reversed as scan_order_reversed
from iirspy.georef.georef import scene_glt as scene_glt
from iirspy.georef.georef import scene_topo as scene_topo
from iirspy.georef.georef import slope_aspect as slope_aspect
from iirspy.georef.georef import stereo_crs as stereo_crs
from iirspy.georef.georef import sun_geometry as sun_geometry
from iirspy.georef.georef import sun_geometry_rows as sun_geometry_rows
from iirspy.georef.georef import to_lonlat as to_lonlat
from iirspy.georef.georef import to_stereo as to_stereo
from iirspy.georef.georef import use_wac as use_wac
from iirspy.georef.georef import window_of as window_of
from iirspy.georef.georef import xy_to_lonlat as xy_to_lonlat
from iirspy.georef.georef import xyz_to_lonlat as xyz_to_lonlat

GEOREF: dict[str, Callable[..., Geometry]] = {"issdc": issdc.geometry, "arosics": arosics.geometry}


def run_georef(method: str, *, ws=None, **inputs) -> Geometry:
    return GEOREF[method](ws=ws, **inputs)
