"""The "issdc" georef method: rebuild ISSDC's own geometry (csv lon/lat, spm angles per line) as a `Geometry`.

No registration and no DEM: the flat-Moon reference for comparing against "arosics" solves.
"""

import numpy as np

from iirspy import io
from iirspy.georef import ancillary
from iirspy.georef import backplanes as bp
from iirspy.georef.geometry import Geometry, quality
from iirspy.georef.georef import MOON_RADIUS_M


def geometry(
    *, sid: str, group: str, fgeom, flabel, fspm, shape: tuple[int, int], row0: int = 0, work=None
) -> Geometry:
    """Reproduces ISSDC geometry per the SIS: csv lon/lat, spm angles per line."""
    ny, nx = shape
    scans = row0 + np.arange(ny)
    tl = ancillary.get_line_times(flabel)[scans]
    lon, lat, _ = ancillary.geom2grid(fgeom, xs=np.arange(nx), ys=scans, xyextent=(0, nx - 1, scans[0], scans[-1]))

    def per_line(col):
        return np.broadcast_to(ancillary.get_spm_interpolator(fspm, col)(tl)[:, None], (ny, nx))

    zenith = 90 - np.abs(per_line("sun_elev"))
    bands = {
        "sun_azimuth": per_line("sun_az"),
        "sun_zenith": zenith,
        "sensor_zenith": np.zeros((ny, nx)),
        "phase": per_line("phase"),
        "facet_cos_i": np.cos(np.radians(zenith)),
    }
    loc = {"lon": lon, "lat": lat, "radius": np.full((ny, nx), MOON_RADIUS_M)}
    nan = np.full((ny, nx), np.nan, "float32")
    qa = bp.build_qa(loc, {"bands": {**dict.fromkeys(bp.OBS_NAMES, nan), **bands}}) | bp.qa_mask("no_topo")
    return Geometry.from_arrays(
        sid,
        group,
        row0,
        "issdc",
        loc,
        bands,
        qa,
        quality("issdc", "warn"),
        provenance=[{"step": "georef", "method": "issdc", "version": io.IIRSPY_VERSION}],
        tags={
            "GEOMETRY_BASIS": "Reproduces ISSDC geometry per the SIS: csv lon/lat, spm angles per line.",
            "SENSOR_SOURCE": "spm",
        },
    )
