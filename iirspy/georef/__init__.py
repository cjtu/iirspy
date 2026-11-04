"""Georeferencing: map each camera pixel of a scene to the Moon, as one `Geometry` per method.

`GEOREF` names the methods: "issdc" reproduces ISSDC's own geometry from the csv and spm; "arosics" registers the
L1 against a LOLA/WAC hillshade chunk by chunk (`iirspy.georef.arosics`). Both return a `Geometry` that
`L1.calibrate(geom=...)` consumes; the cube itself stays in camera space until `to_geotiff`.
"""

from collections.abc import Callable

from iirspy.georef import arosics, issdc
from iirspy.georef.geometry import Geometry

__all__ = ["GEOREF", "Geometry"]

GEOREF: dict[str, Callable[..., Geometry]] = {"issdc": issdc.geometry, "arosics": arosics.run}
