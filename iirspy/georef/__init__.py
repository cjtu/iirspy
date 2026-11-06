"""Georeferencing: map each camera pixel of a scene to the Moon, as one `Geometry` per method.

`GEOREF` names the methods: "issdc" (`Geometry.from_issdc`) reproduces ISSDC's own geometry from the csv and spm;
"arosics" registers the L1 against a LOLA/WAC hillshade chunk by chunk (`iirspy.georef.arosics`). Both return a `Geometry` that
`L1.calibrate(geom=...)` consumes; the cube itself stays in camera space until `to_geotiff`.

NOTE: Automated georeferencing is still under development and may not produce accurate results in all cases.
"""

from collections.abc import Callable

from iirspy.georef import arosics
from iirspy.georef.geometry import Geometry

__all__ = ["GEOREF", "Geometry"]

GEOREF: dict[str, Callable[..., Geometry]] = {"issdc": Geometry.from_issdc, "arosics": arosics.run}
