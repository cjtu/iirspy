"""iirspy module"""

from importlib import metadata

from iirspy.iirs import L0 as L0
from iirspy.iirs import L1 as L1

__version__ = metadata.version(__package__)
del metadata
