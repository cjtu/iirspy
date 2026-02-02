"""iirspy module"""

from importlib import metadata

from iirspy.iirs import L0 as L0
from iirspy.iirs import L1 as L1
from iirspy.iirs import L2 as L2

__version__ = metadata.version(__package__)
del metadata
