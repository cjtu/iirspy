"""iirspy module"""

from importlib import metadata

from iirspy.iirs import L0 as L0
from iirspy.iirs import L1 as L1
from iirspy.iirs import L2 as L2

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:  # running from a source tree, e.g. iirspy.coreg
    __version__ = "0+source"
del metadata
