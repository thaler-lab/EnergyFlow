"""A subpackage for handling multigraph generation.
Note that unless igraph is importable, the functionality of this module will not be available.
In this case, EnergyFlow is still usable with the provided default file of precomputed multigraphs.
"""

from __future__ import absolute_import

from . import gen
from .gen import *
__all__ = gen.__all__