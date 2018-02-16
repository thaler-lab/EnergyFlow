"""This. Is. EnergyFlow."""

from __future__ import absolute_import

# import all submodules
from . import algorithms
from . import multigraphs
from . import observables
from . import utils

# import contents of multigraphs and polynomial modules
from .multigraphs import *
from .observables import *

__all__ = multigraphs.__all__ + observables.__all__

__version__ = '0.5.0'
