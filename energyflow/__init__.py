"""This. Is. EnergyFlow."""
from __future__ import absolute_import

# import all submodules
from . import algorithms
from . import multigraphs
from . import polynomials
from . import utils
__all__ = ['algorithms', 'multigraphs', 'polynomials', 'utils']

# import contents of multigraphs and polynomial modules
from .multigraphs import *
from .polynomials import *
__all__ += multigraphs.__all__ + polynomials.__all__

__version__ = '0.4.3'
