"""The EnergyFlow package."""
from __future__ import absolute_import

# attempt to use 'fork' start method in multiprocessing
import sys
if sys.version_info[:2] >= (3, 4):
    import multiprocessing, warnings
    method = multiprocessing.get_start_method(True)
    if method is None:
        try:
            multiprocessing.set_start_method('fork')
        except:
            pass
    method = multiprocessing.get_start_method()
    if method != 'fork':
        warnings.warn('multiprocessing start method is {},'.format(method)
                      + ' EnergyFlow multicore functionality may not work properly')
    del multiprocessing, warnings
del sys

# import top-level submodules
from . import algorithms
from . import base
from . import datasets
from . import efm
from . import efp
from . import emd
from . import gen
from . import measure
from . import obs
from . import utils

# import top-level attributes
from .datasets import *
from .efm import *
from .efp import *
from .gen import *
from .measure import *
from .obs import *
from .utils import *

__all__ = (datasets.__all__ +
           efm.__all__ +
           efp.__all__ +
           gen.__all__ +
           measure.__all__ +
           obs.__all__ +
           utils.__all__)

__version__ = '1.1.2'
