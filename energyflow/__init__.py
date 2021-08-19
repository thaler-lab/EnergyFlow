r"""

$$$$$$$$\                                                   $$$$$$$$\ $$\
$$  _____|                                                  $$  _____|$$ |
$$ |      $$$$$$$\   $$$$$$\   $$$$$$\   $$$$$$\  $$\   $$\ $$ |      $$ | $$$$$$\  $$\  $$\  $$\
$$$$$\    $$  __$$\ $$  __$$\ $$  __$$\ $$  __$$\ $$ |  $$ |$$$$$\    $$ |$$  __$$\ $$ | $$ | $$ |
$$  __|   $$ |  $$ |$$$$$$$$ |$$ |  \__|$$ /  $$ |$$ |  $$ |$$  __|   $$ |$$ /  $$ |$$ | $$ | $$ |
$$ |      $$ |  $$ |$$   ____|$$ |      $$ |  $$ |$$ |  $$ |$$ |      $$ |$$ |  $$ |$$ | $$ | $$ |
$$$$$$$$\ $$ |  $$ |\$$$$$$$\ $$ |      \$$$$$$$ |\$$$$$$$ |$$ |      $$ |\$$$$$$  |\$$$$$\$$$$  |
\________|\__|  \__| \_______|\__|       \____$$ | \____$$ |\__|      \__| \______/  \_____\____/
                                        $$\   $$ |$$\   $$ |
                                        \$$$$$$  |\$$$$$$  |
                                         \______/  \______/

EnergyFlow - Python package for high-energy particle physics.
Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import absolute_import

# import top-level submodules
from . import algorithms
from . import archs
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

__author__ = 'Patrick T. Komiske III <pkomiske@mit.edu>'
__version__ = '2.0.0a0'

# python < 3.5 warning
import sys, warnings
if sys.version_info < (3, 5):
    m = 'Using EnergyFlow with Python {}.{} is deprecated; support will be removed in the future'
    warnings.warn(FutureWarning(m.format(*sys.version_info[:2])))
del sys, warnings
