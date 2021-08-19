"""# Utilities

Utility functions for the EnergyFlow package. The utilities are grouped into the
following submodules:

- [`data_utils`](#data-utils): Utilities for processing datasets as arrays of
events.
aspects.
- [`fastjet_utils`](#fastjet-utils): Utilities for interfacing with the Python
wrapper of the [FastJet](http://fastjet.fr/) package.
- [`image_utils`](#image-utils): Utilities for creating and standardizing images
from collections of particles.
- [`particle_utils`](#particle-utils): Utilities for manipulating particle
properties, including converting between different kinematic representations,
adding/centering collections of four-vectors, and accessing particle properties
including masses and charges by PDG ID.
- [`random_utils`](#random-utils): Utilities for generating random collections
of (massless) four-vectors.
"""

#  _    _ _______ _____ _       _____
# | |  | |__   __|_   _| |     / ____|
# | |  | |  | |    | | | |    | (___
# | |  | |  | |    | | | |     \___ \
# | |__| |  | |   _| |_| |____ ____) |
#  \____/   |_|  |_____|______|_____/

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from . import arch_utils
from . import data_utils
from . import fastjet_utils
from . import generic_utils
from . import graph_utils
from . import image_utils
from . import particle_utils
from . import random_utils

from .arch_utils import *
from .data_utils import *
from .fastjet_utils import *
from .generic_utils import *
from .image_utils import *
from .particle_utils import *
from .random_utils import *

__all__ = (data_utils.__all__ +
           fastjet_utils.__all__ +
           particle_utils.__all__ +
           random_utils.__all__)
