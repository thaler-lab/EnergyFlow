#           _____   _____ _    _  _____
#     /\   |  __ \ / ____| |  | |/ ____|
#    /  \  | |__) | |    | |__| | (___
#   / /\ \ |  _  /| |    |  __  |\___ \
#  / ____ \| | \ \| |____| |  | |____) |
# /_/    \_\_|  \_\\_____|_|  |_|_____/

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import

import warnings

__all__ = []

# requires keras/tensorflow
try:
    from . import archbase
    from . import cnn
    from . import dnn
    from . import efn
    from .cnn import *
    from .dnn import *
    from .efn import *

    __all__ += cnn.__all__ + dnn.__all__ + efn.__all__

except ImportError as e:
    warnings.warn('could not import some architectures - ' + str(e))

# requires sklearn
try:
    from . import linear
    from .linear import *

    __all__ += linear.__all__

except ImportError as e:
    warnings.warn('could not import some architectures - ' + str(e))
