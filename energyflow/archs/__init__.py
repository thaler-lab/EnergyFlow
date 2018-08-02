from __future__ import absolute_import

import warnings

__all__ = []

# requires keras
try:
    from . import cnn
    from . import dnn
    from . import efn
    from .cnn import *
    from .dnn import *
    from .efn import *

    __all__ += cnn.__all__ + dnn.__all__ + efn.__all__

except ImportError as e:
    warnings.warn('could not import some architectures - ' + e.msg)

# requires sklearn
try:
    from . import linear
    from .linear import *

    __all__ += linear.__all__

except ImportError as e:
    warnings.warn('could not import some architectures - ' + e.msg)
