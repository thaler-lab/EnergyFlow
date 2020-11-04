from __future__ import absolute_import

import warnings

__all__ = []

# requires keras/tensorflow
try:
    from . import archbase
    from . import cnn
    from . import dnn
    from . import efn
    from . import utils
    from .cnn import *
    from .dnn import *
    from .efn import *
    from .utils import *

    __all__ += cnn.__all__ + dnn.__all__ + efn.__all__ + utils.__all__

except ImportError as e:
    warnings.warn('could not import some architectures - ' + str(e))

# requires sklearn
try:
    from . import linear
    from .linear import *

    __all__ += linear.__all__

except ImportError as e:
    warnings.warn('could not import some architectures - ' + str(e))
