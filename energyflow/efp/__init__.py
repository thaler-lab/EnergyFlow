from __future__ import absolute_import, division, print_function

# import base efp support
from . import efp_base
from .efp_base import *
__all__ = efp_base.__all__

# try importing efp gen, will fail without igraph
try:
    from . import efp_gen
    from .efp_gen import *
    __all__ += efp_gen.__all__
except ImportError:
    pass
