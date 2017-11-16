from __future__ import absolute_import, division, print_function

# determine if igraph is installed
try:
    import igraph
except ImportError:
    igraph = False

# import base efp support
from . import efp_base
from .efp_base import *
__all__ = efp_base.__all__

# import efp generation support if igraph is installed
if igraph:
    from . import efp_gen
    from .efp_gen import *
    __all__ += efp_gen.__all__

# unclutter namespace
del igraph
