from __future__ import absolute_import

from energyflow.utils import igraph_import

igraph = igraph_import(__file__)

__all__ = []
if igraph:
    from . import gen
    from .gen import *
    __all__ = gen.__all__

del igraph
