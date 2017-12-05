"""A subpackage for handling multigraph generation."""
from __future__ import absolute_import

from energyflow.utils import igraph_import

# if igraph cannot be imported, do nothing further here
__all__ = []
igraph = igraph_import()
if igraph:
    from . import gen
    from .gen import *
    __all__ = gen.__all__

del igraph
