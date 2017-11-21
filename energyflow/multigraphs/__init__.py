"""A subpackage for handling multigraph generation."""

from __future__ import absolute_import

from energyflow.utils import igraph_import

igraph = igraph_import(warn=False, file=__file__)

# if igraph cannot be imported, do nothing further here
__all__ = []
if igraph:
    from . import gen
    from .gen import *
    __all__ = gen.__all__

del igraph
