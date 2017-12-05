"""Helper functions related to igraph."""
from __future__ import absolute_import

__all__ = ['igraph_import']

def igraph_import():
    """
    Determines if igraph can be imported. 

    Returns
    -------
    output : {igraph, False}
        The igraph module if it was successfully imported, otherwise False.
    """
    
    try:
        import igraph
    except:
        igraph = False
    return igraph

# this file may eventually contain native implementations of igraph functions we use for ve
