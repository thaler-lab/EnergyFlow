"""A subpackage containing utility functions and classes. Not meant to be 
imported directly in energyflow."""
from __future__ import absolute_import

from .measure import *

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