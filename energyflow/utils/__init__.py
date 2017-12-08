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

def kwargs_check(name, kwargs):
    for k in kwargs:
        message = name + '() got an unexpected keyword argument \'{}\''.format(k)
        raise TypeError(message)

comp_map = {'>':  '__gt__', 
            '<':  '__lt__', 
            '>=': '__ge__', 
            '<=': '__le__',
            '==': '__eq__', 
            '!=': '__ne__'}

def explicit_comp(obj, comp, val):
    return getattr(obj, comp_map[comp])(val)
