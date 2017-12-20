"""A subpackage containing utility functions and classes. Not meant to be 
imported directly in energyflow."""
from __future__ import absolute_import
import itertools

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

def nvert(graph):
    return 1 + max((max(edge) for edge in graph))

# assumes graphs have vertices 0-(n-1)
def graph_union(*graphs):
    ns = [nvert(graph) for graph in graphs[:-1]]
    adds = [sum(ns[:i]) for i in range(1,len(graphs))]
    new_comps = [[tuple(a+v for v in edge) for edge in graph] for a,graph in zip(adds,graphs[1:])]
    return list(itertools.chain(graphs[0], *new_comps))