"""Various useful functions on graphs."""
from __future__ import absolute_import

from collections import Counter
import itertools

__all__ = [
    'igraph_import', 
    'get_valency_structure',
    'graph_union', 
    'num_valency_ones',
    'nvert', 
    'valencies'
]

# determine if igraph can be imported, returns either the igraph module or false
def igraph_import():
    try:
        import igraph
    except:
        igraph = False
    return igraph

# standard graph form:
#   - a graph is a list of tuples
#   - vertices are always labeled from 0 to |V|-1
#   - each tuple in the list corresponds to an edge, 
#     specified by a pair of integers which are the 
#     vertices touching that edge
#   - the same edge may appear more than once, in which
#     case the graph is a multigraph
#
# each of the functions below operates on graphs assumed
# to be in this standard form

def get_valency_structure(graph):
    """Turn graph into a dictionary where the keys are the vertices
    and the values are dictionaries where the keys are again vertices 
    and the values are the number of edges shared by those vertices.
    """

    d = {}
    for edge in graph:
        d.setdefault(edge[0], []).append(edge[1])
        d.setdefault(edge[1], []).append(edge[0])
    return {v: Counter(d[v]) for v in d}

def graph_union(*graphs):
    """Returns the union of one or more graphs."""

    ns = [nvert(graph) for graph in graphs[:-1]]
    adds = [sum(ns[:i]) for i in range(1,len(graphs))]
    new_comps = [[tuple(a+v for v in edge) for edge in graph] for a,graph in zip(adds,graphs[1:])]
    return list(itertools.chain(graphs[0], *new_comps))

def num_valency_ones(graph):
    return Counter(valencies(graph).values())[1]

def nvert(graph):
    """Gets the number of vertices, |V|, in the graph."""

    return 1 + max((max(edge) for edge in graph))

def valencies(graph):
    """Gets the valency of each vertex in the graph."""

    return Counter((v for edge in graph for v in edge))
    