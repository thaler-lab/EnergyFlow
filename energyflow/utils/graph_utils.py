"""Various useful functions on graphs."""

#   _____ _____            _____  _    _          _    _ _______ _____ _       _____
#  / ____|  __ \     /\   |  __ \| |  | |        | |  | |__   __|_   _| |     / ____|
# | |  __| |__) |   /  \  | |__) | |__| |        | |  | |  | |    | | | |    | (___
# | | |_ |  _  /   / /\ \ |  ___/|  __  |        | |  | |  | |    | | | |     \___ \
# | |__| | | \ \  / ____ \| |    | |  | | ______ | |__| |  | |   _| |_| |____ ____) |
#  \_____|_|  \_\/_/    \_\_|    |_|  |_||______| \____/   |_|  |_____|______|_____/

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import

from collections import Counter
import itertools

__all__ = [
    'import_igraph', 
    'get_components',
    'get_valency_structure',
    'graph_union',
    'nvert', 
    'valencies'
]

# determine if igraph can be imported, returns either the igraph module or false
def import_igraph():
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

def get_components(graph):
    """Returns a list of lists of vertices in each connected component of the
    graph.
    """

    vds = get_valency_structure(graph)

    verts = set(vds.keys())
    components = []
    while len(verts):
        i = 0
        component = [verts.pop()]
        while i < len(component):
            
            # append all vertices touched by the present one that haven't already been visited
            for v in vds[component[i]]:
                if v in verts:
                    verts.remove(v)
                    component.append(v)      
            i += 1                
        components.append(component)

    return components

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

def nvert(graph):
    """Gets the number of vertices, |V|, in the graph."""

    return 1 + max((max(edge) for edge in graph))

def valencies(graph):
    """Gets the valency of each vertex in the graph."""

    return Counter((v for edge in graph for v in edge))
    