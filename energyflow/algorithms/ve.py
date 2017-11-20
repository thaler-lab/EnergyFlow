from __future__ import absolute_import

import itertools
import sys
import numpy as np

from energyflow.utils import igraph_import

igraph = igraph_import(__file__)

einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

__all__ = ['VariableElimination']

class VariableElimination:

    def __init__(self, ve_alg='numpy', np_optimize='greedy'):
        possible_algs = ['numpy'] + (['ef'] if igraph else [])
        assert ve_alg in possible_algs, 've_alg must be in {}'.format(possible_algs)
        self.ve_alg = ve_alg

        if self.ve_alg == 'numpy':
            self.np_optimize = np_optimize
            self.dummy_dim = 10
            self.X = np.random.rand(self.dummy_dim, self.dummy_dim)
            self.y = np.random.rand(self.dummy_dim)

    def _einstr_from_edges(self, edges, n):
        einstr  = ','.join([einsum_symbols[j] + einsum_symbols[k] for (j, k) in edges]) + ','
        einstr += ','.join([einsum_symbols[v] for v in range(n)])
        return einstr

    def _ve_numpy(self, edges, n):
        d = len(edges)
        self.einstr = self._einstr_from_graph(edges, n)
        self.einpath = np.einsum_path(einstr, *[self.X]*d, *[self.y]*n, optimize=self.np_optimize)
        self.chi = int(einpath[1].split('\n')[2].split(':')[1])

    def _ve_ef_chi_elim_order(self, edges):
        self.edges = edges
        self.elim_order, self.chi = ef_elim_order(self.edges)

    # returns chi
    def set(self, edges, n):
        if self.ve_alg == 'numpy':
            self._ve_numpy(edges, n)
        else:
            self._ve_ef_chi_elim_order(edges)
        return self.chi

    def einspecs(self):
        if self.ve_alg == 'numpy':
            return (self.einstr, self.einpath)
        else:
            return (self._einstr_from_edges(self.edges, max(self.elim_order)+1),
                    ef_einsum_path(self.edges, self.elim_order))

# implements heuristics to find a good elimination ordering for VE
def ef_elim_order(edges):

    # make new igraph Graph
    graph = igraph.Graph(edges)

    # vertex information
    N = graph.vcount()
    active_vertices = list(range(N))

    # storage of computation
    elim_order = []
    chi = 0
    for i in range(N):

        # get degrees of active vertices
        degrees = graph.neighborhood_size(vertices=active_vertices)

        # find lowest degree of connected vertices (disconnected have degree 0)
        min_degree = min(degrees)
        chi = max(chi, min_degree)

        # find vertices corresponding to the minimum degree
        min_vertices = [v for v,d in zip(active_vertices,degrees) if d == min_degree]

        # get all neighborhoods of these vertices
        min_neighborhoods = graph.neighborhood(vertices=min_vertices)

        # iterate over all neighborhoods and select one via min fill heuristic
        best_num_fill = int((min_degree-1)*(min_degree-2)/2)+1
        for min_neighborhood, min_vertex in zip(min_neighborhoods, min_vertices):

            # remove self from neighborhood
            min_neighborhood.pop(min_neighborhood.index(min_vertex))

            # get all pairs of edges that could exist
            pairs = list(itertools.combinations(min_neighborhood, 2))

            # edge ids are -1 if not in the graph
            eids = graph.get_eids(pairs=pairs, error=False)

            # update best so far
            num_fill = np.count_nonzero(eids == -1)
            if num_fill < best_num_fill:
                best_min_vertex = min_vertex
                best_num_fill = num_fill
                best_pairs = pairs
                best_eids = eids
                best_neighborhood = min_neighborhood

        # add selected vertex to elimination ordering
        elim_order.append(best_min_vertex)

        # add fill in edges
        graph.add_edges([p for p,i in zip(best_pairs,best_eids) if i != -1])

        # remove edges associated with selected vertex
        graph.delete_edges(graph.incident(best_min_vertex))

        # remove vertex from active vertices
        active_vertices.pop(active_vertices.index(best_min_vertex))
        
    return elim_order, chi

def ef_einsum_path(edges, elim_order):
    
    # a list of tensors, to be updated as we eliminate vertices
    tensors = [e for e in edges] + [(v,) for v in range(len(elim_order))]
    
    # the path we're building
    einsum_path = ['einsum_path']
    
    # a list of the tensors that get combined in each step
    #ops = []
    
    # iterate over 
    for v in elim_order:
        
        # get positions of/and tensors that contain v
        new_it = [[i,t] for i,t in enumerate(tensors) if v in t]
        
        # aggregate all the tensors
        op = [it[1] for it in new_it]
        
        # aggregate the positions
        formula = tuple(it[0] for it in new_it)
        
        # remove tensors that contained v from tensors list
        for i in formula[::-1]: 
            del tensors[i]
            
        # add new tensor to tensors list
        tensors.append(tuple(set(f for t in op for f in t if f!=v)))
        
        # append new formula and op
        einsum_path.append(formula)
        #ops.append(op)
        
    return einsum_path