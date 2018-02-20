"""Implementation of Variable Elimination (VE) Algorithm."""

from __future__ import absolute_import

import itertools
import numpy as np

from energyflow.utils import igraph_import

igraph = igraph_import()

__all__ = ['VariableElimination']

inds = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

class VariableElimination:

    def __init__(self, ve_alg, np_optimize='greedy'):
        possible_algs = ['numpy'] + (['ef'] if igraph else [])
        if ve_alg not in possible_algs:
            raise ValueError('ve_alg must be one of {}'.format(possible_algs))
        self.ve_alg = ve_alg
        self._use_numpy_ve = (self.ve_alg == 'numpy')

        if self._use_numpy_ve:
            self.np_optimize = np_optimize
            self.X = np.empty((2,2))
            self.y = np.empty(2)

        # set public methods based on which ve_alg is chosen
        setattr(self, 'run', self._ve_numpy if self._use_numpy_ve else self._ve_ef)
        setattr(self, 'einspecs', self._einspecs_numpy if self._use_numpy_ve else self._einspecs_ef)

    def _einstr_from_edges(self, edges, n):
        einstr  = ','.join([inds[j]+inds[k] for (j, k) in edges])+','
        einstr += ','.join([inds[v] for v in range(n)])
        return einstr

    def _ve_numpy(self, edges, n):
        d = len(edges)
        self.einstr = self._einstr_from_edges(edges, n)
        args = [self.X]*d + [self.y]*n
        einpath = np.einsum_path(self.einstr, *args, optimize=self.np_optimize)
        self.einpath = einpath[0]
        self.chi = int(einpath[1].split('\n')[2].split(':')[1])

    def _ve_ef(self, edges, n):
        self.edges = edges
        self._ef_elim_order()

    def _einspecs_numpy(self):
        return self.einstr, self.einpath

    def _einspecs_ef(self):
        return self._einstr_from_edges(self.edges, max(self.elim_order)+1), self._ef_einsum_path()

    # implements heuristics to find a good elimination ordering for VE
    def _ef_elim_order(self):

        # make new igraph Graph
        graph = igraph.Graph(self.edges)

        # vertex information
        N = graph.vcount()
        active_vertices = list(range(N))

        # storage of computation
        self.elim_order = []
        self.chi = 0
        for i in range(N):

            # get degrees of active vertices
            degrees = graph.neighborhood_size(vertices=active_vertices)

            # find lowest degree of connected vertices (disconnected have degree 0)
            min_degree = min(degrees)
            self.chi = max(self.chi, min_degree)

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
            self.elim_order.append(best_min_vertex)

            # add fill in edges
            graph.add_edges([p for p,i in zip(best_pairs,best_eids) if i != -1])

            # remove edges associated with selected vertex
            graph.delete_edges(graph.incident(best_min_vertex))

            # remove vertex from active vertices
            active_vertices.pop(active_vertices.index(best_min_vertex))

    def _ef_einsum_path(self):
        
        # a list of tensors, to be updated as we eliminate vertices
        tensors = [e for e in self.edges] + [(v,) for v in range(len(self.elim_order))]
        
        # the path we're building
        einsum_path = ['einsum_path']
        
        # iterate over 
        for v in self.elim_order:
            
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
            
            # append new formula
            einsum_path.append(formula)
            
        return einsum_path
