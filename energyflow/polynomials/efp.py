from __future__ import absolute_import, division, print_function
from collections import Counter
import numpy as np

from energyflow.algorithms import VariableElimination
from energyflow.polynomials.base import EFPBase, EFPElem

__all__ = ['EFP']

class EFP(EFPBase, EFPElem):

    def __init__(self, edges, measure='hadr_yphi', beta=1.0, normed=True, check_type=False, 
                              ve_alg='numpy', np_optimize='greedy'):

        # initialize base class
        super().__init__(self, measure, beta, normed, check_type)

        # get ve instance
        self.ve = VariableElimination(ve_alg, np_optimize)

        # deal with arbitrary vertex labels
        vertex_set = set(v for edge in edges for v in edge)
        self.vertices = {v: i for i,v in enumerate(sorted(list(vertex_set)))}
        self.n = len(vertex_set)

        # construct new edges with remapped vertices
        self.edges = [tuple(self.vertices[v] for v in edge) for edge in edges]
        self.d = len(self.edges)

        # construct edges of simple graphs
        simple_edges = list(set(self.edges))

        # get weights of edges
        counts = Counter(self.edges)
        self.weights = [counts[edge] for edge in simple_edges]
        self.weight_set = set(self.weights)

        # set internals of ve to these edges
        self.ve(simple_edges, self.n)
        self.c = self.chi
        self.einstr, self.einpath = self.einspecs()

    def _make_thetas_dict(self, thetas):
        return {w: thetas**w for w in self.weight_set}

    def compute(self, event=None, zs_thetas_d=None):
        if zs_thetas_d is not None:
            zs, thetasd = zs_thetas_d
        elif event is not None:
            zs, thetas = self.zs_thetas(event)
            thetas_d = self._make_thetas_dict(thetas)
        else:
            raise ValueError('event and zs_thetas_d cannot both be None')

        einsum_args = [thetas_d[w] for w in self.weights] + self.n*[zs]
        return np.einsum(self.einstr, *einsum_args, optimize=self.einpath)

    def batch_compute(self, events, n_jobs=None):
        return EFPBase.batch_compute(self, events, concat_disc=False, n_jobs=n_jobs)