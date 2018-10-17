"""Implementation of Variable Elimination (VE) Algorithm."""
from __future__ import absolute_import, division

import itertools

import numpy as np

from energyflow.algorithms.einsumfunc import einsum_path
from energyflow.utils import igraph_import

igraph = igraph_import()

__all__ = ['VariableElimination']

# allowed einsum symbols
I = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

###############################################################################
# VariableElimination
###############################################################################
class VariableElimination(object):

    def __init__(self, np_optimize='greedy'):

        self.np_optimize = np_optimize
        self.X = np.empty((2,2))
        self.y = np.empty(2)

    def _einstr_from_edges(self, edges, n):
        einstr  = ','.join([I[j]+I[k] for (j, k) in edges])
        einstr += ',' if len(edges) else ''
        einstr += ','.join([I[v] for v in range(n)])
        einstr += '->' if len(edges) == 0 else ''
        return einstr

    def einspecs(self, edges, n):
        d = len(edges)
        args = [self.X]*d + [self.y]*n

        einstr = self._einstr_from_edges(edges, n)
        einpath = einsum_path(einstr, *args, optimize=self.np_optimize)
        chi = int(einpath[1].split('\n')[2].split(':')[1])

        return einstr, einpath[0], chi
