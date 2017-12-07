from __future__ import absolute_import, division, print_function
import numpy as np

from energyflow.algorithms import VariableElimination
from energyflow.polynomials.base import EFPBase, EFPElem

__all__ = ['EFP']

class EFP(EFPBase):

    def __init__(self, edges, measure='hadr', beta=1.0, normed=True, check_type=True, 
                              ve_alg='numpy', np_optimize='greedy'):

        # initialize base classes
        super().__init__(measure, beta, normed, check_type)

        # store these edges as an EFPElem
        self.efpelem = EFPElem(edges)
        self.weight_set = self.efpelem.weight_set

        # get ve instance
        self.ve = VariableElimination(ve_alg, np_optimize)

        # set internals of ve to these edges
        self.ve.run(self.efpelem.simple_edges, self.efpelem.n)
        self.efpelem.einstr, self.efpelem.einpath = self.ve.einspecs()

    def compute(self, event=None, zs=None, thetas=None):
        zs, thetas_dict = self._get_zs_thetas_dict(event, zs, thetas)
        return self.efpelem.compute(zs, thetas_dict)
