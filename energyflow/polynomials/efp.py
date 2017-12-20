from __future__ import absolute_import, division, print_function
import numpy as np

from energyflow.algorithms import VariableElimination
from energyflow.polynomials.base import EFPBase, EFPElem

__all__ = ['EFP']

class EFP(EFPBase):

    """A class for a storing and computing a single EFP."""

    def __init__(self, edges, measure='hadr', beta=1, normed=True, check_type=True, 
                              ve_alg='numpy', np_optimize='greedy'):
        """
        Arguments
        ----------
        edges : list
            Edges of the EFP graph specified by tuple-pairs of vertices.
        measure : string, optional
            See [Measures](/intro/measures) for options.
        beta : float, optional
            A value greater than zero. 
        normed : bool, optional
            Controls energy normalization.
        check_type : bool, optional
            Whether to check the type of the input or use the first input type.
        ve_alg : string, optional
            Controls which variable elimination algorithm is used, either `numpy` or `ef`. Leave as `numpy` unless you know what you're doing.
        np_optimize : string, optional
            When `ve_alg='numpy'` this is the `optimize` keyword of `numpy.einsum_path`
        """

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
