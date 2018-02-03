"""Implementation of EFP."""

from __future__ import absolute_import, division, print_function
import numpy as np

from energyflow.algorithms import VariableElimination
from energyflow.polynomials.base import EFPBase, EFPElem

__all__ = ['EFP']

class EFP(EFPBase):

    """A class for representing and computing a single EFP."""

    def __init__(self, edges, nfree=0, measure='hadr', beta=1, normed=True, check_type=True, 
                              ve_alg='numpy', np_optimize='greedy'):
        """
        Arguments
        ----------
        edges : list
            - Edges of the EFP graph specified by tuple-pairs of vertices.
        measure : {`'hadr'`, `'hadr-dot'`, `'ee'`}
            - See [Measures](/intro/measures) for additional info.
        beta : float
            - The parameter $\\beta$ appearing in the measure. 
            Must be greater than zero.
        normed : bool
            - Controls normalization of the energies in the measure.
        check_type : bool
            - Whether to check the type of the input each time or use 
            the first input type.
        ve_alg : {`'numpy'`, `'ef'`}
            - Which variable elimination algorithm to use.
        np_optimize : {`True`, `False`, `'greedy'`, `'optimal'`}
            - When `ve_alg='numpy'` this is the `optimize` keyword 
            of `numpy.einsum_path`.
        """

        # initialize EFPBase
        super().__init__(measure, beta, normed, check_type)

        # store these edges as an EFPElem
        self.efpelem = EFPElem(edges)

        # get ve instance
        self.ve = VariableElimination(ve_alg, np_optimize)

        # set internals of ve to these edges
        self.ve.run(self.simple_graph, self.n, nfree)
        self.efpelem.einstr, self.efpelem.einpath = self.ve.einspecs()

    #===============
    # public methods
    #===============

    def compute(self, event=None, zs=None, thetas=None):
        
        # get dictionary of thetas to use for event
        zs, thetas_dict = self._get_zs_thetas_dict(event, zs, thetas)

        # call compute on the EFPElem
        return self.efpelem.compute(zs, thetas_dict)

    def batch_compute(self, events=None, zs=None, thetas=None, n_jobs=-1):

        return super().batch_compute(events, zs, thetas, n_jobs)

    #===========
    # properties
    #===========

    @property
    def _weight_set(self):
        """Set of edge weights for the graph of this EFP."""

        return self.efpelem.weight_set

    @property
    def graph(self):
        """Graph of this EFP represented by a list of edges."""

        return self.efpelem.edges

    @property
    def simple_graph(self):
        """Simple graph of this EFP (forgetting all multiedges)
        represented by a list of edges."""

        return self.efpelem.simple_edges

    @property
    def n(self):
        """Number of vertices in the graph of this EFP."""

        return self.efpelem.n

    @property
    def d(self):
        """Degree, or number of edges, in the graph of this EFP."""

        return self.efpelem.d

    @property
    def c(self):
        """VE complexity $\\chi$ of this EFP."""

        if hasattr(self.ve, 'chi'):
            return self.ve.chi
        else:
            return None
