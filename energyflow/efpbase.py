"""Base and helper classes for EFPs."""

from __future__ import absolute_import

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import Counter
import multiprocessing
import os

import numpy as np
from six import add_metaclass

from energyflow.measure import Measure
from energyflow.utils import timing, transfer

__all__ = ['EFPBase', 'EFPElem']

sysname = os.uname()[0]


###############################################################################
# EFPBase
###############################################################################
@add_metaclass(ABCMeta)
class EFPBase:

    def __init__(self, measure, beta, kappa, normed, check_input):

        self.use_efpm_hybrid = 'efpm' in measure
        measure = measure.replace('efpm', 'efm')
        self.use_efms = 'efm' in measure

        # store measure object
        self._measure = Measure(measure, beta, kappa, normed, check_input)

        # store additional EFP measure object if using EFMs
        if self.use_efpm_hybrid:
            efp_measure_type = 'hadrdot' if 'hadr' in self.measure else 'ee'
            self._efp_measure = Measure(efp_measure_type, 2, kappa, normed, check_input)
        else:
            self._efp_measure = self._measure

    def get_zs_thetas_dict(self, event, zs, thetas):
        if event is not None:
            zs, thetas = self._efp_measure.evaluate(event)
        elif zs is None or thetas is None:
            raise TypeError('if event is None then zs and/or thetas cannot also be None')
        return zs, {w: thetas**w for w in self._weight_set}

    def construct_efms(self, event, zs, phats, efmset):
        if event is not None:
            zs, phats = self._measure.evaluate(event)
        elif zs is None or phats is None:
            raise TypeError('if event is None then zs and/or phats cannot also be None')
        return efmset.construct(zs, phats)

    @abstractproperty
    def _weight_set(self):
        pass

    @property
    def measure(self):
        return self._measure.measure

    @property
    def beta(self):
        return self._measure.beta

    @property
    def kappa(self):
        return self._measure.kappa

    @property
    def normed(self):
        return self._measure.normed

    @property
    def check_input(self):
        return self._measure.check_input

    @property
    def subslicing(self):
        return self._measure.subslicing

    def _batch_compute_func(self, event):
        return self.compute(event, batch_call=True)

    #def _compute_func_ps(self, args):
    #    return self.compute(zs=args[0], ps=args[1])

    @abstractmethod
    def compute(self, *args, **kwargs):
        """Computes the value(s) of the EFP(s) on a single event.

        **Arguments**

        - **event** : array_like or `fastjet.PseudoJet`
            - The event as an array of `[E,px,py,pz]` or `[pT,y,phi]` (if hadronic).
        - **zs** : 1-dim array_like
            - If present, `thetas` must also be present, and `zs` is used in place 
            of the energies of an event.
        - **thetas** : 2-dim array_like
            - If present, `zs` must also be present, and `thetas` is used in place 
            of the pairwise angles of an event.
        - **ps** : _numpy.ndarray_
            - If present, used in place of the dim-vectors returned by the measure
            when using EFMs.

        **Returns**

        - _numpy.ndarray_
            - The answers
        """

        pass

    @abstractmethod
    def batch_compute(self, events, n_jobs=-1):
        """Computes the value(s) of the EFP(s) on several events.

        **Arguments**

        - **events** : array_like or `fastjet.PseudoJet`
            - The events as an array of arrays of `[E,px,py,pz]` or `[pT,y,phi]` 
            (if hadronic).
        - **n_jobs** : int
            - The number of worker processes to use. A value of `-1` will attempt
            to use as many processes as there are CPUs on the machine.

        **Returns**

        - _numpy.ndarray_
            - The answers
        """

        if sysname != 'Linux' and self.use_efms:
            m = 'batch_compute currently not implemented for EFMs on {}!'.format(sysname)
            raise NotImplementedError(m)

        if n_jobs == -1:
            try: 
                self.n_jobs = multiprocessing.cpu_count()
            except:
                self.n_jobs = 4 # choose reasonable value

        # setup processor pool
        with multiprocessing.Pool(self.n_jobs) as pool:
            chunksize = max(len(events)//self.n_jobs, 1)
            results = np.asarray(list(pool.imap(self._batch_compute_func, events, chunksize)))

        return results


###############################################################################
# EFPElem
###############################################################################
class EFPElem:

    # if weights are given, edges are assumed to be simple 
    def __init__(self, edges, weights=None, einstr=None, einpath=None, k=None, 
                       efm_einstr=None, efm_einpath=None, efm_spec=None, M_thresh=0):

        transfer(self, locals(), ['einstr', 'einpath', 'k', 'M_thresh', 
                                  'efm_einstr', 'efm_einpath', 'efm_spec'])

        self.process_edges(edges, weights)

        self.pow2d = 2**self.d
        self.ndk = (self.n, self.d, self.k)

        self.use_efms = self.has_efms = self.efm_spec is not None
        if self.has_efms:
            self.efm_spec_set = frozenset(self.efm_spec)

    def process_edges(self, edges, weights):

        # deal with arbitrary vertex labels
        vertex_set = frozenset(v for edge in edges for v in edge)
        vertices = {v: i for i,v in enumerate(vertex_set)}
        self.n = len(vertex_set)

        # construct new edges with remapped vertices
        self.edges = [tuple(vertices[v] for v in sorted(edge)) for edge in edges]

        # get weights
        if weights is None:
            self.simple_edges = list(frozenset(self.edges))
            counts = Counter(self.edges)
            self.weights = tuple(counts[edge] for edge in self.simple_edges)

            # invalidate einsum quantities because edges got reordered
            self.einstr = self.einpath = None
        else:
            if len(weights) != len(self.edges):
                raise ValueError('length of weights is not number of edges')
            self.simple_edges = self.edges
            self.weights = tuple(weights)
        self.edges = [e for w,e in zip(self.weights, self.simple_edges) for i in range(w)]

        self.e = len(self.simple_edges)
        self.d = sum(self.weights)
        self.weight_set = frozenset(self.weights)

    def determine_efm_compute(self, M):
        self.use_efms = M >= self.M_thresh
        return self.use_efms

    def compute(self, zs, thetas_dict, efms_dict):
        return self.efm_compute(efms_dict) if self.use_efms else self.efp_compute(zs, thetas_dict)

    def set_timer(self):
        self.times = []
        self.efp_compute = timing(self, self.efp_compute)
        self.efm_compute = timing(self, self.efm_compute)

    def efp_compute(self, zs, thetas_dict):
        einsum_args = [thetas_dict[w] for w in self.weights] + self.n*[zs]
        return np.einsum(self.einstr, *einsum_args, optimize=self.einpath)

    def efm_compute(self, efms_dict):
        einsum_args = [efms_dict[sig] for sig in self.efm_spec]
        return np.einsum(self.efm_einstr, *einsum_args, optimize=self.efm_einpath)*self.pow2d

    # properties set above:
    #     n, e, d, k, ndk, edges, simple_edges, weights, weight_set, einstr, einpath,
    #     efm_einstr, efm_einpath, efm_spec, M_thresh
