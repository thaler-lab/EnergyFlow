"""Base classes for EFPs."""

from __future__ import absolute_import

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import Counter
import multiprocessing

import numpy as np
from six import add_metaclass

from energyflow.utils import Measure, transfer, timing

__all__ = ['EFPBase', 'EFPElem']

@add_metaclass(ABCMeta)
class EFPBase:

    def __init__(self, measure, beta, kappa, normed, check_input):

        # store measure object
        self._measure = Measure(measure, beta, kappa, normed, check_input)

    def _get_zs_thetas_dict(self, event, zs, thetas):
        if event is not None:
            zs, thetas = self._measure(event)
        elif zs is None or thetas is None:
            raise TypeError('if event is None then zs and/or thetas cannot also be None')
        return zs, {w: thetas**w for w in self.weight_set}

    @abstractproperty
    def weight_set(self):
        pass

    def construct_efms(self, event, zs, phats):
        if event is not None:
            zs, phats = self._measure(event)
        elif zs is None or phats is None:
            raise TypeError('if event is None then zs and/or phats cannot also be None')
        return self.efmset.construct(zs, phats)

    @abstractproperty
    def efmset(self):
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

    def _compute_func(self, args):
        return self.compute(zs=args[0], angles=args[1])

    @abstractmethod
    def compute(self, *args):
        """Computes the value(s) of the EFP(s) on a single event.

        Arguments
        ---------
        event : array_like or `fastjet.PseudoJet`
            - The event or jet as an array or `PseudoJet`.
        zs : 1-dim array_like
            - If present, `thetas` must also be present, and `zs` is used in place 
            of the energies of an event.
        thetas : 2-dim array_like
            - If present, `zs` must also be present, and `thetas` is used in place 
            of the pairwise angles of an event.
        """

        pass

    @abstractmethod
    def batch_compute(self, events, n_jobs=-1):
        """Computes the value(s) of the EFP(s) on several events.

        Arguments
        ---------
        events : array_like or `fastjet.PseudoJet`
            - The events or jets as an array or list of `PseudoJet`s.
        n_jobs : int
            - The number of worker processes to use. A value of `-1` will attempt
            to use as many processes as there are CPUs on the machine.
        """

        iterable = [self._measure(event) for event in events]
        length = len(events)

        if n_jobs == -1:
            try: 
                n_jobs = multiprocessing.cpu_count()
            except:
                n_jobs = 4 # choose reasonable value

        # setup processor pool
        self._n_jobs = n_jobs
        with multiprocessing.Pool(n_jobs) as pool:
            chunksize = int(length/n_jobs)
            results = np.asarray(list(pool.imap(self._compute_func, iterable, chunksize)))

        return results

class EFPElem:

    # if weights are given, edges are assumed to be simple 
    def __init__(self, edges, weights=None, einstr=None, einpath=None, k=None, 
                       efm_einstr=None, efm_einpath=None, efm_spec=None):

        transfer(self, locals(), ['einstr','einpath','k','efm_einstr','efm_einpath','efm_spec'])

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
        self.pow2 = 2**self.d
        self.weight_set = frozenset(self.weights)

        self.ndk = (self.n, self.d, self.k)

        self.use_efms = self.efm_einstr is not None
        self.set_compute()

    def set_timer(self, repeat, number):
        self.set_compute()
        self.compute = timing(self, repeat, number)(self.compute)

    def set_compute(self):
        self.compute = self.efm_compute if self.use_efms else self.efp_compute

    def efp_compute(self, zs, thetas_dict):
        einsum_args = [thetas_dict[w] for w in self.weights] + self.n*[zs]
        return np.einsum(self.einstr, *einsum_args, optimize=self.einpath)

    def efm_compute(self, efms_dict):
        einsum_args = [efms_dict[sig] for sig in self.efm_spec]
        return np.einsum(self.efm_einstr, *einsum_args, optimize=self.efm_einpath)*self.pow2

    # properties set above:
    #     n, e, d, k, ndk, edges, simple_edges, weights, weight_set, einstr, einpath,
    #     efm_einstr, efm_einpath, efm_spec
