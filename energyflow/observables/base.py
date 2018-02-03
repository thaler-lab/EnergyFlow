from __future__ import absolute_import

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import Counter
import multiprocessing as mp

import numpy as np
from six import add_metaclass

from energyflow.utils.measure import Measure

@add_metaclass(ABCMeta)
class EFPBase:

    def __init__(self, measure='hadr', beta=1.0, normed=True, check_type=False):

        # store measure object
        self._measure = Measure(measure, beta, normed, check_type)

    def _get_zs_thetas_dict(self, event, zs, thetas):
        if event is not None:
            zs, thetas = self._measure(event)
        elif zs is None or thetas is None:
            raise TypeError('if event is None then zs and/or thetas cannot also be None')
        thetas_dict = {w: thetas**w for w in self._weight_set}
        return zs, thetas_dict

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
    def normed(self):
        return self._measure.normed

    @property
    def check_type(self):
        return self._measure.check_type

    def _compute_func(self, args):
        return self.compute(zs=args[0], thetas=args[1])

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
    def batch_compute(self, events=None, zs=None, thetas=None, n_jobs=-1):
        """Computes the value(s) of the EFP(s) on several events.

        Arguments
        ---------
        events : array_like or `fastjet.PseudoJet`
            - The events or jets as an array or list of `PseudoJet`s.
        zs : list of 1-dim array
            - If present, `thetas` must also be present, and `zs` is used in place 
            of the energies of the events.
        thetas : list of 2-dim array
            - If present, `zs` must also be present, and `thetas` is used in place 
            of the pairwise angles of the events.
        n_jobs : int
            - The number of worker processes to use. A value of `-1` will attempt
            to use as many processes as there are CPUs on the machine.
        """

        if events is not None:
            iterable = [self._measure(event) for event in events]
            length = len(events)
        elif zs is None or thetas is None:
            raise TypeError('if events is None then zs and/or thetas cannot also be None')
        else:
            iterable = zip(zs,thetas)
            length = min(len(zs),len(thetas))

        if n_jobs == -1:
            try: 
                n_jobs = mp.cpu_count()
            except:
                n_jobs = 4 # choose reasonable value

        # setup processor pool
        self._n_jobs = n_jobs
        with mp.Pool(n_jobs) as pool:
            chunksize = int(length/n_jobs)
            results = np.asarray(list(pool.imap(self._compute_func, iterable, chunksize)))

        return results

class EFPElem:

    # if weights are given, edges are assumed to be simple 
    def __init__(self, edges, weights=None, einstr=None, einpath=None, k=None):

        self.einstr, self.einpath, self.k = einstr, einpath, k

        # deal with arbitrary vertex labels
        vertex_set = set(v for edge in edges for v in edge)
        vertices = {v: i for i,v in enumerate(sorted(list(vertex_set)))}
        self.n = len(vertex_set)

        # construct new edges with remapped vertices
        self.edges = sorted([tuple(vertices[v] for v in sorted(edge)) for edge in edges])

        # get simple edges
        self.simple_edges = sorted(list(set(self.edges)))
        self.e = len(self.simple_edges)

        # get weights
        if weights is None:
            counts = Counter(self.edges)
            self.weights = tuple(counts[edge] for edge in self.simple_edges)
        else:
            if len(weights) != self.e:
                raise ValueError('length of weights is not number of simple edges')
            self.weights = tuple(weights)
            self.edges = [e for w,e in zip(self.weights, self.simple_edges) for i in range(w)]

        self.d = sum(self.weights)
        self.weight_set = frozenset(self.weights)

        if self.k is not None:
            self.ndk = (self.n, self.d, self.k)

    def compute(self, zs, thetas_dict):
        einsum_args = [thetas_dict[w] for w in self.weights] + self.n*[zs]
        return np.einsum(self.einstr, *einsum_args, optimize=self.einpath)
