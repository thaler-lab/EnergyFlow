from __future__ import absolute_import

from abc import ABCMeta, abstractmethod
from collections import Counter
import multiprocessing as mp

import numpy as np
from six import add_metaclass

from energyflow.utils.measure import Measure

@add_metaclass(ABCMeta)
class EFPBase:

    def __init__(self, measure='hadr', beta=1.0, normed=True, check_type=False):

        # store measure object
        self.measure = Measure(measure, beta, normed, check_type)

    def _get_zs_thetas_dict(self, event, zs, thetas):
        if event is not None:
            zs, thetas = self.measure(event)
        elif zs is None or thetas is None:
            raise TypeError('if event is None then zs and/or thetas cannot also be None')
        thetas_dict = {w: thetas**w for w in self.weight_set}
        return zs, thetas_dict

    def _compute_func(self, args):
        return self.compute(zs=args[0], thetas=args[1])

    @abstractmethod
    def compute(self, *args):
        pass

    def batch_compute(self, events=None, zs=None, thetas=None, n_jobs=None):

        if events is not None:
            iterable = [self.measure(event) for event in events]
        elif zs is None or thetas is None:
            raise TypeError('if events is None then zs and/or thetas cannot also be None')
        else:
            iterable = zip(zs, thetas)

        processes = n_jobs
        if processes is None:
            try: 
                processes = mp.cpu_count()
            except:
                processes = 4 # choose reasonable value

        # setup processor pool
        with mp.Pool(processes) as pool:
            chunksize = int(len(iterable)/processes)
            results = np.asarray(list(pool.imap(self._compute_func, iterable, chunksize)))

        return results

class EFPElem:

    # if weights are given, edges are assumed to be simple 
    def __init__(self, edges, weights=None, einstr=None, einpath=None, k=None):
        self.einstr, self.einpath = einstr, einpath
        self.k = k

        self._remap_edges(edges)
        self._get_simple_edges()
        self._get_weights(weights)

        if self.k is not None:
            self.ndk = (self.n, self.d, self.k)

    def _remap_edges(self, edges):

        # deal with arbitrary vertex labels
        vertex_set = set(v for edge in edges for v in edge)
        vertices = {v: i for i,v in enumerate(sorted(list(vertex_set)))}
        self.n = len(vertex_set)

        # construct new edges with remapped vertices
        self.edges = sorted([tuple(vertices[v] for v in sorted(edge)) for edge in edges])

    def _get_simple_edges(self):
        self.simple_edges = sorted(list(set(self.edges)))
        self.e = len(self.simple_edges)

    # get weights of edges
    def _get_weights(self, weights):
        if weights is None:
            counts = Counter(self.edges)
            self.weights = tuple(counts[edge] for edge in self.simple_edges)
        else:
            if len(weights) != self.e:
                raise ValueError('length of weights is not number of simple edges')
            self.weights = tuple(weights)
        self.d = sum(self.weights)
        self.weight_set = set(self.weights)

    def compute(self, zs, thetas_dict):
        einsum_args = [thetas_dict[w] for w in self.weights] + self.n*[zs]
        return np.einsum(self.einstr, *einsum_args, optimize=self.einpath)




 