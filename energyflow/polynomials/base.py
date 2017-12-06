from __future__ import absolute_import
import abc
import multiprocessing as mp

import numpy as np
import six

from energyflow.utils import Measure

@six.add_metaclass(abc.ABCMeta)
class EFPBase:

    def __init__(self, measure='hadr', beta=1.0, normed=True, check_type=False):

        # store measure object
        self.measure = Measure(measure, beta, normed, check_type)

def _make_thetas_dict(self, thetas):
    return {1: thetas}

def _compute_func(self, arg):
    return self.compute(zs_thetas_d=arg)

def batch_compute(self, events, concat_disc=True, n_jobs=None):

    processes = n_jobs
    if processes is None:
        try: 
            processes = mp.cpu_count()
        except:
            processes = 4 # choose reasonable value

    # construct iterable from events
    iterable = []
    for event in events:
        zs, thetas = self.zs_thetas(event)
        thetas_d = self._make_thetas_dict(thetas)
        iterable.append([zs, thetas_d])

    # setup processor pool
    with mp.Pool(processes) as pool:
        chunksize = int(len(events)/processes)
        results = np.asarray(list(pool.imap(self._compute_func, iterable, chunksize)))

    if concat_disc:
        return self.calc_disc(results, self.disc_formulae, concat=True)
    else: 
        return results

class EFPElem:

    def __init__(self, n, e, d, k, g, w, c, p, edges=None, weights=None):
        self.n, self.e, self.d, self.k, self.g, self.w, self.c, self.p = (n, e, d, k, g, w, c, p)
        self.edges = edges
        self.weights = weights





 