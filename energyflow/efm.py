"""Implementation of Energy Flow Moments (EFMs)."""

from __future__ import absolute_import, division

import numpy as np
from numpy.core.multiarray import c_einsum

from energyflow.utils import valencies, vv_counts

__all__ = ['EFM', 'efp_as_efms']

inds = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

class EFM:

    metric = np.array([1.,-1.,-1.,-1.])

    def __init__(self, dim, nlows=None):

        self.dim = dim
        self.nlows = list(set([0]) | (set([] if nlows is None else nlows) & 
                                      set(range(self.dim+1))))

        # dictionaries to hold quantities for different numbers of lowered indices
        self.einstr_dict = {0: ','.join([inds[0]] + [inds[0] + inds[i+1] \
                                         for i in range(self.dim)])}
        self.einpath_0 = ['einsum_path'] + [(0,1)]*self.dim

        for i in range(1,len(self.nlows)):
            prev = self.nlows[i-1]
            nlow = self.nlows[i]
            self.einstr_dict[nlow] = (','.join([inds[:self.dim]] + list(inds[prev:nlow]))
                                      + '->' + inds[:self.dim])

    def construct(self, zs, p4hats):
        d = {0: np.einsum(self.einstr_dict[0], zs, *[p4hats]*self.dim, optimize=self.einpath_0)}
        for i in range(1,len(self.nlows)):
            prev = self.nlows[i-1]
            nlow = self.nlows[i]
            d[nlow] = c_einsum(self.einstr_dict[nlow], d[prev], *[self.metric]*(nlow - prev))
        return d

def efp_as_efms(graph):
    vert_valencies = sorted(valencies(graph).items(), key=lambda x: x[1], reverse=True)
    vert_vert_counts = vv_counts(graph)
    efms = {}
    ind = 0
    for v,valency in vert_valencies:
        new_efm = {'dim': valency, 'upper_indices': '', 'lower_indices': ''}
        for neighbor,count in vert_vert_counts[v].items():
            if neighbor in efms:
                new_efm['lower_indices'] += efms[neighbor][v]
            else:
                new_inds = inds[ind:ind+count]
                ind += count
                new_efm['upper_indices'] += new_inds
                new_efm[neighbor] = new_inds
        efms[v] = new_efm

    einstr_list, efm_specs = [], []
    for vv in vert_valencies:
        efm = efms[vv[0]]
        lower_indices = efm['lower_indices']
        einstr_list.append(lower_indices + efm['upper_indices'])
        efm_specs.append((efm['dim'], len(lower_indices)))
    einstr = ','.join(einstr_list)
    return einstr, efm_specs
