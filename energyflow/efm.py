"""Implementation of Energy Flow Moments (EFMs)."""

from __future__ import absolute_import, division

from operator import itemgetter

import numpy as np
from numpy.core.multiarray import c_einsum

from energyflow.utils.graph import *
from energyflow.utils.helpers import *

__all__ = ['EFM', 'EFMSet', 'efp2efms']

inds = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

class EFM:

    def __init__(self, nlow, nup, raw=False, rlfrom=None):

        # store inputs
        self.nlow = nlow
        self.nup = nup
        self.raw = raw
        self.rlfrom = rlfrom

        # get useful derived quantities
        self.v = self.nlow + self.nup
        self.sig = (self.nlow, self.nup)

        if self.raw:
            self.raw_einstr = ','.join([inds[0]] + [inds[0] + inds[i+1] for i in range(self.v)])
            self.raw_einpath = ['einsum_path'] + [(0,1)]*self.v
            self.construct = self.raw_construct
        elif self.rlfrom is not None:
            if self.v != sum(self.rlfrom):
                raise ValueError('cannot raise/lower among different valency EFMs')
            other_nlow = rlfrom[0]
            diff = self.nlow - other_nlow
            self.rl_diff = abs(diff)
            lowering = diff > 0
            i1 = other_nlow if lowering else self.nlow
            i2 = self.nlow if lowering else other_nlow
            self.rl_einstr = ','.join([inds[:self.v]] + list(inds[i1:i2])) + '->' + inds[:self.v]
            self.construct = self.rl_construct
        else:
            self.construct = self.subslice_construct

    def raw_construct(self, zsphats):
        zs, phats = zsphats
        self.data = np.einsum(self.raw_einstr, zs, *[phats]*self.v, optimize=self.raw_einpath)
        return self.data

    def rl_construct(self, other_efm):       
        metric = flat_metric(len(other_efm.data))
        self.data = c_einsum(self.rl_einstr, other_efm.data, *[metric]*self.rl_diff)
        return self.data

    def subslice_construct(self, big_efm):
        num_up_subslices = big_efm.nup - self.nup
        num_low_subslices = big_efm.nlow - self.nlow

        # perform check
        if num_up_subslices < 0 or num_low_subslices < 0:
            m = 'cannot perform subslicing from {} to {}'.format(big_efm.sig, self.sig)
            raise RuntimeError(m)

        s = [0]*num_low_subslices + [Ellipsis] + [0]*num_up_subslices
        self.data = big_efm.data[s]
        return self.data

class EFMSet:

    def __init__(self, efm_specs, subslicing=False):

        # store inputs
        self.subslicing = subslicing

        # get unique EFMs 
        self.unique_efms = set(efm_specs)

        # setup EFMs based on whether we can subslice or not
        self.efms, self.efm_args = {}, {}
        if self.subslicing:
            self.subslicing_setup()
        else:
            self.full_setup()

    def subslicing_setup(self):

        # ensure there is at least one EFM of each valency for rl purposes
        maxsig = max(self.unique_efms, key=itemgetter(1))
        self.unique_efms |= set((0,n) for n in range(1,maxsig[1]))

        # sort EFMs by decreasing valency and then increasing nlow
        self.sorted_efms = sorted(self.unique_efms, key=itemgetter(1), reverse=True)
        self.sorted_efms.sort(key=itemgetter(0))

        # take care of empty set
        if not len(self.sorted_efms):
            return

        # the first one must be raw constructed
        sig0 = self.sorted_efms[0]
        self.efms[sig0] = EFM(*sig0, raw=True)
        self.efm_args[sig0] = 'r'

        for sig in self.sorted_efms[1:]:

            # determine if we can subslice
            big_sig = find_subslice(sig, self.efms)
            if big_sig is not None:
                self.efms[sig] = EFM(*sig)
                self.efm_args[sig] = self.efms[big_sig]

            # find best raise/lower available
            else:
                rlsig = find_minimum_rl(sig, self.efms)
                self.efms[sig] = EFM(*sig, rlfrom=rlsig)
                self.efm_args[sig] = self.efms[rlsig]

    def full_setup(self):
        self.sorted_efms = sorted(self.unique_efms, key=itemgetter(0))
        self.sorted_efms.sort(key=sum)

        vprev, sigprev = None, None
        for sig in self.sorted_efms:
            v = sum(sig)

            # construct raw
            if v != vprev:
                self.efms[sig] = EFM(*sig, raw=True)
                self.efm_args[sig] = 'r'

            # construct from lowering
            else:
                self.efms[sig] = EFM(*sig, rlfrom=sigprev)
                self.efm_args[sig] = self.efms[sigprev]

            vprev, sigprev = v, sig

    def construct(self, zs, phats):
        data = {}
        zsphats = (zs, phats)
        for sig in self.sorted_efms:
            arg = self.efm_args[sig]
            if arg == 'r':
                arg = zsphats
            data[sig] = self.efms[sig].construct(arg)
        return data

def efp2efms(graph):

    # build convenient data structure to hold graph information
    vds = get_valency_structure(graph)

    # dictionary to hold efm terms
    efms = {}

    # counter to store how to get fresh dummy indices
    ind = 0

    # iterate over vertices sorted by valency in decreasing order
    sorted_verts = sorted(valencies(graph).items(), key=lambda x: x[1], reverse=True)
    for vert,valency in sorted_verts:

        # dict holding info for new efm term
        new_efm = {'upper_indices': '', 'lower_indices': ''}

        # iterate over neighboring vertices
        for neighbor,n_shared_edges in vds[vert].items():

            # if a new neighbor, assign fresh inds
            if neighbor not in efms:
                new_inds = inds[ind:ind+n_shared_edges]
                ind += n_shared_edges
                new_efm['upper_indices'] += new_inds

                # store inds shared with that neighbor
                new_efm[neighbor] = new_inds

            # if neighbor already has an efm factor, add already assigned indices to lower_indices
            else:
                new_efm['lower_indices'] += efms[neighbor][vert]

        # store new efm factor
        efms[vert] = new_efm

    einstr_list, efm_specs = [], []
    for vert,valency in sorted_verts:
        efm = efms[vert]
        lower_indices = efm['lower_indices']

        # conventionally put lowered indices before upper indices
        einstr_list.append(lower_indices + efm['upper_indices'])

        # add spec which is (nlow, nup) of efm
        nlow = len(lower_indices)
        efm_specs.append((nlow, valency - nlow))

    # return comma joined einstr and efm_specs
    return ','.join(einstr_list), efm_specs
