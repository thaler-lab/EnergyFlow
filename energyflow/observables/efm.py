"""Implementation of Energy Flow Moments (EFMs)."""

from __future__ import absolute_import, division

import numpy as np

__all__ = ['EFM']

inds = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
metric = np.array([1.,-1.,-1.,-1.])
fake_p4s = np.zeros((10,4))

class EFM:

    def __init__(self, dim, nlow=0, normed=True, np_optimize='greedy'):

        assert nlow <= dim, 'nlow cannot be bigger than dim'
        
        self.dim = dim
        self.nlow = nlow
        self.nup = self.dim - self.nlow
        self.normed = normed

        self.einstr = ','.join([inds[0]]+[inds[0]+inds[i+1] for i in range(self.dim)])
        self.einpath = np.einsum_path(self.einstr1, fake_p4s[:,0], *[fake_p4s]*self.dim, optimize=np_optimize)[0]

    def construct(self, zs, p4hats_up, p4hats_low):
        p4hat_args = [p4hats_low]*self.nlow + [p4hats_up]*self.nup
        return np.einsum(self.einstr, zs, *p4hat_args, optimize=self.einpath)


