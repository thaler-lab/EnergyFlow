"""A submodule for computing the Earth Mover's Distance."""
from __future__ import absolute_import, division, print_function

import warnings

import numpy as np

try:
    import ot
    from scipy.spatial.distance import cdist # ot imports this anyway
except:
    warnings.warn('cannot import module \'ot\', module \'emd\' unavailable')
    ot = False

if ot:
    __all__ = ['emd', 'pairwise_emds']

    def emd(ev0, ev1, R=1.0, norm=False, return_flow=False):
        """Compute the EMD between two events.

        **Arguments**

        - **ev0**, **ev1** : _numpy.ndarray_
            - The two events to compute the EMD between, given as two-dimensional
            arrays.
        - **R** : _float_
            - The R parameter.
        - **norm** : _bool_
            - Flag
        - **return_flow** : _bool_
            - Flag

        **Returns**

        - _float_
            - The EMD.
        """

        pTs0, pTs1 = np.ascontiguousarray(ev0[:,0]), np.ascontiguousarray(ev1[:,0])
        coords0, coords1 = ev0[:,1:], ev1[:,1:]

        pT0, pT1 = pTs0.sum(), pTs1.sum()
        if norm:
            pTs0 /= pT0
            pTs1 /= pT1
            thetas = cdist(coords0, coords1, metric='euclidean')/R

        else:
            pTdiff = pT1 - pT0
            if pTdiff > 0:
                pTs0 = np.concatenate((pTs0, pTdiff))
                thetas = cdist(np.vstack((coords0, np.zeros(coords.shape[1]))), coords1, metric='euclidean')/R
                thetas[-1,:] = 1.0

            elif pTdiff < 0:
                pTs1 = np.concatenate((pTs1, -pTdiff))
                thetas = cdist(coords0, np.vstack((coords1, np.zeros(coords1.shape[1]))), metric='euclidean')/R
                thetas[:,-1] = 1.0

            else:
                thetas = cdist(coords0, coords1, metric='euclidean')/R

        if return_flow:
            G, log = ot.emd(pTs0, pTs1, thetas, log=True)
            return log['cost'], G
        else:
            return ot.emd2(pTs0, pTs1, thetas)

    def _emd(ev0, ev1, R, normed):

        pTs0, coords0 = ev0
        pTs1, coords1 = ev1

        if normed:
            thetas = cdist(coords0, coords1, metric='euclidean')/R

        else:
            pT0, pT1 = pTs0.sum(), pTs1.sum()
            pTdiff = pT1 - pT0
            if pTdiff > 0:
                pTs0[-1] = pTdiff
            elif pTdiff < 0:
                pTs1[-1] = -pTdiff
            thetas = cdist(coords0, coords1, metric='euclidean')/R
            thetas[:,-1] = 1.0
            thetas[-1,:] = 1.0

        cost = ot.emd2(pTs0, pTs1, thetas)
        pTs0[-1] = pTs1[-1] = 0
        return cost

    def _process_for_emd(event, norm):
        if norm:
            pts = event[:,0]/event[:,0].sum()
        else:
            event = np.vstack((event, np.zeros(event.shape[1])))
            pts = event[:,0]

        return np.ascontiguousarray(pts), event[:,1:]

    def pairwise_emds(X0, X1=None, R=1.0, norm=False):
        X0 = [_process_for_emd(x, norm) for x in X0]
        if X1 is None:
            emds = np.zeros((len(X0), len(X0)))
            for i in range(len(X0)):
                for j in range(i):
                    emds[i,j] = _emd(X0[i], X0[j], R, norm)
            emds += emds.T
        else:
            X1 = [_process_for_emd(x, norm) for x in X1]
            emds = np.zeros((len(X0), len(X1)))
            for i in range(len(X0)):
                for j in range(len(X1)):
                    emds[i,j] = _emd(X0[i], X1[j], R, norm)
        return emds

else:
    __all__ = []
