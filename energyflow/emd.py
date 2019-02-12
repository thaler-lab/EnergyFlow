r"""The Energy Mover's Distance (EMD), also known as the Earth Mover's Distance,
is a metric between particle collider events introduced in [1902.02346](https:
//arxiv.org/abs/1902.02346). This submodule contains convenient functions for 
computing EMDs between individual events and collections of events. The core
of the computation is done using the [Python Optimal Transport (POT)](https:
//pot.readthedocs.io) library, which must be installed in order to use this
submodule.

From Eq. 1 in [1902.02346](https://arxiv.org/abs/1902.02346), the EMD between
two events is the minimum ''work'' required to rearrange one event $\mathcal E$
into the other $\mathcal E'$ by movements of energy $f_{ij}$ from particle $i$ 
in one event to particle $j$ in the other:
$$
\text{EMD} = \min_{\{f_{ij}\}}\sum_{ij}f_{ij}\frac{\theta_{ij}}{R} + 
\left|\sum_iE_i-\sum_jE'_j\right|,\\
f_{ij}\ge 0, \quad \sum_jf_{ij}\le E_i, \quad \sum_if_{ij}\le E'_j, \quad 
\sum_{ij}f_{ij}=E_\text{min},
$$
where $E_i,\,E'_j$ are the energies of the particles in the two events, 
$\theta_{ij}$ is an angular distance between particles, and 
$E_\text{min}=\min\left(\sum_iE_i,\,\sum_jE'_j\right)$ is the smaller of the 
two total energies. In a hadronic context, transverse momenta are used 
instead of energies.
"""
from __future__ import absolute_import, division, print_function

import itertools
import multiprocessing
import time
import warnings

import numpy as np

try:
    import ot
    from scipy.spatial.distance import _distance_wrap # ot imports scipy anyway
except:
    warnings.warn('cannot import module \'ot\', module \'emd\' will be empty')
    ot = False

if ot:

    __all__ = ['emd', 'pairwise_emds']

    # faster than scipy's cdist function because we can guarantee things and avoid their checks
    def _cdist_euclidean(X, Y):
        out = np.empty((len(X), len(Y)), dtype=np.double)
        _distance_wrap.cdist_euclidean_double_wrap(X, Y, out)
        return out

    def emd(ev0, ev1, R=1.0, norm=False, return_flow=False):
        """Compute the EMD between two events.

        **Arguments**

        - **ev0** : _numpy.ndarray_
            - The first event, given as a two-dimensional array.
        - **ev1** : _numpy.ndarray_
            - The other event, same format as **ev0**.
        - **R** : _float_
            - The R parameter in Eq. 1 that controls the relative importance of
            the two terms. Must be greater than or equal to half of the maximum
            ground distance in the space in order for the EMD to be a valid metric.
        - **norm** : _bool_
            - Whether or not to normalize the pT values of the events prior to 
            computing the EMD.
        - **return_flow** : _bool_
            - Whether or not to return the flow matrix describing the optimal 
            transport found during the computation of the EMD. Note that since
            the second term in Eq. 1 is implemented by including an additional 
            particle in the event with lesser total pT, this will be reflected 
            in the flow matrix.

        **Returns**

        - _float_
            - The EMD value.
        - [_numpy.ndarray_]
            - The flow matrix found while solving for the EMD. The `(i,j)`th 
            entry is the amount of `pT` that flows between particle i in `ev0`
            and particle j in `ev1`.
        """

        ev0, ev1 = np.atleast_2d(ev0), np.atleast_2d(ev1)
        pTs0 = np.ascontiguousarray(ev0[:,0], dtype=np.double)
        pTs1 = np.ascontiguousarray(ev1[:,0], dtype=np.double)
        coords0 = np.ascontiguousarray(ev0[:,1:], dtype=np.double)
        coords1 = np.ascontiguousarray(ev1[:,1:], dtype=np.double)

        pT0, pT1 = pTs0.sum(), pTs1.sum()

        # if norm, then we normalize the pts to 1
        if norm:
            pTs0 /= pT0
            pTs1 /= pT1
            thetas = _cdist_euclidean(coords0, coords1)/R

        else:

            # implement the EMD in Eq. 1 of the paper by adding an appropriate extra particle
            pTdiff = pT1 - pT0
            if pTdiff > 0:
                pTs0 = np.concatenate((pTs0, pTdiff))
                coords0_extra = np.vstack((coords0, np.zeros(coords.shape[1], dtype=np.double)))
                thetas = _cdist_euclidean(coords0_extra, coords1)/R
                thetas[-1,:] = 1.0

            elif pTdiff < 0:
                pTs1 = np.concatenate((pTs1, -pTdiff))
                coords1_extra = np.vstack((coords1, np.zeros(coords1.shape[1], dtype=np.double)))
                thetas = _cdist_euclidean(coords0, coords1_extra)/R
                thetas[:,-1] = 1.0

            # in this case, the pts were exactly equal already so no need to add a particle
            else:
                thetas = _cdist_euclidean(coords0, coords1)/R

        if return_flow:
            G, log = ot.emd(pTs0, pTs1, thetas, log=True)
            return log['cost'], G
        else:
            return ot.emd2(pTs0, pTs1, thetas)

    # process events for EMD calculation using _emd
    def _process_for_emd(event, norm):
        event = np.atleast_2d(event)
        if norm:
            pts = event[:,0]/event[:,0].sum()
        else:
            event = np.vstack((event, np.zeros(event.shape[1])))
            pts = event[:,0]

        return (np.ascontiguousarray(pts, dtype=np.double), 
                np.ascontiguousarray(event[:,1:], dtype=np.double))

    # internal use only by pairwise_emds, makes assumptions about input format
    def _emd(ev0, ev1, R, normed):

        pTs0, coords0 = ev0
        pTs1, coords1 = ev1

        # pts already normed if going in here
        if normed:
            thetas = _cdist_euclidean(coords0, coords1)/R

        # extra particles (with zero pt) already added if going in here
        else:
            pT0, pT1 = pTs0.sum(), pTs1.sum()
            pTdiff = pT1 - pT0
            if pTdiff > 0:
                pTs0[-1] = pTdiff
            elif pTdiff < 0:
                pTs1[-1] = -pTdiff
            thetas = _cdist_euclidean(coords0, coords1)/R
            thetas[:,-1] = 1.0
            thetas[-1,:] = 1.0

        # compute the emd with POT
        cost = ot.emd2(pTs0, pTs1, thetas)

        # important! must reset extra particles to have pt zero
        pTs0[-1] = pTs1[-1] = 0

        return cost

    def pairwise_emds(X0, X1=None, R=1.0, norm=False, 
                      n_jobs=None, verbose=0, print_every=10**6, chunksize=10**4):
        """Compute the EMD between collections of events. This can be used to
        compute EMDs between all pairs of events in a set or between events in
        two difference sets.

        **Arguments**

        - **X0** : _list_
            - Iterable collection of events. Each event is assumed to be an
            `(M,1+d)` array of particles, where `M` is the multiplicity and
            `d` is the dimension of the ground space in which to compute
            euclidean distances between particles. The zeroth column is
            assumed to be the energies (or equivalently, the transverse
            momenta) of the particles. For typical hadron collider applications, 
            each particle will be of the form `(pT,y,phi)` where `y` is the 
            rapidity and `phi` is the azimuthal angle.
        - **X1** : _list_ or `None`
            - Iterable collection of events in the same format as `X0`, 
            or `None`. If the latter, the pairwise distances between events
            in `X0` will be computed and the returned matrix will be symmetric.
       - **R** : _float_
            - The R parameter in Eq. 1 that controls the relative importance of
            the two terms. Must be greater than or equal to half of the maximum
            ground distance in the space in order for the EMD to be a valid metric.
        - **norm** : _bool_
            - Whether or not to normalize the pT values of the events prior to 
            computing the EMD.
        - **n_jobs** : _int_ or `None`
            - The number of worker processes to use. A value of `None` will use 
            as many processes as there are CPUs on the machine.
        - **verbose** : _int_
            - Controls the verbosity level. A value greater than `0` will print
            the progress of the computation at intervals specified by `print_every`.
        - **print_every** : _int_
            - The number of computations to do in between printing the progress.
            Even if the verbosity level is zero, this still plays a role in 
            determining when the worker processes report the results back to the
            main process.
        - **chunksize** : _int_
            - The size of chunks distributed to the worker processes. It is
            recommended to not make this value too small, and it should typically
            not be larger than `print_every`/`cpu_count`.

        **Returns**

        - _numpy.ndarray_
            - The EMD values as a two-dimensional array. If `X1` was `None`, then 
            the shape will be `(len(X0), len(X0))` and the array will be symmetric,
            otherwise it will have shape `(len(X0), len(X1))`.
        """

        # determine if we're doing symmetric pairs
        sym = X1 is None

        # process events into convenient form for EMD
        X0 = [_process_for_emd(x, norm) for x in X0]
        X1 = X0 if sym else [_process_for_emd(x, norm) for x in X1]

        # get iterator for indices
        pairs = (itertools.combinations(range(len(X0)), r=2) if sym else 
                 itertools.product(range(len(X0)), range(len(X1))))
        npairs = len(X0)*(len(X0)-1)//2 if sym else len(X0)*len(X1)

        # handle kwarg options
        if isinstance(print_every, float):
            print_every = int(npairs*print_event)

        # setup container for EMDs
        emds = np.zeros((len(X0), len(X1)))

        # use some number of worker processes to calculate EMDs
        start = time.time()
        if n_jobs != 1 or n_jobs is None:

            def _emd4map(pair):
                return _emd(X0[pair[0]], X1[pair[1]], R, norm)

            # create process pool
            with multiprocessing.Pool(n_jobs) as pool:

                # iterate over pairs of events
                begin = end = 0
                while end < npairs:
                    end += print_every
                    end = min(end, npairs)

                    # only hold this many pairs in memory
                    local_pairs = [next(pairs) for i in range(end - begin)]

                    # map function and store results
                    results = list(pool.imap(_emd4map, local_pairs, chunksize=chunksize))
                    for k,(i,j) in enumerate(local_pairs):
                        emds[i,j] = results[k]

                    # set begin to end
                    begin = end

                    # print update if verbose
                    if verbose >= 1:
                        print('computed {} EMDs, {:.3f}% finished in {:.2f}s'.format(end, end/npairs*100, time.time() - start))

        # run EMDs in this process
        elif n_jobs == 1:
            for k,(i,j) in enumerate(pairs):
                emds[i,j] = _emd(X0[i], X1[j], R, norm)
                if verbose >= 1 and (k % print_every) == 0:
                    print('computed {} EMDs, {:.3f}% finished in {:.2f}s'.format(k, k/npairs*100, time.time() - start))

        # unrecognized n_jobs value
        else:
            raise ValueError('n_jobs must be a positive integer or -1')

        # if doing an array with itself, symmetrize the distance matrix
        if sym:
            emds += emds.T

        return emds

# if we don't have ot
else:

    __all__ = []
