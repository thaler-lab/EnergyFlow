r"""The Energy Mover's Distance (EMD), also known as the Earth Mover's 
Distance, is a metric between particle collider events introduced in
[1902.02346](https://arxiv.org/abs/1902.02346). This submodule contains
convenient functions for computing EMDs between individual events and
collections of events. The core of the computation is done using the
[Python Optimal Transport (POT)](https://pot.readthedocs.io) library,
which must be installed in order to use this submodule.

From Eq. 1 in [1902.02346](https://arxiv.org/abs/1902.02346), the EMD between
two events is the minimum ''work'' required to rearrange one event $\mathcal E$
into the other $\mathcal E'$ by movements of energy $f_{ij}$ from particle $i$ 
in one event to particle $j$ in the other:
$$
\text{EMD}(\mathcal E,\mathcal E^\prime)=\min_{\{f_{ij}\}}\sum_{ij}f_{ij}\frac{
\theta_{ij}}{R} + \left|\sum_iE_i-\sum_jE^\prime_j\right|,\\
f_{ij}\ge 0, \quad \sum_jf_{ij}\le E_i, \quad \sum_if_{ij}\le E^\prime_j, \quad 
\sum_{ij}f_{ij}=E_\text{min},
$$
where $E_i,E^\prime_j$ are the energies of the particles in the two events, 
$\theta_{ij}$ is an angular distance between particles, and 
$E_\text{min}=\min\left(\sum_iE_i,\,\sum_jE^\prime_j\right)$ is the smaller
of the two total energies. In a hadronic context, transverse momenta are used 
instead of energies.
"""
from __future__ import absolute_import, division, print_function

import itertools
import multiprocessing
import sys
import time
import warnings

import numpy as np

ot = True
try:
    from ot.lp import emd_c, check_result
    from scipy.spatial.distance import _distance_wrap # ot imports scipy anyway
except:
    warnings.warn('cannot import module \'ot\', module \'emd\' will be empty')
    ot = False

from energyflow.utils import create_pool

if ot:

    __all__ = ['emd', 'emds']

    # parameter checks
    def _check_params(norm, gdim, phi_col):
        if norm is None:
            raise ValueError('\'norm\' cannot be None')
        if phi_col < 1:
            raise ValueError('\'phi_col\' cannot be smaller than 1')
        if gdim is not None:
            if gdim < 1:
                raise ValueError('\'gdim\' must be greater than or equal to 1')
            if phi_col > gdim + 1:
                raise ValueError('\'phi_col\' must be less than or equal to gdim')

    # faster than scipy's cdist function because we can avoid their checks
    def _cdist_euclidean(X, Y, periodic_phi, phi_col):

        if periodic_phi:

            # delta phis (taking into account periodicity)
            # aleady guaranteed for values to be in [0, 2pi]
            d_phis = np.pi - np.abs(np.pi - np.abs(X[:,phi_col,np.newaxis] - Y[:,phi_col]))

            # split out common case of having only one other dimension
            if X.shape[1] == 2:
                non_phi_col = 1 - phi_col
                d_ys = X[:,non_phi_col,np.newaxis] - Y[:,non_phi_col]
                out = np.sqrt(d_ys**2 + d_phis**2)

            # general case
            else:
                non_phi_cols = [i for i in range(X.shape[1]) if i != phi_col]
                d_others2 = (X[:,np.newaxis,non_phi_cols] - Y[:,non_phi_cols])**2
                out = np.sqrt(d_others2.sum(axis=-1) + d_phis**2)

        else:
            out = np.empty((len(X), len(Y)), dtype=np.double)
            _distance_wrap.cdist_euclidean_double_wrap(X, Y, out)
        
        return out

    # process events for EMD calculation
    two_pi = 2*np.pi
    def _process_for_emd(event, norm, gdim, periodic_phi, phi_col):
        
        event = np.atleast_2d(event)
        
        if norm:
            pts = event[:,0]/event[:,0].sum()
        elif norm is None:
            pts = event[:,0]
        else:
            event = np.vstack((event, np.zeros(event.shape[1])))
            pts = event[:,0]

        pts = np.ascontiguousarray(pts, dtype=np.float64)
        if gdim is None:
            coords = np.ascontiguousarray(event[:,1:], dtype=np.float64)
        else:
            coords = np.ascontiguousarray(event[:,1:gdim+1], dtype=np.float64)

        if periodic_phi:
            if phi_col >= event.shape[1] - 1:
                evgdim = str(event.shape[1] - 1)
                raise ValueError('\'phi_col\' cannot be larger than the ground space '
                                 'dimension, which is ' + evgdim + ' for one of the events')
            coords[:,phi_col] %= two_pi

        return pts, coords

    def emd(ev0, ev1, R=1.0, norm=False, return_flow=False, gdim=None, n_iter_max=100000,
                      periodic_phi=False, phi_col=2):
        r"""Compute the EMD between two events.

        **Arguments**

        - **ev0** : _numpy.ndarray_
            - The first event, given as a two-dimensional array. The event is 
            assumed to be an `(M,1+gdim)` array of particles, where `M` is the 
            multiplicity and `gdim` is the dimension of the ground space in 
            which to compute euclidean distances between particles (as specified
            by the `gdim` keyword argument. The zeroth column is assumed to be
            the energies (or equivalently, the transverse momenta) of the 
            particles. For typical hadron collider jet applications, each 
            particle will be of the form `(pT,y,phi)` where  `y` is the rapidity
            and `phi` is the azimuthal angle.
        - **ev1** : _numpy.ndarray_
            - The other event, same format as **ev0**.
        - **R** : _float_
            - The R parameter in the EMD definition that controls the relative 
            importance of the two terms. Must be greater than or equal to half 
            of the maximum ground distance in the space in order for the EMD 
            to be a valid metric.
        - **norm** : _bool_
            - Whether or not to normalize the pT values of the events prior to 
            computing the EMD.
        - **return_flow** : _bool_
            - Whether or not to return the flow matrix describing the optimal 
            transport found during the computation of the EMD. Note that since
            the second term in Eq. 1 is implemented by including an additional 
            particle in the event with lesser total pT, this will be reflected 
            in the flow matrix.
        - **gdim** : _int_
            - The dimension of the ground metric space. Useful for restricting
            which dimensions are considered part of the ground space. Can be
            larger than the number of dimensions present in the events (in
            which case all dimensions will be included). If `None`, has no
            effect.
        - **n_iter_max** : _int_
            - Maximum number of iterations for solving the optimal transport 
            problem.
        - **periodic_phi** : _bool_
            - Whether to expect (and therefore properly handle) periodicity
            in the coordinate corresponding to the azimuthal angle $\phi$.
            Should typically be `True` for event-level applications but can
            be set to `False` (which is slightly faster) for jet applications
            where all $\phi$ differences are less than or equal to $\pi$.
        - **phi_col** : _int_
            - The index of the column of $\phi$ values in the event array.

        **Returns**

        - _float_
            - The EMD value.
        - [_numpy.ndarray_], optional
            - The flow matrix found while solving for the EMD. The `(i,j)`th 
            entry is the amount of `pT` that flows between particle i in `ev0`
            and particle j in `ev1`.
        """

        # parameter checks
        _check_params(norm, gdim, phi_col)

        # handle periodicity
        phi_col_m1 = phi_col - 1

        # process events
        pTs0, coords0 = _process_for_emd(ev0, None, gdim, periodic_phi, phi_col_m1)
        pTs1, coords1 = _process_for_emd(ev1, None, gdim, periodic_phi, phi_col_m1)

        pT0, pT1 = pTs0.sum(), pTs1.sum()

        # if norm, then we normalize the pts to 1
        if norm:
            pTs0 /= pT0
            pTs1 /= pT1
            thetas = _cdist_euclidean(coords0, coords1, periodic_phi, phi_col_m1)/R
            rescale = 1.0

        # implement the EMD in Eq. 1 of the paper by adding an appropriate extra particle
        else:
            pTdiff = pT1 - pT0
            if pTdiff > 0:
                pTs0 = np.hstack((pTs0, pTdiff))
                coords0_extra = np.vstack((coords0, np.zeros(coords0.shape[1], dtype=np.float64)))
                thetas = _cdist_euclidean(coords0_extra, coords1, periodic_phi, phi_col_m1)/R
                thetas[-1,:] = 1.0

            elif pTdiff < 0:
                pTs1 = np.hstack((pTs1, -pTdiff))
                coords1_extra = np.vstack((coords1, np.zeros(coords1.shape[1], dtype=np.float64)))
                thetas = _cdist_euclidean(coords0, coords1_extra, periodic_phi, phi_col_m1)/R
                thetas[:,-1] = 1.0

            # in this case, the pts were exactly equal already so no need to add a particle
            else:
                thetas = _cdist_euclidean(coords0, coords1, periodic_phi, phi_col_m1)/R

            # change units for numerical stability
            rescale = max(pT0, pT1)

        G, cost, _, _, result_code = emd_c(pTs0/rescale, pTs1/rescale, thetas, n_iter_max)
        check_result(result_code)

        # need to change units back
        return (cost * rescale, G * rescale) if return_flow else cost * rescale

    # helper function for pool imap
    def _emd4imap(x):
        (i, j), (X0, X1, R, norm, n_iter_max, periodic_phi, phi_col) = x
        return _emd(X0[i], X1[j], R, norm, n_iter_max, periodic_phi, phi_col)

    # internal use only by emds, makes assumptions about input format
    def _emd(ev0, ev1, R, norm, n_iter_max, periodic_phi, phi_col):

        pTs0, coords0 = ev0
        pTs1, coords1 = ev1

        thetas = _cdist_euclidean(coords0, coords1, periodic_phi, phi_col)/R

        # extra particles (with zero pt) already added if going in here
        rescale = 1.0
        if not norm:
            pT0, pT1 = pTs0.sum(), pTs1.sum()
            pTdiff = pT1 - pT0
            if pTdiff > 0:
                pTs0[-1] = pTdiff
            elif pTdiff < 0:
                pTs1[-1] = -pTdiff
            thetas[:,-1] = 1.0
            thetas[-1,:] = 1.0

            # change units for numerical stability
            rescale = max(pT0, pT1)

        # compute the emd with POT
        _, cost, _, _, result_code = emd_c(pTs0/rescale, pTs1/rescale, thetas, n_iter_max)
        check_result(result_code)

        # important! must reset extra particles to have pt zero
        if not norm:
            pTs0[-1] = pTs1[-1] = 0

        return cost * rescale

    def emds(X0, X1=None, R=1.0, norm=False, gdim=None, n_iter_max=100000,
                          periodic_phi=False, phi_col=2,
                          n_jobs=None, verbose=0, print_every=10**6):
        r"""Compute the EMD between collections of events. This can be used to
        compute EMDs between all pairs of events in a set or between events in
        two difference sets.

        **Arguments**

        - **X0** : _list_
            - Iterable collection of events. Each event is assumed to be an 
            `(M,1+gdim)` array of particles, where `M` is the multiplicity and 
            `gdim` is the dimension of the ground space in which to compute
            euclidean distances between particles (specified by the `gdim`
            keyword argument). The zeroth column is assumed to be the energies
            (or equivalently, the transverse momenta) of the particles. For
            typical hadron collider jet applications, each particle will be of
            the form `(pT,y,phi)` where  `y` is the rapidity and `phi` is the
            azimuthal angle.
        - **X1** : _list_ or `None`
            - Iterable collection of events in the same format as `X0`, 
            or `None`. If the latter, the pairwise distances between events
            in `X0` will be computed and the returned matrix will be symmetric.
       - **R** : _float_
            - The R parameter in the EMD definition that controls the relative 
            importance of the two terms. Must be greater than or equal to half 
            of the maximum ground distance in the space in order for the EMD 
            to be a valid metric.
        - **norm** : _bool_
            - Whether or not to normalize the pT values of the events prior to 
            computing the EMD.
        - **gdim** : _int_
            - The dimension of the ground metric space. Useful for restricting
            which dimensions are considered part of the ground space. Can be
            larger than the number of dimensions present in the events (in
            which case all dimensions will be included). If `None`, has no
            effect.
        - **n_iter_max** : _int_
            - Maximum number of iterations for solving the optimal transport 
            problem.
        - **periodic_phi** : _bool_
            - Whether to expect (and therefore properly handle) periodicity
            in the coordinate corresponding to the azimuthal angle $\phi$.
            Should typically be `True` for event-level applications but can
            be set to `False` (which is slightly faster) for jet applications
            where all $\phi$ differences are less than or equal to $\pi$.
        - **phi_col** : _int_
            - The index of the column of $\phi$ values in the event array.
        - **n_jobs** : _int_ or `None`
            - The number of worker processes to use. A value of `None` will use 
            as many processes as there are CPUs on the machine. Note that for
            smaller numbers of events, a smaller value of `n_jobs` can be faster.
        - **verbose** : _int_
            - Controls the verbosity level. A value greater than `0` will print
            the progress of the computation at intervals specified by `print_every`.
        - **print_every** : _int_
            - The number of computations to do in between printing the progress.
            Even if the verbosity level is zero, this still plays a role in 
            determining when the worker processes report the results back to the
            main process.

        **Returns**

        - _numpy.ndarray_
            - The EMD values as a two-dimensional array. If `X1` was `None`, then 
            the shape will be `(len(X0), len(X0))` and the array will be symmetric,
            otherwise it will have shape `(len(X0), len(X1))`.
        """

        _check_params(norm, gdim, phi_col)

        # determine if we're doing symmetric pairs
        sym = X1 is None

        # period handling
        phi_col_m1 = phi_col - 1

        # process events into convenient form for EMD
        X0 = [_process_for_emd(x, norm, gdim, periodic_phi, phi_col_m1) for x in X0]
        X1 = X0 if sym else [_process_for_emd(x, norm, gdim, periodic_phi, phi_col_m1) for x in X1]

        # get iterator for indices
        pairs = (itertools.combinations(range(len(X0)), r=2) if sym else 
                 itertools.product(range(len(X0)), range(len(X1))))
        npairs = len(X0)*(len(X0)-1)//2 if sym else len(X0)*len(X1)

        # handle kwarg options
        if isinstance(print_every, float):
            print_every = int(npairs*print_event)
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count() or 1

        # setup container for EMDs
        emds = np.zeros((len(X0), len(X1)))

        # use some number of worker processes to calculate EMDs
        start = time.time()
        if n_jobs != 1:

            # verbose printing
            if verbose >= 1:
                print('Using {} worker process{}:'.format(n_jobs, 'es' if n_jobs > 1 else ''))

            # create process pool
            with create_pool(n_jobs) as pool:

                # iterate over pairs of events
                begin = end = 0
                other_params = [X0, X1, R, norm, n_iter_max, periodic_phi, phi_col_m1]
                imap_args = ([pair, other_params] for pair in pairs)
                while end < npairs:
                    end += print_every
                    end = min(end, npairs)
                    chunksize = max(1, (end - begin)//n_jobs)

                    # only hold this many pairs in memory
                    local_imap_args = [next(imap_args) for i in range(end - begin)]

                    # map function and store results
                    results = list(pool.map(_emd4imap, local_imap_args, chunksize=chunksize))
                    for arg,r in zip(local_imap_args, results):
                        i, j = arg[0]
                        emds[i, j] = r

                    # setup for next iteration of while loop
                    begin = end

                    # print update if verbose
                    if verbose >= 1:
                        args = (end, end/npairs*100, time.time() - start)
                        print('  Computed {} EMDs, {:.2f}% done in {:.2f}s'.format(*args))

        # run EMDs in this process
        elif n_jobs == 1:
            for k,(i,j) in enumerate(pairs):
                emds[i, j] = _emd(X0[i], X1[j], R, norm, n_iter_max, periodic_phi, phi_col_m1)
                if verbose >= 1 and (k % print_every) == 0:
                    args = (k, k/npairs*100, time.time() - start)
                    print('Computed {} EMDs, {:.2f}% done in {:.2f}s'.format(*args))

        # unrecognized n_jobs value
        else:
            raise ValueError('n_jobs must be a positive integer or -1')

        # if doing an array with itself, symmetrize the distance matrix
        if sym:
            emds += emds.T

        if verbose >= 1:
            print()

        return emds

# if we don't have ot
else:

    __all__ = []
