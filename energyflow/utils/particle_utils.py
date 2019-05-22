r"""### Particle Tools

Tools to compute particle kinematic quantities from four-vectors, such as
transverse momentum $p_T$, rapidity $y$, azimuthal angle $\phi$, and mass 
$m$, and vice versa.
"""
from __future__ import absolute_import, division

import warnings

import numpy as np

__all__ = [

    # from_p4s functions
    'ptyphims_from_p4s',
    'pts_from_p4s',
    'ys_from_p4s',
    'phis_from_p4s',
    'ms_from_p4s',

    # from_ptyphims functions
    'p4s_from_ptyphims',
    'p4s_from_ptyphis',
    'p4s_from_ptyphipids',

    # combination functions
    'combine_ptyphims',
    'combine_ptyphipids',

    # pid functions
    'pids2ms',

    # other functions
    'phi_fix',
    'flat_metric',
]

def ptyphims_from_p4s(p4s, phi_ref=None, keep_allzeros=True):
    r"""Compute the `[pt,y,phi,m]` representation of a four-vector for each
    Euclidean four-vector given as input. All-zero four-vectors are removed
    unless `keep_shape` is `True`.

    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - An event as an `(M,4)` array of four-vectors `[E,px,py,pz]` for each
        particle. A single particle as a one-dimensional array or list is also
        accepted.
    - **phi_ref** : _float_
        - A reference value used so that all phis will be within $\pm\pi$ of
        this value. A value of `None` means that no phi fixing will be applied.
    - **keep_allzeros** : _bool_
        - Flag to determine if all-zero four-vectors will be retained as such.
        Otherwise, they are removed (resulting in a change in the shape of the 
        output).

    **Returns**

    - _numpy.ndarray_
        - An array of size `(M,4)` consisting of the transverse momentum, 
        rapidity, azimuthal angle, and mass of each particle. If a single
        particle was given as input, a one-dimensional array is returned.
    """

    # ensure a two-dimensional array
    particles = np.copy(np.atleast_2d(p4s))

    # find non-zero particles
    nonzero_mask = np.count_nonzero(particles, axis=1) > 0
    nonzero_particles = particles[nonzero_mask]

    # get quantities
    pts = pts_from_p4s(nonzero_particles)
    ys = ys_from_p4s(nonzero_particles)
    phis = phis_from_p4s(nonzero_particles, phi_ref=phi_ref)
    ms = ms_from_p4s(nonzero_particles)
    ptyphims = np.vstack((pts, ys, phis, ms)).T

    # keep the all-zero particles
    if keep_allzeros:
        particles[nonzero_mask] = ptyphims
        return np.squeeze(particles)

    # return just the ptyphims for the non-zero particles
    return np.squeeze(ptyphims)


def pts_from_p4s(p4s):
    """Calculate the transverse momenta of a collection of four-vectors
    
    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - An event as an `(M,4)` array of four-vectors `[E,px,py,pz]` for each
        particle. A single particle as a one-dimensional array or list is also 
        accepted.

    **Returns**

    - _numpy.ndarray_
        - An `M`-length array consisting of the transverse momentum of each
        particle. If a single particle was given as input, a single float is
        returned.
    """

    pts = np.sqrt(p4s[...,1]**2 + p4s[...,2]**2)
    return np.squeeze(pts)


def ys_from_p4s(p4s):
    """Calculate the rapidities of a collection of four-vectors
    
    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - An event as an `(M,4)` array of four-vectors `[E,px,py,pz]` for each
        particle. A single particle as a one-dimensional array or list is also
        accepted.

    **Returns**

    - _numpy.ndarray_
        - An `M`-length array consisting of the rapidity of each particle.
        If a single particle was given as input, a single float is returned.
    """

    ys = 0.5*np.log((p4s[...,0] + p4s[...,3])/(p4s[...,0] - p4s[...,3]))
    return np.squeeze(ys)


def phis_from_p4s(p4s, phi_ref=None):
    r"""Calculate the azimuthal angles of a collection of four-vectors. If
    `phi_ref` is not `None`, then `phi_fix` is called using this value. 
    Otherwise, the angles are chosen to be in the inverval $[0,2\pi]$.
    
    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - An event as an `(M,4)` array of four-vectors `[E,px,py,pz]` for each
        particle. A single particle as a one-dimensional array or list is also
        accepted.
    - **phi_ref** : _float_
        - See [`phi_fix`](#phi_fix)

    **Returns**

    - _numpy.ndarray_
        - An `M`-length array consisting of the azimuthal angle of each 
        particle. If a single particle was given as input, a single float is
        returned.
    """

    phis = np.arctan2(p4s[...,2], p4s[...,1])
    phis[phis<0] += 2*np.pi

    # ensure close to reference value
    if phi_ref is not None:
        phis = phi_fix(phis, phi_ref)

    return np.squeeze(phis)


def ms_from_p4s(p4s):
    """Calculate the masses of a collection of four-vectors.
    
    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - An event as an `(M,4)` array of four-vectors `[E,px,py,pz]` for each
        particle. A single particle as a one-dimensional array or list is also
        accepted.

    **Returns**

    - _numpy.ndarray_
        - An `M`-length array consisting of the mass of each particle. If a
        single particle was given as input, a single float is returned.
    """

    p4s = np.atleast_2d(p4s)
    m2s = np.squeeze(p4s[...,0]**2 - np.sum(p4s[...,1:]**2, axis=-1))
    ms = np.sign(m2s)*np.sqrt(np.abs(m2s))
    return np.squeeze(ms)

def p4s_from_ptyphims(ptyphims):
    """Calculate Euclidean four-vectors from transverse momentum, rapidity,
    azimuthal angle, and (optionally) mass for each input.
    
    **Arguments**

    - **ptyphims** : _numpy.ndarray_ or _list_
        - An array with shape `(M,4)` of `[pT,y,phi,m]` for each particle. An
        array with shape `(M,3)` is also accepted where the masses are taken to
        be zero. A single particle is also accepted.

    **Returns**

    - _numpy.ndarray_
        - An event as an `(M,4)` array of four-vectors `[E,px,py,pz]` for each
        particle. If a single particle was given as input, a single four-vector
        will be returned.
    """

    # ensure a two-dimensional array
    ptyphims = np.atleast_2d(ptyphims)

    # get pts, ys, phis
    pts, ys, phis = [ptyphims[:,i] for i in range(3)]

    # get masses
    ms = ptyphims[:,3] if ptyphims.shape[1] == 4 else np.zeros(len(ptyphims))

    Ets = np.sqrt(pts**2 + ms**2)
    p4s = np.vstack([Ets*np.cosh(ys), pts*np.cos(phis), 
                     pts*np.sin(phis), Ets*np.sinh(ys)]).T

    return np.squeeze(p4s)

def p4s_from_ptyphipids(ptyphipids, error_on_uknown=False):
    """Calculate Euclidean four-vectors from transverse momentum, rapidity,
    azimuthal angle, and particle ID (which is used to determine the mass)
    for each input.
    
    **Arguments**

    - **ptyphipids** : _numpy.ndarray_ or _list_
        - An array with shape `(M,4)` of `[pT,y,phi,pdgid]` for each particle.
        A single particle is also accepted.
    - **error_on_unknown** : _bool_
        - See [`pids2ms`](#pids2ms).

    **Returns**

    - _numpy.ndarray_
        - An event as an `(M,4)` array of four-vectors `[E,px,py,pz]` for each
        particle. If a single particle was given as input, a single four-vector
        will be returned.
    """

    # ensure a two-dimensional array
    ptyphipids = np.atleast_2d(ptyphipids)

    # get pts, ys, phis
    pts, ys, phis, pids = [ptyphipids[:,i] for i in range(4)]

    # get masses
    ms = pids2ms(pids, error_on_unknown)

    Ets = np.sqrt(pts**2 + ms**2)
    p4s = np.vstack([Ets*np.cosh(ys), pts*np.cos(phis), 
                     pts*np.sin(phis), Ets*np.sinh(ys)]).T

    return np.squeeze(p4s)

def p4s_from_ptyphis(ptyphis):
    """_Legacy function_: Will be removed in version 1.0. Use 
    `p4s_from_ptyphims` for equivalent functionality.
    """

    warnings.warn('This function is deprecated and will be removed in version '
                  '1.0. Use p4s_from_ptyphims for equivalent functionality.')

    return p4s_from_ptyphims(ptyphis)

def combine_ptyphims(ptyphims, scheme='escheme'):
    """Combine (add) a collection of four-vectors that are expressed in
    hadronic coordinates.

    **Arguments**

    - **ptyphims** : _numpy.ndarray_ or _list_
        - An array with shape `(M,4)` of `[pT,y,phi,m]` for each particle. An
        array with shape `(M,3)` is also accepted where the masses are taken to
        be zero.
    - **scheme** : {`'escheme'`}
        - A string specifying how the four-vectors are to be combined.
        Currently, there is only one option, `'escheme'`, which adds the 
        four-vectors in euclidean coordinates.

    **Returns**

    - _1-d numpy.ndarray_
        - The combined four-vector, expressed as `[pT,y,phi,m]`.
    """

    if scheme == 'escheme':
        p4s = np.atleast_2d(p4s_from_ptyphims(ptyphims))
        tot = np.sum(p4s, axis=0)

    else:
        raise ValueError("Combination scheme '{}' not supported.".format(scheme))

    return ptyphims_from_p4s(tot)

def combine_ptyphipids(ptyphipids, scheme='escheme'):
    """Combine (add) a collection of four-vectors that are expressed as 
    `[pT,y,phi,pdgid]`.

    **Arguments**

    - **ptyphipids** : _numpy.ndarray_ or _list_
        - An array with shape `(M,4)` of `[pT,y,phi,pdgid]` for each particle.
    - **scheme** : {`'escheme'`}
        - A string specifying how the four-vectors are to be combined.
        Currently, there is only one option, `'escheme'`, which adds the 
        four-vectors in euclidean coordinates.

    **Returns**

    - _1-d numpy.ndarray_
        - The combined four-vector, expressed as `[pT,y,phi,m]`.
    """

    if scheme == 'escheme':
        p4s = np.atleast_2d(p4s_from_ptyphipids(ptyphipids))
        tot = np.sum(p4s, axis=0)

    else:
        raise ValueError("Combination scheme '{}' not supported.".format(scheme))

    return ptyphims_from_p4s(tot)

# masses (in GeV) of particles by pdgid
# obtained from the Pythia8 Particle Data page
PARTICLE_MASSES = {
    0:    0.,      # void
    1:    0.33,    # down
    2:    0.33,    # up
    3:    0.50,    # strange
    4:    1.50,    # charm
    5:    4.80,    # bottom
    6:    171.,    # top
    11:   5.11e-4, # e-
    12:   0.,      # nu_e
    13:   0.10566, # mu-
    14:   0.,      # nu_mu
    15:   1.77682, # tau-
    16:   0.,      # nu_tau
    21:   0.,      # gluon
    22:   0.,      # photon
    23:   91.1876, # Z
    24:   80.385,  # W+
    25:   125.,    # Higgs
    111:  0.13498, # pi0
    130:  0.49761, # K0-long
    211:  0.13957, # pi+
    310:  0.49761, # K0-short
    321:  0.49368, # K+
    2112: 0.93957, # neutron
    2212: 0.93827, # proton
    3122: 1.11568, # Lambda0
    3222: 1.18937, # Sigma+
    3312: 1.32171, # Xi-
    3322: 1.31486, # Xi0
    3334: 1.67245, # Omega-
}

def pids2ms(pids, error_on_uknown=False):
    r"""Map an array of [Particle Data Group IDs](http://pdg.lbl.gov/2018/
    reviews/rpp2018-rev-monte-carlo-numbering.pdf) to an array of the
    corresponding particle masses (in GeV).

    **Arguments**

    - **pids** : _1-d numpy.ndarray_ or _list_
        - An array of numeric (float or integer) PDGID values.
    - **error_on_unknown** : _bool_
        - Controls whether a `KeyError` is raised if an unknown PDGID is
        encountered. If `False`, unknown PDGIDs will map to zero.

    **Returns**

    - _1-d numpy.ndarray_
        - An array of masses in GeV.
    """

    pids_arr = np.asarray(pids, dtype=int)

    if error_on_uknown:
        masses = [PARTICLE_MASSES[pid] for pid in pids_arr]
    else:
        masses = [PARTICLE_MASSES.get(pid, 0.) for pid in pids_arr]

    return np.asarray(masses, dtype=float)

TWOPI = 2*np.pi
def phi_fix(phis, phi_ref, copy=False):
    r"""A function to ensure that all phi values are within $\pi$ of `phi_ref`.
    It is assumed that all starting phi values are within $2\pi$ of `phi_ref`.

    **Arguments**

    - **phis** : _numpy.ndarray_ or _list_
        - One-dimensional array of phi values.
    - **phi_ref** : _float_
        - A reference value used so that all phis will be within $\pm\pi$ of
        this value.
    - **copy** : _bool_
        - Determines if `phis` are copied or not. If `False` then `phis` may be
        modified in place.

    **Returns**

    - _numpy.ndarray_
        - An array of the fixed phi values.
    """

    phis = np.asarray(phis)
    diff = phis - phi_ref
    new_phis = np.copy(phis) if copy else phis
    new_phis[diff > np.pi] -= TWOPI
    new_phis[diff < -np.pi] += TWOPI
    return new_phis

LONG_METRIC = np.array([1.] + [-1.]*100)
def flat_metric(dim):
    """The Minkowski metric in `dim` spacetime dimensions in the mostly-minus
    convention.
    
    **Arguments**

    - **dim** : _int_
        - The number of spacetime dimensions (thought to be four in our 
        universe).

    **Returns**

    - _1-d numpy.ndarray_
        - A `dim`-length, one-dimensional (not matrix) array equal to 
        `[+1,-1,...,-1]`.
    """

    if dim <= 101:
        return LONG_METRIC[:dim]
    return np.asarray([1.] + [-1.]*(dim-1))
