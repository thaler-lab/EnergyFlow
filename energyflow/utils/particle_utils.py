"""### Particle Tools

Tools to compute particle kinematic quantities from four-vectors,
such as transverse momentum $p_T$, rapidity $y$, and azimuthal angle
$\phi$, and vice versa.
"""
from __future__ import absolute_import, division

import warnings

import numpy as np

__all__ = [
    'p4s_from_ptyphims',
    'p4s_from_ptyphis',
    'ptyphims_from_p4s',
    'pts_from_p4s',
    'ys_from_p4s',
    'phis_from_p4s',
    'ms_from_p4s',
    'phi_fix',
    'flat_metric',
]

def ptyphims_from_p4s(p4s, phi_ref=None, keep_allzeros=True):
    """Compute the `[pt,y,phi,m]` representation of a four-vector for each Euclidean
    four-vector given as input. All-zero four-vectors are removed unless `keep_shape` 
    is `True`.

    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - An event as an `(M,4)` array of four-vectors `[E,px,py,pz]` for each particle.
        A single particle as a one-dimensional array or list is also accepted.
    - **phi_ref** : _float_
        - A reference value used so that all phis will be within $\pm\pi$ of this value.
        A value of `None` means that no phi fixing will be applied.
    - **keep_allzeros** : _bool_
        - Flag to determine if all-zero four-vectors will be retained as such. Otherwise,
        they are removed (resulting in a change in the shape of the output).

    **Returns**

    - _numpy.ndarray_
        - An array of size `(M,4)` consisting of the transverse momentum, rapidity,
        azimuthal angle, and mass of each particle. If a single particle was given as 
        input, a one-dimensional array is returned.
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
        - An event as an `(M,4)` array of four-vectors `[E,px,py,pz]` for each particle.
        A single particle as a one-dimensional array or list is also accepted.

    **Returns**

    - _numpy.ndarray_
        - An `M`-length array consisting of the transverse momentum of each particle.
        If a single particle was given as input, a single float is returned.
    """

    pts = np.sqrt(p4s[...,1]**2 + p4s[...,2]**2)
    return np.squeeze(pts)


def ys_from_p4s(p4s):
    """Calculate the rapidities of a collection of four-vectors
    
    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - An event as an `(M,4)` array of four-vectors `[E,px,py,pz]` for each particle.
        A single particle as a one-dimensional array or list is also accepted.

    **Returns**

    - _numpy.ndarray_
        - An `M`-length array consisting of the rapidity of each particle.
        If a single particle was given as input, a single float is returned.
    """

    ys = 0.5*np.log((p4s[...,0] + p4s[...,3])/(p4s[...,0] - p4s[...,3]))
    return np.squeeze(ys)


def phis_from_p4s(p4s, phi_ref=None):
    """Calculate the azimuthal angles of a collection of four-vectors. If `phi_ref` is 
    not `None`, then `phi_fix` is called using this value. Otherwise, 
    the angles are chosen to be in the inverval $[0,2\pi]$.
    
    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - An event as an `(M,4)` array of four-vectors `[E,px,py,pz]` for each particle.
        A single particle as a one-dimensional array or list is also accepted.
    - **phi_ref** : _float_
        - See 

    **Returns**

    - _numpy.ndarray_ or _list_
        - An `M`-length array consisting of the azimuthal angle of each particle.
        If a single particle was given as input, a single float is returned.
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
        - An event as an `(M,4)` array of four-vectors `[E,px,py,pz]` for each particle.
        A single particle as a one-dimensional array or list is also accepted.

    **Returns**

    - _numpy.ndarray_
        - An `M`-length array consisting of the mass of each particle. If a single 
        particle was given as input, a single float is returned.
    """

    p4s = np.atleast_2d(p4s)
    m2s = np.squeeze(p4s[...,0]**2 - np.sum(p4s[...,1:]**2, axis=-1))
    ms = np.sign(m2s)*np.sqrt(np.abs(m2s))
    return np.squeeze(ms)


def p4s_from_ptyphims(ptyphims):
    """Calculate Euclidean four-vectors from transverse momentum, rapidity, azimuthal angle,
    and (optionally) mass for each input.
    
    **Arguments**

    - **ptyphims** : _numpy.ndarray_ or _list_
        - An array with shape `(M,4)` of `[pT,y,phi,m]` for each particle. An array with 
        shape `(M,3)` is also accepted where the masses are taken to be zero. A single 
        particle is also accepted.

    **Returns**

    - _numpy.ndarray_
        - An event as an `(M,4)` array of four-vectors `[E,px,py,pz]` for each particle.
        If a single particle was given as input, a single four-vector will be returned.
    """

    # ensure a two-dimensional array
    ptyphims = np.atleast_2d(ptyphims)

    # get pts, ys, phis
    pts, ys, phis = [ptyphims[:,i] for i in range(3)]

    # get masses
    ms = ptyphims[:,3] if ptyphims.shape[1] == 4 else np.zeros(len(ptyphims))

    Ets = np.sqrt(pts**2 + ms**2)
    p4s = np.vstack([Ets*np.cosh(ys), pts*np.cos(phis), pts*np.sin(phis), Ets*np.sinh(ys)]).T

    return np.squeeze(p4s)


def p4s_from_ptyphis(ptyphis):
    """_Legacy function_: Will be removed in version 1.0. Use `p4s_from_ptyphims`
    for equivalent functionality.
    """

    warnings.warn(('This function is deprecated and will be removed in version 1.0. ' +
                   'Use p4s_from_ptyphims for equivalent functionality.'))

    return p4s_from_ptyphims(ptyphis)


twopi = 2*np.pi
def phi_fix(phis, phi_ref, copy=False):
    """A function to ensure that all phi values are within $\pi$ of `phi_ref`. 
    It is assumed that all starting phi values are within $2\pi$ of `phi_ref`.

    **Arguments**

    - **phis** : _numpy.ndarray_ or _list_
        - One-dimensional array of phi values.
    - **phi_ref** : _float_
        - A reference value used so that all phis will be within $\pm\pi$ of this value.
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
    new_phis[diff > np.pi] -= twopi
    new_phis[diff < -np.pi] += twopi
    return new_phis

long_metric = np.array([1.] + [-1.]*100)
def flat_metric(dim):
    """The Minkowski metric in `dim` spacetime dimensions in the mostly-minus convention.
    
    **Arguments**

    - **dim** : _int_
        - The number of spacetime dimensions (thought to be four in our universe).

    **Returns**

    - _1-d numpy.ndarray_
        - A `dim`-length, one-dimensional (not matrix) array equal to `[+1,-1,...,-1]`
    """

    if dim <= 101:
        return long_metric[:dim]
    return np.asarray([1.] + [-1.]*(dim-1))
