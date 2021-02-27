r"""# Utilities

## Particle Tools

Tools for dealing with particle momenta four-vectors. A four-vector can either
be in Cartesian coordinates, `[e,px,py,pz]` (energy, momentum in `x` direction,
momentum in `y` direction, momentum in `z` direction), or hadronic coordinates, 
`[pt,y,phi,m]` (transverse momentum, rapidity, azimuthal angle, mass), which
are related via:

\[p_T=\sqrt{p_x^2+p_y^2},\quad y=\text{arctanh}\,\frac{p_z}{E},\quad 
\phi=\arctan_2\frac{p_y}{p_x},\quad m=\sqrt{E^2-p_x^2-p_y^2-p_z^2}\]

and inversely:

\[E=\cosh y\sqrt{p_T^2+m^2},\quad p_x=p_T\cos\phi,\quad 
p_y=p_T\sin\phi,\quad p_z=\sinh y\sqrt{p_T^2+m^2}.\]

The pseudorapidity `eta` can be obtained from a Cartesian four-momentum as:

\[\eta=\text{arctanh}\,\frac{p_z}{|\vec p|},\quad 
|\vec p|\equiv\sqrt{p_x^2+p_y^2+p_z^2},\]

and is related to the rapidity via

\[\eta=\text{arcsinh}\left(\sinh y\,\left(1+m^2/p_T^2\right)^{1/2}\right),\quad 
y=\text{arcsinh}\left(\sinh \eta\,\left(1+m^2/p_T^2\right)^{-1/2}\right).\]

Note that the above formulas are numerically stable up to values of rapidity or
pseudorapidity of a few hundred, above which the formulas have numerical issues. 
In this case, a different but equivalent formulae are used that are numerically
stable in this region. In all cases, the $p_T\to0$ limit produces infinite
values.

In the context of this package, an "event" is a two-dimensional numpy array
with shape `(M,4)` where `M` is the multiplicity. An array of events is a 
three-dimensional array with shape `(N,M,4)` where `N` is the number of events.
The valid inputs and outputs of the functions here will be described using
this terminology.
"""

#  _____        _____ _______ _____ _____ _      ______          _    _ _______ _____ _       _____
# |  __ \ /\   |  __ \__   __|_   _/ ____| |    |  ____|        | |  | |__   __|_   _| |     / ____|
# | |__) /  \  | |__) | | |    | || |    | |    | |__           | |  | |  | |    | | | |    | (___
# |  ___/ /\ \ |  _  /  | |    | || |    | |    |  __|          | |  | |  | |    | | | |     \___ \
# | |  / ____ \| | \ \  | |   _| || |____| |____| |____  ______ | |__| |  | |   _| |_| |____ ____) |
# |_| /_/    \_\_|  \_\ |_|  |_____\_____|______|______||______| \____/   |_|  |_____|______|_____/

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

import warnings

import numpy as np
import six

__all__ = [

    # from_p4s functions
    'ptyphims_from_p4s',
    'pts_from_p4s',
    'pt2s_from_p4s',
    'ys_from_p4s',
    'etas_from_p4s',
    'phis_from_p4s',
    'm2s_from_p4s',
    'ms_from_p4s',
    'ms_from_ps',

    # eta/y conversions
    'etas_from_pts_ys_ms',
    'ys_from_pts_etas_ms',

    # from_ptyphims functions
    'p4s_from_ptyphims',
    'p4s_from_ptyphipids',

    # combination functions
    'sum_ptyphims',
    'sum_ptyphipids',

    # transformation functions
    'center_ptyphims',
    'rotate_ptyphims',
    'reflect_ptyphims',

    # pid functions
    'pids2ms',
    'pids2chrgs',
    'ischrgd',

    # other functions
    'phi_fix',
    'flat_metric',
]

def ptyphims_from_p4s(p4s, phi_ref=None, mass=True):
    r"""Convert to hadronic coordinates `[pt,y,phi,m]` from Cartesian
    coordinates. All-zero four-vectors are left alone.

    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - A single particle, event, or array of events in Cartesian coordinates.
    - **phi_ref** : {`None`, `'hardest'`, _float_, _numpy.ndarray_}
        - Used to help deal with the fact that $\phi$ is a periodic coordinate.
        If a float (which should be in $[0,2\pi)$), all phi values will be
        within $\pm\pi$ of this reference value. If `'\hardest'`, the phi of
        the hardest particle is used as the reference value. If `None`, all
        phis will be in the range $[0,2\pi)$. An array is accepted in the case
        that `p4s` is an array of events, in which case the `phi_ref` array
        should have shape `(N,)` where `N` is the number of events.
    - **mass** : _bool_
        - Whether or not to include particle masses.

    **Returns**

    - _numpy.ndarray_
        - An array of hadronic four-momenta with the same shape as the input.
    """

    p4s = np.asarray(p4s, dtype=float)
    if p4s.shape[-1] != 4:
        raise ValueError("Last dimension of 'p4s' must have size 4.")

    out = np.zeros(p4s.shape[:-1] + (4 if mass else 3,), dtype=float)
    out[...,0] = pts_from_p4s(p4s)
    out[...,1] = ys_from_p4s(p4s)
    out[...,2] = phis_from_p4s(p4s, phi_ref, _pts=out[...,0])
    if mass:
        out[...,3] = ms_from_p4s(p4s)

    return out

def pt2s_from_p4s(p4s):
    """Calculate the squared transverse momenta of a collection of four-vectors.
    
    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - A single particle, event, or array of events in Cartesian coordinates.

    **Returns**

    - _numpy.ndarray_
        - An array of squared transverse momenta with shape `p4s.shape[:-1]`.
    """

    p4s = np.asarray(p4s, dtype=float)
    return p4s[...,1]**2 + p4s[...,2]**2

def pts_from_p4s(p4s):
    """Calculate the transverse momenta of a collection of four-vectors.
    
    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - A single particle, event, or array of events in Cartesian coordinates.

    **Returns**

    - _numpy.ndarray_
        - An array of transverse momenta with shape `p4s.shape[:-1]`.
    """

    return np.sqrt(pt2s_from_p4s(p4s))

def ys_from_p4s(p4s):
    """Calculate the rapidities of a collection of four-vectors. Returns zero
    for all-zero particles
    
    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - A single particle, event, or array of events in Cartesian coordinates.

    **Returns**

    - _numpy.ndarray_
        - An array of rapidities with shape `p4s.shape[:-1]`.
    """

    p4s = np.asarray(p4s, dtype=float)
    out = np.zeros(p4s.shape[:-1], dtype=float)

    nz_mask = np.any(p4s != 0., axis=-1)
    nz_p4s = p4s[nz_mask]
    out[nz_mask] = np.arctanh(nz_p4s[...,3]/nz_p4s[...,0])

    return out

def etas_from_p4s(p4s):
    """Calculate the pseudorapidities of a collection of four-vectors. Returns
    zero for all-zero particles
    
    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - A single particle, event, or array of events in Cartesian coordinates.

    **Returns**

    - _numpy.ndarray_
        - An array of pseudorapidities with shape `p4s.shape[:-1]`.
    """

    p4s = np.asarray(p4s, dtype=float)
    out = np.zeros(p4s.shape[:-1], dtype=float)

    nz_mask = np.any(p4s != 0., axis=-1)
    nz_p4s = p4s[nz_mask]
    out[nz_mask] = np.arctanh(nz_p4s[...,3]/np.sqrt(nz_p4s[...,1]**2 + nz_p4s[...,2]**2 + nz_p4s[...,3]**2))

    return out

# phis_from_p4s(p4s, phi_ref=None)
def phis_from_p4s(p4s, phi_ref=None, _pts=None):
    r"""Calculate the azimuthal angles of a collection of four-vectors.
    
    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - A single particle, event, or array of events in Cartesian coordinates.
    - **phi_ref** : {_float_, _numpy.ndarray_, `None`, `'hardest'`}
        - Used to help deal with the fact that $\phi$ is a periodic coordinate.
        If a float (which should be in $[0,2\pi)$), all phi values will be
        within $\pm\pi$ of this reference value. If `'\hardest'`, the phi of
        the hardest particle is used as the reference value. If `None`, all
        phis will be in the range $[0,2\pi)$. An array is accepted in the case
        that `p4s` is an array of events, in which case the `phi_ref` array
        should have shape `(N,)` where `N` is the number of events.

    **Returns**

    - _numpy.ndarray_
        - An array of azimuthal angles with shape `p4s.shape[:-1]`.
    """

    # get phis
    p4s = np.asarray(p4s, dtype=float)
    phis = np.asarray(np.arctan2(p4s[...,2], p4s[...,1]))
    phis[phis<0] += 2*np.pi

    # ensure close to reference value
    if phi_ref is not None:
        if isinstance(phi_ref, six.string_types) and phi_ref == 'hardest':
            ndim = phis.ndim

            # here the particle is already phi fixed with respect to itself
            if ndim == 0:
                return phis

            # get pts if needed (pt2s are fine for determining hardest)
            if _pts is None:
                _pts = pt2s_from_p4s(p4s)
            hardest = np.argmax(_pts, axis=-1)

            # indexing into vector
            if ndim == 1:
                phi_ref = phis[hardest]

            # advanced indexing
            elif ndim == 2:
                phi_ref = phis[np.arange(len(phis)), hardest]

            else:
                raise ValueError("'p4s' should not have more than three dimensions.")

        phis = phi_fix(phis, phi_ref, copy=False)

    return phis

TWOPI = 2*np.pi
def phi_fix(phis, phi_ref, copy=True):
    r"""A function to ensure that all phis are within $\pi$ of `phi_ref`. It is
    assumed that all starting phi values are $\pm 2\pi$ of `phi_ref`.

    **Arguments**

    - **phis** : _numpy.ndarray_ or _list_
        - Array of phi values.
    - **phi_ref** : {_float_ or _numpy.ndarray_}
        - A reference value used so that all phis will be within $\pm\pi$ of
        this value. Should have a shape of `phis.shape[:-1]`.
    - **copy** : _bool_
        - Determines if `phis` are copied or not. If `False` then `phis` is
        modified in place.

    **Returns**

    - _numpy.ndarray_
        - An array of the fixed phi values.
    """

    phis, phi_ref = np.asarray(phis, dtype=float), np.asarray(phi_ref, dtype=float)
    phi_ref = phi_ref[...,np.newaxis] if phi_ref.ndim > 0 else phi_ref

    diff = phis - phi_ref

    new_phis = np.copy(phis) if copy else phis
    new_phis[diff > np.pi] -= TWOPI
    new_phis[diff < -np.pi] += TWOPI

    return new_phis

def m2s_from_p4s(p4s):
    """Calculate the squared masses of a collection of four-vectors.
    
    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - A single particle, event, or array of events in Cartesian coordinates.

    **Returns**

    - _numpy.ndarray_
        - An array of squared masses with shape `p4s.shape[:-1]`.
    """

    p4s = np.asarray(p4s, dtype=float)
    return p4s[...,0]**2 - p4s[...,1]**2 - p4s[...,2]**2 - p4s[...,3]**2

def ms_from_p4s(p4s):
    """Calculate the masses of a collection of four-vectors.
    
    **Arguments**

    - **p4s** : _numpy.ndarray_ or _list_
        - A single particle, event, or array of events in Cartesian coordinates.

    **Returns**

    - _numpy.ndarray_
        - An array of masses with shape `p4s.shape[:-1]`.
    """

    m2s = m2s_from_p4s(p4s)
    return np.sign(m2s)*np.sqrt(np.abs(m2s))

def ms_from_ps(ps):
    r"""Calculate the masses of a collection of Lorentz vectors in two or more
    spacetime dimensions.

    **Arguments**

    - **ps** : _numpy.ndarray_ or _list_
        - A single particle, event, or array of events in Cartesian
        coordinates in $d\ge2$ spacetime dimensions.

    **Returns**

    - _numpy.ndarray_
        - An array of masses with shape `ps.shape[:-1]`.
    """

    nps = np.asarray(ps, dtype=float)
    m2s = nps[...,0]**2 - np.sum(nps[...,1:]**2, axis=-1)
    return np.sign(m2s)*np.sqrt(np.abs(m2s))

# etas_from_pts_ys_ms(pts, ys, ms)
def etas_from_pts_ys_ms(pts, ys, ms, _cutoff=50.):
    """Calculate pseudorapidities from transverse momenta, rapidities, and masses.
    All input arrays should have the same shape.

    **Arguments**

    - **pts** : _numpy.ndarray_
        - Array of transverse momenta.
    - **ys** : _numpy.ndarray_
        - Array of rapidities.
    - **ms** : _numpy.ndarray_
        - Array of masses.

    **Returns**

    - _numpy.ndarray_
        - Array of pseudorapidities with the same shape as `ys`.
    """

    pts, ys, ms = np.asarray(pts), np.asarray(ys), np.asarray(ms)

    abs_ys, x2s = np.abs(ys), (ms/pts)**2
    sqrt1plusx2s = np.sqrt(1 + x2s)

    if np.max(abs_ys) < _cutoff:
        return np.arcsinh(np.sinh(ys)*sqrt1plusx2s)

    # have to use different formulas for large and small ys
    large_mask = (abs_ys > _cutoff)
    small_mask = ~large_mask
    out = np.zeros(ys.shape, dtype=float)

    large_abs_ys = abs_ys[large_mask]

    # note that the commented term can be ignored since it is numerically 1 for |y| > 20
    out[large_mask] = large_abs_ys + np.log(#(1. - np.exp(-2.*large_abs_ys))*
                        (sqrt1plusx2s[large_mask] + 
                         np.sqrt(x2s[large_mask] + 1./np.tanh(large_abs_ys)**2))/2.)
    out[large_mask] *= np.sign(ys[large_mask])

    out[small_mask] = np.arcsinh(np.sinh(ys[small_mask])*sqrt1plusx2s[small_mask])

    return out

# ys_from_pts_etas_ms(pts, etas, ms)
def ys_from_pts_etas_ms(pts, etas, ms, _cutoff=50.):
    """Calculate rapidities from transverse momenta, pseudorapidities, and masses.
    All input arrays should have the same shape.

    **Arguments**

    - **pts** : _numpy.ndarray_
        - Array of transverse momenta.
    - **etas** : _numpy.ndarray_
        - Array of pseudorapidities.
    - **ms** : _numpy.ndarray_
        - Array of masses.

    **Returns**

    - _numpy.ndarray_
        - Array of rapidities with the same shape as `etas`.
    """

    pts, etas, ms = np.asarray(pts), np.asarray(etas), np.asarray(ms)

    abs_etas, x2s = np.abs(etas), (ms/pts)**2
    sqrt1plusx2s = np.sqrt(1 + x2s)

    if np.max(abs_etas) < _cutoff:
        return np.arcsinh(np.sinh(etas)/sqrt1plusx2s)

    # have to use different formulas for large and small etas
    large_mask = (abs_etas > _cutoff)
    small_mask = ~large_mask
    out = np.zeros(etas.shape, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        large_abs_etas = abs_etas[large_mask]

        # note that the commented term can be ignored since it is numerically 1 for |eta| > 20
        out[large_mask] = large_abs_etas + np.log(#(1. - np.exp(-2.*large_abs_etas))*
                                   (1. + np.sqrt(1./np.tanh(large_abs_etas)**2 + 
                                                 x2s[large_mask]/np.sinh(large_abs_etas)**2))/
                                   (2.*sqrt1plusx2s[large_mask]))
        out[large_mask] *= np.sign(etas[large_mask])

    out[small_mask] = np.arcsinh(np.sinh(etas[small_mask])/sqrt1plusx2s[small_mask])

    return out

def p4s_from_ptyphims(ptyphims):
    """Calculate Cartesian four-vectors from transverse momenta, rapidities,
    azimuthal angles, and (optionally) masses for each input.
    
    **Arguments**

    - **ptyphims** : _numpy.ndarray_ or _list_
        - A single particle, event, or array of events in hadronic coordinates.
        The mass is optional and if left out will be taken to be zero.

    **Returns**

    - _numpy.ndarray_
        - An array of Cartesian four-vectors.
    """

    # get pts, ys, phis
    ptyphims = np.asarray(ptyphims, dtype=float)
    pts, ys, phis = (ptyphims[...,0,np.newaxis], 
                     ptyphims[...,1,np.newaxis], 
                     ptyphims[...,2,np.newaxis])

    # get masses
    ms = ptyphims[...,3,np.newaxis] if ptyphims.shape[-1] == 4 else np.zeros(pts.shape)

    Ets = np.sqrt(pts**2 + ms**2)
    p4s = np.concatenate((Ets*np.cosh(ys), pts*np.cos(phis), 
                          pts*np.sin(phis), Ets*np.sinh(ys)), axis=-1)
    return p4s

def p4s_from_ptyphipids(ptyphipids, error_on_unknown=False):
    """Calculate Cartesian four-vectors from transverse momenta, rapidities,
    azimuthal angles, and particle IDs for each input. The particle IDs are
    used to lookup the mass of the particle. Transverse momenta should have
    units of GeV when using this function.
    
    **Arguments**

    - **ptyphipids** : _numpy.ndarray_ or _list_
        - A single particle, event, or array of events in hadronic coordinates
        where the mass is replaced by the PDG ID of the particle.
    - **error_on_unknown** : _bool_
        - See the corresponding argument of [`pids2ms`](#pids2ms).

    **Returns**

    - _numpy.ndarray_
        - An array of Cartesian four-vectors with the same shape as the input.
    """

    # get pts, ys, phis
    ptyphipids = np.asarray(ptyphipids, dtype=float)
    pts, ys, phis = (ptyphipids[...,0,np.newaxis],
                     ptyphipids[...,1,np.newaxis],
                     ptyphipids[...,2,np.newaxis])

    # get masses
    ms = pids2ms(ptyphipids[...,3,np.newaxis], error_on_unknown)

    Ets = np.sqrt(pts**2 + ms**2)
    p4s = np.concatenate((Ets*np.cosh(ys), pts*np.cos(phis), 
                          pts*np.sin(phis), Ets*np.sinh(ys)), axis=-1)
    return p4s

def sum_ptyphims(ptyphims, scheme='escheme'):
    r"""Add a collection of four-vectors that are expressed in hadronic
    coordinates by first converting to Cartesian coordinates and then summing.

    **Arguments**

    - **ptyphims** : _numpy.ndarray_ or _list_
        - An event in hadronic coordinates. The mass is optional and if left
        out will be taken to be zero.
    - **scheme** : _str_
        - A string specifying a recombination scheme for adding four-vectors
        together. Currently supported options are `'escheme'`, which adds the
        vectors in Cartesian coordinates, and `'ptscheme'`, which sums the pTs
        of each particle and places the jet axis at the pT-weighted centroid
        in the rapidity-azimuth plane. Note that `'ptscheme'` will return a
        three-vector consisting of the jet `[pT,y,phi]` with no mass value.

    **Returns**

    - _numpy.ndarray_
        - Array of summed four-vectors, in hadronic coordinates. Note that when
        `scheme` is `'escheme'`, the $\phi$ value of the hardest particle is
        used as the `phi_ref` when converting back to hadronic coordinates.
    """

    ptyphims = np.asarray(ptyphims, dtype=float)

    if ptyphims.ndim <= 1 or ptyphims.size == 0:
        return ptyphims

    if scheme == 'escheme':
        phi = ptyphims[np.argmax(ptyphims[:,0]),2]
        sum_p4 = np.sum(p4s_from_ptyphims(ptyphims), axis=-2)
        return ptyphims_from_p4s(sum_p4, phi_ref=phi)

    elif scheme == 'ptscheme':
        y, phi = np.average(ptyphims[:,1:3], weights=ptyphims[:,0], axis=0)
        return np.asarray([np.sum(ptyphims[:,0]), y, phi])

    else:
        raise ValueError('Unknown recombination scheme {}'.format(scheme))

def sum_ptyphipids(ptyphipids, scheme='escheme', error_on_unknown=False):
    r"""Add a collection of four-vectors that are expressed as
    `[pT,y,phi,pdgid]`.

    **Arguments**

    - **ptyphipids** : _numpy.ndarray_ or _list_
        - A single particle or event in hadronic coordinates where the mass
        is replaced by the PDG ID of the particle.
    - **scheme** : _str_
        - See the argument of the same name of [`sum_ptyphims`](#sum_ptyphims).
    - **error_on_unknown** : _bool_
        - See the corresponding argument of [`pids2ms`](#pids2ms).

    **Returns**

    - _numpy.ndarray_
        - Array of summed four-vectors, in hadronic coordinates. Note that when
        `scheme` is `'escheme'`, the $\phi$ value of the hardest particle is
        used as the `phi_ref` when converting back to hadronic coordinates.
    """

    ptyphipids = np.asarray(ptyphipids, dtype=float)

    if ptyphipids.ndim <= 1 or ptyphipids.size == 0:
        return ptyphipids

    if scheme == 'escheme':
        phi = ptyphipids[np.argmax(ptyphipids[:,0]),2]
        sum_p4 = np.sum(p4s_from_ptyphipids(ptyphipids, error_on_unknown), axis=-2)
        return ptyphims_from_p4s(sum_p4, phi_ref=phi)

    elif scheme == 'ptscheme':
        return sum_ptyphims(ptyphipids, scheme=scheme)

    else:
        raise ValueError('Unknown recombination scheme {}'.format(scheme))

def center_ptyphims(ptyphims, axis=None, center='escheme', copy=True):
    """Center a collection of four-vectors according to a calculated or 
    provided axis.

    **Arguments**

    - **ptyphims** : _numpy.ndarray_ or _list_
        - An event in hadronic coordinates. The mass is optional and if left
        out will be taken to be zero.
    - **axis** : _numpy.ndarray_
        - If not `None`, the `[y,phi]` values to use for centering.
    - **center** : _str_
        - The centering scheme to be used. Valid options are the same as the
        `scheme` argument of [`sum_ptyphims`](#sum_ptyphims).
    - **copy** : _bool_
        - Whether or not to copy the input array.

    **Returns**

    - _numpy.ndarray_
        - An array of hadronic four-momenta with the positions centered around
        the origin.
    """

    if axis is None:
        axis = sum_ptyphims(ptyphims, scheme=center)[1:3]

    if copy:
        ptyphims = np.copy(ptyphims)

    ptyphims[:,1:3] -= axis

    return ptyphims

def _do_reflection(zs, coords):
    return np.sum(zs[coords > 0.]) < np.sum(zs[coords < 0.])

def rotate_ptyphims(ptyphims, rotate='ptscheme', center=None, copy=True):
    """Rotate a collection of four-vectors to vertically align the principal
    component of the energy flow. The principal component is obtained as the
    eigenvector of the energy flow with the largest eigenvalue. It is only
    defined up to a sign, however it is ensured that there is more total pT in 
    the top half of the rapidity-azimuth plane.

    **Arguments**

    - **ptyphims** : _numpy.ndarray_ or _list_
        - An event in hadronic coordinates. The mass is optional and if left
        out will be taken to be zero.
    - **rotate** : _str_
        - The rotation scheme to be used. Currently, only `'ptscheme'` is
        supported, which causes the rotation to take place in the 
        rapidity-azimuth plane.
    - **center** : _str_ or `None`
        - If not `None`, the event will be centered prior to rotation and this
        argument will be passed on to `center_ptyphims` as the centering
        scheme.
    - **copy** : _bool_
        - Whether or not to copy the input array.

    **Returns**

    - _numpy.ndarray_
        - An array of hadronic four-momenta with the positions rotated around
        the origin.
    """

    if copy:
        ptyphims = np.copy(ptyphims)

    if center is not None:
        ptyphims = center_ptyphims(ptyphims, center=center, copy=False)

    if rotate == 'ptscheme':

        zs, phats = ptyphims[:,0], ptyphims[:,1:3]
        efm2 = np.einsum('a,ab,ac->bc', zs, phats, phats, optimize=['einsum_path', (0,1), (0,1)])
        eigvals, eigvecs = np.linalg.eigh(efm2)

        ptyphims[:,1:3] = np.dot(phats, eigvecs)

        if _do_reflection(zs, ptyphims[:,2]):
            ptyphims[:,1:3] *= -1.

    else:
        raise ValueError('Unknown rotation scheme {}'.format(rotate))

    return ptyphims

def reflect_ptyphims(ptyphims, which='both', center=None, copy=True):
    """Reflect a collection of four-vectors to arrange the highest-pT
    half or quadrant to have positive rapidity-azimuth coordinates.

    **Arguments**

    - **ptyphims** : _numpy.ndarray_
        - An event in hadronic coordinates. The mass is optional and is not
        used as a part of this function.
    - **which** : {`'both'`, `'x'`, `'y'`}
        - Controls which axes to consider reflecting over. `'both'` includes
        `'x'` and `'y'`.
    - **center** : _str_ or `None`
        - If not `None`, the centering scheme to use prior to performing
        reflections.
    - **copy** : _bool_
        - Whether or not to copy the input array.
    """

    if copy:
        ptyphims = np.copy(ptyphims)

    if center is not None:
        ptyphims = center_ptyphims(ptyphims, center=center, copy=False)

    zs = ptyphims[:,0]
    if (which == 'both' or which == 'x') and _do_reflection(zs, ptyphims[:,1]):
        ptyphims[:,1] *= -1.

    if (which == 'both' or which == 'y') and _do_reflection(zs, ptyphims[:,2]):
        ptyphims[:,2] *= -1.

    return ptyphims

# charges and masses (in GeV) of particles by pdgid
# obtained from the Pythia8 Particle Data page
# http://home.thep.lu.se/~torbjorn/pythia82html/ParticleData.html
# includes fundamental particles and most ground state uds mesons and baryons 
# as well as some things that have shown up at Pythia parton level
PARTICLE_PROPERTIES = {
#   PDGID     CHARGE MASS          NAME
    0:       ( 0.,   0.,      ), # void
    1:       (-1./3, 0.33,    ), # down
    2:       ( 2./3, 0.33,    ), # up
    3:       (-1./3, 0.50,    ), # strange
    4:       ( 2./3, 1.50,    ), # charm
    5:       (-1./3, 4.80,    ), # bottom
    6:       ( 2./3, 171.,    ), # top
    11:      (-1.,   5.11e-4, ), # e-
    12:      ( 0.,   0.,      ), # nu_e
    13:      (-1.,   0.10566, ), # mu-
    14:      ( 0.,   0.,      ), # nu_mu
    15:      (-1.,   1.77682, ), # tau-
    16:      ( 0.,   0.,      ), # nu_tau
    21:      ( 0.,   0.,      ), # gluon
    22:      ( 0.,   0.,      ), # photon
    23:      ( 0.,   91.1876, ), # Z
    24:      ( 1.,   80.385,  ), # W+
    25:      ( 0.,   125.,    ), # Higgs
    111:     ( 0.,   0.13498, ), # pi0
    113:     ( 0.,   0.77549, ), # rho0
    130:     ( 0.,   0.49761, ), # K0-long
    211:     ( 1.,   0.13957, ), # pi+
    213:     ( 1.,   0.77549, ), # rho+
    221:     ( 0.,   0.54785, ), # eta
    223:     ( 0.,   0.78265, ), # omega
    310:     ( 0.,   0.49761, ), # K0-short
    321:     ( 1.,   0.49368, ), # K+
    331:     ( 0.,   0.95778, ), # eta'
    333:     ( 0.,   1.01946, ), # phi
    445:     ( 0.,   3.55620, ), # chi_2c
    555:     ( 0.,   9.91220, ), # chi_2b
    2101:    ( 1./3, 0.57933, ), # ud_0
    2112:    ( 0.,   0.93957, ), # neutron
    2203:    ( 4./3, 0.77133, ), # uu_1
    2212:    ( 1.,   0.93827, ), # proton
    1114:    (-1.,   1.232,   ), # Delta-
    2114:    ( 0.,   1.232,   ), # Delta0
    2214:    ( 1.,   1.232,   ), # Delta+
    2224:    ( 2.,   1.232,   ), # Delta++
    3122:    ( 0.,   1.11568, ), # Lambda0
    3222:    ( 1.,   1.18937, ), # Sigma+
    3212:    ( 0.,   1.19264, ), # Sigma0
    3112:    (-1.,   1.19745, ), # Sigma-
    3312:    (-1.,   1.32171, ), # Xi-
    3322:    ( 0.,   1.31486, ), # Xi0
    3334:    (-1.,   1.67245, ), # Omega-
    10441:   ( 0.,   3.41475, ), # chi_0c
    10551:   ( 0.,   9.85940, ), # chi_0b
    20443:   ( 0.,   3.51066, ), # chi_1c
    9940003: ( 0.,   3.29692, ), # J/psi[3S1(8)]
    9940005: ( 0.,   3.75620, ), # chi_2c[3S1(8)]
    9940011: ( 0.,   3.61475, ), # chi_0c[3S1(8)]
    9940023: ( 0.,   3.71066, ), # chi_1c[3S1(8)]
    9940103: ( 0.,   3.88611, ), # psi(2S)[3S1(8)]
    9941003: ( 0.,   3.29692, ), # J/psi[1S0(8)]
    9942003: ( 0.,   3.29692, ), # J/psi[3PJ(8)]
    9942033: ( 0.,   3.97315, ), # psi(3770)[3PJ(8)]
    9950203: ( 0.,   10.5552, ), # Upsilon(3S)[3S1(8)]
}

# dictionaries derived from the main one above
PARTICLE_CHARGES = {pdgid: props[0] for pdgid,props in PARTICLE_PROPERTIES.items()}
PARTICLE_MASSES  = {pdgid: props[1] for pdgid,props in PARTICLE_PROPERTIES.items()}
CHARGED_PIDS = frozenset(pdgid for pdgid,charge in PARTICLE_CHARGES.items() if charge != 0.)

def pids2ms(pids, error_on_unknown=False):
    r"""Map an array of [Particle Data Group IDs](http://pdg.lbl.gov/2018/
    reviews/rpp2018-rev-monte-carlo-numbering.pdf) to an array of the
    corresponding particle masses (in GeV).

    **Arguments**

    - **pids** : _numpy.ndarray_ or _list_
        - An array of numeric (float or integer) PDG ID values.
    - **error_on_unknown** : _bool_
        - Controls whether a `KeyError` is raised if an unknown PDG ID is
        encountered. If `False`, unknown PDG IDs will map to zero.

    **Returns**

    - _numpy.ndarray_
        - An array of masses in GeV.
    """

    abspids = np.abs(np.asarray(pids, dtype=int))
    orig_shape = abspids.shape
    abspids = abspids.reshape(-1)

    if error_on_unknown:
        masses = [PARTICLE_MASSES[pid] for pid in abspids]
    else:
        masses = [PARTICLE_MASSES.get(pid, 0.) for pid in abspids]

    return np.asarray(masses, dtype=float).reshape(orig_shape)

def pids2chrgs(pids, error_on_unknown=False):
    r"""Map an array of [Particle Data Group IDs](http://pdg.lbl.gov/2018/
    reviews/rpp2018-rev-monte-carlo-numbering.pdf) to an array of the
    corresponding particle charges (in fundamental units where the charge
    of the electron is -1).

    **Arguments**

    - **pids** : _numpy.ndarray_ or _list_
        - An array of numeric (float or integer) PDG ID values.
    - **error_on_unknown** : _bool_
        - Controls whether a `KeyError` is raised if an unknown PDG ID is
        encountered. If `False`, unknown PDG IDs will map to zero.

    **Returns**

    - _numpy.ndarray_
        - An array of charges as floats.
    """

    signs = np.sign(np.asarray(pids, dtype=float))
    abspids = np.abs(np.asarray(pids, dtype=int))
    orig_shape = abspids.shape
    abspids = abspids.reshape(-1)

    if error_on_unknown:
        charges = [PARTICLE_CHARGES[pid] for pid in abspids]
    else:
        charges = [PARTICLE_CHARGES.get(pid, 0.) for pid in abspids]

    return signs * np.asarray(charges, dtype=float).reshape(orig_shape)

def ischrgd(pids, ignored_pids=None):
    """Compute a boolean mask according to if the given PDG ID corresponds
    to a particle of non-zero charge.

    **Arguments**

    - **pids** : _numpy.ndarray_
        - An array of numeric (float or integer) PDG ID values.
    - **ignored_pids** : _numpy.ndarray_ or `None`
        - If not `None`, the PDG IDs in this array will not be considered
        charged, for instance to avoid counting quarks as charged particles.

    **Returns**

    - _numpy.ndarray_
        - A boolean mask corresponding to which particles are charged.
    """
    
    abspids = np.abs(np.asarray(pids, dtype=int))
    orig_shape = abspids.shape
    abspids = abspids.reshape(-1)

    if ignored_pids is None:
        charged = np.asarray([pid in CHARGED_PIDS for pid in abspids], dtype=bool)
    else:
        charged = np.asarray([(pid in CHARGED_PIDS) and (pid not in ignored_pids) 
                              for pid in abspids], dtype=bool)

    return charged.reshape(orig_shape)

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
