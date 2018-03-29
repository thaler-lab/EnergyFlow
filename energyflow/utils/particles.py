"""
### Particle Tools

Tools to compute particle kinematic quantities from four-vectors,
such as transverse momentum $p_T$, rapidity $y$, and azimuth $\phi$.
Also includes functions for converting from $(p_T,y,\phi)$ to 
four-momenta.

"""

from __future__ import absolute_import, division

import numpy as np

__all__ = [
    'flat_metric',
    'p4s_from_ptyphims',
    'p4s_from_ptyphis',
    'pts_from_p4s',
    'ys_from_p4s',
    'phis_from_p4s'
]

###############################################################################
# Measure helpers
###############################################################################

long_metric = np.array([1.] + [-1.]*100)
def flat_metric(dim):
    """The Minkowski metric in `dim` spacetime dimensions in the mostly-minus convention.
    
    **Arguments**

    - **dim** : _int_
        - The number of spacetime dimensions (thought to be four in our universe).

    **Returns**

    - _numpy.ndarray_
        - A `dim`-length array (not matrix) equal to `[+1, -1, ..., -1]`
    """

    if dim <= 101:
        return long_metric[:dim]
    return np.asarray([1.] + [-1.]*(dim-1))

def p4s_from_ptyphims(ptyphim):
    """Convert transverse momenta $p_T$, rapidities $y$, azimuths $\phi$, and masses $m$ to four-vectors
    
    **Arguments**

    - **ptyphims** : _numpy.ndarray_
        - An event as an `(M, 4)` array of `[pT, y, phi, m]` for each particle.

    **Returns**

    - _numpy.ndarray_
        - An event as an `(M, 4)` array of four-vectors `[E, px, py, pz]` for each particle.
    """

    pts, ys, phis, ms = [ptyphim[:,i] for i in range(4)]
    Ets = np.sqrt(pts**2 + ms**2)
    return np.vstack([Ets*np.cosh(ys), pts*np.cos(phis), pts*np.sin(phis), Ets*np.sinh(ys)]).T

def p4s_from_ptyphis(ptyphis):
    """Convert transverse momenta $p_T$, rapidities $y$, and azimuths $\phi$ to massless four-vectors
    
    **Arguments**

    - **ptyphis** : _numpy.ndarray_
        - An event as an `(M, 3)` array of `[pT, y, phi]` for each particle.

    **Returns**

    - _numpy.ndarray_
        - An event as an `(M, 4)` array of four-vectors `[E, px, py, pz]` for each particle.
    """

    pts, ys, phis = ptyphis[:,0], ptyphis[:,1], ptyphis[:,2]
    return (pts*np.vstack([np.cosh(ys), np.cos(phis), np.sin(phis), np.sinh(ys)])).T

def pts_from_p4s(p4s):
    """Calculate the transverse momenta of a collection of four-vectors
    
    **Arguments**

    - **p4s** : _numpy.ndarray_
        - An event as an `(M, 4)` array of four-vectors `[E, px, py, pz]` for each particle.

    **Returns**

    - _numpy.ndarray_
        - An `M`-length array of transverse momenta `pT` for each particle.
    """


    return np.sqrt(p4s[:,1]**2 + p4s[:,2]**2)

def ys_from_p4s(p4s):
    """Calculate the rapidities of a collection of four-vectors
    
    **Arguments**

    - **p4s** : _numpy.ndarray_
        - An event as an `(M, 4)` array of four-vectors `[E, px, py, pz]` for each particle.

    **Returns**

    - _numpy.ndarray_
        - An `M`-length array of rapidities `y` for each particle.
    """


    return 0.5*np.log((p4s[:,0]+p4s[:,3])/(p4s[:,0]-p4s[:,3]))

def phis_from_p4s(p4s):
    """Calculate the azimuthal angles of a collection of four-vectors.
    The angles are chosen to be in the inverval $[0,2\pi]$.
    
    **Arguments**

    - **p4s** : _numpy.ndarray_
        - An event as an `(M, 4)` array of four-vectors `[E, px, py, pz]` for each particle.

    **Returns**

    - _numpy.ndarray_
        - An `M`-length array of azimuthal angles `phi` for each particle.
    """

    phis = np.arctan2(p4s[:,2], p4s[:,1])
    phis[phis<0] += 2*np.pi
    return phis
