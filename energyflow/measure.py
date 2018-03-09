"""
### Energy and Angular Measures

The appropriate notions of energy and angle depend on the collider context. Typically, one wants
to work with observables that respect the appropriate Lorentz subgroup for the collision type
of interest. EnergyFlow is capable of handling two broad classes of measures: $e^+e^-$ and
hadronic, which are selected using the `measure` keyword argument (default is `'hadrdot'`).
For substructure applications, it is often convenient to normalize the energies so that
$\\sum_iz_i=1$. The `normed` keyword argument is provided to control normalization of the
energies (default is `True`).

Each measure comes with a parameter $\\beta>0$ which controls the relative weighting between
smaller and larger anglular structures. This can be set using the `beta` keyword argument
(default is `2`). There is also a $\\kappa$ parameter to control the relative weighting
between soft and hard energies. This can be set using the `kappa` keyword argument
(default is `1`). Only `kappa=1` yields collinear-safe observables.

Beyond the measures implemented here, the user can implement their own custom measure by
passing $\\{z_i\\}$ and $\\{\\theta_{ij}\\}$ in directly to the EFP classes.

#### Hadronic Measures

For hadronic collisions, observables are typically desired to be invariant under boosts along
the beam direction and rotations about the beam direction. Thus, particle transverse momentum
$p_T$ and rapidity-azimuth coordinates $(y,\\phi)$ are often used.

There are two hadronic measures implemented in EnergyFlow: `'hadr'` and `'hadrdot'` (the default).
These are listed explicitly below.

`'hadr'`:
$$z_i=p_{T,i}^{\\kappa},\\quad\\quad \\theta_{ij}=(\\Delta y_{ij}^2 + \\Delta\\phi_{ij}^2)^{\\beta/2}.$$

`'hadrdot'`:
$$z_i=p_{T,i}^{\\kappa},\\quad\\quad \\theta_{ij}=\\left(\\frac{2p^\\mu_ip_{j\\mu}}{p_{T,i}p_{T,j}}
\\right)^{\\beta/2}.$$

#### *e+e-* Measures

For $e^+e^-$ collisions, observables are typically desired to be invariant under the full
group of reotations about the interaction point. Since the center of momentum energy is known,
the particle energy $E$ is typically used. For the angular measure, pairwise Lorentz contractions
of the normalized particle four-momenta are used.

There is one $e^+e^-$ measure implemented in the EnergyFlow framework.

`'ee'`:
$$z_i = E_{i}^{\\kappa},
\\quad\\quad \\theta_{ij} = \\left(\\frac{2p_i^\\mu p_{j \\mu}}{E_i E_j}\\right)^{\\beta/2}.$$
"""

# When using the `'hadrdot'` or `'ee'` measures with $\\beta=2$, the user can also append `'efm'`
# to the measure name in order to perform the calculation in $O(M)$ with
# [Energy Flow Moments](/docs/efm).


from __future__ import absolute_import, division

from abc import ABCMeta, abstractmethod

import numpy as np
from six import add_metaclass

from energyflow.utils import transfer
from energyflow.utils.particles import *

__all__ = ['Measure']

# special value of kappa indicating "particle flow"
pf_marker = 'pf'

# form theta_ij**2 matrix from array of (rapidity,phi) values
# theta_ij**2 = (y_i - y_j)**2 + (phi_i - phi_j)**2
def thetas2_from_yphis(yphis):
    X = yphis[:,np.newaxis] - yphis[np.newaxis,:]
    X[:,:,0] **= 2
    X[:,:,1] = (np.pi - np.abs(np.abs(X[:,:,1]) - np.pi))**2
    return X[:,:,0] + X[:,:,1]

# get theta_ij**2 matrix from four-vectors using combination of above functions
def thetas2_from_p4s(p4s):
    return thetas2_from_yphis(np.vstack([ys_from_p4s(p4s), phis_from_p4s(p4s)]).T)

# kappa is a number, so raise energies to that number and form phats
def kappa_func(Es, ps, kappa):
    return Es**kappa, ps/Es[:,np.newaxis]

# kappa indicates particle flow, so make energies 1 and leave ps alone
def pf_func(Es, ps, kappa):
    return np.ones(Es.shape), ps


###############################################################################
# Measure 
###############################################################################
@add_metaclass(ABCMeta)
class Measure:
    
    """Class for dealing with any kind of measure."""

    def __new__(cls, *args, **kwargs):
        if cls is Measure:
            measure = args[0]
            if 'hadr' in measure:
                return super(Measure, cls).__new__(HadronicMeasure.factory(measure))
            if 'ee' in measure:
                return super(Measure, cls).__new__(EEMeasure.factory(measure))
            raise NotImplementedError('measure {} is unknown'.format(measure))
        else:
            return super(Measure, cls).__new__(cls)

    def __init__(self, measure, beta=2, kappa=1, normed=True, check_input=True):
        """Processes inputs according to the measure choice.

        **Arguments**

        - **measure** : _string_
            - The string specifying the energy and angular measures to use.

        - **beta** : _float_ or 'pf'
            - The angular weighting exponent $\\beta$. Must be positive.

        - **kappa** : _float_
            - If a number, the energy weighting exponent $\\kappa$. If `'pf'`, use $\\kappa=v$
             where $v$ is the valency of the vertex. `'pf'` can be used with measure except `'hadr'`.
             Only IRC-safe for `kappa=1`.

        - **normed** : bool
            - Whether or not to use normalized energies
        
        - **check_input** : bool
            - Whether to check the type of input each time or assume the first input type.
        """

        transfer(self, locals(), ['measure', 'kappa', 'normed', 'check_input'])
        self.beta = float(beta)
        assert self.beta > 0

        self.half_beta = self.beta/2
        self.need_meas_func = True

    def evaluate(self, arg):
        """ Evaluate
        """

        # check type only if needed
        if self.need_meas_func or self.check_input:
            self.set_meas_func(arg)

        # get zs and angles 
        zs, angles = self.meas_func(arg)

        return (zs/np.sum(zs) if self.normed else zs), angles

    def set_meas_func(self, arg):

        # support arg as numpy.ndarray
        if isinstance(arg, np.ndarray):
            self.meas_func = self.array_handler(arg.shape[1])

        # support arg as list (of lists)
        elif isinstance(arg, list):
            array_meas = self.array_handler(len(arg[0]))
            def wrapped_meas(arg):
                return array_meas(np.asarray(arg))
            self.meas_func = wrapped_meas

        # support arg as fastjet pseudojet
        elif hasattr(arg, 'constituents'):
            self.meas_func = self.pseudojet

        # raise error if not one of these types
        else:
            raise TypeError('arg is not one of numpy.ndarray, list, or fastjet.PseudoJet')

        self.need_meas_func = False

    @abstractmethod
    def array_handler(self, dim):
        pass

    @abstractmethod
    def pseudojet(self, arg):
        pass

    def _ps_dot(self, ps):
        return np.abs(2*np.dot(ps[:,np.newaxis]*ps[np.newaxis,:], self.metric))

    def _set_k_func(self):
        self._k_func = kappa_func
        if self.kappa == pf_marker:
            self.normed = False
            self.subslicing = False
            self._k_func = pf_func


###############################################################################
# HadronicMeasure
###############################################################################
class HadronicMeasure(Measure):

    @staticmethod
    def factory(measure):
        if 'efm' in measure:
            return HadronicEFMMeasure
        if 'dot' in measure:
            return HadronicDotMeasure
        return HadronicDefaultMeasure

    def __init__(self, *args, **kwargs):
        super(HadronicMeasure, self).__init__(*args, **kwargs)
        self._set_k_func()

    def array_handler(self, dim):
        if dim == 3:
            return self.ndarray_dim3
        elif dim == 4:
            return self.ndarray_dim4
        else:
            raise ValueError('second dimension of arg must be either 3 or 4')

    @abstractmethod
    def ndarray_dim3(self, arg):
        pass

    @abstractmethod
    def ndarray_dim4(self, arg):
        pass

    @abstractmethod
    def pseudojet(self, arg):
        constituents = arg.constituents()
        pts = np.asarray([c.pt() for c in constituents])
        return pts, constituents


###############################################################################
# EEMeasure
###############################################################################
class EEMeasure(Measure):

    @staticmethod
    def factory(measure):
        if 'efm' in measure:
            return EEEFMMeasure
        return EEDefaultMeasure

    def __init__(self, *args, **kwargs):
        super(EEMeasure, self).__init__(*args, **kwargs)
        self._set_k_func()

    def array_handler(self, dim):
        if dim < 2:
            raise ValueError('second dimension of arg must be >= 2')
        self.metric = flat_metric(dim)
        return self.ndarray_dim_arb

    @abstractmethod
    def ndarray_dim_arb(self, arg):
        pass

    @abstractmethod
    def pseudojet(self, arg):
        constituents = arg.constituents()
        Es = np.asarray([c.e() for c in constituents])
        return Es, constituents


###############################################################################
# HadronicDefaultMeasure
###############################################################################
class HadronicDefaultMeasure(HadronicMeasure):

    subslicing = None

    def __init__(self, *args, **kwargs):
        # skip __init__ of HadronicMeasure
        super(HadronicMeasure, self).__init__(*args, **kwargs)
        if self.kappa == pf_marker:
            raise ValueError('particle flow not available for HadronicDefaultMeasure')

    def ndarray_dim3(self, arg):
        return arg[:,0]**self.kappa, thetas2_from_yphis(arg[:,(1,2)])**self.half_beta

    def ndarray_dim4(self, arg):
        return pts_from_p4s(arg)**self.kappa, thetas2_from_p4s(arg)**self.half_beta

    def pseudojet(self, arg):
        pts, constituents = super(HadronicDefaultMeasure, self).pseudojet(arg)
        thetas = np.asarray([[c1.delta_R(c2) for c2 in constituents] for c1 in constituents])
        return pts**self.kappa, thetas**self.beta


###############################################################################
# HadronicDotMeasure
###############################################################################
class HadronicDotMeasure(HadronicMeasure):

    subslicing = None
    metric = flat_metric(4)

    def ndarray_dim3(self, arg):
        pts, p4s = self._k_func(arg[:,0], p4s_from_ptyphis(arg), self.kappa)
        return pts, self._ps_dot(p4s)**self.half_beta

    def ndarray_dim4(self, arg):
        pts, p4s = self._k_func(pts_from_p4s(arg), arg, self.kappa)
        return pts, self._ps_dot(p4s)**self.half_beta

    def pseudojet(self, arg):
        pts, constituents = super(HadronicDotMeasure, self).pseudojet(arg)
        p4s = np.asarray([[c.e(), c.px(), c.py(), c.pz()] for c in constituents])
        pts, p4s = self._k_func(pts, p4s, self.kappa)
        return pts, self._ps_dot(p4s)**self.half_beta


###############################################################################
# HadronicEFMMeasure
###############################################################################
class HadronicEFMMeasure(HadronicMeasure):

    subslicing = False

    def ndarray_dim3(self, arg):
        return self._k_func(arg[:,0], p4s_from_ptyphis(arg), self.kappa)

    def ndarray_dim4(self, arg):
        return self._k_func(pts_from_p4s(arg), arg, self.kappa)

    def pseudojet(self, arg):
        pts, constituents = super(HadronicEFMMeasure, self).pseudojet(arg)
        p4s = np.asarray([[c.e(), c.px(), c.py(), c.pz()] for c in constituents])
        return self._k_func(pts, p4s, self.kappa)


###############################################################################
# EEDefaultMeasure
###############################################################################
class EEDefaultMeasure(EEMeasure):

    subslicing = None

    def ndarray_dim_arb(self, arg):
        Es, ps = self._k_func(arg[:,0], arg, self.kappa)
        return Es, self._ps_dot(ps)**self.half_beta

    def pseudojet(self, arg):
        Es, constituents =  super(EEDefaultMeasure, self).pseudojet(arg)
        p4s = np.asarray([[c.e(), c.px(), c.py(), c.pz()] for c in constituents])
        Es, p4s = self._k_func(Es, p4s, self.kappa)
        return Es, self._ps_dot(p4s)**self.half_beta


###############################################################################
# EEEFMMeasure
###############################################################################
class EEEFMMeasure(EEMeasure):

    subslicing = True

    def ndarray_dim_arb(self, arg):
        return self._k_func(arg[:,0], arg, self.kappa)

    def pseudojet(self, arg):
        Es, constituents = super(EEEFMMeasure, self).pseudojet(arg)
        p4s = np.asarray([[c.e(), c.px(), c.py(), c.pz()] for c in constituents])
        return self._k_func(Es, p4s, self.kappa)
