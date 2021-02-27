r"""# Energy and Angular Measures

The appropriate notions of energy and angle depend on the collider context.
Typically, one wants to work with observables that respect the appropriate
Lorentz subgroup for the collision type of interest. EnergyFlow is capable of
handling two broad classes of measures: $e^+e^-$ and hadronic, which are
selected using the required `measure` argument. For substructure applications,
it is often convenient to normalize the energies so that $\sum_iz_i=1$. The
`normed` keyword argument is provided to control normalization of the energies
(default is `True`). Measures also deal with converting between different
representations of particle momenta, e.g. Cartesian `[E,px,py,pz]` or hadronic
`[pt,y,phi,m]`.

Each measure comes with a parameter $\beta>0$ which controls the relative
weighting between smaller and larger anglular structures. This can be set using
the `beta` keyword argument (default is `1`). when using an EFM measure, `beta`
is ignored as EFMs require $\beta=2$. There is also a $\kappa$ parameter to
control the relative weighting between soft and hard energies. This can be set
using the `kappa` keyword argument (default is `1`). Only `kappa=1` yields
collinear-safe observables.

Prior to version `1.1.0`, the interaction of the `kappa` and `normed` options
resulted in potentially unexpected behavior. As of version `1.1.0`, the flag
`kappa_normed_behavior` has been added to give the user explicit control over
the behavior when `normed=True` and `kappa!=1`. See the description of this
option below for more detailed information.

The usage of EFMs throughout the EnergyFlow package is also controlled through
the `Measure` interface. There are special measure, `'hadrefm'` and `'eeefm'`
that are used to deploy EFMs.

Beyond the measures implemented here, the user can implement their own custom
measure by passing in $\{z_i\}$ and $\{\theta_{ij}\}$ directly to the EFP
classes. Custom EFM measures can be implemented by passing in $\{z_i\}$ and
$\{\hat n_i\}$.

## Hadronic Measures

For hadronic collisions, observables are typically desired to be invariant
under boosts along the beam direction and rotations about the beam direction.
Thus, particle transverse momentum $p_T$ and rapidity-azimuth coordinates
$(y,\phi)$ are used.

There are two hadronic measures implemented in EnergyFlow that work for any
$\beta$: `'hadr'` and `'hadrdot'`. These are listed explicitly below.

`'hadr'`:

\[z_i=p_{T,i}^{\kappa},\quad\quad \theta_{ij}=(\Delta y_{ij}^2 + 
\Delta\phi_{ij}^2)^{\beta/2}.\]

`'hadrdot'`:

\[z_i=p_{T,i}^{\kappa},\quad\quad \theta_{ij}=\left(\frac{2p^\mu_ip_{j\mu}}
{p_{T,i}p_{T,j}}\right)^{\beta/2}.\]

The hadronic EFM measure is `'hadrefm'`, which is equivalent to `'hadrdot'`
with $\beta=2$ when used to compute EFPs, but works with the EFM-based
implementation.

## *e+e-* Measures

For $e^+e^-$ collisions, observables are typically desired to be invariant
under the full group of rotations about the interaction point. Since the center
of momentum energy is known, the particle energy $E$ is typically used. For the
angular measure, pairwise Lorentz contractions of the normalized particle
four-momenta are used.

There is one $e^+e^-$ measure implemented that works for any $\beta$.

`'ee'`:

\[z_i = E_{i}^{\kappa},\quad\quad \theta_{ij} = \left(\frac{2p_i^\mu p_{j \mu}}
{E_i E_j}\right)^{\beta/2}.\]

The $e^+e^-$ EFM measure is `'eeefm'`, which is equivalent to `'ee'` with
$\beta=2$ when used to compute EFPs, but works with the EFM-based
implementation.
"""

#  __  __ ______           _____ _    _ _____  ______
# |  \/  |  ____|   /\    / ____| |  | |  __ \|  ____|
# | \  / | |__     /  \  | (___ | |  | | |__) | |__
# | |\/| |  __|   / /\ \  \___ \| |  | |  _  /|  __|
# | |  | | |____ / ____ \ ____) | |__| | | \ \| |____
# |_|  |_|______/_/    \_\_____/ \____/|_|  \_\______|

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
import warnings

import numpy as np
import six

from energyflow.utils import transfer
from energyflow.utils.particle_utils import *

__all__ = ['Measure']

# special value of kappa indicating "particle flow"
PF_MARKER = 'pf'

# form theta_ij**2 matrix from array of (rapidity,phi) values
# theta_ij**2 = (y_i - y_j)**2 + (phi_i - phi_j)**2
def _thetas2_from_yphis(yphis):
    X = yphis[:,np.newaxis] - yphis[np.newaxis,:]
    X[...,0] **= 2
    X[...,1] = (np.pi - np.abs(np.abs(X[...,1]) - np.pi))**2
    return X[...,0] + X[...,1]

# get theta_ij**2 matrix from four-vectors using combination of above functions
def _thetas2_from_p4s(p4s):
    return _thetas2_from_yphis(np.vstack([ys_from_p4s(p4s), phis_from_p4s(p4s)]).T)

# phats are normalized by the energies
def _phat_func(Es, ps):
    return ps/Es[:,np.newaxis]

# phats are left alone for particle-flow
def _pf_phat_func(Es, ps):
    return ps

MEASURE_KWARGS = {'measure', 'beta', 'kappa', 'normed', 'coords', 
                  'check_input', 'kappa_normed_behavior'}

###############################################################################
# Measure 
###############################################################################

class Measure(six.with_metaclass(ABCMeta, object)):
    
    """Class for handling measure options, described above."""

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

    # Measure(measure, beta=1, kappa=1, normed=True, coords=None,
    #                  check_input=True, kappa_normed_behavior='new')
    def __init__(self, measure, beta=1, kappa=1, normed=True, coords=None,
                                check_input=True, kappa_normed_behavior='new'):
        r"""Processes inputs according to the measure choice and other options.

        **Arguments**

        - **measure** : _string_
            - The string specifying the energy and angular measures to use.
        - **beta** : _float_
            - The angular weighting exponent $\beta$. Must be positive.
        - **kappa** : {_float_, `'pf'`}
            - If a number, the energy weighting exponent $\kappa$. If `'pf'`,
            use $\kappa=v$ where $v$ is the valency of the vertex. `'pf'`
            cannot be used with measure `'hadr'`. Only IRC-safe for `kappa=1`.
        - **normed** : bool
            - Whether or not to use normalized energies/transverse momenta.
        - **coords** : {`'ptyphim'`, `'epxpypz'`, `None`}
            - Controls which coordinates are assumed for the input. If
            `'ptyphim'`, the fourth column (the masses) is optional and
            massless particles are assumed if it is not present. If `None`,
            coords with be `'ptyphim'` if using a hadronic measure and
            `'epxpypz'` if using the e+e- measure.
        - **check_input** : bool
            - Whether to check the type of input each time or assume the first
            input type.
        - **kappa_normed_behavior** : {`'new'`, `'orig'`}
            - Determines how `'kappa'`!=1 interacts with normalization of the
            energies. A value of `'new'` will ensure that `z` is truly the
            energy fraction of a particle, so that $z_i=E_i^\kappa/\left(
            \sum_{i=1}^ME_i\right)^\kappa$. A value of `'orig'` will keep the
            behavior prior to version `1.1.0`, which used $z_i=E_i^\kappa/
            \sum_{i=1}^M E_i^\kappa$.
        """

        # store parameters
        transfer(self, locals(), ['measure', 'kappa', 'normed', 'coords', 
                                  'check_input', 'kappa_normed_behavior'])

        # check that options are appropriate
        if self.coords not in {None, 'epxpypz', 'ptyphim'}:
            raise ValueError("coords must be one of 'epxpypz', 'ptyphim', or None")
        if self.kappa_normed_behavior not in {'new', 'orig'}:
            raise ValueError("kappa_normed_behavior must be 'new' or 'orig'")

        # verify beta
        self.beta = float(beta)
        self.half_beta = self.beta/2
        assert self.beta > 0

        # measure function is not yet set
        self.need_meas_func = True

        # handle normed and kappa options
        self._z_func, self._phat_func = self._z_unnormed_func, _phat_func
        if self.kappa == PF_MARKER:
            self._phat_func = _pf_phat_func

            # cannot subslice when kappa = pf
            self.subslicing = False

            # if normed was set to True, warn them about this
            if self.normed:
                raise ValueError("Normalization not supported when kappa='pf'")

            self._z_func = self._pf_z_func

        # we're norming the correlators
        elif self.normed:
            if self.kappa_normed_behavior == 'new':
                self._z_func = self._z_normed_new_func
            else:
                self._z_func = self._z_normed_orig_func

    # returns zs for numeric kappa and normed=False
    def _z_unnormed_func(self, Es):
        return Es**self.kappa

    # returns zs for numeric kappa and normed=True, original style
    def _z_normed_orig_func(self, Es):
        zs = Es**self.kappa
        return zs/np.sum(zs)

    # returns zs for numeric kappa and normed=True, new style
    def _z_normed_new_func(self, Es):
        zs = Es**self.kappa
        return zs/np.sum(Es)**self.kappa

    # kappa indicates particle flow, so make energies 1
    def _pf_z_func(self, Es):
        return np.ones(Es.shape)

    def evaluate(self, arg):
        """Evaluate the measure on a set of particles. Returns `zs`, `thetas`
        if using a non-EFM measure and `zs`, `nhats` otherwise.

        **Arguments**

        - **arg** : _2-d numpy.ndarray_
            - A two-dimensional array of the particles with each row being a 
            particle and the columns specified by the `coords` attribute.

        **Returns**

        - (_ 1-d numpy.ndarray_, _2-d numpy.ndarray_)
            - If using a non-EFM measure, (`zs`, `thetas`) where `zs` is a
            vector of the energy fractions for each particle and `thetas`
            is the distance matrix between the particles. If using an EFM
            measure, (`zs`, `nhats`) where `zs` is the same and `nhats` is
            the `[E,px,py,pz]` of each particle divided by its energy (if
            in an $e^+e^-$ context) or transverse momentum (if in a hadronic
            context.)
        """

        # check type only if needed
        if self.need_meas_func or self.check_input:
            self.set_meas_func(arg)

        # get zs and angles (already normalized)
        return self.meas_func(arg)

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

###############################################################################
# HadronicMeasure
###############################################################################

class HadronicMeasure(Measure):

    @staticmethod
    def factory(measure):
        if measure == 'hadrefm':
            return HadronicEFMMeasure
        if measure == 'hadrdot':
            return HadronicDotMeasure
        if measure == 'hadr':
            return HadronicDefaultMeasure
        else:
            raise ValueError('Hadronic measure {} not understood'.format(measure))

    def __init__(self, *args, **kwargs):
        super(HadronicMeasure, self).__init__(*args, **kwargs)
        if self.coords is None:
            self.coords = 'ptyphim'
        self.epxpypz = (self.coords == 'epxpypz')

    def array_handler(self, dim):
        if dim == 3:
            if self.epxpypz:
                raise ValueError('epxpypz coords require second dimension of arg to be 4')
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
        if self.epxpypz:
            return np.atleast_1d(pts_from_p4s(arg)), arg
        else:
            return arg[:,0], np.atleast_2d(p4s_from_ptyphims(arg))

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
        if measure == 'eeefm':
            return EEEFMMeasure
        if measure == 'ee':
            return EEDefaultMeasure
        else:
            raise ValueError('EE measure {} not understood'.format(measure))

    def __init__(self, *args, **kwargs):
        super(EEMeasure, self).__init__(*args, **kwargs)
        if self.coords is None:
            self.coords = 'epxpypz'
        self.epxpypz = self.coords == 'epxpypz'

    def array_handler(self, dim):
        if dim < 2:
            raise ValueError('second dimension of arg must be >= 2')
        if not self.epxpypz and dim not in [3, 4]:
            raise ValueError('ptyphim coords only work with inputs of dimension 3 or 4')
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
        super(HadronicDefaultMeasure, self).__init__(*args, **kwargs)
        if self.kappa == PF_MARKER:
            raise ValueError('particle flow not available for HadronicDefaultMeasure')

    def ndarray_dim3(self, arg):
        return self._z_func(arg[:,0]), _thetas2_from_yphis(arg[:,(1,2)])**self.half_beta

    def ndarray_dim4(self, arg):
        if self.epxpypz:
            return self._z_func(np.atleast_1d(pts_from_p4s(arg))), _thetas2_from_p4s(arg)**self.half_beta
        else:
            return self.ndarray_dim3(arg[:,:3])

    def pseudojet(self, arg):
        pts, constituents = super(HadronicDefaultMeasure, self).pseudojet(arg)
        thetas = np.asarray([[c1.delta_R(c2) for c2 in constituents] for c1 in constituents])
        return self._z_func(pts), thetas**self.beta

###############################################################################
# HadronicDotMeasure
###############################################################################

class HadronicDotMeasure(HadronicMeasure):

    subslicing = None
    metric = flat_metric(4)

    def ndarray_dim3(self, arg):
        phats = self._phat_func(arg[:,0], np.atleast_2d(p4s_from_ptyphims(arg)))
        return self._z_func(arg[:,0]), self._ps_dot(phats)**self.half_beta

    def ndarray_dim4(self, arg):
        pts, ps = super(HadronicDotMeasure, self).ndarray_dim4(arg)
        return self._z_func(pts), self._ps_dot(self._phat_func(pts, ps))**self.half_beta

    def pseudojet(self, arg):
        pts, constituents = super(HadronicDotMeasure, self).pseudojet(arg)
        p4s = np.asarray([[c.e(), c.px(), c.py(), c.pz()] for c in constituents])
        return self._z_func(pts), self._ps_dot(self._phat_func(pts, p4s))**self.half_beta

###############################################################################
# HadronicEFMMeasure
###############################################################################

class HadronicEFMMeasure(HadronicMeasure):

    subslicing = False

    def __init__(self, *args, **kwargs):
        super(HadronicEFMMeasure, self).__init__(*args, **kwargs)
        self.beta, self.half_beta = 2, 1

    def ndarray_dim3(self, arg):
        return self._z_func(arg[:,0]), self._phat_func(arg[:,0], np.atleast_2d(p4s_from_ptyphims(arg)))

    def ndarray_dim4(self, arg):
        pts, ps = super(HadronicEFMMeasure, self).ndarray_dim4(arg)
        return self._z_func(pts), self._phat_func(pts, ps)

    def pseudojet(self, arg):
        pts, constituents = super(HadronicEFMMeasure, self).pseudojet(arg)
        p4s = np.asarray([[c.e(), c.px(), c.py(), c.pz()] for c in constituents])
        return self._z_func(pts), self._phat_func(pts, p4s)

###############################################################################
# EEDefaultMeasure
###############################################################################

class EEDefaultMeasure(EEMeasure):

    subslicing = None

    def ndarray_dim_arb(self, arg):
        if not self.epxpypz:
            arg = np.atleast_2d(p4s_from_ptyphims(arg))
        return self._z_func(arg[:,0]), self._ps_dot(self._phat_func(arg[:,0], arg))**self.half_beta

    def pseudojet(self, arg):
        Es, constituents =  super(EEDefaultMeasure, self).pseudojet(arg)
        p4s = np.asarray([[c.e(), c.px(), c.py(), c.pz()] for c in constituents])
        return self._z_func(Es), self._ps_dot(self._phat_func(Es, p4s))**self.half_beta

###############################################################################
# EEEFMMeasure
###############################################################################

class EEEFMMeasure(EEMeasure):

    subslicing = True

    def __init__(self, *args, **kwargs):
        super(EEEFMMeasure, self).__init__(*args, **kwargs)
        self.beta, self.half_beta = 2, 1

    def ndarray_dim_arb(self, arg):
        if not self.epxpypz:
            arg = np.atleast_2d(p4s_from_ptyphims(arg))
        return self._z_func(arg[:,0]), self._phat_func(arg[:,0], arg)

    def pseudojet(self, arg):
        Es, constituents = super(EEEFMMeasure, self).pseudojet(arg)
        p4s = np.asarray([[c.e(), c.px(), c.py(), c.pz()] for c in constituents])
        return self._z_func(Es), self._phat_func(Es, p4s)
