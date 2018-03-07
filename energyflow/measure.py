"""Implementation of the Measure class and its helpers."""

from __future__ import absolute_import, division

from abc import ABCMeta, abstractmethod

import numpy as np
from six import add_metaclass

from energyflow.utils import transfer

__all__ = [
    'Measure',
    'flat_metric',
    'p4s_from_ptyphis',
    'pts_from_p4s',
    'ys_from_p4s',
    'phis_from_p4s'
]

###############################################################################
# Measure helpers
###############################################################################

# special value of kappa indicating "particle flow"
pf_marker = 'pf'

# Minkowski metric in dim spacetime dimensions
long_metric = np.array([1.] + [-1.]*100)
def flat_metric(dim):
    if dim <= 101:
        return long_metric[:dim]
    return np.asarray([1.] + [-1.]*(dim-1))

# get (massless) four-vectors from (pt,y,phi) values
def p4s_from_ptyphis(ptyphis):
    pts, ys, phis = ptyphis[:,0], ptyphis[:,1], ptyphis[:,2]
    return (pts*np.vstack([np.cosh(ys), np.cos(phis), np.sin(phis), np.sinh(ys)])).T

# get pts from four-vectors
def pts_from_p4s(p4s):
    return np.sqrt(p4s[:,1]**2 + p4s[:,2]**2)

# rapidity from four-vectors
def ys_from_p4s(p4s):
    return 0.5*np.log((p4s[:,0]+p4s[:,3])/(p4s[:,0]-p4s[:,3]))

# azimuthal angle phi in [0,2pi]
def phis_from_p4s(p4s):
    phis = np.arctan2(p4s[:,2], p4s[:,1])
    phis[phis<0] += 2*np.pi
    return phis

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

        Arguments
        ---------
        measure : string
            - The string specifying the measure.
        beta : float/int
            - The exponent $\beta$.
        normed : bool
            - Whether or not to use normalized energies.
        check_input : bool
            - Whether or not to check the input for each new event.
        """

        transfer(self, locals(), ['measure', 'kappa', 'normed', 'check_input'])
        self.beta = float(beta)
        self.half_beta = self.beta/2
        self.need_meas_func = True

    def __call__(self, arg):

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

    def __init__(self, *args):
        super(HadronicMeasure, self).__init__(*args)
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

    def __init__(self, *args):
        super(EEMeasure, self).__init__(*args)
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

    def __init__(self, *args):
        # skip __init__ of HadronicMeasure
        super(HadronicMeasure, self).__init__(*args)
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
