"""Implementation of the Measure class and its helpers."""
from __future__ import absolute_import, division
from abc import ABCMeta, abstractmethod
import numpy as np
from six import add_metaclass

__all__ = ['Measure']

@add_metaclass(ABCMeta)
class Measure:
    
    """Class for dealing with any kind of measure."""

    def __new__(cls, *args, **kwargs):
        if cls is Measure:
            measure = args[0]
            if 'hadr' in measure:
                return super(Measure, cls).__new__(HadronicMeasure._factory(measure))
            if 'ee' in measure:
                return super(Measure, cls).__new__(EEMeasure._factory(measure))
            raise NotImplementedError('measure {} is unknown'.format(measure))
        else:
            return super(Measure, cls).__new__(cls)

    def __init__(self, measure, beta, normed, check_type):

        """ 
        Parameters
        ----------
        measure : 
        normed : 
        """

        assert beta > 0, 'beta must be greater than zero'

        self.measure = measure
        self.beta = float(beta)
        self.normed = normed
        self.check_type = check_type

        self._half_beta = self.beta/2
        self._lacks_meas_func = True

    def __call__(self, arg):

        # check type only if needed
        if self._lacks_meas_func or self.check_type:
            self._set_meas_func(arg)

        # get zs and thetas 
        zs, thetas = self._meas_func(arg)

        return (zs/np.sum(zs) if self.normed else zs), thetas

    def _set_meas_func(self, arg):

        # support arg as numpy.ndarray
        if isinstance(arg, np.ndarray):
            if not self._ndarray_handler(arg.shape[1]):
                raise IndexError('second dimension of arg must be in {}'.format(self._allowed_dims))

        # support arg as list (of lists)
        elif isinstance(arg, list):
            if not self._list_handler(len(arg[0])):
                raise IndexError('second dimension of arg must be in {}'.format(self._allowed_dims))

        # support arg as fastjet pseudojet
        elif hasattr(arg, 'constituents'):
            self._pseudojet_handler()

        # raise error if not one of these types
        else:
            raise TypeError('arg is not one of numpy.ndarray, list, or fastjet.PseudoJet')

        self._lacks_meas_func = False

    @abstractmethod
    def _ndarray_handler(self, dim):
        pass

    @abstractmethod
    def _list_handler(self, dim):
        pass

    @abstractmethod
    def _pseudojet_handler(self):
        pass

    def _p4s_dot(self, p4s, Es):
        p4hats = p4s/Es[:,np.newaxis]
        X = (p4hats[:,np.newaxis]*p4hats[np.newaxis,:]).T
        return (2*np.abs(X[0] - X[1] - X[2] - X[3]))**self._half_beta

class HadronicMeasure(Measure):

    @staticmethod
    def _factory(measure):
        if 'dot' in measure:
            return HadronicDotMeasure
        else:
            return HadronicDefaultMeasure 

    _allowed_dims = [3, 4]

    def _ndarray_handler(self, dim):
        if dim == 3:
            self._meas_func = self._ndarray_dim3
        elif dim == 4:
            self._meas_func = self._ndarray_dim4
        else:
            return False
        return True
        
    def _list_handler(self, dim):
        if dim == 3:
            self._meas_func = self._list_dim3
        elif dim == 4:
            self._meas_func = self._list_dim4
        else:
            return False
        return True

    def _pseudojet_handler(self):
        self._meas_func = self._pseudojet
        return True

    @abstractmethod
    def _ndarray_dim3(self, arg):
        pass

    @abstractmethod
    def _ndarray_dim4(self, arg):
        pass

    def _list_dim3(self, arg):
        return self._ndarray_dim3(np.asarray(arg))

    def _list_dim4(self, arg):
        return self._ndarray_dim4(np.asarray(arg))

    @abstractmethod
    def _pseudojet(self, arg):
        constituents = arg.constituents()
        pts = np.asarray([c.pt() for c in constituents])
        return pts, constituents

    def _pts(self, p4s):
        return np.sqrt(p4s[:,1]**2 + p4s[:,2]**2)

    def _yphis(self, p4s):
        return np.vstack([0.5*np.log((p4s[:,0]+p4s[:,3])/(p4s[:,0]-p4s[:,3])),
                          np.arctan2(p4s[:,2], p4s[:,1])]).T

    def _thetas_from_yphis(self, yphis):
        X = yphis[:,np.newaxis] - yphis[np.newaxis,:]
        X[:,:,0] **= 2
        X[:,:,1] = (np.pi - np.abs(np.abs(X[:,:,1]) - np.pi))**2
        return (X[:,:,0] + X[:,:,1]) ** self._half_beta

    def _p4s_from_ptyphis(self, ptyphis):
        pts, ys, phis = ptyphis[:,0], ptyphis[:,1], ptyphis[:,2]
        return (pts*np.vstack([np.cosh(ys), np.cos(phis), np.sin(phis), np.sinh(ys)])).T

class EEMeasure(Measure):

    @staticmethod
    def _factory(measure):
        return EEDefaultMeasure

    _allowed_dims = [4]

    def _ndarray_handler(self, dim):
        if dim == 4:
            self._meas_func = self._ndarray_dim4
        else:
            return False
        return True
        
    def _list_handler(self, dim):
        if dim == 4:
            self._meas_func = self._list_dim4
        else:
            return False
        return True

    def _pseudojet_handler(self):
        self._meas_func = self._pseudojet
        return True

    @abstractmethod
    def _ndarray_dim4(self, arg):
        pass

    def _list_dim4(self, arg):
        return self._ndarray_dim4(np.asarray(arg))

    @abstractmethod
    def _pseudojet(self, arg):
        constituents = arg.constituents()
        Es = np.asarray([c.e() for c in constituents])
        return Es, constituents

class HadronicDefaultMeasure(HadronicMeasure):

    def _ndarray_dim3(self, arg):
        return arg[:,0], self._thetas_from_yphis(arg[:,(1,2)])

    def _ndarray_dim4(self, arg):
        return self._pts(arg), self._thetas_from_yphis(self._yphis(arg))

    def _pseudojet(self, arg):
        pts, constituents = super()._pseudojet(arg)
        thetas = np.asarray([[c1.delta_R(c2) for c2 in constituents] for c1 in constituents])**self.beta
        return pts, thetas

class HadronicDotMeasure(HadronicMeasure):

    def _ndarray_dim3(self, arg):
        pts = arg[:,0]
        p4s = self._p4s_from_ptyphis(arg)
        return pts, self._p4s_dot(p4s, pts)

    def _ndarray_dim4(self, arg):
        pts = self._pts(arg)
        return pts, self._p4s_dot(arg, pts)

    def _pseudojet(self, arg):
        pts, constituents = super()._pseudojet(arg)
        p4s = np.asarray([[c.e(), c.px(), c.py(), c.pz()] for c in constituents])
        return pts, self._p4s_dot(p4s, pts)

class EEDefaultMeasure(EEMeasure):

    def _ndarray_dim4(self, arg):
        Es = arg[:,0]
        return Es, self._p4s_dot(arg, Es)

    def _pseudojet(self, arg):
        Es, constituents =  super()._pseudojet(arg)
        p4s = np.asarray([[c.e(), c.px(), c.py(), c.pz()] for c in constituents])
        return Es, self._p4s_dot(p4s, Es)
