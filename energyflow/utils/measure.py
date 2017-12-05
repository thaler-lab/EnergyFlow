"""Implementation of the Measure class and its helpers."""
from __future__ import absolute_import, division
import numpy as np

__all__ = ['Measure']

class HadronicMeasure:

    def _hadr_yphi_parse_input(self, arg):

        # support arg as numpy.ndarray
        if isinstance(arg, np.ndarray):
            dim = arg.shape[1]
            if dim == 3:
                self._meas_func = self._hadr_yphi_ndarray_dim3
            if dim == 4:
                self._meas_func = self._hadr_yphi_ndarray_dim4
            else:
                raise IndexError('second dimension of arg must be either size 3 or 4')

        # support arg as list (of lists)
        elif isinstance(arg, list):
            dim = len(arg[0])
            if dim == 3:
                self._meas_func = self._hadr_yphi_list_dim3
            if dim == 4:
                self._meas_func = self._hadr_yphi_list_dim4
            else:
                raise IndexError('second dimension of arg must be either size 3 or 4')

        # support arg as fastjet pseudojet
        elif hasattr(arg, 'constituents'):
            self._meas_func = self._hadr_yphi_pseudojet

        # raise error if not one of these types
        else:
            raise TypeError('arg is not one of numpy.ndarray, list, or fastjet.PseudoJet')

        # set flag to indicate we have (once) determined a type
        self._lacks_type = True

    def _thetas_from_yphi(self, yphis):

        X = yphis[:,np.newaxis] - yphis[np.newaxis,:]
        X[:,:,0] **= 2
        X[:,:,1] = (np.pi - np.abs(np.abs(X[:,:,1]) - np.pi))**2
        return (X[:,:,0] + X[:,:,1]) ** self._half_beta

    def _pts(self, p4s):

        return np.sqrt(p4s[:,1]**2 + p4s[:,2]**2)

    def _yphis(self, p4s):

        return np.arctan2(p4s[:,2], p4s[:,1])

    def _hadr_yphi_ndarray_dim3(self, arg):
        
        return arg[:,0], self._thetas_from_yphi(arg[:,(1,2)])

    def _hadr_yphi_ndarray_dim4(self, arg):
        
        return self._pts(arg), self._thetas_from_yphi(self._yphis(arg))

    def _hadr_yphi_list_dim3(self, arg):
        
        return self._hadr_yphi_ndarray_dim3(np.asarray(arg))

    def _hadr_yphi_list_dim4(self, arg):
        
        return self._hadr_yphi_ndarray_dim4(np.asarray(arg))

    def _hadr_yphi_pseudojet(self, arg):
        
        constituents = arg.constituents()
        pts = np.asarray([c.pt() for c in constituents])
        thetas = np.asarray([[c1.delta_R(c2) for c2 in constituents] for c1 in constituents])**self.beta
        
        return pts, thetas

class EEMeasure:

    def _ee_parse_input(self, arg):
        
        # support arg as numpy.ndarray
        if isinstance(arg, np.ndarray):
            if arg.shape[1] == 4:
                self._meas_func = self._ee_ndarray
            else:
                raise IndexError('second dimension of arg must be size 4')

        # support arg as list (of lists)
        elif isinstance(arg, list):
            if len(arg[0]) == 4:
                self._meas_func = self._ee_list
            else:
                raise IndexError('second dimension of arg must be size 4')

        # support arg as fastjet pseudojet
        elif hasattr(arg, 'constituents'):
            self._meas_func = self._ee_pseudojet

        # raise error if not one of these types
        else:
            raise TypeError('arg is not one of numpy.ndarray, list, or fastjet.PseudoJet')

        # set flag to indicate we have (once) determined a type
        self._lacks_type = True

    def _ee_ndarray(self, arg):
        Es = arg[:,0]
        p4hats = arg/Es[:,np.newaxis]
        X = p4hats[:,np.newaxis]*p4hats[np.newaxis,:]
        return Es, (2*(X[0] - X[1] - X[2] - X[3]))**self._half_beta

    def _ee_list(self, arg):
        return self._ee_ndarray(np.asarray(arg))

    def _ee_pseudojet(self, arg):
        constituents = arg.constituents()
        p4s = np.asarray([[c.e(), c.px(), c.py(), c.pz()] for c in constituents])
        return self._ee_ndarray(p4s)

class Measure(HadronicMeasure, EEMeasure):
    
    """Class for dealing with any kind of measure."""

    supported_measures = ['ee', 'hadr_yphi']

    def __init__(self, measure, beta, normed, check_type):
        """ 
        Parameters
        ----------
        measure : 
        normed : 
        """

        assert measure in self.supported_measures, '{} not supported as measure'.format(measure)
        assert beta > 0, 'beta must be greater than zero'

        self.measure = measure
        self.beta = beta
        self._half_beta = self.beta/2
        self.normed = normed
        self.check_type = check_type

        self._lacks_type = True

        if self.measure == 'hadr_yphi':
            self._parse_inputs = self.hadr_yphi_parse_input
        elif self.measure == 'ee':
            self._parse_inputs = self._ee_parse_input

    def zs_thetas(self, arg):

        # check type only if needed
        if self._lacks_type or self.check_type:
            self._parse_inputs(arg)

        # get zs and thetas 
        zs, thetas = self._meas_func(arg)

        # normalize zs
        if self.normed:
            zs /= np.sum(zs)

        return zs, thetas
