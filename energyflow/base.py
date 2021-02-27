"""Base classes for EnergyFlow."""

#  ____           _____ ______
# |  _ \   /\    / ____|  ____|
# | |_) | /  \  | (___ | |__
# |  _ < / /\ \  \___ \|  __| 
# | |_) / ____ \ ____) | |____
# |____/_/    \_\_____/|______|

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division

from abc import ABCMeta, abstractmethod, abstractproperty
import multiprocessing
import sys
import warnings

import numpy as np
import six

from energyflow.measure import Measure, MEASURE_KWARGS
from energyflow.utils import create_pool, kwargs_check

###############################################################################
# EFBase
###############################################################################

class EFBase(six.with_metaclass(ABCMeta, object)):

    """A base class for EnergyFlow objects that holds a `Measure`."""

    def __init__(self, kwargs):

        kwargs_check('EFBase', kwargs, allowed=MEASURE_KWARGS)
        self._measure = Measure(kwargs.pop('measure'), **kwargs)

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    @property
    def has_measure(self):
        return hasattr(self, '_measure') and isinstance(self._measure, Measure)

    @property
    def measure(self):
        return self._measure.measure if self.has_measure else None

    @property
    def beta(self):
        return self._measure.beta if self.has_measure else None

    @property
    def kappa(self):
        return self._measure.kappa if self.has_measure else None

    @property
    def normed(self):
        return self._measure.normed if self.has_measure else None

    @property
    def coords(self):
        return self._measure.coords if self.has_measure else None

    @property
    def check_input(self):
        return self._measure.check_input if self.has_measure else None

    @property
    def kappa_normed_behavior(self):
        return self._measure.kappa_normed_behavior if self.has_measure else None

    @property
    def subslicing(self):
        return self._measure.subslicing if self.has_measure else None

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    def _batch_compute_func(self, event):
        return self.compute(event)

    def batch_compute(self, events, n_jobs=None):
        """Computes the value of the observable on several events.

        **Arguments**

        - **events** : array_like or `fastjet.PseudoJet`
            - The events as an array of arrays of particles in coordinates
            matching those anticipated by `coords`.
        - **n_jobs** : _int_ or `None`
            - The number of worker processes to use. A value of `None` will
            use as many processes as there are CPUs on the machine.

        **Returns**

        - _1-d numpy.ndarray_
            - A vector of the observable values for each event.
        """

        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count() or 1
        self.n_jobs = n_jobs

        # don't bother setting up a Pool
        if self.n_jobs == 1:
            return np.asarray(list(map(self._batch_compute_func, events)))

        # setup processor pool
        chunksize = min(max(len(events)//self.n_jobs, 1), 10000)
        with create_pool(self.n_jobs) as pool:
            results = np.asarray(list(pool.map(self._batch_compute_func, events, chunksize)))

        return results

###############################################################################
# EFPBase
###############################################################################

class EFPBase(EFBase):

    """Base class for the `EFP` and `EFPSet` classes."""

    def __init__(self, kwargs):

        # initialize base class if measure needed
        if not kwargs.pop('no_measure', False):

            # set default measure for EFPs
            kwargs.setdefault('measure', 'hadr')
            super(EFPBase, self).__init__(kwargs)

            self.use_efms = 'efm' in self.measure

    def get_zs_thetas_dict(self, event, zs, thetas):
        if event is not None:
            zs, thetas = self._measure.evaluate(event)
        elif zs is None or thetas is None:
            raise TypeError('If event is None then zs and thetas cannot also be None')

        return zs, {w: thetas**w for w in self.weight_set}

    def compute_efms(self, event, zs, nhats):
        if event is not None:
            zs, nhats = self._measure.evaluate(event)
        elif zs is None or nhats is None:
            raise TypeError('If event is None then zs and thetas cannot also be None')

        return self.efmset.compute(zs=zs, nhats=nhats)

    @abstractproperty
    def weight_set(self):
        pass

    @abstractproperty
    def efmset(self):
        pass

    def _batch_compute_func(self, event):
        return self.compute(event, batch_call=True)

###############################################################################
# EFMBase
###############################################################################

class EFMBase(EFBase):

    """Base class for the `EFM` and `EFMSet` classes."""

    def __init__(self, kwargs):

        # verify we're using an efm measure
        assert 'efm' in kwargs.setdefault('measure', 'hadrefm'), 'Must use an efm measure.'

        # initialize base class if measure needed
        if not kwargs.pop('no_measure', False):
            super(EFMBase, self).__init__(kwargs)
            self._measure.beta = None

    @abstractmethod
    def compute(self, event=None, zs=None, nhats=None):
        if event is not None:
            return self._measure.evaluate(event)
        elif zs is None or nhats is None:
            raise ValueError('If event is None then zs and nhats cannot be None.')
        return zs, nhats

###############################################################################
# SingleEnergyCorrelatorBase
###############################################################################

class SingleEnergyCorrelatorBase(EFBase):

    """Base class for observables such as D2, C2, and C3."""

    def __init__(self, graphs, measure, beta, strassen, kwargs):

        kwargs.setdefault('measure', measure)
        kwargs.setdefault('beta', beta)
        super(SingleEnergyCorrelatorBase, self).__init__(kwargs)

        # use strassen if requested and it's possible
        self.strassen = strassen
        if self.strassen and ('efm' in self.measure or self.measure == 'hadr'):
            raise ValueError("strassen cannot be True when using 'hadr' or an EFM measure.")

        # include dot as the last graph if not normed
        if not self.normed:
            graphs.append([])

        # prepare EFPSet
        if not self.strassen:
            self._efpset = EFPSet(*graphs, measure=self.measure, beta=self.beta, coords=self.coords)

    @abstractmethod
    def _strassen_compute(self, event, zs, thetas):
        """Abstract method that evaluate the measure on the inputs."""

        if event is not None:
            zs, thetas = self._measure.evaluate(event)
        elif zs is None or thetas is None:
            raise TypeError('If event is None then zs and thetas cannot also be None.')

        return zs, thetas

    @abstractmethod
    def _efp_compute(self, event, zs, thetas, nhats):
        """Abstract method that evaluates the EFPSet on the inputs."""

        return self.efpset.compute(event, zs, thetas, nhats)

    def compute(self, event=None, zs=None, thetas=None, nhats=None):
        """Computes the value of the observable on a single event. Note that
        the observable object is also callable, in which case this method is
        invoked.

        **Arguments**

        - **event** : 2-d array_like or `fastjet.PseudoJet`
            - The event as an array of particles in the coordinates specified
            by `coords`.
        - **zs** : 1-d array_like
            - If present, `thetas` must also be present, and `zs` is used in place 
            of the energies of an event.
        - **thetas** : 2-d array_like
            - If present, `zs` must also be present, and `thetas` is used in place 
            of the pairwise angles of an event.
        - **nhats** : 2-d array like
            - If present, `zs` must also be present, and `nhats` is used in place
            of the scaled particle momenta. Only applicable when EFMs are being
            used.

        **Returns**

        - _float_
            - The observable value.
        """


        if self.strassen:
            return self._strassen_compute(event, zs, thetas)
        else:
            return self._efp_compute(event, zs, thetas, nhats)

    @property
    def efpset(self):
        """`EFPSet` held by the object to compute fundamental EFP values."""

        return None if self.strassen else self._efpset

# put EFPSet import here do it succeeds
from energyflow.efp import EFPSet
