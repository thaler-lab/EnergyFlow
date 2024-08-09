"""## Random Events

Functions to generate random sets of four-vectors. Includes an implementation
of the [RAMBO](https://doi.org/10.1016/0010-4655(86)90119-0) algorithm for
sampling uniform M-body massless phase space. Also includes other functions for
various random, non-center of momentum, and non-uniform sampling.
"""

#  ________      ________ _   _ _______         _    _ _______ _____ _       _____
# |  ____\ \    / /  ____| \ | |__   __|       | |  | |__   __|_   _| |     / ____|
# | |__   \ \  / /| |__  |  \| |  | |          | |  | |  | |    | | | |    | (___
# |  __|   \ \/ / |  __| | . ` |  | |          | |  | |  | |    | | | |     \___ \
# | |____   \  /  | |____| |\  |  | |   ______ | |__| |  | |   _| |_| |____ ____) |
# |______|   \/   |______|_| \_|  |_|  |______| \____/   |_|  |_____|______|_____/

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2022 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

import os

import numpy as np

__all__ = [
    'gen_random_events',
    'gen_random_events_mcom',
    'gen_massless_phase_space'
]

def gen_random_events(nevents, nparticles, dim=4, mass=0., rng=None, seed=None):
    r"""Generate random events with a given number of particles in a given
    spacetime dimension. The spatial components of the momenta are
    distributed uniformly in $[-1,+1]$. These events are not guaranteed to
    uniformly sample phase space.

    **Arguments**

    - **nevents** : _int_
        - Number of events to generate.
    - **nparticles** : _int_
        - Number of particles in each event.
    - **dim** : _int_
        - Number of spacetime dimensions.
    - **mass** : _float_ or `'random'`
        - Mass of the particles to generate. Can be set to `'random'` to obtain
        a different random mass for each particle.
    - **rng** : {None, Generator}, optional
        - A numpy.random.Generator to use.
    - **seed** : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        - A seed to initialize the numpy.random.BitGenerator used for the
        internal numpy.random.Generator if `rng=None`.

    **Returns**

    - _numpy.ndarray_
        - An `(nevents,nparticles,dim)` array of events. The particles
        are specified as `[E,p1,p2,...]`. If `nevents` is 1 then that axis is
        dropped.
    """

    if rng is None:
        rng = np.random.default_rng(seed=seed)

    spatial_ps = 2*rng.random((nevents, nparticles, dim-1)) - 1

    if isinstance(mass, str) and mass == 'random':
        mass = rng.random((nevents, nparticles))

    energies = np.sqrt(mass**2 + np.sum(spatial_ps**2, axis=-1))
    events = np.concatenate((energies[:,:,np.newaxis], spatial_ps), axis=-1)

    return events if nevents > 1 else events[0]

def gen_random_events_mcom(nevents, nparticles, dim=4, rng=None, seed=None):
    r"""Generate random events with a given number of massless particles in a
    given spacetime dimension. The total momentum are made to sum
    to zero. These events are not guaranteed to uniformly sample phase space.

    **Arguments**

    - **nevents** : _int_
        - Number of events to generate.
    - **nparticles** : _int_
        - Number of particles in each event.
    - **dim** : _int_
        - Number of spacetime dimensions.
    - **rng** : {None, Generator}, optional
        - A numpy.random.Generator to use.
    - **seed** : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        - A seed to initialize the numpy.random.BitGenerator used for the
        internal numpy.random.Generator if `rng=None`.

    **Returns**

    - _numpy.ndarray_
        - An `(nevents,nparticles,dim)` array of events. The particles
        are specified as `[E,p1,p2,...]`.
    """
    if rng is None:
        rng = np.random.default_rng(seed=seed)

    events_1_sp = 2*rng.random((nevents, int(np.ceil(nparticles/2)-1), dim-1)) - 1
    events_2_sp = 2*rng.random((nevents, int(np.floor(nparticles/2)-1), dim-1)) - 1

    events_1_sp_com = np.concatenate((events_1_sp, -np.sum(events_1_sp, axis=1)[:,np.newaxis]), axis=1)
    events_2_sp_com = np.concatenate((events_2_sp, -np.sum(events_2_sp, axis=1)[:,np.newaxis]), axis=1)

    events_1_tup = (np.sqrt(np.abs(np.sum(events_1_sp_com**2, axis=-1)))[:,:,np.newaxis], events_1_sp_com)
    events_2_tup = (np.sqrt(np.abs(np.sum(events_2_sp_com**2, axis=-1)))[:,:,np.newaxis], events_2_sp_com)
    events_1 = np.concatenate(events_1_tup, axis=-1)
    events_2 = np.concatenate(events_2_tup, axis=-1)

    events_1_tot, events_2_tot = np.sum(events_1, axis=1), np.sum(events_2, axis=1)
    factors = events_1_tot[:,0]/events_2_tot[:,0]

    events = np.concatenate((events_1, -events_2*factors[:,np.newaxis,np.newaxis]), axis=1)
    events[...,0] = np.abs(events[...,0])

    return events

def gen_massless_phase_space(nevents, nparticles, energy=1., rng=None, seed=None):
    r"""Implementation of the [RAMBO](https://doi.org/10.1016/0010-4655(86)90119-0)
    algorithm for uniformly sampling massless M-body phase space for any center
    of mass energy.

    **Arguments**

    - **nevents** : _int_
        - Number of events to generate.
    - **nparticles** : _int_
        - Number of particles in each event.
    - **energy** : _float_
        - Total center of mass energy of each event.
    - **rng** : {None, Generator}, optional
        - A numpy.random.Generator to use.
    - **seed** : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        - A seed to initialize the numpy.random.BitGenerator used for the
        internal numpy.random.Generator if `rng=None`.

    **Returns**

    - _numpy.ndarray_
        - An `(nevents,nparticles,4)` array of events. The particles
        are specified as `[E,p_x,p_y,p_z]`. If `nevents` is 1 then that axis is
        dropped.
    """

    # qs: to be massless four-momenta uniformly sampled in angle
    qs = np.empty((nevents, nparticles, 4))

    # ps: to be the uniformly sampled n-body four-momenta s.t. sum_i p_i = (energy, 0)
    ps = np.empty((nevents, nparticles, 4))

    # randomly sample from the qs as stated in the RAMBO paper
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    r = rng.random((4, nevents, nparticles))

    c = 2*r[0] - 1
    phi = 2*np.pi*r[1]

    qs[:,:,0] = -np.log(r[2]*r[3])
    tmp = qs[:,:,0]*np.sqrt(1 - c**2)
    qs[:,:,1] = tmp*np.cos(phi)
    qs[:,:,2] = tmp*np.sin(phi)
    qs[:,:,3] = qs[:,:,0]*c

    # define the following quantities to rescale the qs to the ps
    Qs = np.sum(qs, axis=1)
    Ms = np.sqrt(np.abs(Qs[:,0]**2 - Qs[:,1]**2 - Qs[:,2]**2 - Qs[:,3]**2))
    gammas = Qs[:,0]/Ms
    As = 1/(1 + gammas)[:,np.newaxis,np.newaxis]
    bs = (-Qs[:,1:]/Ms[:,np.newaxis])[:,np.newaxis]
    xs = (energy/Ms)[:,np.newaxis]

    bdotq = np.sum(bs*qs[:,:,1:], axis=-1)

    ps[:,:,0] = xs*(gammas[:,np.newaxis]*qs[:,:,0] + bdotq)
    ps[:,:,1:] = xs[:,np.newaxis]*(qs[:,:,1:] +
                                              bs*qs[:,:,0,np.newaxis] +
                                              As*bdotq[:,:,np.newaxis]*bs)
    return ps if nevents > 1 else ps[0]
