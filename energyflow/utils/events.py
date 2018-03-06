from __future__ import absolute_import

import os

import numpy as np

from energyflow.utils.path import *

__all__ = [
    'gen_massless_phase_space',
    'gen_random_events',
    'gen_random_events_massless_com',
    'load_events',
    'load_big_event',
    'mass2'
]

# implementation of RAMBO algorithm
# citation: http://cds.cern.ch/record/164736/files/198601282.pdf
def gen_massless_phase_space(nevents, nparticles, energy=1):
    
    # qs: to be massless four-momenta uniformly sampled in angle
    qs = np.empty((nevents, nparticles, 4))
    
    # ps: to be the uniformly sampled n-body four-momenta s.t. sum_i p_i = (energy, 0)
    ps = np.empty((nevents, nparticles, 4))
    
    # randomly sample from the qs as stated in the RAMBO paper
    r = np.random.random((4, nevents, nparticles))

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
    if nevents == 1:
        return ps[0]
    return ps

# make random events with a given number of particles
# the spacetime dimension and mass of the particles can be controlled
# the spatial vectors are drawn randomly in [-1,1]^(dim-1)
def gen_random_events(nevents, nparticles, dim=4, mass=0):
    spatial_ps = 2*np.random.rand(nevents, nparticles, dim-1) - 1
    energies = np.sqrt(mass**2 + np.sum(spatial_ps**2, axis=-1))
    events = np.concatenate((energies[:,:,np.newaxis], spatial_ps), axis=-1) 
    if nevents == 1:
        return events[0]
    return events

# generate random massless events in the center of momentum frame
def gen_random_events_massless_com(nevents, nparticles, dim=4):
    events_1_sp = 2*np.random.rand(nevents, int(np.ceil(nparticles/2)-1), dim-1) - 1
    events_2_sp = 2*np.random.rand(nevents, int(np.floor(nparticles/2)-1), dim-1) - 1
    
    events_1_sp_com = np.concatenate((events_1_sp, -np.sum(events_1_sp, axis=1)[:,np.newaxis]), axis=1)
    events_2_sp_com = np.concatenate((events_2_sp, -np.sum(events_2_sp, axis=1)[:,np.newaxis]), axis=1)

    events_1_tup = (np.sqrt(np.sum(events_1_sp_com**2, axis=-1))[:,:,np.newaxis], events_1_sp_com)
    events_2_tup = (np.sqrt(np.sum(events_2_sp_com**2, axis=-1))[:,:,np.newaxis], events_2_sp_com)
    events_1 = np.concatenate(events_1_tup, axis=-1)
    events_2 = np.concatenate(events_2_tup, axis=-1)

    events_1_tot, events_2_tot = np.sum(events_1, axis=1), np.sum(events_2, axis=1)
    factors = events_1_tot[:,0]/events_2_tot[:,0]

    return np.concatenate((events_1, -events_2*factors[:,np.newaxis,np.newaxis]), axis=1)

def load_events():
    return np.load(os.path.join(data_dir, 'events.npy'))

def load_big_event(num=None):
    return np.concatenate(load_events(), axis=0)[:num]

def mass2(events):
    return events[...,0]**2 - np.sum(events[...,1:]**2, axis=-1)
    