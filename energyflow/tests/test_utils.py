from __future__ import absolute_import

import numpy as np
import pytest

import energyflow as ef

def epsilon_diff(X, Y, epsilon=10**-14):
    return np.all(np.abs(X - Y) < epsilon)

def epsilon_percent(X, Y, epsilon=10**-14):
    return np.all(2*np.abs(X - Y)/(np.abs(X) + np.abs(Y)) < epsilon)

@pytest.mark.parametrize('nparticles', [10,100])
@pytest.mark.parametrize('nevents', [20,200])
def test_gen_massless_phase_space(nevents, nparticles):
    events = ef.gen_massless_phase_space(nevents, nparticles)
    assert events.shape == (nevents, nparticles, 4)
    assert epsilon_diff(ef.mass2(events), 0)

@pytest.mark.parametrize('nparticles', [10,100])
@pytest.mark.parametrize('nevents', [20,200])
@pytest.mark.parametrize('dim', [3,4,8])
@pytest.mark.parametrize('mass', [0,1.5])
def test_gen_random_events(nevents, nparticles, dim, mass):
    events = ef.gen_random_events(nevents, nparticles, dim=dim, mass=mass)
    assert events.shape == (nevents, nparticles, dim)
    assert epsilon_diff(ef.mass2(events), mass**2)

@pytest.mark.parametrize('nparticles', [10,100])
@pytest.mark.parametrize('nevents', [20,200])
@pytest.mark.parametrize('dim', [3,4,8])
def test_gen_random_events_massless_com(nevents, nparticles, dim):
    events = ef.gen_random_events_massless_com(nevents, nparticles, dim=dim)
    assert events.shape == (nevents, nparticles, dim)
    assert epsilon_diff(ef.mass2(events)/dim, 0, 10**-13)
    assert epsilon_diff(np.sum(events, axis=1), 0, 10**-13)