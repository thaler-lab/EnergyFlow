# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2022 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division

import numpy as np
import pytest

import energyflow as ef

def epsilon_diff(X, Y, epsilon=10**-14):
    return np.all(np.abs(X - Y) < epsilon)

def epsilon_percent(X, Y, epsilon=10**-14):
    return np.all(2*np.abs(X - Y)/(np.abs(X) + np.abs(Y)) < epsilon)

# test event utils

@pytest.mark.utils
@pytest.mark.gen_random
@pytest.mark.parametrize('nparticles', [10, 100])
@pytest.mark.parametrize('nevents', [20, 200])
def test_gen_massless_phase_space(nevents, nparticles):
    events = ef.gen_massless_phase_space(nevents, nparticles)
    assert events.shape == (nevents, nparticles, 4)
    assert epsilon_diff(ef.ms_from_ps(events)**2, 0, 10**-13)

@pytest.mark.utils
@pytest.mark.gen_random
@pytest.mark.parametrize('nparticles', [10, 100])
@pytest.mark.parametrize('nevents', [20, 200])
@pytest.mark.parametrize('dim', [3, 4, 8])
@pytest.mark.parametrize('mass', [0, 1.5, 'random', 'array'])
def test_gen_random_events(nevents, nparticles, dim, mass):
    rng = np.random.default_rng(seed=1234567890)
    if mass == 'array':
        mass = rng.random((nevents, nparticles))

    events = ef.gen_random_events(nevents, nparticles, dim=dim, mass=mass, rng=rng)

    assert events.shape == (nevents, nparticles, dim)

    if not (isinstance(mass, str) and mass == 'random'):
        assert epsilon_diff(ef.ms_from_ps(events)**2, mass**2, 10**-13)

@pytest.mark.utils
@pytest.mark.gen_random
@pytest.mark.parametrize('nparticles', [10, 100])
@pytest.mark.parametrize('nevents', [20, 200])
@pytest.mark.parametrize('dim', [3, 4, 8])
def test_gen_random_events_mcom(nevents, nparticles, dim):
    rng = np.random.default_rng(seed=1234567890)
    events = ef.gen_random_events_mcom(nevents, nparticles, dim=dim, rng=rng)

    assert events.shape == (nevents, nparticles, dim)
    assert epsilon_diff(ef.ms_from_ps(events)**2/dim, 0, 10**-12)
    assert epsilon_diff(np.sum(events, axis=1)[:,1:], 0, 10**-12)

# test particle utils

@pytest.mark.utils
@pytest.mark.parametrize('nparticles', [1, 20])
@pytest.mark.parametrize('nevents', [1, 10])
@pytest.mark.parametrize('method', ['ptyphims_from_p4s', 'pts_from_p4s',
					                'pt2s_from_p4s', 'ys_from_p4s', 'etas_from_p4s',
					                'phis_from_p4s', 'm2s_from_p4s', 'ms_from_p4s'])
def test_shapes_from_p4s(method, nevents, nparticles):
    rng = np.random.default_rng(seed=1234567890)
    events = ef.gen_random_events(nevents, nparticles, dim=4, mass='random', rng=rng).reshape(nevents, nparticles, 4)
    event, particle = events[0], events[0,0]

    func = getattr(ef, method)
    results = func(events)

    assert epsilon_diff(results[0], func(event))
    assert epsilon_diff(results[0,0], func(particle))

@pytest.mark.utils
@pytest.mark.parametrize('nparticles', [1, 20])
@pytest.mark.parametrize('nevents', [1, 10])
@pytest.mark.parametrize('method', ['p4s_from_ptyphims', 'p4s_from_ptyphipids',])
                                    #'sum_ptyphims', 'sum_ptyphipids'])
def test_shapes_from_ptyphis(method, nevents, nparticles):
    rng = np.random.default_rng(seed=1234567890)
    p4s = ef.gen_random_events(nevents, nparticles, dim=4, mass='random', rng=rng).reshape(nevents, nparticles, 4)
    ptyphims = ef.ptyphims_from_p4s(p4s)

    func = getattr(ef, method)

    if 'ms' in method:
        for end in [3,4]:
            results = func(ptyphims[...,:end])

            assert epsilon_diff(results[0], func(ptyphims[0,...,:end]))

    elif 'pids' in method:
        ptyphims[...,3] = (np.random.choice([-1., 1.], size=(nevents, nparticles)) *
                           np.random.choice(list(ef.utils.particle_utils.PARTICLE_MASSES.keys()),
                                            size=(nevents, nparticles)))
        results = func(ptyphims)

        assert epsilon_diff(results[0], func(ptyphims[0]))

@pytest.mark.utils
@pytest.mark.parametrize('nparticles', [1, 20])
@pytest.mark.parametrize('nevents', [1, 10])
def test_pts_from_p4s(nevents, nparticles):
    rng = np.random.default_rng(seed=1234567890)
    p4s = ef.gen_random_events(nevents, nparticles, dim=4, mass='random', rng=rng).reshape(nevents, nparticles, 4)

    pts = ef.pts_from_p4s(p4s)
    slow_pts = []
    for i in range(nevents):
        event_pts = []
        for j in range(nparticles):
            event_pts.append(np.sqrt(p4s[i,j,1]**2 + p4s[i,j,2]**2))
        slow_pts.append(event_pts)

    slow_pts = np.asarray(slow_pts)
    assert epsilon_diff(slow_pts, pts)

@pytest.mark.utils
@pytest.mark.parametrize('nparticles', [1, 20])
@pytest.mark.parametrize('nevents', [1, 10])
def test_pt2s_from_p4s(nevents, nparticles):
    rng = np.random.default_rng(seed=1234567890)
    p4s = ef.gen_random_events(nevents, nparticles, dim=4, mass='random', rng=rng).reshape(nevents, nparticles, 4)

    pt2s = ef.pt2s_from_p4s(p4s)
    slow_pt2s = []
    for i in range(nevents):
        event_pt2s = []
        for j in range(nparticles):
            event_pt2s.append(p4s[i,j,1]**2 + p4s[i,j,2]**2)
        slow_pt2s.append(event_pt2s)

    slow_pt2s = np.asarray(slow_pt2s)
    assert epsilon_diff(slow_pt2s, pt2s)

@pytest.mark.utils
@pytest.mark.parametrize('nparticles', [1, 20])
@pytest.mark.parametrize('nevents', [1, 10])
def test_ys_from_p4s(nevents, nparticles):
    rng = np.random.default_rng(seed=1234567890)
    p4s = ef.gen_random_events(nevents, nparticles, dim=4, mass='random', rng=rng).reshape(nevents, nparticles, 4)
    ys = 0.5*np.log((p4s[...,0] + p4s[...,3])/(p4s[...,0] - p4s[...,3]))

    assert epsilon_diff(ys, ef.ys_from_p4s(p4s))

@pytest.mark.utils
@pytest.mark.parametrize('nparticles', [1, 20])
@pytest.mark.parametrize('nevents', [1, 10])
def test_etas_from_p4s(nevents, nparticles):
    rng = np.random.default_rng(seed=1234567890)
    p4s = ef.gen_random_events(nevents, nparticles, dim=4, mass='random', rng=rng).reshape(nevents, nparticles, 4)

    p3tots = np.linalg.norm(p4s[...,1:], axis=-1)
    etas = 0.5*np.log((p3tots + p4s[...,3])/(p3tots - p4s[...,3]))

    assert epsilon_diff(etas, ef.etas_from_p4s(p4s), 1e-13)

@pytest.mark.utils
@pytest.mark.parametrize('phi_ref', [None, 'hardest', 'array'])
@pytest.mark.parametrize('nparticles', [1, 20])
@pytest.mark.parametrize('nevents', [1, 100])
def test_phis_from_p4s(nevents, nparticles, phi_ref):
    phis = 2*np.pi*np.random.rand(nevents, nparticles, 1)
    ys = 6*np.random.rand(nevents, nparticles, 1) - 3
    pts = 100*np.random.rand(nevents, nparticles, 1)
    ms = np.random.rand(nevents, nparticles, 1)

    p4s = ef.p4s_from_ptyphims(np.concatenate((pts, ys, phis, ms), axis=-1))

    if isinstance(phi_ref, str) and phi_ref == 'array':
        phi_ref = 2*np.pi*np.random.rand(nevents)

    new_phis = ef.phis_from_p4s(p4s, phi_ref=phi_ref)

    if phi_ref is None:
        assert epsilon_diff(new_phis, phis[...,0])

    else:
        if isinstance(phi_ref, str) and phi_ref == 'hardest':
            phi_ref = np.asarray([phis[i,np.argmax(pts[i,:,0]),0] for i in range(nevents)])

        assert np.all(np.abs(new_phis.T - phi_ref) <= np.pi)

@pytest.mark.utils
@pytest.mark.parametrize('nparticles', [1, 20])
@pytest.mark.parametrize('nevents', [1, 10])
def test_ms_from_p4s(nevents, nparticles):
    rng = np.random.default_rng(seed=1234567890)
    p4s = ef.gen_random_events(nevents, nparticles, dim=4, mass='random', rng=rng).reshape(nevents, nparticles, 4)

    ms = ef.ms_from_p4s(p4s)
    slow_ms = []
    for i in range(nevents):
        event_ms = []
        for j in range(nparticles):
            event_ms.append(np.sqrt(p4s[i,j,0]**2 - p4s[i,j,1]**2 - p4s[i,j,2]**2 - p4s[i,j,3]**2))
        slow_ms.append(event_ms)

    slow_ms = np.asarray(slow_ms)
    assert epsilon_diff(slow_ms, ms)

@pytest.mark.utils
@pytest.mark.parametrize('nparticles', [1, 20])
@pytest.mark.parametrize('nevents', [1, 10])
def test_m2s_from_p4s(nevents, nparticles):
    rng = np.random.default_rng(seed=1234567890)
    p4s = ef.gen_random_events(nevents, nparticles, dim=4, mass='random', rng=rng).reshape(nevents, nparticles, 4)

    m2s = ef.m2s_from_p4s(p4s)
    slow_m2s = []
    for i in range(nevents):
        event_m2s = []
        for j in range(nparticles):
            event_m2s.append(p4s[i,j,0]**2 - p4s[i,j,1]**2 - p4s[i,j,2]**2 - p4s[i,j,3]**2)
        slow_m2s.append(event_m2s)

    slow_m2s = np.asarray(slow_m2s)
    assert epsilon_diff(slow_m2s, m2s)

@pytest.mark.utils
@pytest.mark.yeta
@pytest.mark.parametrize('nparticles', [1, 20])
@pytest.mark.parametrize('nevents', [1, 10])
def test_etas_from_pts_ys_ms(nevents, nparticles):
    rng = np.random.default_rng(seed=1234567890)
    p4s = ef.gen_random_events(nevents, nparticles, dim=4, mass='random', rng=rng).reshape(nevents, nparticles, 4)
    ptyphims = ef.ptyphims_from_p4s(p4s)

    etas = ef.etas_from_p4s(p4s)
    etas_primes = ef.etas_from_pts_ys_ms(ptyphims[...,0], ptyphims[...,1], ptyphims[...,3])

    assert epsilon_diff(etas, etas_primes, 1e-12)

    # test cutoff
    pts, ms = 1000*np.random.rand(25), 10*np.random.rand(25)
    ys = np.random.choice([-1., 1.], size=25)*100*np.random.rand(25)

    for c1,c2 in zip(np.linspace(20, 100, 5), 20 + 80*np.random.rand(5)):
        etas_c1 = ef.etas_from_pts_ys_ms(pts, ys, ms, _cutoff=c1)
        etas_c2 = ef.etas_from_pts_ys_ms(pts, ys, ms, _cutoff=c2)

        assert epsilon_diff(etas_c1, etas_c2, 1e-12)

@pytest.mark.utils
@pytest.mark.yeta
@pytest.mark.parametrize('nparticles', [1, 20])
@pytest.mark.parametrize('nevents', [1, 10])
def test_ys_from_pts_etas_ms(nevents, nparticles):
    rng = np.random.default_rng(seed=1234567890)
    p4s = ef.gen_random_events(nevents, nparticles, dim=4, mass='random', rng=rng).reshape(nevents, nparticles, 4)

    ys = ef.ys_from_p4s(p4s)
    y_primes = ef.ys_from_pts_etas_ms(ef.pts_from_p4s(p4s), ef.etas_from_p4s(p4s), ef.ms_from_p4s(p4s))

    assert epsilon_diff(ys, y_primes, 1e-12)

    # test cutoff
    pts, ms = 1000*np.random.rand(25), 10*np.random.rand(25)
    etas = np.random.choice([-1., 1.], size=25)*100*np.random.rand(25)

    for c1,c2 in zip(np.linspace(20, 100, 5), 20 + 80*np.random.rand(5)):
        ys_c1 = ef.ys_from_pts_etas_ms(pts, etas, ms, _cutoff=c1)
        ys_c2 = ef.ys_from_pts_etas_ms(pts, etas, ms, _cutoff=c2)

        assert epsilon_diff(ys_c1, ys_c2, 1e-12)

@pytest.mark.utils
@pytest.mark.parametrize('nparticles', [1, 500])
@pytest.mark.parametrize('nevents', [1, 100])
def test_coordinate_transforms(nevents, nparticles):
    rng = np.random.default_rng(seed=1234567890)
    p4s = ef.gen_random_events(nevents, nparticles, dim=4, mass='random', rng=rng).reshape(nevents, nparticles, 4)
    ptyphims = ef.ptyphims_from_p4s(p4s)
    new_p4s = ef.p4s_from_ptyphims(ptyphims)

    assert epsilon_diff(p4s, new_p4s, 1e-11)

@pytest.mark.utils
@pytest.mark.parametrize('nparticles', [1, 500])
@pytest.mark.parametrize('nevents', [1, 100])
@pytest.mark.parametrize('phi_ref', np.linspace(0, 2*np.pi, 5))
def test_phifix(phi_ref, nevents, nparticles):
    phis = 2*np.pi*np.random.rand(nevents, nparticles)
    phi_list, phi_single = phis[0], phis[0,0]

    results = ef.phi_fix(phis, phi_ref)

    # test shapes
    assert epsilon_diff(results[0],   ef.phi_fix(phi_list, phi_ref))
    assert epsilon_diff(results[0,0], ef.phi_fix(phi_single, phi_ref))

    # test values
    assert np.all(np.abs(results - phi_ref) <= np.pi)

    # test vector of phis
    if nevents > 1:
        phi_ref = 2*np.pi*np.random.rand(nevents)

        results = ef.phi_fix(phis, phi_ref)

        assert np.all(np.abs(results - phi_ref[:,np.newaxis])) <= np.pi

@pytest.mark.utils
@pytest.mark.parametrize('nparticles', [1, 20])
@pytest.mark.parametrize('nevents', [1, 10])
@pytest.mark.parametrize('dim', [2, 4, 8])
def test_ms_from_ps(dim, nevents, nparticles):
    rng = np.random.default_rng(seed=1234567890)
    masses = rng.random((nevents, nparticles))
    events = ef.gen_random_events(nevents, nparticles, mass=masses, dim=dim, rng=rng)
    masses = masses.reshape(events.shape[:-1])

    results = ef.ms_from_ps(events)
    assert epsilon_diff(results, masses, 1e-10)

    if dim == 4:
        assert epsilon_diff(results, ef.ms_from_p4s(events), 1e-10)

@pytest.mark.utils
@pytest.mark.parametrize('scheme', ['escheme', 'ptscheme'])
@pytest.mark.parametrize('nparticles', [1, 20])
def test_sum_ptyphims(nparticles, scheme):
    rng = np.random.default_rng(seed=1234567890)
    p4s = ef.gen_random_events(10, nparticles, dim=4, mass='random', rng=rng)
    ptyphims = ef.ptyphims_from_p4s(p4s)

    if scheme == 'escheme':

        for ev_p4s,ev_ptyphims in zip(p4s, ptyphims):
            tot = ef.p4s_from_ptyphims(ef.sum_ptyphims(ev_ptyphims, scheme=scheme))
            tot_p4 = ev_p4s.sum(axis=0)

            assert epsilon_diff(tot, tot_p4, 10**-12)

    elif scheme == 'ptscheme':

        for ev_ptyphims in ptyphims:
            tot = ef.sum_ptyphims(ev_ptyphims, scheme=scheme)

            pt = ev_ptyphims[:,0].sum()
            y = np.sum(ev_ptyphims[:,0]*ev_ptyphims[:,1])/pt
            phi = np.sum(ev_ptyphims[:,0]*ev_ptyphims[:,2])/pt

            assert epsilon_diff(tot, np.array([pt,y,phi]), 10**-12)

@pytest.mark.utils
@pytest.mark.parametrize('nparticles', [1, 20])
@pytest.mark.parametrize('nevents', [1, 10])
def test_pids2ms(nevents, nparticles):
    pids = (np.random.choice([-1., 1.], size=(nevents, nparticles)) *
            np.random.choice(list(ef.utils.particle_utils.PARTICLE_MASSES.keys()),
                             size=(nevents, nparticles)))

    # test shapes
    results = ef.pids2ms(pids)
    assert epsilon_diff(results[0],   ef.pids2ms(pids[0]))
    assert epsilon_diff(results[0,0], ef.pids2ms(pids[0,0]))

    # electron
    assert ef.pids2ms(11) == 0.000511

    # photon
    assert ef.pids2ms(22) == 0.

    # pion
    assert ef.pids2ms(211) == .139570

@pytest.mark.utils
@pytest.mark.parametrize('nparticles', [1, 20])
@pytest.mark.parametrize('nevents', [1, 10])
def test_pids2chrgs(nevents, nparticles):
    pids = (np.random.choice([-1., 1.], size=(nevents, nparticles)) *
            np.random.choice(list(ef.utils.particle_utils.PARTICLE_MASSES.keys()),
                             size=(nevents, nparticles)))

    # test shapes
    results = ef.pids2chrgs(pids)
    assert epsilon_diff(results[0],   ef.pids2chrgs(pids[0]))
    assert epsilon_diff(results[0,0], ef.pids2chrgs(pids[0,0]))

    # electron/positron
    assert ef.pids2chrgs(11) == -1.
    assert ef.pids2chrgs(-11) == 1.

    # photon
    assert ef.pids2chrgs(22) == 0.
    assert ef.pids2chrgs(-22) == 0.

    # pion
    assert ef.pids2chrgs(211) == 1.
    assert ef.pids2chrgs(-211) == -1.

# test graph utils

@pytest.mark.utils
def test_get_graph_components():
    efpset = ef.EFPSet()
    ps = np.array([len(ef.utils.get_components(graph)) for graph in efpset.graphs()])

    # note that the empty graph is recorded as having 1 connected component by EFPSet
    assert np.all(ps[1:] == efpset.specs[1:,-2])
