# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division

import numpy as np
import pytest

import energyflow as ef

from test_utils import epsilon_percent, epsilon_diff

# test measures
ptyphis = [(10*np.random.rand(25), 6*np.random.rand(25)-3, 2*np.pi*np.random.rand(25)) for i in range(3)]

@pytest.mark.measure
@pytest.mark.parametrize('kappa_normed_behavior', ['new', 'orig'])
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, .5, 1])
@pytest.mark.parametrize('beta', [.2, 1, 2])
@pytest.mark.parametrize('pts,ys,phis', ptyphis)
def test_measure_hadr_ptyphi(pts, ys, phis, beta, kappa, normed, kappa_normed_behavior):
    M = len(pts)
    
    # compute using the energyflow package
    hmeas = ef.Measure('hadr', beta, kappa, normed, 'ptyphim', True, kappa_normed_behavior)
    hzs, hthetas = hmeas.evaluate(np.vstack((pts,ys,phis)).T)
    
    # compute naively
    norm = 1 if not normed else (np.sum(pts**kappa) if kappa_normed_behavior == 'orig' else np.sum(pts)**kappa)
    zs = (pts**kappa)/norm
    thetas = np.asarray([[(ys[i]-ys[j])**2 + min(abs(phis[i]-phis[j]), 2*np.pi-abs(phis[i]-phis[j]))**2
                          for i in range(M)] for j in range(M)])**(beta/2)
    
    assert epsilon_diff(hzs, zs, 10**-13)
    assert epsilon_diff(hthetas, thetas, 10**-13)

@pytest.mark.measure
@pytest.mark.parametrize('kappa_normed_behavior', ['new', 'orig'])
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, .5, 1])
@pytest.mark.parametrize('beta', [.2, 1, 2])
@pytest.mark.parametrize('event', ef.gen_random_events(3, 15))
def test_measure_hadr_p4s(event, beta, kappa, normed, kappa_normed_behavior):
    M = len(event)
    pTs = np.sqrt(event[:,1]**2 + event[:,2]**2)
    ys = 0.5*np.log((event[:,0] + event[:,3])/(event[:,0] - event[:,3]))
    phis = np.arctan2(event[:,2], event[:,1])
    
    # compute using the energyflow package
    hmeas = ef.Measure('hadr', beta, kappa, normed, 'epxpypz', True, kappa_normed_behavior)
    hzs, hthetas = hmeas.evaluate(event)
    
    # compute naively
    norm = 1 if not normed else (np.sum(pTs**kappa) if kappa_normed_behavior == 'orig' else np.sum(pTs)**kappa)
    zs = (pTs**kappa)/norm
    thetas = np.asarray([[(ys[i]-ys[j])**2 + min(abs(phis[i]-phis[j]), 2*np.pi-abs(phis[i]-phis[j]))**2
                          for i in range(M)] for j in range(M)])**(beta/2)
    
    assert epsilon_diff(hzs, zs, 10**-12)
    assert epsilon_diff(hthetas, thetas, 10**-12)

@pytest.mark.measure
@pytest.mark.parametrize('kappa_normed_behavior', ['new', 'orig'])
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, .5, 1, 'pf'])
@pytest.mark.parametrize('beta,theta_eps', [(1, 5.5), (2,10), (np.pi,8)])
@pytest.mark.parametrize('event', [np.vstack(event).T for event in ptyphis])
def test_measure_hadrdot_ptyphi(event, beta, theta_eps, kappa, normed, kappa_normed_behavior):
    if normed and kappa == 'pf':
        pytest.skip()
    
    pTs = event[:,0]
    ps  = np.asarray([pT*np.asarray([np.cosh(y),np.cos(phi),np.sin(phi),np.sinh(y)]) for (pT,y,phi) in event])
   
    # compute using the energyflow package
    hmeas = ef.Measure('hadrdot', beta, kappa, normed, 'ptyphim', True, kappa_normed_behavior)
    hzs, hthetas = hmeas.evaluate(event)
        
    # compute naively
    norm = 1 if not normed else (np.sum(pTs**kappa) if kappa_normed_behavior == 'orig' else np.sum(pTs)**kappa)
    zs = (pTs**kappa)/norm if kappa != 'pf' else np.ones(len(pTs))
    phats = np.asarray([p/(pT if kappa != 'pf' else 1) for p,pT in zip(ps,pTs)])
    thetas = np.asarray([[2*abs(phti[0]*phtj[0]-np.dot(phti[1:],phtj[1:])) for phti in phats] for phtj in phats])**(beta/2)

    assert epsilon_diff(hzs, zs, 10**-13)
    assert epsilon_diff(hthetas, thetas, 10**-theta_eps)

@pytest.mark.measure
@pytest.mark.parametrize('kappa_normed_behavior', ['new', 'orig'])
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, .5, 1, 'pf'])
@pytest.mark.parametrize('beta,theta_eps', [(1, 5.5), (2,12), (np.pi, 10)])
@pytest.mark.parametrize('event', [2*np.random.rand(15,4)-1 for i in range(3)])
def test_measure_hadrdot_p4s(event, beta, theta_eps, kappa, normed, kappa_normed_behavior):
    if normed and kappa == 'pf':
        pytest.skip()
    
    pTs = np.sqrt(event[:,1]**2 + event[:,2]**2)
    ps  = event
   
    # compute using the energyflow package
    hmeas = ef.Measure('hadrdot', beta, kappa, normed, 'epxpypz', True, kappa_normed_behavior)
    hzs, hthetas = hmeas.evaluate(event)
        
    # compute naively
    norm = 1 if not normed else (np.sum(pTs**kappa) if kappa_normed_behavior == 'orig' else np.sum(pTs)**kappa)
    zs = (pTs**kappa)/norm if kappa != 'pf' else np.ones(len(pTs))
    phats = np.asarray([p/(pT if kappa != 'pf' else 1) for p,pT in zip(ps,pTs)])
    thetas = np.asarray([[2*abs(phti[0]*phtj[0]-np.dot(phti[1:],phtj[1:])) for phti in phats] for phtj in phats])**(beta/2)

    assert epsilon_diff(hzs, zs, 10**-13)
    assert epsilon_diff(hthetas, thetas, 10**-theta_eps)

@pytest.mark.measure
@pytest.mark.parametrize('kappa_normed_behavior', ['new', 'orig'])
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, .5, 1, 'pf'])
@pytest.mark.parametrize('beta,theta_eps', [(1, 6), (2,11), (np.pi, 9)])
@pytest.mark.parametrize('event', [2*np.random.rand(15,dim) for dim in [4,6] for i in range(2)])
def test_measure_ee(event, beta, theta_eps, kappa, normed, kappa_normed_behavior):
    if kappa == 'pf' and normed:
        pytest.skip()

    Es = event[:,0]

    emeas = ef.Measure('ee', beta, kappa, normed, 'epxpypz', True, kappa_normed_behavior)
    ezs, ethetas = emeas.evaluate(event)

    # compute naively
    norm = 1 if not normed else (np.sum(Es**kappa) if kappa_normed_behavior == 'orig' else np.sum(Es)**kappa)
    zs = (Es**kappa)/norm if kappa != 'pf' else np.ones(len(Es))
    phats = np.asarray([p/(E if kappa != 'pf' else 1) for p,E in zip(event,Es)])
    thetas = np.asarray([[2*abs(phti[0]*phtj[0]-np.dot(phti[1:],phtj[1:])) for phti in phats] for phtj in phats])**(beta/2)

    assert epsilon_diff(ezs, zs, 10**-13)
    assert epsilon_percent(ethetas, thetas, 10**-theta_eps)

@pytest.mark.measure
@pytest.mark.parametrize('check_input', [True, False])
@pytest.mark.parametrize('event', ef.gen_random_events(2,15))
@pytest.mark.parametrize('measure', ['hadr', 'hadrdot', 'hadrefm', 'ee', 'eeefm'])
def test_measure_list_input(measure, event, check_input):
    meas = ef.Measure(measure, check_input=check_input)
    list_event = event.tolist()
    nd0, nd1 = meas.evaluate(event)
    try:
        list0, list1 = meas.evaluate(list_event)
    except:
        assert not check_input
    else:
        assert check_input
        assert epsilon_diff(nd0, list0, 10**-14)
        assert epsilon_diff(nd1, list1, 10**-14)
