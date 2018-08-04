from __future__ import absolute_import, division

import numpy as np
import pytest

import energyflow as ef

def epsilon_diff(X, Y, epsilon=10**-14):
    return np.all(np.abs(X - Y) < epsilon)

def epsilon_percent(X, Y, epsilon=10**-14):
    return np.all(2*np.abs(X - Y)/(np.abs(X) + np.abs(Y)) < epsilon)

# test fake event generation
@pytest.mark.parametrize('nparticles', [10,100])
@pytest.mark.parametrize('nevents', [20,200])
def test_gen_massless_phase_space(nevents, nparticles):
    events = ef.gen_massless_phase_space(nevents, nparticles)
    assert events.shape == (nevents, nparticles, 4)
    assert epsilon_diff(ef.ms_from_p4s(events)**2, 0, 10**-13)

@pytest.mark.parametrize('nparticles', [10,100])
@pytest.mark.parametrize('nevents', [20,200])
@pytest.mark.parametrize('dim', [3,4,8])
@pytest.mark.parametrize('mass', [0,1.5])
def test_gen_random_events(nevents, nparticles, dim, mass):
    events = ef.gen_random_events(nevents, nparticles, dim=dim, mass=mass)
    assert events.shape == (nevents, nparticles, dim)
    assert epsilon_diff(ef.ms_from_p4s(events)**2, mass**2, 10**-13)

@pytest.mark.parametrize('nparticles', [10,100])
@pytest.mark.parametrize('nevents', [20,200])
@pytest.mark.parametrize('dim', [3,4,8])
def test_gen_random_events_massless_com(nevents, nparticles, dim):
    events = ef.gen_random_events_massless_com(nevents, nparticles, dim=dim)
    assert events.shape == (nevents, nparticles, dim)
    assert epsilon_diff(ef.ms_from_p4s(events)**2/dim, 0, 10**-12)
    assert epsilon_diff(np.sum(events, axis=1), 0, 10**-12)

# test measures
ptyphis = [(10*np.random.rand(25), 6*np.random.rand(25)-3, 2*np.pi*np.random.rand(25)) for i in range(3)]

@pytest.mark.measure
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, .5, 1])
@pytest.mark.parametrize('beta', [.2, 1, 2])
@pytest.mark.parametrize('pts,ys,phis', ptyphis)
def test_measure_hadr_ptyphi(pts, ys, phis, beta, kappa, normed):
    M = len(pts)
    
    # compute using the energyflow package
    hmeas = ef.Measure('hadr', beta, kappa, normed, 'ptyphim', True)
    hzs, hthetas = hmeas.evaluate(np.vstack((pts,ys,phis)).T)
    
    # compute naively
    norm = 1 if not normed else np.sum(pts**kappa)
    zs = (pts**kappa)/norm
    thetas = np.asarray([[(ys[i]-ys[j])**2 + min(abs(phis[i]-phis[j]), 2*np.pi-abs(phis[i]-phis[j]))**2
                          for i in range(M)] for j in range(M)])**(beta/2)
    
    assert epsilon_diff(hzs, zs, 10**-13)
    assert epsilon_diff(hthetas, thetas, 10**-13)

@pytest.mark.measure
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, .5, 1])
@pytest.mark.parametrize('beta', [.2, 1, 2])
@pytest.mark.parametrize('event', ef.gen_random_events(3, 15))
def test_measure_hadr_p4s(event, beta, kappa, normed):
    M = len(event)
    pTs = np.sqrt(event[:,1]**2 + event[:,2]**2)
    ys = 0.5*np.log((event[:,0] + event[:,3])/(event[:,0] - event[:,3]))
    phis = np.arctan2(event[:,2], event[:,1])
    
    # compute using the energyflow package
    hmeas = ef.Measure('hadr', beta, kappa, normed, 'epxpypz', True)
    hzs, hthetas = hmeas.evaluate(event)
    
    # compute naively
    norm = 1 if not normed else np.sum(pTs**kappa)
    zs = (pTs**kappa)/norm
    thetas = np.asarray([[(ys[i]-ys[j])**2 + min(abs(phis[i]-phis[j]), 2*np.pi-abs(phis[i]-phis[j]))**2
                          for i in range(M)] for j in range(M)])**(beta/2)
    
    assert epsilon_diff(hzs, zs, 10**-13)
    assert epsilon_diff(hthetas, thetas, 10**-13)

@pytest.mark.measure
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, .5, 1, 'pf'])
@pytest.mark.parametrize('beta,theta_eps', [(1, 5.5), (2,10), (np.pi,8)])
@pytest.mark.parametrize('event', [np.vstack(event).T for event in ptyphis])
def test_measure_hadrdot_ptyphi(event, beta, theta_eps, kappa, normed):
    if normed and kappa == 'pf':
        pytest.skip()
    
    pTs = event[:,0]
    ps  = np.asarray([pT*np.asarray([np.cosh(y),np.cos(phi),np.sin(phi),np.sinh(y)]) for (pT,y,phi) in event])
   
    # compute using the energyflow package
    hmeas = ef.Measure('hadrdot', beta, kappa, normed, 'ptyphim', True)
    hzs, hthetas = hmeas.evaluate(event)
        
    # compute naively
    norm = 1 if not normed else np.sum(pTs**kappa)
    zs = (pTs**kappa)/norm if kappa != 'pf' else np.ones(len(pTs))
    
    phats = np.asarray([p/(pT if kappa != 'pf' else 1) for p,pT in zip(ps,pTs)])
    thetas = np.asarray([[2*abs(phti[0]*phtj[0]-np.dot(phti[1:],phtj[1:])) for phti in phats] for phtj in phats])**(beta/2)

    assert epsilon_diff(hzs, zs, 10**-13)
    assert epsilon_diff(hthetas, thetas, 10**-theta_eps)

@pytest.mark.measure
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, .5, 1, 'pf'])
@pytest.mark.parametrize('beta,theta_eps', [(1, 5.5), (2,12), (np.pi, 11)])
@pytest.mark.parametrize('event', [2*np.random.rand(15,4)-1 for i in range(3)])
def test_measure_hadrdot_p4s(event, beta, theta_eps, kappa, normed):
    if normed and kappa == 'pf':
        pytest.skip()
    
    pTs = np.sqrt(event[:,1]**2 + event[:,2]**2)
    ps  = event
   
    # compute using the energyflow package
    hmeas = ef.Measure('hadrdot', beta, kappa, normed, 'epxpypz', True)
    hzs, hthetas = hmeas.evaluate(event)
        
    # compute naively
    norm = 1 if not normed else np.sum(pTs**kappa)
    zs = (pTs**kappa)/norm if kappa != 'pf' else np.ones(len(pTs))
    
    phats = np.asarray([p/(pT if kappa != 'pf' else 1) for p,pT in zip(ps,pTs)])
    thetas = np.asarray([[2*abs(phti[0]*phtj[0]-np.dot(phti[1:],phtj[1:])) for phti in phats] for phtj in phats])**(beta/2)

    assert epsilon_diff(hzs, zs, 10**-13)
    assert epsilon_diff(hthetas, thetas, 10**-theta_eps)

@pytest.mark.measure
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, .5, 1, 'pf'])
@pytest.mark.parametrize('beta,theta_eps', [(1, 6), (2,12), (np.pi, 11)])
@pytest.mark.parametrize('event', [2*np.random.rand(15,dim) for dim in [4,6] for i in range(2)])
def test_measure_ee(event, beta, theta_eps, kappa, normed):
    if kappa == 'pf' and normed:
        pytest.skip()

    Es = event[:,0]

    emeas = ef.Measure('ee', beta, kappa, normed, 'epxpypz', True)
    ezs, ethetas = emeas.evaluate(event)

    # compute naively
    if kappa == 'pf':
        zs = np.ones(len(Es))
        phats = event
    else:
        if normed:
            zs = Es**kappa/np.sum(Es**kappa)
        else:
            zs = Es**kappa
        phats = event/Es[:,np.newaxis]

    thetas = np.asarray([[2*abs(phti[0]*phtj[0]-np.dot(phti[1:],phtj[1:])) for phti in phats] for phtj in phats])**(beta/2)

    assert epsilon_diff(ezs, zs, 10**-13)
    assert epsilon_diff(ethetas, thetas, 10**-theta_eps)

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
