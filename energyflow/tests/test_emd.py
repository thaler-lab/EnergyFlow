from __future__ import absolute_import, division, print_function

import itertools
import warnings

import numpy as np
import ot
import pytest

import energyflow as ef
from energyflow import emd
from test_utils import epsilon_percent, epsilon_diff

warnings.filterwarnings('error')

@pytest.mark.emd
def test_has_emd():
    assert emd.emd

@pytest.mark.emd
def test_has_emds():
    assert emd.emds

nev = 15

@pytest.mark.emd
@pytest.mark.parametrize('R', [np.sqrt(2)/2, 1.0, 2])
@pytest.mark.parametrize('norm', [True, False])
@pytest.mark.parametrize('M2', [1,2,5,25])
@pytest.mark.parametrize('M1', [1,2,5,25])
def test_emd_equivalence(M1, M2, norm, R):
    gdim = 2
    events1 = np.random.rand(nev, M1, gdim+1)
    events2 = np.random.rand(nev, M2, gdim+1)

    # test two different sets
    emds1 = np.zeros((nev, nev))
    for i,ev1 in enumerate(events1):
        for j,ev2 in enumerate(events2):
            emds1[i,j] = emd.emd(ev1, ev2, R=R, norm=norm, gdim=gdim)
    emds2 = emd.emds(events1, events2, R=R, norm=norm, verbose=0, n_jobs=1, gdim=gdim)

    assert epsilon_diff(emds1, emds2, 10**-12)

    # test same set
    emds1 = np.zeros((nev, nev))
    for i,ev1 in enumerate(events1):
        for j in range(i):
            emds1[i,j] = emd.emd(ev1, events1[j], R=R, norm=norm, gdim=gdim)
    emds1 += emds1.T
    emds2 = emd.emds(events1, R=R, norm=norm, verbose=0, n_jobs=1, gdim=gdim)

    assert epsilon_diff(emds1, emds2, 10**-12)

@pytest.mark.emd
@pytest.mark.parametrize('R', [np.sqrt(2)/2, 1.0, 2])
@pytest.mark.parametrize('norm', [True, False])
@pytest.mark.parametrize('n_jobs, M', itertools.product([1,2,None], [5,25]))
def test_n_jobs(n_jobs, M, norm, R):
    events = np.random.rand(nev, M, 3)
    emds1 = np.zeros((nev, nev))
    for i,ev1 in enumerate(events):
        for j in range(i):
            emds1[i,j] = emd.emd(ev1, events[j], R=R, norm=norm)
    emds1 += emds1.T
    emds2 = emd.emds(events, R=R, norm=norm, verbose=0, n_jobs=n_jobs)

    assert epsilon_diff(emds1, emds2, 10**-12)

@pytest.mark.emd
@pytest.mark.parametrize('R', [np.sqrt(2)/2, 1.0, 2])
@pytest.mark.parametrize('norm', [True, False])
@pytest.mark.parametrize('M', [1,2,5,25])
@pytest.mark.parametrize('gdim, evdim', [(gdim, gdim+off) for gdim in [1,2,3] for off in [0,1,2]])
def test_gdim(gdim, evdim, M, norm, R):
    if R < np.sqrt(gdim)/2:
        pytest.skip('R too small')
    events = np.random.rand(nev, M, 1+evdim)
    emds1 = emd.emds(events, gdim=gdim, norm=norm, R=R, n_jobs=1, verbose=0)
    emds2 = emd.emds(events[:,:,:1+gdim], gdim=100, norm=norm, R=R, n_jobs=1, verbose=0)

    assert epsilon_diff(emds1, emds2, 10**-13)

@pytest.mark.emd
@pytest.mark.periodic
@pytest.mark.parametrize('M', [1,2,5,25])
@pytest.mark.parametrize('gdim', [1,2,3])
def test_periodic_phi(gdim, M):
    events = np.random.rand(nev, M, 1+gdim)
    for phi_col in range(1,gdim+1):
        emds1 = emd.emds(events, R=1.0, periodic_phi=False, gdim=gdim, n_jobs=1, verbose=0)
        events_c = np.copy(events)
        events_c[:,:,phi_col] += 2*np.pi*np.random.randint(-10, 10, size=(nev, M))
        emds2 = emd.emds(events_c, R=1.0, gdim=gdim, periodic_phi=True, phi_col=phi_col, n_jobs=1, verbose=0)

    assert epsilon_diff(emds1, emds2, 10**-12)

@pytest.mark.emd
@pytest.mark.return_flow
@pytest.mark.parametrize('R', [np.sqrt(2)/2, 1.0])
@pytest.mark.parametrize('M', [1,5,25])
@pytest.mark.parametrize('dt', ['nt', 'nf', 'eq'])
def test_emd_return_flow(dt, M, R):
    events1 = np.random.rand(nev, M, 3)
    events2 = np.random.rand(nev, M, 3)
    for ev1 in events1:
        for ev2 in events2:
            if dt == 'eq':
                s1, s2 = ev1[:,0].sum(), ev2[:,0].sum()
                if s1 < s2:
                    ev1 = np.vstack((ev1, [s2-s1, np.random.rand(), np.random.rand()]))
                else:
                    ev2 = np.vstack((ev2, [s1-s2, np.random.rand(), np.random.rand()]))

            s1, s2 = ev1[:,0].sum(), ev2[:,0].sum()
            if dt == 'nt':
                Gshape = (M, M)
            elif s2 - s1 == 0:
                Gshape = (len(ev1), len(ev2))
            elif s2 - s1 > 0:
                Gshape = (len(ev1)+1, len(ev2))
            else:
                Gshape = (len(ev1), len(ev2)+1)

            cost, G = emd.emd(ev1, ev2, R=R, norm=(dt == 'nt'), return_flow=True)

            assert G.shape == Gshape

@pytest.mark.emd
@pytest.mark.byhand
@pytest.mark.parametrize('R', [np.sqrt(2)/2, 1.0, 2])
@pytest.mark.parametrize('norm', [True, False])
@pytest.mark.parametrize('gdim', [1,2,3])
def test_emd_byhand_1_1(gdim, norm, R):
    for i in range(nev):
        ev1 = np.random.rand(1+gdim)
        for j in range(nev):
            ev2 = np.random.rand(1+gdim)
            ef_emd = emd.emd(ev1, ev2, norm=norm, R=R, gdim=gdim)
            if norm:
                byhand_emd = np.linalg.norm(ev1[1:]-ev2[1:])/R
            else:
                byhand_emd = min(ev1[0], ev2[0])*np.linalg.norm(ev1[1:]-ev2[1:])/R + abs(ev1[0]-ev2[0])
            assert abs(ef_emd - byhand_emd) < 10**-15

# compare to function used in paper (which is off by a factor of R)
def emde(ev0, ev1, R=1.0, return_flow=False):
    pTs0, pTs1 = np.asarray(ev0[:,0], order='c'), np.asarray(ev1[:,0], order='c')
    thetas = ot.dist(np.vstack((ev0[:,1:3], np.zeros(2))), 
                     np.vstack((ev1[:,1:3], np.zeros(2))), metric='euclidean')
    
    # add a fictitious particle to the lower-energy event to balance the energy
    pT0, pT1 = pTs0.sum(), pTs1.sum()       
    pTs0 = np.hstack((pTs0, 0 if pT0 > pT1 else pT1-pT0))
    pTs1 = np.hstack((pTs1, 0 if pT1 > pT0 else pT0-pT1))
    
    # make its distance R to all particles in the other event
    thetas[:,-1] = R
    thetas[-1,:] = R
    
    if return_flow:
        G, log = ot.emd(pTs0, pTs1, thetas, log=True)
        return log['cost'], G
    else:
        return ot.emd2(pTs0, pTs1, thetas)

@pytest.mark.emd
@pytest.mark.parametrize('R', [np.sqrt(2)/2, 1.0, 2])
@pytest.mark.parametrize('M', [1,5,25])
def test_emde(M, R):
    events1 = np.random.rand(nev, M, 3)
    events2 = np.random.rand(nev, M, 3)

    for ev1 in events1:
        for ev2 in events2:
            ef_emd = emd.emd(ev1, ev2, R=R)
            emde_emd = emde(ev1, ev2, R=R)/R
            assert abs(ef_emd - emde_emd) < 10**-14

    for i,ev1 in enumerate(events1):
        for j in range(i):
            ef_emd = emd.emd(ev1, events1[j], R=R)
            emde_emd = emde(ev1, events1[j], R=R)/R
            assert abs(ef_emd - emde_emd) < 10**-14
