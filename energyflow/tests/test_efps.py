from __future__ import absolute_import, division

import os
import sys

import numpy as np
import pytest

import energyflow as ef
from test_utils import epsilon_percent, epsilon_diff

def test_has_efp():
    assert ef.EFP

def test_has_efpset():
    assert ef.EFPSet

# test individual EFPs
@pytest.mark.efpsbyhand
@pytest.mark.parametrize('beta', np.linspace(.2, 2.5, 7))
@pytest.mark.parametrize('zs, thetas', [(np.random.rand(M), np.random.rand(M,M)) for M in [10,20]])
def test_efp_wedge(zs, thetas, beta):
    thetas **= beta
    wedge= 0
    for i1 in range(len(zs)):
        for i2 in range(len(zs)):
            for i3 in range(len(zs)):
                wedge += zs[i1]*zs[i2]*zs[i3]*thetas[i1,i2]*thetas[i1,i3]

    efp_result = ef.EFP([(0,1),(0,2)], beta=beta).compute(zs=zs, thetas=thetas)
    assert epsilon_percent(wedge, efp_result, epsilon=10**-13)

@pytest.mark.efpsbyhand
@pytest.mark.parametrize('beta', np.linspace(.2, 2.5, 7))
@pytest.mark.parametrize('zs, thetas', [(np.random.rand(M), np.random.rand(M,M)) for M in [10,20]])
def test_efp_asymfly(zs, thetas, beta):
    thetas **= beta
    asymfly = 0
    for i1 in range(len(zs)):
        for i2 in range(len(zs)):
            for i3 in range(len(zs)):
                for i4 in range(len(zs)):
                    asymfly += (zs[i1]*zs[i2]*zs[i3]*zs[i4]*
                                thetas[i1,i2]*thetas[i2,i3]*thetas[i3,i4]*thetas[i2,i4]**2)

    efp_result = ef.EFP([(0,1),(1,2),(2,3),(1,3),(1,3)], beta=beta).compute(zs=zs, thetas=thetas)
    assert epsilon_percent(asymfly, efp_result, epsilon=10**-13)

@pytest.mark.efpsbyhand
@pytest.mark.parametrize('beta', np.linspace(.2, 2.5, 7))
@pytest.mark.parametrize('zs, thetas', [(np.random.rand(M), np.random.rand(M,M)) for M in [10,20]])
def test_efp_asymbox(zs, thetas, beta):
    thetas **= beta
    asymbox = 0
    for i1 in range(len(zs)):
        for i2 in range(len(zs)):
            for i3 in range(len(zs)):
                for i4 in range(len(zs)):
                    asymbox += (zs[i1]*zs[i2]*zs[i3]*zs[i4]*thetas[i1,i2]**2*
                                thetas[i2,i3]**3*thetas[i3,i4]**4*thetas[i1,i4]**3)
    
    asymbox_graph = [(0,1),(0,1),(1,2),(1,2),(1,2),(2,3),(2,3),(2,3),(2,3),(3,0),(0,3),(3,0)]
    efp_result = ef.EFP(asymbox_graph, beta=beta).compute(zs=zs, thetas=thetas)
    return epsilon_percent(asymbox, efp_result, epsilon=10**-13)

nogood = pytest.mark.xfail(raises=NotImplementedError) if sys.platform.startswith('linux') else []

@pytest.mark.slow
@pytest.mark.batch_compute
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, 0.5, 1, 'pf'])
@pytest.mark.parametrize('beta', [.5, 1, 2])
@pytest.mark.parametrize('measure', ['hadr', 'hadrdot', 'ee'])
def test_batch_compute_vs_compute(measure, beta, kappa, normed):
    if measure == 'hadr' and kappa == 'pf':
        pytest.skip('hadr does not do pf')
    if kappa == 'pf' and normed:
        pytest.skip('normed not supported with kappa=pf')
    events = ef.gen_random_events(10, 15)
    s = ef.EFPSet('d<=6', measure=measure, beta=beta, kappa=kappa, normed=normed)
    r_batch = s.batch_compute(events, n_jobs=1)
    r = np.asarray([s.compute(event) for event in events])
    assert epsilon_percent(r_batch, r, 10**-14)

# test that efpset matches efps
@pytest.mark.slow
@pytest.mark.efpset
@pytest.mark.parametrize('event', ef.gen_random_events(2, 15))
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, 0.5, 1, 'pf'])
@pytest.mark.parametrize('beta', [.5, 1, 2])
@pytest.mark.parametrize('measure', ['hadr', 'hadrdot', 'ee'])
def test_efpset_vs_efps(measure, beta, kappa, normed, event):
    # handle cases we want to skip
    if measure == 'hadr' and kappa == 'pf':
        pytest.skip('hadr does not do pf')
    if kappa == 'pf' and normed:
        pytest.skip('normed not supported with kappa=pf')
    s1 = ef.EFPSet('d<=6', measure=measure, beta=beta, kappa=kappa, normed=normed)
    efps = [ef.EFP(g, measure=measure, beta=beta, kappa=kappa, normed=normed) for g in s1.graphs()]
    r1 = s1.compute(event)
    r2 = np.asarray([efp.compute(event) for efp in efps])
    assert epsilon_percent(r1, r2, 10**-12)
