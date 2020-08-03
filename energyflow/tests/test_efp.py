from __future__ import absolute_import, division

import os
import sys

import numpy as np
import pytest

import energyflow as ef
from test_utils import epsilon_percent, epsilon_diff

# function to check if a graph is leafless (i.e. has no valency-1 vertices)
def leafless(graph):
    return not 1 in np.unique(graph, return_counts=True)[1]

def test_has_efp():
    assert ef.EFP

def test_has_efpset():
    assert ef.EFPSet

# test individual EFPs
@pytest.mark.efp
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

@pytest.mark.efp
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

@pytest.mark.efp
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
    assert epsilon_percent(asymbox, efp_result, epsilon=10**-13)

@pytest.mark.slow
@pytest.mark.efp
@pytest.mark.efm
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, 0.5, 1, 'pf'])
@pytest.mark.parametrize('beta', [.5, 1, 2])
@pytest.mark.parametrize('measure', ['hadr', 'hadrdot', 'ee', 'hadrefm', 'eeefm'])
def test_batch_compute_vs_compute(measure, beta, kappa, normed):
    if measure == 'hadr' and kappa == 'pf':
        pytest.skip('hadr does not do pf')
    if kappa == 'pf' and normed:
        pytest.skip('normed not supported with kappa=pf')
    if ('efm' in measure) and (beta != 2):
        pytest.skip('only beta=2 can use efm measure')
    events = ef.gen_random_events(10, 15)
    s = ef.EFPSet('d<=6', measure=measure, beta=beta, kappa=kappa, normed=normed)
    r_batch = s.batch_compute(events, n_jobs=1)
    r = np.asarray([s.compute(event) for event in events])
    assert epsilon_percent(r_batch, r, 10**-14)

# test that efpset matches efps
@pytest.mark.slow
@pytest.mark.efp
@pytest.mark.efm
@pytest.mark.parametrize('event', ef.gen_random_events(3, 15))
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, 0.5, 1, 'pf'])
@pytest.mark.parametrize('beta', [.5, 1, 2])
@pytest.mark.parametrize('measure', ['hadr', 'hadrdot', 'ee', 'hadrefm', 'eeefm'])
def test_efpset_vs_efps(measure, beta, kappa, normed, event):
    # handle cases we want to skip
    if measure == 'hadr' and kappa == 'pf':
        pytest.skip('hadr does not do pf')
    if kappa == 'pf' and normed:
        pytest.skip('normed not supported with kappa=pf')    
    if ('efm' in measure) and (beta != 2):
        pytest.skip('only beta=2 can use efm measure')
    s1 = ef.EFPSet('d<=6', measure=measure, beta=beta, kappa=kappa, normed=normed)
    efps = [ef.EFP(g, measure=measure, beta=beta, kappa=kappa, normed=normed) for g in s1.graphs()]
    r1 = s1.compute(event)
    r2 = np.asarray([efp.compute(event) for efp in efps])
    assert epsilon_percent(r1, r2, 10**-12)

@pytest.mark.efp
@pytest.mark.kappa
@pytest.mark.parametrize('kappa_normed_behavior', ['new', 'orig'])
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, 0.5, 1])
@pytest.mark.parametrize('beta', [.5, 1, 2])
@pytest.mark.parametrize('event', ef.gen_random_events(3, 15))
def test_efp_kappa_hadr(event, beta, kappa, normed, kappa_normed_behavior):

    asymbox_graph = [(0,1),(0,1),(1,2),(1,2),(1,2),(2,3),(2,3),(2,3),(2,3),(3,0),(0,3),(3,0)]
    efp_result = ef.EFP(asymbox_graph, measure='hadr', beta=beta, kappa=kappa,
                                       normed=normed, coords='epxpypz',
                                       kappa_normed_behavior=kappa_normed_behavior).compute(event)

    pts, ys, phis, ms = ef.ptyphims_from_p4s(event).T
    thetas = np.asarray([[(ys[i]-ys[j])**2 + min(abs(phis[i]-phis[j]), 2*np.pi-abs(phis[i]-phis[j]))**2
                          for i in range(len(event))] for j in range(len(event))])**(beta/2)

    zs = pts**kappa
    if normed and kappa_normed_behavior == 'orig':
        zs /= np.sum(zs)
    if normed and kappa_normed_behavior == 'new':
        zs /= np.sum(pts)**kappa

    asymbox = 0
    for i1 in range(len(zs)):
        for i2 in range(len(zs)):
            for i3 in range(len(zs)):
                for i4 in range(len(zs)):
                    asymbox += (zs[i1]*zs[i2]*zs[i3]*zs[i4]*thetas[i1,i2]**2*
                                thetas[i2,i3]**3*thetas[i3,i4]**4*thetas[i1,i4]**3)

    assert epsilon_percent(asymbox, efp_result, epsilon=10**-13)

# test that efps, efms, and naive all agree
@pytest.mark.efp
@pytest.mark.efm
@pytest.mark.parametrize('event', ef.gen_random_events(4, 15))
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('measure', ['hadrefm', 'eeefm'])
def test_efp_efm_naive_compute(measure, normed, event):
    # compute the EFP corresponding to the "icecreamcone" graph in the usual way
    EFP_correlator = ef.EFP([(0,1),(0,1),(0,2),(1,2)], measure=measure, normed=normed, coords='epxpypz', beta=2)(event)

    # compute the EFP corresponding to "icecreamcone" as a contraction of EFMs with the metric
    metric = np.diag([1,-1,-1,-1])

    # compute several energy flow moments
    EFM2 = ef.EFM(2, measure=measure, normed=normed, coords='epxpypz')(event)
    EFM3 = ef.EFM(3, measure=measure, normed=normed, coords='epxpypz')(event)
    EFP_contraction = np.einsum('abc,def,gh,ad,be,cg,fh->', EFM3, EFM3, EFM2, *([metric]*4))

    # compute the EFP corresponding to "icecreamcone" with explicit sums
    if measure == 'hadrefm':
        zs = np.sqrt(event[:,1]**2 + event[:,2]**2)
    if measure == 'eeefm':
        zs = event[:,0]
        
    ns = (event/zs[:,np.newaxis])
    thetas = np.asarray([[np.sum(2*ni*nj*np.asarray([1,-1,-1,-1])) for ni in ns] for nj in ns])

    if normed:
        zs /= zs.sum()
    
    EFP_sums = 0
    for i in range(len(event)):
        for j in range(len(event)):
            for k in range(len(event)):
                EFP_sums += zs[i] * zs[j] * zs[k] * thetas[i,j] * thetas[i,j] * thetas[i,k] * thetas[j,k]

    # ensure that the two values agree
    assert epsilon_percent(EFP_correlator, EFP_contraction, 10**-12)
    assert epsilon_percent(EFP_correlator, EFP_sums, 10**-12)
    
# test that efps, efms, and naive all agree
@pytest.mark.efp
@pytest.mark.efm
@pytest.mark.parametrize('measure', ['hadrefm', 'eeefm'])
def test_linear_relations(measure):
    
    graphs ={# d=0
        'dot': [],

        # d=1
        'line': [(0,1)],

        # d=2
        'dumbbell': [(0,1), (0,1)],
        'wedge': [(0,1),(1,2)],
        'linesqd' : [(0,1),(2,3)],

        # d = 3
        'tribell' : [(0,1),(0,1),(0,1)],
        'triangle' : [(0,1),(1,2),(2,0)],
        'asymwedge' : [(0,1),(0,1),(1,2)],
        'birdfoot' : [(0,1),(0,2),(0,3)],
        'chain' : [(0,1),(1,2),(2,3)],
        'linedumbbell' : [(0,1),(2,3),(2,3)],
        'linewedge' : [(0,1),(2,3),(3,4)],
        'linecbd'  : [(0,1),(2,3),(4,5)],

        # d = 4
        'quadbell' : [(0,1),(0,1),(0,1),(0,1)],
        'doublewedge' : [(0,1),(0,1),(1,2),(1,2)],
        'icecreamcone' : [(0,1),(0,1),(1,2),(2,0)],
        'asymwedge2' : [(0,1),(0,1),(0,1),(1,2)],
        'square' : [(0,1),(1,2),(2,3),(3,0)],
        'flyswatter' : [(0,1),(1,2),(2,3),(3,1)],
        'chain2mid' : [(0,1),(1,2),(1,2),(2,3)],
        'chain2end' : [(0,1),(1,2),(2,3),(2,3)],
        'asymbirdfoot' : [(0,1),(0,1),(1,2),(1,3)],
        'bigbirdfoot' : [(0,1),(0,2),(0,3),(0,4)],
        'dog' : [(0,1),(1,2),(2,3),(2,4)],
        'bigchain' : [(0,1),(1,2),(2,3),(3,4)],

        'dumbbellwedge' : [(0,1),(0,1),(2,3),(3,4)],
        'triangleline' : [(0,1),(1,2),(2,0),(3,4)],
        'dumbbellsqd' : [(0,1),(0,1),(2,3),(2,3)],

        # d = 5
        'pentagon' : [(0,1),(1,2),(2,3),(3,4),(4,0)],
        'triangledumbbell': [(0,1),(0,1),(2,3),(3,4),(4,2)]
        }
    
    # pick a random event with 2 particles
    event = ef.gen_random_events(1, 2, dim=4)

    # compute the value of all of the EFPs on this event
    d = {name: ef.EFP(graph, measure=measure, coords='epxpypz')(event) for name,graph in graphs.items()}

    eps = 10**-8
    
    # check that the identities in the EFM paper are valid (i.e. = 0)
    assert epsilon_diff(2 * d['wedge'] - d['dumbbell'], 0, eps)
    assert epsilon_diff(2 * d['triangle'], 0, eps)
    assert epsilon_diff(d['tribell'] - 2 * d['asymwedge'], 0, eps)
    assert epsilon_diff(2 * d['chain'] - d['linedumbbell'] - d['triangle'], 0, eps)
    assert epsilon_diff(d['birdfoot'] + d['chain'] - d['asymwedge'], 0, eps)
    
    # Four Dimensions
    # pick a random event in 4 dimensions
    event = ef.gen_random_events(1, 25, dim=4)

    # compute the value of all of the EFPs on this event
    d = {name: ef.EFP(graph, measure=measure, coords='epxpypz')(event) for name,graph in graphs.items()}

    # check that the identity in the paper is valid (i.e. = 0)
    assert epsilon_percent(6*d['pentagon']-5*d['triangledumbbell'], 0, 10**-11)
    
    # count the number of leafless multigraphs (all or just connected) with degree d
    ds = np.arange(11)
    counts_all, counts_con = [], []

    # for each degree, get the graphs with edges<=d and check whether they are leafless
    for d in ds:
        counts_all.append(np.sum([leafless(graph) for graph in ef.EFPSet(('d<=',d)).graphs()]))
        counts_con.append(np.sum([leafless(graph) for graph in ef.EFPSet(('d<=',d), ('p==',1)).graphs()]))

    # note: computed counts are cumulative, must take the difference to get individual d    
    counts_all = np.asarray(counts_all[1:]) - np.asarray(counts_all[:-1])
    counts_con = np.asarray(counts_con[1:]) - np.asarray(counts_con[:-1])
    
    # ensure agreement with the table in the paper
    assert epsilon_diff(counts_all, [0,1,2,5,11,34,87,279,897,3129], eps)
    assert epsilon_diff(counts_con, [0,1,2,4,9,26,68,217,718,2553], eps)