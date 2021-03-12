# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division

import numpy as np
import pytest

import energyflow as ef
from test_utils import epsilon_percent, epsilon_diff

def test_has_efm():
    assert ef.EFM

def test_has_EFMSet():
    assert ef.EFMSet

def rec_outer(nhat, v, q=None):
    q = nhat if q is None else q
    return q if (q.ndim == v) else rec_outer(nhat, v, q=np.multiply.outer(nhat, q))

def slow_efm(zs, nhats, v):
    return (np.sqrt(2)**v) * np.sum([z*rec_outer(nhat, v) for z, nhat in zip(zs, nhats)], axis=0)

@pytest.mark.efm
@pytest.mark.parametrize('M', [1, 10, 50, 100, 500])
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('measure', ['hadrefm', 'eeefm'])
@pytest.mark.parametrize('v', list(range(0,2)))
def test_efms(v, measure, normed, M):
    
    events = ef.gen_random_events(2, M)
    e = ef.EFM(v, measure=measure, normed=normed, coords='epxpypz')

    for event in events:
        if measure == 'hadrefm':
            zs = np.atleast_1d(ef.pts_from_p4s(event))
        elif measure == 'eeefm':
            zs = event[:,0]
            
        nhats = event/zs[:,np.newaxis]
        if normed:
            zs = zs/zs.sum()

        e_ans = e.compute(event)
        if v == 0:
            assert epsilon_percent(e_ans, zs.sum(), 10**-13)
        else:
            s_ans = slow_efm(zs, nhats, v)
            assert epsilon_percent(s_ans, e_ans, 10**-13)
  
@pytest.mark.efm
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('measure', ['hadrefm', 'eeefm'])
@pytest.mark.parametrize('M', [1, 10, 50, 100, 500])
@pytest.mark.parametrize('v', list(range(0,2)))
def test_efm_batch_compute(v, M, measure, normed):
    events = ef.gen_random_events(2, M)
    e = ef.EFM(v, measure=measure, normed=normed, coords='epxpypz')
    
    r1 = [e.compute(event) for event in events]
    r2 = e.batch_compute(events)
    
    assert epsilon_percent(r1, r2, 10**-1)
            
@pytest.mark.efm
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('measure', ['hadrefm', 'eeefm'])
@pytest.mark.parametrize('M', [1, 10, 50, 100, 500])
@pytest.mark.parametrize('sigs', [[(1,0),(1,1),(3,2),(0,4),(2,3),(1,2)],
                                  [(0,0),(1,0),(0,2),(1,2),(6,2),(1,5)]])
def test_efm_vs_efmset_compute(sigs, M, measure, normed):
    
    efmset = ef.EFMSet(sigs, measure=measure, normed=normed, coords='epxpypz')
    efms = [ef.EFM(*sig, measure=measure, normed=normed, coords='epxpypz') for sig in sigs]

    for event in ef.gen_random_events(2, M):
        efm_dict = efmset.compute(event)
        for sig,efm in zip(sigs,efms):
            print(sig, np.max(np.abs(efm_dict[sig] - efm.compute(event))))
            assert epsilon_percent(efm_dict[sig], efm.compute(event), 10**-10)

@pytest.mark.efm
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('measure', ['hadrefm', 'eeefm'])
@pytest.mark.parametrize('M', [1, 10, 50, 100, 500])
@pytest.mark.parametrize('sigs', [[(1,0),(1,1),(3,2),(0,4),(2,3),(1,2)],
                                  [(0,0),(1,0),(0,2),(1,2),(6,2),(1,5)]])
def test_efm_vs_efmset_batch_compute(sigs, M, measure, normed):
    
    efmset = ef.EFMSet(sigs, measure=measure, normed=normed, coords='epxpypz')
    efms = [ef.EFM(*sig, measure=measure, normed=normed, coords='epxpypz') for sig in sigs]

    events = ef.gen_random_events(2, M)
    efm_dict = efmset.batch_compute(events)
    
    for sig,efm in zip(sigs,efms):
        results = efm.batch_compute(events)
        for i in range(len(events)):
            assert epsilon_percent(efm_dict[i][sig], results[i], 10**-10)

     