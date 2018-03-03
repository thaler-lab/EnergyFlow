from __future__ import absolute_import

import numpy as np

import energyflow as ef

def test_has_efp():
    assert ef.EFP

def test_has_efpset():
    assert ef.EFPSet

event = ef.utils.load_big_event()
def test_efpset_vs_efps():
    meas = 'hadr'
    eps = 10**-13
    s1 = ef.EFPSet('d<=7', measure=meas)
    efps = [ef.EFP(g, measure=meas) for g in s1.graphs()]
    ev = event[:100]
    r1 = s1.compute(ev)
    r2 = np.asarray([efp.compute(ev) for efp in efps])
    assert np.all(np.abs((r1-r2)/(r1+r2)*2) < eps)

def test_efps_vs_efms():
    ev = event[:100]
    s1 = ef.EFPSet('d<=7', measure='hadrefm')
    s2 = ef.EFPSet('d<=7', measure='hadrdot', beta=2)
    eps = 10**-13
    r1 = s1.compute(ev)
    r2 = s2.compute(ev)
    assert np.all(np.abs((r1-r2)/(r1+r2)*2) < eps)