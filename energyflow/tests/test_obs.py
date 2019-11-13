from __future__ import absolute_import, division

import os
import sys

import numpy as np
import pytest

import energyflow as ef
from test_utils import epsilon_percent, epsilon_diff

@pytest.mark.slow
@pytest.mark.obs
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('beta', [.5, 1, 2])
@pytest.mark.parametrize('measure', ['hadr', 'hadrdot', 'ee', 'hadrefm', 'eeefm'])
def test_C2D2C3(measure, beta, normed):
    
    # skip the efm measures for beta other than 2
    if ('efm' in measure) and (beta != 2):
        pytest.skip('only beta=2 can use efm measure')
    
    # generate a random event with 10 particles
    event = ef.gen_random_events(1, 10, dim=4)
    
    # specify the relevant graphs and EFPs to compute C1, D2, C3
    line = ef.EFP([(0,1)], measure=measure, coords='epxpypz', beta=beta, normed=True)(event)
    triangle = ef.EFP([(0,1), (0,2), (1,2)], measure=measure, coords='epxpypz', beta=beta, normed=True)(event)
    kite = ef.EFP([(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)], measure=measure, coords='epxpypz', beta=beta, normed=True)(event)

    # determine the observables
    C2val = triangle/line**2
    D2val = triangle/line**3
    C3val = kite*line/triangle**2
    
    for strassen in [True, False]:
        
        # skip strassen for EFM measures and hadr
        if ('efm' in measure or measure == 'hadr') and strassen:
            continue
        
        D2 = ef.obs.D2(measure=measure, beta=beta, strassen=strassen, normed=normed, coords='epxpypz')
        assert epsilon_diff(D2(event), D2val, 10**-10)

        C2 = ef.obs.C2(measure=measure, beta=beta, strassen=strassen, normed=normed, coords='epxpypz')
        assert epsilon_diff(C2(event), C2val, 10**-10)

    C3 = ef.obs.C3(measure=measure, beta=beta, coords='epxpypz')
    assert epsilon_diff(C3(event), C3val, 10**-10)