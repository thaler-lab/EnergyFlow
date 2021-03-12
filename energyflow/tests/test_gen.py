# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division

import sys

import numpy as np
import pytest

import energyflow as ef

table2a = {'prime': [1,1,2,5,12,33,103,333,1183,4442,17576],
           'all': [1,1,3,8,23,66,212,686,2389,8682,33160]}

table2b = [
[0,0,0,0,0,0,0,0,0,0,0],
[1,0,0,0,0,0,0,0,0,0,0],
[0,1,1,1,1,1,1,1,1,1,1],
[0,0,1,2,3,4,6,7,9,11,13],
[0,0,0,2,5,11,22,37,61,95,141],
[0,0,0,0,3,11,34,85,193,396,771],
[0,0,0,0,0,6,29,110,348,969,2445],
[0,0,0,0,0,0,11,70,339,1318,4457],
[0,0,0,0,0,0,0,23,185,1067,4940],
[0,0,0,0,0,0,0,0,47,479,3294],
[0,0,0,0,0,0,0,0,0,106,1279],
[0,0,0,0,0,0,0,0,0,0,235]
]

@pytest.mark.gen
@pytest.mark.skipif(sys.version_info > (3, 7),
                    reason='order of generated EFPs different on Python 3.8 and higher')
def test_gen_matches_file():
    if sys.version_info[0] == 2:
        pytest.skip
    pytest.importorskip('igraph')
    g_7 = ef.Generator(dmax=7)
    g_7_default = ef.Generator(dmax=7, filename='default')
    assert np.all(g_7_default.specs == g_7.specs)

g_10_default = ef.Generator(dmax=10, filename='default')

sp = g_10_default.specs
c_sp = g_10_default.c_specs

@pytest.mark.gen
def test_table2a():
    for d in range(11):
        num_prime = np.count_nonzero(c_sp[:,g_10_default.d_ind] == d)
        num_comp = np.count_nonzero(sp[:,g_10_default.d_ind] == d)
        assert num_prime == table2a['prime'][d]
        assert num_comp == table2a['all'][d]

@pytest.mark.gen
def test_table2b():
    for d in range(11):
        dmask = c_sp[:,g_10_default.d_ind] == d
        for n in range(g_10_default.nmax+1):
            nmask = c_sp[:,g_10_default.n_ind] == n
            num = np.count_nonzero(dmask & nmask)
            assert num == table2b[n][d]
