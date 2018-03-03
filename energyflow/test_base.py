from __future__ import absolute_import

import numpy as np
import pytest

import energyflow as ef

table2a = {'prime': [0,1,2,5,12,33,103,333,1183,4442,17576],
           'comp': [0,1,3,8,23,66,212,686,2389,8682,33160]}

table2b = [
[0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0],
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

g_7_default = ef.Generator(dmax=7, filename='default')
g_10_default = ef.Generator(dmax=10, filename='default')
g_7 = ef.Generator(dmax=7)


def test_gen_matches_file():
    assert np.all(g_7_default.specs == g_7.specs)

sp = g_10_default.specs
c_sp = g_10_default.c_specs
def test_table2a():
    for d in range(1, g_10_default.dmax+1):
        num_prime = np.count_nonzero(c_sp[:,g_10_default.d_ind] == d)
        num_comp = np.count_nonzero(sp[:,g_10_default.d_ind] == d)
        assert num_prime == table2a['prime'][d]
        assert num_comp == table2a['comp'][d]

def test_table2b():
    for d in range(g_10_default.dmax+1):
        dmask = c_sp[:,g_10_default.d_ind] == d
        for n in range(g_10_default.nmax+1):
            nmask = c_sp[:,g_10_default.n_ind] == n
            num = np.count_nonzero(dmask & nmask)
            assert num == table2b[n][d]



