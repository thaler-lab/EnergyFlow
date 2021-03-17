# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

import gzip
import os

import numpy as np
import pytest

import energyflow as ef
import pyfjcore

from test_utils import epsilon_percent, epsilon_diff, epsilon_either

@pytest.mark.pyfjcore
@pytest.mark.parametrize('mass', [0, 'random'])
@pytest.mark.parametrize('nparticles', [20, 50, (25, 100)])
def test_pjs_array_conversion(nparticles, mass):

    events = ef.gen_random_events(100, nparticles, mass=mass)
    for event in events:
        ptyphims = ef.ptyphims_from_p4s(event)

        for f in [lambda x: x, tuple]:

            # make pjs
            pjs_epxpypz = f(ef.pjs_from_p4s(event))
            pjs_ptyphim = f(ef.pjs_from_ptyphims(ptyphims))

            # check cartesian to cartesian
            assert np.all(event == ef.p4s_from_pjs(pjs_epxpypz))

            # check ptyphim to ptyphim
            ptyphims_pjs = ef.ptyphims_from_pjs(pjs_ptyphim)
            assert epsilon_either(ptyphims[:,:3], ptyphims_pjs[:,:3], 1e-13)
            assert epsilon_either(ptyphims[:,3], ptyphims_pjs[:,3], 1e-6)

            # check cartesian to ptyphim
            ptyphims_pjs = ef.ptyphims_from_pjs(pjs_epxpypz)
            assert epsilon_either(ptyphims[:,:3], ptyphims_pjs[:,:3], 1e-12, 1e-12)
            assert epsilon_either(ptyphims[:,3], ptyphims_pjs[:,3], 1e-6)

            # check ptyphims to cartesian
            p4s_pjs = ef.p4s_from_pjs(pjs_ptyphim)
            assert epsilon_either(event, p4s_pjs, 1e-12, 1e-10)

@pytest.mark.pyfjcore
@pytest.mark.parametrize('jet_R', [0.1, 0.2, 0.4, 0.8, 1, 4.25])
@pytest.mark.parametrize('algorithm', ['ca', 'kt', 'akt'])
def test_clustering(algorithm, jet_R):

    # load single event from fastjet
    with gzip.open(os.path.join(ef.utils.generic_utils.EF_DATA_DIR, 'single-event.dat.gz'), 'rb') as f:
        event = np.asarray([np.asarray(line.split(), dtype=float) for line in f.readlines()])[:,(3,0,1,2)]

    pjs = ef.pjs_from_p4s(event)
    jets = pyfjcore.sorted_by_pt(ef.cluster(pjs, algorithm=algorithm, R=jet_R))
    jets2 = pyfjcore.sorted_by_pt(ef.cluster(tuple(pjs), algorithm=algorithm, R=jet_R))
    jets3 = pyfjcore.sorted_by_pt(ef.jet_def(algorithm=algorithm, R=jet_R)(pjs))

    for j,j2,j3 in zip(jets, jets2, jets3):
        for i in range(4):
            epsilon_percent(j[i], j2[i])
            epsilon_percent(j[i], j3[i])

        for jj in [j, j2, j3]:
            inds = jj.constituents().user_indices()
            assert np.all(inds >= 0) and np.all(inds < len(event))

@pytest.mark.pyfjcore
@pytest.mark.parametrize('etamax', [-0.0000001, 0.7536, 3.9])
@pytest.mark.parametrize('etamin', [-4, -np.pi, -1.1, 0.5, 2.2])
@pytest.mark.parametrize('ptmax', [2, 5, 100, np.inf])
@pytest.mark.parametrize('ptmin', [0, 0.1, 0.5, 1.0, 10.])
def test_selectors(ptmin, ptmax, etamin, etamax):

    if ptmin >= ptmax or etamin >= etamax:
        pytest.skip()

    # load single event from fastjet
    with gzip.open(os.path.join(ef.utils.generic_utils.EF_DATA_DIR, 'single-event.dat.gz'), 'rb') as f:
        event = np.asarray([np.asarray(line.split(), dtype=float) for line in f.readlines()])[:,(3,0,1,2)]

    pjs = ef.pjs_from_p4s(event)
    pt_sel = pyfjcore.SelectorPtRange(ptmin, ptmax)
    pjs_sel = pt_sel(pjs)
    for pj in pjs_sel:
        assert pj.pt() >= ptmin and pj.pt() < ptmax

    sel = pt_sel | pyfjcore.SelectorEtaRange(etamin, etamax)
    pjs_sel = sel(pjs)
    for pj in pjs_sel:
        assert (pj.pt() >= ptmin and pj.pt() < ptmax) or (pj.eta() >= etamin and pj.eta() < etamax)

