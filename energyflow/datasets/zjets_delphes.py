# coding: utf-8
r"""## Z + Jets with Delphes Simulation

Datasets of QCD jets used for studying unfolding in [OmniFold: A Method to
Simultaneously Unfold All Observables](https://arxiv.org/abs/1911.09107). Four
different datasets are present:

- Herwig 7.1.5 with the default tune
- Pythia 8.243 with tune 21 (ATLAS A14 central tune with NNPDF2.3LO)
- Pythia 8.243 with tune 25 (ATLAS A14 variation 2+ of tune 21)
- Pythia 8.243 with tune 26 (ATLAS A14 variation 2- of tune 21)

$Z$ + jet events (with the $Z$ set to be stable) were generated for each of the
above generator/tune pairs with the $Z$ boson $\hat p_T^\text{min} > 150$ GeV
and $\sqrt{s}=14$ TeV. Events were then passed through the Delphes 3.4.2 fast
detector simulation of the CMS detector. Jets with radius parameter $R=0.4$
were found with the anti-kt algorithm at both particle level ("gen"), where all
non-neutrino, non-$Z$ particle are used, and detector level ("sim"), where
reconstructed energy flow objects (tracks, electromagnetic calorimeter cells,
and hadronic calorimeter cells) are used. Only jets with transverse momentum
greater than  are kept (note that sim jets have a simple jet energy correction
applied by Delphes). The hardest jet from events with a $Z$ boson with a final
transverse momentum of 200 GeV or greater are kept, yielding approximately 1.6
million jets at both gen-level and sim-level for each generator/tune pair.

Each zipped NumPy file consists of several arrays, the names of which begin
with either 'gen_' or 'sim_' depending on which set of jets they correspond
to. The name of each array ends in a key word indicating what it contains. With
the exception of 'gen_Zs' (which contains the $(p_T,\,y,\,\phi)$ of the final
$Z$ boson), there is both a gen and sim version of each array. The included
arrays are (listed by their key words):

- `'jets'` - The jet axis four vector, as $(p_T^\text{jet},\,y^\text{jet},\,
\phi^\text{jet},\,m^\text{jet})$ where $y^\text{jet}$ is the jet rapidity,
$\phi^\text{jet}$ is the jet azimuthal angle, and $m^\text{jet}$ is the jet
mass.
- `'particles'` - The rescaled and translated constituents of the jets as $(p_T/100,\,y-y^\text{jet},\,\phi-\phi^\text{jet},\,f_\text{PID})$ where
$f_\text{PID}$ is a small float corresponding to the PDG ID of the particle.
The PIDs are remapped according to $22\to0.0$, $211\to0.1$, $-211\to0.2$, $130
\to0.3$, $11\to0.4$, $-11\to0.5$, $13\to0.6$, $-13\to0.7$, $321\to0.8$, $-321
\to0.9$, $2212\to1.0$, $-2212\to1.1$, $2112\to1.2$, $-2112\to1.3$. Note that
ECAL cells are treated as photons (id 22) and HCAL cells are treated as 
$K_L^0$ (id 130). 
- `'mults'` - The constituent multiplicity of the jet.
- `'lhas'` - The Les Houches ($\beta=1/2$) angularity.
- `'widths'` - The jet width ($\beta=1 angularity).
- `'ang2s'` - The $\beta=2$ angularity (note that this is very similar to the jet
mass, but does not depend on particle masses).
- `'tau2s'` - The 2-subjettiness value with $\beta=1$.
- `'sdms'` - The groomed mass with Soft Drop parameters $z_\text{cut}=0.1$ and
$\beta=0$.
- `'zgs'` - The groomed momentum fraction (same Soft Drop parameters as above).

If you use this dataset, please cite the Zenodo record as well as the
corresponding paper:

[![DOI](/img/zenodo.3548091.svg)](https://
doi.org/10.5281/zenodo.3548091) - Pythia/Herwig + Delphes samples

- A. Andreassen, P. T. Komiske, E. M. Metodiev, B. Nachman, J. Thaler, 
OmniFold: A Method to Simultaneously Unfold All Observables,
[arXiv:1911.09107](https://arxiv.org/abs/1911.09107).
"""

#  ______    _ ______ _______ _____          _____  ______ _      _____  _    _ ______  _____
# |___  /   | |  ____|__   __/ ____         |  __ \|  ____| |    |  __ \| |  | |  ____|/ ____|
#    / /    | | |__     | | | (___          | |  | | |__  | |    | |__) | |__| | |__  | (___
#   / / _   | |  __|    | |  \___ \         | |  | |  __| | |    |  ___/|  __  |  __|  \___ \
#  / /_| |__| | |____   | |  ____) | ______ | |__| | |____| |____| |    | |  | | |____ ____) |
# /_____\____/|______|  |_| |_____/ |______||_____/|______|______|_|    |_|  |_|______|_____/

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

import json
import os
import warnings

import numpy as np

from energyflow.utils import EF_DATA_DIR, ZENODO_URL_PATTERN
from energyflow.utils.data_utils import _get_filepath, _pad_events_axis1

__all__ = ['load']

DROPBOX_URL_PATTERN = 'https://www.dropbox.com/s/{}/{}?dl=1'
FILENAME_PATTERNS = {
    'pythia21': 'Pythia21_Zjet_pTZ-200GeV_{}.npz',
    'pythia25': 'Pythia25_Zjet_pTZ-200GeV_{}.npz',
    'pythia26': 'Pythia26_Zjet_pTZ-200GeV_{}.npz',
    'herwig': 'Herwig_Zjet_pTZ-200GeV_{}.npz'
}
DATASETS = frozenset(FILENAME_PATTERNS.keys())
KEYS = {
    # jets
    'jets': 'Jet four vectors, [pt, y, phi, m]',
    'particles': 'Jet constituents, [pt, y, phi, pid_float]',

    # Zs
    'Zs': 'Z momenta, [pt, y, phi]',

    # observables
    'ang2s': 'Jet angularity, beta = 2',
    'lhas': 'Les Houches Angularity, beta = 1/2',
    'mults': 'Jet constituent multiplicity',
    'sdms': 'Soft Dropped jet mass, zcut = 0.1, beta = 0',
    'tau2s': '2-subjettiness, beta = 1',
    'widths': 'Jet widths, beta = 1',
    'zgs': 'Groomed momentum fraction, zcut = 0.1, beta = 0',
}
NUM_FILES = 17
SOURCES = ['dropbox', 'zenodo']
ZENODO_RECORD = '3548091'

# load(dataset, num_data=100000, pad=False, cache_dir='~/.energyflow',
#               source='zenodo', which='all',
#               include_keys=None, exclude_keys=None)
def load(dataset, num_data=100000, pad=False, cache_dir='~/.energyflow', source='zenodo', 
                  which='all', include_keys=None, exclude_keys=None):
    """Loads in the Z+jet Pythia/Herwig + Delphes datasets. Any file that is
    needed that has not been cached will be automatically downloaded.
    Downloaded files are cached for later use. Checksum verification is
    performed.

    **Arguments**

    - **datasets**: {`'Herwig'`, `'Pythia21'`, `'Pythia25'`, `'Pythia26'`}
        - The dataset (specified by which generator/tune was used to produce
        it) to load. Note that this argument is not sensitive to
        capitalization.
    - **num_data**: _int_
        - The number of events to read in. A value of `-1` means to load all
        available events.
    - **pad**: _bool_
        - Whether to pad the particles with zeros in order to form contiguous
        arrays.
    - **cache_dir**: _str_
        - Path to the directory where the dataset files should be stored.
    - **source**: {`'dropbox'`, `'zenodo'`}
        - Which location to obtain the files from.
    - **which**: {`'gen'`, `'sim'`, `'all'`}
        - Which type(s) of events to read in. Each dataset has corresponding
        generated events at particle-level and simulated events at
        detector-level.
    - **include_keys**: _list_ or_tuple_ of _str_, or `None`
        - If not `None`, load these keys from the dataset files. A value of
        `None` uses all available keys (the `KEYS` global variable of this
        module contains the available keys as keys of the dictionary and
        brief descriptions as values). Note that keys do not have 'sim' or
        'gen' prepended to the front yet.
    - **exclude_keys**: _list_ or _tuple_ or _str_, or `None`
        - Any keys to exclude from loading. Most useful when a small number of
        keys should be excluded from the default set of keys.

    **Returns**

    - _dict_ of _numpy.ndarray_
        - A dictionary of the jet, particle, and observable arrays for the
        specified dataset.
    """

    # load info from JSON file
    if 'INFO' not in globals():
        global INFO
        with open(os.path.join(EF_DATA_DIR, 'ZjetsDelphes.json'), 'r') as f:
            INFO = json.load(f)

    # check that options are valid
    dataset_low = dataset.lower()
    if dataset_low not in DATASETS:
        raise ValueError("Invalid dataset '{}'".format(dataset))
    if source not in SOURCES:
        raise ValueError("Invalud source '{}'".format(source))

    # handle selecting keys
    keys = set(KEYS.keys()) if include_keys is None else set(include_keys)
    keys -= set() if exclude_keys is None else set(exclude_keys)
    for key in keys:
        if key not in KEYS:
            raise ValueError("Unrecognized key '{}'".format(key))

    # create dictionray to store values to be returned
    levels = ['gen', 'sim'] if which == 'all' else [which.lower()]
    for level in levels:
        if level != 'gen' and level != 'sim':
            raise ValueError("Unrecognized specification '{}' ".format(level) +
                             "for argument 'which', allowed options are 'all', 'gen', 'sim'")
    vals = {'{}_{}'.format(level, key) : [] for level in levels for key in keys}
    if 'sim_Zs' in vals:
        del vals['sim_Zs']

    # get filenames
    filenames = [FILENAME_PATTERNS[dataset_low].format(i) for i in range(NUM_FILES)]

    # get urls
    if source == 'dropbox':
        db_link_hashes = INFO['dropbox_link_hashes'][dataset_low]
        urls = [DROPBOX_URL_PATTERN.format(dl, fn) for dl,fn in zip(db_link_hashes, filenames)]
    elif source == 'zenodo':
        urls = [ZENODO_URL_PATTERN.format(ZENODO_RECORD, fn) for fn in filenames]

    # get hashes
    hashes = INFO['hashes'][dataset_low]['sha256']

    n = 0
    subdir = os.path.join('datasets', 'ZjetsDelphes')
    for filename, url, h in zip(filenames, urls, hashes):

        # check if we have enough events
        if n >= num_data and num_data != -1:
            break

        # load file
        with np.load(_get_filepath(filename, url, cache_dir, cache_subdir=subdir, file_hash=h)) as f:

            # add relevant arrays to vals
            for i,(key,val) in enumerate(vals.items()):

                if 'particles' not in key or pad:
                    val.append(f[key])
                else:
                    val.append([np.array(ps[ps[:,0] > 0]) for ps in f[key]])

                # increment number of events
                if i == 0:
                    n += len(val[-1])

    # warn if we don't have enough events
    if num_data > n:
        warnings.warn('Only have {} events when {} were requested'.format(n, num_data))

    # concatenate arrays
    s = slice(0, num_data if num_data > -1 else None)
    for key,val in vals.items():

        if 'particles' not in key or not pad:
            vals[key] = np.concatenate(val, axis=0)[s]
        else:
            max_len_axis1 = max([X.shape[1] for X in val])
            vals[key] = np.vstack([_pad_events_axis1(x, max_len_axis1) for x in val])[s]

    return vals
