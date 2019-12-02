r"""## Z + Jets with Delphes Simulation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3548091.svg)](https://doi.org/10.5281/zenodo.3548091) - Pythia/Herwig + Delphes samples
"""
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
        f = np.load(_get_filepath(filename, url, cache_dir, cache_subdir=subdir, file_hash=h))

        # add relevant arrays to vals
        for i,(key,val) in enumerate(vals.items()):

            if 'particles' not in key or pad:
                val.append(f[key])
            else:
                val.append([np.array(ps[ps[:,0] > 0]) for ps in f[key]])

            # increment number of events
            if i == 0:
                n += len(val[-1])

        f.close()

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
