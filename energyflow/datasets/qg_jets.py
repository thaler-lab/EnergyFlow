r"""## Quark and Gluon Jets

Four datasets of quark and gluon jets, each having two million total jets, have
been generated with [Pythia](http://home.thep.lu.se/~torbjorn/Pythia.html) and
[Herwig](https://herwig.hepforge.org/) and are accessible through this
submodule of EnergyFlow. The four datasets are:

- Pythia 8.226 quark (uds) and gluon jets.
- Pythia 8.235 quark (udscb) and gluon jets.
- Herwig 7.1.4 quark (uds) and gluon jets.
- Herwig 7.1.4 quark (udscb) and gluon jets

To avoid downloading unnecessary samples, the datasets are contained in twenty
files with 100k jets each, and only the required files are downloaded. These
are based on the samples used in 
[1810.05165](https://arxiv.org/abs/1810.05165). Splitting the data into 
1.6M/200k/200k train/validation/test sets is recommended for standardized
comparisons.

Each dataset consists of two components:

- `X` : a three-dimensional numpy array of the jets with shape 
`(num_data,max_num_particles,4)`.
- `y` : a numpy array of quark/gluon jet labels (quark=`1` and gluon=`0`).

The jets are padded with zero-particles in order to make a contiguous array.
The particles are given as `(pt,y,phi,pid)` values, where `pid` is the
particle's [PDG id](http://pdg.lbl.gov/2018/reviews/rpp2018-rev-monte
-carlo-numbering.pdf). Quark jets either include or exclude $c$ and $b$
quarks depending on the `with_bc` argument.

The samples are generated from $q\bar q\to Z(\to\nu\bar\nu)+g$ and
$qg\to Z(\to\nu\bar\nu)+(uds[cb])$ processes in $pp$ collisions at
$\sqrt{s}=14$ TeV. Hadronization and multiple parton interactions (i.e.
underlying event) are turned on and the default tunings and shower parameters
are used. Final state non-neutrino particles are clustered into $R=0.4$
anti-$k_T$ jets using FastJet 3.3.0. Jets with transverse momentum
$p_T\in[500,550]$ GeV and rapidity $|y|<1.7$ are kept. Particles are ensured
have to $\phi$ values within $\pi$ of the jet (i.e. no $\phi$-periodicity 
issues). No detector simulation is performed.

The samples are also hosted on Zenodo and we ask that you cite them
appropriately if they are useful to your research. For BibTex entries,
see the [FAQs](/faqs/#how-do-i-cite-the-energyflow-package).

[![DOI](/img/zenodo.3164691.svg)](https://doi.org/10.5281/zenodo.3164691) - Pythia samples
<br>
[![DOI](/img/zenodo.3066475.svg)](https://doi.org/10.5281/zenodo.3066475) - Herwig samples
"""

#   ____   _____               _ ______ _______ _____
#  / __ \ / ____|             | |  ____|__   __/ ____|
# | |  | | |  __              | | |__     | | | (___
# | |  | | | |_ |         _   | |  __|    | |  \___ \
# | |__| | |__| | ______ | |__| | |____   | |  ____) |
#  \___\_\\_____||______| \____/|______|  |_| |_____/

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

import json
import os
import warnings

import numpy as np

from energyflow.utils.data_utils import _get_filepath, _pad_events_axis1
from energyflow.utils.generic_utils import EF_DATA_DIR, DROPBOX_URL_PATTERN, ZENODO_URL_PATTERN

__all__ = ['load']

GENERATORS = ['pythia', 'herwig']
NUM_PER_FILE = 100000
MAX_NUM_FILES = 20

# load(num_data=100000, pad=True, ncol=4, generator='pythia',
#      with_bc=False, cache_dir='~/.energyflow')
def load(num_data=100000, pad=True, ncol=4, generator='pythia', source='zenodo',
         with_bc=False, cache_dir=None, dtype='float64'):
    """Loads samples from the dataset (which in total is contained in twenty 
    files). Any file that is needed that has not been cached will be 
    automatically downloaded. Downloading a file causes it to be cached for
    later use. Basic checksums are performed.

    **Arguments**

    - **num_data** : _int_
        - The number of events to return. A value of `-1` means read in all
        events.
    - **pad** : _bool_
        - Whether to pad the events with zeros to make them the same length.
        Note that if set to `False`, the returned `X` array will be an object
        array and not a 3-d array of floats.
    - **ncol** : _int_
        - Number of columns to keep in each event.
    - **generator** : _str_
        - Specifies which Monte Carlo generator the events should come from.
        Currently, the options are `'pythia'` and `'herwig'`.
    - **with_bc** : _bool_
        - Whether to include jets coming from bottom or charm quarks. Changing
        this flag does not mask out these jets but rather accesses an entirely
        different dataset. The datasets with and without b and c quarks should
        not be combined.
    - **cache_dir** : _str_ or `None`
        - The directory where to store/look for the files. If `None`, the
        [`determine_cache_dir`](../utils/#determine_cache_dir) function will be
        used to get the default path. Note that in either case, `'datasets'` is
        appended to the end of the path.
    - **dtype** : _str_ or _numpy.dtype_
        - The dtype of the resulting NumPy arrays. For ML applications it may be
        preferred to use 32-bit floats.

    **Returns**

    - _3-d numpy.ndarray_, _1-d numpy.ndarray_
        - The `X` and `y` components of the dataset as specified above. If
        `pad` is `False` then these will be object arrays holding the events,
        each of which is a 2-d ndarray.
    """

    # load info from JSON file
    if 'QG_INFO' not in globals():
        global QG_INFO
        with open(os.path.join(EF_DATA_DIR, 'qg_jets.json'), 'r') as f:
            QG_INFO = json.load(f)

    # check for valid options
    if generator not in GENERATORS:
        raise ValueError("'generator' must be in " + str(GENERATORS))

    # get number of files we need
    num_files = int(np.ceil(num_data/NUM_PER_FILE)) if num_data > -1 else MAX_NUM_FILES
    if num_files > MAX_NUM_FILES:
        warnings.warn('More data requested than available. Providing the full dataset.')
        num_files = MAX_NUM_FILES
        num_data = -1

    # index into global info
    qg_info = QG_INFO[generator]['bc' if with_bc else 'nobc']
    filenames = qg_info['filenames'][:num_files]

    # get urls
    if source == 'dropbox':
        dropbox_hashes = qg_info['dropbox_link_hashes']
        urls = [DROPBOX_URL_PATTERN.format(h, f) for h,f in zip(dropbox_hashes, filenames)]
    elif source == 'zenodo':
        zenodo_record = qg_info['zenodo_record']
        urls = [ZENODO_URL_PATTERN.format(zenodo_record, f) for f in filenames]
    else:
        raise ValueError("source '{}' not recognized".format(source))

    # get hashes
    hashes = qg_info['hashes']['sha256']
    
    # obtain files
    Xs, ys = [], []
    for filename, url, h in zip(filenames, urls, hashes):
        try:
            fpath = _get_filepath(filename, url, cache_dir, cache_subdir='datasets', file_hash=h)

        except Exception as e:
            print(str(e))

            m = 'Failed to download {} from {}.'.format(filename, source)
            raise RuntimeError(m)

        # load file and append arrays
        with np.load(fpath) as f:
            fX = np.asarray(f['X'], dtype=dtype)
            ys.append(np.asarray(f['y'], dtype=dtype))
            if pad:
                Xs.append(fX)
            else:
                Xs.extend([x[x[:,0] > 0,:ncol] for x in fX])

    # get X array
    if pad:
        max_len_axis1 = max([X.shape[1] for X in Xs])
        X = np.vstack([_pad_events_axis1(x[...,:ncol], max_len_axis1) for x in Xs])
    else:
        X = np.asarray(Xs, dtype='O')

    # get y array
    y = np.concatenate(ys)

    # chop down to specified amount of data
    if num_data > -1:
        X, y = X[:num_data], y[:num_data]

    return X, y
    