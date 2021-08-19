r"""## Quark and Gluon Nsubs

A dataset consisting of 45 $N$-subjettiness observables for 100k quark and 
gluon jets generated with Pythia 8.230. Following [1704.08249](https:
//arxiv.org/abs/1704.08249), the observables are in the following order:

\[\{\tau_1^{(\beta=0.5)},\tau_1^{(\beta=1.0)},\tau_1^{(\beta=2.0)},
\tau_2^{(\beta=0.5)},\tau_2^{(\beta=1.0)},\tau_2^{(\beta=2.0)},
\ldots,
\tau_{15}^{(\beta=0.5)},\tau_{15}^{(\beta=1.0)},\tau_{15}^{(\beta=2.0)}\}.\]

The dataset contains two members: `'X'` which is a numpy array of the nsubs
that has shape `(100000,45)` and `'y'` which is a numpy array of quark/gluon 
labels (quark=`1` and gluon=`0`).
"""

#   ____   _____          _   _  _____ _    _ ____   _____
#  / __ \ / ____|        | \ | |/ ____| |  | |  _ \ / ____|
# | |  | | |  __         |  \| | (___ | |  | | |_) | (___
# | |  | | | |_ |        | . ` |\___ \| |  | |  _ < \___ \
# | |__| | |__| | ______ | |\  |____) | |__| | |_) |____) |
#  \___\_\\_____||______||_| \_|_____/ \____/|____/|_____/

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

import numpy as np

from energyflow.utils.data_utils import _get_filepath

__all__ = ['load']

def load(num_data=-1, cache_dir=None):
    """Loads the dataset. The first time this is called, it will automatically
    download the dataset. Future calls will attempt to use the cached dataset 
    prior to redownloading.

    **Arguments**

    - **num_data** : _int_
        - The number of events to return. A value of `-1` means read in all events.
    - **cache_dir** : _str_ or `None`
        - The directory where to store/look for the files. If `None`, the
        [`determine_cache_dir`](../utils/#determine_cache_dir) function will be
        used to get the default path. Note that in either case, `'datasets'` is
        appended to the end of the path.

    **Returns**

    - _3-d numpy.ndarray_, _1-d numpy.ndarray_
        - The `X` and `y` components of the dataset as specified above.
    """

    fpath = _get_filepath('QG_nsubs.npz', 
                      url='https://www.dropbox.com/s/y1l6avj5yj7jn9t/QG_nsubs.npz?dl=1',
                      file_hash='a99f771147af9b207356c990430cfeba6b6aa96fe5cff8263450ff3a31ab0997',
                      cache_subdir='datasets',
                      cache_dir=cache_dir)

    with np.load(fpath) as f:
        X, y = f['X'], f['y']

    if num_data > -1:
        X, y = X[:num_data], y[:num_data]

    return X, y
    