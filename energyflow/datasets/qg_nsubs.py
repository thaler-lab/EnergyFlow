from __future__ import absolute_import

import numpy as np

from energyflow.utils import get_file

def load(num_data=-1, filename='QG_nsubs.npz', cache_dir=None):

    fpath = get_file(filename, 
                     url='https://www.dropbox.com/s/y1l6avj5yj7jn9t/QG_nsubs.npz?dl=1',
                     file_hash='a99f771147af9b207356c990430cfeba6b6aa96fe5cff8263450ff3a31ab0997',
                     cache_dir=cache_dir)

    f = np.load(fpath)
    X, y = f['X'], f['y']
    f.close()

    if num_data > -1:
        X, y = X[:num_data], y[:num_data]

    return X, y