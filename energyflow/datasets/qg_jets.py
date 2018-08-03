from __future__ import absolute_import

import numpy as np

from energyflow.utils import get_file

def load(num_data=-1, filename='QG_jets.npz', cache_dir=None):

    fpath = get_file(filename, 
                     url='https://www.dropbox.com/s/fclsl7pukcpobsb/QG_jets.npz?dl=1',
                     file_hash='3f27a02eab06e8b83ccc9d25638021e6e24c9361341730961f9d560dee12c257',
                     cache_dir=cache_dir)

    f = np.load(fpath)
    X, y = f['X'], f['y']
    f.close()

    if num_data > -1:
        X, y = X[:num_data], y[:num_data]

    return X, y