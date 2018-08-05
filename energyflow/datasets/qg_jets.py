"""###QG_jets

A dataset consisting of 100k quark and gluon jets generated with Pythia 8.230.
The dataset contains two members: `'X'` which is a numpy array of the jets that
has shape `(100000,139,4)` and `'y'` which is a numpy array of quark/gluon 
labels (quark=`1` and gluon=`0`). The jets are padded with zero-particles in order
to make a contiguous array. The particles are given as `(pt,y,phi,pid)` values 
where `pid` is the particle's [PDG id](http://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf).
"""
from __future__ import absolute_import

import numpy as np

from energyflow.utils.data_utils import _get_file

__all__ = ['load']

def load(num_data=-1, filename='QG_jets.npz', cache_dir=None):
    """Loads the dataset. The first time this is called, it will automatically
    download the dataset. Future calls will attempt to use the cached dataset 
    prior to redownloading.

    **Arguments**

    - **num_data** : _int_
        - The number of events to return. A value of `-1` means read in all events.
    - **filename** : _str_
        - The filename where to store/look for the file.
    - **cache_dir** : _str_
        - The directory where to store/look for the file.

    **Returns**

    - _3-d numpy.ndarray_, _1-d numpy.ndarray
        - The `X` and `y` components of the dataset as specified above.
    """

    fpath = _get_file(filename, 
                      url='https://www.dropbox.com/s/fclsl7pukcpobsb/QG_jets.npz?dl=1',
                      file_hash='3f27a02eab06e8b83ccc9d25638021e6e24c9361341730961f9d560dee12c257',
                      cache_dir=cache_dir)

    f = np.load(fpath)
    X, y = f['X'], f['y']
    f.close()

    if num_data > -1:
        X, y = X[:num_data], y[:num_data]

    return X, y