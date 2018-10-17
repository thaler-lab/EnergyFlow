"""### Quark and Gluon Jets

A dataset consisting of up to 2 million total quark and gluon jets generated with PYTHIA 8.226.
To avoid downloading unnecessary samples, the dataset is contained in twenty
files with 100k jets each, and only the required files are downloaded.
These samples are used in [1810.05165](https://arxiv.org/abs/1810.05165).
Splitting the data into 1.6M/200k/200k train/validation/test sets is recommended for standardized comparisons.

The dataset `qg_jets` consists of two components:

* `X` : a three-dimensional numpy array of the jets with shape `(num_data, max_num_particles, 4)`
* `y` : a numpy array of quark/gluon jet labels 
(quark=`1` and gluon=`0`).

The jets are padded with zero-particles in order to make a contiguous 
array. The particles are given as `(pt,y,phi,pid)` values, where `pid` is the particle's 
[PDG id](http://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf).


The samples are $Z(\\to\\nu\\bar\\nu)+g$ and $Z(\\to\\nu\\bar\\nu)+(u,d,s)$
 events generated with PYTHIA for $pp$ collisions at $\sqrt{s}=14$ TeV using the `WeakBosonAndParton:qqbar2gmZg` 
 and `WeakBosonAndParton:qg2gmZq` processes, ignoring the photon contribution and requiring
 the $Z$ to decay invisibly to neutrinos.
Hadronization and multiple parton interactions (i.e. underlying event) are turned on
 and the default tunings and shower parameters are used.
Final state non-neutrino particles are clustered into $R=0.4$ anti-$k_T$ jets
 using FASTJET 3.3.0.
Jets with transverse momentum $p_T\in[500,550]$ GeV and rapidity $|y|<2.0$ are kept.
Particles are ensured to have $\phi$ values within $\pi$ of the jet (i.e. no $\phi$-periodicity issues).
No detector simulation is performed.
"""
from __future__ import absolute_import

import warnings

import numpy as np

from energyflow.utils.data_utils import _get_file

__all__ = ['load']

QG_jets_urls = [
    'https://www.dropbox.com/s/fclsl7pukcpobsb/QG_jets.npz?dl=1',
    'https://www.dropbox.com/s/ztzd1a6lkmgovuy/QG_jets_1.npz?dl=1',
    'https://www.dropbox.com/s/jzgc9e786tbk1m5/QG_jets_2.npz?dl=1',
    'https://www.dropbox.com/s/tiwz2ck3wnzvlcr/QG_jets_3.npz?dl=1',
    'https://www.dropbox.com/s/3miwek1n0brbd2i/QG_jets_4.npz?dl=1',
    'https://www.dropbox.com/s/tsq80wc6ngen9kn/QG_jets_5.npz?dl=1',
    'https://www.dropbox.com/s/5oba2h15ufa57ie/QG_jets_6.npz?dl=1',
    'https://www.dropbox.com/s/npl6b2rts82r1ya/QG_jets_7.npz?dl=1',
    'https://www.dropbox.com/s/7pldxfqdb4n0kaw/QG_jets_8.npz?dl=1',
    'https://www.dropbox.com/s/isw4clv7n370nfb/QG_jets_9.npz?dl=1',
    'https://www.dropbox.com/s/prw7myb889v2y12/QG_jets_10.npz?dl=1',
    'https://www.dropbox.com/s/10r4ydro3e6nsmc/QG_jets_11.npz?dl=1',
    'https://www.dropbox.com/s/42p10sv9jedmtn0/QG_jets_12.npz?dl=1',
    'https://www.dropbox.com/s/crqdeg4arjti7cy/QG_jets_13.npz?dl=1',
    'https://www.dropbox.com/s/1e7ss2quxhkbhwy/QG_jets_14.npz?dl=1',
    'https://www.dropbox.com/s/psje9feje43buc7/QG_jets_15.npz?dl=1',
    'https://www.dropbox.com/s/8qw5bcswgrr9fl1/QG_jets_16.npz?dl=1',
    'https://www.dropbox.com/s/gcdp98bgupfk05x/QG_jets_17.npz?dl=1',
    'https://www.dropbox.com/s/jvgt17z1ufxz1ly/QG_jets_18.npz?dl=1',
    'https://www.dropbox.com/s/gbbfvy2e0slmm8v/QG_jets_19.npz?dl=1',
]

QG_jets_hashes = [
    '3f27a02eab06e8b83ccc9d25638021e6e24c9361341730961f9d560dee12c257',
    '648e49cd59b5353e0064e7b1a3388d9c2f4a454d3ca67afaa8d0344c836ecb35',
    '09f7b16fa7edb312c0f652bb8504de45f082c4193df65204d693155017272fe9',
    '7dc9a50bb38e9f6fc1f11db18f9bd04f72823c944851746b848dee0bba808537',
    '3e6217aad8e0502f5ce3b6371c61396dfc48a6cf4f26ee377cc7b991b1d2b543',
    'b5b7d742b2599bcbe1d7a639895bca64c28da513dc3620b0e5bbb5801f8c88fd',
    '7d31bc48c15983401e0dbe8fd5ee938c3809d9ee3c909f4adab6daf8b73c14f1',
    'cec0d7b2afa9d955543c597f9b7f3b3767812a68b2401ec870caf3a2ceb98401',
    'e984620f57abe06fc5d0b063f9f84ba54bd3e8c295d2b2419a7b1c6175079ed4',
    '6e3b69196995d6eb3b8e7af874e2b9f93d904624f7a7a73b8ff39f151e3bd189',
    'fa3d386f230b806058ff17e5bd77326ff4bf01d72aa5eb3325c1df2a8825927c',
    'acd49ab7bea8f72ecf699a9a898bccacc8730474259d68406656a5a43d407fb0',
    '2edd55b8bc30c686a0637855e1ba068586eb97041e8114d5540d96db2a7a2e17',
    '7276a8a0e573f9795a47f9d5addc10d2af903c2a0ffa5c848a720ccae93daa90',
    '2068ecfa912e94cd3ce7273b7c77af0bbd5ec57940997e7483b56f03434a6869',
    '41a732ce6321dd593214225b03fb87329607ccae768c705e3896ffecc28bfcca',
    '9d68caeb18f3ccf127b9032f52e63ee011c4381293a3a503f894e5c0741ae215',
    '086053ca611bb04d97fa0b6509b4ffb6955421b067c7b277498f0e5188879331',
    'cdc595f5fedef7db9411a9f93f2786f110073b4d17a523700f625846588b1e44',
    'd07781139320ae134ce4824bc0cefa43fd5003cd97cdf3aed90d4fb12fad8a1d',
]

num_per_file = 100000
max_num_files = len(QG_jets_urls)

def load(num_data=100000, cache_dir=None):
    """Loads samples from the dataset (which in total is contained in twenty files). 
    Any file that is needed that has not been cached will be automatically downloaded.
    Downloading a file causes it to be cached for later use. Basic checksums are
    performed.

    **Arguments**

    - **num_data** : _int_
        - The number of events to return. A value of `-1` means read in all events.
    - **cache_dir** : _str_
        - The directory where to store/look for the file.

    **Returns**

    - _3-d numpy.ndarray_, _1-d numpy.ndarray_
        - The `X` and `y` components of the dataset as specified above.
    """

    num_files = int(np.ceil(num_data/num_per_file)) if num_data > -1 else max_num_files

    # handle request for too much data
    if num_files > max_num_files:
        warnings.warn('More data requested than available. Providing the full dataset.')
        num_files = max_num_files
        num_data = -1

    Xs, ys = [], []
    for i in range(num_files):

        # preserve old first file
        filename = 'QG_jets_{}.npz'.format(i) if i > 0 else 'QG_jets.npz'

        f = np.load(_get_file(filename, url=QG_jets_urls[i], 
                                        file_hash=QG_jets_hashes[i], 
                                        cache_dir=cache_dir))
        Xs.append(f['X'])
        ys.append(f['y'])
        f.close()

    max_len_axis1 = max([X.shape[1] for X in Xs])

    X = np.vstack([_pad_events_axis1(x, max_len_axis1) for x in Xs])
    y = np.concatenate(ys)

    if num_data > -1:
        X, y = X[:num_data], y[:num_data]

    return X, y


def _pad_events_axis1(events, axis1_shape):
    """Pads the first axis of the NumPy array `events` with zero subarrays
    such that the first dimension of the results has size `axis1_shape`.
    """


    num_zeros = axis1_shape - events.shape[1]
    if num_zeros > 0:
        zeros = np.zeros([s if i != 1 else num_zeros for i,s in enumerate(events.shape)])
        return np.concatenate((events, zeros), axis=1)

    return events
    