"""### Image Tools

Functions for dealing with image representations of events. These are 
not importable from the top level `energyflow` module, but must 
instead be imported from `energyflow.utils`.
"""
from __future__ import absolute_import, division, print_function

import numpy as np

__all__ = [
    'pixelate',
    'standardize',
    'zero_center',
]

# PDGid to isCharged dictionary
pid2abschg_mapping = {22: 0,             # photon
                      211: 1, -211: 1,   # pi+-
                      321: 1, -321: 1,   # K+-
                      130: 0,            # K-long
                      2112: 0, -2112: 0, # neutron, anti-neutron
                      2212: 1, -2212: 1, # proton, anti-proton
                      11: 1, -11: 1,     # electron, positron
                      13: 1, -13: 1}     # muon, anti-muon

def pixelate(jet, npix=33, img_width=0.8, nb_chan=1, norm=True, charged_counts_only=False):
    """A function for creating a jet image from an array of particles.

    **Arguments**

    - **jet** : _numpy.ndarray_
        - An array of particles where each particle is of the form 
        `[pt,y,phi,pid]` where the particle id column is only 
        used if `nb_chan=2` and `charged_counts_only=True`.
    - **npix** : _int_
        - The number of pixels on one edge of the jet image, which is
        taken to be a square.
    - **img_width** : _float_
        - The size of one edge of the jet image in the rapidity-azimuth
        plane.
    - **nb_chan** : {`1`, `2`}
        - The number of channels in the jet image. If `1`, then only a
        $p_T$ channel is constructed (grayscale). If `2`, then both a 
        $p_T$ channel and a count channel are formed (color).
    - **norm** : _bool_
        - Whether to normalize the $p_T$ pixels to sum to `1`.
    - **charged_counts_only** : _bool_
        - If making a count channel, whether to only include charged 
        particles. Requires that `pid` information be given.

    **Returns**

    - _3-d numpy.ndarray_
        - The jet image as a `(nb_chan, npix, npix)` array.
    """

    # set columns
    (pT_i, rap_i, phi_i, pid_i) = (0, 1, 2, 3)

    # the image is (img_width x img_width) in size
    pix_width = img_width / npix
    jet_image = np.zeros((nb_chan, npix, npix))

    # remove particles with zero pt
    jet = jet[jet[:,pT_i] > 0]

    # get pt centroid values
    rap_avg = np.average(jet[:,rap_i], weights=jet[:,pT_i])
    phi_avg = np.average(jet[:,phi_i], weights=jet[:,pT_i])
    rap_pt_cent_index = np.ceil(rap_avg/pix_width - 0.5) - np.floor(npix / 2)
    phi_pt_cent_index = np.ceil(phi_avg/pix_width - 0.5) - np.floor(npix / 2)

    # center image and transition to indices
    rap_indices = np.ceil(jet[:,rap_i]/pix_width - 0.5) - rap_pt_cent_index
    phi_indices = np.ceil(jet[:,phi_i]/pix_width - 0.5) - phi_pt_cent_index

    # delete elements outside of range
    mask = np.ones(jet[:,rap_i].shape).astype(bool)
    mask[rap_indices < 0] = False
    mask[phi_indices < 0] = False
    mask[rap_indices >= npix] = False
    mask[phi_indices >= npix] = False
    rap_indices = rap_indices[mask].astype(int)
    phi_indices = phi_indices[mask].astype(int)

    # construct grayscale image
    if nb_chan == 1: 
        for pt,y,phi in zip(jet[:,pT_i][mask], rap_indices, phi_indices): 
            jet_image[0, phi, y] += pt

    # construct two-channel image
    elif nb_chan == 2:
        if charged_counts_only:
            for pt,y,phi,pid in zip(jet[:,pT_i][mask], rap_indices, 
                                    phi_indices, jet[:,pid_i][mask].astype(int)):
                jet_image[0, phi, y] += pt
                jet_image[1, phi, y] += pid2abschg_mapping.get(pid, 0)
        else:
            for pt,y,phi in zip(jet[:,pT_i][mask], rap_indices, phi_indices): 
                jet_image[0, phi, y] += pt
                jet_image[1, phi, y] += 1
    else:
        raise ValueError('nb_chan must be 1 or 2')

    # L1-normalize the pt channels of the jet image
    if norm:
        normfactor = np.sum(jet_image[0])
        if normfactor == 0:
            raise FloatingPointError('Image had no particles!')
        else: 
            jet_image[0] /= normfactor

    return jet_image


# standardize(*args, channels=None, copy=False, reg=10**-10)
def standardize(*args, **kwargs):
    """Normalizes each argument by the standard deviation of the pixels in 
    arg[0]. The expected use case would be `standardize(X_train, X_val, X_test)`.

    **Arguments**

    - ***args** : arbitrary _numpy.ndarray_ datasets
        - An arbitrary number of datasets, each required to have
        the same shape in all but the first axis.
    - **channels** : _int_
        - A list of which channels (assumed to be the second axis)
        to standardize. `None` is interpretted to mean every channel.
    - **copy** : _bool_
        - Whether or not to copy the input arrays before modifying them.
    - **reg** : _float_
        - Small parameter used to avoid dividing by zero. It's important
        that this be kept consistent for images used with a given model.

    **Returns**

    - _list_ 
        - A list of the now-standardized arguments.
    """

    channels = kwargs.pop('channels', [])
    copy = kwargs.pop('copy', False)
    reg = kwargs.pop('reg', 10**-10)

    if len(kwargs):
        raise TypeError('following kwargs are invalid: {}'.format(kwargs))

    assert len(args) > 0

    # treat channels properly
    if channels is None: 
        channels = np.arange(args[0].shape[1])

    # compute stds
    stds = np.std(args[0], axis=0) + reg

    # copy arguments if requested
    if copy: 
        X = [np.copy(arg) for arg in args]
    else: 
        X = args

    # iterate through arguments and channels
    for x in X: 
        for chan in channels: 
            x[:,chan] /= stds[chan]
    return X


def zero_center(*args, **kwargs):
    """Subtracts the mean of arg[0] from the arguments. The expected 
    use case would be `standardize(X_train, X_val, X_test)`.

    **Arguments**

    - ***args** : arbitrary _numpy.ndarray_ datasets
        - An arbitrary number of datasets, each required to have
        the same shape in all but the first axis.
    - **channels** : _int_
        - A list of which channels (assumed to be the second axis)
        to zero center. `None` is interpretted to mean every channel.
    - **copy** : _bool_
        - Whether or not to copy the input arrays before modifying them.

    **Returns**

    - _list_ 
        - A list of the zero-centered arguments.
    """

    channels = kwargs.pop('channels', None)
    copy = kwargs.pop('copy', False)

    if len(kwargs):
        raise TypeError('following kwargs are invalid: {}'.format(kwargs))

    assert len(args) > 0

    # treat channels properly
    if channels is None: 
        channels = np.arange(args[0].shape[1])

    # compute mean of the first argument
    mean = np.mean(args[0], axis=0)

    # copy arguments if requested
    if copy: 
        X = [np.copy(arg) for arg in args]
    else: 
        X = args

    # iterate through arguments and channels
    for x in X: 
        for chan in channels: 
            x[:,chan] -= mean[chan]

    return X
