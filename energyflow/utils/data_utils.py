"""Functions for dealing with datasets. 

URL handling and hashing functions copied from Keras GitHub repo. 
The required license and copyright notice are included below.

-------------------------------------------------------------------------------

COPYRIGHT

All contributions by Francois Chollet:
Copyright (c) 2015 - 2018, Francois Chollet.
All rights reserved.

All contributions by Google:
Copyright (c) 2015 - 2018, Google, Inc.
All rights reserved.

All contributions by Microsoft:
Copyright (c) 2017 - 2018, Microsoft, Inc.
All rights reserved.

All other contributions:
Copyright (c) 2015 - 2018, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import absolute_import, division, print_function

from contextlib import closing
import hashlib
import os
import sys

import numpy as np

from six.moves.urllib.error import HTTPError, URLError

__all__ = ['data_split', 'get_examples', 'get_file', 'pixelate', 'remap_pids', 
           'standardize', 'to_categorical', 'zero_center']

def data_split(*args, **kwargs):
    """A function to split an arbitrary number of arrays into train, 
    validation, and test sets. If val_frac = 0, then we don't split any 
    events into the validation set. If exactly two arguments are given 
    (an "X" and "Y") then we return (X_train, [X_val], X_test, Y_train, 
    [Y_val], Y_test), otherwise i lists corresponding to the different args
    are returned with each list being [train, [val], test]. Note that all 
    arguments must have the same number of samples otherwise an exception
    will be raised.
    """

    # handle valid kwargs
    train, val, test = kwargs.pop('train', -1), kwargs.pop('val', 0.1), kwargs.pop('test', 0.1)
    shuffle = kwargs.pop('shuffle', True)
    if len(kwargs):
        raise TypeError('following kwargs are invalid: {}'.format(kwargs))

    # validity checks
    if len(args) == 0: 
        raise RuntimeError('Need to pass at least one argument to data_split')

    n_samples = len(args[0])
    for arg in args[1:]: 
        assert len(arg) == n_samples, 'args to data_split have different length'

    # determine numbers
    num_val = int(n_samples*val) if val<=1 else val
    num_test = int(n_samples*test) if test <=1 else test
    num_train = n_samples - num_val - num_test if train==-1 else (int(n_samples*train) if train<=1 else train)
    assert num_train >= 0, 'bad parameters: negative num_train'
    assert num_train + num_val + num_test <= n_samples, 'too few samples for requested data split'
    
    # calculate masks 
    perm = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)
    train_mask = perm[:num_train]
    val_mask = perm[-num_val:]
    test_mask = perm[num_train:num_train+num_test]

    # apply masks
    masks = [train_mask, val_mask, test_mask] if num_val > 0 else [train_mask, test_mask]

    # return list of new datasets
    return [arg[mask] for arg in args for mask in masks]

def get_examples(which='all', path='~/.energyflow'):

    all_examples = {'efn_example.py', 'pfn_example.py', 'cnn_example.py', 'dnn_example.py'}
    if which == 'all':
        examples = all_examples
    else:
        if not isinstance(which, (tuple, list)):
            which = [which]
        examples = all_examples.intersection(which)

    base_url = 'https://github.com/pkomiske/EnergyFlow/raw/master/examples/'
    for example in examples:
        fpath = get_file(example, 
                 url=base_url+example,
                 cache_dir=os.path.expanduser(path),
                 cache_subdir='examples')

# PDGid to isCharged dictionary
pid2abschg_mapping = {22: 0,             # photon
                      211: 1, -211: 1,   # pi+-
                      321: 1, -321: 1,   # K+-
                      130: 0,            # K-long
                      2112: 0, -2112: 0, # neutron, anti-neutron
                      2212: 1, -2212: 1, # proton, anti-proton
                      11: 1, -11: 1,     # electron, positron
                      13: 1, -13: 1}     # muon, anti-muon

def pixelate(jet, npix=33, img_width=0.8, nb_chan=1, charged_counts_only=False, norm=True):
    """A function for creating a jet image from a list of particles.

    jet: an array containing the list of particles in a jet with each row 
         representing a particle and the columns being (rapidity, phi, pT, 
         pdgid), the latter not being necessary for a grayscale image.
    npix: number of pixels along one dimension of the image.
    img_width: the image will be size img_width x img_width
    nb_chan: 1 - returns a grayscale image of total pt
             2 - returns a two-channel image with total pt and total counts
    norm: whether to normalize the pT channels to 1 according to the L1 norm
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

# PDGid to small float dictionary
pid2float_mapping = {22: 0, 
                     211: .1, -211: .2, 
                     321: .3, -321: .4, 
                     130: .5, 
                     2112: .6, -2112: .7, 
                     2212: .8, -2212: .9, 
                     11: 1.0, -11: 1.1, 
                     13: 1.2, -13: 1.3}

def remap_pids(events, pid_i=3):
    events_shape = events.shape
    pids = events[:,:,pid_i].astype(int).reshape((events_shape[0]*events_shape[1]))
    events[:,:,pid_i] = np.asarray([pid2float_mapping.get(pid, 0) for pid in pids]).reshape(events_shape[:2])

def standardize(*args, **kwargs):
    """ Normalizes each argument by the standard deviation of the pixels in 
    arg[0]. The expected use case would be standardize(X_train, X_val, X_test).

    channels: which channels to zero_center. The default will lead to all
              channels being affected.
    copy: if True, the arguments are unaffected. if False, the arguments
          themselves may be modified
    reg: used to prevent divide by zero 
    """

    channels = kwargs.pop('channels', [])
    copy = kwargs.pop('copy', False)
    reg = kwargs.pop('reg', 10**-10)

    if len(kwargs):
        raise TypeError('following kwargs are invalid: {}'.format(kwargs))

    assert len(args) > 0

    # treat channels properly
    if len(channels)==0: 
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

def to_categorical(vector, num_classes=None):
    if num_classes is None:
        num_classes = np.max(vector) + 1

    y = np.asarray(vector, dtype=int)
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def zero_center(*args, **kwargs):
    """ Subtracts the mean of arg[0,channels] from the other arguments.
    Assumes that the arguments are numpy arrays. The expected use case would
    be zero_center(X_train, X_val, X_test).

    channels: list of which channels to zero_center. The default will lead to 
              all channels being affected.
    copy: if True, the arguments are unaffected. if False, the arguments
          themselves may be modified
    """

    channels = kwargs.pop('channels', [])
    copy = kwargs.pop('copy', False)

    if len(kwargs):
        raise TypeError('following kwargs are invalid: {}'.format(kwargs))

    assert len(args) > 0

    # treat channels properly
    if len(channels)==0: 
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

# begin code copied from Keras
if sys.version_info[0] == 2:
    from six.moves.urllib.request import urlopen
    def urlretrieve(url, filename, reporthook=None, data=None):
        """Replacement for `urlretrive` for Python 2.
        Under Python 2, `urlretrieve` relies on `FancyURLopener` from legacy
        `urllib` module, known to have issues with proxy management.
        # Arguments
            url: url to retrieve.
            filename: where to store the retrieved data locally.
            reporthook: a hook function that will be called once
                on establishment of the network connection and once
                after each block read thereafter.
                The hook will be passed three arguments;
                a count of blocks transferred so far,
                a block size in bytes, and the total size of the file.
            data: `data` argument passed to `urlopen`.
        """

        def chunk_read(response, chunk_size=8192, reporthook=None):
            content_type = response.info().get('Content-Length')
            total_size = -1
            if content_type is not None:
                total_size = int(content_type.strip())
            count = 0
            while True:
                chunk = response.read(chunk_size)
                count += 1
                if reporthook is not None:
                    reporthook(count, chunk_size, total_size)
                if chunk:
                    yield chunk
                else:
                    break

        with closing(urlopen(url, data)) as response, open(filename, 'wb') as fd:
                for chunk in chunk_read(response, reporthook=reporthook):
                    fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve

def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    """Calculates a file sha256 or md5 hash.
    # Example
    ```python
        >>> from keras.data_utils import _hash_file
        >>> _hash_file('/path/to/file.zip')
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```
    # Arguments
        fpath: path to the file being validated
        algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    # Returns
        The file hash
    """
    if (algorithm is 'sha256') or (algorithm is 'auto' and len(hash) is 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()

def _validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.
    # Arguments
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    # Returns
        Whether the file is valid
    """
    if ((algorithm is 'sha256') or
            (algorithm is 'auto' and len(file_hash) is 64)):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False
# end code copied from Keras

# the following function is closely based on the matching Keras function
def get_file(filename, url, cache_dir=None, cache_subdir='datasets', file_hash=None):

    # cache_dir = None means use default cache
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.energyflow')
    datadir_base = os.path.expanduser(cache_dir)

    # ensure that directory exists
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    # handle case where cache is not writeable
    if not os.access(datadir_base, os.W_OK):
        datadir = os.path.join('/tmp', '.energyflow', cache_subdir)
        if not os.path.exists(datadir):
            os.makedirs(datadir)

    fpath = os.path.join(datadir, filename)

    # determine if file needs to be downloaded
    download = False
    if os.path.exists(fpath):
        if file_hash is not None and not _validate_file(fpath, file_hash, 'sha256'):
            print('local file hash does not match so we will redownload')
            download = True
    else:
        download = True

    if download:
        print('Downloading {} from {} to {}'.format(filename, url, datadir))

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(url, fpath)
            except URLError as e:
                raise Exception(error_msg.format(url, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(url, e.code, e.msg))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

        if file_hash is not None:
            assert _validate_file(fpath, file_hash, 'sha256')

    return fpath
