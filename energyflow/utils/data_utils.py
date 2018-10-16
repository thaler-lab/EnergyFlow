"""### Data Tools

Functions for dealing with datasets. These are not importable from
the top level `energyflow` module, but must instead be imported 
from `energyflow.utils`.
"""
from __future__ import absolute_import, division, print_function

from contextlib import closing
import hashlib
import os
import sys

import numpy as np
from six.moves.urllib.error import HTTPError, URLError

__all__ = [
    'get_examples', 
    'data_split', 
    'to_categorical',
    'remap_pids'
]

def get_examples(path='~/.energyflow', which='all', overwrite=False):
    """Pulls examples from GitHub. To ensure availability of all examples
    update EnergyFlow to the latest version.

    **Arguments**

    - **path** : _str_
        - The destination for the downloaded files.
    - **which** : {_list_, `'all'`}
        - List of examples to download, or the string `'all'` in which 
        case all the available examples are downloaded.
    - **overwrite** : _bool_
        - Whether to overwrite existing files or not.
    """

    # all current examples 
    all_examples = {
        'efn_example.py',
        'pfn_example.py',
        'cnn_example.py',
        'dnn_example.py',
        'efp_example.py',
    }

    # process which examples are selected
    if which == 'all':
        examples = all_examples
    else:
        if not isinstance(which, (tuple, list)):
            which = [which]
        examples = all_examples.intersection(which)

    base_url = 'https://github.com/pkomiske/EnergyFlow/raw/master/examples/'
    cache_dir = os.path.expanduser(path)

    # get each example
    files = []
    for example in examples:

        # remove file if necessary
        file_path = os.path.join(cache_dir, 'examples', example)
        if overwrite and os.path.exists(file_path):
            os.remove(file_path)

        files.append(_get_file(example, url=base_url+example, 
                                        cache_dir=cache_dir, cache_subdir='examples'))

    # print summary
    print()
    print('Summary of examples:')
    for f in files:
        path, fname = os.path.split(f)
        print(fname, 'exists at', path)
    print()


# data_split(*args, train=-1, val=0.0, test=0.1, shuffle=True)
def data_split(*args, **kwargs):
    """A function to split a dataset into train, test, and optionally 
    validation datasets.

    **Arguments**

    - ***args** : arbitrary _numpy.ndarray_ datasets
        - An arbitrary number of datasets, each required to have
        the same number of elements, as numpy arrays.
    - **train** : {_int_, _float_}
        - If a float, the fraction of elements to include in the training
        set. If an integer, the number of elements to include in the
        training set. The value `-1` is special and means include the
        remaining part of the dataset in the training dataset after
        the test and (optionally) val parts have been removed
    - **val** : {_int_, _float_}
        - If a float, the fraction of elements to include in the validation
        set. If an integer, the number of elements to include in the
        validation set. The value `0` is special and means do not form
        a validation set.
    - **test** : {_int_, _float_}
        - If a float, the fraction of elements to include in the test
        set. If an integer, the number of elements to include in the
        test set.
    - **shuffle** : _bool_
        - A flag to control whether the dataset is shuffle prior to 
        being split into parts.

    **Returns**

    - _list_
        - A list of the split datasets in train, [val], test order. If 
        datasets `X`, `Y`, and `Z` were given as `args` (and assuming a
        non-zero `val`), then [`X_train`, `X_val`, `X_test`, `Y_train`, 
        `Y_val`, `Y_test`, `Z_train`, `Z_val`, `Z_test`] will be returned.
    """

    # handle valid kwargs
    train, val, test = kwargs.pop('train', -1), kwargs.pop('val', 0.0), kwargs.pop('test', 0.1)
    shuffle = kwargs.pop('shuffle', True)
    if len(kwargs):
        raise TypeError('following kwargs are invalid: {}'.format(kwargs))

    # validity checks
    if len(args) == 0: 
        raise RuntimeError('Need to pass at least one argument to data_split')

    # check for consistent length
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


def to_categorical(labels, num_classes=None):
    """One-hot encodes class labels.

    **Arguments**

    - **labels** : _1-d numpy.ndarray_
        - Labels in the range `[0,num_classes)`.
    - **num_classes** : {_int_, `None`}
        - The total number of classes. If `None`, taken to be the 
        maximum label plus one.

    **Returns**

    - _2-d numpy.ndarray_
        - The one-hot encoded labels.
    """

    # get num_classes from max label if None
    if num_classes is None:
        num_classes = np.max(labels) + 1

    y = np.asarray(labels, dtype=int)
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))

    # index into array and set appropriate values to 1
    categorical[np.arange(n), y] = 1

    return categorical


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
    """Remaps PDG id numbers to small floats for use in a neural network.
    `events` are modified in place and nothing is returned.

    **Arguments**

    - **events** : _3-d numpy.ndarray_
        - The events as an array of arrays of particles.
    - **pid_i** : _int_
        - The index corresponding to pid information along the last 
        axis of `events`.
    """

    events_shape = events.shape
    pids = events[:,:,pid_i].astype(int).reshape((events_shape[0]*events_shape[1]))
    events[:,:,pid_i] = np.asarray([pid2float_mapping.get(pid, 0) for pid in pids]).reshape(events_shape[:2])


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
def _get_file(filename, url, cache_dir=None, cache_subdir='datasets', file_hash=None):
    """Pulls file from the internet."""

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
