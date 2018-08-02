"""Functions for dealing with data manipulation. 

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

__all__ = ['data_split', 'get_file', 'to_categorical']

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

def to_categorical(vector, num_classes=None):
    if num_classes is None:
        num_classes = np.max(vector) + 1

    y = np.asarray(vector, dtype=int)
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

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

def hash_file(fpath, algorithm='sha256', chunk_size=65535):
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


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
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

    if str(hash_file(fpath, hasher, chunk_size)) == str(file_hash):
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
    if os.path.exists(fpath) and file_hash is not None:
        if not validate_file(fpath, file_hash, 'sha256'):
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

        assert validate_file(fpath, file_hash, 'sha256')

    return fpath
