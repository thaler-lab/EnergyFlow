"""Utilities for EnergyFlow architectures, split out from the utils submodule
because these import tensorflow, which the main package avoids doing."""

from __future__ import absolute_import, division, print_function

import warnings

import numpy as np
import tensorflow as tf

__all__ = [
    'tf_point_cloud_dataset',
    'tf_gen'
]

def tf_gen(*args):
    """Returns a function that when called returns a generator that yields
    samples from the given arrays. Designed to work with
    `tf.data.Dataset.from_generator`, though commonly this is handled by
    [`tf_point_cloud_dataset`](#tf_point_cloud_dataset).

    **Arguments**

    - ***args** : arbitrary _numpy.ndarray_ datasets
        - An arbitrary number of arrays.

    **Returns**

    - _function_
        - A function that when called returns a generator that yields samples
        from the given arrays.
    """

    def gen_func():
        if len(args) == 1:
            return iter(args[0])
        return zip(*args)

    return gen_func

# tf_point_cloud_dataset(data_arrs, batch_size=None, dtype='float32')
def tf_point_cloud_dataset(data_arrs, batch_size=None, dtype='float32', prefetch=5, xyweights=True):
    """"""

    # check for proper data_arrs
    if not isinstance(data_arrs, (list, tuple)):
        data_arrs = [data_arrs]
        #raise TypeError("'data_arrs' must be a list or tuple of arrays")
    data_arrs = list(data_arrs)

    # check if this is a top-level call (i.e. xyweights is True)
    need_padding, nx = False, 1
    if xyweights is True:

        # check for proper length
        if len(data_arrs) not in {1, 2, 3}:
            raise ValueError("'data_arrs' should be length 1, 2, or 3 if xyweights is True")

        # make data_arrs[0] a list or tuple
        if isinstance(data_arrs[0], (list, tuple)):
            nx = len(data_arrs[0])
            data_arrs[0], need_padding = tf_point_cloud_dataset(data_arrs[0], dtype=dtype, xyweights='internal')

    # process each dataset
    tfds, arr_len = [], None
    for arr in data_arrs:
        try:
            # skip if already a dataset
            if isinstance(arr, tf.data.Dataset):
                tfds.append(arr)
                continue
            else:
                assert isinstance(arr, np.ndarray), 'array must be a numpy array'

            # check size of array
            if arr_len is None:
                arr_len = len(arr)
            else:
                assert len(arr) == arr_len, 'lengths of arrays do not match'

            # we have an array of arrays here
            if arr.ndim == 1 and isinstance(arr[0], np.ndarray):
                if arr[0].ndim == 2:
                    tfd_shape = (None, arr[0].shape[1])
                elif arr[0].ndim == 1:
                    tfd_shape = (None,)
                else:
                    raise IndexError('array dimensions not understood')

                tfds.append(tf.data.Dataset.from_generator(tf_gen(arr), tf.as_dtype(dtype), tfd_shape))
                need_padding = True

            # form dataset from array
            else:
                tfds.append(tf.data.Dataset.from_tensor_slices(np.asarray(arr, dtype=dtype)))

        except Exception as e:
            e.args = ('cannot properly form tensorflow dataset - ' + e.args[0],) + e.args[1:]
            raise e

    # zip datasets if needed
    if len(tfds) > 1 or nx > 1:
        tfds = tf.data.Dataset.zip(tuple(tfds))
    else:
        tfds = tfds[0]

    # set batch size
    if batch_size:
        if need_padding:
            tfds = tfds.padded_batch(batch_size)
        else:
            tfds = tfds.batch(batch_size)

        # set prefetch amount
        if prefetch:
            tfds = tfds.prefetch(prefetch)

    # ensure that the need for padding is communicated internally
    if xyweights == 'internal':
        return tfds, need_padding
    return tfds
