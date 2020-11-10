"""## Utilities 

Utilities for EnergyFlow architectures, split out from the utils submodule
because these import tensorflow, which the main package avoids doing.
"""

from __future__ import absolute_import, division, print_function

import types
import warnings

import numpy as np
import tensorflow as tf

from energyflow.utils import iter_or_rep

__all__ = [
    'tf_point_cloud_dataset',
    'to_generator'
]

# tf_point_cloud_dataset(data_arrs, batch_size=None, dtype='float32')
def tf_point_cloud_dataset(data_arrs, batch_size=100, dtype='float32',
                                      prefetch=10, pad_val=0.,
                                      generator_shapes=None, _xyweights=True):
    """Creates a TensorFlow dataset from NumPy arrays of events of particles,
    designed to be used as input to EFN and PFN models. The function uses a
    generator to spool events from the arrays as needed and pad them on the fly.
    As of EnergyFlow version 1.3.0, it is suggested to use this function to
    create TensorFlow datasets to use as input to EFN and PFN training as it can
    yield a slight improvement in training and evaluation time.


    Here are some examples of using this function. For a standard EFN without
    event weights, one would specify the arrays as:
    ```python
    data_arrs = [[event_zs, event_phats], Y]
    ```
    For a PFN, let's say with event weights, the arrays look like:
    ```python
    data_arrs = [event_ps, Y, weights]
    ```
    For an EFN model with global features, we would do:
    ```python
    data_arrs = [[event_zs, event_phats, X_global], Y]
    ```
    For a test dataset, where there are no target values of weights, it is
    important to use a nested list in the case where there are multiple inputs.
    For instance, for a test dataset for an EFN model, we would have:
    ```python
    data_arrs = [[test_event_zs, test_event_phats]]
    ```

    **Arguments**

    - **data_arrs** : {_tuple_, _list_} of _numpy.ndarray_
        - The NumPy arrays to build the dataset from. A single array may be
        given, in which case samples from it alone are used. If a list of arrays
        are given, then it should be length 1, 2, or 3, corresponding to the
        `(X,)`, `(X, Y)`, or `(X, Y, weights)`, respectively, where `X` are the
        features, `Y` are the targets and `weights` are the sample weights. In
        the case where multiple inputs or multiple target arrays are to be used,
        a nested list may be used, (see above).
    - **batch_size** : _int_ or `None`
        - If an integer, the dataset will provide batches with that number of
        events when queried. If `None`, no batching is done. Setting this option
        should replace padding a `batch_size` argument directly to the `.fit`
        method of the EFN or PFN.
    - **dtype** : _str_
        - The datatype to use in the TensorFlow dataset. Note that 32-bit
        floats are typically sufficient for ML models and so this is the
        default.
    - **prefetch** : _int_
        - The maximum number of samples to prepare in advance of their usage
        during training or evaluation. See the [TensorFlow documentation](https:
        //www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch) for more
        details.
    - **pad_val** : _float_
        - Events will be padded with particles consisting of this value repeated
        as many times as necessary. This should match the `mask_val` option of
        the EFN or PFN model.

    **Returns**

    - _tensorflow.data.Dataset_
        - The TensorFlow dataset built from the provided arrays and options. To
        view samples from the dataset, for instance the first five batches, one
        can do:

    ```python
    for sample in tfdataset.take(5).as_numpy_iterator():
        print(sample.shape)
    ```
    """

    # handle generator shapes
    generator_shapes = iter_or_rep(generator_shapes)

    # check for proper data_arrs
    if not isinstance(data_arrs, (list, tuple)):
        data_arrs = [data_arrs]
    data_arrs = list(data_arrs)

    # check if this is a top-level call (i.e. xyweights is True)
    need_padding, nx = False, 1
    if _xyweights:

        # check for proper length
        if len(data_arrs) not in {1, 2, 3}:
            raise ValueError("'data_arrs' should be length 1, 2, or 3 if _xyweights is True")

        # if data_arrs[i] is a list or tuple, process it further
        for i, gen_shapes in zip(range(len(data_arrs)), generator_shapes):
            if isinstance(data_arrs[i], (list, tuple)):
                nx = len(data_arrs[i])
                data_arrs[i], need_padding_i = tf_point_cloud_dataset(data_arrs[i], batch_size=None,
                                                                      prefetch=None, dtype=dtype,
                                                                      generator_shapes=gen_shapes,
                                                                      _xyweights=False)
                need_padding |= need_padding_i

    # process each dataset
    tfds, arr_len = [], None
    for arr, gen_shape in zip(data_arrs, generator_shapes):
        try:
            # skip if already a dataset
            if isinstance(arr, tf.data.Dataset):
                tfds.append(arr)
                continue

            # handle the case of a generator
            elif isinstance(arr, types.GeneratorType):
                tfds.append(tf_dataset_from_generator(arr, output_shapes=gen_shape, dtype=dtype))
                continue

            # ensure we have a numpy array with the right dtype
            arr = convert_dtype(arr, dtype)

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

                # if a generator_shape was specified, ensure it matches
                if gen_shape is not None and gen_shape != tfd_shape:
                    raise ValueError('improper generator_shape for array - use None for non-generator input')

                tfds.append(tf_dataset_from_generator(arr, tfd_shape, dtype))
                need_padding = True

            # form dataset from array
            else:
                tfds.append(tf.data.Dataset.from_tensor_slices(arr))

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
            tfds = tfds.padded_batch(batch_size, padding_values=float(pad_val))
        else:
            tfds = tfds.batch(batch_size)

    # set prefetch amount
    if prefetch:
        tfds = tfds.prefetch(prefetch)

    # ensure that the need for padding is communicated internally
    if not _xyweights:
        return tfds, need_padding

    return tfds

def convert_dtype(X, dtype):

    # check for proper argument type
    if not isinstance(X, np.ndarray):
        raise TypeError("argument 'X' must be a numpy ndarray")

    # object arrays are special
    if X.dtype == np.dtype('O'):
        return np.asarray([convert_dtype(x, dtype) for x in X], dtype='O')
    else:
        return X.astype(dtype, copy=False)

def to_generator(*args, batch_size, seed=None, infinite=False):
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

    def generator():

        len_args = len(args)

        # no shuffling required
        if seed is None:

            # loop over epochs
            while True:

                # loop over dataset
                for it in (iter(args[0]) if len_args == 1 else zip(*args)):
                    yield it

                # consider ending iteration
                if not infinite:
                    return

        # get rng from seed
        if isinstance(seed, str):
            seed = int(seed)
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        
        # loop over epochs
        while True:

            # get a new permutation each epoch
            perm = rng.permutation(len(args[0]))

            # special case 1
            if len_args == 1:
                arg0 = args[0]
                for p in perm:
                    yield arg0[p]

            # special case 2
            elif len_args == 2:
                arg0, arg1 = args
                for p in perm:
                    yield (arg0[p], arg1[p])

            # special case 3
            elif len_args == 3:
                arg0, arg1, arg2 = args
                for p in perm:
                    yield (arg0[p], arg1[p], arg2[p])

            # general case
            else:
                for p in perm:
                    yield tuple(arg[p] for arg in args)

            # consider ending iteration
            if not infinite:
                return

    return generator

def tf_dataset_from_generator(X, output_shapes=None, dtype='float32', args=None):

    if not isinstance(X, types.GeneratorType):
        X = to_generator(convert_dtype(X, dtype))

    return tf.data.Dataset.from_generator(X, tf.as_dtype(dtype),
                                          output_shapes=output_shapes,
                                          args=args)

