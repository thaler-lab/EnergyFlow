"""## Utilities 

Utilities for EnergyFlow architectures, split out from the utils submodule
because these import tensorflow, which the main package avoids doing.
"""

#           _____   _____ _    _          _    _ _______ _____ _       _____
#     /\   |  __ \ / ____| |  | |        | |  | |__   __|_   _| |     / ____|
#    /  \  | |__) | |    | |__| |        | |  | |  | |    | | | |    | (___
#   / /\ \ |  _  /| |    |  __  |        | |  | |  | |    | | | |     \___ \
#  / ____ \| | \ \| |____| |  | | ______ | |__| |  | |   _| |_| |____ ____) |
# /_/    \_\_|  \_\\_____|_|  |_||______| \____/   |_|  |_____|______|_____/

from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod, abstractproperty
import math
import types
import warnings

import numpy as np
import six

from energyflow.utils.random_utils import random

__all__ = [

    # helper functions
    'convert_dtype',
    'pad_events',
    'pair_and_pad_events',
    'product_and_pad_weights',
    'pair_and_pad_weighted_events',
    'pair_3d_array_axis1',
    'product_2d_weights',

    # classes representing point cloud datasets
    'PointCloudDataset',
    'WeightedPointCloudDataset',
    'PairedPointCloudDataset',
    'PairedWeightedPointCloudDataset',
]

def convert_dtype(X, dtype):

    # check for proper argument type
    if not isinstance(X, np.ndarray):
        raise TypeError('argument must be a numpy ndarray')

    # object arrays are special
    if X.dtype == 'O':
        return np.asarray([convert_dtype(x, dtype) for x in X], dtype='O')
    else:
        return X.astype(dtype, copy=False)

def pad_events(X, pad_val=0.):

    # get a rectangular array which will hold the padded events
    lens = [len(x) for x in X]
    if pad_val == 0.:
        output = np.zeros(((len(X), max(lens),) + X[0].shape[1:]), dtype=X[0].dtype)
    else:
        output = np.full(((len(X), max(lens),) + X[0].shape[1:]), pad_val, dtype=X[0].dtype)

    # set events into the array
    for i, (x, lenx) in enumerate(zip(X, lens)):
        output[i,:lenx] = x

    return output

def pair_and_pad_events(X, pad_val=0., max_len=None):

    # get a rectangular array which will hold the padded events
    if max_len is None:
        max_len = max([len(x) for x in X])

    nfeatures = X[0].shape[1]
    two_nfeatures = 2*nfeatures
    if pad_val == 0.:
        output = np.zeros((len(X), max_len*max_len, two_nfeatures), dtype=X[0].dtype)
    else:
        output = np.full((len(X), max_len*max_len, two_nfeatures), pad_val, dtype=X[0].dtype)

    # set events in the array
    for i, x in enumerate(X):
        lenx = len(x)
        paired_shape = (lenx, lenx, nfeatures)
        x0 = np.broadcast_to(x[None,:], paired_shape)
        x1 = np.broadcast_to(x[:,None], paired_shape)
        pairedx = np.concatenate((x0, x1), axis=2).reshape(-1, two_nfeatures)
        output[i,:len(pairedx)] = pairedx

    return output

def product_and_pad_weights(weights, max_len=None):

    # get a rectangular array which will hold the padded events
    if max_len is None:
        max_len = max([len(x) for x in weights])

    # get array to hold result
    output = np.zeros((len(weights), max_len*max_len), dtype=weights[0].dtype)

    # set events in the array
    for i, x in enumerate(weights):
        x_prod = (x[None,:] * x[:,None]).reshape(-1)
        output[i,:len(x_prod)] = x_prod

    return output

def pair_and_pad_weighted_events(weights, X, pad_val=0.):

    # determine the max length
    max_len = max([len(x) for x in X])

    return product_and_pad_weights(weights, max_len), pair_and_pad_events(X, pad_val, max_len)

def pair_3d_array_axis1(X, pad_val=0.):
    max_len = X.shape[1]
    paired_shape = (len(X), max_len, max_len, X.shape[2])
    X0 = np.broadcast_to(X[:,None,:], paired_shape)
    X1 = np.broadcast_to(X[:,:,None], paired_shape)
    return np.concatenate((X0, X1), axis=3).reshape(len(X), -1, 2*X.shape[2])

def product_2d_weights(weights):
    return (weights[:,None,:] * weights[:,:,None]).reshape(len(weights), -1)

class PointCloudDataset(object):

    def __init__(self, data_args, batch_size=100, dtype='float32',
                                  shuffle=True, seed=None, infinite=False):
        """Creates a TensorFlow dataset from NumPy arrays of events of particles,
        designed to be used as input to EFN and PFN models. The function uses a
        generator to spool events from the arrays as needed and pad them on the fly.
        As of EnergyFlow version 1.3.0, it is suggested to use this function to
        create TensorFlow datasets to use as input to EFN and PFN training as it can
        yield a slight improvement in training and evaluation time.


        Here are some examples of using this function. For a standard EFN without
        event weights, one would specify the arrays as:
        ```python
        data_args = [[event_zs, event_phats], Y]
        ```
        For a PFN, let's say with event weights, the arrays look like:
        ```python
        data_args = [event_ps, Y, weights]
        ```
        For an EFN model with global features, we would do:
        ```python
        data_args = [[event_zs, event_phats, X_global], Y]
        ```
        For a test dataset, where there are no target values of weights, it is
        important to use a nested list in the case where there are multiple inputs.
        For instance, for a test dataset for an EFN model, we would have:
        ```python
        data_args = [[test_event_zs, test_event_phats]]
        ```

        **Arguments**

        - **data_args** : {_tuple_, _list_} of _numpy.ndarray_
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

        # store inputs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.dtype = dtype
        self.infinite = infinite
        self._wrap = False

        # check for proper data_args
        if not isinstance(data_args, (list, tuple)):
            data_args = [data_args]
        data_args = list(data_args)

        # wrap lists and tuples in a PointCloudDataset
        self.data_args = []
        for i,data_arg in enumerate(data_args):

            # wrap in a PointCloudDataset
            if isinstance(data_arg, (list, tuple)):
                data_arg = self.__class__(data_arg)

            # set/check length of overall dataset
            if i == 0:
                self._len = len(data_arg)
            elif len(self) != len(data_arg):
                m = 'arguments have different length, {} vs. {}'
                raise IndexError(m.format(len(self), len(data_arg)))

            self.data_args.append(data_arg)

    def __len__(self):
        return self._len

    def __repr__(self):

        # ensure we're initialized
        self._init()

        s = '{}\n'.format(self.__class__.__name__)
        s += '  length - {}\n'.format(len(self))
        s += '  batch_size - {}\n'.format(self.batch_size)
        s += '  batch_dtypes - {}\n'.format(repr(self.batch_dtypes))
        s += '  batch_shapes - {}\n'.format(repr(self.batch_shapes))
        s += '  data_args - {}\n'.format([arg.__class__.__name__ for arg in self.data_args])

        return s

    def wrap(self):
        self._wrap = True
        return self

    @property
    def _state(self):
        return (self.batch_size, self.shuffle, self.seed, self.dtype, self.infinite)

    @_state.setter
    def _state(self, state):
        self.batch_size, self.shuffle, self.seed, self.dtype, self.infinite = state

    @property
    def batch_dtypes(self):
        if hasattr(self, '_batch_dtypes'):

            # unpack single element lists and convert rest to tuple
            bdt = self._batch_dtypes[0] if len(self._batch_dtypes) == 1 else tuple(self._batch_dtypes)
            return (bdt,) if self._wrap else bdt

    @property
    def batch_shapes(self):
        if hasattr(self, '_batch_shapes'):

            # unpack single element lists and convert rest to tuple
            bs = self._batch_shapes[0] if len(self._batch_shapes) == 1 else tuple(self._batch_shapes)
            return (bs,) if self._wrap else bs

    @property
    def steps_per_epoch(self):
        return math.ceil(len(self)/self.batch_size)

    def _check_compatibility(self, other):
        assert isinstance(other, PointCloudDataset), 'other is not instance of PointCloudDataset'
        if self.__class__ != PointCloudDataset:
            raise TypeError('{} should not contain other instances of PointCloudDataset'.format(self.__class__))
        if len(self) != len(other):
            m = 'arguments have different length, {} vs. {}'
            raise IndexError(m.format(len(self), len(other)))
        if self.dtype != other.dtype:
            raise ValueError('inconsistent dtypes')
        if self.batch_size != other.batch_size:
            raise ValueError('inconsistent batch_sizes')
        if self.shuffle != other.shuffle:
            raise ValueError('inconsistent shuffling')
        if self.seed != other.seed:
            warnings.warn('seeds do not match')
        if self.infinite != other.infinite:
            raise ValueError('inconsistent setting for infinite')

    # function to enable lazy init
    def _init(self, state=None):
        import tensorflow as tf

        # ensure consistent state (randomness in particular)
        if state is not None:
            self.toplevel = False
            self._state = state
        else:
            self.toplevel = True
        
        # determine if we need to get a new rng
        if self.shuffle and self.seed is None:
            self.seed = random._bit_generator._seed_seq.spawn(1)[0]
        self._rng = np.random.default_rng(self.seed) if self.shuffle else random

        # initialize subcomponents of this dataset
        for arg in self.data_args:
            if isinstance(arg, PointCloudDataset):
                arg._init(self._state)

        # process data_args
        self._batch_dtypes, self._batch_shapes, self._zero_pad = [], [], []
        self.tensor_batch_size = self.batch_size if self.steps_per_epoch*self.batch_size == len(self) else None
        for i,data_arg in enumerate(self.data_args):

            # handle PointCloudDataset
            if isinstance(data_arg, PointCloudDataset):
                self._check_compatibility(data_arg)
                self._batch_dtypes.append(data_arg.batch_dtypes)
                self._batch_shapes.append(data_arg.batch_shapes)

            # tf dataset not handled yet
            elif isinstance(data_arg, tf.data.Dataset):
                print(data_arg.element_spec)
                raise TypeError('tensorflow dataset not supported here yet')

            # ensure numpy array
            elif not isinstance(data_arg, np.ndarray):
                raise TypeError('expected numpy array and got {}'.format(type(data_arg)))

            # numpy array
            else:

                # object array
                if data_arg.ndim == 1 and len(data_arg) and isinstance(data_arg[0], np.ndarray):
                    if data_arg[0].ndim == 2:
                        self._batch_shapes.append((self.tensor_batch_size, None, data_arg[0].shape[1]))
                    elif data_arg[0].ndim == 1:
                        self._batch_shapes.append((self.tensor_batch_size, None,))
                    else:
                        raise IndexError('array dimensions not understood')

                    # mark this data_arg for zero padding
                    self._zero_pad.append(i)

                # rectangular array
                else:
                    self._batch_shapes.append((self.tensor_batch_size,) + data_arg.shape[1:])    

                self._batch_dtypes.append(self.dtype)
                self.data_args[i] = convert_dtype(data_arg, self.dtype)

    def as_tf_dataset(self, prefetch=4):

        # ensure we're initialized
        self._init()

        # get tensorflow dataset from generator
        import tensorflow as tf
        tfds = tf.data.Dataset.from_generator(self.get_batch_generator(),
                                              self.batch_dtypes,
                                              output_shapes=self.batch_shapes)

        # set prefetch amount
        if prefetch:
            tfds = tfds.prefetch(prefetch)

        return tfds

    def batch_callback(self, args):
        return args

    def _construct_batch(self, args):
        for z in self._zero_pad:
            args[z] = pad_events(args[z])

        # apply callback to batch
        batch = self.batch_callback(args)

        # unpack single element lists
        ret = batch[0] if len(batch) == 1 else tuple(batch)
        return (ret,) if self._wrap else ret

    def get_batch_generator(self):
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

        # these quantities don't change inbetween instantiations of the generator
        gens = [isinstance(arg, PointCloudDataset) for arg in self.data_args]
        self.batch_inds = [(i*self.batch_size, min((i+1)*self.batch_size, len(self)))
                           for i in range(self.steps_per_epoch)]

        def batch_generator():

            # get generators anew, in case infinite=False and we have subgenerators
            data_args = [arg.get_batch_generator()() if g else arg
                         for arg,g in zip(self.data_args, gens)]
            
            # loop over epochs
            arr_func = lambda arg, start, end: arg[start:end]
            while True:

                # get a new permutation each epoch
                if self.shuffle:
                    perm = self._rng.permutation(len(self))
                    arr_func = lambda arg, start, end: arg[perm[start:end]]

                # special case 1 (for speed)
                if len(data_args) == 1:
                    arg0 = data_args[0]
                    if gens[0]:
                        for _ in self.batch_inds:
                            yield self._construct_batch([next(arg0)])
                    else:
                        for start, end in self.batch_inds:
                            yield self._construct_batch([arr_func(arg0, start, end)])

                # special case 2 (for speed)
                elif len(data_args) == 2:
                    arg0, arg1 = data_args
                    if gens[0]:
                        if gens[1]:
                            for _ in self.batch_inds:
                                yield self._construct_batch([next(arg0), next(arg1)])
                        else:
                            for start, end in self.batch_inds:
                                yield self._construct_batch([next(arg0), arr_func(arg1, start, end)])
                    else:
                        if gens[1]:
                            for start, end in self.batch_inds:
                                yield self._construct_batch([arr_func(arg0, start, end), next(arg1)])
                        else:
                            for start, end in self.batch_inds:
                                yield self._construct_batch([arr_func(arg0, start, end), arr_func(arg1, start, end)])

                # general case
                else:
                    for start, end in self.batch_inds:
                        yield self._construct_batch([next(arg) if g else arr_func(arg, start, end)
                                                     for arg, g in zip(data_args, gens)])

                # consider ending iteration
                if not self.infinite:
                    break

        return batch_generator

class WeightedPointCloudDataset(PointCloudDataset):

    def _init(self, state=None):
        super()._init(state)
        self._assume_padded = True

        # duplicate batch dtypes
        self._batch_dtypes = [bdtype for bdtype in self._batch_dtypes for i in range(2)]

        # modify batch shapes
        batch_shapes = []
        for batch_shape in self._batch_shapes:
            batch_shapes.extend([batch_shape[:2], batch_shape[:2] + (batch_shape[2]-1,)])
        self._batch_shapes = batch_shapes

    def batch_callback(self, args):
        weighted_args = []
        if self._assume_padded:
            for arg in args:
                weighted_args.extend([arg[:,:,0], arg[:,:,1:]])
        else:
            for arg in args:
                weighted_args.extend([[x[:,0] for x in arg], [x[:,1:] for x in arg]])

        return weighted_args

class PairedPointCloudDataset(PointCloudDataset):

    def _init(self, state=None):
        super()._init(state)

        self._paired_args_need_padding = len(self.data_args)*[False]
        for i in range(len(self.data_args)):

            # add to our own list
            if i in self._zero_pad:
                self._paired_args_need_padding[i] = True

            # get new shape resulting from pairing
            self._update_shape(i)

        # zero padding handled separately
        self._zero_pad.clear()

    def _update_shape(self, i):
        self._batch_shapes[i] = self._batch_shapes[i][:2] + (2*self._batch_shapes[i][2],)

    # pair objects in a point cloud dataset
    def batch_callback(self, args):
        return [pair_and_pad_events(arg) if need_padding else pair_3d_array_axis1(arg)
                for arg, need_padding in zip(args, self._paired_args_need_padding)]

class PairedWeightedPointCloudDataset(PairedPointCloudDataset, WeightedPointCloudDataset):

    def _init(self, state=None):
        super()._init(state)
        self._assume_padded = False

    # update shape for features only, not weights
    def _update_shape(self, i):
        j = 2*i + 1
        self._batch_shapes[j] = self._batch_shapes[j][:2] + (2*self._batch_shapes[j][2],)

    def batch_callback(self, args):

        # run args through weighted callback first, ignoring PairedPointCloudDataset.batch_callback
        args = super(PairedPointCloudDataset, self).batch_callback(args)

        paired_weighted_args = []
        for i, need_padding in enumerate(self._paired_args_need_padding):
            weights, features = args[2*i], args[2*i+1]
            if need_padding:
                paired_weighted_args.extend(pair_and_pad_weighted_events(weights, features))
            else:
                paired_weighted_args.extend([product_2d_weights(weights), pair_3d_array_axis1(features)])

        return paired_weighted_args
