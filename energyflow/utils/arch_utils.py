"""## Architecture Utils

Utilities for EnergyFlow architectures. These are available in both the
`energyflow.archs` and `energyflow.utils` submodules.
"""

#           _____   _____ _    _          _    _ _______ _____ _       _____
#     /\   |  __ \ / ____| |  | |        | |  | |__   __|_   _| |     / ____|
#    /  \  | |__) | |    | |__| |        | |  | |  | |    | | | |    | (___
#   / /\ \ |  _  /| |    |  __  |        | |  | |  | |    | | | |     \___ \
#  / ____ \| | \ \| |____| |  | | ______ | |__| |  | |   _| |_| |____ ____) |
# /_/    \_\_|  \_\\_____|_|  |_||______| \____/   |_|  |_____|______|_____/

from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
import math
import types
import warnings

import numpy as np
import six

from energyflow.utils.data_utils import convert_dtype, pad_events
from energyflow.utils.random_utils import random

__all__ = [

    # classes representing point cloud datasets
    'PointCloudDataset',
    'WeightedPointCloudDataset',
    'PairedPointCloudDataset',
    'PairedWeightedPointCloudDataset',

    # classes to help with pairing features
    'PairedFeatureCombiner',
    'ConcatenatePairer',
    'ParticleDistancePairer'
]

class PointCloudDataset(object):
    """"""

    def __init__(self, data_args, batch_size=100, dtype='float32',
                                   shuffle=True, seed=None, pad_val=0.,
                                   _wrap=False, _enumerate=False):
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

        # public options
        self.batch_size = batch_size
        self.dtype = dtype
        self.shuffle = shuffle
        self.seed = seed
        self.pad_val = pad_val

        # private options
        self._wrap = _wrap
        self._enumerate = _enumerate

        # check for proper data_args
        if not isinstance(data_args, (list, tuple)):
            data_args = [data_args]
        data_args = list(data_args)

        # wrap lists and tuples in a PointCloudDataset
        self.data_args = []
        for i,data_arg in enumerate(data_args):

            # consider sublists
            if isinstance(data_arg, (list, tuple)):

                # unpack single argument lists
                if len(data_arg) == 1:
                    data_arg = data_arg[0]
                else:
                    data_arg = self.__class__(data_arg)

            # set/check length of overall dataset
            if i == 0:
                self._len = len(data_arg)
            elif len(self) != len(data_arg):
                m = 'arguments have different length, {} vs. {}'
                raise IndexError(m.format(len(self), len(data_arg)))

            self.data_args.append(data_arg)

        if self._enumerate:
            self.enumerate(check_for_repeat=False)

    # allows len() to be used with this class
    def __len__(self):
        return getattr(self, '_len', 0)

    # makes class nicely printable
    def __repr__(self):

        # ensure we're initialized
        self._init()

        s = '{}\n'.format(self.__class__.__name__)
        s += '  length: {}\n'.format(len(self))
        s += '  batch_size: {}\n'.format(self.batch_size)
        if callable(self.shuffle):
            s += '  shuffle: custom\n'
        else:
            s += '  shuffle: {}\n'.format(bool(self.shuffle))
        s += '  seed: {}\n'.format(self.seed)
        s += '  batch_dtypes: {}\n'.format(repr(self.batch_dtypes))
        s += '  batch_shapes: {}\n'.format(repr(self.batch_shapes))
        s += '  data_args:\n'
        for arg in self.data_args:
            if isinstance(arg, PointCloudDataset):
                s += ('    - ' + repr(arg)[:-1]).replace('\n', '\n      ')
            elif isinstance(arg, np.ndarray):
                s += '    - numpy.ndarray | {} | {}'.format(arg.dtype, arg.shape)
            else:
                s += '    - {}'.format(arg.__class__.__name__)
            s += '\n'

        return s

    # turns on wrapping
    def wrap(self):
        self._wrap = True
        return self

    # turns off wrapping
    def unwrap(self):
        self._wrap = False
        return self

    # appends index array as final data arg
    def enumerate(self, check_for_repeat=True):
        if check_for_repeat and self._enumerate:
            raise RuntimeError('`enumerate` already called, cannot be called twice')

        self._enumerate = True
        self.data_args.append(np.arange(len(self)))

        return self

    def split(self, split_arg):

        # check split arg
        if callable(split_arg):
            split_func = split_arg
        elif not isinstance(split_arg, np.ndarray):
            raise ValueError('`split_arg` should be callable or a numpy array')
        else:
            split_func = lambda x: x[split_arg]

        # split data_args
        new_data_args = []
        for data_arg in self.data_args:

            # handle PointCloudDataset
            if isinstance(data_arg, PointCloudDataset):
                new_data_args.append(data_arg.split(split_func))

            # numpy array
            else:
                new_data_args.append(split_func(data_arg))

        # create new object from clone of current one
        return self.join(new_data_args, state=self._state, **self._join_kwargs)

    # note that the settings of the primary dataset will be used for the new one
    def chain(self, other, chain_method='concat'):

        # check chaining method
        if isinstance(chain_method, six.string_types):
            if chain_method != 'concat':
                raise ValueError('unrecognized chain_method `{}`'.format(chain_method))
            chain_method = lambda x, y: np.concatenate((x, y))
        elif not callable(chain_method):
            raise ValueError("`chain_method` should be 'concat' or a callable")

        self._init()
        other._init()

        # chain data_args
        new_data_args = []
        for data_arg, other_data_arg in zip(self.data_args, other.data_args):

            # check that types match
            assert type(data_arg) == type(other_data_arg), 'cannot chain incompatible types'

            # handle PointCloudDataset
            if isinstance(data_arg, PointCloudDataset):

                # check some basic compatibility
                batch_shapes_match = self._check_batch_compatibility(data_arg.batch_shapes, other_data_arg.batch_shapes)
                assert data_arg.batch_dtypes == other_data_arg.batch_dtypes, 'batch_dtypes must match'
                assert batch_shapes_match, 'batch_shapes must match'

                # chain datasets
                new_data_args.append(data_arg.chain(other_data_arg, chain_method))

            # numpy array
            else:
                new_data_args.append(chain_method(data_arg, other_data_arg))

        return self.join(new_data_args, state=self._state, **self._join_kwargs)

    @classmethod
    def join(cls, args, state=None, transfer_state=True, **kwargs):

        new_dset = cls(args, **kwargs)
        if state is None and transfer_state:
            
            # finds first available state to use
            for arg in args:
                if isinstance(arg, PointCloudDataset):
                    state = arg._state
                    break

        # this uses explicitly provided state first, then a transferred one if it exists, finally None
        new_dset._state = state

        return new_dset

    @property
    def _state(self):
        return (self.batch_size, self.shuffle, self.seed, self.dtype, self.pad_val)

    @_state.setter
    def _state(self, state):
        if state is not None:
            (self.batch_size, self.shuffle, self.seed, self.dtype, self.pad_val) = state

    # we don't want _enumerate here since that could cause double enumeration
    @property
    def _join_kwargs(self):
        return {'_wrap': self._wrap}

    # checks if batch shapes are compatible (None matches anything)
    @staticmethod
    def _check_batch_compatibility(batch_shapes, other_batch_shapes):

        # they don't match if they're different lengths
        if len(batch_shapes) != len(other_batch_shapes):
            return False

        # check each element in the shape
        for bs1, bs2 in zip(batch_shapes, other_batch_shapes):

            # check for compatibility of two shape tuples
            assert isinstance(bs1, tuple) and isinstance(bs2, tuple), 'expected tuples'

            # check lengths
            if len(bs1) != len(bs2_):
                return False

            # detect if these contain any nested structures
            if any([not isinstance(b, (None, int)) for bs in [bs1, bs2] for b in bs]):
                return PointCloudDataset._check_batch_compatibility(bs1, bs2)

            # we have tuples of int/None here
            return ((bs1[1:] == bs2[1:]) and ((bs1[0] == bs2[0]) or (bs1[0] is None) or (bs2[0] is None)))

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

    @property
    def rng(self):
        return getattr(self, '_rng', None)

    def _check_option_compatibility(self, other):
        assert isinstance(other, PointCloudDataset), 'other is not instance of PointCloudDataset'
        if self.__class__ != PointCloudDataset:
            raise TypeError('{} should not contain other instances of PointCloudDataset'.format(self.__class__))
        if len(self) != len(other):
            raise IndexError('arguments have different length, {} vs. {}'.format(len(self), len(other)))
        if self.dtype != other.dtype:
            raise ValueError('inconsistent dtypes')
        if self.batch_size != other.batch_size:
            raise ValueError('inconsistent batch_sizes')
        if self.shuffle != other.shuffle:
            raise ValueError('inconsistent shuffling')
        if self.seed != other.seed:
            raise ValueError('seeds do not match')

    # function to enable lazy init
    def _init(self, state=None, final_init=False):
        import tensorflow as tf

        # ensure consistent state (randomness in particular)
        self._state = state
        
        # determine if we need to get a new rng
        if final_init:
            if self.shuffle and self.seed is None:
                self.seed = random.bit_generator._seed_seq.spawn(1)[0]
            self._rng = np.random.default_rng(self.seed) if self.shuffle else random

        # initialize subcomponents of this dataset
        for arg in self.data_args:
            if isinstance(arg, PointCloudDataset):
                arg._init(state=self._state, final_init=final_init)

        # process data_args
        self._batch_dtypes, self._batch_shapes, self._zero_pad = [], [], []
        self.tensor_batch_size = self.batch_size if self.steps_per_epoch*self.batch_size == len(self) else None
        for i,data_arg in enumerate(self.data_args):

            # handle PointCloudDataset
            if isinstance(data_arg, PointCloudDataset):
                self._check_option_compatibility(data_arg)
                self._batch_dtypes.append(data_arg.batch_dtypes)
                self._batch_shapes.append(data_arg.batch_shapes)

            # tf dataset not handled yet
            elif isinstance(data_arg, tf.data.Dataset):
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
                self.data_args[i] = (convert_dtype(data_arg, getattr(np, self.dtype)) 
                                     if final_init else data_arg)

    # method to convert PointCloudDataset to a Tensorflow dataset
    def as_tf_dataset(self, prefetch=None, shuffle_override=None):

        prev_shuffle = self.shuffle
        if shuffle_override is not None:
            self.shuffle = shuffle_override

        # ensure we're initialized
        self._init(final_init=True)

        # get tensorflow dataset from generator
        import tensorflow as tf
        tfds = tf.data.Dataset.from_generator(self.get_batch_generator(),
                                              self.batch_dtypes,
                                              output_shapes=self.batch_shapes)

        # set prefetch amount
        if prefetch is None:
            prefetch = 4
        if prefetch:
            tfds = tfds.prefetch(prefetch)

        # restore shuffle
        self.shuffle = prev_shuffle

        return tfds

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

        # use shuffle at time of calling get_batch_generator
        shuffle = self.shuffle
        def batch_generator():

            # get generators anew, in case infinite=False and we have subgenerators
            # prefetch casues generators to run over, but validation resets generator between epochs
            data_args = [arg.get_batch_generator()() if g else arg
                         for arg,g in zip(self.data_args, gens)]
            
            # loop over epochs
            arr_func = lambda arg, start, end: arg[start:end]
            while True:

                # get a new permutation each epoch
                if shuffle:

                    # allow for custom shuffling
                    # function should take in a random generator and a number of samples per epoch
                    # function should return a permutation to use
                    perm = shuffle(self.rng, len(self)) if callable(shuffle) else self.rng.permutation(len(self))
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

        return batch_generator

    # function called by generator to construct a given batch
    def _construct_batch(self, args):
        for z in self._zero_pad:
            args[z] = pad_events(args[z], self.pad_val)

        # apply callback to batch
        batch = self.batch_callback(args)

        # unpack single element lists
        ret = batch[0] if len(batch) == 1 else tuple(batch)
        return (ret,) if self._wrap else ret

    # overriding this method provides a way to modify the batch once it's formed
    def batch_callback(self, args):
        return args

# splits all data args into weights (assumed to be first column) and features
class WeightedPointCloudDataset(PointCloudDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assume_padded = True

    def _init(self, **kwargs):
        super()._init(**kwargs)

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

    def __init__(self, *args, **kwargs):
        self.pairing = kwargs.pop('pairing', 'concat')

        super().__init__(*args, **kwargs)

        if not isinstance(self.pairing, (tuple, list)):
            self.pairing = [self.pairing]

        pairers = []
        for pairing in self.pairing:
            if isinstance(pairing, six.string_types):
                if pairing == 'concat':
                    pairer = ConcatenatePairer
                elif pairing == 'distance':
                    pairer = ParticleDistancePairer
                else:
                    raise ValueError("pairing '{}' not recognized".format(pairing))
            elif isinstance(pairing, FeaturePairerBase) or issubclass(pairing, FeaturePairerBase):
                pairer = pairing()
            else:
                raise ValueError("pairing '{}' not recognized".format(pairing))

            pairers.append(pairer)

        if len(pairers) == 0:
            raise ValueError('at least one pairing expected')
        elif len(pairers) == 1:
            self.pairer = pairers[0]()
        else:
            self.pairer = PairedFeatureCombiner(pairers)

        self.pair_and_pad_func = self.pairer.get_pair_and_pad_func()
        self.pair_array_func = self.pairer.get_pair_array_func()

    def __repr__(self):
        s = super().__repr__()
        return s + '  ' + repr(self.pairer)[:-1].replace('\n', '\n  ') + '\n'

    @property
    def nfeatures(self):
        self._init()
        if isinstance(self.batch_shapes[-1], int):
            return self.batch_shapes[-1]
        elif isinstance(self.batch_shapes[-1][-1],int):
            return self.batch_shapes[-1][-1]
        else:
            raise ValueError('unable to parse number of features from batch_shapes')

    @property
    def _join_kwargs(self):
        kwargs = super()._join_kwargs
        kwargs.update({'pairing': self.pairing})
        return kwargs

    def _init(self, **kwargs):
        super()._init(**kwargs)

        self._paired_args_need_padding = len(self.data_args)*[False]
        for i in range(len(self.data_args)):

            # add to our own list
            if i in self._zero_pad:
                self._assume_padded = False
                self._paired_args_need_padding[i] = True

            # get new shape resulting from pairing
            self._update_shape(i)

        # zero padding handled separately
        self._zero_pad.clear()

    def _update_shape(self, i):
        self.pairer._update_shape(self._batch_shapes, i)

    # pair objects in a point cloud dataset
    def batch_callback(self, args):
        return [self.pair_and_pad_func(arg, self.pad_val, max([len(x) for x in arg])) 
                if need_padding else self.pair_array_func(arg)
                for arg, need_padding in zip(args, self._paired_args_need_padding)]

class PairedWeightedPointCloudDataset(PairedPointCloudDataset, WeightedPointCloudDataset):

    def _update_shape(self, i):

        # features are located at 2*i+1, update them via update_shape
        super()._update_shape(2*i + 1)

        # if explicit number of particles is known, need to square it
        if self._batch_shapes[2*i][1] is not None:
            self._batch_shapes[2*i] = (self._batch_shapes[2*i][0], self._batch_shapes[2*i][1]**2)

    def batch_callback(self, args):

        # run args through weighted callback first, ignoring PairedPointCloudDataset.batch_callback
        args = super(PairedPointCloudDataset, self).batch_callback(args)

        paired_weighted_args = []
        for i, need_padding in enumerate(self._paired_args_need_padding):
            weights, features = args[2*i], args[2*i+1]
            if need_padding:
                max_len = max([len(w) for w in weights])
                paired_weighted_args.extend([self.product_and_pad_weights(weights, max_len),
                                             self.pair_and_pad_func(features, self.pad_val, max_len)])
            else:
                paired_weighted_args.extend([self.product_2d_weights(weights),
                                             self.pair_array_func(features)])

        return paired_weighted_args

    @staticmethod
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

    @staticmethod
    def product_2d_weights(weights):
        return (weights[:,None,:] * weights[:,:,None]).reshape(len(weights), -1)

class FeaturePairerBase(six.with_metaclass(ABCMeta, object)):
    """"""

    def __init__(self):
        """"""
        self.features_will_be_combined = False

    def __call__(self):
        return self

    def __repr__(self):
        return self.__class__.__name__ + '\n'

    def _update_shape(self, batch_shapes, i):
        new_nf = self.get_new_nfeatures(batch_shapes, i)
        if batch_shapes[i][1] is not None:
            batch_shapes[i] = (batch_shapes[i][0], batch_shapes[i][1]**2, new_nf)
        else:
            batch_shapes[i] = batch_shapes[i][:2] + (new_nf,)

    @abstractmethod
    def get_new_nfeatures(self, batch_shapes, i):
        pass

    def get_pair_and_pad_func(self):
        pair_func = self.get_pair_func()
        if self.features_will_be_combined:
            return pair_func
        else:
            return lambda X, pad_val, max_len: pad_events(pair_func(X), pad_val, max_len*max_len)

    @abstractmethod
    def get_pair_func(self):
        pass

    @abstractmethod
    def get_pair_array_func(self):
        pass

class PairedFeatureCombiner(FeaturePairerBase):
    """"""

    def __init__(self, pairers):
        """"""

        super().__init__()
        self.pairers = [pairer() for pairer in pairers]
        for pairer in self.pairers:
            pairer.features_will_be_combined = True

    def __call__(self):
        for pairer in self.pairers:
            pairer().features_will_be_combined = True
        return self

    def __repr__(self):
        s = 'PairedFeatureCombiner:\n'
        for pairer in self.pairers:
            s += '  - {}'.format(repr(pairer))
        return s

    def get_new_nfeatures(self, batch_shapes, i):
        return sum([pairer.get_new_nfeatures(batch_shapes, i) for pairer in self.pairers])

    @staticmethod
    def pair_and_pad_func(pair_funcs):
        n2combine = len(pair_funcs)
        def combined_pair_func(X, pad_val, max_len):
            pairs = [pair_func(X) for pair_func in pair_funcs]
            nfeatures = [p[0].shape[1] for p in pairs]

            output_shape = (len(X), max_len*max_len, sum(nfeatures))
            if pad_val == 0.:
                output = np.zeros(output_shape, dtype=X[0].dtype)
            else:
                output = np.full(output_shape, pad_val, dtype=X[0].dtype)

            # set events in padded array
            feature_slices = n2combine*[None]
            start = 0
            for i,nf in enumerate(nfeatures):
                end = start + nf
                feature_slices[i] = slice(start, end)
                start = end

            for i in range(len(X)):
                start = 0
                len_i = len(pairs[0][i])
                for p,feature_slice in zip(pairs, feature_slices):
                    end = start + nf
                    output[i,:len_i,feature_slice] = p[i]
                    start = end

            return output
        return combined_pair_func

    @staticmethod
    def pair_array_func(pair_array_funcs):
        def combined_pair_array_func(X):
            return np.concatenate([pair_array_func(X) for pair_array_func in pair_array_funcs], axis=2)
        return combined_pair_array_func

    def get_pair_and_pad_func(self):
        return self.pair_and_pad_func([pairer.get_pair_func() for pairer in self.pairers])

    def get_pair_func(self):
        raise RuntimeError('this method should never be called')

    def get_pair_array_func(self):
        return self.pair_array_func([pairer.get_pair_array_func() for pairer in self.pairers])

class ConcatenatePairer(FeaturePairerBase):
    """"""

    def __init__(self, *args, **kwargs):
        """"""

        self.cols = kwargs.pop('cols', None)
        super().__init__(*args, **kwargs)

    @staticmethod
    def pair_func(X, cols=None):

        paired_X = np.empty(len(X), dtype='O')
        if cols is None:
            nfeatures = X[0].shape[1]
            two_nfeatures = 2*nfeatures

            for i, x in enumerate(X):
                lenx = len(x)
                paired_shape = (lenx, lenx, nfeatures)
                x0 = np.broadcast_to(x[:,None], paired_shape)
                x1 = np.broadcast_to(x[None,:], paired_shape)
                paired_X[i] = np.concatenate((x0, x1), axis=2).reshape(-1, two_nfeatures)

        else:
            nfeatures = cols.stop - cols.start if isinstance(cols, slice) else len(cols)
            two_nfeatures = 2*nfeatures

            for i, x in enumerate(X):
                lenx = len(x)
                paired_shape = (lenx, lenx, nfeatures)
                x = x[:,cols]
                x0 = np.broadcast_to(x[:,None], paired_shape)
                x1 = np.broadcast_to(x[None,:], paired_shape)
                paired_X[i] = np.concatenate((x0, x1), axis=2).reshape(-1, two_nfeatures)

        return paired_X

    @staticmethod
    def pair_array_func(X, cols=None):
        if cols is None:
            nfeatures = X.shape[2]
        else:
            nfeatures = cols.stop - cols.start if isinstance(cols, slice) else len(cols)
            X = X[:,:,cols]

        max_len = X.shape[1]
        paired_shape = (len(X), max_len, max_len, nfeatures)
        X0 = np.broadcast_to(X[:,:,None], paired_shape)
        X1 = np.broadcast_to(X[:,None,:], paired_shape)
        return np.concatenate((X0, X1), axis=3).reshape(len(X), -1, 2*nfeatures)

    def get_new_nfeatures(self, batch_shapes, i):
        if self.cols is None:
            nfeatures = batch_shapes[i][2]
        else:
            nfeatures = (self.cols.stop - self.cols.start 
                         if isinstance(self.cols, slice) else len(self.cols))

        return 2*nfeatures

    def get_pair_func(self):
        return lambda X: self.pair_func(X, self.cols)

    def get_pair_array_func(self):
        return lambda X: self.pair_array_func(X, self.cols)

class DistancePairerBase(object):
    """"""

    def __init__(self, *args, **kwargs):
        """"""

        self.coord_cols = kwargs.pop('coord_cols', slice(0, 2))
        super().__init__(*args, **kwargs)

    def get_pair_func(self):
        return lambda X: self.pair_func(X, self.coord_cols)

    def get_pair_array_func(self):
        return lambda X: self.pair_array_func(X, self.coord_cols)

class ParticleDistancePairer(DistancePairerBase, FeaturePairerBase):
    """"""

    @staticmethod
    def pair_func(X, coord_cols=slice(0, 2)):
        paired_X = np.empty(len(X), dtype='O')
        for i, x in enumerate(X):
            paired_X[i] = np.linalg.norm(x[None,:,coord_cols] - x[:,None,coord_cols], axis=2).reshape(-1, 1)
        return paired_X

    @staticmethod
    def pair_array_func(X, coord_cols=slice(0, 2)):
        distance_matrices = np.linalg.norm(X[:,None,:,coord_cols] - X[:,:,None,coord_cols], axis=3)
        return distance_matrices.reshape(len(X), -1, 1)

    def get_new_nfeatures(self, batch_shapes, i):
        return 1

#class OriginDistancePairer(DistancePairerBase, FeaturePairerBase):
#
#    @staticmethod
#    def pair_func(X, coord_cols=slice(0, 2)):
#        dists_to_origin = np.asarray([np.linalg.norm(x[:,coord_cols], axis=1)[:,None] for x in X], dtype='O')
#        return ConcatenatePairer.pair_func(dists_to_origin)
#    
#    @staticmethod
#    def pair_array_func(X, coord_cols=slice(0, 2)):
#        distance_matrices = np.linalg.norm(X[:,:,coord_cols], axis=2)[:,:,None]
#        return ConcatenatePairer.pair_array_func(distance_matrices)
#
#    def get_new_nfeatures(self, batch_shapes, i):
#        return 2
