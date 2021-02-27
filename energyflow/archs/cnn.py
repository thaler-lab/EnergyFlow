#   _____ _   _ _   _
#  / ____| \ | | \ | |
# | |    |  \| |  \| |
# | |    | . ` | . ` |
# | |____| |\  | |\  |
#  \_____|_| \_|_| \_|

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, SpatialDropout2D
from tensorflow.keras.models import Sequential

from energyflow.archs.archbase import NNBase
from energyflow.utils import iter_or_rep

__all__ = ['CNN']

###############################################################################
# CNN
###############################################################################

class CNN(NNBase):

    """Convolutional Neural Network architecture."""

    # CNN(*args, **kwargs)
    def _process_hps(self):
        """See [`ArchBase`](#archbase) for how to pass in hyperparameters as
        well as defaults common to all EnergyFlow neural network models.

        **Required CNN Hyperparameters**

        - **input_shape** : {_tuple_, _list_} of _int_
            - The shape of a single jet image. Assuming that `data_format`
            is set to `channels_first`, this is `(nb_chan,npix,npix)`.
        - **filter_sizes** : {_tuple_, _list_} of _int_
            - The size of the filters, which are taken to be square, in each 
            convolutional layer of the network. The length of the list will be
            the number of convolutional layers in the network.
        - **num_filters** : {_tuple_, _list_} of _int_
            - The number of filters in each convolutional layer. The length of 
            `num_filters` must match that of `filter_sizes`.

        **Default CNN Hyperparameters**

        - **dense_sizes**=`None` : {_tuple_, _list_} of _int_
            - The sizes of the dense layer backend. A value of `None` is 
            equivalent to an empty list.
        - **pool_sizes**=`0` : {_tuple_, _list_} of _int_
            - Size of maxpooling filter, taken to be a square. A value of 
            `0` will not use maxpooling.
        - **conv_acts**=`'relu'` : {_tuple_, _list_} of _str_  or Keras activation
            - Activation function(s) for the conv layers. A single string or
            activation layer will apply the same activation to all conv layers.
            Keras advanced activation layers are also accepted, either as
            strings (which use the default arguments) or as Keras `Layer` 
            instances. If passing a single `Layer` instance, be aware that this
            layer will be used for all activations and may introduce weight 
            sharing (such as with `PReLU`); it is recommended in this case to 
            pass as many activations as there are layers in the model.See the
            [Keras activations docs](https://keras.io/activations/) for more 
            detail.
        - **dense_acts**=`'relu'` : {_tuple_, _list_} of _str_  or Keras activation
            - Activation functions(s) for the dense layers. A single string 
            or activation layer will apply the same activation to all dense 
            layers.
        - **conv_k_inits**=`'he_uniform'` : {_tuple_, _list_} of _str_ or Keras initializer
            - Kernel initializers for the convolutional layers. A single
            string will apply the same initializer to all layers. See the
            [Keras initializer docs](https://keras.io/initializers/) for 
            more detail.
        - **dense_k_inits**=`'he_uniform'` : {_tuple_, _list_} of _str_ or Keras initializer
            - Kernel initializers for the dense layers. A single string will 
            apply the same initializer to all layers.
        - **conv_dropouts**=`0` : {_tuple_, _list_} of _float_
            - Dropout rates for the convolutional layers. A single float will
            apply the same dropout rate to all conv layers. See the [Keras
            Dropout layer](https://keras.io/layers/core/#dropout) for more 
            detail.
        - **num_spatial2d_dropout**=`0` : _int_
            - The number of convolutional layers, starting from the beginning
            of the model, for which to apply [SpatialDropout2D](https://keras
            .io/layers/core/#spatialdropout2d) instead of Dropout.
        - **dense_dropouts**=`0` : {_tuple_, _list_} of _float_
            - Dropout rates for the dense layers. A single float will apply 
            the same dropout rate to all dense layers.
        - **paddings**=`'valid'` : {_tuple_, _list_} of _str_
            - Controls how the filters are convoled with the inputs. See
            the [Keras Conv2D layer](https://keras.io/layers/convolutional/#conv2d) 
            for more detail.
        - **data_format**=`'channels_last'` : {`'channels_first'`, `'channels_last'`}
            - Sets which axis is expected to contain the different channels.
            `'channels_first'` appears to have issues with newer versions of 
            tensorflow, so prefer `'channels_last'`.
        """

        # process generic NN hps
        super(CNN, self)._process_hps()

        # required hyperparameters
        self.input_shape = self._proc_arg('input_shape')
        self.filter_sizes = self._proc_arg('filter_sizes')
        self.num_filters = self._proc_arg('num_filters')

        # required checks
        m = 'filter_sizes and num_filters must be the same length'
        assert len(self.filter_sizes) == len(self.num_filters), m

        # optional (but likely provided) hyperparameters with defaults
        self.pool_sizes = iter_or_rep(self._proc_arg('pool_sizes', default=0))
        self.dense_sizes = self._proc_arg('dense_sizes', default=None)
        if self.dense_sizes is None:
            self.dense_sizes = []

        # activations
        self.conv_acts = iter_or_rep(self._proc_arg('conv_acts', default='relu'))
        self.dense_acts = iter_or_rep(self._proc_arg('dense_acts', default='relu'))

        # initializations
        self.conv_k_inits = iter_or_rep(self._proc_arg('conv_k_inits', default='he_uniform'))
        self.dense_k_inits = iter_or_rep(self._proc_arg('dense_k_inits', default='he_uniform'))

        # regularization
        self.conv_dropouts = iter_or_rep(self._proc_arg('conv_dropouts', default=0.))
        self.num_spatial2d_dropout = self._proc_arg('num_spatial2d_dropout', default=0)
        self.dense_dropouts = iter_or_rep(self._proc_arg('dense_dropouts', default=0.))

        # padding
        self.paddings = iter_or_rep(self._proc_arg('padding', default='valid'))
        self.data_format = self._proc_arg('data_format', default='channels_last')

        self._verify_empty_hps()

    def _construct_model(self):

        # fresh model
        self._model = Sequential()

        # iterate over conv specifications
        conv_z = zip(self.filter_sizes, self.num_filters, self.pool_sizes, self.conv_acts,
                     self.conv_dropouts, self.conv_k_inits, self.paddings)
        num_dropout = 0
        looped = False
        for i,(filter_size, num_filter, pool_size, act, dropout, k_init, pad) in enumerate(conv_z):
            looped = True

            # add conv2d layer to model using provided hyperparameters
            kwargs = {} if i > 0 else {'input_shape': self.input_shape}
            self.model.add(Conv2D(num_filter, filter_size, kernel_initializer=k_init,
                                                           padding=pad, data_format=self.data_format,
                                                           name=self._proc_name('conv2d_'+str(i)), **kwargs))
            self._add_act(act)

            # add pooling layer if we have a non-zero pool size
            if pool_size > 0:
                self.model.add(MaxPooling2D(pool_size=pool_size, data_format=self.data_format, 
                                            name=self._proc_name('max_pool_'+str(i))))

            # add dropout layer if we have a non-zero dropout rate
            if dropout > 0.:
                d_layer = SpatialDropout2D if i < self.num_spatial2d_dropout else Dropout
                self.model.add(d_layer(dropout, name=self._proc_name('dropout_'+str(i))))
                num_dropout += 1

        if not looped:
            raise ValueError('need to specify at least one conv layer')

        # flatten model for dense layers
        self.model.add(Flatten(name=self._proc_name('flatten')))

        # iterate over dense specifications
        dense_z = zip(self.dense_sizes, self.dense_acts, self.dense_dropouts, self.dense_k_inits)
        for i,(num_dense, act, dropout, k_init) in enumerate(dense_z):

            # add dense layer
            self.model.add(Dense(num_dense, kernel_initializer=k_init, name=self._proc_name('dense_'+str(i))))
            self._add_act(act)

            # add dropout layer if dropout is nonzero
            if dropout > 0.:
                self.model.add(Dropout(dropout, name=self._proc_name('dropout_'+str(i+num_dropout))))

        # output layer
        self.model.add(Dense(self.output_dim, name=self._proc_name('output')))
        self._add_act(self.output_act)

        # compile model
        self._compile_model()
