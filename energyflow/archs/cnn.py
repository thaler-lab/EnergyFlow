from __future__ import absolute_import, division, print_function

from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, SpatialDropout2D
from keras.models import Sequential

from energyflow.archs.archbase import NNBase
from energyflow.utils import iter_or_rep, transfer

__all__ = ['CNN']

###############################################################################
# CNN
###############################################################################

class CNN(NNBase):

    def process_hps(self):

        # process generic NN hps
        super(CNN, self).process_hps()

        # required hyperparameters
        transfer(self, self.hps, ['input_shape', 'filter_sizes', 'num_filters'])

        # required checks
        m = 'filter_sizes and num_filters must be the same length'
        assert len(self.filter_sizes) == len(self.num_filters), m

        # optional (but likely provided) hyperparameters with defaults
        self.pool_sizes = iter_or_rep(self.hps.get('pool_sizes', None))
        self.dense_sizes = self.hps.get('dense_sizes', None)

        # activations
        self.conv_acts = iter_or_rep(self.hps.get('conv_acts', 'relu'))
        self.dense_acts = iter_or_rep(self.hps.get('dense_acts', 'relu'))

        # initializations
        self.conv_k_inits = iter_or_rep(self.hps.get('conv_k_inits', 'he_uniform'))
        self.dense_k_inits = iter_or_rep(self.hps.get('dense_k_inits', 'he_uniform'))

        # regularization
        self.conv_dropouts = iter_or_rep(self.hps.get('conv_dropouts', 0))
        self.num_spatial2d_dropout = self.hps.get('num_spatial2d_dropout', 0)
        self.dense_dropouts = iter_or_rep(self.hps.get('dense_dropouts', 0))

        # padding
        self.paddings = iter_or_rep(self.hps.get('padding', 'valid'))

    def construct_model(self):

        # fresh model
        self._model = Sequential()

        # iterate over conv specifications
        conv_z = zip(self.filter_sizes, self.num_filters, self.pool_sizes, self.conv_acts,
                     self.conv_dropouts, self.conv_k_inits, self.paddings)
        num_dropout = 0
        for i,(filter_size, num_filter, pool_size, act, dropout, k_init, pad) in enumerate(conv_z):

            # add conv2d layer to model using provided hyperparameters
            kwargs = {} if i > 0 else {'input_shape': self.input_shape}
            self.model.add(Conv2D(num_filter, filter_size, activation=act, kernel_initializer=k_init,
                                                           padding=pad, name='conv2d_'+str(i), **kwargs))

            # add pooling layer if we have a non-zero pool size
            if pool_size > 0:
                self.model.add(MaxPooling2D(pool_size=pool_size, name='max_pool_' + str(i)))

            # add dropout layer if we have a non-zero dropout rate
            if dropout > 0:
                d_layer = SpatialDropout2D if i < self.num_spatial2d_dropout else Dropout
                self.model.add(d_layer(dropout, name='dropout_' + str(i)))
                num_dropout += 1

        # flatten model for dense layers
        self.model.add(Flatten(name='flatten'))

        # iterate over dense specifications
        dense_z = zip(self.dense_sizes, self.dense_acts, self.dense_dropouts, self.dense_k_inits)
        for i,(num_dense, act, dropout, k_init) in enumerate(dense_z):

            # add dense layer
            self.model.add(Dense(num_dense, activation=act, kernel_initializer=k_init, name='dense_'+str(i)))

            # add dropout layer if dropout is nonzero
            if dropout > 0:
                self.model.add(Dropout(dropout, name='dropout_' + str(i + num_dropout)))

        # output layer
        self.model.add(Dense(self.output_dim, activation=self.output_act, name='output'))

        # compile model
        self.compile_model()

