from __future__ import absolute_import, division, print_function

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2

from energyflow.archs.archbase import NNBase
from energyflow.utils import iter_or_rep

__all__ = ['DNN']

###############################################################################
# DNN
###############################################################################

class DNN(NNBase):

    def process_hps(self):

        # process generic NN hps
        super(DNN, self).process_hps()

        # required hyperparameters
        self.input_dim = self.hps['input_dim']
        self.dense_sizes = self.hps['dense_sizes']

        # activations
        self.acts = iter_or_rep(self.hps.get('acts', 'relu'))

        # initializations
        self.k_inits = iter_or_rep(self.hps.get('k_inits', 'he_uniform'))

        # regularization
        self.dropouts = iter_or_rep(self.hps.get('dropouts', 0))
        self.l2_regs = iter_or_rep(self.hps.get('l2_regs', 0))

    def construct_model(self):

        # fresh model
        self._model = Sequential()

        # iterate over specified dense layers
        z = zip(self.dense_sizes, self.acts, self.dropouts, self.l2_regs, self.k_inits)
        for i,(dim, act, dropout, l2_reg, k_init) in enumerate(z):

            # construct variable argument dict
            kwargs = {} if i > 0 else {'input_dim': self.input_dim}
            if l2_reg > 0:
                kwargs.update({'kernel_regularizer': l2(l2_reg),
                               'bias_regularizer': l2(l2_reg)})

            # add dense layer
            self.model.add(Dense(dim, activation=act, kernel_initializer=k_init, 
                                      name='dense_'+str(i), **kwargs))

            # add dropout layer if nonzero
            if dropout > 0:
                self.model.add(Dropout(dropout, name='dropout_' + str(i)))

        # output layer
        self.model.add(Dense(self.output_dim, activation=self.output_act, name='output'))    

        # compile model
        self.compile_model()
