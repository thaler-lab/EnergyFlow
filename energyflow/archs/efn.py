from __future__ import absolute_import, division, print_function

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2

from energyflow.archs.archbase import NNBase
from energyflow.utils import iter_or_rep

__all__ = ['EFN']

###############################################################################
# EFN
###############################################################################

class EFN(NNBase):

    def process_hps(self):

        # process generic NN hps
        super(EFN, self).process_hps()

        # required hyperparameters
        self.input_dim = self.hps['input_dim']
        self.ppm_sizes = self.hps['ppm_sizes']
        self.dense_sizes = self.hps['dense_sizes']

        # activations
        self.ppm_acts = iter_or_rep(self.hps.get('ppm_acts', 'relu'))
        self.dense_acts = iter_or_rep(self.hps.get('dense_acts', 'relu'))

        # initializations
        self.ppm_k_inits = iter_or_rep(self.hps.get('ppm_k_inits', 'he_uniform'))
        self.dense_k_inits = iter_or_rep(self.hps.get('dense_k_inits', 'he_uniform'))

        # regularizations
        self.ppm_dropouts = iter_or_rep(self.hps.get('ppm_dropouts', 0))
        self.latent_dropout = self.hps.get('latent_dropout', 0)
        self.dense_dropouts = iter_or_rep(self.hps.get('dense_dropouts', 0))

        # masking
        self.mask_val = self.hps.get('mask_val', 0.)

    def construct_model(self):

        # common model preprocessing
        super(EFN, self).construct_model()

    # structure
    e_weight = hps.get('e_weight', False)
    mask_val = hps.get('mask_val', 0.)


    # define input layer for events
    input_layers = [Input(batch_shape=(None, None, input_dim), name='input_0')]

    # mask inputs to remove all-zero particles (makes physical sense and undoes padding)
    masked_layer = Masking(mask_value=mask_val, name='mask_0')(input_layers[0])

    # define vector of dense layers to be used first
    dense_layers_1 = [Dense(s, activation=act, name='dense_' + str(i), kernel_initializer=k_init) 
                      for i,(s, act, k_init) in enumerate(zip(tdist_sizes, tdist_acts, tdist_k_inits))]

    # apply TimeDistributed layers to dense layers
    num_tdist = len(dense_layers_1)
    if num_tdist:
        tdist_layers = [TimeDistributed(dense_layers_1[0], name='tdist_0')(masked_layer)]

        for i,dl in enumerate(dense_layers_1[1:]):
            tdist_layers.append(TimeDistributed(dl, name='tdist_'+str(i+1))(tdist_layers[i]))
    else:
        tdist_layers = [masked_layer]

    # sum over all the particles
    if e_weight:
        input_layers.append(Input(batch_shape=(None, None), name='input_1'))
        latent_layer = Dot(0, name='dot')([tdist_layers[-1], input_layers[1]])
        input_layers.reverse()
    else:
        latent_layer = Lambda(lambda x: K.sum(x, axis=1), name='sum')(tdist_layers[-1])

    dropout_layer = Dropout(latent_dropout)(latent_layer)

    # second dense layers
    if len(dense_sizes):
        dense_z = list(zip(dense_sizes, dense_acts, dense_k_inits))
        dense_layers_2 = [Dense(dense_z[0][0], activation=dense_z[0][1], name='dense_0', 
                                kernel_initializer=dense_z[0][2])(dropout_layer)]
        for i,(s, act, k_init) in enumerate(dense_z[1:]):
            dense_layers_2.append(Dense(s, activation=act, name='dense_'+str(i+1), 
                                        kernel_initializer=k_init)(dense_layers_2[i]))
    else:
        dense_layers_2 = [latent_layer]

    # define output layer
    output_layer = Dense(output_dim, activation=output_act, name='output')(dense_layers_2[-1])

    # define model
    model = Model(inputs=input_layers, outputs=output_layer)

    # compile models with standard choices
    if hps.get('compile', True):
        model.compile(loss=loss, optimizer=opt(lr=lr), metrics=hps.get('metrics', ['accuracy']))

    if hps.get('summary', True):
        model.summary()
    
    return model
