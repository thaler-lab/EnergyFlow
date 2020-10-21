from itertools import repeat

from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate, Dense, Dropout, Input, Lambda, Masking, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from energyflow.archs import PFN
from energyflow.archs.archbase import _apply_act
from energyflow.utils import iter_or_rep

def make_dense_layers(sizes, input_layer=None, input_shape=None, activations='relu', dropouts=0., l2_regs=0., k_inits='he_uniform'):

    # process options
    activations, dropouts = iter_or_rep(activations), iter_or_rep(dropouts)
    l2_regs, k_inits = iter_or_rep(l2_regs), iter_or_rep(k_inits)

    if input_shape is not None:
        input_layer = Input(shape=input_shape)

    # a list to store the layers
    dense_layers = [input_layer]

    # iterate over specified dense layers
    z = zip(sizes, activations, k_inits, dropouts, l2_regs)
    for i,(s, act, k_init, dropout, l2_reg) in enumerate(z):

        # construct variable argument dict
        kwargs = {'kernel_initializer': k_init}
        if l2_reg > 0.:
            kwargs.update({'kernel_regularizer': l2(l2_reg), 'bias_regularizer': l2(l2_reg)})

        # a new dense layer
        new_layer = _apply_act(act, Dense(s, **kwargs)(dense_layers[-1]))

        # apply dropout (does nothing if dropout is zero)
        if dropout > 0.:
            new_layer = Dropout(dropout)(new_layer)

        # apply new layer to previous and append to list
        dense_layers.append(new_layer)

    return dense_layers

# get two PFNs for muons and electrons
muon_pfn = PFN(input_dim=5, Phi_sizes=[100, 100], F_sizes=[50], compile=False, name_layers=False)
electron_pfn = PFN(input_dim=5, Phi_sizes=[100, 100], F_sizes=[50], compile=False, name_layers=False)

# make some dense layers (including an input layer) for the jet variables dnn
jet_vars_dnn = make_dense_layers([100, 100], input_shape=(10,))

# a list of the input layers
inputs = muon_pfn.inputs + electron_pfn.inputs + [jet_vars_dnn[0]]

# the concatenated layer
concat_layer = concatenate([muon_pfn.F[-1], electron_pfn.F[-1], jet_vars_dnn[-1]])

# a DNN to combine things on the backend
combo_dnn = make_dense_layers([100, 100], concat_layer)

# a binary-classification-like output
output = Dense(2, activation='softmax', name='output')(combo_dnn[-1])

# make the model and compile it
model = Model(inputs=inputs, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# print summary
model.summary()
