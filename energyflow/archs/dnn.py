#  _____  _   _ _   _
# |  __ \| \ | | \ | |
# | |  | |  \| |  \| |
# | |  | | . ` | . ` |
# | |__| | |\  | |\  |
# |_____/|_| \_|_| \_|

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from energyflow.archs.archbase import NNBase, _get_act_layer, _import_keras
from energyflow.utils.generic_utils import iter_or_rep

__all__ = ['DNN']

###############################################################################
# Dense Network Construction Function
###############################################################################

def construct_dense(input_tensor, sizes,
                    acts='relu', k_inits='he_uniform',
                    dropouts=0., l2_regs=0.,
                    names=None, act_names=None):
    """"""

    _import_keras(globals())
    
    # repeat options if singletons
    acts, k_inits = iter_or_rep(acts), iter_or_rep(k_inits)
    dropouts, l2_regs = iter_or_rep(dropouts), iter_or_rep(l2_regs)
    names, act_names = iter_or_rep(names), iter_or_rep(act_names)

    # lists of layers and tensors
    layers, tensors = [], [input_tensor]

    # iterate to make specified layers
    z = zip(sizes, acts, k_inits, dropouts, l2_regs, names, act_names)
    for s, act, k_init, dropout, l2_reg, name, act_name in z:

        # get layers and append them to list
        kwargs = ({'kernel_regularizer': keras.regularizers.L2(l2_reg), 'bias_regularizer': keras.regularizers.L2(l2_reg)} 
                  if l2_reg > 0. else {})
        dense_layer = keras.layers.Dense(s, kernel_initializer=k_init, name=name, **kwargs)
        act_layer = _get_act_layer(act, name=act_name)
        layers.extend([dense_layer, act_layer])

        # get tensors and append them to list
        tensors.append(dense_layer(tensors[-1]))
        tensors.append(act_layer(tensors[-1]))

        # apply dropout if specified
        if dropout > 0.:
            dr_name = None if name is None else '{}_dropout'.format(name)
            layers.append(keras.layers.Dropout(dropout, name=dr_name))
            tensors.append(layers[-1](tensors[-1]))

    return layers, tensors[1:]


###############################################################################
# DNN - Fully-connected neural network
###############################################################################

class DNN(NNBase):

    """Dense Neural Network architecture."""

    # DNN(*args, **kwargs)
    def _process_hps(self):
        """See [`ArchBase`](#archbase) for how to pass in hyperparameters as
        well as defaults common to all EnergyFlow neural network models.

        **Required DNN Hyperparameters**

        - **input_dim** : _int_
            - The number of inputs to the model.
        - **dense_sizes** : {_tuple_, _list_} of _int_
            - The number of nodes in the dense layers of the model.

        **Default DNN Hyperparameters**

        - **acts**=`'relu'` : {_tuple_, _list_} of _str_ or Keras activation
            - Activation functions(s) for the dense layers. A single string or
            activation layer will apply the same activation to all dense layers.
            Keras advanced activation layers are also accepted, either as
            strings (which use the default arguments) or as Keras `Layer` 
            instances. If passing a single `Layer` instance, be aware that this
            layer will be used for all activations and may introduce weight 
            sharing (such as with `PReLU`); it is recommended in this case to 
            pass as many activations as there are layers in the model.See the
            [Keras activations docs](https://www.tensorflow.org/api_docs/python/
            tf/keras/activations) for more detail.
        - **k_inits**=`'he_uniform'` : {_tuple_, _list_} of _str_ or Keras 
        initializer
            - Kernel initializers for the dense layers. A single string 
            will apply the same initializer to all layers. See the
            [Keras initializer docs](https://www.tensorflow.org/api_docs/python/
            tf/keras/initializers) for more detail.
        - **dropouts**=`0` : {_tuple_, _list_} of _float_
            - Dropout rates for the dense layers. A single float will apply the
            same dropout rate to all layers. See the [Keras Dropout layer](https
            ://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) for
            more detail.
        - **l2_regs**=`0` : {_tuple_, _list_} of _float_
            - $L_2$-regulatization strength for both the weights and biases
            of the dense layers. A single float will apply the same
            $L_2$-regulatization to all layers.
        """

        # process generic NN hps
        super(DNN, self)._process_hps()

         # required hyperparameters
        self.input_dim = self._proc_arg('input_dim')
        self.dense_sizes = self._proc_arg('dense_sizes')

        # activations
        self.acts = iter_or_rep(self._proc_arg('acts', default='relu'))

        # initializations
        self.k_inits = iter_or_rep(self._proc_arg('k_inits', default='he_uniform'))

        # regularization
        self.dropouts = iter_or_rep(self._proc_arg('dropouts', default=0.))
        self.l2_regs = iter_or_rep(self._proc_arg('l2_regs', default=0.))

        self._verify_empty_hps()

    def _construct_model(self):

        _import_keras(globals())

        # get an input tensor
        self._input = keras.layers.Input(shape=(self.input_dim,), name=self._proc_name('input'))

        # get potential names of layers
        names = [self._proc_name('dense_'+str(i)) for i in range(len(self.dense_sizes))]
        act_names = [self._proc_act_name(act) for size,act in zip(self.dense_sizes, self.acts)]

        # construct layers and tensors
        self._layers, tensors = construct_dense(self._input, self.dense_sizes,
                                                acts=self.acts, k_inits=self.k_inits,
                                                dropouts=self.dropouts, l2_regs=self.l2_regs,
                                                names=names, act_names=act_names)
        self._tensors = [self._input] + tensors

        # get output layers
        out_layer = keras.layers.Dense(self.output_dim, name=self._proc_name('output'))
        act_layer = _get_act_layer(self.output_act, name=self._proc_act_name(self.output_act))
        self._layers.extend([out_layer, act_layer])

        # append output tensors
        self._tensors.append(out_layer(self._tensors[-1]))
        self._tensors.append(act_layer(self._tensors[-1]))
        self._output = self._tensors[-1]

        # construct a new model
        self._model = keras.models.Model(inputs=self._input, outputs=self._output, name=self.model_name)

        # compile model
        self._compile_model()
