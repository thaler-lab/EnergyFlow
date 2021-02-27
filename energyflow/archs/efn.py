#  ______ ______ _   _
# |  ____|  ____| \ | |
# | |__  | |__  |  \| |
# |  __| |  __| | . ` |
# | |____| |    | |\  |
# |______|_|    |_| \_|
                      
# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

from abc import abstractmethod, abstractproperty

import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras import __version__ as __keras_version__
from tensorflow.keras.layers import Concatenate, Dense, Dot, Dropout, Input, Lambda, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from energyflow.archs.archbase import NNBase, _get_act_layer
from energyflow.archs.dnn import construct_dense
from energyflow.utils import iter_or_rep

__all__ = [

    # input constructor functions
    #'construct_weighted_input', 'construct_input',

    # weight mask constructor functions
    #'construct_efn_weight_mask', 'construct_pfn_weight_mask',

    # network consstructor functions
    #'construct_distributed_dense', 'construct_latent', 'construct_dense', 

    # full model classes
    'EFN', 'PFN'
]

################################################################################
# Keras 2.2.5 fixes bug in 2.2.4 that affects our usage of the Dot layer
################################################################################

if __keras_version__.endswith('-tf'):
    __keras_version__ = __keras_version__[:-3]
keras_version_tuple = tuple(map(int, __keras_version__.split('.')))
DOT_AXIS = 1

################################################################################
# INPUT FUNCTIONS
################################################################################

def construct_weighted_input(input_dim, zs_name=None, phats_name=None):

    # construct input tensors
    zs_input = Input(batch_shape=(None, None), name=zs_name)
    phats_input = Input(batch_shape=(None, None, input_dim), name=phats_name)

    return [zs_input, phats_input]

def construct_input(input_dim, nnone=2, name=None):

    # construct input tensor
    return [Input(batch_shape=nnone*(None,) + (input_dim,), name=name)]


################################################################################
# WEIGHT MASK FUNCTIONS
################################################################################

def construct_efn_weight_mask(input_tensor, mask_val=0., name=None):
    """"""

    # define a function which maps the given mask_val to zero
    def efn_mask_func(X, mask_val=mask_val):
    
        # map mask_val to zero and leave everything else alone
        return X * K.cast(K.not_equal(X, mask_val), K.dtype(X))

    mask_layer = Lambda(efn_mask_func, name=name)

    # return as lists for consistency
    return [mask_layer], mask_layer(input_tensor)

def construct_pfn_weight_mask(input_tensor, mask_val=0., name=None):
    """"""

    # define a function which maps the given mask_val to zero
    def pfn_mask_func(X, mask_val=mask_val):

        # map mask_val to zero and return 1 elsewhere
        return K.cast(K.any(K.not_equal(X, mask_val), axis=-1), K.dtype(X))

    mask_layer = Lambda(pfn_mask_func, name=name)

    # return as lists for consistency
    return [mask_layer], mask_layer(input_tensor)


################################################################################
# NETWORK FUNCTIONS
################################################################################

def construct_distributed_dense(input_tensor, sizes, acts='relu', k_inits='he_uniform', 
                                                                  names=None, l2_regs=0.):
    """"""

    # repeat options if singletons
    acts, k_inits, names = iter_or_rep(acts), iter_or_rep(k_inits), iter_or_rep(names)
    l2_regs = iter_or_rep(l2_regs)
    
    # list of tensors
    layers, tensors = [], [input_tensor]

    # iterate over specified layers
    for s, act, k_init, name, l2_reg in zip(sizes, acts, k_inits, names, l2_regs):
        
        # define a dense layer that will be applied through time distributed
        kwargs = {} 
        if l2_reg > 0.:
            kwargs.update({'kernel_regularizer': l2(l2_reg), 'bias_regularizer': l2(l2_reg)})
        d_layer = Dense(s, kernel_initializer=k_init, **kwargs)

        # get layers and append them to list
        tdist_layer = TimeDistributed(d_layer, name=name)
        act_layer = _get_act_layer(act)
        layers.extend([tdist_layer, act_layer])

        # get tensors and append them to list
        tensors.append(tdist_layer(tensors[-1]))
        tensors.append(act_layer(tensors[-1]))

    return layers, tensors[1:]

def construct_latent(input_tensor, weight_tensor, dropout=0., name=None):
    """"""

    # lists of layers and tensors
    layers = [Dot(DOT_AXIS, name=name)]
    tensors = [layers[-1]([weight_tensor, input_tensor])]

    # apply dropout if specified
    if dropout > 0.:
        dr_name = None if name is None else '{}_dropout'.format(name)
        layers.append(Dropout(dropout, name=dr_name))
        tensors.append(layers[-1](tensors[-1]))

    return layers, tensors


################################################################################
# SymmetricPerParticleNN - Base class for EFN-like models
################################################################################

class SymmetricPerParticleNN(NNBase):

    # EFN(*args, **kwargs)
    def _process_hps(self):
        r"""See [`ArchBase`](#archbase) for how to pass in hyperparameters as
        well as defaults common to all EnergyFlow neural network models.

        **Required EFN Hyperparameters**

        - **input_dim** : _int_
            - The number of features for each particle.
        - **Phi_sizes** (formerly `ppm_sizes`) : {_tuple_, _list_} of _int_
            - The sizes of the dense layers in the per-particle frontend
            module $\Phi$. The last element will be the number of latent 
            observables that the model defines.
        - **F_sizes** (formerly `dense_sizes`) : {_tuple_, _list_} of _int_
            - The sizes of the dense layers in the backend module $F$.

        **Default EFN Hyperparameters**

        - **Phi_acts**=`'relu'` (formerly `ppm_acts`) : {_tuple_, _list_} of
        _str_ or Keras activation
            - Activation functions(s) for the dense layers in the 
            per-particle frontend module $\Phi$. A single string or activation
            layer will apply the same activation to all layers. Keras advanced
            activation layers are also accepted, either as strings (which use
            the default arguments) or as Keras `Layer` instances. If passing a
            single `Layer` instance, be aware that this layer will be used for
            all activations and may introduce weight sharing (such as with 
            `PReLU`); it is recommended in this case to pass as many activations
            as there are layers in the model. See the [Keras activations 
            docs](https://keras.io/activations/) for more detail.
        - **F_acts**=`'relu'` (formerly `dense_acts`) : {_tuple_, _list_} of
        _str_ or Keras activation
            - Activation functions(s) for the dense layers in the 
            backend module $F$. A single string or activation layer will apply
            the same activation to all layers.
        - **Phi_k_inits**=`'he_uniform'` (formerly `ppm_k_inits`) : {_tuple_,
        _list_} of _str_ or Keras initializer
            - Kernel initializers for the dense layers in the per-particle
            frontend module $\Phi$. A single string will apply the same
            initializer to all layers. See the [Keras initializer docs](https:
            //keras.io/initializers/) for more detail.
        - **F_k_inits**=`'he_uniform'` (formerly `dense_k_inits`) : {_tuple_,
        _list_} of _str_ or Keras initializer
            - Kernel initializers for the dense layers in the backend 
            module $F$. A single string will apply the same initializer 
            to all layers.
        - **latent_dropout**=`0` : _float_
            - Dropout rates for the summation layer that defines the
            value of the latent observables on the inputs. See the [Keras
            Dropout layer](https://keras.io/layers/core/#dropout) for more 
            detail.
        - **F_dropouts**=`0` (formerly `dense_dropouts`) : {_tuple_, _list_}
        of _float_
            - Dropout rates for the dense layers in the backend module $F$. 
            A single float will apply the same dropout rate to all dense layers.
        - **Phi_l2_regs**=`0` : {_tuple_, _list_} of _float_
            - $L_2$-regulatization strength for both the weights and biases
            of the layers in the $\Phi$ network. A single float will apply the
            same $L_2$-regulatization to all layers.
        - **F_l2_regs**=`0` : {_tuple_, _list_} of _float_
            - $L_2$-regulatization strength for both the weights and biases
            of the layers in the $F$ network. A single float will apply the
            same $L_2$-regulatization to all layers.
        - **mask_val**=`0` : _float_
            - The value for which particles with all features set equal to
            this value will be ignored. The [Keras Masking layer](https://
            keras.io/layers/core/#masking) appears to have issues masking
            the biases of a network, so this has been implemented in a
            custom (and correct) manner since version `0.12.0`.
        - **num_global_features**=`None` : _int_
            - Number of additional features to be concatenated with the latent
            space observables to form the input to F. If not `None`, then the
            features are to be provided at the end of the list of inputs.
        """

        # process generic NN hps
        super(SymmetricPerParticleNN, self)._process_hps()

        # required hyperparameters
        self.input_dim = self._proc_arg('input_dim')
        self.Phi_sizes = self._proc_arg('Phi_sizes', old='ppm_sizes')
        self.F_sizes = self._proc_arg('F_sizes', old='dense_sizes')

        # activations
        self.Phi_acts = iter_or_rep(self._proc_arg('Phi_acts', default='relu', 
                                                               old='ppm_acts'))
        self.F_acts = iter_or_rep(self._proc_arg('F_acts', default='relu', 
                                                           old='dense_acts'))

        # initializations
        self.Phi_k_inits = iter_or_rep(self._proc_arg('Phi_k_inits', default='he_uniform', 
                                                                     old='ppm_k_inits'))
        self.F_k_inits = iter_or_rep(self._proc_arg('F_k_inits', default='he_uniform', 
                                                                 old='dense_k_inits'))

        # regularizations
        self.latent_dropout = self._proc_arg('latent_dropout', default=0.)
        self.F_dropouts = iter_or_rep(self._proc_arg('F_dropouts', default=0., 
                                                                   old='dense_dropouts'))
        self.Phi_l2_regs = iter_or_rep(self._proc_arg('Phi_l2_regs', default=0.))
        self.F_l2_regs   = iter_or_rep(self._proc_arg('F_l2_regs', default=0.))

        # masking
        self.mask_val = self._proc_arg('mask_val', default=0.)

        # additional network modifications
        self.num_global_features = self._proc_arg('num_global_features', default=None)

        self._verify_empty_hps()

    def _construct_model(self):

        # initialize dictionaries for holding indices of subnetworks
        self._layer_inds, self._tensor_inds = {}, {}

        # construct earlier parts of the model
        self._construct_inputs()
        self._construct_Phi()
        self._construct_latent()
        self._construct_F()

        # get output layers
        out_layer = Dense(self.output_dim, name=self._proc_name('output'))
        act_layer = _get_act_layer(self.output_act)
        self._layers.extend([out_layer, act_layer])

        # append output tensors
        self._tensors.append(out_layer(self.tensors[-1]))
        self._tensors.append(act_layer(self.tensors[-1]))

        # construct a new model
        self._model = Model(inputs=self.inputs, outputs=self.output)

        # compile model
        self._compile_model()

    @abstractmethod
    def _construct_inputs(self):
        pass

    def _construct_Phi(self):

        # get names
        names = [self._proc_name('tdist_{}'.format(i)) for i in range(len(self.Phi_sizes))]

        # determine begin inds
        layer_inds, tensor_inds = [len(self.layers)], [len(self.tensors)]

        # construct Phi
        Phi_layers, Phi_tensors = construct_distributed_dense(self.tensors[-1], self.Phi_sizes, 
                                                              acts=self.Phi_acts, 
                                                              k_inits=self.Phi_k_inits, 
                                                              names=names, 
                                                              l2_regs=self.Phi_l2_regs)
        
        # add layers and tensors to internal lists
        self._layers.extend(Phi_layers)
        self._tensors.extend(Phi_tensors)

        # determine end inds
        layer_inds.append(len(self.layers))
        tensor_inds.append(len(self.tensors))

        # store inds
        self._layer_inds['Phi'] = layer_inds
        self._tensor_inds['Phi'] = tensor_inds

    def _construct_latent(self):

        # determine begin inds
        layer_inds, tensor_inds = [len(self.layers)], [len(self.tensors)]

        # construct latent tensors
        latent_layers, latent_tensors = construct_latent(self.tensors[-1], self.weights, 
                                                         dropout=self.latent_dropout, 
                                                         name=self._proc_name('sum'))
        
        # add layers and tensors to internal lists
        self._layers.extend(latent_layers)
        self._tensors.extend(latent_tensors)

        # determine end inds
        layer_inds.append(len(self.layers))
        tensor_inds.append(len(self.tensors))

        # store inds
        self._layer_inds['latent'] = layer_inds
        self._tensor_inds['latent'] = tensor_inds

    def _construct_F(self):

        # get names
        names = [self._proc_name('dense_{}'.format(i)) for i in range(len(self.F_sizes))]

        # determine begin inds
        layer_inds, tensor_inds = [len(self.layers)], [len(self.tensors)]


        # construct F
        F_layers, F_tensors = construct_dense(self.tensors[-1], self.F_sizes,
                                              acts=self.F_acts, k_inits=self.F_k_inits, 
                                              dropouts=self.F_dropouts, names=names,
                                              l2_regs=self.F_l2_regs)

        # add layers and tensors to internal lists
        self._layers.extend(F_layers)
        self._tensors.extend(F_tensors)

        # determine end inds
        layer_inds.append(len(self.layers))
        tensor_inds.append(len(self.tensors))

        # store inds
        self._layer_inds['F'] = layer_inds
        self._tensor_inds['F'] = tensor_inds

    @abstractproperty
    def inputs(self):
        pass

    @abstractproperty
    def weights(self):
        pass

    @property
    def Phi(self):
        r"""List of tensors corresponding to the layers in the $\Phi$ network."""

        begin, end = self._tensor_inds['Phi']
        return self._tensors[begin:end]

    @property
    def latent(self):
        """List of tensors corresponding to the summation layer in the
        network, including any dropout layer if present.
        """

        begin, end = self._tensor_inds['latent']
        return self._tensors[begin:end]

    @property
    def F(self):
        """List of tensors corresponding to the layers in the $F$ network."""

        begin, end = self._tensor_inds['F']
        return self._tensors[begin:end]

    @property
    def output(self):
        """Output tensor for the model."""

        return self._tensors[-1]

    @property
    def layers(self):
        """List of all layers in the model."""

        return self._layers

    @property
    def tensors(self):
        """List of all tensors in the model. Order may be arbitrary given that
        not every model can be unambiguously flattened."""

        return self._tensors


################################################################################
# Construction helper functions
################################################################################

def _new_symppnn(symppnn, cls, args, kwargs):
    pfn = (symppnn is PFN)
    if cls is symppnn:
        if kwargs.get('num_global_features') is not None:
            return super(symppnn, cls).__new__(PFNGlobalFeatures if pfn else EFNGlobalFeatures)
        else:
            return super(symppnn, cls).__new__(symppnn)
    else:
        return super(symppnn, cls).__new__(cls)


################################################################################
# EFN - Energy flow network class
################################################################################

class EFN(SymmetricPerParticleNN):

    """Energy Flow Network (EFN) architecture."""

    # customize which EFN instance is created
    def __new__(cls, *args, **kwargs):
        return _new_symppnn(EFN, cls, args, kwargs)

    def _construct_inputs(self):

        # construct input tensors
        self._inputs = construct_weighted_input(self.input_dim, 
                                           zs_name=self._proc_name('zs_input'), 
                                           phats_name=self._proc_name('phats_input'))

        # construct weight tensor and begin list of layers
        self._layers, self._weights = construct_efn_weight_mask(self.inputs[0], 
                                                                mask_val=self.mask_val, 
                                                                name=self._proc_name('mask'))

        # begin list of tensors with the inputs
        self._tensors = [self.weights] + self.inputs

    @property
    def inputs(self):
        """List of input tensors to the model. EFNs have two input tensors:
        `inputs[0]` corresponds to the `zs` input and `inputs[1]` corresponds
        to the `phats` input.
        """

        return self._inputs

    @property
    def weights(self):
        """Weight tensor for the model. This is the `zs` input where entries
        equal to `mask_val` have been set to zero.
        """

        return self._weights

    # eval_filters(patch, n=100, prune=True)
    def eval_filters(self, patch, n=100, prune=True):
        """Evaluates the latent space filters of this model on a patch of the 
        two-dimensional geometric input space.

        **Arguments**

        - **patch** : {_tuple_, _list_} of _float_
            - Specifies the patch of the geometric input space to be evaluated.
            A list of length 4 is interpretted as `[xmin, ymin, xmax, ymax]`.
            Passing a single float `R` is equivalent to `[-R,-R,R,R]`.
        - **n** : {_tuple_, _list_} of _int_
            - The number of grid points on which to evaluate the filters. A list 
            of length 2 is interpretted as `[nx, ny]` where `nx` is the number of
            points along the x (or first) dimension and `ny` is the number of points
            along the y (or second) dimension.
        - **prune** : _bool_
            - Whether to remove filters that are all zero (which happens sometimes
            due to dying ReLUs).

        **Returns**

        - (_numpy.ndarray_, _numpy.ndarray_, _numpy.ndarray_)
            - Returns three arrays, `(X, Y, Z)`, where `X` and `Y` have shape `(nx, ny)` 
            and are arrays of the values of the geometric inputs in the specified patch.
            `Z` has shape `(num_filters, nx, ny)` and is the value of the different
            filters at each point.
        """

        # determine patch of xy space to evaluate filters on
        if isinstance(patch, (float, int)):
            if patch > 0:
                xmin, ymin, xmax, ymax = -patch, -patch, patch, patch
            else:
                ValueError('patch must be positive when passing as a single number.')
        else:
            xmin, ymin, xmax, ymax = patch

        # determine number of pixels in each dimension
        if isinstance(n, int):
            nx = ny = n
        else:
            nx, ny = n

        # construct grid of inputs
        xs, ys = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        XY = np.asarray([X, Y]).reshape((1, 2, nx*ny)).transpose((0, 2, 1))

        # handle weirdness of Keras/tensorflow
        old_keras = (keras_version_tuple <= (2, 2, 5))
        s = self.Phi_sizes[-1] if len(self.Phi_sizes) else self.input_dim
        in_t, out_t = self.inputs[1], self._tensors[self._tensor_inds['latent'][0] - 1]

        # construct function
        kf = K.function([in_t] if old_keras else in_t, [out_t] if old_keras else out_t)

        # evaluate function
        Z = kf([XY] if old_keras else XY)[0].reshape(nx, ny, s).transpose((2, 0, 1))

        # prune filters that are off
        if prune:
            return X, Y, Z[[not (z == 0).all() for z in Z]]
        
        return X, Y, Z


################################################################################
# PFN - Particle flow network class
################################################################################

class PFN(SymmetricPerParticleNN):

    """Particle Flow Network (PFN) architecture. Accepts the same 
    hyperparameters as the [`EFN`](#EFN)."""

    # customize which PFN instance is created
    def __new__(cls, *args, **kwargs):
        return _new_symppnn(PFN, cls, args, kwargs)

    # PFN(*args, **kwargs)
    def _construct_inputs(self):
        """""" # need this for autogen docs

        # construct input tensor
        self._inputs = construct_input(self.input_dim, name=self._proc_name('input'))

        # construct weight tensor and begin list of layers
        self._layers, self._weights = construct_pfn_weight_mask(self.inputs[0], 
                                                                mask_val=self.mask_val, 
                                                                name=self._proc_name('mask'))

        # begin list of tensors with the inputs
        self._tensors = [self.weights] + self.inputs

    @property
    def inputs(self):
        """List of input tensors to the model. PFNs have one input tensor
        corresponding to the `ps` input.
        """

        return self._inputs

    @property
    def weights(self):
        """Weight tensor for the model. A weight of `0` is assigned to any
        particle which has all features equal to `mask_val`, and `1` is
        assigned otherwise.
        """

        return self._weights

################################################################################
# Mixin class for concatenating features to F (eventually there may be more mixins)
################################################################################

class GlobalFeaturesMixin(object):

    def _construct_inputs(self):

        # do normal construction of inputs
        super(GlobalFeaturesMixin, self)._construct_inputs()

        # get new input tensor and insert it at position 1 in tensors list
        self.inputs.extend(construct_input(self.num_global_features, nnone=1,
                                           name=self._proc_name('num_global_features')))
        self.tensors.insert(1, self._global_feature_tensor)

    def _construct_latent(self):

        # do normal construction of the latent layer
        super(GlobalFeaturesMixin, self)._construct_latent()

        # add concatenate layer and tensor to respective lists
        self.layers.append(Concatenate(axis=-1, name=self._proc_name('concat')))
        self.tensors.append(self.layers[-1]([self.tensors[-1], self._global_feature_tensor]))

    @property
    def _global_feature_tensor(self):
        return self.inputs[-1]

class PFNGlobalFeatures(GlobalFeaturesMixin, PFN):
    pass

class EFNGlobalFeatures(GlobalFeaturesMixin, EFN):
    pass
