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
import warnings

import numpy as np

from energyflow.archs.archbase import NNBase, _get_act_layer, _import_keras
from energyflow.archs.dnn import construct_dense
from energyflow.utils.arch_utils import PointCloudDataset
from energyflow.utils.generic_utils import iter_or_rep, kwargs_check

__all__ = [

    # input constructor functions
    #'construct_point_cloud_weighted_inputs', 'construct_point_cloud_inputs',

    # weight mask constructor functions
    #'construct_weighted_point_cloud_mask', 'construct_point_cloud_mask',

    # network consstructor functions
    #'construct_distributed_dense', 'construct_latent', 'construct_dense', 

    # full model classes
    'EFN', 'PFN'
]

################################################################################
# Keras 2.2.5 fixes bug in 2.2.4 that affects our usage of the Dot layer
################################################################################

def _keras_version_tuple():
    from tensorflow.keras import __version__ as __keras_version__
    if __keras_version__.endswith('-tf'):
        __keras_version__ = __keras_version__[:-3]
    return tuple(map(int, __keras_version__.split('.')))

def _dot_axis():
    return 0 if _keras_version_tuple() <= (2, 2, 4) else 1

################################################################################
# Input Functions
################################################################################

def construct_point_cloud_weighted_inputs(*input_dims, **kwargs):
    """Builds the input tensors for multiple weighted point cloud inputs with
    different dimensions.

    **Arguments**

    - ***input_dims** : arbitrary position arguments of type _int_
        - The dimensions of the different point clouds, not including the weight
        dimension.
    - **zs_names** : _list_ of _str_ or `None`
        - The names of the input weight tensors, or None to use default names.
    - **ps_names** : _list_ of _str_ or `None`
        - The names of the input point tensors, or None to use default names.

    **Returns**

    - _list_ of _tensorflow.keras.Input_ tensors.
    """

    _import_keras(globals())

    zs_names, ps_names = kwargs.pop('zs_names', None), kwargs.pop('ps_names', None)
    kwargs_check('construct_point_cloud_weighted_inputs', kwargs)

    # handle names
    if zs_names is None:
        zs_names = len(input_dims)*[None]
    elif (not isinstance(zs_names, list)) or len(input_dims) != len(zs_names):
        raise ValueError('zs_names must be a list of the same length as number of input_dims')
    if ps_names is None:
        ps_names = len(input_dims)*[None]
    elif (not isinstance(ps_names, list)) or len(input_dims) != len(ps_names):
        raise ValueError('ps_names must be a list of the same length as number of  input_dims')

    inputs = []
    for i,input_dim in enumerate(input_dims):
        inputs.append(keras.layers.Input(batch_shape=(None, None), name=zs_names[i]))
        inputs.append(keras.layers.Input(batch_shape=(None, None, input_dim), name=ps_names[i]))

    return inputs

def construct_point_cloud_inputs(*input_dims, **kwargs):
    """Builds the input tensors for multiple point cloud inputs with different
    dimensions.

    **Arguments**

    - ***input_dims** : arbitrary position arguments of type _int_
        - The dimensions of the different point clouds, not including the weight
        dimension.
    - **names** : _list_ of _str_ or `None`
        - The names of the input tensors for the weights, or None to use default
        names.

    **Returns**

    - _list_ of _tensorflow.keras.Input_ tensors.
    """

    _import_keras(globals())

    names = kwargs.pop('names', None)
    kwargs_check('construct_point_cloud_inputs', kwargs)

    if names is None:
        names = len(input_dims)*[None]
    elif len(input_dims) != len(names):
        raise ValueError('names must be the same length as the number of input dims')

    # construct input tensors
    return [keras.layers.Input(batch_shape=(None, None, input_dim), name=name)
            for input_dim,name in zip(input_dims, names)]


################################################################################
# Weight Mask Functions
################################################################################

def construct_weighted_point_cloud_mask(input_tensors, mask_val=0., name=None):
    """"""

    _import_keras(globals())

    mask_layer = keras.layers.Lambda(lambda X: X * K.cast(K.not_equal(X, mask_val), K.dtype(X)), name=name)

    # return layer and tensors
    return [mask_layer], [mask_layer(input_tensor) for input_tensor in input_tensors]

def construct_point_cloud_mask(input_tensors, mask_val=0., name=None, coeffs=None):
    """"""

    _import_keras(globals())

    if coeffs is None:
        mask_layer = keras.layers.Lambda(lambda X: K.cast(K.any(K.not_equal(X, mask_val), axis=-1), K.dtype(X)), name=name)
        return [mask_layer], [mask_layer(input_tensor) for input_tensor in input_tensors]

    else:
        mask_layers, mask_tensors = [], []
        for i, (tensor, coeff) in enumerate(zip(input_tensors, coeffs)):
            name_i = None if name is None else '{}_{}'.format(name, i)
            mask_layer = keras.layers.Lambda(lambda X: coeff*K.cast(K.any(K.not_equal(X, mask_val), axis=-1), K.dtype(X)), name=name_i)
            mask_layers.append(mask_layer)
            mask_tensors.append(mask_layer(tensor))

        return mask_layers, mask_tensors


################################################################################
# Network Construction Functions
################################################################################

def construct_distributed_dense(input_tensor, sizes, acts='relu', k_inits='he_uniform', 
                                                     l2_regs=0., names=None, act_names=None):
    """"""

    _import_keras(globals())

    # repeat options if singletons
    acts, k_inits = iter_or_rep(acts), iter_or_rep(k_inits)
    l2_regs = iter_or_rep(l2_regs)
    names, act_names = iter_or_rep(names), iter_or_rep(act_names)
    
    # list of tensors
    layers, tensors = [], [input_tensor]

    # iterate over specified layers
    z = zip(sizes, acts, k_inits, l2_regs, names, act_names)
    for s, act, k_init, l2_reg, name, act_name in z:
        
        # define a dense layer that will be applied through time distributed
        kwargs = {} 
        if l2_reg > 0.:
            kwargs.update({'kernel_regularizer': keras.regularizers.L2(l2_reg), 'bias_regularizer': keras.regularizers.L2(l2_reg)})
        d_layer = keras.layers.Dense(s, kernel_initializer=k_init, **kwargs)

        # get layers and append them to list
        tdist_layer = keras.layers.TimeDistributed(d_layer, name=name)
        act_layer = _get_act_layer(act, name=act_name)
        layers.extend([tdist_layer, act_layer])

        # get tensors and append them to list
        tensors.append(tdist_layer(tensors[-1]))
        tensors.append(act_layer(tensors[-1]))

    return layers, tensors

def construct_latent(input_tensor, weight_tensor, dropout=0., name=None):
    """"""

    _import_keras(globals())

    # lists of layers and tensors
    layers = [keras.layers.Dot(_dot_axis(), name=name)]
    tensors = [layers[-1]([weight_tensor, input_tensor])]

    # apply dropout if specified
    if dropout > 0.:
        dr_name = None if name is None else '{}_dropout'.format(name)
        layers.append(keras.layers.Dropout(dropout, name=dr_name))
        tensors.append(layers[-1](tensors[-1]))

    return layers, tensors


################################################################################
# SymmetricPointCloudNN - Base class for EFN-like models
################################################################################

class SymmetricPointCloudNN(NNBase):

    # EFN(*args, **kwargs)
    def _process_hps(self):
        r"""See [`ArchBase`](#archbase) for how to pass in hyperparameters as
        well as defaults common to all EnergyFlow neural network models.

        **Required EFN Hyperparameters**

        - **input_dim** : _int_
            - The number of features for each particle. As of version 1.3.0,
            `input_dim` may also be a tuple or list of integers, in which case
            multiple Phi components will be used and their latent spaces
            concatenated (see `additional_input_dims`).
        - **Phi_sizes** (formerly `ppm_sizes`) : {_tuple_, _list_} of _int_
            - The sizes of the dense layers in the per-particle frontend module
            $\Phi$. The last element will be the number of latent observables
            that the model defines.
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
            docs](https://www.tensorflow.org/api_docs/python/tf/keras/
            activations) for more detail.
        - **F_acts**=`'relu'` (formerly `dense_acts`) : {_tuple_, _list_} of
        _str_ or Keras activation
            - Activation functions(s) for the dense layers in the backend module
            $F$. A single string or activation layer will apply the same
            activation to all layers.
        - **Phi_k_inits**=`'he_uniform'` (formerly `ppm_k_inits`) : {_tuple_,
        _list_} of _str_ or Keras initializer
            - Kernel initializers for the dense layers in the per-particle
            frontend module $\Phi$. A single string will apply the same
            initializer to all layers. See the [Keras initializer docs](https://
            www.tensorflow.org/api_docs/python/tf/keras/initializers) for more
            detail.
        - **F_k_inits**=`'he_uniform'` (formerly `dense_k_inits`) : {_tuple_,
        _list_} of _str_ or Keras initializer
            - Kernel initializers for the dense layers in the backend module
            $F$. A single string will apply the same initializer to all layers.
        - **latent_dropout**=`0` : _float_
            - Dropout rates for the summation layer that defines the value of
            the latent observables on the inputs. See the [Keras Dropout layer](
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)
            for more detail.
        - **F_dropouts**=`0` (formerly `dense_dropouts`) : {_tuple_, _list_}
        of _float_
            - Dropout rates for the dense layers in the backend module $F$. A
            single float will apply the same dropout rate to all dense layers.
        - **Phi_l2_regs**=`0` : {_tuple_, _list_} of _float_
            - $L_2$-regulatization strength for both the weights and biases of
            the layers in the $\Phi$ network. A single float will apply the same
            $L_2$-regulatization to all layers.
        - **F_l2_regs**=`0` : {_tuple_, _list_} of _float_
            - $L_2$-regulatization strength for both the weights and biases of
            the layers in the $F$ network. A single float will apply the same
            $L_2$-regulatization to all layers.
        - **mask_val**=`0` : _float_
            - The value for which particles with all features set equal to
            this value will be ignored. The [Keras Masking layer](https://www.
            tensorflow.org/api_docs/python/tf/keras/layers/Masking) does not
            work with the TimeDistributed layer, so this has been implemented in
            a custom manner since version `0.12.0`.

        **Extended EFN Hyperparameters**

        - **additional_input_dims**=`None` : {_tuple_, _list_} of _int_
            - If multiple Phi components are to be used to create several latent
            space embeddings, this list specifies the input dimensions of the
            subsequent input tensors. Note that the same behavior may be induced
            by passing a tuple or list of ints as `input_dim`. If `None`, then
            no additional achitecture components are constructed. If not `None`,
            or if `input_dim` is a tuple or list of length greater than one,
            then the above `Phi` options (as well as `latent_dropout`) are used
            to specify aspects of each Phi component; lists or tuples should be
            used to specify the options for the different architectures. For
            instance, if there is an EFN1 and EFN2 architecture, the the
            `Phi_sizes` are specified as:
        ```python
        Phi_sizes = [(Phi_sizes_EFN1_0, Phi_sizes_EFN1_1, ...), 
                     (Phi_sizes_EFN2_0, Phi_sizes_EFN2_1, ...)]
        ```
        - **num_global_features**=`None` : _int_
            - Number of additional features to be concatenated with the latent
            space observables to form the input to F. If not `None`, then the
            features are to be provided at the end of the list of inputs.
        """

        _import_keras(globals())

        # process generic NN hps
        super()._process_hps()

        # input dimensions
        self.input_dims = self._proc_arg('input_dim')
        self.additional_input_dims = self._proc_arg('additional_input_dims', default=None)
        self.num_global_features = self._proc_arg('num_global_features', default=None)

        # ensure we end up with input_dims as a list
        if isinstance(self.input_dims, int):
            self.input_dims = [self.input_dims]
        elif not isinstance(self.input_dims, (tuple, list)):
            raise TypeError("'input_dim' must be an integer, tuple, or list")
        else:
            self.input_dims = list(self.input_dims)

        # network sizes
        self.Phi_sizes = self._proc_arg('Phi_sizes', old='ppm_sizes')
        self.F_sizes = self._proc_arg('F_sizes', old='dense_sizes')

        # determine if we have multiple Phi components
        self._prepare_multiple_Phis()

        # activations
        self.Phi_acts = self._proc_Phi_arg('Phi_acts', default='relu', old='ppm_acts')
        self.F_acts = iter_or_rep(self._proc_arg('F_acts', default='relu', old='dense_acts'))

        # initializations
        self.Phi_k_inits = self._proc_Phi_arg('Phi_k_inits', default='he_uniform', old='ppm_k_inits')
        self.F_k_inits = iter_or_rep(self._proc_arg('F_k_inits', default='he_uniform', old='dense_k_inits'))

        # regularizations
        self.Phi_l2_regs = self._proc_Phi_arg('Phi_l2_regs', default=0.)
        self.latent_dropout = self._proc_arg('latent_dropout', default=0.)
        self.F_dropouts = iter_or_rep(self._proc_arg('F_dropouts', default=0., old='dense_dropouts'))
        self.F_l2_regs = iter_or_rep(self._proc_arg('F_l2_regs', default=0.))

        # handle latent dropout
        if not isinstance(self.latent_dropout, (tuple, list)):
            self.latent_dropout = self.num_Phi_components*[self.latent_dropout]
        elif len(self.latent_dropout) != self.num_Phi_components:
            raise ValueError('number of latent dropouts does not match number of Phi components')

        # masking
        self.mask_val = self._proc_arg('mask_val', default=0.)
        self.weight_coeffs = iter_or_rep(self._proc_arg('weight_coeffs', default=1.))

        self._verify_empty_hps()

    def _prepare_multiple_Phis(self):

        # form input dimensions
        if self.additional_input_dims is not None:
            if not isinstance(self.additional_input_dims, (tuple, list)):
                self.additional_input_dims = [self.additional_input_dims]
            self.input_dims.extend(self.additional_input_dims) 

        # handle multiple Phis
        self.num_Phi_components = len(self.input_dims)
        if self.num_Phi_components > 1:
            self.additional_input_dims = self.input_dims[1:]

            # check Phi sizes
            for x in self.Phi_sizes:
                if not isinstance(x, (tuple, list)):
                    raise TypeError('multiple Phi components being used - Phi_sizes should be a list of lists')

            self._proc_Phi_arg = self._proc_multiple_Phi_arg

        else:
            if len(self.Phi_sizes) and not isinstance(self.Phi_sizes[0], (tuple, list)):
                self.Phi_sizes = [self.Phi_sizes]
            self._proc_Phi_arg = self._proc_single_Phi_arg

    def _proc_single_Phi_arg(self, name, **kwargs):
        return [iter_or_rep(self._proc_arg(name, **kwargs))]

    def _proc_multiple_Phi_arg(self, name, **kwargs):

        # ensure we have a list of length num_Phi_components
        arg = self._proc_arg(name, **kwargs)
        if not isinstance(arg, (tuple, list)):
            arg = self.num_Phi_components*[arg]
        elif len(arg) != self.num_Phi_components:
            raise ValueError('multiple Phi components being used - {}'.format(name)
                             + ' length should match number of Phi components')

        # process arguments parts, one for each Phi component
        arg = list(arg)
        for i,a in enumerate(arg):
            arg[i] = iter_or_rep(a)

        return arg

    def _construct_model(self):

        # initialize dictionaries for holding indices of subnetworks
        self._layer_inds, self._tensor_inds = {}, {}

        # construct parts of the model
        self._construct_point_cloud_inputs()
        self._construct_global_inputs()
        self._construct_Phi()
        self._construct_latent()
        self._construct_F()

        # get output layers
        out_layer = keras.layers.Dense(self.output_dim, name=self._proc_name('output'))
        act_layer = _get_act_layer(self.output_act, name=self._proc_act_name(self.output_act))
        self._layers.extend([out_layer, act_layer])

        # append output tensors
        self._tensors.append(out_layer(self.tensors[-1]))
        self._tensors.append(act_layer(self.tensors[-1]))

        # construct a new model
        self._model = keras.models.Model(inputs=self.inputs, outputs=self.output, name=self.model_name)

        # compile model
        self._compile_model()

    @abstractmethod
    def _construct_point_cloud_inputs(self):
        pass

    def _construct_global_inputs(self):

        # get new input tensor and insert it at position 1 in tensors list
        if self.num_global_features:
            self.inputs.append(keras.layers.Input(batch_shape=(None, self.num_global_features), 
                                                     name=self._proc_name('num_global_features')))
            self.tensor_inds['global_features'] = len(self.tensors)
            self.tensors.append(self.global_feature_tensor)

    def _construct_Phi(self):

        # iterate over each Phi architecture
        for i in range(self.num_Phi_components):

            # get names
            names = [self._proc_name('tdist_{}_{}'.format(i, j)) for j in range(len(self.Phi_sizes[i]))]
            act_names = [self._proc_act_name(act) for Phi_size,act in zip(self.Phi_sizes[i], self.Phi_acts[i])]

            # determine begin inds
            layer_inds, tensor_inds = [len(self.layers)], [len(self.tensors)]

            # construct Phi
            Phi_layers, Phi_tensors = construct_distributed_dense(self._ps_input_tensors[i], self.Phi_sizes[i], 
                                                        acts=self.Phi_acts[i], k_inits=self.Phi_k_inits[i], 
                                                        l2_regs=self.Phi_l2_regs[i],
                                                        names=names, act_names=act_names)

            # add layers and tensors to internal lists
            self.layers.extend(Phi_layers)
            self.tensors.extend(Phi_tensors)

            # determine end inds
            layer_inds.append(len(self.layers))
            tensor_inds.append(len(self.tensors))

            # store inds
            self.layer_inds['Phi_{}'.format(i)] = tuple(layer_inds)
            self.tensor_inds['Phi_{}'.format(i)] = tuple(tensor_inds)

    def _construct_latent(self):

        # iterate over each Phi architecture
        for i in range(self.num_Phi_components):

            # determine begin inds
            layer_inds, tensor_inds = [len(self.layers)], [len(self.tensors)]

            # construct latent tensors
            ps_tensor = self.tensors[self.tensor_inds['Phi_{}'.format(i)][1] - 1]
            latent_layers, latent_tensors = construct_latent(ps_tensor, self.weights[i],
                                                             dropout=self.latent_dropout[i],
                                                             name=self._proc_name('sum_{}'.format(i)))
            
            # add layers and tensors to internal lists
            self.layers.extend(latent_layers)
            self.tensors.extend(latent_tensors)

            # determine end inds
            layer_inds.append(len(self.layers))
            tensor_inds.append(len(self.tensors))

            # store inds
            self.layer_inds['latent_{}'.format(i)] = tuple(layer_inds)
            self.tensor_inds['latent_{}'.format(i)] = tuple(tensor_inds)

        # get tensors to concatenate
        tensors_to_concat = [latents[-1] for latents in self.latent]
        if self.num_global_features:
            tensors_to_concat.append(self.global_feature_tensor)

        if len(tensors_to_concat) > 1:
            self.layer_inds['concat'] = len(self.layers)
            self.tensor_inds['F_input'] = self._tensor_inds['concat'] = len(self.tensors)
            self.layers.append(keras.layers.Concatenate(axis=-1, name=self._proc_name('concat')))
            self.tensors.append(self.layers[-1](tensors_to_concat))
        else:
            self.tensor_inds['F_input'] = len(self.tensors) - 1

    def _construct_F(self):

        # get names
        names = [self._proc_name('dense_{}'.format(i)) for i in range(len(self.F_sizes))]
        act_names = [self._proc_act_name(act) for F_size,act in zip(self.F_sizes, self.F_acts)]

        # determine begin inds
        layer_inds, tensor_inds = [len(self.layers)], [len(self.tensors)]

        # construct F
        F_layers, F_tensors = construct_dense(self.tensors[self.tensor_inds['F_input']], self.F_sizes,
                                              acts=self.F_acts, k_inits=self.F_k_inits, 
                                              dropouts=self.F_dropouts, l2_regs=self.F_l2_regs,
                                              names=names, act_names=act_names)

        # add layers and tensors to internal lists
        self.layers.extend(F_layers)
        self.tensors.extend(F_tensors)

        # determine end inds
        layer_inds.append(len(self.layers))
        tensor_inds.append(len(self.tensors))

        # store inds
        self.layer_inds['F'] = tuple(layer_inds)
        self.tensor_inds['F'] = tuple(tensor_inds)


    def fit(self, *args, **kwargs):
        prefetch = kwargs.pop('prefetch', None)

        # handle being passed a PointCloudDataset to fit on
        if len(args) and isinstance(args[0], PointCloudDataset):
            kwargs.setdefault('steps_per_epoch', args[0].steps_per_epoch)
            args = (args[0].as_tf_dataset(prefetch=prefetch),) + args[1:]

        # handle validation_data as PointCloudDataset
        if 'validation_data' in kwargs and isinstance(kwargs['validation_data'], PointCloudDataset):
            kwargs.setdefault('validation_steps', kwargs['validation_data'].steps_per_epoch)
            kwargs['validation_data'] = kwargs['validation_data'].as_tf_dataset(prefetch=prefetch, shuffle_override=False)

        return super().fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        
        # handle predicting on a PointCloudDataset
        wrapped = False
        if len(args) and isinstance(args[0], PointCloudDataset):
            kwargs.setdefault('steps', args[0].steps_per_epoch)
            args[0]._init()
            if len(args[0].batch_dtypes) != 1:
                args[0].wrap()
                wrapped = args[0]

            prefetch = kwargs.pop('prefetch', None)
            args = (args[0].as_tf_dataset(prefetch=prefetch, shuffle_override=False),) + args[1:]

        # get predictions
        preds = super().predict(*args, **kwargs)

        # undo wrapping, if we did it
        if wrapped:
            wrapped.unwrap()

        return preds

    @abstractproperty
    def inputs(self):
        pass

    @abstractproperty
    def weights(self):
        pass

    @property
    def _ps_input_tensors(self):
        return [self.inputs[i] for i in self.tensor_inds['ps_inputs']]

    @property
    def global_feature_tensor(self):
        return self.inputs[-1] if self.num_global_features else None

    @property
    def Phi(self):
        r"""List of lists of tensors corresponding to the layers in the $\Phi$
        network(s). The outer list will have length equal to the number of Phi
        components and each sublist contains the tensors for that particular
        component.
        """

        return [self._tensors[slice(*self._tensor_inds['Phi_{}'.format(i)])]
                for i in range(self.num_Phi_components)]

    @property
    def latent(self):
        """List of lists of tensors corresponding to the summation layer(s) in
        the network, including any latent dropout layers if present. The outer
        list will have length equal to the number of Phi components and each
        sublist contains the latent tensors for that particular component.
        """

        return [self._tensors[slice(*self._tensor_inds['latent_{}'.format(i)])]
                for i in range(self.num_Phi_components)]

    @property
    def F(self):
        """List of tensors corresponding to the layers in the $F$ network."""

        return self._tensors[slice(*self._tensor_inds['F'])]

    @property
    def output(self):
        """Output tensor for the model."""

        return self._tensors[-1]

    @property
    def layers(self):
        """List of all layers in the model. Order may be arbitrary since not
        every model can be unambiguously flattened. See also `layer_inds`.
        """

        return self._layers

    @property
    def tensors(self):
        """List of all tensors in the model. Order may be arbitrary since not
        every model can be unambiguously flattened. See also `tensor_inds`.
        """

        return self._tensors

    @property
    def layer_inds(self):
        """Dictionary whose keys are name of layers or groups of layers in the
        network and whose values contain the corresponding indices in the
        `layers` list. Values that are tuples indicate ranges of indices whereas
        lists indicate explicit indices.
        """

        return self._layer_inds

    @property
    def tensor_inds(self):
        """Dictionary whose keys are name of tensors or groups of tensors in the
        network and whose values contain the corresponding indices in the
        `tensors` list. Values that are tuples indicate ranges of indices whereas
        lists indicate explicit indices.
        """

        return self._tensor_inds


################################################################################
# EFN - Energy flow network class
################################################################################

class EFN(SymmetricPointCloudNN):

    """Energy Flow Network (EFN) architecture."""

    def _construct_point_cloud_inputs(self):

        # construct input tensors
        zs_names = [self._proc_name('zs_input_{}'.format(i)) for i in range(self.num_Phi_components)]
        ps_names = [self._proc_name('ps_input_{}'.format(i)) for i in range(self.num_Phi_components)]
        self._inputs = construct_point_cloud_weighted_inputs(*self.input_dims, zs_names=zs_names,
                                                                               ps_names=ps_names)

        # begin list of tensors in the model
        self._tensors = list(self.inputs)
        self.tensor_inds['inputs'] = (0, len(self.inputs))
        self.tensor_inds['zs_inputs'] = [i for i in range(len(self.inputs)) if i % 2 == 0]
        self.tensor_inds['ps_inputs'] = [i for i in range(len(self.inputs)) if i % 2 == 1]

        # construct weight tensor and begin list of layers
        weight_mask_layer, self._weights = construct_weighted_point_cloud_mask(self._zs_input_tensors,
                                                mask_val=self.mask_val, name=self._proc_name('mask'))
        self._layers = weight_mask_layer
        self.layer_inds['weight_mask'] = 0

        # add weights to list of tensors
        self.tensors.extend(self.weights)
        self.tensor_inds['weights'] = (len(self.inputs), len(self.tensors))

    @property
    def inputs(self):
        """List of input tensors to the model. EFNs have two input tensors per
        Phi component: `inputs[2*i]` corresponds to the `zs` input of Phi
        component `i` and `inputs[2*i+1]` corresponds to the `ps` input.
        Additionally, if any global features are present, they are final tensor
        in this list.
        """

        return self._inputs

    @property
    def _zs_input_tensors(self):
        return [self.inputs[i] for i in self.tensor_inds['zs_inputs']]

    @property
    def weights(self):
        """List of weight tensors for the model, one for each Phi component.
        For each of the Phi components, this is the `zs` input where entries
        equal to `mask_val` have been set to zero.
        """

        return self._weights

    # eval_filters(patch, n=100, Phi_i=None, prune=True)
    def eval_filters(self, patch, n=100, Phi_i=None, prune=True):
        """Evaluates the latent space filters of this model on a patch of the 
        two-dimensional geometric input space.

        **Arguments**

        - **patch** : {_tuple_, _list_} of _float_
            - Specifies the patch of the geometric input space to be evaluated.
            A list of length 4 is interpretted as `[xmin, ymin, xmax, ymax]`.
            Passing a single float `R` is equivalent to `[-R,-R,R,R]`.
        - **n** : {_tuple_, _list_} of _int_
            - The number of grid points on which to evaluate the filters. A list 
            of length 2 is interpretted as `[nx, ny]` where `nx` is the number
            of points along the x (or first) dimension and `ny` is the number of
            points along the y (or second) dimension.
        - **prune** : _bool_
            - Whether to remove filters that are all zero (which happens
            sometimes due to dying ReLUs).

        **Returns**

        - (_numpy.ndarray_, _numpy.ndarray_, _numpy.ndarray_)
            - Returns three arrays, `(X, Y, Z)`, where `X` and `Y` have shape
            `(nx, ny)` and are arrays of the values of the geometric inputs in
            the specified patch. `Z` has shape `(num_filters, nx, ny)` and is
            the value of the different filters at each point.
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
        old_keras = _keras_version_tuple() <= (2, 2, 5)
        
        # iterate over latent spaces
        if Phi_i is None:
            Phi_i = range(self.num_Phi_components)
        elif isinstance(Phi_i, int):
            Phi_i = [Phi_i]
            
        Zs = []
        for i in Phi_i:
            nfilt = self.Phi_sizes[i][-1] if len(self.Phi_sizes[i]) else self.input_dims[i]
            in_t, out_t = self._ps_input_tensors[i], self.Phi[i][-1]

            # construct function
            kf = keras.backend.function([in_t] if old_keras else in_t, [out_t] if old_keras else out_t)

            # evaluate function
            Z = kf([XY] if old_keras else XY)[0].reshape(nx, ny, nfilt).transpose((2, 0, 1))
            Zs.append(Z)
            
        Z = np.concatenate(Zs, axis=0)

        # prune filters that are off
        if prune:
            return X, Y, Z[np.any(Z != 0., axis=(1,2))]

        return X, Y, Z


################################################################################
# PFN - Particle flow network class
################################################################################

class PFN(SymmetricPointCloudNN):

    """Particle Flow Network (PFN) architecture. Accepts the same 
    hyperparameters as the [`EFN`](#EFN)."""

    # PFN(*args, **kwargs)
    def _construct_point_cloud_inputs(self):
        """""" # need this for autogen docs

        # construct input tensor
        ps_names = [self._proc_name('ps_input_{}'.format(i)) for i in range(self.num_Phi_components)]
        self._inputs = construct_point_cloud_inputs(*self.input_dims, names=ps_names)

        # begin list of tensors in the model
        self._tensors = list(self.inputs)
        self.tensor_inds['inputs'] = (0, len(self.inputs))
        self.tensor_inds['ps_inputs'] = list(range(self.num_Phi_components))

        # construct weight tensor and begin list of layers
        weight_mask_layer, self._weights = construct_point_cloud_mask(self._ps_input_tensors,
                                                mask_val=self.mask_val, name=self._proc_name('mask'),
                                                coeffs=self.weight_coeffs)
        self._layers = weight_mask_layer
        self.layer_inds['weight_mask'] = list(range(len(self._layers)))

        # add weights to list of tensors
        self.tensors.extend(self.weights)
        self.tensor_inds['weights'] = (len(self.inputs), len(self.tensors))

    @property
    def inputs(self):
        """List of input tensors to the model. PFNs have one input tensor per
        Phi component corresponding to the `ps` input. Additionally, if there
        are any global features present, they are final tensor in this list.
        """

        return self._inputs

    @property
    def weights(self):
        """List of weight tensors for the model, one for each Phi component.
        A weight of `0` is assigned to any particle which has all features
        equal to `mask_val`, and `1` is assigned otherwise.
        """

        return self._weights
