from __future__ import absolute_import, division, print_function

from abc import abstractmethod, abstractproperty

import numpy as np

from keras import backend as K
from keras.layers import Dense, Dot, Dropout, Input, Lambda, TimeDistributed, Masking
from keras.models import Model
from keras.regularizers import l2

from energyflow.archs.archbase import NNBase, _apply_act
from energyflow.utils import iter_or_rep

__all__ = ['EFN', 'PFN']


###############################################################################
# SymmetricPerParticleNN - Base class for EFN-like models
###############################################################################
class SymmetricPerParticleNN(NNBase):

    # EFN(*args, **kwargs)
    def _process_hps(self):
        r"""See [`ArchBase`](#archbase) for how to pass in hyperparameters as
        well as hyperparameters common to all EnergyFlow neural network models.

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
        - **mask_val**=`0` : _float_
            - The value for which particles with all features set equal to
            this value will be ignored. See the [Keras Masking layer](https://
            keras.io/layers/core/#masking) for more detail.
        """

        # process generic NN hps
        super(SymmetricPerParticleNN, self).process_hps()

        # required hyperparameters
        self.input_dim = self._proc_arg('input_dim')
        self.Phi_sizes = self._proc_arg('Phi_sizes', old='ppm_sizes')
        self.F_sizes = self._proc_arg('F_sizes', old='dense_sizes')

        # activations
        self.Phi_acts = iter_or_rep(self._proc_arg('Phi_acts', default='relu', old='ppm_acts'))
        self.F_acts = iter_or_rep(self._proc_arg('F_acts', default='relu', old='dense_acts'))

        # initializations
        self.Phi_k_inits = iter_or_rep(self._proc_arg('Phi_k_inits', default='he_uniform', old='ppm_k_inits'))
        self.F_k_inits = iter_or_rep(self._proc_arg('F_k_inits', default='he_uniform', old='dense_k_inits'))

        # regularizations
        #self.ppm_dropouts = iter_or_rep(self.hps.pop('ppm_dropouts', 0))
        self.latent_dropout = self._proc_arg('latent_dropout', default=0)
        self.F_dropouts = iter_or_rep(self._proc_arg('F_dropouts', default=0, old='dense_dropouts'))

        # masking
        self.mask_val = self._proc_arg('mask_val', default=0.)

        self._verify_empty_hps()

    def _construct_model(self):

        # construct earlier parts of the model
        self._construct_inputs()
        self._construct_Phi()
        self._construct_latent()
        self._construct_F()

        # output layer, applied to the last backend layer
        out_name = self._proc_name('output')
        d_tensor = Dense(self.output_dim, name=out_name)(self._F[-1])
        self._output = _apply_act(self.output_act, d_tensor)

        # construct a new model
        self._model = Model(inputs=self.inputs, outputs=self.output)

        # compile model
        self._compile_model()

    @abstractmethod
    def _construct_inputs(self):
        pass

    def _construct_Phi(self):

        # a list of the per-particle layers
        self._Phi = [self.inputs[-1]]

        # iterate over specified layers
        for i,(s, act, k_init) in enumerate(zip(self.Phi_sizes, self.Phi_acts, self.Phi_k_inits)):

            # define a dense layer that will be applied through time distributed
            d_layer = Dense(s, kernel_initializer=k_init)

            # append time distributed layer to list of ppm layers
            td_name = self._proc_name('tdist_'+str(i))
            tdist_tensor = TimeDistributed(d_layer, name=td_name)(self._Phi[-1])
            self._Phi.extend([tdist_tensor, _apply_act(act, tdist_tensor)])

    def _construct_latent(self):

        name = self._proc_name('sum')
        self._latent = [Dot(0, name=name)([self.weights, self._Phi[-1]])]
        if self.latent_dropout > 0.:
            ld_name = self._proc_name('latent_dropout')
            self._latent.append(Dropout(self.latent_dropout, name=ld_name)(self._latent[0]))

    def _construct_F(self):
        
        # a list of backend layers
        self._F = [self.latent[-1]]

        # iterate over specified backend layers
        z = zip(self.F_sizes, self.F_acts, self.F_k_inits, self.F_dropouts)
        for i,(s, act, k_init, dropout) in enumerate(z):

            # a new dense layer
            d_name = self._proc_name('dense_'+str(i))
            d_tensor = Dense(s, kernel_initializer=k_init, name=d_name)(self._F[-1])
            act_tensor = _apply_act(act, d_tensor)
            self._F.extend([d_tensor, act_tensor])

            # apply dropout (does nothing if dropout is zero)
            if dropout > 0.:
                dr_name = self._proc_name('dropout_'+str(i))
                dr_tensor = Dropout(dropout, name=dr_name)(act_tensor)
                self._F.append(dr_tensor)

    @abstractproperty
    def inputs(self):
        pass

    @abstractproperty
    def weights(self):
        pass

    @property
    def Phi(self):
        r"""List of tensors corresponding to the layers in the $\Phi$
        network."""

        return self._Phi[1:]

    @property
    def latent(self):
        """List of tensors corresponding to the summation layer in the
        network, including any dropout layer if present.
        """

        return self._latent

    @property
    def F(self):
        """List of tensors corresponding to the layers in the $F$ network."""

        return self._F[1:]

    @property
    def output(self):
        """Output tensor for the model."""

        return self._output


###############################################################################
# EFN - Energy flow network class
###############################################################################

class EFN(SymmetricPerParticleNN):

    """Energy Flow Network (EFN) architecture."""

    def _construct_inputs(self):

        # make input tensors
        zs_input = Input(batch_shape=(None, None), name=self._proc_name('zs_input'))
        phats_input = Input(batch_shape=(None, None, self.input_dim), 
                            name=self._proc_name('phats_input'))
        self._inputs = [zs_input, phats_input]

        # make weight tensor
        self._weights = Lambda(self._efn_mask, name=self._proc_name('mask'))(self.inputs[0])

    def _efn_mask(self, x):
        return x * K.cast(K.not_equal(x, self.mask_val), K.dtype(x))

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
        XY = np.asarray([X, Y]).reshape((2, nx*ny)).T

        # construct function 
        kf = K.function([self.inputs[1]], [self._Phi[-1]])

        # evaluate function
        s = self.Phi_sizes[-1] if len(self.Phi_sizes) else self.input_dim
        Z = kf([[XY]])[0][0].reshape(nx, ny, s).transpose((2,0,1))

        # prune filters that are off
        if prune:
            return X, Y, Z[[not (z == 0).all() for z in Z]]
        
        return X, Y, Z


###############################################################################
# PFN - Particle flow network class
###############################################################################

class PFN(SymmetricPerParticleNN):

    """Particle Flow Network (PFN) architecture. Accepts the same 
    hyperparameters as the [`EFN`](#EFN)."""

    # PFN(*args, **kwargs)
    def _construct_inputs(self):
        """""" # need this for autogen docs

        # construct inputs
        self._inputs = [Input(batch_shape=(None, None, self.input_dim), 
                              name=self._proc_name('input'))]

        self._weights = Lambda(self._pfn_mask, name=self._proc_name('mask'))(self.inputs[0])

    def _pfn_mask(self, x):
        return K.cast(K.any(K.not_equal(x, self.mask_val), axis=-1), K.dtype(x))

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
