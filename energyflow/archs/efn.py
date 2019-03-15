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
        r"""See [`ArchBase`](#archbase) for how to pass in hyperparameters.

        **Required EFN Hyperparameters**

        - **input_dim** : _int_
            - The number of features for each particle.
        - **Phi_sizes** : {_tuple_, _list_} of _int_
            - The sizes of the dense layers in the per-particle frontend
            module $\Phi$. The last element will be the number of latent 
            observables that the model defines.
        - **F_sizes** : {_tuple_, _list_} of _int_
            - The sizes of the dense layers in the backend module $F$.

        **Default EFN Hyperparameters**

        - **Phi_acts**=`'relu'` : {_tuple_, _list_} of _str_ or Keras activation
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
        - **F_acts**=`'relu'` : {_tuple_, _list_} of _str_ or Keras activation
            - Activation functions(s) for the dense layers in the 
            backend module $F$. A single string or activation layer will apply
            the same activation to all layers.
        - **Phi_k_inits**=`'he_uniform'` : {_tuple_, _list_} of _str_
            - Kernel initializers for the dense layers in the per-particle
            frontend module $\Phi$. A single string will apply the same initializer 
            to all layers. See the [Keras initializer docs](https://
            keras.io/initializers/) for more detail.
        - **F_k_inits**=`'he_uniform'` : {_tuple_, _list_} of _str_
            - Kernel initializers for the dense layers in the backend 
            module $F$. A single string will apply the same initializer 
            to all layers.
        - **latent_dropout**=`0` : _float_
            - Dropout rates for the summation layer that defines the
            value of the latent observables on the inputs. See the [Keras
            Dropout layer](https://keras.io/layers/core/#dropout) for more 
            detail.
        - **F_dropouts**=`0` : {_tuple_, _list_} of _float_
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

    @abstractmethod
    def _construct_inputs(self):
        pass

    def _construct_Phi_network(self):

        # a list of the per-particle layers, starting with the masking layer operating on input 0
        masking_name = self._proc_name('mask_0')
        self._Phi_tensors = [Masking(mask_value=self.mask_val, 
                                     name=masking_name)(self.input_tensors[-1])]

        # iterate over specified layers
        for i,(s, act, k_init) in enumerate(zip(self.Phi_sizes, self.Phi_acts, self.Phi_k_inits)):

            # define a dense layer that will be applied through time distributed
            d_layer = Dense(s, kernel_initializer=k_init)

            # append time distributed layer to list of ppm layers
            td_name = self._proc_name('tdist_'+str(i))
            tdist_tensor = TimeDistributed(d_layer, name=td_name)(self.Phi_tensors[-1])
            self._Phi_tensors.extend([tdist_tensor, _apply_act(act, tdist_tensor)])

    @abstractmethod
    def _construct_latent_tensor(self):
        pass

    def _construct_F_network(self):
        
        # a list of backend layers
        self._F_tensors = [self.latent_tensor]

        # iterate over specified backend layers
        z = zip(self.F_sizes, self.F_acts, self.F_k_inits, self.F_dropouts)
        for i,(s, act, k_init, dropout) in enumerate(z):

            # a new dense layer
            d_name = self._proc_name('dense_'+str(i))
            d_tensor = Dense(s, kernel_initializer=k_init, name=d_name)(self.F_tensors[-1])
            act_tensor = _apply_act(act, d_tensor)
            self._F_tensors.extend([d_tensor, act_tensor])

            # apply dropout (does nothing if dropout is zero)
            if dropout > 0.:
                dr_name = self._proc_name('dropout_'+str(i))
                dr_tensor = Dropout(dropout, name=dr_name)(act_tensor)
                self._F_tensors.append(dr_tensor)

    def _construct_model(self):

        # construct earlier parts of the model
        self._construct_inputs()
        self._construct_Phi_network()
        self._construct_latent_tensor()
        self._construct_F_network()

        # output layer, applied to the last backend layer
        out_name = self._proc_name('output')
        d_tensor = Dense(self.output_dim, name=out_name)(self.F_tensors[-1])
        self._output_tensor = _apply_act(self.output_act, d_tensor)

        # construct a new model
        self._model = Model(inputs=self.input_tensors, outputs=self.output_tensor)

        # compile model
        self._compile_model()

    @abstractproperty
    def input_tensors(self):
        pass

    @property
    def Phi_tensors(self):
        return self._Phi_tensors

    @abstractproperty
    def latent_tensor(self):
        pass

    @property
    def F_tensors(self):
        return self._F_tensors

    @property
    def output_tensor(self):
        return self._output_tensor


###############################################################################
# EFN - Energy flow network class
###############################################################################

class EFN(SymmetricPerParticleNN):

    """Energy Flow Network (EFN) architecture."""

    def _construct_inputs(self):

        zs_input = Input(batch_shape=(None, None), name=self._proc_name('zs_input'))
        phats_input = Input(batch_shape=(None, None, self.input_dim), 
                            name=self._proc_name('phats_input'))
        self._input_tensors = [zs_input, phats_input]

    def _construct_latent_tensor(self):

        name = self._proc_name('dot')
        self._latent_tensor = Dot(0, name=name)([self.input_tensors[0], self.Phi_tensors[-1]])
        if self.latent_dropout > 0.:
            ld_name = self._proc_name('latent_dropout')
            self._latent_tensor = Dropout(self.latent_dropout, name=ld_name)(self.latent_tensor)

    @property
    def input_tensors(self):
        return self._input_tensors

    @property
    def latent_tensor(self):
        return self._latent_tensor

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
        kf = K.function([self.input_tensors[1]], [self.Phi_tensors[-1]])

        # evaluate function
        Z = kf([[XY]])[0][0].reshape(nx, ny, self.Phi_sizes[-1]).transpose((2,0,1))

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

        self._input_tensors = [Input(batch_shape=(None, None, self.input_dim), 
                                    name=self._proc_name('input'))]

    def _construct_latent_tensor(self):

        name = self._proc_name('sum')
        self._latent_tensor = Lambda(lambda x: K.sum(x, axis=1), name=name)(self.Phi_tensors[-1])
        if self.latent_dropout > 0.:
            ld_name = self._proc_name('latent_dropout')
            self._latent_tensor = Dropout(self.latent_dropout, name=ld_name)(self._latent_tensor)

    @property
    def input_tensors(self):
        return self._input_tensors

    @property
    def latent_tensor(self):
        return self._latent_tensor
