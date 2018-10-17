from __future__ import absolute_import, division, print_function

from abc import abstractmethod

from keras import backend as K
from keras.layers import Dense, Dot, Dropout, Input, Lambda, TimeDistributed, Masking
from keras.models import Model
from keras.regularizers import l2
import numpy as np

from energyflow.archs.archbase import NNBase
from energyflow.utils import iter_or_rep

__all__ = ['EFN', 'PFN']

###############################################################################
# SymmetricPerParticleNN - Base class for EFN-like models
###############################################################################
class SymmetricPerParticleNN(NNBase):

    # EFN(*args, **kwargs)
    def process_hps(self):
        """See [`ArchBase`](#archbase) for how to pass in hyperparameters.

        **Required EFN Hyperparameters**

        - **input_dim** : _int_
            - The number of features for each particle.
        - **ppm_sizes** : {_tuple_, _list_} of _int_
            - The sizes of the dense layers in the per-particle frontend
            module $\\Phi$. The last element will be the number of latent 
            observables that the model defines.
        - **dense_sizes** : {_tuple_, _list_} of _int_
            - The sizes of the dense layers in the backend module $F$.

        **Default EFN Hyperparameters**

        - **ppm_acts**=`'relu'` : {_tuple_, _list_} of _str_
            - Activation functions(s) for the dense layers in the 
            per-particle frontend module $\\Phi$. A single string will apply 
            the same activation to all layers. See the [Keras activations 
            docs](https://keras.io/activations/) for more detail.
        - **dense_acts**=`'relu'` : {_tuple_, _list_} of _str_
            - Activation functions(s) for the dense layers in the 
            backend module $F$. A single string will apply  the same activation 
            to all layers.
        - **ppm_k_inits**=`'he_uniform'` : {_tuple_, _list_} of _str_
            - Kernel initializers for the dense layers in the per-particle
            frontend module $\\Phi$. A single string will apply the same initializer 
            to all layers. See the [Keras initializer docs](https://
            keras.io/initializers/) for more detail.
        - **dense_k_inits**=`'he_uniform'` : {_tuple_, _list_} of _str_
            - Kernel initializers for the dense layers in the backend 
            module $F$. A single string will apply the same initializer 
            to all layers.
        - **latent_dropout**=`0` : _float_
            - Dropout rates for the summation layer that defines the
            value of the latent observables on the inputs. See the [Keras
            Dropout layer](https://keras.io/layers/core/#dropout) for more 
            detail.
        - **dense_dropouts**=`0` : {_tuple_, _list_} of _float_
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
        #self.ppm_dropouts = iter_or_rep(self.hps.get('ppm_dropouts', 0))
        self.latent_dropout = self.hps.get('latent_dropout', 0)
        self.dense_dropouts = iter_or_rep(self.hps.get('dense_dropouts', 0))

        # masking
        self.mask_val = self.hps.get('mask_val', 0.)

    @abstractmethod
    def construct_input_layers(self):
        pass

    def construct_per_particle_module(self):

        # a list of the per-particle layers, starting with the masking layer operating on input 0
        self.ppm_layers = [Masking(mask_value=self.mask_val, name='mask_0')(self.input_layers[-1])]

        # iterate over specified layers
        for i,(s, act, k_init) in enumerate(zip(self.ppm_sizes, self.ppm_acts, self.ppm_k_inits)):

            # define a dense layer that will be applied through time distributed
            d_layer = Dense(s, activation=act, kernel_initializer=k_init)

            # append time distributed layer to list of ppm layers
            self.ppm_layers.append(TimeDistributed(d_layer, name='tdist_'+str(i))(self.ppm_layers[-1]))

    @abstractmethod
    def construct_latent_layer(self):
        pass

    def construct_backend_module(self):
        
        # a list of backend layers
        self.backend_layers = [self.latent_layer]

        # iterate over specified backend layers
        z = zip(self.dense_sizes, self.dense_acts, self.dense_k_inits, self.dense_dropouts)
        for i,(s, act, k_init, dropout) in enumerate(z):

            # a new dense layer
            new_layer = Dense(s, activation=act, kernel_initializer=k_init, name='dense_'+str(i))

            # apply dropout if specified 
            if dropout > 0:
                new_layer = Dropout(dropout, name='dropout_'+str(i))(new_layer)

            # apply new layer to previous and append to list
            self.backend_layers.append(new_layer(self.backend_layers[-1]))

    def construct_model(self):

        # construct earlier parts of the model
        self.construct_input_layers()
        self.construct_per_particle_module()
        self.construct_latent_layer()
        self.construct_backend_module()

        # output layer, applied to the last backend layer
        output_layer = Dense(self.output_dim, activation=self.output_act, 
                                              name='output')(self.backend_layers[-1])

        # construct a new model
        self._model = Model(inputs=self.input_layers, outputs=output_layer)

        # compile model
        self.compile_model()

    @property
    def input_layers(self):
        return self._input_layers

    @property
    def latent_layer(self):
        return self._latent_layer


###############################################################################
# EFN - Energy flow network class
###############################################################################

class EFN(SymmetricPerParticleNN):

    """Energy Flow Network (EFN) architecture."""

    def construct_input_layers(self):

        zs_input = Input(batch_shape=(None, None), name='zs_input')
        phats_input = Input(batch_shape=(None, None, self.input_dim), name='phats_input')
        self._input_layers = [zs_input, phats_input]

    def construct_latent_layer(self):

        self._latent_layer = Dot(0, name='dot')([self.input_layers[0], self.ppm_layers[-1]])

        if self.latent_dropout > 0:
            self._latent_layer = Dropout(self.latent_dropout, name='latent_dropout')(self._latent_layer)

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
        kf = K.function([self.input_layers[1]], [self.ppm_layers[-1]])

        # evaluate function
        Z = kf([[XY]])[0][0].reshape(nx, ny, self.ppm_sizes[-1]).transpose((2,0,1))

        # prune filters that are off
        if prune:
            return X, Y, Z[[not (z == 0).all() for z in Z]]
        
        return X, Y, Z


###############################################################################
# PFN - Particle flow network class
###############################################################################

class PFN(SymmetricPerParticleNN):

    """Particle Flow Network (PFN) architecture. Accepts the same 
    hperparameters as the [`EFN`](#EFN)."""

    # PFN(*args, **kwargs)
    def construct_input_layers(self):
        """""" # need this for autogen docs

        self._input_layers = [Input(batch_shape=(None, None, self.input_dim), name='input')]

    def construct_latent_layer(self):

        self._latent_layer = Lambda(lambda x: K.sum(x, axis=1), name='sum')(self.ppm_layers[-1])

        if self.latent_dropout > 0:
            self._latent_layer = Dropout(self.latent_dropout, name='latent_dropout')(self._latent_layer)
