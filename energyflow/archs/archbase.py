r"""# Architectures

Energy Flow Networks (EFNs) and Particle Flow Networks (PFNs) are model
architectures designed for learning from collider events as unordered,
variable-length sets of particles. Both EFNs and PFNs are parameterized by a
learnable per-particle function $\Phi$ and latent space function $F$.

An EFN takes the following form:

$$\text{EFN}=F\left(\sum_{i=1}^M z_i \Phi(\hat p_i)\right)$$

where $z_i$ is a measure of the energy of particle $i$, such as $z_i=p_{T,i}$,
and $\hat p_i$ is a measure of the angular information of particle $i$, such as
$\hat p_i = (y_i,\phi_i)$. Any infrared- and collinear-safe observable can be
parameterized in this form.

A PFN takes the following form:

$$\text{PFN}=F\left(\sum_{i=1}^M \Phi(p_i)\right)$$

where $p_i$ is the information of particle $i$, such as its four-momentum,
charge, or flavor. Any observable can be parameterized in this form. See the
[Deep Sets](https://arxiv.org/abs/1703.06114) framework for additional 
discussion.

Since these architectures are not used by the core EnergyFlow code, and require
the external [TensorFlow](https://www.tensorflow.org) and [scikit-learn](http:
//scikit-learn.org/) libraries, they are not imported by default but must be
explicitly imported, e.g. `from energyflow.archs import *`. EnergyFlow also
contains several additional model architectures for ease of using common models
that frequently appear in the intersection of particle physics and machine
learning.
"""

#           _____   _____ _    _ ____           _____ ______
#     /\   |  __ \ / ____| |  | |  _ \   /\    / ____|  ____|
#    /  \  | |__) | |    | |__| | |_) | /  \  | (___ | |__
#   / /\ \ |  _  /| |    |  __  |  _ < / /\ \  \___ \|  __|
#  / ____ \| | \ \| |____| |  | | |_) / ____ \ ____) | |____
# /_/    \_\_|  \_\\_____|_|  |_|____/_/    \_\_____/|______|

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod, abstractproperty
import gc
import os
import sys
import warnings

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Activation, Layer, LeakyReLU, PReLU, ThresholdedReLU

import six

from energyflow.utils import iter_or_rep

__all__ = ['ArchBase', 'NNBase']

###############################################################################
# ArchBase
###############################################################################

class ArchBase(six.with_metaclass(ABCMeta, object)):

    """Base class for all architectures contained in EnergyFlow. The mechanism of
    specifying hyperparameters for all architectures is described here. Methods
    common to all architectures are documented here. Note that this class cannot
    be instantiated directly as it is an abstract base class.
    """

    # ArchBase(*args, **kwargs)
    def __init__(self, *args, **kwargs):
        """Accepts arbitrary arguments. Positional arguments (if present) are
        dictionaries of hyperparameters, keyword arguments (if present) are 
        hyperparameters directly. Keyword hyperparameters take precedence over
        positional hyperparameter dictionaries.

        **Arguments**

        - ***args** : arbitrary positional arguments
            - Each argument is a dictionary containing hyperparameter (name, value)
            pairs.
        - ***kwargs** : arbitrary keyword arguments
            - Hyperparameters as keyword arguments. Takes precedence over the 
            positional arguments.
        """
        
        # store all options
        self.hps = {}
        for d in args:
            self.hps.update(d)
        self.hps.update(kwargs)

        # process hyperparameters
        self._process_hps()

        # construct model
        self._construct_model()

    def _proc_arg(self, name, **kwargs):
        if 'old' in kwargs and kwargs['old'] in self.hps:
            old = kwargs['old']
            m = ('\'{}\' is deprecated and will be removed in the future, '
                 'use \'{}\' instead.').format(old, name)
            warnings.warn(FutureWarning(m))
            kwargs['default'] = self.hps.pop(old)

        return (self.hps.pop(name, kwargs['default']) if 'default' in kwargs 
                                                      else self.hps.pop(name))

    def _verify_empty_hps(self):

        # hps should be all empty now
        for k in self.hps:
            raise ValueError('unrecognized keyword argument {}'.format(k))

        del self.hps

    @abstractmethod
    def _process_hps(self):
        pass

    @abstractmethod
    def _construct_model(self):
        pass

    # fit(*args, **kwargs)
    @abstractmethod
    def fit(self):
        """Train the model by fitting the provided training dataset and labels.
        Transparently calls the `.fit()` method of the underlying model.

        **Arguments**

        - ***args** : _numpy.ndarray_ or _tensorflow.data.Dataset_
            - Either the `X_train` and `Y_train` NumPy arrays or a TensorFlow
            dataset.
        - **kwargs** : _dict_
            - Keyword arguments passed on to the `.fit()` method of the
            underlying model. Most relevant for neural network models, where the
            [TensorFlow/Keras model docs](https://www.tensorflow.org/api_docs/
            python/tf/keras/Model#fit) contain detailed information on the
            possible arguments.

        **Returns**

        - The return value of the the underlying model's `.fit()` method.
        """

        pass

    # predict(X_test, **kwargs)
    @abstractmethod
    def predict(self):
        """Evaluate the model on a dataset. Note that for the `LinearClassifier`
        this corresponds to the `predict_proba` method of the underlying 
        scikit-learn model.

        **Arguments**

        - **X_test** : _numpy.ndarray_
            - The dataset to evaluate the model on.
        - **kwargs** : _dict_
            - Keyword arguments passed on to the underlying model when
            predicting on a dataset.

        **Returns**

        - _numpy.ndarray_
            - The value of the model on the input dataset.
        """

        pass

    @abstractproperty
    def model(self):
        """The underlying model held by this architecture. Note that accessing
        an attribute that the architecture does not have will resulting in
        attempting to retrieve the attribute from this model. This allows for
        interrogation of the EnergyFlow architecture in the same manner as the
        underlying model.

        **Examples**

        - For neural network models:
            - `model.layers` will return a list of the layers, where 
            `model` is any EnergFlow neural network.
        - For linear models:
            - `model.coef_` will return the coefficients, where `model`
            is any EnergyFlow `LinearClassifier` instance.
        """

        pass

    # pass on unknown attribute lookups to the underlying model
    def __getattr__(self, attr):

        if hasattr(self.model, attr):
            return getattr(self.model, attr)

        else:
            name = self.__class__.__name__
            raise AttributeError("'{}' object has no attribute '{}', ".format(name, attr)
                                 + "check of underlying model failed")


###############################################################################
# NNBase
###############################################################################

class NNBase(ArchBase):        

    def _process_hps(self):
        """**Default NN Hyperparameters**

        Common hyperparameters that apply to all architectures except for
        [`LinearClassifier`](#linearclassifier).

        **Compilation Options**

        - **loss**=`'categorical_crossentropy'` : _str_
            - The loss function to use for the model. See the [Keras loss 
            function docs](https://keras.io/losses/) for available loss
            functions.
        - **optimizer**=`'adam'` : Keras optimizer or _str_
            - A [Keras optimizer](https://keras.io/optimizers/) instance or a
            string referring to one (in which case the default arguments are 
            used).
        - **metrics**=`['accuracy']` : _list_ of _str_
            - The [Keras metrics](https://keras.io/metrics/) to apply to the
            model.
        - **compile_opts**=`{}` : _dict_
            - Dictionary of keyword arguments to be passed on to the
            [`compile`](https://keras.io/models/model/#compile) method of the
            model. `loss`, `optimizer`, and `metrics` (see above) are included
            in this dictionary. All other values are the Keras defaults.

        **Output Options**

        - **output_dim**=`2` : _int_
            - The output dimension of the model.
        - **output_act**=`'softmax'` : _str_ or Keras activation
            - Activation function to apply to the output.

        **Callback Options**

        - **filepath**=`None` : _str_
            - The file path for where to save the model. If `None` then the
            model will not be saved.
        - **save_while_training**=`True` : _bool_
            - Whether the model is saved during training (using the 
            [`ModelCheckpoint`](https://keras.io/callbacks/#modelcheckpoint)
            callback) or only once training terminates. Only relevant if
            `filepath` is set.
        - **save_weights_only**=`False` : _bool_
            - Whether only the weights of the model or the full model are
            saved. Only relevant if `filepath` is set.
        - **modelcheck_opts**=`{'save_best_only':True, 'verbose':1}` : _dict_
            - Dictionary of keyword arguments to be passed on to the
            [`ModelCheckpoint`](https://keras.io/callbacks/#modelcheckpoint)
            callback, if it is present. `save_weights_only` (see above) is
            included in this dictionary. All other arguments are the Keras
            defaults.
        - **patience**=`None` : _int_
            - The number of epochs with no improvement after which the training
            is stopped (using the [`EarlyStopping`](https://keras.io/
            callbacks/#earlystopping) callback). If `None` then no early stopping
            is used.
        - **earlystop_opts**=`{'restore_best_weights':True, 'verbose':1}` : _dict_
            - Dictionary of keyword arguments to be passed on to the
            [`EarlyStopping`](https://keras.io/callbacks/#earlystopping)
            callback, if it is present. `patience` (see above) is included in
            this dictionary. All other arguments are the Keras defaults.

        **Flags**

        - **name_layers**=`True` : _bool_
            - Whether to give the layers of the model explicit names or let
            them be named automatically. One reason to set this to `False`
            would be in order to use parts of this model in another model
            (all Keras layers in a model are required to have unique names).
        - **compile**=`True` : _bool_
            - Whether the model should be compiled or not.
        - **summary**=`True` : _bool_
            - Whether a summary should be printed or not.
        """

        # compilation
        self.compile_opts = {'loss': self._proc_arg('loss', default='categorical_crossentropy'),
                             'optimizer': self._proc_arg('optimizer', default='adam'),
                             'metrics': self._proc_arg('metrics', default=['acc'])}
        self.compile_opts.update(self._proc_arg('compile_opts', default={}))

        # add these attributes for historical reasons
        self.loss = self.compile_opts['loss']
        self.optimizer = self.compile_opts['optimizer']
        self.metrics = self.compile_opts['metrics']

        # output
        self.output_dim = self._proc_arg('output_dim', default=2)
        self.output_act = self._proc_arg('output_act', default='softmax')

        # callbacks
        self.filepath = self._proc_arg('filepath', default=None)
        if self.filepath is not None:
            self.filepath = os.path.expanduser(self.filepath)
        self.save_while_training = self._proc_arg('save_while_training', default=True)
        self.modelcheck_opts = {'save_best_only': True, 'verbose': 1, 
                'save_weights_only': self._proc_arg('save_weights_only', default=False)}
        self.modelcheck_opts.update(self._proc_arg('modelcheck_opts', default={}))
        self.save_weights_only = self.modelcheck_opts['save_weights_only']

        self.earlystop_opts = {'restore_best_weights': True, 'verbose': 1, 
                               'patience': self._proc_arg('patience', default=None)}
        self.earlystop_opts.update(self._proc_arg('earlystop_opts', default={}))
        self.patience = self.earlystop_opts['patience']

        # flags
        self.name_layers = self._proc_arg('name_layers', default=True)
        self.compile = self._proc_arg('compile', default=True)
        self.summary = self._proc_arg('summary', default=True)

    def _add_act(self, act):

        # handle case of act as a layer
        if isinstance(act, Layer):
            self.model.add(act)

        # handle case of act being a string and in ACT_DICT
        elif isinstance(act, six.string_types) and act in ACT_DICT:
            self.model.add(ACT_DICT[act]())

        # default case of regular activation
        else:
            self.model.add(Activation(act))

    def _proc_name(self, name):
        return name if self.name_layers else None

    def _compile_model(self):

        # compile model if specified
        if self.compile: 
            self.model.compile(**self.compile_opts)

            # print summary
            if self.summary: 
                self.model.summary()

    def fit(self, *args, **kwargs):

        # list of callback functions
        callbacks = []

        # do model checkpointing, used mainly to save model during training instead of at end
        if self.filepath and self.save_while_training:
            callbacks.append(ModelCheckpoint(self.filepath, **self.modelcheck_opts))

        # do early stopping, which now also handle loading best weights at the end
        if self.patience is not None:
            callbacks.append(EarlyStopping(**self.earlystop_opts))

        # update any callbacks that were passed with the two we build in explicitly
        kwargs.setdefault('callbacks', []).extend(callbacks)

        # do the fitting
        hist = self.model.fit(*args, **kwargs)

        # handle saving at the end, if we weren't already saving throughout 
        if self.filepath and not self.save_while_training:
            if self.save_weights_only:
                self.model.save_weights(self.filepath)
            else:
                self.model.save(self.filepath)

        # take out the trash
        gc.collect()

        return hist

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    @property
    def model(self):
        if hasattr(self, '_model'):
            return self._model
        else:
            name = self.__class__.__name__
            raise AttributeError("'{}' object has no underlying model".format(name))


###############################################################################
# Activation Functions
###############################################################################

ACT_DICT = {'LeakyReLU': LeakyReLU, 'PReLU': PReLU, 'ThresholdedReLU': ThresholdedReLU}

def _get_act_layer(act):

    # handle case of act as a layer
    if isinstance(act, Layer):
        return act

    # handle case of act being a string and in ACT_DICT
    if isinstance(act, six.string_types) and act in ACT_DICT:
        return ACT_DICT[act]()

    # default case of passing act into layer
    return Activation(act)
