from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod, abstractproperty

from keras.optimizers import Adam
from six import with_metaclass

__all__ = [
    'ArchBase',
    'NNBase'
]

###############################################################################
# ArchBase
###############################################################################
class ArchBase(with_metaclass(ABCMeta, object)):

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
        self.process_hps()

        # construct model
        self.construct_model()

    @abstractmethod
    def process_hps(self):
        pass

    @abstractmethod
    def construct_model(self):
        pass

    # fit(X_train, Y_train, **kwargs)
    @abstractmethod
    def fit(self):
        """Train the model by fitting the provided training dataset and labels.
        Transparently calls the `fit()` method of the underlying model.

        **Arguments**

        - **X_train** : _numpy.ndarray_
            - The training dataset as an array of features for each sample.
        - **Y_train** : _numpy.ndarray_
            - The labels for the training dataset. May need to be one-hot encoded
            depending on the requirements of the underlying model (typically Keras
            models will use one-hot encoding whereas the linear model does not.)
        - **kwargs** : _dict_
            - Keyword arguments passed on to the Keras method of the same name.
            See the [Keras model docs](https://keras.io/models/model/#fit) for
            details on available parameters.

        **Returns**

        - Whatever the underlying model's `fit()` returns.
        """

        pass

    # predict(X_test, **kwargs)
    @abstractmethod
    def predict(self):
        """Evaluate the model on a dataset. 

        **Arguments**

        - **X_test** : _numpy.ndarray_
            - The dataset to evaluate the model on.
        - **kwargs** : _dict_
            - Keyword arguments passed on to the Keras method of the same name.
            See the [Keras model docs](https://keras.io/models/model/#fit) for
            details on available parameters.

        **Returns**

        - _numpy.ndarray_
            - The value of the model on the input dataset.
        """

        pass

    @abstractproperty
    def model(self):
        """The underlying model held by this architecture."""

        pass

###############################################################################
# NNBase
###############################################################################
class NNBase(ArchBase):        

    def process_hps(self):
        """**Default NN Hyperparameters**

        Common hyperparameters that apply to all architectures except 
        for [`LinearClassifier`](#linearclassifier).

        - **loss**=`'categorical_crossentropy'` : _str_
            - The loss function to use for the model. See the [Keras
            loss function docs](https://keras.io/losses/) for available
            loss functions.
        - **lr**=`0.001` : _float_
            - The learning rate for the model.
        - **opt**=`Adam` : Keras optimizer
            - A [Keras optimizer](https://keras.io/optimizers/).
        - **output_dim**=`2` : _int_
            - The output dimension of the model.
        - **output_act**=`'softmax'` : _str_
            - Activation function to apply to the output.
        - **metrics**=`['accuracy']` : _list_ of _str_
            - The [Keras metrics](https://keras.io/metrics/) to apply
            to the model.
        - **compile**=`True` : _bool_
            - Whether the model should be compiled or not.
        - **summary**=`True` : _bool_
            - Whether a summary should be printed or not.
        """

        # optimization
        self.loss = self.hps.get('loss', 'categorical_crossentropy')
        self.lr = self.hps.get('lr', 0.001)
        self.opt = self.hps.get('opt', Adam)

        # output
        self.output_dim = self.hps.get('output_dim', 2)
        self.output_act = self.hps.get('output_act', 'softmax')

        # metrics
        self.metrics = self.hps.get('metrics', ['accuracy'])

        # flags
        self.compile = self.hps.get('compile', True)
        self.summary = self.hps.get('summary', True)

    def compile_model(self):

        # compile model if specified
        if self.compile: 
            self.model.compile(loss=self.loss, optimizer=self.opt(lr=self.lr), metrics=self.metrics)

            # print summary
            if self.summary: 
                self.model.summary()

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    @property
    def model(self):
        return self._model
