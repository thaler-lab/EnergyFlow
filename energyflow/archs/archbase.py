from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod

from keras.optimizers import Adam
from six import add_metaclass

from energyflow.utils import kwargs_check

__all__ = ['NNBase', 'LinearBase']

###############################################################################
# ArchBase
###############################################################################

@add_metaclass(ABCMeta)
class ArchBase:

    def __init__(self, *args, **kwargs):
        
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

    @abstractproperty
    def model(self):
        pass


###############################################################################
# NNBase
###############################################################################

class NNBase(ArchBase):        

    def process_hps(self):

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

    def construct_model(self):
        pass

    @property
    def model(self):
        return self._model


###############################################################################
# LinearBase
###############################################################################

class LinearBase(ArchBase):

    def process_hps(self):
        pass

    def construct_model(self):
        pass

    @property
    def model(self):
        return self._model
