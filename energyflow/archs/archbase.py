from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod, abstractproperty

from keras.optimizers import Adam
from six import with_metaclass

###############################################################################
# ArchBase
###############################################################################

class ArchBase(with_metaclass(ABCMeta, object)):

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

    @abstractmethod
    def fit(self):
        pass

    @abstractproperty
    def predict(self):
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
