from __future__ import absolute_import, division, print_function

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from energyflow.archs.archbase import ArchBase

__all__ = ['LinearClassifier']

class LinearClassifier(ArchBase):

    def process_hps(self):

        # which type of linear model we're using
        self.linclass_type = self.hps.get('linclass_type', 'lda')

        # LDA hyperparameters
        self.solver = self.hps.get('solver', 'svd')
        self.tol = self.hps.get('tol', 10**-10)

        # logistic regression hyperparameter dictionary
        self.LR_hps = self.hps.get('LR_hps', {})

    def construct_model(self):

        # setup linear model according to linclass_type
        if self.linclass_type == 'lda':
            self._model = LinearDiscriminantAnalysis(solver=self.solver, tol=self.tol)
        elif self.linclass_type == 'lr':
            self._model = LogisticRegression(**self.LR_hps)
        else:
            raise ValueError('linclass_type can only be lda or lr')
