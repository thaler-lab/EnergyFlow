from __future__ import absolute_import, division, print_function

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from energyflow.archs.archbase import ArchBase

__all__ = ['LinearClassifier']

###############################################################################
# LinearClassifier
###############################################################################
class LinearClassifier(ArchBase):

    """Linear classifier that can be either Fisher's linear discriminant
    or logistic regression. Relies on the [scikit-learn](https://scikit-learn.org/)
    implementations of these classifiers."""

    # LinearClassifier(*args, **kwargs)
    def process_hps(self):
        """See [`ArchBase`](#archbase) for how to pass in hyperparameters.

        **Default Hyperparameters**

        - **linclass_type**=`'lda'` : {`'lda'`, `'lr'`}
            - Controls which type of linear classifier is used. `'lda'`
            corresponds to [`LinearDisciminantAnalysis`](http://scikit-
            learn.org/stable/modules/generated/sklearn.discriminant_analysis.
            LinearDiscriminantAnalysis.html) and `'lr'` to [`Logistic
            Regression`](http://scikit-learn.org/stable/modules/generated/
            sklearn.linear_model.LogisticRegression.html). If using `'lr'`
            all arguments are passed on directly to the scikit-learn
            class.

        **LDA Hyperparameters**

        - **solver**=`'svd'` : {`'svd'`, `'lsqr'`, `'eigen'`}
            - Which LDA solver to use.
        - **tol**=`1e-10` : _float_
            - Threshold used for rank estimation. Notably not a
            convergence parameter.
        """

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

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict_proba(*args, **kwargs)

    @property
    def model(self):
        return self._model
