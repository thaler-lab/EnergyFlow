"""An example involving Energy Flow Polynomials (EFPs) and a linear classifier
(Fisher's Linear Discriminant by default). First, the [`EFPSet`](../docs/
efp/#efpset) class is used to compute the EFPs up to the specified `dmax`, the
default being `dmax=5`. Then linear classifiers are trained for different
numbers of EFPs as input, determined by taking all EFPs up to degree `d` with
`d` from `1` to `dmax`. The output of the example is a plot of the ROC curves
for the classifiers with different numbers of EFP inputs.
"""

#  ______ ______ _____
# |  ____|  ____|  __ \
# | |__  | |__  | |__) |
# |  __| |  __| |  ___/
# | |____| |    | |
# |______|_|    |_|
#  ________   __          __  __ _____  _      ______
# |  ____\ \ / /    /\   |  \/  |  __ \| |    |  ____|
# | |__   \ V /    /  \  | \  / | |__) | |    | |__
# |  __|   > <    / /\ \ | |\/| |  ___/| |    |  __|
# | |____ / . \  / ____ \| |  | | |    | |____| |____
# |______/_/ \_\/_/    \_\_|  |_|_|    |______|______|

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2020 Patrick T. Komiske III and Eric Metodiev

# standard library imports
from __future__ import absolute_import, division, print_function
import sys

# standard scientific python libraries
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

# energyflow imports
import energyflow as ef

################################### SETTINGS ###################################

# data controls
num_data = 20000
test_frac = 0.2

# efp parameters
dmax = 5
measure = 'hadr'
beta = 0.5

# plotting
colors = ['tab:red', 'tab:orange', 'tab:olive', 'tab:green', 'tab:blue']

################################################################################

# load data
X, y = ef.qg_jets.load(num_data, pad=False)

print('Loaded quark and gluon jets')

# calculate EFPs
print('Calculating d <= {} EFPs for {} jets... '.format(dmax, num_data), end='')
efpset = ef.EFPSet(('d<=', dmax), measure='hadr', beta=beta)
X = efpset.batch_compute(X)
print('Done')

# train models with different numbers of EFPs as input
rocs = []
for d in range(1, dmax+1):

    # build architecture
    model = ef.archs.LinearClassifier(linclass_type='lda')

    # select EFPs with degree <= d
    X_d = X[:,efpset.sel(('d<=', d))]

    # do train/val/test split 
    (X_train, X_test, y_train, y_test) = ef.utils.data_split(X_d, y, val=0, test=test_frac)
    print('Done train/val/test split')

    # train model
    model.fit(X_train, y_train)

    # get predictions on test data
    preds = model.predict(X_test)

    # get ROC curve if we have sklearn
    rocs.append(sklearn.metrics.roc_curve(y_test, preds[:,1]))

    # get area under the ROC curve
    auc = sklearn.metrics.roc_auc_score(y_test, preds[:,1])
    print()
    print('EFPs d <= {} AUC:'.format(d), auc)
    print()

if len(sys.argv) == 1 or bool(sys.argv[1]):

    # some nicer plot settings 
    plt.rcParams['figure.figsize'] = (4,4)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True

    # iterate over the ROC curves and plot them
    for i,d in enumerate(range(1, dmax+1)):
        plt.plot(rocs[i][1], 1-rocs[i][0], '-', color=colors[i], 
                                                label='LDA: d <= {} EFPs'.format(d))

    # axes labels
    plt.xlabel('Quark Jet Efficiency')
    plt.ylabel('Gluon Jet Rejection')

    # axes limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # make legend and show plot
    plt.legend(loc='lower left', frameon=False)
    plt.show()
