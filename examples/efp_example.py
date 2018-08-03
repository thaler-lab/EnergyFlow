from __future__ import absolute_import, division, print_function

import numpy as np

import energyflow as ef
from energyflow.archs import LinearClassifier
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, to_categorical

# attempt to import sklearn
try:
    from sklearn.metrics import roc_auc_score, roc_curve
except:
    print('please install scikit-learn in order to make ROC curves')
    roc_curve = False

# attempt to import matplotlib
try:
    import matplotlib.pyplot as plt
except:
    print('please install matploltib in order to make plots')
    plt = False


################################### SETTINGS ###################################

# data controls
num_data = 15000
test_frac = 0.2

# efp parameters
dmax = 5
measure = 'hadr'
beta = 1

# plotting
colors = ['tab:red', 'tab:orange', 'tab:olive', 'tab:green', 'tab:blue']

################################################################################

X, y = qg_jets.load(num_data)

print()
print('Loaded quark and gluon jets')

print('Calculating d <= {} EFPs for {} jets... '.format(dmax, num_data), end='')
efpset = ef.EFPSet(('d<=', dmax), measure='hadr', beta=beta)
masked_X = [x[x[:,0] > 0] for x in X]
X = efpset.batch_compute(masked_X)
print('Done')

rocs = []
for d in range(1, dmax+1):

    # build architecture
    model = LinearClassifier(linclass_type='lda')

    # select EFPs with degree <= d
    X_d = X[:,efpset.sel(('d<=', d))]

    # do train/val/test split 
    (X_train, X_test, y_train, y_test) = data_split(X_d, y, val=0, test=test_frac)

    print('Done train/val/test split')

    # train model
    model.fit(X_train, y_train)

    # get ROC curve
    preds = model.predict(X_test)

    if roc_curve:
        rocs.append(roc_curve(y_test, preds[:,1]))
        auc = roc_auc_score(y_test, preds[:,1])
        print()
        print('EFPs d <= {} AUC:'.format(d), auc)
        print()

if plt:

    plt.rcParams['figure.figsize'] = (4,4)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True

    for i,d in enumerate(range(1, dmax+1)):
        plt.plot(rocs[i][1], 1-rocs[i][0], '-', color=colors[i], label='LDA: d <= {} EFPs'.format(d))

    plt.xlabel('Quark Jet Efficiency')
    plt.ylabel('Gluon Jet Rejection')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.legend(loc='lower left', frameon=False)
    plt.show()
