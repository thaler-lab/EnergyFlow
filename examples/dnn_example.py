"""An example involving deep, fully-connected neural networks (DNNs). The
[`DNN`](../docs/archs/#dnn) class is used to construct the network architecture.

The inputs are taken to be the $N$-subjettiness observables as specified as part
of the phase space basis from [1704.08249](https://arxiv.org/abs/1704.08249),
cut off at some total number of observables. The output of the example is a plot
showing the ROC curves obtained from training the DNN on different numbers of
$N$-subjettiness observables.
"""

#  _____  _   _ _   _
# |  __ \| \ | | \ | |
# | |  | |  \| |  \| |
# | |  | | . ` | . ` |
# | |__| | |\  | |\  |
# |_____/|_| \_|_| \_|
#  ________   __          __  __ _____  _      ______
# |  ____\ \ / /    /\   |  \/  |  __ \| |    |  ____|
# | |__   \ V /    /  \  | \  / | |__) | |    | |__
# |  __|   > <    / /\ \ | |\/| |  ___/| |    |  __|
# | |____ / . \  / ____ \| |  | | |    | |____| |____
# |______/_/ \_\/_/    \_\_|  |_|_|    |______|______|

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2022 Patrick T. Komiske III and Eric Metodiev

# standard library imports
from __future__ import absolute_import, division, print_function

# standard numerical library imports
import numpy as np

# energyflow imports
import energyflow as ef
from energyflow.archs import DNN
from energyflow.datasets import qg_nsubs
from energyflow.utils import data_split, to_categorical

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

################################### SETTINGS ###################################

# data controls
num_data = 100000
val_frac, test_frac = 0.1, 0.15

# network architecture parameters
dense_sizes = (100, 100)

# network training parameters
num_epoch = 10
batch_size = 100

# sweep parameters
num_nsubs = [1, 2, 4, 8, 16, 32]
colors = ['tab:red', 'tab:orange', 'tab:olive', 'tab:green', 'tab:blue', 'tab:purple']

################################################################################

# load data
X, y = qg_nsubs.load(num_data=num_data)

# convert labels to categorical
Y = to_categorical(y, num_classes=2)

print('Loaded quark and gluon jets')
print('Model summary:')

# train models with different numbers of nsubs as input
rocs = []
for i,num_nsub in enumerate(num_nsubs):

    # build architecture
    dnn = DNN(input_dim=num_nsub, dense_sizes=dense_sizes, summary=(i==0))

    # do train/val/test split 
    (X_train, X_val, X_test,
     Y_train, Y_val, Y_test) = data_split(X[:,:num_nsub], Y, val=val_frac, test=test_frac)

    print('Done train/val/test split')

    # train model
    dnn.fit(X_train, Y_train,
            epochs=num_epoch,
            batch_size=batch_size,
            validation_data=(X_val, Y_val),
            verbose=1)

    # get predictions on test data
    preds = dnn.predict(X_test, batch_size=1000)

    # get ROC curve if we have sklearn
    if roc_curve:
        rocs.append(roc_curve(Y_test[:,1], preds[:,1]))

        # get area under the ROC curve
        auc = roc_auc_score(Y_test[:,1], preds[:,1])
        print()
        print('{} nsubs DNN AUC:'.format(num_nsub), auc)
        print()

# some nicer plot settings 
plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.autolayout'] = True

# iterate over the ROC curves and plot them
for i in range(len(rocs)):
    plt.plot(rocs[i][1], 1-rocs[i][0], '-', color=colors[i], 
                                            label='DNN: {} N-subs'.format(num_nsubs[i]))

# axes labels
plt.xlabel('Quark Jet Efficiency')
plt.ylabel('Gluon Jet Rejection')

# axes limits
plt.xlim(0, 1)
plt.ylim(0, 1)

# make legend and show plot
plt.legend(loc='lower left', frameon=False)
plt.show()
