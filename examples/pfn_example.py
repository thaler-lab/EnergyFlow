"""An example involving Particle Flow Networks (PFNs), which were introduced in
[1810.05165](https://arxiv.org/abs/1810.05165). The [`PFN`](../docs/archs/#pfn)
class is used to construct the network architecture. The output of the example
is a plot of the ROC curves obtained by the PFN as well as the jet mass and
constituent multiplicity observables.
"""

#  _____  ______ _   _ 
# |  __ \|  ____| \ | |
# | |__) | |__  |  \| |
# |  ___/|  __| | . ` |
# | |    | |    | |\  |
# |_|    |_|    |_| \_|
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

# energyflow
import energyflow as ef

################################### SETTINGS ###################################
# the commented values correspond to those in 1810.05165
###############################################################################

# data controls, can go up to 2000000 for full dataset
train, val, test = 75000, 10000, 15000
# train, val, test = 1000000, 200000, 200000
use_pids = True
use_global_features = False

# network architecture parameters
Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)
# Phi_sizes, F_sizes = (100, 100, 256), (100, 100, 100)

# network training parameters
num_epoch = 1
batch_size = 250

################################################################################

# load data
ncol = 4 if use_pids else 3
X, y = ef.qg_jets.load(train + val + test, ncol=ncol, pad=False)

# convert labels to categorical
Y = ef.utils.to_categorical(y, num_classes=2)

print('Loaded quark and gluon jets')

# preprocess by centering jets and rescaling pts to O(1) numbers
global_features = np.asarray([ef.sum_ptyphims(x) for x in X]) if use_global_features else np.random.rand(len(X),4)
global_features[:,(0,3)] /= 500.
for x in X:
    yphi_avg = np.average(x[:,1:3], weights=x[:,0], axis=0)
    x[:,1:3] -= yphi_avg
    x[:,0] /= 100.

# handle particle id channel
if use_pids:
    ef.utils.remap_pids(X, pid_i=3)

print('Finished preprocessing')

# do train/val/test split 
(X_train, X_val, X_test,
 Y_train, Y_val, Y_test,
 g_train, g_val, g_test) = ef.utils.data_split(X, Y, global_features, val=val, test=test)

print('Done train/val/test split')
print('Model summary:')

# build architecture
pfn = ef.archs.PFN(input_dim=ncol, Phi_sizes=Phi_sizes, F_sizes=F_sizes,
                   num_global_features=(4 if use_global_features else None))

# specify inputs
d_train = [ef.utils.pad_events(X_train)] + ([g_train] if use_pids else [])
d_val = [ef.utils.pad_events(X_val)] + ([g_val] if use_pids else [])
d_test = [ef.utils.pad_events(X_test)] + ([g_test] if use_pids else [])

# train model
pfn.fit(d_train, Y_train,
        epochs=num_epoch,
        batch_size=batch_size,
        validation_data=(d_val, Y_val),
        verbose=1)

# get predictions on test data
preds = pfn.predict(d_test, batch_size=1000)

# get ROC curve
pfn_fp, pfn_tp, threshs = sklearn.metrics.roc_curve(Y_test[:,1], preds[:,1])

# get area under the ROC curve
auc = sklearn.metrics.roc_auc_score(Y_test[:,1], preds[:,1])
print()
print('PFN AUC:', auc)
print()

if len(sys.argv) == 1 or bool(sys.argv[1]):

    # get multiplicity and mass for comparison
    masses = np.asarray([ef.ms_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in X])
    mults = np.asarray([np.count_nonzero(x[:,0]) for x in X])
    mass_fp, mass_tp, threshs = sklearn.metrics.roc_curve(Y[:,1], -masses)
    mult_fp, mult_tp, threshs = sklearn.metrics.roc_curve(Y[:,1], -mults)

    # some nicer plot settings 
    plt.rcParams['figure.figsize'] = (4,4)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True

    # plot the ROC curves
    plt.plot(pfn_tp, 1-pfn_fp, '-', color='black', label='PFN')
    plt.plot(mass_tp, 1-mass_fp, '-', color='blue', label='Jet Mass')
    plt.plot(mult_tp, 1-mult_fp, '-', color='red', label='Multiplicity')

    # axes labels
    plt.xlabel('Quark Jet Efficiency')
    plt.ylabel('Gluon Jet Rejection')

    # axes limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # make legend and show plot
    plt.legend(loc='lower left', frameon=False)
    plt.show()
