"""An example involving Particle Flow Networks (PFNs), which were introduced in
[1810.05165](https://arxiv.org/abs/1810.05165). The [`PFN`](../docs/archs/#pfn)
class is used to construct the network architecture. This example is meant to
highlight the usafe of PFNs with Tensorflow datasets, in particular the function
`tf_point_cloud_dataset` which helpfully formats things in the proper way. Like
the bse PFN example, the output is a plot of the ROC curves obtained by the PFN
as well as the jet mass and constituent multiplicity observables.
"""

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
train, val, test = 7500, 1000, 1500
# train, val, test = 1000000, 200000, 200000
use_pids = False
use_global_features = False

# network architecture parameters
Phi_sizes, F_sizes = [(100, 100, 128), (10, 20)], (100, 100, 100)
# Phi_sizes, F_sizes = (100, 100, 256), (100, 100, 100)

# network training parameters
num_epoch = 1
batch_size = 100

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
pfn = ef.archs.PFN(input_dim=(ncol, 2*ncol), Phi_sizes=Phi_sizes, F_sizes=F_sizes, weight_coeffs=(1.0, 0.001))

# construct lists of dataset inputs
X_train = [X_train, ef.archs.PairedPointCloudDataset(X_train).unpack()] + ([g_train] if use_global_features else [])
X_val = [X_val, ef.archs.PairedPointCloudDataset(X_val).unpack()] + ([g_val] if use_global_features else [])
X_test = [X_test, ef.archs.PairedPointCloudDataset(X_test).unpack()] + ([g_test] if use_global_features else [])

# construct point cloud datasets
d_train = ef.archs.PointCloudDataset([X_train, Y_train], batch_size=batch_size)
d_val = ef.archs.PointCloudDataset([X_val, Y_val])
d_test = ef.archs.PointCloudDataset([X_test, Y_test])

print('training', d_train)
print('validation', d_val)
print('testing', d_test)

# train model
pfn.fit(d_train, epochs=num_epoch, validation_data=d_val)

# get predictions on test data
preds = pfn.predict(d_test)

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
