"""An example using Energy Flow Networks (EFNs), which were introduced in
[1810.05165](https://arxiv.org/abs/1810.05165), to classify quark and gluon
jets. The [`EFN`](../docs/archs/#efn) class is used to construct the network
architecture. This example is meant to highlight the usafe of PFNs with
Tensorflow datasets, in particular the function `tf_point_cloud_dataset` which
helpfully formats things in the proper way.  The output of the example is a
plot of the ROC curves obtained by the EFN as well as the jet mass and
constituent multiplicity observables.
"""

# standard library imports
from __future__ import absolute_import, division, print_function
import sys

# standard scientific python libraries
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

# energyflow imports
import energyflow as ef

################################### SETTINGS ##################################
# the commented values correspond to those in 1810.05165
###############################################################################

# data controls, can go up to 2000000 total for full dataset
train, val, test = 75000, 10000, 15000
# train, val, test = 1000000, 200000, 200000
use_global_features = False

# network architecture parameters
Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)
# Phi_sizes, F_sizes = (100, 100, 256), (100, 100, 100)

# network training parameters
num_epoch = 2
batch_size = 100

###############################################################################

# load data
X, y = ef.qg_jets.load(train + val + test, ncol=3, pad=False)

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

print('Finished preprocessing')

# do train/val/test split 
(X_train, X_val, X_test,
 Y_train, Y_val, Y_test,
 g_train, g_val, g_test) = ef.utils.data_split(X, Y, global_features, val=val, test=test)

print('Done train/val/test split')
print('Model summary:')

# build architecture
efn = ef.archs.EFN(input_dim=2, Phi_sizes=Phi_sizes, F_sizes=F_sizes,
                   num_global_features=(4 if use_global_features else None))

# get datasets
if use_global_features:
    d_train = ef.archs.PointCloudDataset([[ef.archs.WeightedPointCloudDataset(X_train), g_train], Y_train],
                                         batch_size=batch_size)
    d_val = ef.archs.PointCloudDataset([[ef.archs.WeightedPointCloudDataset(X_val), g_val], Y_val])
    d_test = ef.archs.PointCloudDataset([ef.archs.WeightedPointCloudDataset(X_test), g_test]).wrap()
else:
    d_train = ef.archs.PointCloudDataset([ef.archs.WeightedPointCloudDataset(X_train), Y_train],
                                         batch_size=batch_size)
    d_val = ef.archs.PointCloudDataset([ef.archs.WeightedPointCloudDataset(X_val), Y_val])
    d_test = ef.archs.WeightedPointCloudDataset(X_test).wrap()

print('training', d_train)
print('validation', d_val)
print('testing', d_test)

# train model
efn.fit(d_train, epochs=num_epoch, validation_data=d_val)

# get predictions on test data
preds = efn.predict(d_test)

# get ROC curve
efn_fp, efn_tp, threshs = sklearn.metrics.roc_curve(Y_test[:,1], preds[:,1])

# get area under the ROC curve
auc = sklearn.metrics.roc_auc_score(Y_test[:,1], preds[:,1])
print()
print('EFN AUC:', auc)
print()

if len(sys.argv) == 1 or bool(sys.argv[1]):

    # some nicer plot settings 
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True

    fig, axes = plt.subplots(1, 2, figsize=(8,4))

    ######################### ROC Curve Plot #########################

    # get multiplicity and mass for comparison
    masses = np.asarray([ef.ms_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in X])
    mults = np.asarray([np.count_nonzero(x[:,0]) for x in X])
    mass_fp, mass_tp, threshs = sklearn.metrics.roc_curve(Y[:,1], -masses)
    mult_fp, mult_tp, threshs = sklearn.metrics.roc_curve(Y[:,1], -mults)

    # plot the ROC curves
    axes[0].plot(efn_tp, 1-efn_fp, '-', color='black', label='EFN')
    axes[0].plot(mass_tp, 1-mass_fp, '-', color='blue', label='Jet Mass')
    axes[0].plot(mult_tp, 1-mult_fp, '-', color='red', label='Multiplicity')

    # axes labels
    axes[0].set_xlabel('Quark Jet Efficiency')
    axes[0].set_ylabel('Gluon Jet Rejection')

    # axes limits
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    # make legend and show plot
    axes[0].legend(loc='lower left', frameon=False)

    ######################### Filter Plot #########################

    # plot settings
    R, n = 0.4, 100
    colors = ['Reds', 'Oranges', 'Greens', 'Blues', 'Purples', 'Greys']
    grads = np.linspace(0.45, 0.55, 4)

    # evaluate filters
    X, Y, Z = efn.eval_filters(R, n=n, Phi_i=0)

    # plot filters
    for i,z in enumerate(Z):
        axes[1].contourf(X, Y, z/np.max(z), grads, cmap=colors[i%len(colors)])

    axes[1].set_xticks(np.linspace(-R, R, 5))
    axes[1].set_yticks(np.linspace(-R, R, 5))
    axes[1].set_xticklabels(['-R', '-R/2', '0', 'R/2', 'R'])
    axes[1].set_yticklabels(['-R', '-R/2', '0', 'R/2', 'R'])
    axes[1].set_xlabel('Translated Rapidity y')
    axes[1].set_ylabel('Translated Azimuthal Angle phi')
    axes[1].set_title('Energy Flow Network Latent Space', fontdict={'fontsize': 10})

    plt.show()
