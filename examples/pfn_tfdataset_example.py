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

# standard numerical library imports
import numpy as np

# energyflow imports
import energyflow as ef
from energyflow.archs import PFN, tf_point_cloud_dataset
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

################################### SETTINGS ###################################
# the commented values correspond to those in 1810.05165
###############################################################################

# data controls, can go up to 2000000 for full dataset
train, val, test = 75000, 10000, 15000
# train, val, test = 1000000, 200000, 200000
use_pids = True

# network architecture parameters
Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)
# Phi_sizes, F_sizes = (100, 100, 256), (100, 100, 100)

# network training parameters
num_epoch = 1
batch_size = 500

################################################################################

# load data
ncol = 4 if use_pids else 3
X, y = qg_jets.load(train + val + test, pad=False, ncol=ncol)

# convert labels to categorical
Y = to_categorical(y, num_classes=2)

print('Loaded quark and gluon jets')

# preprocess by centering jets and normalizing pts
for x in X:
    yphi_avg = np.average(x[:,1:3], weights=x[:,0], axis=0)
    x[:,1:3] -= yphi_avg
    x[:,0] /= x[:,0].sum()

# handle particle id channel
if use_pids:
    remap_pids(X, pid_i=3)

print('Finished preprocessing')

# do train/val/test split 
(X_train, X_val, X_test,
 Y_train, Y_val, Y_test) = data_split(X, Y, val=val, test=test)

print('Done train/val/test split')
print('Model summary:')

# build architecture
pfn = PFN(input_dim=ncol, Phi_sizes=Phi_sizes, F_sizes=F_sizes)

# get tf datasets
d_train = tf_point_cloud_dataset([[X_train], Y_train], batch_size)
d_val = tf_point_cloud_dataset([X_val, Y_val], batch_size)
d_test = tf_point_cloud_dataset(X_test, batch_size)
print('training dataset', d_train)
print('validation dataset', d_val)
print('testing dataset', d_test)

# train model
pfn.fit(d_train, epochs=num_epoch, validation_data=d_val)

# get predictions on test data
preds = pfn.predict(d_test)

# get ROC curve
pfn_fp, pfn_tp, threshs = roc_curve(Y_test[:,1], preds[:,1])

# get area under the ROC curve
auc = roc_auc_score(Y_test[:,1], preds[:,1])
print()
print('PFN AUC:', auc)
print()

# get multiplicity and mass for comparison
masses = np.asarray([ef.ms_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in X])
mults = np.asarray([np.count_nonzero(x[:,0]) for x in X])
mass_fp, mass_tp, threshs = roc_curve(Y[:,1], -masses)
mult_fp, mult_tp, threshs = roc_curve(Y[:,1], -mults)

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
