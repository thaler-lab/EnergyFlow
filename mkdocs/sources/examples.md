# Examples

There are 11 examples provided for the EnergyFlow package. They currently focus on demonstrating the various architectures included as part of EnergyFlow (see [Architectures](../docs/archs)). For examples involving the computation of EFPs or EMDs, see the [Demos](../demos).

To install the examples to the default directory, `~/.energyflow/examples/`, simply run 

```python
python3 -c "import energyflow; energyflow.utils.get_examples()"
```

See the [`get_examples`](../docs/utils/#get_examples) function for more detailed information. Some examples require [Tensorflow](https://tensorflow.org), [matplotlib](https://matplotlib.org/), or [scikit-learn](https://scikit-learn.org/stable/) to be installed.

### efn_example.py

An example using Energy Flow Networks (EFNs), which were introduced in
[1810.05165](https://arxiv.org/abs/1810.05165), to classify quark and gluon
jets. The [`EFN`](../docs/archs/#efn) class is used to construct the network
architecture. The output of the example is a plot of the ROC curves obtained
by the EFN as well as the jet mass and constituent multiplicity observables.

```python
#  ______ ______ _   _
# |  ____|  ____| \ | |
# | |__  | |__  |  \| |
# |  __| |  __| | . ` |
# | |____| |    | |\  |
# |______|_|    |_| \_|
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
num_epoch = 1
batch_size = 250

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
X_padded = ef.utils.pad_events(X)
(z_train, z_val, z_test, 
 p_train, p_val, p_test,
 Y_train, Y_val, Y_test,
 g_train, g_val, g_test) = ef.utils.data_split(X_padded[:,:,0], X_padded[:,:,1:], Y, global_features, val=val, test=test)

print('Done train/val/test split')
print('Model summary:')

# build architecture
efn = ef.archs.EFN(input_dim=2, Phi_sizes=Phi_sizes, F_sizes=F_sizes,
                   num_global_features=(4 if use_global_features else None))

# train model
efn.fit([z_train, p_train] + ([g_train] if use_global_features else []), Y_train,
        epochs=num_epoch,
        batch_size=batch_size,
        validation_data=([z_val, p_val] + ([g_val] if use_global_features else []), Y_val),
        verbose=1)

# get predictions on test data
preds = efn.predict([z_test, p_test] + ([g_test] if use_global_features else []), batch_size=1000)

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
    X, Y, Z = efn.eval_filters(R, n=n)

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

```

### efn_regression_example.py

An example involving Energy Flow Networks (EFNs), which were introduced in
1810.05165, to regress the jet constituents to the jet mass. The EFN class is
used to construct the network architecture. The output of the example is a plot
of the predicted and actual mass distributions.

```python
#  ______ ______ _   _
# |  ____|  ____| \ | |
# | |__  | |__  |  \| |
# |  __| |  __| | . ` |
# | |____| |    | |\  |
# |______|_|    |_| \_|
#  _____  ______ _____ _____  ______  _____ _____ _____ ____  _   _ 
# |  __ \|  ____/ ____|  __ \|  ____|/ ____/ ____|_   _/ __ \| \ | |
# | |__) | |__ | |  __| |__) | |__  | (___| (___   | || |  | |  \| |
# |  _  /|  __|| | |_ |  _  /|  __|  \___ \\___ \  | || |  | | . ` |
# | | \ \| |___| |__| | | \ \| |____ ____) |___) |_| || |__| | |\  |
# |_|  \_\______\_____|_|  \_\______|_____/_____/|_____\____/|_| \_|
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

# fundamental python package imports
import numpy as np
import matplotlib.pyplot as plt

# energyflow imports
import energyflow as ef

################################### SETTINGS ##################################

# data controls, can go up to 2000000 total for full dataset
train, val, test = 75000, 10000, 15000

# network architecture parameters
Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)
output_act, output_dim = 'linear', 1
loss = 'mse'

# network training parameters
num_epoch = 5
batch_size = 250

###############################################################################

# load data
X, y = ef.qg_jets.load(train + val + test, ncol=3, pad=False)

print('Loaded quark and gluon jets')

# preprocess by centering jets and normalizing pts
event_mask = []
for x in X:
    mask = x[:,0] > 0
    event_mask.append(np.count_nonzero(mask) > 1)
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[:,0].sum()
X = X[np.asarray(event_mask)]

print('Finished preprocessing')

# compute the jet "mass" as an angularity with exponent 2
# it's easier for the network to predict the log of the observable, shifted and scaled
obs = np.log10(np.asarray([np.sum(x[:,0]*(x[:,1:3]**2).sum(1))/x[:,0].sum() for x in X]))
obs_mean, obs_std = np.mean(obs), np.std(obs)
obs -= obs_mean
obs /= obs_std

print('Finished computing observables')

# do train/val/test split
X_padded = ef.utils.pad_events(X)
(z_train, z_val, z_test, 
 p_train, p_val, p_test,
 y_train, y_val, y_test) = ef.utils.data_split(X_padded[:,:,0], X_padded[:,:,1:], obs, val=val, test=test)

print('Done train/val/test split')

# build architecture
efn = ef.archs.EFN(input_dim=2, Phi_sizes=Phi_sizes, F_sizes=F_sizes, 
                   output_act=output_act, output_dim=output_dim, loss=loss, metrics=[])

# train model
efn.fit([z_train, p_train], y_train,
        epochs=num_epoch,
        batch_size=batch_size,
        validation_data=([z_val, p_val], y_val),
        verbose=1)

# get predictions on test data
preds = efn.predict([z_test, p_test], batch_size=1000)[:,0]*obs_std + obs_mean

######################### Observable Distributions Plot #########################

# some nicer plot settings 
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = (4, 4)

# plot the ROC curves
bins = np.linspace(-4, 0, 51)
obs_test = y_test*obs_std + obs_mean
plt.hist(obs_test, bins=bins, density=True, histtype='step', color='black', label='Actual')
plt.hist(preds, bins=bins, density=True, histtype='step', color='red', label='EFN Pred.')

# axes labels
plt.xlabel('log10(lambda2/pT)')
plt.ylabel('Differential Cross Section')

# make legend and show plot
plt.legend(loc='upper right', frameon=False)

plt.show()

######################### Percent Error Plot #########################

plt.hist(2*(preds - obs_test)/(obs_test + preds)*100,
         bins=np.linspace(-2.5, 2.5, 51),
         histtype='step', density=True)
plt.xlabel('Percent Error')
plt.ylabel('Probability Density')
plt.show()

######################### EFN Latent Space #########################

# plot settings
R, n = 0.4, 100
colors = ['Reds', 'Oranges', 'Greens', 'Blues', 'Purples', 'Greys']
grads = np.linspace(0.45, 0.55, 4)

# evaluate filters
X, Y, Z = efn.eval_filters(R, n=n)

# for sorting filters according to position
def get_filter_size_and_position(filt, zfrac=0.5):   
    filt /= np.max(filt)
    maxind = np.argmax(filt)
    j, k = maxind//n, maxind%n
    
    angle = np.sqrt((j-n/2)**2 + (k-n/2)**2)*2/n
    size = np.count_nonzero(filt > zfrac)/n**2
    
    return size, angle

sizes, angles = [], []
for z in Z:
    size, angle = get_filter_size_and_position(z)
    sizes.append(size)
    angles.append(angle)
qg_sizes, qg_angles = np.asarray(sizes), np.asarray(angles)

# plot filters
for i,z in enumerate(Z[np.argsort(qg_angles)[::-1]]):
    plt.contourf(X, Y, z/np.max(z), grads, cmap=colors[i%len(colors)])

plt.xticks(np.linspace(-R, R, 5), ['-R', '-R/2', '0', 'R/2', 'R'])
plt.yticks(np.linspace(-R, R, 5), ['-R', '-R/2', '0', 'R/2', 'R'])
plt.xlabel('Translated Rapidity y')
plt.ylabel('Translated Azimuthal Angle phi')
plt.title('Energy Flow Network Latent Space', fontdict={'fontsize': 10})

plt.show()
```

### efn_point_cloud_dataset_example.py

An example using Energy Flow Networks (EFNs), which were introduced in
[1810.05165](https://arxiv.org/abs/1810.05165), to classify quark and gluon
jets. The [`EFN`](../docs/archs/#efn) class is used to construct the network
architecture. This example is meant to highlight the usafe of PFNs with
Tensorflow datasets, in particular the function `tf_point_cloud_dataset` which
helpfully formats things in the proper way.  The output of the example is a
plot of the ROC curves obtained by the EFN as well as the jet mass and
constituent multiplicity observables.

```python
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
use_global_features = True

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
                   num_global_features=(global_features.shape[1] if use_global_features else None))

# get datasets
if use_global_features:
    d_train = ef.archs.PointCloudDataset([[ef.archs.WeightedPointCloudDataset(X_train), g_train], Y_train],
                                         batch_size=batch_size)
    d_val = ef.archs.PointCloudDataset([[ef.archs.WeightedPointCloudDataset(X_val), g_val], Y_val])
    d_test = ef.archs.PointCloudDataset([ef.archs.WeightedPointCloudDataset(X_test), g_test])#.wrap()
else:
    d_train = ef.archs.PointCloudDataset([ef.archs.WeightedPointCloudDataset(X_train), Y_train],
                                         batch_size=batch_size)
    d_val = ef.archs.PointCloudDataset([ef.archs.WeightedPointCloudDataset(X_val), Y_val])
    d_test = ef.archs.WeightedPointCloudDataset(X_test)#.wrap()

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

```

### efn_multiple_phis.py

An example using Energy Flow Networks (EFNs), which were introduced in
[1810.05165](https://arxiv.org/abs/1810.05165), to classify quark and gluon
jets. The [`EFN`](../docs/archs/#efn) class is used to construct the network
architecture. This example is meant to highlight the usafe of PFNs with
Tensorflow datasets, in particular the function `tf_point_cloud_dataset` which
helpfully formats things in the proper way.  The output of the example is a
plot of the ROC curves obtained by the EFN as well as the jet mass and
constituent multiplicity observables.

```python
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
Phi_sizes, F_sizes = [(100, 100, 128), (10, 20)], (100, 100, 100)
# Phi_sizes, F_sizes = (100, 100, 256), (100, 100, 100)

# network training parameters
num_epoch = 1
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
efn = ef.archs.EFN(input_dim=(2, 1), Phi_sizes=Phi_sizes, F_sizes=F_sizes,
                   num_global_features=(4 if use_global_features else None))

# get datasets
X_train = [ef.archs.WeightedPointCloudDataset(X_train), ef.archs.PairedWeightedPointCloudDataset(X_train, pairing='distance')]
X_val = [ef.archs.WeightedPointCloudDataset(X_val), ef.archs.PairedWeightedPointCloudDataset(X_val, pairing='distance')]
X_test = [ef.archs.WeightedPointCloudDataset(X_test), ef.archs.PairedWeightedPointCloudDataset(X_test, pairing='distance')]
if use_global_features:
    X_train += [g_train]
    X_val += [g_val]
    X_test += [g_test]

d_train = ef.archs.PointCloudDataset([X_train, Y_train], batch_size=batch_size)
d_val = ef.archs.PointCloudDataset([X_val, Y_val])
d_test = ef.archs.PointCloudDataset(X_test)

print('training dataset', d_train)
print('validation dataset', d_val)
print('testing dataset', d_test)

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

```

### pfn_example.py

An example involving Particle Flow Networks (PFNs), which were introduced in
[1810.05165](https://arxiv.org/abs/1810.05165). The [`PFN`](../docs/archs/#pfn)
class is used to construct the network architecture. The output of the example
is a plot of the ROC curves obtained by the PFN as well as the jet mass and
constituent multiplicity observables.

```python
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

```

### pfn_point_cloud_dataset_example.py

An example involving Particle Flow Networks (PFNs), which were introduced in
[1810.05165](https://arxiv.org/abs/1810.05165). The [`PFN`](../docs/archs/#pfn)
class is used to construct the network architecture. This example is meant to
highlight the usafe of PFNs with Tensorflow datasets, in particular the function
`tf_point_cloud_dataset` which helpfully formats things in the proper way. Like
the bse PFN example, the output is a plot of the ROC curves obtained by the PFN
as well as the jet mass and constituent multiplicity observables.

```python
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
use_pids = False
use_global_features = False

# network architecture parameters
Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)
# Phi_sizes, F_sizes = (100, 100, 256), (100, 100, 100)

# network training parameters
num_epoch = 2
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
pfn = ef.archs.PFN(input_dim=ncol, Phi_sizes=Phi_sizes, F_sizes=F_sizes,
                   num_global_features=(global_features.shape[1] if use_global_features else None))

# get datasets
if use_global_features:
    d_train = ef.archs.PointCloudDataset([[X_train, g_train], Y_train], batch_size=batch_size)
    d_val = ef.archs.PointCloudDataset([[X_val, g_val], Y_val])
    d_test = ef.archs.PointCloudDataset([X_test, g_test])
else:
    d_train = ef.archs.PointCloudDataset([X_train, Y_train], batch_size=batch_size)
    d_val = ef.archs.PointCloudDataset([X_val, Y_val])
    d_test = ef.archs.PointCloudDataset([X_test])
    
print('training dataset', d_train)
print('validation dataset', d_val)
print('testing dataset', d_test)

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

```

### pfn_multiple_phis.py

An example involving Particle Flow Networks (PFNs), which were introduced in
[1810.05165](https://arxiv.org/abs/1810.05165). The [`PFN`](../docs/archs/#pfn)
class is used to construct the network architecture. This example is meant to
highlight the usafe of PFNs with Tensorflow datasets, in particular the function
`tf_point_cloud_dataset` which helpfully formats things in the proper way. Like
the bse PFN example, the output is a plot of the ROC curves obtained by the PFN
as well as the jet mass and constituent multiplicity observables.

```python
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
X_train = [X_train, ef.archs.PairedPointCloudDataset(X_train)] + ([g_train] if use_global_features else [])
X_val = [X_val, ef.archs.PairedPointCloudDataset(X_val)] + ([g_val] if use_global_features else [])
X_test = [X_test, ef.archs.PairedPointCloudDataset(X_test)] + ([g_test] if use_global_features else [])

# construct point cloud datasets
d_train = ef.archs.PointCloudDataset([X_train, Y_train], batch_size=batch_size)
d_val = ef.archs.PointCloudDataset([X_val, Y_val])
d_test = ef.archs.PointCloudDataset(X_test)

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

```

### cnn_example.py

An example involving jet images and convolutional neural networks (CNNs). The
[`CNN`](../docs/archs/#cnn) class is used to provide a network architecture
based on that described in [1612.01551](https://arxiv.org/abs/1612.01551). 

Jet images are constructed using the [`pixelate`](../docs/utils/#pixelate)
function and can be either one-channel (grayscale), meaning that only $p_T$
information is used, or two-channel (color), meaning that $p_T$ information and
local charged particle counts are used. The images are preprocessed by
subtracting the average image in the training set and dividing by the per-pixel
standard deviations, using the [`zero_center`](../docs/utils/#zero_center) and 
[`standardize`](../docs/utils/#standardize) functions, respectively. The output
of the example is a plot of the ROC curves of the CNN as well as the jet mass
and constituent multiplicity observables.

Note that the number of epochs is quite small because it is quite time consuming
to train a CNN without a GPU (which will speed up this example immensely).

```python
#   _____ _   _ _   _
#  / ____| \ | | \ | |
# | |    |  \| |  \| |
# | |    | . ` | . ` |
# | |____| |\  | |\  |
#  \_____|_| \_|_| \_|
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

# data controls
num_data = 100000
val_frac, test_frac = 0.1, 0.15

# image parameters
R = 0.4
img_width = 2*R
npix = 33
nb_chan = 2
norm = True

# required network architecture parameters
input_shape = (npix, npix, nb_chan)
filter_sizes = [8, 4, 4]
num_filters = [8, 8, 8] # very small so can run on non-GPUs in reasonable time

# optional network architecture parameters
dense_sizes = [50]
pool_sizes = 2

# network training parameters
num_epoch = 1
batch_size = 500

################################################################################

# load data
X, y = ef.qg_jets.load(num_data=num_data, pad=False)

# convert labels to categorical
Y = ef.utils.to_categorical(y, num_classes=2)

print('Loaded quark and gluon jets')

# make jet images
images = np.asarray([ef.utils.pixelate(x, npix=npix, img_width=img_width, nb_chan=nb_chan, 
                                          charged_counts_only=True, norm=norm) for x in X])

print('Done making jet images')

# do train/val/test split 
(X_train, X_val, X_test,
 Y_train, Y_val, Y_test) = ef.utils.data_split(images, Y, val=val_frac, test=test_frac)

print('Done train/val/test split')

# preprocess by zero centering images and standardizing each pixel
X_train, X_val, X_test = ef.utils.standardize(*ef.utils.zero_center(X_train, X_val, X_test))

print('Finished preprocessing')
print('Model summary:')

# build architecture
hps = {'input_shape': input_shape,
       'filter_sizes': filter_sizes,
       'num_filters': num_filters,
       'dense_sizes': dense_sizes,
       'pool_sizes': pool_sizes}
cnn = ef.archs.CNN(hps)

# train model
cnn.fit(X_train, Y_train,
        epochs=num_epoch,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        verbose=1)

# get predictions on test data
preds = cnn.predict(X_test, batch_size=1000)

# get ROC curve
cnn_fp, cnn_tp, threshs = sklearn.metrics.roc_curve(Y_test[:,1], preds[:,1])

# get area under the ROC curve
auc = sklearn.metrics.roc_auc_score(Y_test[:,1], preds[:,1])
print()
print('CNN AUC:', auc)
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
    plt.plot(cnn_tp, 1-cnn_fp, '-', color='black', label='CNN')
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

```

### dnn_example.py

An example involving deep, fully-connected neural networks (DNNs). The
[`DNN`](../docs/archs/#dnn) class is used to construct the network architecture.

The inputs are taken to be the $N$-subjettiness observables as specified as part
of the phase space basis from [1704.08249](https://arxiv.org/abs/1704.08249),
cut off at some total number of observables. The output of the example is a plot
showing the ROC curves obtained from training the DNN on different numbers of
$N$-subjettiness observables.

```python
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
X, y = ef.qg_nsubs.load(num_data=num_data)

# convert labels to categorical
Y = ef.utils.to_categorical(y, num_classes=2)

print('Loaded quark and gluon jets')
print('Model summary:')

# train models with different numbers of nsubs as input
rocs = []
for i,num_nsub in enumerate(num_nsubs):

    # build architecture
    dnn = ef.archs.DNN(input_dim=num_nsub, dense_sizes=dense_sizes, summary=(i==0))

    # do train/val/test split 
    (X_train, X_val, X_test,
     Y_train, Y_val, Y_test) = ef.utils.data_split(X[:,:num_nsub], Y, val=val_frac, test=test_frac)

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
    rocs.append(sklearn.metrics.roc_curve(Y_test[:,1], preds[:,1]))

    # get area under the ROC curve
    auc = sklearn.metrics.roc_auc_score(Y_test[:,1], preds[:,1])
    print()
    print('{} nsubs DNN AUC:'.format(num_nsub), auc)
    print()

if len(sys.argv) == 1 or bool(sys.argv[1]):

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

```

### efp_example.py

An example involving Energy Flow Polynomials (EFPs) and a linear classifier
(Fisher's Linear Discriminant by default). First, the [`EFPSet`](../docs/
efp/#efpset) class is used to compute the EFPs up to the specified `dmax`, the
default being `dmax=5`. Then linear classifiers are trained for different
numbers of EFPs as input, determined by taking all EFPs up to degree `d` with
`d` from `1` to `dmax`. The output of the example is a plot of the ROC curves
for the classifiers with different numbers of EFP inputs.

```python
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

```

### animation_example.py

An example that makes an animation between two events using the EMD. Note
that `ffmpeg` must be installed in order for matplotlib to be able to render
the animation. Strange errors may result if there are issues with required
software components.

```python
#           _   _ _____ __  __       _______ _____ ____  _   _ 
#     /\   | \ | |_   _|  \/  |   /\|__   __|_   _/ __ \| \ | |
#    /  \  |  \| | | | | \  / |  /  \  | |    | || |  | |  \| |
#   / /\ \ | . ` | | | | |\/| | / /\ \ | |    | || |  | | . ` |
#  / ____ \| |\  |_| |_| |  | |/ ____ \| |   _| || |__| | |\  |
# /_/    \_\_| \_|_____|_|  |_/_/    \_\_|  |_____\____/|_| \_|
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

# standard numerical library imports
import numpy as np

# matplotlib is required for this example
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (4,4)

#############################################################
# NOTE: ffmpeg must be installed
# on macOS this can be done with `brew install ffmpeg`
# on Ubuntu this would be `sudo apt-get install ffmpeg`
#############################################################

# on windows, the following might need to be uncommented
#plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'

import energyflow as ef
from matplotlib import animation, rc

# helper function to interpolate between the optimal transport of two events
def merge(ev0, ev1, R=1, lamb=0.5):    
    emd, G = ef.emd.emd(ev0, ev1, R=R, return_flow=True)
    
    merged = []
    for i in range(len(ev0)):
        for j in range(len(ev1)):
            if G[i, j] > 0:
                merged.append([G[i,j], lamb*ev0[i,1] + (1-lamb)*ev1[j,1], 
                                       lamb*ev0[i,2] + (1-lamb)*ev1[j,2]])

    # detect which event has more pT
    if np.sum(ev0[:,0]) > np.sum(ev1[:,0]):
        for i in range(len(ev0)):
            if G[i,-1] > 0:
                merged.append([G[i,-1]*lamb, ev0[i,1], ev0[i,2]])
    else:
        for j in range(len(ev1)):
            if G[-1,j] > 0:
                merged.append([G[-1,j]*(1-lamb), ev1[j,1], ev1[j,2]])            
            
    return np.asarray(merged)


#############################################################
# ANIMATION OPTIONS
#############################################################
zf = 2           # size of points in scatter plot
lw = 1           # linewidth of flow lines
fps = 40         # frames per second, increase this for sharper resolution
nframes = 10*fps # total number of frames
R = 0.5          # jet radius


#############################################################
# LOAD IN JETS
#############################################################
specs = ['375 <= corr_jet_pts <= 425', 'abs_jet_eta < 1.9', 'quality >= 2']
events = ef.mod.load(*specs, dataset='cms', amount=0.01)

# events go here as lists of particle [pT,y,phi]
event0 = events.particles[14930][:,:3]
event1 = events.particles[19751][:,:3]

# center the jets
event0[:,1:3] -= np.average(event0[:,1:3], weights=event0[:,0], axis=0)
event1[:,1:3] -= np.average(event1[:,1:3], weights=event1[:,0], axis=0)

# mask out particles outside of the cone
event0 = event0[np.linalg.norm(event0[:,1:3], axis=1) < R]
event1 = event1[np.linalg.norm(event1[:,1:3], axis=1) < R]

ev0 = np.copy(event0)
ev1 = np.copy(event1)

#############################################################
# MAKE ANIMATION
#############################################################

fig, ax = plt.subplots()

merged = merge(ev0, ev1, lamb=0, R=R)
pts, ys, phis = merged[:,0], merged[:,1], merged[:,2]

scatter = ax.scatter(ys, phis, color='blue', s=pts, lw=0)

# animation function. This is called sequentially
def animate(i):
    ax.clear()
    
    nstages = 4
    
    # first phase is a static image of event0
    if i < nframes / nstages:
        lamb = nstages*i/(nframes-1)
        ev0  = event0
        ev1  = event0
        color = (1,0,0)
    
    # second phase is a transition from event0 to event1
    elif i < 2 * nframes / nstages:
        lamb = nstages*(i - nframes/nstages)/(nframes-1)
        ev0  = event1
        ev1  = event0
        color = (1-lamb)*np.asarray([1,0,0]) + (lamb)*np.asarray([0,0,1])
    
    # third phase is a static image of event1
    elif i < 3 * nframes / nstages:
        lamb = nstages*(i - 2*nframes/nstages)/(nframes-1)
        ev0  = event1
        ev1  = event1
        color = (0,0,1)
    
    # fourth phase is a transition from event1 to event0
    else:
        lamb = nstages*(i - 3*nframes/nstages)/(nframes-1)
        ev0  = event0
        ev1  = event1
        color = (lamb)*np.asarray([1,0,0]) + (1-lamb)*np.asarray([0,0,1])

    merged = merge(ev0, ev1, lamb=lamb, R=0.5)
    pts, ys, phis = merged[:,0], merged[:,1], merged[:,2]
    scatter = ax.scatter(ys, phis, color=color, s=zf*pts, lw=0)
    
    ax.set_xlim(-R, R); ax.set_ylim(-R, R);
    ax.set_axis_off()
    
    return scatter,

anim = animation.FuncAnimation(fig, animate, frames=nframes, repeat=True)
anim.save('energyflowanimation.mp4', fps=fps, dpi=200)

# uncomment these lines if running in a jupyter notebook
# from IPython.display import HTML
# HTML(anim.to_html5_video())

```

