from __future__ import absolute_import, division

import numpy as np
import pytest

from tensorflow.keras import backend as K
from tensorflow.keras.layers import PReLU
from tensorflow.keras.initializers import Constant

from energyflow import archs
from test_utils import epsilon_percent, epsilon_diff

@pytest.mark.arch
@pytest.mark.dnn
@pytest.mark.parametrize('k_inits', ['he_uniform', 'glorot_uniform', 'uniform', [Constant(), Constant(), Constant()]])
@pytest.mark.parametrize('acts', ['relu', 'LeakyReLU', 'sigmoid', [PReLU(), PReLU(), PReLU()]])
@pytest.mark.parametrize('sizes', [pytest.param([], marks=pytest.mark.xfail), [10], [10, 10]])
def test_DNN1(sizes, acts, k_inits):
    n, input_dim = 50, 10
    X_train = np.random.rand(n, input_dim)
    Y_train = np.random.rand(n, 2)
    dnn = archs.DNN(input_dim=input_dim, dense_sizes=sizes, acts=acts, k_inits=k_inits, summary=False, output_dim=2)
    dnn.fit(X_train, Y_train, epochs=1, batch_size=10)

@pytest.mark.arch
@pytest.mark.dnn
@pytest.mark.parametrize('output_dim', [2, 4])
@pytest.mark.parametrize('l2_regs', [0, 0.1, [0.2, 0.3, 0.5]])
@pytest.mark.parametrize('dropouts', [0, 0.1, [0.2, 0.3, 0.5]])
@pytest.mark.parametrize('sizes', [[10], [10, 10]])
def test_DNN2(sizes, dropouts, l2_regs, output_dim):
    n, input_dim = 50, 10
    X_train = np.random.rand(n, input_dim)
    Y_train = np.random.rand(n, output_dim)
    dnn = archs.DNN(input_dim=input_dim, dense_sizes=sizes, dropouts=dropouts, 
              l2_regs=l2_regs, summary=False, output_dim=output_dim)
    dnn.fit(X_train, Y_train, epochs=1, batch_size=10)

@pytest.mark.arch
@pytest.mark.cnn
@pytest.mark.parametrize('pool_sizes', [0, (2, 0)])
@pytest.mark.parametrize('dense_sizes', [None, [10]])
@pytest.mark.parametrize('num_filters', [(2, 2), (3, 1)])
@pytest.mark.parametrize('filter_sizes', [(2, 2), (3, 1)])
@pytest.mark.parametrize('npix', [14])
@pytest.mark.parametrize('nb_chan', [2])
@pytest.mark.parametrize('data_format', [pytest.param('channels_first', marks=pytest.mark.xfail), 'channels_last'])
def test_CNN_required(data_format, nb_chan, npix, filter_sizes, num_filters, dense_sizes, pool_sizes):
    if data_format == 'channels_first':
        input_shape = (nb_chan, npix, npix)
    else:
        input_shape = (npix, npix, nb_chan)

    X_train = np.random.rand(50, *input_shape)
    Y_train = np.random.rand(50, 2)
    cnn = archs.CNN(input_shape=input_shape, filter_sizes=filter_sizes, num_filters=num_filters, 
                    dense_sizes=dense_sizes, pool_sizes=pool_sizes, summary=False, data_format=data_format)
    cnn.fit(X_train, Y_train, epochs=1, batch_size=10)

@pytest.mark.arch
@pytest.mark.efn
@pytest.mark.parametrize('F_sizes', [[], [5], [5, 5]])
@pytest.mark.parametrize('Phi_sizes', [[], [5], [5, 5]])
@pytest.mark.parametrize('input_dim', [1, 2])
def test_EFN_required(input_dim, Phi_sizes, F_sizes):
    n, m = 50, 10
    X_train = [np.random.rand(n, m), np.random.rand(n, m, input_dim)]
    Y_train = np.random.rand(n, 2)
    efn = archs.EFN(input_dim=input_dim, Phi_sizes=Phi_sizes, F_sizes=F_sizes, summary=False)
    efn.fit(X_train, Y_train, epochs=1, batch_size=10)
    efn.inputs, efn.latent, efn.weights, efn.outputs, efn.Phi, efn.F

@pytest.mark.arch
@pytest.mark.pfn
@pytest.mark.parametrize('F_sizes', [[], [5], [5, 5]])
@pytest.mark.parametrize('Phi_sizes', [[], [5], [5, 5]])
@pytest.mark.parametrize('input_dim', [1, 2])
def test_PFN_required(input_dim, Phi_sizes, F_sizes):
    n, m = 50, 10
    X_train = np.random.rand(n, m, input_dim)
    Y_train = np.random.rand(n, 2)
    pfn = archs.PFN(input_dim=input_dim, Phi_sizes=Phi_sizes, F_sizes=F_sizes, summary=False)
    pfn.fit(X_train, Y_train, epochs=1, batch_size=10)
    pfn.inputs, pfn.latent, pfn.weights, pfn.outputs, pfn.Phi, pfn.F

@pytest.mark.arch
@pytest.mark.archbase
@pytest.mark.parametrize('modelcheck_opts', [{}, {'save_best_only': False}])
@pytest.mark.parametrize('save_weights_only', [True, False])
@pytest.mark.parametrize('save_while_training', [True, False])
@pytest.mark.parametrize('model_path', ['', 'efn_test_model.h5'])
def test_EFN_modelcheck(model_path, save_while_training, save_weights_only, modelcheck_opts):
    n, m = 50, 10
    X_train = [np.random.rand(n, m), np.random.rand(n, m, 2)]
    Y_train = np.random.rand(n, 2)
    X_val = [np.random.rand(n//10, m), np.random.rand(n//10, m, 2)]
    Y_val = np.random.rand(n//10, 2)
    efn = archs.EFN(input_dim=2, Phi_sizes=[10], F_sizes=[10], summary=False, filepath=model_path, 
                    save_while_training=save_while_training, save_weights_only=save_weights_only,
                    modelcheck_opts=modelcheck_opts)
    hist = efn.fit(X_train, Y_train, epochs=1, batch_size=10, validation_data=[X_val, Y_val])

@pytest.mark.arch
@pytest.mark.masking
@pytest.mark.efn
@pytest.mark.parametrize('mask_val', [0.0, 10, np.pi])
def test_EFN_masking(mask_val):
    n,m1,m2 = (50, 10, 20)
    input_dim = 3

    X_train = [np.random.rand(n,m1), np.random.rand(n,m1,input_dim)]
    y_train = np.random.rand(n, 2)

    efn = archs.EFN(input_dim=input_dim, Phi_sizes=[10], F_sizes=[10], mask_val=mask_val)
    efn.fit(X_train, y_train, epochs=1)

    X_test = [np.random.rand(n,m2), np.random.rand(n,m2,input_dim)]
    X_test_mask = [np.concatenate((X_test[0], mask_val*np.ones((n,5))), axis=1),
                   np.concatenate((X_test[1], mask_val*np.ones((n,5,input_dim))), axis=1)]

    assert epsilon_diff(efn.predict(X_test), efn.predict(X_test_mask), 10**-15)

    kf = K.function(inputs=efn.inputs, outputs=efn.latent)
    pure_mask = kf([0*X_test[0] + mask_val, 0*X_test[1] + mask_val ])[0]
    assert epsilon_diff(pure_mask, 0, 10**-15)

@pytest.mark.arch
@pytest.mark.masking
@pytest.mark.pfn
@pytest.mark.parametrize('mask_val', [0.0, 10, np.pi])
def test_PFN_masking(mask_val):
    n,m1,m2 = (50, 10, 20)
    input_dim = 3

    X_train = np.random.rand(n,m1,input_dim)
    y_train = np.random.rand(n, 2)

    pfn = archs.PFN(input_dim=input_dim, Phi_sizes=[10], F_sizes=[10], mask_val=mask_val)
    pfn.fit(X_train, y_train, epochs=1)

    X_test = np.random.rand(n,m2,input_dim)
    X_test_mask = np.concatenate((X_test, mask_val*np.ones((n,5,input_dim))), axis=1)

    assert epsilon_diff(pfn.predict(X_test), pfn.predict(X_test_mask), 10**-15)

    kf = K.function(inputs=pfn.inputs, outputs=pfn.latent)
    pure_mask = kf([0*X_test + mask_val])[0]
    assert epsilon_diff(pure_mask, 0, 10**-15)

@pytest.mark.arch
@pytest.mark.efn
@pytest.mark.globalfeatures
@pytest.mark.parametrize('nglobal', [1, 12])
def test_EFN_global_features(nglobal):
    n, m = 50, 10
    X_train = [np.random.rand(n, m), np.random.rand(n, m, 2), np.random.rand(n, nglobal)]
    Y_train = np.random.rand(n, 2)
    X_val = [np.random.rand(n//10, m), np.random.rand(n//10, m, 2), np.random.rand(n//10, nglobal)]
    Y_val = np.random.rand(n//10, 2)
    efn = archs.EFN(input_dim=2, Phi_sizes=[10], F_sizes=[10], num_global_features=nglobal, summary=False)
    hist = efn.fit(X_train, Y_train, epochs=1, batch_size=5, validation_data=[X_val, Y_val])
    efn._global_feature_tensor

@pytest.mark.arch
@pytest.mark.pfn
@pytest.mark.globalfeatures
@pytest.mark.parametrize('nglobal', [1, 12])
def test_PFN_required(nglobal):
    n, m = 50, 10
    X_train = [np.random.rand(n, m, 3), np.random.rand(n, nglobal)]
    Y_train = np.random.rand(n, 2)
    X_val = [np.random.rand(n//10, m, 3), np.random.rand(n//10, nglobal)]
    Y_val = np.random.rand(n//10, 2)
    pfn = archs.PFN(input_dim=3, Phi_sizes=[10], F_sizes=[10], num_global_features=nglobal, summary=False)
    hist = pfn.fit(X_train, Y_train, epochs=1, batch_size=5, validation_data=[X_val, Y_val])
    pfn._global_feature_tensor

