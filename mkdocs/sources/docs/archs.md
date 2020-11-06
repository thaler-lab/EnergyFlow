# Architectures

Energy Flow Networks (EFNs) and Particle Flow Networks (PFNs) are model
architectures designed for learning from collider events as unordered,
variable-length sets of particles. Both EFNs and PFNs are parameterized by a
learnable per-particle function $\Phi$ and latent space function $F$.

An EFN takes the following form:
$$\text{EFN}=F\left(\sum_{i=1}^M z_i \Phi(\hat p_i)\right)$$
where $z_i$ is a measure of the energy of particle $i$, such as $z_i=p_{T,i}$,
and $\hat p_i$ is a measure of the angular information of particle $i$, such as
$\hat p_i = (y_i,\phi_i)$. Any infrared- and collinear-safe observable can be
parameterized in this form.

A PFN takes the following form:
$$\text{PFN}=F\left(\sum_{i=1}^M \Phi(p_i)\right)$$
where $p_i$ is the information of particle $i$, such as its four-momentum,
charge, or flavor. Any observable can be parameterized in this form. See the
[Deep Sets](https://arxiv.org/abs/1703.06114) framework for additional 
discussion.

Since these architectures are not used by the core EnergyFlow code, and require
the external [TensorFlow](https://www.tensorflow.org) and [scikit-learn](http:
//scikit-learn.org/) libraries, they are not imported by default but must be
explicitly imported, e.g. `from energyflow.archs import *`. EnergyFlow also
contains several additional model architectures for ease of using common models
that frequently appear in the intersection of particle physics and machine
learning.

----

## ArchBase

Base class for all architectures contained in EnergyFlow. The mechanism of
specifying hyperparameters for all architectures is described here. Methods
common to all architectures are documented here. Note that this class cannot
be instantiated directly as it is an abstract base class.

```python
energyflow.archs.archbase.ArchBase(*args, **kwargs)
```

Accepts arbitrary arguments. Positional arguments (if present) are
dictionaries of hyperparameters, keyword arguments (if present) are 
hyperparameters directly. Keyword hyperparameters take precedence over
positional hyperparameter dictionaries.

**Arguments**

- ***args** : arbitrary positional arguments
    - Each argument is a dictionary containing hyperparameter (name, value)
    pairs.
- ***kwargs** : arbitrary keyword arguments
    - Hyperparameters as keyword arguments. Takes precedence over the 
    positional arguments.

**Default NN Hyperparameters**

Common hyperparameters that apply to all architectures except for
[`LinearClassifier`](#linearclassifier).

**Compilation Options**

- **loss**=`'categorical_crossentropy'` : _str_
    - The loss function to use for the model. See the [Keras loss 
    function docs](https://keras.io/losses/) for available loss
    functions.
- **optimizer**=`'adam'` : Keras optimizer or _str_
    - A [Keras optimizer](https://keras.io/optimizers/) instance or a
    string referring to one (in which case the default arguments are 
    used).
- **metrics**=`['accuracy']` : _list_ of _str_
    - The [Keras metrics](https://keras.io/metrics/) to apply to the
    model.
- **compile_opts**=`{}` : _dict_
    - Dictionary of keyword arguments to be passed on to the
    [`compile`](https://keras.io/models/model/#compile) method of the
    model. `loss`, `optimizer`, and `metrics` (see above) are included
    in this dictionary. All other values are the Keras defaults.

**Output Options**

- **output_dim**=`2` : _int_
    - The output dimension of the model.
- **output_act**=`'softmax'` : _str_ or Keras activation
    - Activation function to apply to the output.

**Callback Options**

- **filepath**=`None` : _str_
    - The file path for where to save the model. If `None` then the
    model will not be saved.
- **save_while_training**=`True` : _bool_
    - Whether the model is saved during training (using the 
    [`ModelCheckpoint`](https://keras.io/callbacks/#modelcheckpoint)
    callback) or only once training terminates. Only relevant if
    `filepath` is set.
- **save_weights_only**=`False` : _bool_
    - Whether only the weights of the model or the full model are
    saved. Only relevant if `filepath` is set.
- **modelcheck_opts**=`{'save_best_only':True, 'verbose':1}` : _dict_
    - Dictionary of keyword arguments to be passed on to the
    [`ModelCheckpoint`](https://keras.io/callbacks/#modelcheckpoint)
    callback, if it is present. `save_weights_only` (see above) is
    included in this dictionary. All other arguments are the Keras
    defaults.
- **patience**=`None` : _int_
    - The number of epochs with no improvement after which the training
    is stopped (using the [`EarlyStopping`](https://keras.io/
    callbacks/#earlystopping) callback). If `None` then no early stopping
    is used.
- **earlystop_opts**=`{'restore_best_weights':True, 'verbose':1}` : _dict_
    - Dictionary of keyword arguments to be passed on to the
    [`EarlyStopping`](https://keras.io/callbacks/#earlystopping)
    callback, if it is present. `patience` (see above) is included in
    this dictionary. All other arguments are the Keras defaults.

**Flags**

- **name_layers**=`True` : _bool_
    - Whether to give the layers of the model explicit names or let
    them be named automatically. One reason to set this to `False`
    would be in order to use parts of this model in another model
    (all Keras layers in a model are required to have unique names).
- **compile**=`True` : _bool_
    - Whether the model should be compiled or not.
- **summary**=`True` : _bool_
    - Whether a summary should be printed or not.

### fit

```python
fit(*args, **kwargs)
```

Train the model by fitting the provided training dataset and labels.
Transparently calls the `.fit()` method of the underlying model.

**Arguments**

- ***args** : _numpy.ndarray_ or _tensorflow.data.Dataset_
    - Either the `X_train` and `Y_train` NumPy arrays or a TensorFlow
    dataset. See [`tf_point_cloud_dataset`](#tf_point_cloud_dataset) for
    contruction of TensorFlow datasets for EFN and PFN models.
- **kwargs** : _dict_
    - Keyword arguments passed on to the `.fit()` method of the
    underlying model. Most relevant for neural network models, where the
    [TensorFlow/Keras model docs](https://www.tensorflow.org/api_docs/
    python/tf/keras/Model#fit) contain detailed information on the
    possible arguments.

**Returns**

- The return value of the the underlying model's `.fit()` method.

### predict

```python
predict(X_test, **kwargs)
```

Evaluate the model on a dataset. Note that for the `LinearClassifier`
this corresponds to the `predict_proba` method of the underlying 
scikit-learn model.

**Arguments**

- **X_test** : _numpy.ndarray_
    - The dataset to evaluate the model on.
- **kwargs** : _dict_
    - Keyword arguments passed on to the underlying model when
    predicting on a dataset.

**Returns**

- _numpy.ndarray_
    - The value of the model on the input dataset.

### properties

#### model

```python
model
```

The underlying model held by this architecture. Note that accessing
an attribute that the architecture does not have will resulting in
attempting to retrieve the attribute from this model. This allows for
interrogation of the EnergyFlow architecture in the same manner as the
underlying model.

**Examples**

- For neural network models:
    - `model.layers` will return a list of the layers, where 
    `model` is any EnergFlow neural network.
- For linear models:
    - `model.coef_` will return the coefficients, where `model`
    is any EnergyFlow `LinearClassifier` instance.


----


## EFN

Energy Flow Network (EFN) architecture.

```python
energyflow.archs.EFN(*args, **kwargs)
```

See [`ArchBase`](#archbase) for how to pass in hyperparameters as
well as defaults common to all EnergyFlow neural network models.

**Required EFN Hyperparameters**

- **input_dim** : _int_
    - The number of features for each particle.
- **Phi_sizes** (formerly `ppm_sizes`) : {_tuple_, _list_} of _int_
    - The sizes of the dense layers in the per-particle frontend
    module $\Phi$. The last element will be the number of latent 
    observables that the model defines.
- **F_sizes** (formerly `dense_sizes`) : {_tuple_, _list_} of _int_
    - The sizes of the dense layers in the backend module $F$.

**Default EFN Hyperparameters**

- **Phi_acts**=`'relu'` (formerly `ppm_acts`) : {_tuple_, _list_} of
_str_ or Keras activation
    - Activation functions(s) for the dense layers in the 
    per-particle frontend module $\Phi$. A single string or activation
    layer will apply the same activation to all layers. Keras advanced
    activation layers are also accepted, either as strings (which use
    the default arguments) or as Keras `Layer` instances. If passing a
    single `Layer` instance, be aware that this layer will be used for
    all activations and may introduce weight sharing (such as with 
    `PReLU`); it is recommended in this case to pass as many activations
    as there are layers in the model. See the [Keras activations 
    docs](https://keras.io/activations/) for more detail.
- **F_acts**=`'relu'` (formerly `dense_acts`) : {_tuple_, _list_} of
_str_ or Keras activation
    - Activation functions(s) for the dense layers in the 
    backend module $F$. A single string or activation layer will apply
    the same activation to all layers.
- **Phi_k_inits**=`'he_uniform'` (formerly `ppm_k_inits`) : {_tuple_,
_list_} of _str_ or Keras initializer
    - Kernel initializers for the dense layers in the per-particle
    frontend module $\Phi$. A single string will apply the same
    initializer to all layers. See the [Keras initializer docs](https:
    //keras.io/initializers/) for more detail.
- **F_k_inits**=`'he_uniform'` (formerly `dense_k_inits`) : {_tuple_,
_list_} of _str_ or Keras initializer
    - Kernel initializers for the dense layers in the backend 
    module $F$. A single string will apply the same initializer 
    to all layers.
- **latent_dropout**=`0` : _float_
    - Dropout rates for the summation layer that defines the
    value of the latent observables on the inputs. See the [Keras
    Dropout layer](https://keras.io/layers/core/#dropout) for more 
    detail.
- **F_dropouts**=`0` (formerly `dense_dropouts`) : {_tuple_, _list_}
of _float_
    - Dropout rates for the dense layers in the backend module $F$. 
    A single float will apply the same dropout rate to all dense layers.
- **Phi_l2_regs**=`0` : {_tuple_, _list_} of _float_
    - $L_2$-regulatization strength for both the weights and biases
    of the layers in the $\Phi$ network. A single float will apply the
    same $L_2$-regulatization to all layers.
- **F_l2_regs**=`0` : {_tuple_, _list_} of _float_
    - $L_2$-regulatization strength for both the weights and biases
    of the layers in the $F$ network. A single float will apply the
    same $L_2$-regulatization to all layers.
- **mask_val**=`0` : _float_
    - The value for which particles with all features set equal to
    this value will be ignored. The [Keras Masking layer](https://
    keras.io/layers/core/#masking) appears to have issues masking
    the biases of a network, so this has been implemented in a
    custom (and correct) manner since version `0.12.0`.
- **num_global_features**=`None` : _int_
    - Number of additional features to be concatenated with the latent
    space observables to form the input to F. If not `None`, then the
    features are to be provided at the end of the list of inputs.

### eval_filters

```python
eval_filters(patch, n=100, prune=True)
```

Evaluates the latent space filters of this model on a patch of the 
two-dimensional geometric input space.

**Arguments**

- **patch** : {_tuple_, _list_} of _float_
    - Specifies the patch of the geometric input space to be evaluated.
    A list of length 4 is interpretted as `[xmin, ymin, xmax, ymax]`.
    Passing a single float `R` is equivalent to `[-R,-R,R,R]`.
- **n** : {_tuple_, _list_} of _int_
    - The number of grid points on which to evaluate the filters. A list 
    of length 2 is interpretted as `[nx, ny]` where `nx` is the number of
    points along the x (or first) dimension and `ny` is the number of points
    along the y (or second) dimension.
- **prune** : _bool_
    - Whether to remove filters that are all zero (which happens sometimes
    due to dying ReLUs).

**Returns**

- (_numpy.ndarray_, _numpy.ndarray_, _numpy.ndarray_)
    - Returns three arrays, `(X, Y, Z)`, where `X` and `Y` have shape `(nx, ny)` 
    and are arrays of the values of the geometric inputs in the specified patch.
    `Z` has shape `(num_filters, nx, ny)` and is the value of the different
    filters at each point.

### properties

#### inputs

```python
inputs
```

List of input tensors to the model. EFNs have two input tensors:
`inputs[0]` corresponds to the `zs` input and `inputs[1]` corresponds
to the `phats` input.

#### weights

```python
weights
```

Weight tensor for the model. This is the `zs` input where entries
equal to `mask_val` have been set to zero.

#### Phi

```python
Phi
```

List of tensors corresponding to the layers in the $\Phi$ network.

#### latent

```python
latent
```

List of tensors corresponding to the summation layer in the
network, including any dropout layer if present.

#### F

```python
F
```

List of tensors corresponding to the layers in the $F$ network.

#### output

```python
output
```

Output tensor for the model.


----

## PFN

Particle Flow Network (PFN) architecture. Accepts the same 
hyperparameters as the [`EFN`](#EFN).

```python
energyflow.archs.PFN(*args, **kwargs)
```



### properties

#### inputs

```python
inputs
```

List of input tensors to the model. PFNs have one input tensor
corresponding to the `ps` input.

#### weights

```python
weights
```

Weight tensor for the model. A weight of `0` is assigned to any
particle which has all features equal to `mask_val`, and `1` is
assigned otherwise.

#### Phi

```python
Phi
```

List of tensors corresponding to the layers in the $\Phi$ network.

#### latent

```python
latent
```

List of tensors corresponding to the summation layer in the
network, including any dropout layer if present.

#### F

```python
F
```

List of tensors corresponding to the layers in the $F$ network.

#### output

```python
output
```

Output tensor for the model.


----

## CNN

Convolutional Neural Network architecture.

```python
energyflow.archs.CNN(*args, **kwargs)
```

See [`ArchBase`](#archbase) for how to pass in hyperparameters as
well as defaults common to all EnergyFlow neural network models.

**Required CNN Hyperparameters**

- **input_shape** : {_tuple_, _list_} of _int_
    - The shape of a single jet image. Assuming that `data_format`
    is set to `channels_first`, this is `(nb_chan,npix,npix)`.
- **filter_sizes** : {_tuple_, _list_} of _int_
    - The size of the filters, which are taken to be square, in each 
    convolutional layer of the network. The length of the list will be
    the number of convolutional layers in the network.
- **num_filters** : {_tuple_, _list_} of _int_
    - The number of filters in each convolutional layer. The length of 
    `num_filters` must match that of `filter_sizes`.

**Default CNN Hyperparameters**

- **dense_sizes**=`None` : {_tuple_, _list_} of _int_
    - The sizes of the dense layer backend. A value of `None` is 
    equivalent to an empty list.
- **pool_sizes**=`0` : {_tuple_, _list_} of _int_
    - Size of maxpooling filter, taken to be a square. A value of 
    `0` will not use maxpooling.
- **conv_acts**=`'relu'` : {_tuple_, _list_} of _str_  or Keras activation
    - Activation function(s) for the conv layers. A single string or
    activation layer will apply the same activation to all conv layers.
    Keras advanced activation layers are also accepted, either as
    strings (which use the default arguments) or as Keras `Layer` 
    instances. If passing a single `Layer` instance, be aware that this
    layer will be used for all activations and may introduce weight 
    sharing (such as with `PReLU`); it is recommended in this case to 
    pass as many activations as there are layers in the model.See the
    [Keras activations docs](https://keras.io/activations/) for more 
    detail.
- **dense_acts**=`'relu'` : {_tuple_, _list_} of _str_  or Keras activation
    - Activation functions(s) for the dense layers. A single string 
    or activation layer will apply the same activation to all dense 
    layers.
- **conv_k_inits**=`'he_uniform'` : {_tuple_, _list_} of _str_ or Keras initializer
    - Kernel initializers for the convolutional layers. A single
    string will apply the same initializer to all layers. See the
    [Keras initializer docs](https://keras.io/initializers/) for 
    more detail.
- **dense_k_inits**=`'he_uniform'` : {_tuple_, _list_} of _str_ or Keras initializer
    - Kernel initializers for the dense layers. A single string will 
    apply the same initializer to all layers.
- **conv_dropouts**=`0` : {_tuple_, _list_} of _float_
    - Dropout rates for the convolutional layers. A single float will
    apply the same dropout rate to all conv layers. See the [Keras
    Dropout layer](https://keras.io/layers/core/#dropout) for more 
    detail.
- **num_spatial2d_dropout**=`0` : _int_
    - The number of convolutional layers, starting from the beginning
    of the model, for which to apply [SpatialDropout2D](https://keras
    .io/layers/core/#spatialdropout2d) instead of Dropout.
- **dense_dropouts**=`0` : {_tuple_, _list_} of _float_
    - Dropout rates for the dense layers. A single float will apply 
    the same dropout rate to all dense layers.
- **paddings**=`'valid'` : {_tuple_, _list_} of _str_
    - Controls how the filters are convoled with the inputs. See
    the [Keras Conv2D layer](https://keras.io/layers/convolutional/#conv2d) 
    for more detail.
- **data_format**=`'channels_last'` : {`'channels_first'`, `'channels_last'`}
    - Sets which axis is expected to contain the different channels.
    `'channels_first'` appears to have issues with newer versions of 
    tensorflow, so prefer `'channels_last'`.


----

## DNN

Dense Neural Network architecture.

```python
energyflow.archs.DNN(*args, **kwargs)
```

See [`ArchBase`](#archbase) for how to pass in hyperparameters as
well as defaults common to all EnergyFlow neural network models.

**Required DNN Hyperparameters**

- **input_dim** : _int_
    - The number of inputs to the model.
- **dense_sizes** : {_tuple_, _list_} of _int_
    - The number of nodes in the dense layers of the model.

**Default DNN Hyperparameters**

- **acts**=`'relu'` : {_tuple_, _list_} of _str_ or Keras activation
    - Activation functions(s) for the dense layers. A single string or
    activation layer will apply the same activation to all dense layers.
    Keras advanced activation layers are also accepted, either as
    strings (which use the default arguments) or as Keras `Layer` 
    instances. If passing a single `Layer` instance, be aware that this
    layer will be used for all activations and may introduce weight 
    sharing (such as with `PReLU`); it is recommended in this case to 
    pass as many activations as there are layers in the model.See the
    [Keras activations docs](https://keras.io/activations/) for more 
    detail.
- **k_inits**=`'he_uniform'` : {_tuple_, _list_} of _str_ or Keras 
initializer
    - Kernel initializers for the dense layers. A single string 
    will apply the same initializer to all layers. See the
    [Keras initializer docs](https://keras.io/initializers/) for 
    more detail.
- **dropouts**=`0` : {_tuple_, _list_} of _float_
    - Dropout rates for the dense layers. A single float will
    apply the same dropout rate to all layers. See the [Keras
    Dropout layer](https://keras.io/layers/core/#dropout) for more 
    detail.
- **l2_regs**=`0` : {_tuple_, _list_} of _float_
    - $L_2$-regulatization strength for both the weights and biases
    of the dense layers. A single float will apply the same
    $L_2$-regulatization to all layers.


----


## LinearClassifier

Linear classifier that can be either Fisher's linear discriminant
or logistic regression. Relies on the [scikit-learn](https://scikit-learn.org/)
implementations of these classifiers.

```python
energyflow.archs.LinearClassifier(*args, **kwargs)
```

See [`ArchBase`](#archbase) for how to pass in hyperparameters.

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

**Linear Discriminant Analysis Hyperparameters**

- **solver**=`'svd'` : {`'svd'`, `'lsqr'`, `'eigen'`}
    - Which LDA solver to use.
- **tol**=`1e-12` : _float_
    - Threshold used for rank estimation. Notably not a
    convergence parameter.

**Logistic Regression Hyperparameters**

- **LR_hps**=`{}` : _dict_
    - Dictionary of keyword arguments to pass on to the underlying
    `LogisticRegression` model.


----

## Utilities 

Utilities for EnergyFlow architectures, split out from the utils submodule
because these import tensorflow, which the main package avoids doing.

----

### tf_point_cloud_dataset

```python
energyflow.archs.tf_point_cloud_dataset(data_arrs, batch_size=None, dtype='float32')
```

Creates a TensorFlow dataset from NumPy arrays of events of particles,
designed to be used as input to EFN and PFN models. The function uses a
generator to spool events from the arrays as needed and pad them on the fly.
As of EnergyFlow version 1.3.0, it is suggested to use this function to
create TensorFlow datasets to use as input to EFN and PFN training as it can
yield a slight improvement in training and evaluation time.

**Arguments**

- **data_arrs** : _list_ or _tuple_ of _numpy.ndarray_
    -
- **batch_size** : _int_ or `None`
    - If an integer, the dataset will provide batches with that number of
    events when queried. If `None`, no batching is done. Setting this option
    should replace padding a `batch_size` argument directly to the `.fit`
    method of the EFN or PFN.
- **dtype** : _str_
    - The datatype to use in the TensorFlow dataset. Note that 32-bit
    floats are typically sufficient for ML models and so this is the
    default.
- **prefetch** : _int_
    - The maximum number of batches to prepare in advance of their usage
    during training or evaluation. See the [TensorFlow documentation](https:
    //www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch) for more
    details.
- **pad_val** : _float_
    - Events will be padded with particles consisting of this value repeated
    as many times as necessary. This should match the `mask_val` option of
    the EFN or PFN model.

**Returns**

- _tensorflow.data.Dataset_
    - The TensorFlow dataset built from the provided arrays and options. To
    view samples from the dataset, for instance the first five batches, one
    can do:

```python
for sample in tfdataset.take(5).as_numpy_iterator():
    print(sample.shape)
```


----

### tf_gen

```python
energyflow.archs.tf_gen(args)
```

Returns a function that when called returns a generator that yields
samples from the given arrays. Designed to work with
`tf.data.Dataset.from_generator`, though commonly this is handled by
[`tf_point_cloud_dataset`](#tf_point_cloud_dataset).

**Arguments**

- ***args** : arbitrary _numpy.ndarray_ datasets
    - An arbitrary number of arrays.

**Returns**

- _function_
    - A function that when called returns a generator that yields samples
    from the given arrays.


----

