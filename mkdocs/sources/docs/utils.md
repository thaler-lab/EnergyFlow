# Utilities

Utility functions for the EnergyFlow package. The utilities are grouped into the
following submodules:

- [`data_utils`](#data-utils): Utilities for processing datasets as arrays of
events.
aspects.
- [`fastjet_utils`](#fastjet-utils): Utilities for interfacing with the Python
wrapper of the [FastJet](http://fastjet.fr/) package.
- [`image_utils`](#image-utils): Utilities for creating and standardizing images
from collections of particles.
- [`particle_utils`](#particle-utils): Utilities for manipulating particle
properties, including converting between different kinematic representations,
adding/centering collections of four-vectors, and accessing particle properties
including masses and charges by PDG ID.
- [`random_utils`](#random-utils): Utilities for generating random collections
of (massless) four-vectors.

----

## Data Utils

Functions for handling with datasets, including facilitating [train/val/test 
splits](#data_split), [converting](#convert_dtype) the numpy dtype of a
(possibly ragged) array, [padding events](#pad_events) with different numbers
of particles, and [mapping](#remap_pids) PDG ID values to small floating point
values.

----

### convert_dtype

```python
energyflow.convert_dtype(X, dtype=None)
```

Converts the numpy dtype of the given array to the provided value. This
function can handle a ragged array, that is, an object array where the
elements are numpy arrays of a possibly different type, in which case the
function will be recursively applied.

**Arguments**

**Returns**

- _numpy.ndarray_


----

### data_split

```python
energyflow.data_split(*args, train=-1, val=0.0, test=0.1, shuffle=True, perm=None,
                             include_empty=False, return_perm=False)
```

A function to split a dataset into train, validation, and test datasets.

**Arguments**

- ***args** : arbitrary _numpy.ndarray_ datasets
    - An arbitrary number of datasets, each required to have the same number
    of elements, as numpy arrays.
- **train** : {_int_, _float_}
    - If a float, the fraction of elements to include in the training set.
    If an integer, the number of elements to include in the training set.
    The value `-1` is special and means include the remaining part of the
    dataset in the training dataset after the test and (optionally) val
    parts have been removed.
- **val** : {_int_, _float_}
    - If a float, the fraction of elements to include in the validation set.
    If an integer, the number of elements to include in the validation set.
    The value `0` is special and means do not form a validation set.
- **test** : {_int_, _float_}
    - If a float, the fraction of elements to include in the test set. If an
    integer, the number of elements to include in the test set. The value `0`
    is special and means do not form a validation set.
- **shuffle** : _bool_
    - A flag to control whether the dataset is shuffled prior to being split
    into parts.
- **perm** : _numpy.ndarray_ dataset of integers
    - If not `None`, the permutation to use for shuffling the dataset(s).
- **include_empty** : _bool_
    - Whether or not to return empty arrays for datasets that would have
    zero elements in them. This can be useful for setting e.g. `val` or
    `test` to 0 without having to change the unpacking of the result.
- **return_perm** : _bool_
    - Whether or not to return the permutation used for shuffling the
    events. If `True`, it will be included as the final array in the
    returned list.

**Returns**

- _list_
    - A list of the split datasets in train, val, test order. If datasets
    `X`, `Y`, and `Z` were given as `args` (and assuming a non-zero `val`
    and `test`), then [`X_train`, `X_val`, `X_test`, `Y_train`, `Y_val`,
    `Y_test`, `Z_train`, `Z_val`, `Z_test`] will be returned. If, for
    instance, `val` is zero and `include_empty` is `False` then [`X_train`,
    `X_test`, `Y_train`, `Y_test`, `Z_train`, `Z_test`] will be returned. If
    `return_perm` is `True`, the final array will be the permutation used
    for shuffling.


----

### determine_cache_dir

```python
energyflow.determine_cache_dir(cache_dir=None, cache_subdir=None)
```

Determines the path to the specified directory used for caching files. If
`cache_dir` is `None`, the default is to use `'~/.energyflow'` unless the
environment variable `ENERGYFLOW_CACHE_DIR` is set, in which case it is
used.

**Arguments**

- **cache_dir** : _str_ or `None`
    - The path to the top-level cache directory. Defaults to the environment
    variable `ENERGYFLOW_CACHE_DIR`, or `'~/.energyflow'` if that is unset.
- **cache_subdir** : _str_ or `None`
    - Further path component to join to `cache_dir`. Ignored if `None`.

**Returns**

- _str_
    - The path to the cache directory specified by the supplied arguments.


----

### get_examples

```python
energyflow.get_examples(cache_dir=None, which='all', overwrite=False, branch='master')
```

Pulls examples from GitHub. To ensure availability of all examples
update EnergyFlow to the latest version.

**Arguments**

- **cache_dir** : _str_ or `None`
    - The directory where to store/look for the files. If `None`, the
    [`determine_cache_dir`](../utils/#determine_cache_dir) function will be
    used to get the default path. Note that in either case, `'datasets'` is
    appended to the end of the path.
- **which** : {_list_, `'all'`}
    - List of examples to download, or the string `'all'` in which 
    case all the available examples are downloaded.
- **overwrite** : _bool_
    - Whether to overwrite existing files or not.
- **branch** : _str_
    - The EnergyFlow branch from which to get the examples.


----

### pad_events

```python
energyflow.pad_events(X, pad_val=0.0, max_len=None)
```




----

### to_categorical

```python
energyflow.to_categorical(labels, num_classes=None, dtype=None)
```

One-hot encodes class labels.

**Arguments**

- **labels** : _1-d numpy.ndarray_
    - Labels in the range `[0,num_classes)`.
- **num_classes** : {_int_, `None`}
    - The total number of classes. If `None`, taken to be the 
    maximum label plus one.

**Returns**

- _2-d numpy.ndarray_
    - The one-hot encoded labels.


----

### remap_pids

```python
energyflow.remap_pids(events, pid_i=3, error_on_unknown=True)
```

Remaps PDG id numbers to small floats for use in a neural network.
`events` are modified in place and nothing is returned.

**Arguments**

- **events** : _numpy.ndarray_
    - The events as an array of arrays of particles.
- **pid_i** : _int_
    - The column index corresponding to pid information in an event.
- **error_on_unknown** : _bool_
    - Controls whether a `KeyError` is raised if an unknown PDG ID is
    encountered. If `False`, unknown PDG IDs will map to zero.


----

## FastJet Utils

The [FastJet package](http://fastjet.fr/) provides, among other things, fast
jet clustering utilities. Since EnergyFlow 2.0, the [PyFJCore](https://github.
com/pkomiske/PyFJCore) package has been used to provide Python access to
FastJet's classes and algorithms. Keep in mind that if you use these utilities
in EnergyFlow for published research, you are relying on the FastJet library so
please [cite FastJet appropriately](http://fastjet.fr/about.html).

See the [PyFJCore README](https://github.com/pkomiske/PyFJCore/blob/main/
README.md) for more documentation on its functions and classes.

----

### pjs_from_ptyphims

```python
energyflow.pjs_from_ptyphims(ptyphims)
```

Converts an array of particles in hadronic coordinates to FastJet
PseudoJets. See the [`ptyphim_array_to_pseudojets`](https://github.com/
pkomiske/PyFJCore/blob/main/README.md/#NumPy-conversion-functions) method
of PyFJCore.

**Arguments**

- **ptyphims** : _2d numpy.ndarray_
    - An array of particles in hadronic coordinates, `(pt, y, phi, [mass])`;
    the mass is optional. Any additional features are added as user info for
    each PseudoJet, accessible using the `.python_info()` method.

**Returns**

- _tuple_ of _PseudoJet_
    - A Python tuple of `PseudoJet`s corresponding to the input particles.


----

### pjs_from_p4s

```python
energyflow.pjs_from_p4s(p4s)
```

Converts particles in Cartesian coordinates to FastJet PseudoJets. See
the [`epxpypz_array_to_pseudojets`](https://github.com/pkomiske/PyFJCore/
blob/main/README.md/#NumPy-conversion-functions) method of PyFJCore.

**Arguments**

- **p4s** : _2d numpy.ndarray_
    - An array of particles in Cartesian coordinates, `(E, px, py, pz)`. Any
    additional features are added as user info for each PseudoJet,
    accessible using the `.python_info()` method.

**Returns**

- _tuple_ of _PseudoJet_
    - A Python tuple of `PseudoJet`s corresponding to the input particles.


----

### ptyphims_from_pjs

```python
energyflow.ptyphims_from_pjs(pjs, phi_ref=False, mass=True, phi_std=False, float32=False)
```

Extracts hadronic four-vectors from FastJet PseudoJets. See the
[`pseudojets_to_ptyphim_array`](https://github.com/pkomiske/PyFJCore/blob/
main/README.md/#NumPy-conversion-functions) method of PyFJCore.

**Arguments**

- **pjs** : iterable of _PseudoJet_
    - An iterable of PseudoJets (list, tuple, array, etc).
- **phi_ref** : _float_ or `None` or `False`
    - The reference phi value to use for phi fixing. If `False`, then no
    phi fixing is performed. If `None`, then the phi value of the first
    particle is used.
- **mass** : _bool_
    - Whether or not to include the mass in the extracted four-vectors.

**Returns**

- _numpy.ndarray_
    - A 2D array of four-vectors corresponding to the given PseudoJets as
    `(pT, y, phi, [mass])`, where the mass is optional.


----

### p4s_from_pjs

```python
energyflow.p4s_from_pjs(pjs, float32=False)
```

Extracts Cartesian four-vectors from FastJet PseudoJets. See the
[`pseudojets_to_epxpypz_array`](https://github.com/pkomiske/PyFJCore/blob/
main/README.md/#NumPy-conversion-functions) method of PyFJCore.

**Arguments**

- **pjs** : iterable of _PseudoJet_
    - An iterable of PseudoJets (list, tuple, array, etc).

**Returns**

- _numpy.ndarray_
    - A 2D array of four-vectors corresponding to the given PseudoJets as
    `(E, px, py, pz)`.


----

### jet_definition

```python
energyflow.jet_definition(algorithm='ca', R=fastjet.JetDefinition.max_allowable_R, recomb='E_scheme')
```

Creates a JetDefinition from the specified arguments.

**Arguments**

- **algorithm** : _str_ or _int_
    - A string such as `'kt'`, `'akt'`, `'antikt'`, `'ca'`, 
    `'cambridge'`, or `'cambridge_aachen'`; or an integer corresponding to a
    fastjet.JetAlgorithm value.
- **R** : _float_
    - The jet radius. The default value corresponds to `max_allowable_R` as
    defined by the FastJet package.
- **extra** : _float_ or `None`
    - Some jet algorithms, like generalized $k_T$, take an extra parameter.
    If not `None`, `extra` can be used to provide that parameter.
- **recomb** : _str_ or _int_
    - An integer corresponding to a RecombinationScheme, or a string
    specifying a name which is looked up in the PyFJCore module.

**Returns**

- _JetDefinition_
    - A JetDefinition instance corresponding to the given arguments.


----

### cluster

```python
energyflow.cluster(pjs, jetdef=None, N=None, dcut=None, ptmin=0., return_cs=False, **kwargs)
```

Clusters an iterable of PseudoJets. Uses a jet definition that can
either be provided directly or specified using the same keywords as the
`jet_def` function. The jets returned can either be includive, the 
default, or exclusive according to either a maximum number of subjets
or a particular `dcut`.

**Arguments**

- **pjs** : iterable of _PseudoJet_
    - A list of Pseudojets representing particles or other kinematic
    objects that are to be clustered into jets.
- **jetdef** : _JetDefinition_ or `None`
    - The `JetDefinition` used for the clustering. If `None`, the
    keyword arguments are passed on to `jet_def` to create a
    `JetDefinition`.
- **N** : _int_ or `None`
    - If not `None`, then the `exclusive_jets_up_to` method of the
    `ClusterSequence` class is used to get up to `N` exclusive jets.
- **dcut** : _float_ or `None`
    - If not `None`, then the `exclusive_jets` method of the
    `ClusterSequence` class is used to get exclusive jets with the
    provided `dcut` parameter.
- **ptmin** : _float_
    - If both `N` and `dcut` are `None`, then inclusive jets are
    returned using this value as the minimum transverse momentum value.
- ***kwargs** : keyword arguments
    - If `jetdef` is `None`, then these keyword arguments are passed on
    to the `jet_def` function.

**Returns**

- _tuple_ of _PseudoJet_
    - A tuple of PseudoJets corresponding to the clustered jets.


----

### softdrop

```python
energyflow.softdrop(jet, zcut=0.1, beta=0, R=1.0)
```

Implements the SoftDrop grooming algorithm on a jet that has been
found via clustering. Specifically, given a jet, it is recursively
declustered and the softer branch removed until the SoftDrop condition
is satisfied:

$$
\frac{\min(p_{T,1},p_{T,2})}{p_{T,1}+p_{T,2}} > z_{\rm cut}
\left(\frac{\Delta R_{12}}{R}\right)^\beta
$$

where $1$ and $2$ refer to the two PseudoJets declustered at this stage.
See the [SoftDrop paper](https://arxiv.org/abs/1402.2657) for a
complete description of SoftDrop. If you use this function for your
research, please cite [1402.2657](https://doi.org/10.1007/
JHEP05(2014)146).

**Arguments**

- **jet** : _PseudoJet_
    - A PseudoJet that has been obtained from a suitable clustering
    (typically Cambridge/Aachen).
- **zcut** : _float_
    - The $z_{\rm cut}$ parameter of SoftDrop. Should be between `0.0` and
    `1.0`.
- **beta** : _int_ or _float_
    - The $\beta$ parameter of SoftDrop.
- **R** : _float_
    - The jet radius to use for the grooming. Only relevant if `beta != 0.`.

**Returns**

- _PseudoJet_
    - The groomed jet. Note that it will not necessarily have all of the
    same associated structure as the original jet, but it is suitable for
    obtaining kinematic quantities, e.g. [$z_g$](/docs/obs/#zg_from_pj).


----

## Image Utils

Functions for dealing with image representations of events. These are 
not importable from the top level `energyflow` module, but must 
instead be imported from `energyflow.utils`.

----

### pixelate

```python
energyflow.utils.pixelate(jet, npix=33, img_width=0.8, nb_chan=1, norm=True, charged_counts_only=False)
```

A function for creating a jet image from an array of particles.

**Arguments**

- **jet** : _numpy.ndarray_
    - An array of particles where each particle is of the form 
    `[pt,y,phi,pid]` where the particle id column is only 
    used if `nb_chan=2` and `charged_counts_only=True`.
- **npix** : _int_
    - The number of pixels on one edge of the jet image, which is
    taken to be a square.
- **img_width** : _float_
    - The size of one edge of the jet image in the rapidity-azimuth
    plane.
- **nb_chan** : {`1`, `2`}
    - The number of channels in the jet image. If `1`, then only a
    $p_T$ channel is constructed (grayscale). If `2`, then both a 
    $p_T$ channel and a count channel are formed (color).
- **norm** : _bool_
    - Whether to normalize the $p_T$ pixels to sum to `1`.
- **charged_counts_only** : _bool_
    - If making a count channel, whether to only include charged 
    particles. Requires that `pid` information be given.

**Returns**

- _3-d numpy.ndarray_
    - The jet image as a `(npix, npix, nb_chan)` array. Note that the order
    of the channels changed in version 1.0.3.


----

### standardize

```python
energyflow.utils.standardize(*args, channels=None, copy=False, reg=10**-10)
```

Normalizes each argument by the standard deviation of the pixels in 
args[0]. The expected use case would be `standardize(X_train, X_val, 
X_test)`.

**Arguments**

- ***args** : arbitrary _numpy.ndarray_ datasets
    - An arbitrary number of datasets, each required to have
    the same shape in all but the first axis.
- **channels** : _int_
    - A list of which channels (assumed to be the last axis)
    to standardize. `None` is interpretted to mean every channel.
- **copy** : _bool_
    - Whether or not to copy the input arrays before modifying them.
- **reg** : _float_
    - Small parameter used to avoid dividing by zero. It's important
    that this be kept consistent for images used with a given model.

**Returns**

- _list_ 
    - A list of the now-standardized arguments.


----

### zero_center

```python
energyflow.utils.zero_center(args, kwargs)
```

Subtracts the mean of arg[0] from the arguments. The expected 
use case would be `standardize(X_train, X_val, X_test)`.

**Arguments**

- ***args** : arbitrary _numpy.ndarray_ datasets
    - An arbitrary number of datasets, each required to have
    the same shape in all but the first axis.
- **channels** : _int_
    - A list of which channels (assumed to be the last axis)
    to zero center. `None` is interpretted to mean every channel.
- **copy** : _bool_
    - Whether or not to copy the input arrays before modifying them.

**Returns**

- _list_ 
    - A list of the zero-centered arguments.


----

## Particle Utils

Tools for dealing with particle momenta four-vectors. A four-vector can either
be in Cartesian coordinates, `[e,px,py,pz]` (energy, momentum in `x` direction,
momentum in `y` direction, momentum in `z` direction), or hadronic coordinates, 
`[pt,y,phi,m]` (transverse momentum, rapidity, azimuthal angle, mass), which
are related via:

\[p_T=\sqrt{p_x^2+p_y^2},\quad y=\text{arctanh}\,\frac{p_z}{E},\quad 
\phi=\arctan_2\frac{p_y}{p_x},\quad m=\sqrt{E^2-p_x^2-p_y^2-p_z^2}\]

and inversely:

\[E=\cosh y\sqrt{p_T^2+m^2},\quad p_x=p_T\cos\phi,\quad 
p_y=p_T\sin\phi,\quad p_z=\sinh y\sqrt{p_T^2+m^2}.\]

The pseudorapidity `eta` can be obtained from a Cartesian four-momentum as:

\[\eta=\text{arctanh}\,\frac{p_z}{|\vec p|},\quad 
|\vec p|\equiv\sqrt{p_x^2+p_y^2+p_z^2},\]

and is related to the rapidity via

\[\eta=\text{arcsinh}\left(\sinh y\,\left(1+m^2/p_T^2\right)^{1/2}\right),\quad 
y=\text{arcsinh}\left(\sinh \eta\,\left(1+m^2/p_T^2\right)^{-1/2}\right).\]

Note that the above formulas are numerically stable up to values of rapidity or
pseudorapidity of a few hundred, above which the formulas have numerical issues. 
In this case, a different but equivalent formulae are used that are numerically
stable in this region. In all cases, the $p_T\to0$ limit produces infinite
values.

In the context of this package, an "event" is a two-dimensional numpy array
with shape `(M,4)` where `M` is the multiplicity. An array of events is a 
three-dimensional array with shape `(N,M,4)` where `N` is the number of events.
The valid inputs and outputs of the functions here will be described using
this terminology.

----

### ptyphims_from_p4s

```python
energyflow.ptyphims_from_p4s(p4s, phi_ref=None, mass=True)
```

Convert to hadronic coordinates `[pt,y,phi,m]` from Cartesian
coordinates. All-zero four-vectors are left alone.

**Arguments**

- **p4s** : _numpy.ndarray_ or _list_
    - A single particle, event, or array of events in Cartesian coordinates.
- **phi_ref** : {`None`, `'hardest'`, _float_, _numpy.ndarray_}
    - Used to help deal with the fact that $\phi$ is a periodic coordinate.
    If a float (which should be in $[0,2\pi)$), all phi values will be
    within $\pm\pi$ of this reference value. If `'\hardest'`, the phi of
    the hardest particle is used as the reference value. If `None`, all
    phis will be in the range $[0,2\pi)$. An array is accepted in the case
    that `p4s` is an array of events, in which case the `phi_ref` array
    should have shape `(N,)` where `N` is the number of events.
- **mass** : _bool_
    - Whether or not to include particle masses.

**Returns**

- _numpy.ndarray_
    - An array of hadronic four-momenta with the same shape as the input.


----

### pts_from_p4s

```python
energyflow.pts_from_p4s(p4s)
```

Calculate the transverse momenta of a collection of four-vectors.

**Arguments**

- **p4s** : _numpy.ndarray_ or _list_
    - A single particle, event, or array of events in Cartesian coordinates.

**Returns**

- _numpy.ndarray_
    - An array of transverse momenta with shape `p4s.shape[:-1]`.


----

### pt2s_from_p4s

```python
energyflow.pt2s_from_p4s(p4s)
```

Calculate the squared transverse momenta of a collection of four-vectors.

**Arguments**

- **p4s** : _numpy.ndarray_ or _list_
    - A single particle, event, or array of events in Cartesian coordinates.

**Returns**

- _numpy.ndarray_
    - An array of squared transverse momenta with shape `p4s.shape[:-1]`.


----

### ys_from_p4s

```python
energyflow.ys_from_p4s(p4s)
```

Calculate the rapidities of a collection of four-vectors. Returns zero
for all-zero particles

**Arguments**

- **p4s** : _numpy.ndarray_ or _list_
    - A single particle, event, or array of events in Cartesian coordinates.

**Returns**

- _numpy.ndarray_
    - An array of rapidities with shape `p4s.shape[:-1]`.


----

### etas_from_p4s

```python
energyflow.etas_from_p4s(p4s)
```

Calculate the pseudorapidities of a collection of four-vectors. Returns
zero for all-zero particles

**Arguments**

- **p4s** : _numpy.ndarray_ or _list_
    - A single particle, event, or array of events in Cartesian coordinates.

**Returns**

- _numpy.ndarray_
    - An array of pseudorapidities with shape `p4s.shape[:-1]`.


----

### phis_from_p4s

```python
energyflow.phis_from_p4s(p4s, phi_ref=None)
```

Calculate the azimuthal angles of a collection of four-vectors.

**Arguments**

- **p4s** : _numpy.ndarray_ or _list_
    - A single particle, event, or array of events in Cartesian coordinates.
- **phi_ref** : {_float_, _numpy.ndarray_, `None`, `'hardest'`}
    - Used to help deal with the fact that $\phi$ is a periodic coordinate.
    If a float (which should be in $[0,2\pi)$), all phi values will be
    within $\pm\pi$ of this reference value. If `'\hardest'`, the phi of
    the hardest particle is used as the reference value. If `None`, all
    phis will be in the range $[0,2\pi)$. An array is accepted in the case
    that `p4s` is an array of events, in which case the `phi_ref` array
    should have shape `(N,)` where `N` is the number of events.

**Returns**

- _numpy.ndarray_
    - An array of azimuthal angles with shape `p4s.shape[:-1]`.


----

### m2s_from_p4s

```python
energyflow.m2s_from_p4s(p4s)
```

Calculate the squared masses of a collection of four-vectors.

**Arguments**

- **p4s** : _numpy.ndarray_ or _list_
    - A single particle, event, or array of events in Cartesian coordinates.

**Returns**

- _numpy.ndarray_
    - An array of squared masses with shape `p4s.shape[:-1]`.


----

### ms_from_p4s

```python
energyflow.ms_from_p4s(p4s)
```

Calculate the masses of a collection of four-vectors.

**Arguments**

- **p4s** : _numpy.ndarray_ or _list_
    - A single particle, event, or array of events in Cartesian coordinates.

**Returns**

- _numpy.ndarray_
    - An array of masses with shape `p4s.shape[:-1]`.


----

### ms_from_ps

```python
energyflow.ms_from_ps(ps)
```

Calculate the masses of a collection of Lorentz vectors in two or more
spacetime dimensions.

**Arguments**

- **ps** : _numpy.ndarray_ or _list_
    - A single particle, event, or array of events in Cartesian
    coordinates in $d\ge2$ spacetime dimensions.

**Returns**

- _numpy.ndarray_
    - An array of masses with shape `ps.shape[:-1]`.


----

### p4s_from_ptyphims

```python
energyflow.p4s_from_ptyphims(ptyphims)
```

Calculate Cartesian four-vectors from transverse momenta, rapidities,
azimuthal angles, and (optionally) masses for each input.

**Arguments**

- **ptyphims** : _numpy.ndarray_ or _list_
    - A single particle, event, or array of events in hadronic coordinates.
    The mass is optional and if left out will be taken to be zero.

**Returns**

- _numpy.ndarray_
    - An array of Cartesian four-vectors.


----

### p4s_from_ptyphipids

```python
energyflow.p4s_from_ptyphipids(ptyphipids, error_on_unknown=False)
```

Calculate Cartesian four-vectors from transverse momenta, rapidities,
azimuthal angles, and particle IDs for each input. The particle IDs are
used to lookup the mass of the particle. Transverse momenta should have
units of GeV when using this function.

**Arguments**

- **ptyphipids** : _numpy.ndarray_ or _list_
    - A single particle, event, or array of events in hadronic coordinates
    where the mass is replaced by the PDG ID of the particle.
- **error_on_unknown** : _bool_
    - See the corresponding argument of [`pids2ms`](#pids2ms).

**Returns**

- _numpy.ndarray_
    - An array of Cartesian four-vectors with the same shape as the input.


----

### etas_from_pts_ys_ms

```python
energyflow.etas_from_pts_ys_ms(pts, ys, ms)
```

Calculate pseudorapidities from transverse momenta, rapidities, and masses.
All input arrays should have the same shape.

**Arguments**

- **pts** : _numpy.ndarray_
    - Array of transverse momenta.
- **ys** : _numpy.ndarray_
    - Array of rapidities.
- **ms** : _numpy.ndarray_
    - Array of masses.

**Returns**

- _numpy.ndarray_
    - Array of pseudorapidities with the same shape as `ys`.


----

### ys_from_pts_etas_ms

```python
energyflow.ys_from_pts_etas_ms(pts, etas, ms)
```

Calculate rapidities from transverse momenta, pseudorapidities, and masses.
All input arrays should have the same shape.

**Arguments**

- **pts** : _numpy.ndarray_
    - Array of transverse momenta.
- **etas** : _numpy.ndarray_
    - Array of pseudorapidities.
- **ms** : _numpy.ndarray_
    - Array of masses.

**Returns**

- _numpy.ndarray_
    - Array of rapidities with the same shape as `etas`.


----

### phi_fix

```python
energyflow.phi_fix(phis, phi_ref, copy=True, dtype=<class 'float'>)
```

A function to ensure that all phis are within $\pi$ of `phi_ref`. It is
assumed that all starting phi values are $\pm 2\pi$ of `phi_ref`.

**Arguments**

- **phis** : _numpy.ndarray_ or _list_
    - Array of phi values.
- **phi_ref** : {_float_ or _numpy.ndarray_}
    - A reference value used so that all phis will be within $\pm\pi$ of
    this value. Should have a shape of `phis.shape[:-1]`.
- **copy** : _bool_
    - Determines if `phis` are copied or not. If `False` then `phis` is
    modified in place.

**Returns**

- _numpy.ndarray_
    - An array of the fixed phi values.


----

### sum_ptyphims

```python
energyflow.sum_ptyphims(ptyphims, scheme='escheme')
```

Add a collection of four-vectors that are expressed in hadronic
coordinates by first converting to Cartesian coordinates and then summing.

**Arguments**

- **ptyphims** : _numpy.ndarray_ or _list_
    - An event in hadronic coordinates. The mass is optional and if left
    out will be taken to be zero.
- **scheme** : _str_
    - A string specifying a recombination scheme for adding four-vectors
    together. Currently supported options are `'escheme'`, which adds the
    vectors in Cartesian coordinates, and `'ptscheme'`, which sums the pTs
    of each particle and places the jet axis at the pT-weighted centroid
    in the rapidity-azimuth plane. Note that `'ptscheme'` will return a
    three-vector consisting of the jet `[pT,y,phi]` with no mass value.

**Returns**

- _numpy.ndarray_
    - Array of summed four-vectors, in hadronic coordinates. Note that when
    `scheme` is `'escheme'`, the $\phi$ value of the hardest particle is
    used as the `phi_ref` when converting back to hadronic coordinates.


----

### sum_ptyphipids

```python
energyflow.sum_ptyphipids(ptyphipids, scheme='escheme', error_on_unknown=False)
```

Add a collection of four-vectors that are expressed as
`[pT,y,phi,pdgid]`.

**Arguments**

- **ptyphipids** : _numpy.ndarray_ or _list_
    - A single particle or event in hadronic coordinates where the mass
    is replaced by the PDG ID of the particle.
- **scheme** : _str_
    - See the argument of the same name of [`sum_ptyphims`](#sum_ptyphims).
- **error_on_unknown** : _bool_
    - See the corresponding argument of [`pids2ms`](#pids2ms).

**Returns**

- _numpy.ndarray_
    - Array of summed four-vectors, in hadronic coordinates. Note that when
    `scheme` is `'escheme'`, the $\phi$ value of the hardest particle is
    used as the `phi_ref` when converting back to hadronic coordinates.


----

### center_ptyphims

```python
energyflow.center_ptyphims(ptyphims, axis=None, center='escheme', copy=True)
```

Center a collection of four-vectors according to a calculated or 
provided axis.

**Arguments**

- **ptyphims** : _numpy.ndarray_ or _list_
    - An event in hadronic coordinates. The mass is optional and if left
    out will be taken to be zero.
- **axis** : _numpy.ndarray_
    - If not `None`, the `[y,phi]` values to use for centering.
- **center** : _str_
    - The centering scheme to be used. Valid options are the same as the
    `scheme` argument of [`sum_ptyphims`](#sum_ptyphims).
- **copy** : _bool_
    - Whether or not to copy the input array.

**Returns**

- _numpy.ndarray_
    - An array of hadronic four-momenta with the positions centered around
    the origin.


----

### rotate_ptyphims

```python
energyflow.rotate_ptyphims(ptyphims, rotate='ptscheme', center=None, copy=True)
```

Rotate a collection of four-vectors to vertically align the principal
component of the energy flow. The principal component is obtained as the
eigenvector of the energy flow with the largest eigenvalue. It is only
defined up to a sign, however it is ensured that there is more total pT in 
the top half of the rapidity-azimuth plane.

**Arguments**

- **ptyphims** : _numpy.ndarray_ or _list_
    - An event in hadronic coordinates. The mass is optional and if left
    out will be taken to be zero.
- **rotate** : _str_
    - The rotation scheme to be used. Currently, only `'ptscheme'` is
    supported, which causes the rotation to take place in the 
    rapidity-azimuth plane.
- **center** : _str_ or `None`
    - If not `None`, the event will be centered prior to rotation and this
    argument will be passed on to `center_ptyphims` as the centering
    scheme.
- **copy** : _bool_
    - Whether or not to copy the input array.

**Returns**

- _numpy.ndarray_
    - An array of hadronic four-momenta with the positions rotated around
    the origin.


----

### reflect_ptyphims

```python
energyflow.reflect_ptyphims(ptyphims, which='both', center=None, copy=True)
```

Reflect a collection of four-vectors to arrange the highest-pT
half or quadrant to have positive rapidity-azimuth coordinates.

**Arguments**

- **ptyphims** : _numpy.ndarray_
    - An event in hadronic coordinates. The mass is optional and is not
    used as a part of this function.
- **which** : {`'both'`, `'x'`, `'y'`}
    - Controls which axes to consider reflecting over. `'both'` includes
    `'x'` and `'y'`.
- **center** : _str_ or `None`
    - If not `None`, the centering scheme to use prior to performing
    reflections.
- **copy** : _bool_
    - Whether or not to copy the input array.


----

### pids2ms

```python
energyflow.pids2ms(pids, error_on_unknown=False)
```

Map an array of [Particle Data Group IDs](http://pdg.lbl.gov/2018/
reviews/rpp2018-rev-monte-carlo-numbering.pdf) to an array of the
corresponding particle masses (in GeV).

**Arguments**

- **pids** : _numpy.ndarray_ or _list_
    - An array of numeric (float or integer) PDG ID values.
- **error_on_unknown** : _bool_
    - Controls whether a `KeyError` is raised if an unknown PDG ID is
    encountered. If `False`, unknown PDG IDs will map to zero.

**Returns**

- _numpy.ndarray_
    - An array of masses in GeV.


----

### pids2chrgs

```python
energyflow.pids2chrgs(pids, error_on_unknown=False)
```

Map an array of [Particle Data Group IDs](http://pdg.lbl.gov/2018/
reviews/rpp2018-rev-monte-carlo-numbering.pdf) to an array of the
corresponding particle charges (in fundamental units where the charge
of the electron is -1).

**Arguments**

- **pids** : _numpy.ndarray_ or _list_
    - An array of numeric (float or integer) PDG ID values.
- **error_on_unknown** : _bool_
    - Controls whether a `KeyError` is raised if an unknown PDG ID is
    encountered. If `False`, unknown PDG IDs will map to zero.

**Returns**

- _numpy.ndarray_
    - An array of charges as floats.


----

### ischrgd

```python
energyflow.ischrgd(pids, ignored_pids=None)
```

Compute a boolean mask according to if the given PDG ID corresponds
to a particle of non-zero charge.

**Arguments**

- **pids** : _numpy.ndarray_
    - An array of numeric (float or integer) PDG ID values.
- **ignored_pids** : _numpy.ndarray_ or `None`
    - If not `None`, the PDG IDs in this array will not be considered
    charged, for instance to avoid counting quarks as charged particles.

**Returns**

- _numpy.ndarray_
    - A boolean mask corresponding to which particles are charged.


----

### particle_properties

```python
energyflow.particle_properties()
```

Accesses the global dictionary of particle properties. The keys are
non-negative PDGIDs and the values are tuples of properties. Currently,
each tuple has two values, the first is the charge in fundamental units
and the second is the mass in GeV.


----

### particle_masses

```python
energyflow.particle_masses()
```

Accesses the global dictionary of particle masses. The keys are
non-negative PDGIDs and the values are the particle masses in GeV.


----

### particle_charges

```python
energyflow.particle_charges()
```

Accesses the global dictionary of particle masses. The keys are
non-negative PDGIDs and the values are the particle charged in fundamental
units.


----

### charged_pids

```python
energyflow.charged_pids()
```

Accesses the global set of PDGID values that have a non-zero charge.


----

### flat_metric

```python
energyflow.flat_metric(dim)
```

The Minkowski metric in `dim` spacetime dimensions in the mostly-minus
convention.

**Arguments**

- **dim** : _int_
    - The number of spacetime dimensions (thought to be four in our 
    universe).

**Returns**

- _1-d numpy.ndarray_
    - A `dim`-length, one-dimensional (not matrix) array equal to 
    `[+1,-1,...,-1]`.


----

## Random Utils

Functions to generate random sets of four-vectors. Includes an implementation
of the [RAMBO](https://doi.org/10.1016/0010-4655(86)90119-0) algorithm for
sampling uniform M-body massless phase space. Also includes other functions for
various random, non-center of momentum, and non-uniform sampling.

----

### gen_random_events

```python
energyflow.gen_random_events(nevents, nparticles, dim=4, mass=0.0)
```

Generate random events with a given number of particles in a given
spacetime dimension. The spatial components of the momenta are
distributed uniformly in $[-1,+1]$. These events are not guaranteed to 
uniformly sample phase space.

**Arguments**

- **nevents** : _int_
    - Number of events to generate.
- **nparticles** : _int_ or _tuple_/_list_
    - If an integet, the exact number of particles in each event. If a
    tuple/list of length 2, then this is treated as the interval
    `[low, high)` and the particle multiplicities will be uniformly sampled
    from this interval.
- **dim** : _int_
    - Number of spacetime dimensions.
- **mass** : _float_ or `'random'`
    - Mass of the particles to generate. Can be set to `'random'` to obtain
    a different random mass for each particle.

**Returns**

- _numpy.ndarray_
    - An `(nevents,nparticles,dim)` array of events. The particles are
    specified as `[E,p1,p2,...]`. If `nevents` is 1, then that axis is
    dropped.


----

### gen_random_events_mcom

```python
energyflow.gen_random_events_mcom(nevents, nparticles, dim=4)
```

Generate random events with a given number of massless particles in a
given spacetime dimension. The total momentum are made to sum to zero. These
events are not guaranteed to uniformly sample phase space.

**Arguments**

- **nevents** : _int_
    - Number of events to generate.
- **nparticles** : _int_
    - Number of particles in each event.
- **dim** : _int_
    - Number of spacetime dimensions.

**Returns**

- _numpy.ndarray_
    - An `(nevents,nparticles,dim)` array of events. The particles are
    specified as `[E,p1,p2,...]`.


----

### gen_massless_phase_space

```python
energyflow.gen_massless_phase_space(nevents, nparticles, energy=1.0)
```

Implementation of the [RAMBO](https://doi.org/10.1016/0010-4655(86)
90119-0) algorithm for uniformly sampling massless M-body phase space for
any center of mass energy.

**Arguments**

- **nevents** : _int_
    - Number of events to generate.
- **nparticles** : _int_
    - Number of particles in each event.
- **energy** : _float_
    - Total center of mass energy of each event.

**Returns**

- _numpy.ndarray_
    - An `(nevents,nparticles,4)` array of events. The particles are
    specified as `[E,p_x,p_y,p_z]`. If `nevents` is 1 then that axis is
    dropped.


----

### random

```python
energyflow.random
```

In NumPy versions >= 1.16, this object is obtained from
`numpy.random.default_rng()`. Otherwise, it's equivalent to `numpy.random`.


----

