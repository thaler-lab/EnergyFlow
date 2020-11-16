# Utilities

## Particle Tools

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
energyflow.ptyphims_from_p4s(p4s, phi_ref=None)
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

### phi_fix

```python
energyflow.phi_fix(phis, phi_ref, copy=True)
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

## Random Events

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
- **nparticles** : _int_
    - Number of particles in each event.
- **dim** : _int_
    - Number of spacetime dimensions.
- **mass** : _float_ or `'random'`
    - Mass of the particles to generate. Can be set to `'random'` to obtain
    a different random mass for each particle.

**Returns**

- _numpy.ndarray_
    - An `(nevents,nparticles,dim)` array of events. The particles 
    are specified as `[E,p1,p2,...]`. If `nevents` is 1 then that axis is
    dropped.


----

### gen_random_events_mcom

```python
energyflow.gen_random_events_mcom(nevents, nparticles, dim=4)
```

Generate random events with a given number of massless particles in a
given spacetime dimension. The total momentum are made to sum
to zero. These events are not guaranteed to uniformly sample phase space.

**Arguments**

- **nevents** : _int_
    - Number of events to generate.
- **nparticles** : _int_
    - Number of particles in each event.
- **dim** : _int_
    - Number of spacetime dimensions.

**Returns**

- _numpy.ndarray_
    - An `(nevents,nparticles,dim)` array of events. The particles 
    are specified as `[E,p1,p2,...]`.


----

### gen_massless_phase_space

```python
energyflow.gen_massless_phase_space(nevents, nparticles, energy=1.0)
```

Implementation of the [RAMBO](https://doi.org/10.1016/0010-4655(86)90119-0)
algorithm for uniformly sampling massless M-body phase space for any center
of mass energy.

**Arguments**

- **nevents** : _int_
    - Number of events to generate.
- **nparticles** : _int_
    - Number of particles in each event.
- **energy** : _float_
    - Total center of mass energy of each event.

**Returns**

- _numpy.ndarray_
    - An `(nevents,nparticles,4)` array of events. The particles 
    are specified as `[E,p_x,p_y,p_z]`. If `nevents` is 1 then that axis is
    dropped.


----

## Data Tools

Functions for dealing with datasets. These are not importable from
the top level `energyflow` module, but must instead be imported 
from `energyflow.utils`.

----

### get_examples

```python
energyflow.utils.get_examples(path='~/.energyflow', which='all', overwrite=False)
```

Pulls examples from GitHub. To ensure availability of all examples
update EnergyFlow to the latest version.

**Arguments**

- **path** : _str_
    - The destination for the downloaded files. Note that `examples`
    is automatically appended to the end of this path.
- **which** : {_list_, `'all'`}
    - List of examples to download, or the string `'all'` in which 
    case all the available examples are downloaded.
- **overwrite** : _bool_
    - Whether to overwrite existing files or not.


----

### data_split

```python
energyflow.utils.data_split(*args, train=-1, val=0.0, test=0.1, shuffle=True)
```

A function to split a dataset into train, test, and optionally 
validation datasets.

**Arguments**

- ***args** : arbitrary _numpy.ndarray_ datasets
    - An arbitrary number of datasets, each required to have
    the same number of elements, as numpy arrays.
- **train** : {_int_, _float_}
    - If a float, the fraction of elements to include in the training
    set. If an integer, the number of elements to include in the
    training set. The value `-1` is special and means include the
    remaining part of the dataset in the training dataset after
    the test and (optionally) val parts have been removed
- **val** : {_int_, _float_}
    - If a float, the fraction of elements to include in the validation
    set. If an integer, the number of elements to include in the
    validation set. The value `0` is special and means do not form
    a validation set.
- **test** : {_int_, _float_}
    - If a float, the fraction of elements to include in the test
    set. If an integer, the number of elements to include in the
    test set.
- **shuffle** : _bool_
    - A flag to control whether the dataset is shuffled prior to
    being split into parts.

**Returns**

- _list_
    - A list of the split datasets in train, [val], test order. If 
    datasets `X`, `Y`, and `Z` were given as `args` (and assuming a
    non-zero `val`), then [`X_train`, `X_val`, `X_test`, `Y_train`, 
    `Y_val`, `Y_test`, `Z_train`, `Z_val`, `Z_test`] will be returned.


----

### to_categorical

```python
energyflow.utils.to_categorical(labels, num_classes=None)
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
energyflow.utils.remap_pids(events, pid_i=3, error_on_unknown=True)
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

## Image Tools

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

## FastJet Tools

The [FastJet package](http://fastjet.fr/) provides, among other things, fast
jet clustering utilities. It is written in C++ and includes a Python interface
that is easily installed at compile time by passing the `--enable-pyext` flag
to `configure`. If you use this module for published research, please [cite
FastJet appropriately](http://fastjet.fr/about.html).

The core of EnergyFlow does not rely on FastJet, and hence it is not required
to be installed, but the following utilities are available assuming that
`import fastjet` succeeds in your Python environment (if not, no warnings or
errors will be issued but this module will not be usable).

----

### pjs_from_ptyphims

```python
energyflow.pjs_from_ptyphims(ptyphims)
```

Converts particles in hadronic coordinates to FastJet PseudoJets.

**Arguments**

- **ptyphims** : _2d numpy.ndarray_
    - An array of particles in hadronic coordinates. The mass is
    optional and will be taken to be zero if not present.

**Returns**

- _list_ of _fastjet.PseudoJet_
    - A list of PseudoJets corresponding to the particles in the given
    array.


----

### ptyphims_from_pjs

```python
energyflow.ptyphims_from_pjs(pjs, phi_ref=None, mass=True)
```

Extracts hadronic four-vectors from FastJet PseudoJets.

**Arguments**

- **pjs** : _list_ of _fastjet.PseudoJet_
    - An iterable of PseudoJets.
- **phi_ref** : _float_ or `None`
    - The reference phi value to use for phi fixing. If `None`, then no
    phi fixing is performed.
- **mass** : _bool_
    - Whether or not to include the mass in the extracted four-vectors.

**Returns**

- _numpy.ndarray_
    - An array of four-vectors corresponding to the given PseudoJets as
    `[pT, y, phi, m]`, where the mass is optional.


----

### cluster

```python
energyflow.cluster(pjs, algorithm='ca', R=1000.0)
```

Clusters a list of PseudoJets according to a specified jet
algorithm and jet radius.

**Arguments**

- **pjs** : _list_ of _fastjet.PseudoJet_
    - A list of Pseudojets representing particles or other kinematic
    objects that are to be clustered into jets.
- **algorithm** : {'kt', 'antikt', 'ca', 'cambridge', 'cambridge_aachen'}
    - The jet algorithm to use during the clustering. Note that the
    last three options all refer to the same strategy and are provided
    because they are all used by the FastJet Python package.
- **R** : _float_
    - The jet radius. The default value corresponds to
    `max_allowable_R` as defined by the FastJet python package.

**Returns**

- _list_ of _fastjet.PseudoJet_
    - A list of PseudoJets corresponding to the clustered jets.


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

- **jet** : _fastjet.PseudoJet_
    - A FastJet PseudoJet that has been obtained from a suitable
    clustering (typically Cambridge/Aachen).
- **zcut** : _float_
    - The $z_{\rm cut}$ parameter of SoftDrop. Should be between `0`
    and `1`.
- **beta** : _int_ or _float_
    - The $\beta$ parameter of SoftDrop.
- **R** : _float_
    - The jet radius to use for the grooming. Only relevant if `beta!=0`.

**Returns**

- _fastjet.PseudoJet_
    - The groomed jet. Note that it will not necessarily have all of
    the same associated structure as the original jet, but it is
    suitable for obtaining kinematic quantities, e.g. [$z_g$](/docs/
    obs/#zg_from_pj).


----

