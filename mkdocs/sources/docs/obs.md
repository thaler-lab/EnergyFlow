# Observables

Implementations of come collider physics observables. Some observables
require the [FastJet](http://fastjet.fr/) Python interface to be importable;
if it's not, no warnings or errors will be issued, the observables will simply
not be included in this module.

----

## image_activity

```python
energyflow.image_activity(ptyphis, f=0.95, R=1.0, npix=33, center=None, axis=None)
```

Image activity, also known as $N_f$, is the minimum number of pixels
in an image that contain a fraction $f$ of the total pT.

**Arguments**

- **ptyphis** : _2d numpy.ndarray_
    - Array of particles in hadronic coordinates; the mass is optional
    since it is not used in the computation of this observable.
- **f** : _float_
    - The fraction $f$ of total pT that is to be contained by the pixels.
- **R** : _float_
    - Half of the length of one side of the square space to tile with
    pixels when forming the image. For a conical jet, this should typically
    be the jet radius.
- **npix** : _int_
    - The number of pixels along one dimension of the image, such that the
    image has shape `(npix,npix)`.
- **center** : _str_ or `None`
    - If not `None`, the centering scheme to use to center the particles
    prior to calculating the image activity. See the option of the same
    name for [`center_ptyphims`](/docs/utils/#center_ptyphims).
- **axis** : _numpy.ndarray_ or `None`
    - If not `None`, the `[y,phi]` values to use for centering. If `None`,
    the center of the image will be at `(0,0)`.

**Returns**

- _int_
    - The image activity defined for the specified image paramters.


----

## zg

```python
energyflow.zg(ptyphims, zcut=0.1, beta=0, R=1.0, algorithm='ca')
```

Groomed momentum fraction of a jet, as calculated on an array of
particles in hadronic coordinates. First, the particles are converted
to FastJet PseudoJets and clustered according to the specified
algorithm. Second, the jet is groomed according to the specified
SoftDrop parameters and the momentum fraction of the surviving pair of
Pseudojets is computed. See the [SoftDrop paper](https://arxiv.org/abs/
1402.2657) for a complete description of SoftDrop.

**Arguments**

- **ptyphims** : _numpy.ndarray_
    - An array of particles in hadronic coordinates that will be
    clustered into a single jet and groomed.
- **zcut** : _float_
    - The $z_{\rm cut}$ parameter of SoftDrop. Should be between `0`
    and `1`.
- **beta** : _int_ or _float_
    - The $\beta$ parameter of SoftDrop.
- **R** : _float_
    - The jet radius to use for the grooming. Only relevant if `beta!=0`.
- **algorithm** : {'kt', 'ca', 'antikt'}
    - The jet algorithm to use when clustering the particles. Same as
    the argument of the same name of [`cluster`](/docs/utils/#cluster).

**Returns**

- _float_
    - The groomed momentum fraction of the given jet.


----

## zg_from_pj

```python
energyflow.zg_from_pj(pseudojet, zcut=0.1, beta=0, R=1.0)
```

Groomed momentum fraction $z_g$, as calculated on an ungroomed (but
already clustered) FastJet PseudoJet object. First, the jet is groomed
according to the specified SoftDrop parameters and then the momentum
fraction of the surviving pair of Pseudojets is computed. See the
[SoftDrop paper](https://arxiv.org/abs/1402.2657) for a complete
description of SoftDrop. This version of $z_g$ is provided in addition
to the above function so that a jet does not need to be reclustered if
multiple grooming parameters are to be used.

**Arguments**

- **pseudojet** : _fastjet.PseudoJet_
    - A FastJet PseudoJet that has been obtained from a suitable
    clustering (typically Cambridge/Aachen for SoftDrop).
- **zcut** : _float_
    - The $z_{\rm cut}$ parameter of SoftDrop. Should be between `0`
    and `1`.
- **beta** : _int_ or _float_
    - The $\beta$ parameter of SoftDrop.
- **R** : _float_
    - The jet radius to use for the grooming. Only relevant if `beta!=0`.

**Returns**

- _float_
    - The groomed momentum fraction of the given jet.


----

## D2

Ratio of EFPs (specifically, energy correlation functions) designed to
tag two prong signals. In graphs, the formula is:

<img src="https://github.com/thaler-lab/EnergyFlow/raw/images/D2.png"
class="obs_center" width="20%"/>

For additional information, see the [original paper](https://arxiv.org/
abs/1409.6298).

```python
energyflow.D2(measure='hadr', beta=2, strassen=False, reg=0., kappa=1, normed=True,
              coords=None, check_input=True)
```

Since a `D2` defines and holds a `Measure` instance, all `Measure`
keywords are accepted.

**Arguments**

- **measure** : {`'hadr'`, `'hadrdot'`, `'hadrefm'`, `'ee'`, `'eeefm'`}
    - The choice of measure. See [Measures](../measures) for additional
    info.
- **beta** : _float_
    - The parameter $\beta$ appearing in the measure. Must be greater
    than zero.
- **strassen** : _bool_
    - Whether to use matrix multiplication to speed up the evaluation.
    Not recommended when $\beta=2$ since EFMs are faster.
- **reg** : _float_
    - A regularizing value to be added to the denominator in the event
    that it is zero. Should typically be something less than 1e-30.
- **kappa** : {_float_, `'pf'`}
    - If a number, the energy weighting parameter $\kappa$. If `'pf'`,
    use $\kappa=v-1$ where $v$ is the valency of the vertex.
- **normed** : _bool_
    - Controls normalization of the energies in the measure.
- **coords** : {`'ptyphim'`, `'epxpypz'`, `None`}
    - Controls which coordinates are assumed for the input. See
    [Measures](../measures) for additional info.
- **check_input** : _bool_
    - Whether to check the type of the input each time or assume the
    first input type.

### compute

```python
compute(event=None, zs=None, thetas=None, nhats=None)
```

Computes the value of the observable on a single event. Note that
the observable object is also callable, in which case this method is
invoked.

**Arguments**

- **event** : 2-d array_like or `fastjet.PseudoJet`
    - The event as an array of particles in the coordinates specified
    by `coords`.
- **zs** : 1-d array_like
    - If present, `thetas` must also be present, and `zs` is used in place
    of the energies of an event.
- **thetas** : 2-d array_like
    - If present, `zs` must also be present, and `thetas` is used in place
    of the pairwise angles of an event.
- **nhats** : 2-d array like
    - If present, `zs` must also be present, and `nhats` is used in place
    of the scaled particle momenta. Only applicable when EFMs are being
    used.

**Returns**

- _float_
    - The observable value.

### batch_compute

```python
batch_compute(events, n_jobs=None)
```

Computes the value of the observable on several events.

**Arguments**

- **events** : array_like or `fastjet.PseudoJet`
    - The events as an array of arrays of particles in coordinates
    matching those anticipated by `coords`.
- **n_jobs** : _int_ or `None`
    - The number of worker processes to use. A value of `None` will
    use as many processes as there are CPUs on the machine.

**Returns**

- _1-d numpy.ndarray_
    - A vector of the observable values for each event.

### properties

#### efpset

```python
efpset
```

`EFPSet` held by the object to compute fundamental EFP values.


----

## C2

Ratio of Energy Correlation Functions designed to tag two prong signals.
In graphs, the formula is:

<img src="https://github.com/thaler-lab/EnergyFlow/raw/images/C2.png"
class="obs_center" width="20%"/>

For additional information, see the [original paper](https://arxiv.org/
abs/1305.0007).

```python
energyflow.C2(measure='hadr', beta=2, strassen=False, reg=0., kappa=1, normed=True,
              coords=None, check_input=True)
```

Since a `C2` defines and holds a `Measure` instance, all `Measure`
keywords are accepted.

**Arguments**

- **measure** : {`'hadr'`, `'hadrdot'`, `'hadrefm'`, `'ee'`, `'eeefm'`}
    - The choice of measure. See [Measures](../measures) for additional
    info.
- **beta** : _float_
    - The parameter $\beta$ appearing in the measure. Must be greater
    than zero.
- **strassen** : _bool_
    - Whether to use matrix multiplication to speed up the evaluation.
    Not recommended when $\beta=2$ since EFMs are faster.
- **reg** : _float_
    - A regularizing value to be added to the denominator in the event
    that it is zero. Should typically be something less than 1e-30.
- **kappa** : {_float_, `'pf'`}
    - If a number, the energy weighting parameter $\kappa$. If `'pf'`,
    use $\kappa=v-1$ where $v$ is the valency of the vertex.
- **normed** : _bool_
    - Controls normalization of the energies in the measure.
- **coords** : {`'ptyphim'`, `'epxpypz'`, `None`}
    - Controls which coordinates are assumed for the input. See
    [Measures](../measures) for additional info.
- **check_input** : _bool_
    - Whether to check the type of the input each time or assume the
    first input type.

### compute

```python
compute(event=None, zs=None, thetas=None, nhats=None)
```

Computes the value of the observable on a single event. Note that
the observable object is also callable, in which case this method is
invoked.

**Arguments**

- **event** : 2-d array_like or `fastjet.PseudoJet`
    - The event as an array of particles in the coordinates specified
    by `coords`.
- **zs** : 1-d array_like
    - If present, `thetas` must also be present, and `zs` is used in place
    of the energies of an event.
- **thetas** : 2-d array_like
    - If present, `zs` must also be present, and `thetas` is used in place
    of the pairwise angles of an event.
- **nhats** : 2-d array like
    - If present, `zs` must also be present, and `nhats` is used in place
    of the scaled particle momenta. Only applicable when EFMs are being
    used.

**Returns**

- _float_
    - The observable value.

### batch_compute

```python
batch_compute(events, n_jobs=None)
```

Computes the value of the observable on several events.

**Arguments**

- **events** : array_like or `fastjet.PseudoJet`
    - The events as an array of arrays of particles in coordinates
    matching those anticipated by `coords`.
- **n_jobs** : _int_ or `None`
    - The number of worker processes to use. A value of `None` will
    use as many processes as there are CPUs on the machine.

**Returns**

- _1-d numpy.ndarray_
    - A vector of the observable values for each event.

### properties

#### efpset

```python
efpset
```

`EFPSet` held by the object to compute fundamental EFP values.


----

## C3

Ratio of Energy Correlation Functions designed to tag three prong
signals. In graphs, the formula is:

<img src="https://github.com/thaler-lab/EnergyFlow/raw/images/C3.png"
class="obs_center" width="30%"/>

For additional information, see the [original paper](https://arxiv.org/
abs/1305.0007).

```python
energyflow.C3(measure='hadr', beta=2, reg=0., kappa=1, normed=True,
              coords=None, check_input=True)
```

Since a `D2` defines and holds a `Measure` instance, all `Measure`
keywords are accepted.

**Arguments**

- **measure** : {`'hadr'`, `'hadrdot'`, `'hadrefm'`, `'ee'`, `'eeefm'`}
    - The choice of measure. See [Measures](../measures) for additional
    info.
- **beta** : _float_
    - The parameter $\beta$ appearing in the measure. Must be greater
    than zero.
- **reg** : _float_
    - A regularizing value to be added to the denominator in the event
    that it is zero. Should typically be something less than 1e-30.
- **kappa** : {_float_, `'pf'`}
    - If a number, the energy weighting parameter $\kappa$. If `'pf'`,
    use $\kappa=v-1$ where $v$ is the valency of the vertex.
- **normed** : _bool_
    - Controls normalization of the energies in the measure.
- **coords** : {`'ptyphim'`, `'epxpypz'`, `None`}
    - Controls which coordinates are assumed for the input. See
    [Measures](../measures) for additional info.
- **check_input** : _bool_
    - Whether to check the type of the input each time or assume the
    first input type.

### compute

```python
compute(event=None, zs=None, thetas=None, nhats=None)
```

Computes the value of the observable on a single event. Note that
the observable object is also callable, in which case this method is
invoked.

**Arguments**

- **event** : 2-d array_like or `fastjet.PseudoJet`
    - The event as an array of particles in the coordinates specified
    by `coords`.
- **zs** : 1-d array_like
    - If present, `thetas` must also be present, and `zs` is used in place
    of the energies of an event.
- **thetas** : 2-d array_like
    - If present, `zs` must also be present, and `thetas` is used in place
    of the pairwise angles of an event.
- **nhats** : 2-d array like
    - If present, `zs` must also be present, and `nhats` is used in place
    of the scaled particle momenta. Only applicable when EFMs are being
    used.

**Returns**

- _float_
    - The observable value.

### batch_compute

```python
batch_compute(events, n_jobs=None)
```

Computes the value of the observable on several events.

**Arguments**

- **events** : array_like or `fastjet.PseudoJet`
    - The events as an array of arrays of particles in coordinates
    matching those anticipated by `coords`.
- **n_jobs** : _int_ or `None`
    - The number of worker processes to use. A value of `None` will
    use as many processes as there are CPUs on the machine.

**Returns**

- _1-d numpy.ndarray_
    - A vector of the observable values for each event.

### properties

#### efpset

```python
efpset
```

`EFPSet` held by the object to compute fundamental EFP values.


----
