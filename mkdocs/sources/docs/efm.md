# Energy Flow Moments

Energy Flow Moments (EFMs) are tensors that can be computed in
$\mathcal O(M)$ where $M$ is the number of particles. They are useful for many
things, including providing a fast way of computing the $\beta=2$ EFPs, which
are the scalar contractions of products of EFMs.

The expression for a (normalized) hadronic EFM in terms of transverse momenta
$\{p_{Ti}\}$ and particle momenta $\{p_i^\mu\}$ is:
$$
\mathcal I^{\mu_1\cdots\mu_v} = 2^{v/2}\sum_{i=1}^Mz_in_i^{\mu_1}\cdots n_i^{\mu_v},
$$
where
$$
z_i=\frac{p_{Ti}}{\sum_jp_{Tj}},\quad\quad n_i^\mu=\frac{p_i^\mu}{p_{Ti}}.
$$
Note that for an EFM in an $e^+e^-$ context, transverse momenta are replaced
with energies.


Support for using EFMs to compute $\beta=2$ EFPs is built in to the `EFP` and
`EFPSet` classes using the classes and functions in this module. The `EFM` and
`EFMSet` classes can also be used on their own, as can the `efp2efms` function.

----

### efp2efms

```python
energyflow.efp2efms(graph)
```

Translates an EFP formula, specified by its graph, to an expression
involving EFMs. The input is a graph as a list of edges and the output is a
tuple where the first argument is a string to be used with einsum and the
second is a list of EFM signatures (the number of raised indices followed
by the number of lowered indices).

**Arguments**

- **graph** : _list_ of _tuple_
    - The EFP graph given as a list of edges.

**Returns**

- (_str_, _list_ of _tuple_)
    - The einstring to be used with einsum for performing the contraction
    of EFMs followed by a list of the EFM specs. If `r` is the result of
    this function, and `efms` is a dictionary containing EFM tensors
    indexed by their signatures, then the value of the EFP is given as
    `np.einsum(r[0], *[efms[sig] for sig in r[1]])`.


----

### EFM

A class representing and computing a single EFM.

```python
energyflow.EFM(nup, nlow=0, measure='hadrefm', beta=2, kappa=1, normed=None, 
                            coords=None, check_input=True)
```

Since EFMs are fully symmetric tensors, they can be specified by
just two integers: the number of raised and number of lowered indices
that they carry. Thus we use a tuple of two ints as an EFM "spec" or
signature throughout EnergyFlow. By convention the raised indices come
before the lowered indices.

Since a standalone `EFM` defines and holds a `Measure` instance, all
`Measure` keywords are accepted. Note that `beta` is ignored as EFMs
require $\beta=2$.

**Arguments**

- **nup** : _int_
    - The number of uppered indices of the EFM.
- **nlow** : _int_
    - The number of lowered indices of the EFM.
- **measure** : {`'hadrefm'`, `'eeefm'`}
    - The choice of measure. See [Measures](../measures) for additional
    info. Note that EFMs can only use the `'hadrefm'` and `'eeefm'`
    measures.
- **beta** : _float_
    - The parameter $\beta$ appearing in the measure. Must be greater
    than zero.
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

#### compute

```python
compute(event=None, zs=None, nhats=None)
```

Evaluates the EFM on a single event. Note that `EFM` also is
callable, in which case this method is invoked.

**Arguments**

- **event** : 2-d array_like or `fastjet.PseudoJet`
    - The event as an array of particles in the coordinates specified
    by `coords`.
- **zs** : 1-d array_like
    - If present, `nhats` must also be present, and `zs` is used in place 
    of the energies of an event.
- **nhats** : 2-d array like
    - If present, `zs` must also be present, and `nhats` is used in place
    of the scaled particle momenta.

**Returns**

- _numpy.ndarray_ of rank `v`
    - The values of the EFM tensor on the event. The raised indices
    are the first `nup` and the lowered indices are the last `nlow`.

#### batch_compute

```python
batch_compute(events, n_jobs=None)
```

Evaluates the EFM on several events.

**Arguments**

- **events** : array_like or `fastjet.PseudoJet`
    - The events as an array of arrays of particles in coordinates
    matching those anticipated by `coords`.
- **n_jobs** : _int_ or `None`
    - The number of worker processes to use. A value of `None` will
    use as many processes as there are CPUs on the machine.

**Returns**

- _numpy.ndarray_ of rank `v+1`
    - Array of EFM tensor values on the events.

#### properties

##### nup

```python
nup
```

The number of uppered indices on the EFM.

##### nlow

```python
nlow
```

The number of lowered indices on the EFM.

##### spec

```python
spec
```

The signature of the EFM as `(nup, nlow)`.

##### v

```python
v
```

The valency, or total number of indices, of the EFM.


----

### EFMSet

A class for holding and efficiently constructing a collection of EFMs.

```python
energyflow.EFMSet(efm_specs=None, vmax=None, measure='hadrefm', beta=2, kappa=1,
                  normed=None, coords=None, check_input=True)
```

An `EFMSet` can be initialized two ways (in order of precedence):

1. **EFM Specs** - Pass in a list of EFM specs (`nup`, `nlow`).
2. **Max Valency** - Specify a maximum valency and each EFM with up to
that many indices will be constructed, with all indices raised.

Since a standalone `EFMSet` defines and holds a `Measure` instance,
all `Measure` keywords are accepted. Note that `beta` is ignored as
EFMs require $\beta=2$.

**Arguments**

- **efm_specs** : {_list_, _tuple_, _set_} of _tuple_ or `None`
    - A collection of tuples of length two specifying which EFMs this
    object is to hold. Each spec is of the form `(nup, nlow)` where these
    are the number of upper and lower indices, respectively, that the EFM 
    is to have.
- **vmax** : _int_
    - Only used if `efm_specs` is None. The maximum EFM valency to
    include in the `EFMSet`. Note that all EFMs will have `nlow=0`.
- **measure** : {`'hadrefm'`, `'eeefm'`}
    - The choice of measure. See [Measures](../measures) for additional
    info. Note that EFMs can only use the `'hadrefm'` and `'eeefm'`
    measures.
- **beta** : _float_
    - The parameter $\beta$ appearing in the measure. Must be greater
    than zero.
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

#### compute

```python
compute(event=None, zs=None, nhats=None)
```

Evaluates the EFMs held by this `EFMSet` according to the
predetermined strategy on a single event. Note that `EFMSet` also is
callable, in which case this method is invoked.

**Arguments**

- **event** : 2-d array_like or `fastjet.PseudoJet`
    - The event as an array of particles in the coordinates specified
    by `coords`.
- **zs** : 1-d array_like
    - If present, `nhats` must also be present, and `zs` is used in place 
    of the energies of an event.
- **nhats** : 2-d array like
    - If present, `zs` must also be present, and `nhats` is used in place
    of the scaled particle momenta.

**Returns**

- _dict_ of _numpy.ndarray_ of rank `v`
    - A dictionary of EFM tensors indexed by their signatures.

#### batch_compute

```python
batch_compute(events, n_jobs=None)
```

Evaluates the EFMs held by the `EFMSet` on several events.

**Arguments**

- **events** : array_like or `fastjet.PseudoJet`
    - The events as an array of arrays of particles in coordinates
    matching those anticipated by `coords`.
- **n_jobs** : _int_ or `None`
    - The number of worker processes to use. A value of `None` will
    use as many processes as there are CPUs on the machine.

**Returns**

- _numpy.ndarray_ of _dict_
    - Object array of dictionaries of EFM tensors indexed by their
    signatures.

#### properties

##### efms

```python
efms
```

A dictionary of the `EFM` objects held by this `EFMSet` where the
keys are the signatures of the EFM.

##### rules

```python
rules
```

An ordered dictionary of the construction method used for each `EFM`
where the order is the same as `sorted_efms`.


----

