# Energy Flow Polynomials

Energy Flow Polynomials (EFPs) are a set of observables, indexed by
non-isomorphic multigraphs, which linearly span the space of infrared and
collinear (IRC) safe observables.

An EFP, indexed by a multigraph $G$, takes the following form:

\[\text{EFP}_G=\sum_{i_1=1}^M\cdots\sum_{i_N=1}^Mz_{i_1}\cdots z_{i_N}
\prod_{(k,\ell)\in G}\theta_{i_ki_\ell}\]

where $z_i$ is a measure of the energy of particle $i$ and $\theta_{ij}$ is a
measure of the angular separation between particles $i$ and $j$. The specific
choices for "energy" and "angular" measure depend on the collider context and
are discussed in the [Measures](../measures) section.

----

## EFP

A class for representing and computing a single EFP.

```python
energyflow.EFP(edges, measure='hadr', beta=1, kappa=1, normed=None, coords=None,
                      check_input=True, np_optimize=True)
```

Since a standalone EFP defines and holds a `Measure` instance, all
`Measure` keywords are accepted.

**Arguments**

- **edges** : _list_
    - Edges of the EFP graph specified by pairs of vertices.
- **weights** : _list_ of _int_ or `None`
    - If not `None`, the multiplicities of each edge.
- **measure** : {`'hadr'`, `'hadrdot'`, `'hadrefm'`, `'ee'`, `'eeefm'`}
    - The choice of measure. See [Measures](../measures) for additional
    info.
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
- **np_optimize** : {`True`, `False`, `'greedy'`, `'optimal'`}
    - The `optimize` keyword of `numpy.einsum_path`.

### compute

```python
compute(event=None, zs=None, thetas=None, nhats=None)
```

Computes the value of the EFP on a single event. Note that `EFP`
also is callable, in which case this method is invoked.

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
    - The EFP value.

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

#### graph

```python
graph
```

Graph of this EFP represented by a list of edges.

#### simple_graph

```python
simple_graph
```

Simple graph of this EFP (forgetting all multiedges)
represented by a list of edges.

#### weights

```python
weights
```

Edge weights (counts) for the graph of this EFP.

#### weight_set

```python
weight_set
```

Set of edge weights (counts) for the graph of this EFP.

#### einstr

```python
einstr
```

Einstein summation string for the EFP computation.

#### einpath

```python
einpath
```

NumPy einsum path specification for EFP computation.

#### efm_spec

```python
efm_spec
```

List of EFM signatures corresponding to efm_einstr.

#### efm_einstr

```python
efm_einstr
```

Einstein summation string for the EFM computation.

#### efm_einpath

```python
efm_einpath
```

NumPy einsum path specification for EFM computation.

#### efmset

```python
efmset
```

Instance of `EFMSet` help by this EFP if using EFMs.

#### np_optimize

```python
np_optimize
```

The np_optimize keyword argument that initialized this EFP instance.

#### n

```python
n
```

Number of vertices in the graph of this EFP.

#### e

```python
e
```

Number of edges in the simple graph of this EFP.

#### d

```python
d
```

Degree, or number of edges, in the graph of this EFP.

#### v

```python
v
```

Maximum valency of any vertex in the graph.

#### k

```python
k
```

Index of this EFP. Determined by EFPSet or -1 otherwise.

#### c

```python
c
```

VE complexity $\chi$ of this EFP.

#### p

```python
p
```

Number of connected components of this EFP. Note that the empty
graph conventionally has one connected component.

#### h

```python
h
```

Number of valency 1 vertices ('hanging chads) of this EFP.

#### spec

```python
spec
```

Specification array for this EFP.

#### ndk

```python
ndk
```

Tuple of `n`, `d`, and `k` values which form a unique identifier of
this EFP within an `EFPSet`.


----

## EFPSet

A class that holds a collection of EFPs and computes their values on
events. Note that all keyword arguments are stored as properties of the
`EFPSet` instance.

```python
energyflow.EFPSet(*args, filename=None, measure='hadr', beta=1, kappa=1, normed=None, 
                         coords=None, check_input=True, verbose=0)
```

`EFPSet` can be initialized in one of three ways (in order of
precedence):

1. **Graphs** - Pass in graphs as lists of edges, just as for
individual EFPs.
2. **Generator** - Pass in a custom `Generator` object as the first
positional argument.
3. **Custom File** - Pass in the name of a `.npz` file saved with a
custom `Generator`.
4. **Default** - Use the $d\le10$ EFPs that come installed with the
`EnergFlow` package.

To control which EFPs are included, `EFPSet` accepts an arbitrary
number of specifications (see [`sel`](#sel)) and only EFPs meeting each
specification are included in the set. Note that no specifications
should be passed in when initializing from explicit graphs.

Since an EFP defines and holds a `Measure` instance, all `Measure`
keywords are accepted.

**Arguments**

- ***args** : _arbitrary positional arguments_
    - Depending on the method of initialization, these can be either
    1) graphs to store, as lists of edges 2) a Generator instance
    followed by some number of valid arguments to `sel` or 3,4) valid
    arguments to `sel`. When passing in specific graphs, no arguments
    to `sel` should be given.
- **filename** : _string_
    - Path to a `.npz` file which has been saved by a valid
    `energyflow.Generator`. A value of `None` will use the provided
    graphs, if a file is needed at all.
- **measure** : {`'hadr'`, `'hadr-dot'`, `'ee'`}
    - See [Measures](../measures) for additional info.
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
- **verbose** : _int_
    - Controls printed output when initializing `EFPSet` from a file or
    `Generator`.

### compute

```python
compute(event=None, zs=None, thetas=None, nhats=None)
```

Computes the values of the stored EFPs on a single event. Note that
`EFPSet` also is callable, in which case this method is invoked.

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

- _1-d numpy.ndarray_
    - A vector of the EFP values.

### batch_compute

```python
batch_compute(events, n_jobs=None)
```

Computes the value of the stored EFPs on several events.

**Arguments**

- **events** : array_like or `fastjet.PseudoJet`
    - The events as an array of arrays of particles in coordinates
    matching those anticipated by `coords`.
- **n_jobs** : _int_ or `None`
    - The number of worker processes to use. A value of `None` will
    attempt to use as many processes as there are CPUs on the machine.

**Returns**

- _2-d numpy.ndarray_
    - An array of the EFP values for each event.

### calc_disc

```python
calc_disc(X)
```

Computes disconnected EFPs according to the internal 
specifications using the connected EFPs provided as input. Note that
this function has no effect if the `EFPSet` was initialized with
specific graphs.

**Arguments**

- **X** : _numpy.ndarray_
    - Array of connected EFPs. Rows are different events, columns are
    the different EFPs. Can handle a single event (a 1-dim array) as
    input. EFPs are assumed to be in the order expected by the instance
    of `EFPSet`; the safest way to ensure this is to use the same
    `EFPSet` to calculate both connected and disconnected EFPs. This
    function is used internally in `compute` and `batch_compute`.

**Returns**

- _numpy.ndarray_
    - A concatenated array of the connected and disconnected EFPs.

### sel

```python
sel(*args)
```

Computes a boolean mask of EFPs matching each of the
specifications provided by the `args`. 

**Arguments**

- ***args** : arbitrary positional arguments
    - Each argument can be either a string or a length-two iterable. If
    the argument is a string, it should consist of three parts: a
    character which is a valid element of `cols`, a comparison
    operator (one of `<`, `>`, `<=`, `>=`, `==`, `!=`), and a number.
    Whitespace between the parts does not matter. If the argument is a
    tuple, the first element should be a string containing a column
    header character and a comparison operator; the second element is
    the value to be compared. The tuple version is useful when the
    value is a variable that changes (such as in a list comprehension).

**Returns**

- _1-d numpy.ndarray_
    - A boolean array of length the number of EFPs stored by this object. 

### csel

```python
csel(*args)
```

Same as `sel` except using `cspecs` to select from.

### count

```python
count(*args)
```

Counts the number of EFPs meeting the specifications
of the arguments using `sel`.

**Arguments** 

- ***args** : arbitrary positional arguments
    - Valid arguments to be passed to `sel`.

**Returns**

- _int_
    - The number of EFPs meeting the specifications provided.

### graphs

```python
graphs(*args)
```

Graphs meeting provided specifications.

**Arguments** 

- ***args** : arbitrary positional arguments
    - Valid arguments to be passed to `sel`, or, if a single integer, 
    the index of a particular graph.

**Returns**

- _list_, if single integer argument is given
    - The list of edges corresponding to the specified graph
- _1-d numpy.ndarray_, otherwise
    - An array of graphs (as lists of edges) matching the
    specifications.

### simple_graphs

```python
simple_graphs(*args)
```

Simple graphs meeting provided specifications.

**Arguments** 

- ***args** : arbitrary positional arguments
    - Valid arguments to be passed to `sel`, or, if a single integer, 
    the index of particular simple graph.

**Returns**

- _list_, if single integer argument is given
    - The list of edges corresponding to the specified simple graph
- _1-d numpy.ndarray_, otherwise
    - An array of simple graphs (as lists of edges) matching the
    specifications.

### properties

#### efps

```python
efps
```

List of EFPs held by the `EFPSet`.

#### efmset

```python
efmset
```

The `EFMSet` held by the `EFPSet`, if using EFMs.

#### specs

```python
specs
```

An array of EFP specifications. Each row represents an EFP 
and the columns represent the quantities indicated by `cols`.

#### cspecs

```python
cspecs
```

Specification array for connected EFPs.

#### weight_set

```python
weight_set
```

The union of all weights needed by the EFPs stored by the 
`EFPSet`.

#### cols

```python
cols
```

Column labels for `specs`. Each EFP has a property corresponding to
each column.

- `n` : Number of vertices.
- `e` : Number of simple edges.
- `d` : Degree, or number of multiedges.
- `v` : Maximum valency (number of edges touching a vertex).
- `k` : Unique identifier within EFPs of this (n,d).
- `c` : VE complexity $\chi$.
- `p` : Number of prime factors (or connected components).
- `h` : Number of valency 1 vertices (a.k.a. 'hanging chads').


----

