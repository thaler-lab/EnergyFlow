# Multigraph Generation

Implementation of EFP/EFM Generator class.

----

## Generator

Generates non-isomorphic multigraphs according to provided specifications.

```python
energyflow.Generator(dmax=None, nmax=None, emax=None, cmax=None, vmax=None, comp_dmaxs=None,
                     filename=None, gen_efms=True, np_optimize='greedy', verbose=False)
```

Doing a fresh generation of connected multigraphs (`filename=None`)
requires that `igraph` be installed.

**Arguments**

- **dmax** : _int_
    - The maximum number of edges of the generated connected graphs.
- **nmax** : _int_
    - The maximum number of vertices of the generated connected graphs.
- **emax** : _int_
    - The maximum number of edges of the generated connected simple
    graphs.
- **cmax** : _int_
    - The maximum VE complexity $\chi$ of the generated connected
    graphs.
- **vmax** : _int_
    - The maximum valency of the generated connected graphs.
- **comp_dmaxs** : {_dict_, _int_}
    - If an integer, the maximum number of edges of the generated
    disconnected graphs. If a dictionary, the keys are numbers of
    vertices and the values are the maximum number of edges of the
    generated disconnected graphs with that number of vertices.
- **filename** : _str_
    - If `None`, do a complete generation from scratch. If set to a
    string, read in connected graphs from the file given, restrict them
    according to the various 'max' parameters, and do a fresh
    disconnected generation. The special value `filename='default'`
    means to read in graphs from the default file. This is useful when
    various disconnected graph parameters are to be varied since the
    generation of large simple graphs is the most computationlly
    intensive part.
- **gen_efms** : _bool_
    - Controls whether EFM information is generated.
- **np_optimize** : {`True`, `False`, `'greedy'`, `'optimal'`}
    - The `optimize` keyword of `numpy.einsum_path`.
- **verbose** : _bool_
    - A flag to control printing.

### save

```python
save(filename, protocol='npz', compression=True)
```

Save the current generator to file.

**Arguments**

- **filename** : _str_
    - The path to save the file.
- **protocol** : {`'npz'`, `'json'`}
    - The file format to be used.
- **compression** : _bool_
    - Whether to compress the resulting file or not.R

### properties

#### specs

```python
specs
```

An array of EFP specifications. Each row represents an EFP 
and the columns represent the quantities indicated by `cols`.


----

