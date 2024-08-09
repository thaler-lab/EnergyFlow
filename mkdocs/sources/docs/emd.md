# Energy Mover's Distance

<video width="100%" autoplay loop controls>
    <source src="https://github.com/thaler-lab/EnergyFlow/raw/images/CMS2011AJets_EventSpaceTriangulation.mp4"
            type="video/mp4">
</video>
<br>

The Energy Mover's Distance (EMD), also known as the Earth Mover's Distance, is
a metric between particle collider events introduced in [1902.02346](https://
arxiv.org/abs/1902.02346). This submodule contains convenient functions for
computing EMDs between individual events and collections of events. The core of
the computation is handled by either the [Wasserstein](https://github.com/
thaler-lab/Wasserstein) library or the [Python Optimal Transport (POT)](https://
pot.readthedocs.io) library, one of which must be installed in order to use this
submodule.

From Eqs. (1.2) and (1.3) in [2004.04159](https://arxiv.org/abs/2004.04159), the
EMD between two events is the minimum ''work'' required to rearrange one event
$\mathcal E$ into the other $\mathcal E'$ by movements of energy $f_{ij}$ from
particle $i$ in one event to particle $j$ in the other:

\[\text{EMD}_{\beta,R}(\mathcal E,\mathcal E^\prime)=\min_{\{f_{ij}\ge0\}}\sum_{i=1}^M\sum_{j=1}^{M'}f_{ij}\left(\frac{
\theta_{ij}}{R}\right)^\beta + \left|\sum_{i=1}^ME_i-\sum_{j=1}^{M'}E^\prime_j
\right|,\]

\[\sum_{j=1}^{M'}f_{ij}\le E_i, \quad \sum_{i=1}^Mf_{ij}\le E^\prime_j,
\quad\sum_{i=1}^M\sum_{j=1}^{M'}f_{ij}=E_\text{min},\]

where $E_i,E^\prime_j$ are the energies of the particles in the two events,
$\theta_{ij}$ is an angular distance between particles, and
$E_\text{min}=\min\left(\sum_{i=1}^ME_i,\,\sum_{j=1}^{M'}E^\prime_j\right)$ is
the smaller of the two total energies. In a hadronic context, transverse momenta
are used instead of energies.

----

### emd

```python
energyflow.emd.emd(ev0, ev1, dists=None, R=1.0, beta=1.0, norm=False, gdim=None, dtype='float64',
                              periodic_phi=False, mask=False, return_flow=False,
                              n_iter_max=100000,
                              epsilon_large_factor=10000.0, epsilon_small_factor=1.0)
```

Compute the EMD between two events using the Wasserstein library.

**Arguments**

- **ev0** : _numpy.ndarray_
    - The first event, given as a two-dimensional array. The event is
    assumed to be an `(M,1+gdim)` array of particles, where `M` is the
    multiplicity and `gdim` is the dimension of the ground space in
    which to compute euclidean distances between particles (as specified
    by the `gdim` keyword argument). The zeroth column is the weights of
    the particles, typically their energies or transverse momenta. For
    typical hadron collider jet applications, each particle will be of
    the form `(pT,y,phi)` where  `y` is the rapidity and `phi` is the
    azimuthal angle. If `dists` are provided, then the columns after the
    zeroth are ignored; alternatively a one-dimensional array consisting
    of just the particle weights may be passed in this case.
- **ev1** : _numpy.ndarray_
    - The other event, same format as `ev0`.
- **dists** : _numpy.ndarray_
    - A distance matrix between particles in `ev0` and `ev1`. If `None`,
    then the columns of the events after the zeroth are taken to be
    coordinates and the `gdim`-dimensional Euclidean distance is used.
- **R** : _float_
    - The R parameter in the EMD definition that controls the relative
    importance of the two terms. Must be greater than or equal to half
    of the maximum ground distance in the space in order for the EMD
    to be a valid metric satisfying the triangle inequality.
- **beta** : _float_
    - The angular weighting exponent. The internal pairwsie distance
    matrix is raised to this power prior to solving the optimal
    transport problem.
- **norm** : _bool_
    - Whether or not to normalize the particle weights to sum to one
    prior to computing the EMD.
- **gdim** : _int_
    - The dimension of the ground metric space. Useful for restricting
    which dimensions are considered part of the ground space when using
    the internal euclidean distances between particles. Has no effect if
    `dists` are provided or if `None`.
- **dtype** : {`'float64'`, `'float32'`}
    - The floating point precision to use in the computation.
- **periodic_phi** : _bool_
    - Whether to expect (and therefore properly handle) periodicity
    in the second coordinate, corresponding to the azimuthal angle
    $\phi$. Should typically be `True` for event-level applications but
    can be set to `False` (which is slightly faster) for jet
    applications where all $\phi$ differences are less than or equal to
    $\pi$ for properly processed events.
- **mask** : _bool_
    - If `True`, masks out particles farther than `R` away from the
    origin. Has no effect if `dists` are provided.
- **return_flow** : _bool_
    - Whether or not to return the flow matrix describing the optimal
    transport found during the computation of the EMD. Note that since
    the second term in Eq. 1 is implemented by including an additional
    particle in the event with lesser total weight, this will be
    reflected in the flow matrix.
- **n_iter_max** : _int_
    - Maximum number of iterations for solving the optimal transport
    problem.
- **epsilon_large_factor** : _float_
    - Controls some tolerances in the optimal transport solver. This
    value is multiplied by the floating points epsilon (around 1e-16 for
    64-bit floats) to determine the actual tolerance.
- **epsilon_small_factor** : _float_
    - Analogous to `epsilon_large_factor` but used where the numerical
    tolerance can be stricter.

**Returns**

- _float_
    - The EMD value.
- [_numpy.ndarray_], optional
    - The flow matrix found while solving for the EMD. The `(i,j)`th
    entry is the amount of `pT` that flows between particle i in `ev0`
    and particle j in `ev1`.


----

### emds

```python
energyflow.emd.emds(events0, events1=None, R=1.0, beta=1.0, norm=False, gdim=None, dtype='float64',
               #.                            pairwise_emd=None,
                                           periodic_phi=False, mask=False,
                                           external_emd_handler=None,
                                           n_jobs=-1, print_every=0, verbose=0,
                                           throw_on_error=True, n_iter_max=100000,
                                           epsilon_large_factor=10000.0,
                                           epsilon_small_factor=1.0)
```

Compute the EMDs between collections of events using the Wasserstein
 library. This can be used to compute EMDs between all pairs of events in
 a set or between events in two different sets.

 **Arguments**

 - **events0** : _list_
     - Iterable collection of events. Each event is assumed to be an
     `(M,1+gdim)` array of particles, where `M` is the multiplicity and
     `gdim` is the dimension of the ground space in which to compute
     euclidean distances between particles (as specified by the `gdim`
     keyword argument). The zeroth column is the weights of the
     particles, typically their energies or transverse momenta. For
     typical hadron collider jet applications, each particle will be of
     the form `(pT,y,phi)` where  `y` is the rapidity and `phi` is the
     azimuthal angle. If `dists` are provided, then the columns after the
     zeroth are ignored; alternatively a one-dimensional array consisting
     of just the particle weights may be passed in this case.
 - **events1** : _list_ or `None`
     - Iterable collection of events in the same format as `events0`, or
     `None`. If the latter, the pairwise distances between events in
     `events0` will be computed and the returned matrix will be
     symmetric.
- **R** : _float_
     - The R parameter in the EMD definition that controls the relative
     importance of the two terms. Must be greater than or equal to half
     of the maximum ground distance in the space in order for the EMD
     to be a valid metric satisfying the triangle inequality.
 - **norm** : _bool_
     - Whether or not to normalize the particle weights to sum to one
     prior to computing the EMD.
 - **beta** : _float_
     - The angular weighting exponent. The internal pairwsie distance
     matrix is raised to this power prior to solving the optimal
     transport problem.
  - **gdim** : _int_
     - The dimension of the ground metric space. Useful for restricting
     which dimensions are considered part of the ground space when using
     the internal euclidean distances between particles. Has no effect if
     `None`.
 - **dtype** : {`'float64'`, `'float32'`}
     - The floating point precision to use in the computation.
 - **pairwise_emd** : _wasserstein.PairwiseEMD_ or `None`
     - If not `None`, the computation uses this pairwise EMD object.
     Otherwise, one is instantiated using the provided parameters.
 - **periodic_phi** : _bool_
     - Whether to expect (and therefore properly handle) periodicity
     in the second coordinate, corresponding to the azimuthal angle
     $\phi$. Should typically be `True` for event-level applications but
     can be set to `False` (which is slightly faster) for jet
     applications where all $\phi$ differences are less than or equal to
     $\pi$ for properly processed events.
 - **mask** : _bool_
     - If `True`, ignores particles farther than `R` away from the
     origin.
 - **external_emd_handler** : _wasserstein.ExternalEMDHandler_
     - An instance of an external EMD handler from the wasserstein
     module, e.g. `CorrelationDimension`.
 - **n_jobs** : _int_ or `None`
     - The number of cpu cores to use. A value of `None` or `-1` will use
     as many threads as there are CPUs on the machine.
 - **print_every** : _int_
     - The number of computations to do in between printing the
     progress. Even if the verbosity level is zero, this still plays a
     role in determining when the worker threads report the results
     back to the main thread and check for interrupt signals.
 - **verbose** : _int_
     - Controls the verbosity level. A value greater than `0` will print
     the progress of the computation at intervals specified by
     `print_every`.
 - **throw_on_error** : _bool_
     - Whether or not to raise an exception when an issue is encountered.
     Can be useful when debugging.
 - **n_iter_max** : _int_
     - Maximum number of iterations for solving the optimal transport
     problem.
 - **epsilon_large_factor** : _float_
     - Controls some tolerances in the optimal transport solver. This
     value is multiplied by the floating points epsilon (around 1e-16 for
     64-bit floats) to determine the actual tolerance.
 - **epsilon_small_factor** : _float_
     - Analogous to `epsilon_large_factor` but used where the numerical
     tolerance can be stricter.

 **Returns**

 - _numpy.ndarray_
     - The EMD values as a two-dimensional array, except if an external
     EMD handler was provided, in which case no value is returned. If
     `events1` was `None`, then the shape will be `(len(events0),
     len(events0))` and the array will be symmetric, otherwise it will
     have shape `(len(events0), len(events1))`.



----

### emd_pot

```python
energyflow.emd.emd_pot(ev0, ev1, R=1.0, norm=False, beta=1.0, measure='euclidean', coords='hadronic',
                                 return_flow=False, gdim=None, mask=False, n_iter_max=100000,
                                 periodic_phi=False, phi_col=2, empty_policy='error')
```

Compute the EMD between two events using the Python Optimal
Transport library.

**Arguments**

- **ev0** : _numpy.ndarray_
    - The first event, given as a two-dimensional array. The event is
    assumed to be an `(M,1+gdim)` array of particles, where `M` is the
    multiplicity and `gdim` is the dimension of the ground space in
    which to compute euclidean distances between particles (as specified
    by the `gdim` keyword argument. The zeroth column is assumed to be
    the energies (or equivalently, the transverse momenta) of the
    particles. For typical hadron collider jet applications, each
    particle will be of the form `(pT,y,phi)` where  `y` is the rapidity
    and `phi` is the azimuthal angle.
- **ev1** : _numpy.ndarray_
    - The other event, same format as `ev0`.
- **R** : _float_
    - The R parameter in the EMD definition that controls the relative
    importance of the two terms. Must be greater than or equal to half
    of the maximum ground distance in the space in order for the EMD
    to be a valid metric satisfying the triangle inequality.
- **beta** : _float_
    - The angular weighting exponent. The internal pairwsie distance
    matrix is raised to this power prior to solving the optimal
    transport problem.
- **norm** : _bool_
    - Whether or not to normalize the pT values of the events prior to
    computing the EMD.
- **measure** : _str_
    - Controls which metric is used to calculate the ground distances
    between particles. `'euclidean'` uses the euclidean metric in
    however many dimensions are provided and specified by `gdim`.
    `'spherical'` uses the opening angle between particles on the
    sphere (note that this is not fully tested and should be used
    cautiously).
- **coords** : _str_
    - Only has an effect if `measure='spherical'`, in which case it
    controls if `'hadronic'` coordinates `(pT,y,phi,[m])` are expected
    versus `'cartesian'` coordinates `(E,px,py,pz)`.
- **return_flow** : _bool_
    - Whether or not to return the flow matrix describing the optimal
    transport found during the computation of the EMD. Note that since
    the second term in Eq. 1 is implemented by including an additional
    particle in the event with lesser total pT, this will be reflected
    in the flow matrix.
- **gdim** : _int_
    - The dimension of the ground metric space. Useful for restricting
    which dimensions are considered part of the ground space. Can be
    larger than the number of dimensions present in the events (in
    which case all dimensions will be included). If `None`, has no
    effect.
- **mask** : _bool_
    - If `True`, ignores particles farther than `R` away from the
    origin.
- **n_iter_max** : _int_
    - Maximum number of iterations for solving the optimal transport
    problem.
- **periodic_phi** : _bool_
    - Whether to expect (and therefore properly handle) periodicity
    in the coordinate corresponding to the azimuthal angle $\phi$.
    Should typically be `True` for event-level applications but can
    be set to `False` (which is slightly faster) for jet applications
    where all $\phi$ differences are less than or equal to $\pi$.
- **phi_col** : _int_
    - The index of the column of $\phi$ values in the event array.
- **empty_policy** : _float_ or `'error'`
    - Controls behavior if an empty event is passed in. When set to
    `'error'`, a `ValueError` is raised if an empty event is
    encountered. If set to a float, that value is returned is returned
    instead on an empty event.

**Returns**

- _float_
    - The EMD value.
- [_numpy.ndarray_], optional
    - The flow matrix found while solving for the EMD. The `(i,j)`th
    entry is the amount of `pT` that flows between particle i in `ev0`
    and particle j in `ev1`.


----

### emds_pot

```python
energyflow.emd.emds_pot(X0, X1=None, R=1.0, norm=False, beta=1.0, measure='euclidean', coords='hadronic',
                            gdim=None, mask=False, n_iter_max=100000,
                            periodic_phi=False, phi_col=2, empty_policy='error',
                            n_jobs=None, verbose=0, print_every=10**6)
```

Compute the EMDs between collections of events. This can be used to
 compute EMDs between all pairs of events in a set or between events in
 two different sets.

 **Arguments**

 - **X0** : _list_
     - Iterable collection of events. Each event is assumed to be an
     `(M,1+gdim)` array of particles, where `M` is the multiplicity and
     `gdim` is the dimension of the ground space in which to compute
     euclidean distances between particles (specified by the `gdim`
     keyword argument). The zeroth column is assumed to be the energies
     (or equivalently, the transverse momenta) of the particles. For
     typical hadron collider jet applications, each particle will be of
     the form `(pT,y,phi)` where  `y` is the rapidity and `phi` is the
     azimuthal angle.
 - **X1** : _list_ or `None`
     - Iterable collection of events in the same format as `X0`,
     or `None`. If the latter, the pairwise distances between events
     in `X0` will be computed and the returned matrix will be symmetric.
- **R** : _float_
     - The R parameter in the EMD definition that controls the relative
     importance of the two terms. Must be greater than or equal to half
     of the maximum ground distance in the space in order for the EMD
     to be a valid metric satisfying the triangle inequality.
 - **norm** : _bool_
     - Whether or not to normalize the pT values of the events prior to
     computing the EMD.
 - **beta** : _float_
     - The angular weighting exponent. The internal pairwsie distance
     matrix is raised to this power prior to solving the optimal
     transport problem.
 - **measure** : _str_
     - Controls which metric is used to calculate the ground distances
     between particles. `'euclidean'` uses the euclidean metric in
     however many dimensions are provided and specified by `gdim`.
     `'spherical'` uses the opening angle between particles on the
     sphere (note that this is not fully tested and should be used
     cautiously).
 - **coords** : _str_
     - Only has an effect if `measure='spherical'`, in which case it
     controls if `'hadronic'` coordinates `(pT,y,phi,[m])` are expected
     versus `'cartesian'` coordinates `(E,px,py,pz)`.
 - **gdim** : _int_
     - The dimension of the ground metric space. Useful for restricting
     which dimensions are considered part of the ground space. Can be
     larger than the number of dimensions present in the events (in
     which case all dimensions will be included). If `None`, has no
     effect.
 - **mask** : _bool_
     - If `True`, ignores particles farther than `R` away from the
     origin.
 - **n_iter_max** : _int_
     - Maximum number of iterations for solving the optimal transport
     problem.
 - **periodic_phi** : _bool_
     - Whether to expect (and therefore properly handle) periodicity
     in the coordinate corresponding to the azimuthal angle $\phi$.
     Should typically be `True` for event-level applications but can
     be set to `False` (which is slightly faster) for jet applications
     where all $\phi$ differences are less than or equal to $\pi$.
 - **phi_col** : _int_
     - The index of the column of $\phi$ values in the event array.
 - **empty_policy** : _float_ or `'error'`
     - Controls behavior if an empty event is passed in. When set to
     `'error'`, a `ValueError` is raised if an empty event is
     encountered. If set to a float, that value is returned is returned
     instead on an empty event.
 - **n_jobs** : _int_ or `None`
     - The number of worker processes to use. A value of `None` will use
     as many processes as there are CPUs on the machine. Note that for
     smaller numbers of events, a smaller value of `n_jobs` can be
     faster.
 - **verbose** : _int_
     - Controls the verbosity level. A value greater than `0` will print
     the progress of the computation at intervals specified by
     `print_every`.
 - **print_every** : _int_
     - The number of computations to do in between printing the
     progress. Even if the verbosity level is zero, this still plays a
     role in determining when the worker processes report the results
     back to the main process.

 **Returns**

 - _numpy.ndarray_
     - The EMD values as a two-dimensional array. If `X1` was `None`,
     then the shape will be `(len(X0), len(X0))` and the array will be
     symmetric, otherwise it will have shape `(len(X0), len(X1))`.



----
