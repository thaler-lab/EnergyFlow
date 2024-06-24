# Energy and Angular Measures

The appropriate notions of energy and angle depend on the collider context.
Typically, one wants to work with observables that respect the appropriate
Lorentz subgroup for the collision type of interest. EnergyFlow is capable of
handling two broad classes of measures: $e^+e^-$ and hadronic, which are
selected using the required `measure` argument. For substructure applications,
it is often convenient to normalize the energies so that $\sum_iz_i=1$. The
`normed` keyword argument is provided to control normalization of the energies
(default is `True`). Measures also deal with converting between different
representations of particle momenta, e.g. Cartesian `[E,px,py,pz]` or hadronic
`[pt,y,phi,m]`.

Each measure comes with a parameter $\beta>0$ which controls the relative
weighting between smaller and larger anglular structures. This can be set using
the `beta` keyword argument (default is `1`). when using an EFM measure, `beta`
is ignored as EFMs require $\beta=2$. There is also a $\kappa$ parameter to
control the relative weighting between soft and hard energies. This can be set
using the `kappa` keyword argument (default is `1`). Only `kappa=1` yields
collinear-safe observables.

Prior to version `1.1.0`, the interaction of the `kappa` and `normed` options
resulted in potentially unexpected behavior. As of version `1.1.0`, the flag
`kappa_normed_behavior` has been added to give the user explicit control over
the behavior when `normed=True` and `kappa!=1`. See the description of this
option below for more detailed information.

The usage of EFMs throughout the EnergyFlow package is also controlled through
the `Measure` interface. There are special measure, `'hadrefm'` and `'eeefm'`
that are used to deploy EFMs.

Beyond the measures implemented here, the user can implement their own custom
measure by passing in $\{z_i\}$ and $\{\theta_{ij}\}$ directly to the EFP
classes. Custom EFM measures can be implemented by passing in $\{z_i\}$ and
$\{\hat n_i\}$.

## Hadronic Measures

For hadronic collisions, observables are typically desired to be invariant
under boosts along the beam direction and rotations about the beam direction.
Thus, particle transverse momentum $p_T$ and rapidity-azimuth coordinates
$(y,\phi)$ are used.

There are two hadronic measures implemented in EnergyFlow that work for any
$\beta$: `'hadr'` and `'hadrdot'`. These are listed explicitly below.

`'hadr'`:

\[z_i=p_{T,i}^{\kappa},\quad\quad \theta_{ij}=(\Delta y_{ij}^2 +
\Delta\phi_{ij}^2)^{\beta/2}.\]

`'hadrdot'`:

\[z_i=p_{T,i}^{\kappa},\quad\quad \theta_{ij}=\left(\frac{2p^\mu_ip_{j\mu}}
{p_{T,i}p_{T,j}}\right)^{\beta/2}.\]

The hadronic EFM measure is `'hadrefm'`, which is equivalent to `'hadrdot'`
with $\beta=2$ when used to compute EFPs, but works with the EFM-based
implementation.

## *e+e-* Measures

For $e^+e^-$ collisions, observables are typically desired to be invariant
under the full group of rotations about the interaction point. Since the center
of momentum energy is known, the particle energy $E$ is typically used. For the
angular measure, pairwise Lorentz contractions of the normalized particle
four-momenta are used.

There is one $e^+e^-$ measure implemented that works for any $\beta$.

`'ee'`:

\[z_i = E_{i}^{\kappa},\quad\quad \theta_{ij} = \left(\frac{2p_i^\mu p_{j \mu}}
{E_i E_j}\right)^{\beta/2}.\]

The $e^+e^-$ EFM measure is `'eeefm'`, which is equivalent to `'ee'` with
$\beta=2$ when used to compute EFPs, but works with the EFM-based
implementation.

----

## Measure

Class for handling measure options, described above.

```python
energyflow.Measure(measure, beta=1, kappa=1, normed=True, coords=None,
                            check_input=True, kappa_normed_behavior='new')
```

Processes inputs according to the measure choice and other options.

**Arguments**

- **measure** : _string_
    - The string specifying the energy and angular measures to use.
- **beta** : _float_
    - The angular weighting exponent $\beta$. Must be positive.
- **kappa** : {_float_, `'pf'`}
    - If a number, the energy weighting exponent $\kappa$. If `'pf'`,
    use $\kappa=v$ where $v$ is the valency of the vertex. `'pf'`
    cannot be used with measure `'hadr'`. Only IRC-safe for `kappa=1`.
- **normed** : bool
    - Whether or not to use normalized energies/transverse momenta.
- **coords** : {`'ptyphim'`, `'epxpypz'`, `None`}
    - Controls which coordinates are assumed for the input. If
    `'ptyphim'`, the fourth column (the masses) is optional and
    massless particles are assumed if it is not present. If `None`,
    coords with be `'ptyphim'` if using a hadronic measure and
    `'epxpypz'` if using the e+e- measure.
- **check_input** : bool
    - Whether to check the type of input each time or assume the first
    input type.
- **kappa_normed_behavior** : {`'new'`, `'orig'`}
    - Determines how `'kappa'`!=1 interacts with normalization of the
    energies. A value of `'new'` will ensure that `z` is truly the
    energy fraction of a particle, so that $z_i=E_i^\kappa/\left(
    \sum_{i=1}^ME_i\right)^\kappa$. A value of `'orig'` will keep the
    behavior prior to version `1.1.0`, which used $z_i=E_i^\kappa/
    \sum_{i=1}^M E_i^\kappa$.

### evaluate

```python
evaluate(arg)
```

Evaluate the measure on a set of particles. Returns `zs`, `thetas`
if using a non-EFM measure and `zs`, `nhats` otherwise.

**Arguments**

- **arg** : _2-d numpy.ndarray_
    - A two-dimensional array of the particles with each row being a
    particle and the columns specified by the `coords` attribute.

**Returns**

- (_ 1-d numpy.ndarray_, _2-d numpy.ndarray_)
    - If using a non-EFM measure, (`zs`, `thetas`) where `zs` is a
    vector of the energy fractions for each particle and `thetas`
    is the distance matrix between the particles. If using an EFM
    measure, (`zs`, `nhats`) where `zs` is the same and `nhats` is
    the `[E,px,py,pz]` of each particle divided by its energy (if
    in an $e^+e^-$ context) or transverse momentum (if in a hadronic
    context.)


----
