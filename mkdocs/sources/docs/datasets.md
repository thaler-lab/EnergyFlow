# Datasets

## CMS Open Data and the MOD HDF5 Format

Starting in 2014, the CMS Collaboration began to release research-grade
recorded and simulated datasets on the [CERN Open Data Portal](http://opendata.
cern.ch/). These fantastic resources provide a unique opportunity for
researchers with diverse connections to experimental particle phyiscs world to
engage with cutting edge particle physics by developing tools and testing novel
strategies on actual LHC data. Our goal in making portions of the CMS Open
Data available in a reprocessed format is to ease as best as possible the
technical complications that have thus far been present when attempting to use
Open Data (see also [recent efforts by the CMS Collaboration](http://opendata.
cern.ch/docs/cms-releases-open-data-for-machine-learning) to make the data more
accessible).

To facilitate access to Open Data, we have developed a format utilizing the
widespread [HDF5 file format](https://www.hdfgroup.org/solutions/hdf5/) that
stores essential information for some particle physics analyses. This "MOD HDF5
Format" is currently optimized for studies based on jets, but may be updated in
the future to support other types of analyses.

To further the goals of Open Data, we have made our reprocessed samples
available on the [Zenodo platform](https://zenodo.org/). Currently, the only
"collection" of datasets that is available is `CMS2011AJets`, which was used in
[Exploring the Space of Jets with CMS Open Data](https://arxiv.org/abs/
1908.08542) for [EMD](/docs/emd)-based studies. More collections may be added
in the future as our research group completes more studies with the Open Data.

For now, this page focuses on the `CMS2011AJets` collection. This collection
includes datasets of jets that are CMS-recorded (CMS), Pythia-generated (GEN),
and detector-simulated (SIM), or in code `'cms'`, `'gen'`, `'sim'`,
respectively. The datasets include all available jets above 375 GeV, which is
where the HLT_Jet300 trigger was found to be fully efficient in both data and
simulation. Note that the pT values referenced in the name of the SIM/GEN
datasets are those of the generator-level hard parton. The DOIs of
`CMS2011AJets` MOD HDF5 datasets are:

[![DOI](/img/zenodo.3340205.svg)](https://
doi.org/10.5281/zenodo.3340205) - CMS 2011A Jets, pT > 375 GeV
<br>
[![DOI](/img/zenodo.3341500.svg)](https://
doi.org/10.5281/zenodo.3341500) - SIM/GEN QCD Jets 170-300 GeV
<br>
[![DOI](/img/zenodo.3341498.svg)](https://
doi.org/10.5281/zenodo.3341498) - SIM/GEN QCD Jets 300-470 GeV
<br>
[![DOI](/img/zenodo.3341419.svg)](https://
doi.org/10.5281/zenodo.3341419) - SIM/GEN QCD Jets 470-600 GeV
<br>
[![DOI](/img/zenodo.3364139.svg)](https://
doi.org/10.5281/zenodo.3364139) - SIM/GEN QCD Jets 600-800 GeV
<br>
[![DOI](/img/zenodo.3341413.svg)](https://
doi.org/10.5281/zenodo.3341413) - SIM/GEN QCD Jets 800-1000 GeV
<br>
[![DOI](/img/zenodo.3341502.svg)](https://
doi.org/10.5281/zenodo.3341502) - SIM/GEN QCD Jets 1000-1400 GeV
<br>
[![DOI](/img/zenodo.3341770.svg)](https://
doi.org/10.5281/zenodo.3341770) - SIM/GEN QCD Jets 1400-1800 GeV
<br>
[![DOI](/img/zenodo.3341772.svg)](https://
doi.org/10.5281/zenodo.3341772) - SIM/GEN QCD Jets 1800-$\infty$ GeV

For more details regarding the creation of these samples, as well as for the
DOIs of the original CMS Open Datasets, see [Exploring the Space of Jets with
CMS Open Data](https://arxiv.org/abs/1908.08542). To get started using the
samples, see the [MOD Jet Demo](/demos/#mod-jet-demo) which makes use of the 
[`load`](#load) function.

----

### load

```python
energyflow.mod.load(*args, amount=1, cache_dir='~/.energyflow', collection='CMS2011AJets', 
                           dataset='cms', subdatasets=None, validate_files=False,
                           store_pfcs=True, store_gens=True, verbose=0)
```

Loads samples from the specified MOD dataset. Any file that is needed
that has not been cached will be automatically downloaded from Zenodo.
Downloaded files are cached for later use. File checksums are optionally
validated to ensure dataset fidelity.

**Arguments**

- ***args** : _arbitrary positional arguments_
    - Used to specify cuts to be made to the dataset while loading; see
    the detailed description of the positional arguments accepted by
    [`MODDataset`](#moddataset).
- **amount** : _int_ or _float_
    - Approximate amount of the dataset to load. If an integer, this is the
    number of files to load (a warning is issued if more files are
    requested than are available). If a float, this is the fraction of the
    number of available files to load, rounded up to the nearest whole
    number. Note that since ints and floats are treated different, a value
    of `1` loads one file whereas `1.0` loads the entire dataset. A value
    of `-1` also loads the entire dataset. 
- **cache_dir** : _str_
    - The directory where to store/look for the files. Note that 
    `'datasets'` is automatically appended to the end of this path, as well
    as the collection name. For example, the default is to download/look
    for files in the directory `'~/.energyflow/datasets/CMS2011AJets'`.
- **collection** : _str_
    - Name of the collection of datasets to consider. Currently the only
    collection is `'CMS2011AJets'`, though more may be added in the future.
- **dataset** : _str_
    - Which dataset in the collection to load. Currently the
    `'CMS2011AJets'` collection has `'cms'`, `'sim'`, and `'gen'` datasets.
- **subdatasets** : {_tuple_, _list_} of _str_ or `None`
    - The names of subdatasets to use. A value of `None` uses all available
    subdatasets. Currently, for the `'CMS2011AJets'` collection, the
    `'cms'` dataset has one subdataset, `'CMS_Jet300_pT375-infGeV'`, the
    `'sim'` dataset has eight subdatasets arrange according to generator
    $\hat p_T$, e.g. `'SIM470_Jet300_pT375-infGeV'`, and the `'gen'`
    dataset also has eight subdatasets arranged similaraly, e.g.
    `'GEN470_pT375-infGeV'`.
- **validate_files** : _bool_
    - Whether or not to validate files according to their MD5 hashes. It
    is a good idea to set this to `True` when first downloading the files
    from Zenodo in order to ensure they downloaded properly.
- **store_pfcs** : _bool_
    - Whether or not to store PFCs if they are present in the dataset.
- **store_gens** : _bool_
    - Whether or not to store gen-level particles (referred to as
    "gens") if they are present in the dataset.
- **verbose** : _int_
    - Verbosity level to use when loading. `0` is the least verbose, `1`
    is more verbose, and `2` is the most verbose.

**Returns**

- _MODDataset_
    - A `MODDataset` object containing the selected events or jets from the
    specified collection, dataset, and subdatasets.


----

### filter_particles

```python
energyflow.mod.filter_particles(particles, which='all', pt_cut=None, chs=False,
                                           pt_i=0, pid_i=4, vertex_i=5)
```

Constructs a mask that will select particles according to specified
properties. Currently supported are selecting particles according to their
charge, removing particles associated to a pileup vertex, and implementing
a minimum particle-level pT cut

**Arguments**

- **particles** : _numpy.ndarray_
    - Two-dimensional array of particles.
- **which** : {`'all'`, `'charged'`, `'neutral'`}
    - Selects particles according to their charge.
- **pt_cut** : _float_ or `None`
    - If not `None`, the minimum transverse momentum a particle can have to
    be selected.
- **chs** : _bool_
    - Whether or not to implement charged hadron subtraction (CHS), which
    removes particles associated to a non-leading vertex (i.e. with vertex
    ids greater than or equal to 1).
- **pt_i** : _int_
    - Column index of the transverse momentum values of the particles.
- **pid_i** : _int_
    - Column index of the particle IDs (used to select by charge).
- **vertex_i** : _int_
    - Column index of the vertex IDs (used to implement CHS).

**Returns**

- _numpy.ndarray_
    - A boolean mask which selects the specified particles, i.e.
    `particles[filter_particles(particles, ...)]` will be an array of only
    those particles passing the specified cuts.


----

### kfactors

```python
energyflow.mod.kfactors(dataset, pts, npvs=None, collection='CMS2011AJets',
                                      apply_residual_correction=True)
```

Evaluates k-factors used by a particular collection. Currently, since
CMS2011AJets is the only supported collection, some of the arguments are
specific to the details of this collection (such as the use of jet pTs) and
may change in future versions of this function.

**Arguments**

- **dataset** : {`'sim'`, `'gen'`}
    - Specifies which type of k-factor to use. `'sim'` includes a reweighting
    to match the distribution of the number of primary vertices between
    the simulation dataset and the CMS data whereas `'gen'` does not.
- **pts** : _numpy.ndarray_
    - The transverse momenta of the jets, used to determine the
    pT-dependent k-factor due to using only leading order matrix elements
    in the event generation. For the CMS2011AJets collection, these are
    derived from Figure 5 of [this reference](https://doi.org/10.1016/j.
    physletb.2014.01.034).
- **npvs** : _numpy.ndarray_ of integer type or `None`
    - The number of primary vertices of a simulated event, used to
    reweight a simulated event to match the pileup distribution of data.
    Should be the same length as `pts` and correspond to the same events.
    Not used if `dataset` is `'gen'`.
- **collection** : _str_
    - Name of the collection of datasets to consider. Currently the only
    collection is `'CMS2011AJets'`, though more may be added in the future.
- **apply_residual_correction** : _bool_
    - Whether or not to apply a residual correction derived from the first
    bin of the pT spectrum that corrects for the remaining difference
    between data and simulation.

**Returns**

- _numpy.ndarray_
    - An array of k-factors corresponding to the events specified by the
    `pts` and (optionally) the `npvs` arrays. These should be multiplied
    into any existing weight for the simulated or generated event.


----

### MODDataset

Loads and provides access to datasets in MOD HDF5 format. Jets can be
selected when loading from file according to a number of kinematic
attributes. MOD HDF5 datasets are created via the [`save`](#save) method.

Currently, the MOD HDF5 format consists of an HDF5 file with the following
arrays, each of which are stored as properties of the `MODDataset`:

- `/jets_i` - _int64_
    - An array of integer jet attributes, which are currently:
        - `fn` : The file number of the jet, used to index the
        [`filenames`](#filenames) array.
        - `rn` : The run number of the jet.
        - `lbn` : The lumiblock number (or lumisection) of the jet.
        - `evn` : The event number of the jet.
        - `npv` (CMS/SIM only) : The number of primary vertices of the
        event containing the jet.
        - `quality` (CMS/SIM only) : The quality of the jet, where `0`
        means no quality, `1` is "loose", `2` is "medium", and `3` is
        "tight".
        - `hard_pid` (SIM/GEN only) : The particle ID of the hard parton
        associated to the jet (`0` if not associated).
- `/jets_f` - _float64_
    - An array of floating point jet attributes, which are currently:
        - `jet_pt` : Transverse momentum of the jet.
        - `jet_y` : Rapidity of the jet.
        - `jet_phi` : Azimuthal angle of the jet.
        - `jet_m` : Mass of the jet.
        - `jet_eta` : Pseudorapidity of the jet.
        - `jec` (CMS/SIM only) : Jet energy correction.
        - `jet_area` (CMS/SIM only) : Area of the jet.
        - `jet_max_nef` (CMS/SIM only) : Maximum of the hadronic and
        electromagnetic energy fractions of the jet.
        - `gen_jet_pt` (SIM only) : Transverse momentum of an associated
        GEN jet. `-1` if not associated.
        - `gen_jet_y` (SIM only) : Rapidity of an associated GEN jet. `-1`
        if not associated.
        - `gen_jet_phi` (SIM only) : Azimuthal angle of an associated GEN
        jet. `-1` if not associated.
        - `gen_jet_m` (SIM only) : Mass of an associated GEN jet. `-1` if
        not associated.
        - `gen_jet_eta` (SIM only) : Pseudorapidity of an associated GEN
        jet. `-1` if not associated.
        - `hard_pt` (SIM/GEN only) : Transverse momentum of an associated
        hard parton. `-1` if not associated.
        - `hard_y` (SIM/GEN only) : Rapidity of an associated hard parton.
        `-1` if not associated.
        - `hard_phi` (SIM/GEN only) : Azimuthal angle of an associated hard
        parton. `-1` if not associated.
        - `weight` :  Contribution of this jet to the cross-section, in
        nanobarns.
- `/pfcs` - _float64_ (CMS/SIM only)
    - An array of all particle flow candidates, with attributes listed
    below. There is a separate `/pfcs_index` array in the file which
    contains information for `MODDataset` to separate these particles into
    separate jets. The columns of the array are currently:
        - `pt` : Transverse momentum of the PFC.
        - `y` : Rapidity of the PFC.
        - `phi` : Azimuthal angle of the PFC.
        - `m` : Mass of the PFC.
        - `pid` : PDG ID of the PFC.
        - `vertex` : Vertex ID of the PFC. `0` is leading vertex, `>0` is
        a pileup vertex, and `-1` is unknown. Neutral particles are
        assigned to the leading vertex.
- `/gens` - _float64_ (SIM/GEN only)
    - An array of all generator-level particles, currently with the same 
    columns as the `pfcs` array (the vertex column contains all `0`s). For
    the SIM dataset, these are the particles of jets associated to the SIM
    jets which are described in the `jets_i` and `jets_f` arrays. As with
    `pfcs`, there is a separate `gens_index` array which tells `MODDataset`
    how to separate these gen particles into distinct jets.
- `/filenames` - _str_
    - An array of strings indexed by the `fn` attribute of each jet. For
    CMS, this array is one dimensional and contains the CMS-provided
    filenames. For SIM/GEN, this array is two dimensional where the first
    column is the pT value that appears in the name of the dataset and the
    second column is the CMS-provided filename. In all cases, indexing this
    array with the `fn` attribute of a jet gives the file information in
    which the event containing that jet is to be found.

Note that the column names of the `jets_i`, `jets_f`, `pfcs`, and `gens`
arrays are stored as lists of strings in the attributes `jets_i_cols`,
`jets_f_cols`, `pfcs_cols`, and `gens_cols`.

For each of the above arrays, `MODDataset` stores the index of the column
as an attribute with the same name as the column. For example, for an
instance called `modds`, `modds.fn` has a value of `0` since it is the
first column in the `jets_i` array, `modds.jet_phi` has a value of `2`,
`modds.m` has a value of `3`, etc.

Even more helpfully, a view of each column of the jets arrays is stored
as an attribute as well, so that `modds.jet_pts` is the same as
`modds.jets_f[:,modds.jet_pt]`, `modds.evns` is the same as 
`modds.jets_i[:,modds.evn]`, etc. Additionally, one special view is stored,
`corr_jet_pts`, which is equal to the product of the jet pTs and the JECs,
i.e. `modds.jet_pts*modds.jecs`.

`MODDataset` supports the builtin `len()` method, which returns the number
of jets currently stored in the dataset, as well as the `print()` method,
which prints a summary of the dataset.

```python
energyflow.mod.MODDataset(*args, datasets=None, path=None, num=-1, shuffle=True, 
                                 store_pfcs=True, store_gens=True)
```

`MODDataset` can be initialized from a MOD HDF5 file or from a list
of existing `MODDataset`s. In the first case, the filename should be
given as the first positional argument. In the second case, the 
`datasets` keyword argument should be set to a list of `MODDataset`
objects.

**Arguments**

- ***args** : _arbitrary positional arguments_
    - Each argument specifies a requirement for an event to be selected
    and kept in the `MODDataset`. All requirements are ANDed together.
    Each specification can be a string or a tuple/list of objects that
    will be converted to strings and concatenated together. Each string
    specifies the name of one of the columns of one of the jets arrays
    (`'corr_jet_pts'` is also accepted, see above, as well as
    `'abs_jet_eta'`, `'abs_gen_jet_y'`, etc, which use the absolute
    values of the [pseudo]rapidities of the jets) as well as one or
    more comparisons to be performed using the values of that column
    and the given values in the string. For example,
    `('corr_jet_pts >', 400)`, which is the same as
    `'corr_jet_pts>400'`, will select jets with a corrected pT above
    400 GeV. Ampersands may be used within one string to indicated
    multiple requirements, e.g. `'corr_jet_pts > 400 & abs_jet_eta'`,
    which has the same effect as using multiple arguements each with a
    single requirement.
- **datasets** : {_tuple_, _list_} of `MODDataset` instances or `None`
    - `MODDataset`s from which to initialize this dataset. Effectively
    what this does is to concatenate the arrays held by the datasets.
    Should always be `None` when initializing from an existing file.
- **path** : _str_ or `None`
    - If not `None`, then `path` is prepended to the filename when
    initializing from file. Has no effect when initializing from
    existing datasets.
- **num** : _int_
    - The number of events or jets to keep after subselections are
    applied. A value of `-1` keeps the entire dataset. The weights
    are properly rescaled to preserve the total cross section of the
    selection.
- **shuffle** : _bool_
    - When subselecting a fraction of the dataset (i.e. `num!=-1`),
    if `False` the first `num` events passing cuts will be kept, if
    `True` then a random subset of `num` events will be kept. Note that
    this has no effect when `num` is `-1`, and also that this flag only
    affects which events are selected and does not randomize the order
    of the events that are ultimately stored by the `MODDataset` object.
- **store_pfcs** : _bool_
    - Whether or not to store PFCs if they are present in the dataset.
- **store_gens** : _bool_
    - Whether or not to store gen-level particles (referred to as
    "gens") if they are present in the dataset.

#### sel

```python
sel(args, kwargs)
```

Returns a boolean mask that selects jets according to the specified
requirements.

**Arguments**

- ***args** : _arbitrary positional arguments_
    - Used to specify cuts to be made to the dataset while loading; see
    the detailed description of the positional arguments accepted by
    [`MODDataset`](#moddataset).

**Returns**

- _numpy.ndarray_ of type _bool_
    - A boolean mask that will select jets that pass all of the
    specified requirements.

#### apply_mask

```python
apply_mask(mask, preserve_total_weight=False)
```

Subselects jets held by the `MODDataset` according to a boolean
mask.

**Arguments**

- **mask** : _numpy.ndarray_ or type _bool_
    - A boolean mask used to select which jets are to be kept. Should
    be the same length as the `MODDataset` object.
- **preserve_total_weight** : _bool_
    - Whether or not to keep the cross section of the `MODDataset`
    fixed after the selection.

#### save

```python
save(filepath, npf=-1, compression=None, verbose=1, n_jobs=1)
```

Saves a `MODDataset` in the MOD HDF5 format.

**Arguments**

- **filepath** : _str_
    - The filepath (with or without the `'.h5'` suffix) where the saved
    file will be located.
- **npf** : _int_
    - The number of jets per file. If not `-1`, multiple files will be
    created with `npf` jets as the maximum number stored in each file,
    in which case `'_INDEX'`, where `INDEX` is the index of that file,
    will be appended to the filename.
- **compression** : _int_ or `None`
    - If not `None`, the gzip compression level to use when saving the
    arrays in the HDF5 file. If not `None`, `'_compressed'` will be
    appended to the end of the filename.
- **verbose** : _int_
    - Verbosity level to use when saving the files.
- **n_jobs** : _int_
    - The number of processes to use when saving the files; only
    relevant when `npf!=-1`.

#### close

```python
close()
```

Closes the underlying HDF5 file, if one is associated with the
`MODDataset` object. Note that associated HDF5 files are closed by
default when the `MODDataset` object is deleted.

#### properties

##### jets_i

```python
jets_i
```

The `jets_i` array, described under [`MODDataset`](#moddataset).

##### jets_f

```python
jets_f
```

The `jets_f` array, described under [`MODDataset`](#moddataset).

##### pfcs

```python
pfcs
```

The `pfcs` array, described under [`MODDataset`](#moddataset).

##### gens

```python
gens
```

The `gens` array, described under [`MODDataset`](#moddataset).

##### particles

```python
particles
```

If this is a CMS or SIM dataset, `particles` is the same as `pfcs`;
for GEN it is the same as `gens`.

##### filenames

```python
filenames
```

The `filenames` array, described under [`MODDataset`](#moddataset).

##### hf

```python
hf
```

The underlying HDF5 file, if one is associated to the `MODDataset`.


----

## Z + Jets with Delphes Simulation

Datasets of QCD jets used for studying unfolding in [OmniFold: A Method to
Simultaneously Unfold All Observables](https://arxiv.org/abs/1911.09107). Four
different datasets are present:

- Herwig 7.1.5 with the default tune
- Pythia 8.243 with tune 21 (ATLAS A14 central tune with NNPDF2.3LO)
- Pythia 8.243 with tune 25 (ATLAS A14 variation 2+ of tune 21)
- Pythia 8.243 with tune 26 (ATLAS A14 variation 2- of tune 21)

$Z$ + jet events (with the $Z$ set to be stable) were generated for each of the
above generator/tune pairs with the $Z$ boson $\hat p_T^\text{min} > 150$ GeV
and $\sqrt{s}=14$ TeV. Events were then passed through the Delphes 3.4.2 fast
detector simulation of the CMS detector. Jets with radius parameter $R=0.4$
were found with the anti-kt algorithm at both particle level ("gen"), where all
non-neutrino, non-$Z$ particle are used, and detector level ("sim"), where
reconstructed energy flow objects (tracks, electromagnetic calorimeter cells,
and hadronic calorimeter cells) are used. Only jets with transverse momentum
greater than  are kept (note that sim jets have a simple jet energy correction
applied by Delphes). The hardest jet from events with a $Z$ boson with a final
transverse momentum of 200 GeV or greater are kept, yielding approximately 1.6
million jets at both gen-level and sim-level for each generator/tune pair.

Each zipped NumPy file consists of several arrays, the names of which begin
with either 'gen_' or 'sim_' depending on which set of jets they correspond
to. The name of each array ends in a key word indicating what it contains. With
the exception of 'gen_Zs' (which contains the $(p_T,\,y,\,\phi)$ of the final
$Z$ boson), there is both a gen and sim version of each array. The included
arrays are (listed by their key words):

- `'jets'` - The jet axis four vector, as $(p_T^\text{jet},\,y^\text{jet},\,
\phi^\text{jet},\,m^\text{jet})$ where $y^\text{jet}$ is the jet rapidity,
$\phi^\text{jet}$ is the jet azimuthal angle, and $m^\text{jet}$ is the jet
mass.
- `'particles'` - The rescaled and translated constituents of the jets as $(p_T/100,\,y-y^\text{jet},\,\phi-\phi^\text{jet},\,f_\text{PID})$ where
$f_\text{PID}$ is a small float corresponding to the PDG ID of the particle.
The PIDs are remapped according to $22\to0.0$, $211\to0.1$, $-211\to0.2$, $130
\to0.3$, $11\to0.4$, $-11\to0.5$, $13\to0.6$, $-13\to0.7$, $321\to0.8$, $-321
\to0.9$, $2212\to1.0$, $-2212\to1.1$, $2112\to1.2$, $-2112\to1.3$. Note that
ECAL cells are treated as photons (id 22) and HCAL cells are treated as 
$K_L^0$ (id 130). 
- `'mults'` - The constituent multiplicity of the jet.
- `'lhas'` - The Les Houches ($\beta=1/2$) angularity.
- `'widths'` - The jet width ($\beta=1 angularity).
- `'ang2s'` - The $\beta=2$ angularity (note that this is very similar to the jet
mass, but does not depend on particle masses).
- `'tau2s'` - The 2-subjettiness value with $\beta=1$.
- `'sdms'` - The groomed mass with Soft Drop parameters $z_\text{cut}=0.1$ and
$\beta=0$.
- `'zgs'` - The groomed momentum fraction (same Soft Drop parameters as above).

If you use this dataset, please cite the Zenodo record as well as the
corresponding paper:

[![DOI](/img/zenodo.3548091.svg)](https://
doi.org/10.5281/zenodo.3548091) - Pythia/Herwig + Delphes samples

- A. Andreassen, P. T. Komiske, E. M. Metodiev, B. Nachman, J. Thaler, 
OmniFold: A Method to Simultaneously Unfold All Observables, [arXiv:1911.09107](https://arxiv.org/abs/1911.09107).

----

#### load

```python
energyflow.zjets_delphes.load(dataset, num_data=100000, pad=False, cache_dir='~/.energyflow',
                                       source='zenodo', which='all',
                                       include_keys=None, exclude_keys=None)
```

Loads in the Z+jet Pythia/Herwig + Delphes datasets. Any file that is
needed that has not been cached will be automatically downloaded.
Downloaded files are cached for later use. Checksum verification is
performed.

**Arguments**

- **datasets**: {`'Herwig'`, `'Pythia21'`, `'Pythia25'`, `'Pythia26'`}
    - The dataset (specified by which generator/tune was used to produce
    it) to load. Note that this argument is not sensitive to
    capitalization.
- **num_data**: _int_
    - The number of events to read in. A value of `-1` means to load all
    available events.
- **pad**: _bool_
    - Whether to pad the particles with zeros in order to form contiguous
    arrays.
- **cache_dir**: _str_
    - Path to the directory where the dataset files should be stored.
- **source**: {`'dropbox'`, `'zenodo'`}
    - Which location to obtain the files from.
- **which**: {`'gen'`, `'sim'`, `'all'`}
    - Which type(s) of events to read in. Each dataset has corresponding
    generated events at particle-level and simulated events at
    detector-level.
- **include_keys**: _list_ or_tuple_ of _str_, or `None`
    - If not `None`, load these keys from the dataset files. A value of
    `None` uses all available keys (the `KEYS` global variable of this
    module contains the available keys as keys of the dictionary and
    brief descriptions as values). Note that keys do not have 'sim' or
    'gen' prepended to the front yet.
- **exclude_keys**: _list_ or _tuple_ or _str_, or `None`
    - Any keys to exclude from loading. Most useful when a small number of
    keys should be excluded from the default set of keys.

**Returns**

- _dict_ of _numpy.ndarray_
    - A dictionary of the jet, particle, and observable arrays for the
    specified dataset.


----

## Quark and Gluon Jets

Four datasets of quark and gluon jets, each having two million total jets, have
been generated with [Pythia](http://home.thep.lu.se/~torbjorn/Pythia.html) and
[Herwig](https://herwig.hepforge.org/) and are accessible through this
submodule of EnergyFlow. The four datasets are:

- Pythia 8.226 quark (uds) and gluon jets.
- Pythia 8.235 quark (udscb) and gluon jets.
- Herwig 7.1.4 quark (uds) and gluon jets.
- Herwig 7.1.4 quark (udscb) and gluon jets

To avoid downloading unnecessary samples, the datasets are contained in twenty
files with 100k jets each, and only the required files are downloaded. These
are based on the samples used in 
[1810.05165](https://arxiv.org/abs/1810.05165). Splitting the data into 
1.6M/200k/200k train/validation/test sets is recommended for standardized
comparisons.

Each dataset consists of two components:

- `X` : a three-dimensional numpy array of the jets with shape 
`(num_data,max_num_particles,4)`.
- `y` : a numpy array of quark/gluon jet labels (quark=`1` and gluon=`0`).

The jets are padded with zero-particles in order to make a contiguous array.
The particles are given as `(pt,y,phi,pid)` values, where `pid` is the
particle's [PDG id](http://pdg.lbl.gov/2018/reviews/rpp2018-rev-monte
-carlo-numbering.pdf). Quark jets either include or exclude $c$ and $b$
quarks depending on the `with_bc` argument.

The samples are generated from $q\bar q\to Z(\to\nu\bar\nu)+g$ and
$qg\to Z(\to\nu\bar\nu)+(uds[cb])$ processes in $pp$ collisions at
$\sqrt{s}=14$ TeV. Hadronization and multiple parton interactions (i.e.
underlying event) are turned on and the default tunings and shower parameters
are used. Final state non-neutrino particles are clustered into $R=0.4$
anti-$k_T$ jets using FastJet 3.3.0. Jets with transverse momentum
$p_T\in[500,550]$ GeV and rapidity $|y|<1.7$ are kept. Particles are ensured
have to $\phi$ values within $\pi$ of the jet (i.e. no $\phi$-periodicity 
issues). No detector simulation is performed.

The samples are also hosted on Zenodo and we ask that you cite them
appropriately if they are useful to your research. For BibTex entries,
see the [FAQs](/faqs/#how-do-i-cite-the-energyflow-package).

[![DOI](/img/zenodo.3164691.svg)](https://doi.org/10.5281/zenodo.3164691) - Pythia samples
<br>
[![DOI](/img/zenodo.3066475.svg)](https://doi.org/10.5281/zenodo.3066475) - Herwig samples

----

#### load

```python
energyflow.qg_jets.load(num_data=100000, pad=True, ncol=4, generator='pythia',
                        with_bc=False, cache_dir='~/.energyflow')
```

Loads samples from the dataset (which in total is contained in twenty 
files). Any file that is needed that has not been cached will be 
automatically downloaded. Downloading a file causes it to be cached for
later use. Basic checksums are performed.

**Arguments**

- **num_data** : _int_
    - The number of events to return. A value of `-1` means read in all
    events.
- **pad** : _bool_
    - Whether to pad the events with zeros to make them the same length.
    Note that if set to `False`, the returned `X` array will be an object
    array and not a 3-d array of floats.
- **ncol** : _int_
    - Number of columns to keep in each event.
- **generator** : _str_
    - Specifies which Monte Carlo generator the events should come from.
    Currently, the options are `'pythia'` and `'herwig'`.
- **with_bc** : _bool_
    - Whether to include jets coming from bottom or charm quarks. Changing
    this flag does not mask out these jets but rather accesses an entirely
    different dataset. The datasets with and without b and c quarks should
    not be combined.
- **cache_dir** : _str_
    - The directory where to store/look for the files. Note that 
    `'datasets'` is automatically appended to the end of this path.

**Returns**

- _3-d numpy.ndarray_, _1-d numpy.ndarray_
    - The `X` and `y` components of the dataset as specified above. If
    `pad` is `False` then these will be object arrays holding the events,
    each of which is a 2-d ndarray.


----

## Quark and Gluon Nsubs

A dataset consisting of 45 $N$-subjettiness observables for 100k quark and 
gluon jets generated with Pythia 8.230. Following [1704.08249](https:
//arxiv.org/abs/1704.08249), the observables are in the following order:

$$
\{\tau_1^{(\beta=0.5)},\tau_1^{(\beta=1.0)},\tau_1^{(\beta=2.0)},
\tau_2^{(\beta=0.5)},\tau_2^{(\beta=1.0)},\tau_2^{(\beta=2.0)},
\ldots,
\tau_{15}^{(\beta=0.5)},\tau_{15}^{(\beta=1.0)},\tau_{15}^{(\beta=2.0)}\}.
$$

The dataset contains two members: `'X'` which is a numpy array of the nsubs
that has shape `(100000,45)` and `'y'` which is a numpy array of quark/gluon 
labels (quark=`1` and gluon=`0`).

----

#### load

```python
energyflow.qg_nsubs.load(num_data=-1, cache_dir='~/.energyflow')
```

Loads the dataset. The first time this is called, it will automatically
download the dataset. Future calls will attempt to use the cached dataset 
prior to redownloading.

**Arguments**

- **num_data** : _int_
    - The number of events to return. A value of `-1` means read in all events.
- **cache_dir** : _str_
    - The directory where to store/look for the file.

**Returns**

- _3-d numpy.ndarray_, _1-d numpy.ndarray_
    - The `X` and `y` components of the dataset as specified above.


----

