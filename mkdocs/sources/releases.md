# Release Notes

## 1.4.x

- EFN/PFN architectures setup to have multiple Phi components that embed their inputs in separate latent spaces which are then concatenated together.
- Activation function layers have more descriptive names.
- PointCloudDataset and subclasses defined to aid with providing data to EFN and PFN models.
- `archs` submodule now imported by default; tensorflow/sklearn imports delayed until needed.
- Renamed the `event_utils` submodule to `random_utils` since it deals with the generation of random collections of momenta.
- Environment variable added for energyflow cache_dir.
- Renamed `path` keyword argument to `cache_dir` in the `get_examples` function.
- `arch_utils` created and imported at the top level.
- `data_utils` moved to the top level.
- Generated new default EFPs (the order has changed on Python 3.8 and higher, due either to a change in igraph or to dictionaries being ordered by default); removed availability of `npz` encoding of default EFP file.
- Removed `check_input` from Measure keyword arguments.
- EnergyFlow architecture objects are now callable, by forward `__call__` to the underlying model.
- Renamed `summary` argument of neural network models to `print_summary`.
- Added functions in `fastjet_utils` to convert to/from PseudoJet's and events in Cartesian coordinates.
- Modified remap_pids to operate on individual events
- Added float option to MODDataset.
- Added split/chain to PointCloudDataset.
- `pyfjcore` used to provide access to FastJet.
- Increased memory efficiency in `qg_jets.load`.

- Improved `data_split` function to optionally return empty arrays if `train`, `val`, or `test` is zero.

## 1.3.x

**1.3.2**

- Fixed typo in `MODDataset` code that caused `abs_gen_jet_y` and `abs_get_jet_eta` to be invalid selectors.

**1.3.1**

- Added `mass` option to `ptyphims_from_p4s` to mirror `ptyphims_from_pjs`.

**1.3.0**

- EMD module now uses Wasserstein package for optimal transport computations by default. This should yield some speed and stability improvements while being mostly transparent to the user.
- EMD Demo updated to use Wasserstein package for EMD computation and correlation dimension calculation.
- `remap_pids` now works on arrays of events (rather than arrays of padded events only.)

## 1.2.x

**1.2.0**

- Keras is now imported via Tensorflow.
- Added the ability to concatenate global features directly into the F funciton of an EFN/PFN model. See the `num_global_features` option of the EFN/PFN models.
- In list of tensors in EFN/PFN models, changed how input tensors are listed in order to produce a flattened list.
- Modified internals so that DNN and EFN/PFN models use the same code to construct fully-connected network fragments.

## 1.1.x

**1.1.3**

- Added `phi_ref` option to `ptyphims_from_pjs`.
- Simplified `pjs_from_ptyphims` to use `fastjet.PtYPhiM`.
- Simplified multiprocessing usage to avoid setting global start context when trying to use fork.
- Added EFN regression example.

**1.1.2**

- Remove extraneous warning from setting multiprocessing start method on OSX.

**1.1.1**

- Try to set multiprocessing start method to 'fork' in order for EMD multicore functionality to work, warn if the context has already been set otherwise.

**1.1.0**

- Changed default EFP behavior when `kappa≠1` and added option to revert to original behavior if desired. See the [EFP Measures](/docs/measures) page for more details.
- Explicitly cast some numpy arrays as object arrays to avoid deprecation warnings.
- Cached EFP file info after the first time it is accessed to improve speed.
- Added Python 3.8 to Travis CI testing.
- Deployment to PyPI via Travis CI.


## 1.0.x

**1.0.3**

- Increased the speed of `pjs_from_ptyphims`.
- Channels now default to the last axis for images to accomodate the limitations of newer versions of Keras/Tensorflow.
- `pixelate`, `standardize`, `zero_center` functions now designed to work with `channels_last`.
- Added EMD animation example.

**1.0.2**

- Added `ptyphims_from_pjs` function in fastjet_utils.
- Removed eroneous print statement in `D2` when using strassen.

**1.0.1**

- Added Pythia/Herwig + Delphes samples used for OmniFold unfolding study to the `datasets` submodule of EnergyFlow.
- Added `beta` option to EMD module.

**1.0.0**

- Reintroduced EFMs, with full documentation, testing, and integration with the core EFP code.
- Fixed bug in qg_nsubs where `cache_dir` was set to `None` which caused an error. Thanks to Serhii Kryhin for catching this!
- Fixed bug where `emd` module could modify the inputs in place.
- Changed the `emds` function to use global arrays in order to consume less memory when using multiple jobs.
- Updated the `eval_filters` method of the `EFN` class to adjust for new Keras behavior.
- Implemented the `D2`, `C2`, and `C3` observables using EFPs.
- Two new demos, [EFM Demo](/demos/#efm-demo) and [Counting Leafless Multigaphs with Nauty](/demos/#counting-leafless-multigaphs-with-nauty).


## 0.13.x

**0.13.2**

- Keras 2.2.5 fixes a bug in their `batch_dot` function that is used by their `Dot` layer which is used by the `EFN` and `PFN` classes. This necessitates adjusting our code to account for the new behavior.

**0.13.1**

- When loading MOD HDF5 files, jets are now made from copies of the particle arrays rather than from views, 
enabling the large arrays to be freed and only the selected jets to remain in memory (which can have substantial memory savings).

**0.13.0**

- Added support for downloading and reading MOD datasets containing CMS Open Data and Simulation from Zenodo.
- EMD module now has support for spherical measure.
- Added particle utility functions to map PDG IDs to electric charges.
- Added `sum_ptyphims` and `sum_ptyphipids` functions to sum four-vectors given in hadronic coordinates.
- Added scheme choices for summing four-vectors of particles.
- Added preprocessing functions to particle utilities, including `center_ptyphims`, 
`rotate_ptyphims` and `reflect_ptyphims`.
- A `~` is now expanded to the user's home directory properly in the `filepath` option to architectures.
- Added `h5py` install dependency for MOD Datasets.
- Improved binder environment with fastjet, latex, and default matplotlib settings.
- Added observables submodule which currently includes image_activity and zg.
- EMD module now imported when importing toplevel energyflow.


## 0.12.x

**0.12.3**

- Set allow_pickle to True explicitly in Generator (recently changed default in NumPy).
- Added Herwig7.1 dataset to `qg_jets`. A big thanks to Aditya Pathak for generating these Herwig samples!
- Quark and gluon dataset files can now be obtained from Zenodo in addition to Dropbox.
- Changed internals of EFN to use standalone functions for easier use of subnetwork components.
- Added some particle utility functions to deal with PDG IDs.
- Added particle utilities to deal with pseudorapidities.
- Changed `gen_random_events_mcom` to have positive energies (momenta still sum to zero).
- Added l2 regularization to layers in EFN and PFN architectures. Thanks to Anders Andreassen for submiting this pull request!
- Added tests for particle utils.
- Particle utilities now accept arrays of events.

**0.12.2**

- Added another periodic phi test for event EMD.
- Changed gdim default to None (to reduce potentially unexpected behavior).
- Increased numerical stability of EMD computation by including an internal change of units.
- Added verbosity functionality to EFP Generator.

**0.12.1**

- Named lambda functions inside EFNs and PFNs (necessary for saving models).
- Fixed typo in archbase code.
- Added tests for architecture code.

**0.12.0**

- Fixed potential [issue](https://github.com/keras-team/keras/issues/12495) involving the Keras `Masking` 
layer not functioning as documented. This is not expected to affect any EFN models that
were padded with zeros, nor any PFN models for which the padding was consistent across training and testing
sets. Thanks to Anders Andreassen for pointing this out!
- Added arbitrary attribute lookup in the underlying model for all EnergyFlow architectures.
- Deprecated old EFN/PFN parameter names.
- Built-in support for ModelCheckpoint and EarlyStopping callbacks for neural network models.
- Made naming of neural network layers optional, allowing pieces to be reused more easily.
- Support for periodic phi values in EMD module.
- Added support for passing arbitrary compilation options to Keras models.
- Added EMD Demo notebook


## 0.11.x

**0.11.2**

- Added advanced activations support for neural network architectures. Thanks to Kevin Bauer for this suggestion!

**0.11.1**

- Fixed issue when using Python 2 caused by not importing division in dataset loading code. Thanks to Matt LeBlanc 
for pointing this out!
- Added `n_iter_max` option to EMD functions.

**0.11.0**

- Added `emd` module to EnergyFlow. This new module is not imported by default and relies on
the [Python Optimal Transport](https://pot.readthedocs.io) library and [SciPy](http://scipy.github.io/devdocs/).
- Included binder support for the jupyter notebook demos. Thanks to Matthew Feickert for contributing this feature!


## 0.10.x

**0.10.5**

- Minor improvement and fixes. Thanks to Preksha Naik for pointing out a typo!

**0.10.4**

- Updates to the documentation and enhanced examples provided.

**0.10.3**

- Finalized initial documentation pages.
- Minor improvement and fixes.

**0.10.2**

- Minor improvement and fixes.

**0.10.1**

- Minor improvement and fixes.

**0.10.0**

- Added `archs` module containing EFN, PFN, DNN, CNN, and Linear models.


## <0.9.x

- Rapid development of EFP code.