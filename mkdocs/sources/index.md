# Welcome to EnergyFlow

<img src="https://github.com/pkomiske/EnergyFlow/raw/images/QG_256_plain.jpg" width="47.5%"/>
<img src="https://github.com/pkomiske/EnergyFlow/raw/images/JetCustom_29522924_24981665_EMD.jpg" width="47.5%"/>

## Features

EnergyFlow is a Python package containing a suite of particle physics tools:

- [Energy Flow Polynomials](docs/efp): EFPs are a collection of jet substructure observables which form a complete linear basis of IRC-safe observables. EnergyFlow provides tools to compute EFPs on events for several energy and angular measures as well as custom measures.
<br><br>
- [Energy Flow Networks](docs/archs): EFNs are infrared- and collinear-safe models designed for learning from collider events as unordered, variable-length sets of particles. EnergyFlow contains customizable Keras implementations of EFNs. Available from version `0.10.0` onward.
<br><br>
- [Particle Flow Networks](docs/archs): PFNs are general models designed for learning from collider events as unordered, variable-length sets of particles, based on the [Deep Sets](https://arxiv.org/abs/1703.06114) framework. EnergyFlow contains customizable Keras implementations of PFNs. Available from version `0.10.0` onward.
<br><br>
- [Energy Mover's Distance](docs/emd): The EMD is a common metric between probability distributions that has been adapted for use as a metric between collider events. EnergyFlow contains code to facilitate the computation of the EMD between events based on an underlying implementation provided by the [Python Optimal Transport (POT)](https://pot.readthedocs.io) library. Available from version `0.11.0` onward.
<br><br>
- [Energy Flow Moments](docs/efm): EFMs are moments built out of particle energies and momenta that can be evaluated in linear time in the number of particles. They provide a highly efficient means of implementing $\beta=2$ EFPs and are also very useful for reasoning about linear redundancies that appear between EFPs. Available from version `1.0.0` onward.

The EnergyFlow package also provides easy access to particle phyiscs datasets and useful supplementary features:

- [CMS Open Data in MOD HDF5 Format](docs/datasets/#cms-open-data-and-the-mod-hdf5-format): Reprocessed datasets from the [CMS Open Data](http://opendata.cern.ch/docs/cms-guide-for-research), currently including jets with $p_T>375$ GeV and associated detector-level and generator-level Monte Carlo samples. EnergyFlow provides tools for downloading, reading, and manipulating these datasets. 
<br><br>
- [Jet Tagging Datasets](docs/datasets/#quark-and-gluon-jets): Datasets of Pythia 8 and Herwig 7.1 simulated quark and gluon jets are provided, along with tools for downloading and reading them.
<br><br>
- [Additional Architectures](docs/archs): Implementations of other architectures useful for particle physics are also provided, such as convolutional neural networks (CNNs) for jet images.
<br><br>
- [Demos](demos): Jupyter notebook demos that can run in your browser (without any installation) via [Binder](https://mybinder.org).
<br><br>
- [Detailed Examples](examples): Examples showcasing EFPs, EFNs, PFNs, EMDs, and more.


The current version is `1.3.2`. Changes are summarized in the [Release Notes](releases). Using the most up-to-date version is recommended. As of version `0.7.0`, tests have been written covering the majority of the EFP and EMD code. The architectures code is currently tested by running the examples. The source code can be found on [GitHub](https://github.com/pkomiske/EnergyFlow).

Get started by [installing EnergyFlow](installation), [exploring the demos](demos), and [running the examples](examples)!


## Authors

EnergyFlow is developed and maintained by:

- [Patrick Komiske](https://pkomiske.com), primary developer
- [Eric Metodiev](https://www.ericmetodiev.com/)
- [Jesse Thaler](http://jthaler.net/)


## References

[1] P. T. Komiske, E. M. Metodiev, and J. Thaler, _Energy Flow Polynomials: A complete linear basis for jet substructure_, [JHEP __04__ (2018) 013](https://doi.org/10.1007/JHEP04(2018)013) [[1712.07124](https://arxiv.org/abs/1712.07124)].

[2] P. T. Komiske, E. M. Metodiev, and J. Thaler, _Energy Flow Networks: Deep Sets for Particle Jets_, [JHEP __01__ (2019) 121](https://doi.org/10.1007/JHEP01(2019)121) [[1810.05165](https://arxiv.org/abs/1810.05165)].

[3] P. T. Komiske, E. M. Metodiev, and J. Thaler, _The Metric Space of Collider Events_, [Phys. Rev. Lett. __123__ (2019) 041801](https://doi.org/10.1103/PhysRevLett.123.041801) [[1902.02346](https://arxiv.org/abs/1902.02346)].

[4] P. T. Komiske, R. Mastandrea, E. M. Metodiev, P. Naik, and J. Thaler, _Exploring the Space of Jets with CMS Open Data_, [Phys. Rev. D **101** (2020) 034009](https://doi.org/10.1103/PhysRevD.101.034009) [[1908.08542](https://arxiv.org/abs/1908.08542)].

[5] P. T. Komiske, E. M. Metodiev, and J. Thaler, _Cutting Multiparticle Correlators Down to Size_, [Phys. Rev. D **101** (2020) 036019](https://doi.org/10.1103/PhysRevD.101.036019) [[1911.04491](https://arxiv.org/abs/1911.04491)].

[6] A. Andreassen, P. T. Komiske, E. M. Metodiev, B. Nachman, and J. Thaler, _OmniFold: A Method to Simultaneously Unfold All Observables_, [Phys. Rev. Lett. __124__ (2020) 182001](https://doi.org/10.1103/PhysRevLett.124.182001) [[1911.09107](https://arxiv.org/abs/1911.09107)].

[7] P. T. Komiske, E. M. Metodiev, and J. Thaler, _The Hidden Geometry of Particle Collisions_, [JHEP __07__ (2020) 006](https://doi.org/10.1007/JHEP07(2020)006) [[2004.04159](https://arxiv.org/abs/2004.04159)].


## Copyright

EnergyFlow is licensed under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html). See the [LICENSE](https://github.com/pkomiske/EnergyFlow/blob/master/LICENSE) for detailed copyright information. EnergyFlow uses a customized `einsumfunc.py` from the [NumPy GitHub](https://github.com/numpy/numpy) repository as well as a few functions relating to downloading files copied from the [Keras GitHub](https://github.com/keras-team/keras) repository. The copyrights for these parts of the code are attributed to their respective owners in the [LICENSE](https://github.com/pkomiske/EnergyFlow/blob/master/LICENSE) file.