# Installation

The EnergyFlow package is written in pure Python and the core depends only on `NumPy`, the fundamental package for scientific computing with Python, and `h5py`, which is used to interface with HDF5 files. The extra features require additional packages as specified below:

- Architectures: [Tensorflow](https://www.tensorflow.org), [scikit-learn](https://scikit-learn.org).
- EMD: [Wasserstein](https://thaler-lab.github.io/Wasserstein) or [POT](https://pythonot.github.io/)
- Multigraph Generation: [iGraph](http://igraph.org/redirect.html)

The EnergyFlow package is designed to work with Python 3.6 and higher, though it may work with previous versions as well, including Python 2.7. These can be installed from [here](https://www.python.org/downloads/). A recent version of Python 3 is highly recommended, ideally 3.6 or higher.

## Install via `pip`

To install from PyPI using `pip`, make sure you have one of the supported versions of Python installed and that `pip` is available in the system path. Simply execute `pip install energyflow` and EnergyFlow will be installed in your default location for Python packages.

## NumPy

As of version `0.8.2`, EnergyFlow has used a modified version of `numpy.einsum` to do the heavy lifting for the computation of EFPs. NumPy 1.14.0 changed `einsum` to use `tensordot` when possible compared to `1.13.3`, which only used `c_einsum`. It was found that the multi-process approach used by [`batch_compute`](../docs/efp/#batch_compute_1) is typically much faster when using the inherently single-threaded `c_einsum` versus `tensordot`, which can call BLAS. Hence the custom version of `einsum` shipped with EnergyFlow disables all calls to `tensordot` and uses only `c_einsum`.
