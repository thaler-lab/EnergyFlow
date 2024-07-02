# Interactive Demos

Each of the following demos are provided as Jupyter notebooks that are available on GitHub. [Binder](https://mybinder.org) provides an awesome platform for trying out these notebooks without installing anything whatsoever.

For more computationally intensive tasks, such as training EFNs/PFNs or other neural networks, check out the [Examples](../examples).

## EnergyFlow Demo

The EnergyFlow Demo provides an introduction to using EnergyFlow to compute Energy Flow Polynomials.

- [View or download](https://github.com/pkomiske/EnergyFlow/blob/master/demos/EnergyFlow%20Demo.ipynb) the notebook from GitHub
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pkomiske/EnergyFlow/master?filepath=demos/EnergyFlow%20Demo.ipynb)

## EMD Demo

The EMD Demo provides an introduction to using EnergyFlow for computing the Energy Mover's Distance (EMD) between jets.

- [View or download](https://github.com/pkomiske/EnergyFlow/blob/master/demos/EMD%20Demo.ipynb) the notebook from GitHub
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pkomiske/EnergyFlow/master?filepath=demos/EMD%20Demo.ipynb)

## MOD Jet Demo

The MOD Jet Demo provides an introduction to using CMS Open Data in the MOD HDF5 format via EnergyFlow. Jets from the [CMS 2011A Jet Primary Dataset](http://doi.org/10.7483/OPENDATA.CMS.UP77.P6PQ) have been processed into this easy-to-use format and are [available on Zenodo](https://doi.org/10.5281/zenodo.3340205) along with corresponding simulated datasets.

- [View or download](https://github.com/pkomiske/EnergyFlow/blob/master/demos/MOD%20Jet%20Demo.ipynb) the notebook from GitHub
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pkomiske/EnergyFlow/master?filepath=demos/MOD%20Jet%20Demo.ipynb)

## EFM Demo

The EFM Demo provides an introduction to Energy Flow Moments to compute EFPs and verify linear relations among them.

- [View or download](https://github.com/pkomiske/EnergyFlow/blob/master/demos/EFM%20Demo.ipynb) the notebook from GitHub
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pkomiske/EnergyFlow/master?filepath=demos/EFM%20Demo.ipynb)

## Counting Leafless Multigaphs with Nauty

Though not demonstrating features of the EnergyFlow library, this script uses the graph program [Nauty](http://pallini.di.uniroma1.it/) to count leafless multigraphs, a task which relates to EFMs as described in [1911.04491](https://arxiv.org/abs/1911.04491). To run this script, first [download Nauty](http://pallini.di.uniroma1.it/nauty26r12.tar.gz) and compile it. Then point the Python script to the Nauty installation directory. Note that running with `-d 16` has been known to take half a month.

- [View or download](https://github.com/pkomiske/EnergyFlow/blob/master/demos/count_leafless_multigraphs_nauty.py) the script from GitHub
