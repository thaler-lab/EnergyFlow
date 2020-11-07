# Frequently Asked EnergyFlow Questions

- [How do I cite the EnergyFlow package?](#how-do-i-cite-the-energyflow-package)
- [How do I cite the datasets available through EnergyFlow?](#how-do-i-cite-the-datasets-available-through-energyflow)
- [Why Python instead of C++?](#why-python-instead-of-c)
- [Can I contribute to the code?](#can-i-contribute-to-the-code)
- [How do I report an issue or a bug?](#how-do-i-report-an-issue)
- [Where can I get graph image files?](#where-can-i-get-graph-image-files)

---

## How do I cite the EnergyFlow package?

Please cite the relevant papers if they or this package help your research. Here are the BibTeX entries to use:

```
@article{Komiske:2017aww,
  author        = "Komiske, Patrick T. and Metodiev, Eric M. and Thaler, Jesse",
  title         = "{Energy flow polynomials: A complete linear basis for jet substructure}",
  eprint        = "1712.07124",
  archivePrefix = "arXiv",
  primaryClass  = "hep-ph",
  reportNumber  = "MIT-CTP-4965",
  doi           = "10.1007/JHEP04(2018)013",
  journal       = "JHEP",
  volume        = "04",
  pages         = "013",
  year          = "2018"
}

@article{Komiske:2018cqr,
  author        = "Komiske, Patrick T. and Metodiev, Eric M. and Thaler, Jesse",
  title         = "{Energy Flow Networks: Deep Sets for Particle Jets}",
  eprint        = "1810.05165",
  archivePrefix = "arXiv",
  primaryClass  = "hep-ph",
  reportNumber  = "MIT-CTP 5064",
  doi           = "10.1007/JHEP01(2019)121",
  journal       = "JHEP",
  volume        = "01",
  pages         = "121",
  year          = "2019"
}

@article{Komiske:2019fks,
  author        = "Komiske, Patrick T. and Metodiev, Eric M. and Thaler, Jesse",
  title         = "{Metric Space of Collider Events}",
  eprint        = "1902.02346",
  archivePrefix = "arXiv",
  primaryClass  = "hep-ph",
  reportNumber  = "MIT-CTP 5102",
  doi           = "10.1103/PhysRevLett.123.041801",
  journal       = "Phys. Rev. Lett.",
  volume        = "123",
  number        = "4",
  pages         = "041801",
  year          = "2019"
}

@article{Komiske:2019jim,
  author        = "Komiske, Patrick T. and Mastandrea, Radha and Metodiev, Eric M. and Naik, Preksha and Thaler, Jesse",
  title         = "{Exploring the Space of Jets with CMS Open Data}",
  eprint        = "1908.08542",
  archivePrefix = "arXiv",
  primaryClass  = "hep-ph",
  reportNumber  = "MIT-CTP 5129",
  doi           = "10.1103/PhysRevD.101.034009",
  journal       = "Phys. Rev. D",
  volume        = "101",
  number        = "3",
  pages         = "034009",
  year          = "2020"
}

@article{Komiske:2019asc,
  author        = "Komiske, Patrick T. and Metodiev, Eric M. and Thaler, Jesse",
  title         = "{Cutting Multiparticle Correlators Down to Size}",
  eprint        = "1911.04491",
  archivePrefix = "arXiv",
  primaryClass  = "hep-ph",
  reportNumber  = "MIT-CTP 5150",
  doi           = "10.1103/PhysRevD.101.036019",
  journal       = "Phys. Rev. D",
  volume        = "101",
  number        = "3",
  pages         = "036019",
  year          = "2020"
}

@article{Andreassen:2019cjw,
  author        = "Andreassen, Anders and Komiske, Patrick T. and Metodiev, Eric M. and Nachman, Benjamin and Thaler, Jesse",
  title         = "{OmniFold: A Method to Simultaneously Unfold All Observables}",
  eprint        = "1911.09107",
  archivePrefix = "arXiv",
  primaryClass  = "hep-ph",
  reportNumber  = "MIT-CTP 5155",
  doi           = "10.1103/PhysRevLett.124.182001",
  journal       = "Phys. Rev. Lett.",
  volume        = "124",
  number        = "18",
  pages         = "182001",
  year          = "2020"
}

@article{Komiske:2020qhg,
  author        = "Komiske, Patrick T. and Metodiev, Eric M. and Thaler, Jesse",
  title         = "{The Hidden Geometry of Particle Collisions}",
  eprint        = "2004.04159",
  archivePrefix = "arXiv",
  primaryClass  = "hep-ph",
  reportNumber  = "MIT-CTP 5185",
  doi           = "10.1007/JHEP07(2020)006",
  journal       = "JHEP",
  volume        = "07",
  pages         = "006",
  year          = "2020"
}
```

## How do I cite the datasets available through EnergyFlow?

If any of the datasets provided through EnergyFlow are used in your research, we ask that you cite their Zenodo records, which are provided below in BibTeX format (confirmed to give a sensible citation in the JHEP and RevTeX bibliography styles).

[**CMS Open Data in MOD HDF5 Format**](/docs/datasets/#cms-open-data-and-the-mod-hdf5-format)

```
@article{Zenodo:MODCMS2011A:Jets,
  author  = {Komiske, Patrick and Mastandrea, Radha and Metodiev, Eric and
             Naik, Preksha and Thaler, Jesse},
  title   = {{CMS 2011A Open Data $|$ Jet Primary Dataset $|$ pT $>$ 
              375 GeV $|$ MOD HDF5 Format}},
  journal = "Zenodo",
  year    = 2019,
  doi     = {10.5281/zenodo.3340205}
}

@article{Zenodo:MODCMS2011A:MC170,
  author  = {Komiske, Patrick and Mastandrea, Radha and Metodiev, Eric and
             Naik, Preksha and Thaler, Jesse},
  title   = {{CMS 2011A Simulation $|$ Pythia 6 QCD 170-300 $|$ pT $>$ 
              375 GeV $|$ MOD HDF5 Format}},
  journal = "Zenodo",
  year    = 2019,
  doi     = {10.5281/zenodo.3341500}
}

@article{Zenodo:MODCMS2011A:MC300,
  author  = {Komiske, Patrick and Mastandrea, Radha and Metodiev, Eric and
             Naik, Preksha and Thaler, Jesse},
  title   = {{CMS 2011A Simulation $|$ Pythia 6 QCD 300-470 $|$ pT $>$ 
              375 GeV $|$ MOD HDF5 Format}},
  journal = "Zenodo",
  year    = 2019,
  doi     = {10.5281/zenodo.3341498}
}

@article{Zenodo:MODCMS2011A:MC470,
  author  = {Komiske, Patrick and Mastandrea, Radha and Metodiev, Eric and
             Naik, Preksha and Thaler, Jesse},
  title   = {{CMS 2011A Simulation $|$ Pythia 6 QCD 470-600 $|$ pT $>$ 
              375 GeV $|$ MOD HDF5 Format}},
  journal = "Zenodo",
  year    = 2019,
  doi     = {10.5281/zenodo.3341419}
}

@article{Zenodo:MODCMS2011A:MC600,
  author  = {Komiske, Patrick and Mastandrea, Radha and Metodiev, Eric and
             Naik, Preksha and Thaler, Jesse},
  title   = {{CMS 2011A Simulation $|$ Pythia 6 QCD 600-800 $|$ pT $>$ 
              375 GeV $|$ MOD HDF5 Format}},
  journal = "Zenodo",
  year    = 2019,
  doi     = {10.5281/zenodo.3364139}
}

@article{Zenodo:MODCMS2011A:MC800,
  author  = {Komiske, Patrick and Mastandrea, Radha and Metodiev, Eric and
             Naik, Preksha and Thaler, Jesse},
  title   = {{CMS 2011A Simulation $|$ Pythia 6 QCD 800-1000 $|$ pT $>$ 
              375 GeV $|$ MOD HDF5 Format}},
  journal = "Zenodo",
  year    = 2019,
  doi     = {10.5281/zenodo.3341413}
}

@article{Zenodo:MODCMS2011A:MC1000,
  author  = {Komiske, Patrick and Mastandrea, Radha and Metodiev, Eric and
             Naik, Preksha and Thaler, Jesse},
  title   = {{CMS 2011A Simulation $|$ Pythia 6 QCD 1000-1400 $|$ pT $>$
              375 GeV $|$ MOD HDF5 Format}},
  journal = "Zenodo",
  year    = 2019,
  doi     = {10.5281/zenodo.3341502}
}

@article{Zenodo:MODCMS2011A:MC1400,
  author  = {Komiske, Patrick and Mastandrea, Radha and Metodiev, Eric and
             Naik, Preksha and Thaler, Jesse},
  title   = {{CMS 2011A Simulation $|$ Pythia 6 QCD 1400-1800 $|$ pT $>$
              375 GeV $|$ MOD HDF5 Format}},
  journal = "Zenodo",
  year    = 2019,
  doi     = {10.5281/zenodo.3341770}
}

@article{Zenodo:MODCMS2011A:MC1800,
  author  = {Komiske, Patrick and Mastandrea, Radha and Metodiev, Eric and
             Naik, Preksha and Thaler, Jesse},
  title   = {{CMS 2011A Simulation $|$ Pythia 6 QCD1800-inf $|$ pT $>$ 
              375 GeV $|$ MOD HDF5 Format}},
  journal = "Zenodo",
  year    = 2019,
  doi     = {10.5281/zenodo.3341772}
}
```

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

[**Quark and Gluon Datasets**](/docs/datasets/#quark-and-gluon-jets)

```
@article{Zenodo:EnergyFlow:Pythia8QGs,
  author  = {Komiske, Patrick and Metodiev, Eric and Thaler, Jesse},
  title   = {Pythia8 Quark and Gluon Jets for Energy Flow},
  journal = "Zenodo",
  year    = 2019,
  doi     = {10.5281/zenodo.3164691}
}

@article{Zenodo:EnergyFlow:Herwig7QGs,
  author  = {Pathak, Aditya and Komiske, Patrick and
             Metodiev, Eric and Schwartz, Matthew},
  title   = {Herwig7.1 Quark and Gluon Jets},
  journal = "Zenodo",
  year    = 2019,
  doi     = {10.5281/zenodo.3066475}
}
```

[![DOI](/img/zenodo.3164691.svg)](https://doi.org/10.5281/zenodo.3164691) - Pythia samples
<br>
[![DOI](/img/zenodo.3066475.svg)](https://doi.org/10.5281/zenodo.3066475) - Herwig samples

[**Z + Jets with Delphes Datasets**](/docs/datasets/#z-jets-with-delphes-simulation)

```
@article{Zenodo:EnergyFlow:ZJetsDelphes,
  author  = {Andreassen, Anders and Komiske, Patrick and Metodiev, Eric and
             Nachman, Benjamin and Thaler, Jesse},
  title   = {{Pythia/Herwig + Delphes Jet Datasets for OmniFold Unfolding}},
  journal = "Zenodo",
  year    = 2019,
  doi     = {10.5281/zenodo.3548091}
}
```

[![DOI](/img/zenodo.3548091.svg)](https://doi.org/10.5281/zenodo.3548091) - Pythia/Herwig + Delphes samples

## Why Python instead of C++?

Computing the EFPs requires a function such as NumPy's [einsum](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.einsum.html) that can efficiently evaluate arbitrary tensor contractions. To write such a function from scratch in C++ is difficult, and there is no obvious library in C++ to use (though if one were to attempt this the [tensor algebra compiler](http://tensor-compiler.org/) seems like a promising tool).

[NumPy](http://www.numpy.org) is a highly-optimized Python library written in C that provides all of the tools required to efficiently compute the EFPs. Libraries like NumPy take advantage of optimizations that the physicist-programmer typically does not, such as architecture-optimized libraries like BLAS or LAPACK and low-level features such as SSE/AVX instructions.

Beyond just computing EFPs, EnergyFlow makes use of other libraries written in python including [Keras](https://keras.io/) and [scikit-learn](https://scikit-learn.org/) for the architecture implementations, [POT](https://pot.readthedocs.io/en/stable/) and [SciPy](https://www.scipy.org/) for EMD computations, and [matplotlib](https://matplotlib.org/index.html) for the examples.

## Can I contribute to the code?

All of our code is open source and hosted on [GitHub](https://www.github.com/pkomiske/EnergyFlow/). We welcome additional contributors, and if you are interested in getting involved please contact us directly. Contact information is included in the relevant Energy Flow papers and our GitHub repository.

## How do I report an issue?

Please let us know of any issues you encounter as soon as possible by creating a GitHub [Issue](https://github.com/pkomiske/EnergyFlow).

## Where can I get graph image files?

Image files for all connected multigraphs with up to 7 edges in the EFP style are available as PDF files [here](https://github.com/pkomiske/EnergyFlow/tree/images/graphs). You are free to use them with the proper attribution.
