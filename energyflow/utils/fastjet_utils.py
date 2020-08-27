"""## FastJet Tools

The [FastJet package](http://fastjet.fr/) provides, among other things, fast
jet clustering utilities. It is written in C++ and includes a Python interface
that is easily installed at compile time by passing the `--enable-pyext` flag
to `configure`. If you use this module for published research, please [cite
FastJet appropriately](http://fastjet.fr/about.html).

The core of EnergyFlow does not rely on FastJet, and hence it is not required
to be installed, but the following utilities are available assuming that
`import fastjet` succeeds in your Python environment (if not, no warnings or
errors will be issued but this module will not be usable).
"""
from __future__ import absolute_import, division, print_function

import numpy as np

from energyflow.utils.generic_utils import import_fastjet

fj = import_fastjet()

__all__ = []

if fj:

    __all__ = ['pjs_from_ptyphims', 'ptyphims_from_pjs', 'cluster', 'softdrop']

    def pjs_from_ptyphims(ptyphims):
        """Converts particles in hadronic coordinates to FastJet PseudoJets.

        **Arguments**

        - **ptyphims** : _2d numpy.ndarray_
            - An array of particles in hadronic coordinates. The mass is
            optional and will be taken to be zero if not present.

        **Returns**

        - _list_ of _fastjet.PseudoJet_
            - A list of PseudoJets corresponding to the particles in the given
            array.
        """

        pjs = [fj.PseudoJet() for i in range(len(ptyphims))]
        for pj,ptyphim in zip(pjs, ptyphims):
            if len(ptyphim) >= 4:
                pj.reset_PtYPhiM(ptyphim[0], ptyphim[1], ptyphim[2], ptyphim[3])
            else:
                pj.reset_PtYPhiM(ptyphim[0], ptyphim[1], ptyphim[2])

        return pjs

    def ptyphims_from_pjs(pjs, mass=True):
        """Extracts hadronic four-vectors from FastJet PseudoJets.

        **Arguments**

        - **pjs** : _list_ of _fastjet.PseudoJet_
            - An iterable of PseudoJets.
        - **mass** : _bool_
            - Whether or not to include the mass in the extracted four-vectors.

        **Returns**

        - _numpy.ndarray_
            - An array of four-vectors corresponding to the given PseudoJets as
            `[pT, y, phi, m]`, where the mass is optional.
        """

        if mass:
            return np.asarray([[pj.pt(), pj.rap(), pj.phi(), pj.m()] for pj in pjs])
        else:
            return np.asarray([[pj.pt(), pj.rap(), pj.phi()] for pj in pjs])

    def cluster(pjs, algorithm='ca', R=fj.JetDefinition.max_allowable_R):
        """Clusters a list of PseudoJets according to a specified jet
        algorithm and jet radius.

        **Arguments**

        - **pjs** : _list_ of _fastjet.PseudoJet_
            - A list of Pseudojets representing particles or other kinematic
            objects that are to be clustered into jets.
        - **algorithm** : {'kt', 'antikt', 'ca', 'cambridge', 'cambridge_aachen'}
            - The jet algorithm to use during the clustering. Note that the
            last three options all refer to the same strategy and are provided
            because they are all used by the FastJet Python package.
        - **R** : _float_
            - The jet radius. The default value corresponds to
            `max_allowable_R` as defined by the FastJet python package.

        **Returns**

        - _list_ of _fastjet.PseudoJet_
            - A list of PseudoJets corresponding to the clustered jets.
        """

        algorithm_l = algorithm.lower()
        if algorithm_l  == 'kt':
            jet_alg = fj.kt_algorithm
        elif algorithm_l == 'antikt' or algorithm_l == 'akt':
            jet_alg = fj.antikt_algorithm
        elif algorithm_l in {'ca', 'cambridge', 'cambridge_aachen'}:
            jet_alg = fj.cambridge_algorithm
        else:
            raise ValueError("algorithm '{}' not understood".format(algorithm))

        return fj.JetDefinition(jet_alg, float(R))(pjs)

    def softdrop(jet, zcut=0.1, beta=0, R=1.0):
        r"""Implements the SoftDrop grooming algorithm on a jet that has been
        found via clustering. Specifically, given a jet, it is recursively
        declustered and the softer branch removed until the SoftDrop condition
        is satisfied:

        $$
        \frac{\min(p_{T,1},p_{T,2})}{p_{T,1}+p_{T,2}} > z_{\rm cut}
        \left(\frac{\Delta R_{12}}{R}\right)^\beta
        $$

        where $1$ and $2$ refer to the two PseudoJets declustered at this stage.
        See the [SoftDrop paper](https://arxiv.org/abs/1402.2657) for a
        complete description of SoftDrop. If you use this function for your
        research, please cite [1402.2657](https://doi.org/10.1007/
        JHEP05(2014)146).

        **Arguments**

        - **jet** : _fastjet.PseudoJet_
            - A FastJet PseudoJet that has been obtained from a suitable
            clustering (typically Cambridge/Aachen).
        - **zcut** : _float_
            - The $z_{\rm cut}$ parameter of SoftDrop. Should be between `0`
            and `1`.
        - **beta** : _int_ or _float_
            - The $\beta$ parameter of SoftDrop.
        - **R** : _float_
            - The jet radius to use for the grooming. Only relevant if `beta!=0`.

        **Returns**

        - _fastjet.PseudoJet_
            - The groomed jet. Note that it will not necessarily have all of
            the same associated structure as the original jet, but it is
            suitable for obtaining kinematic quantities, e.g. [$z_g$](/docs/
            obs/#zg_from_pj).
        """

        parent1, parent2 = fj.PseudoJet(), fj.PseudoJet()
        if not jet.has_parents(parent1, parent2):
            return jet
        
        pt1, pt2 = parent1.pt(), parent2.pt()
        z = min(pt1, pt2)/(pt1 + pt2)
 
        if z >= (zcut if beta == 0 else zcut * (parent1.delta_R(parent2)/R)**beta):
            return jet
        else:
            return softdrop(parent1 if pt1 >= pt2 else parent2, zcut=zcut, beta=beta, R=R)
