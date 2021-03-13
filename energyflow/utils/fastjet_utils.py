"""## FastJet Utils

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

#  ______       _____ _______   _ ______ _______        _    _ _______ _____ _       _____
# |  ____/\    / ____|__   __| | |  ____|__   __|      | |  | |__   __|_   _| |     / ____|
# | |__ /  \  | (___    | |    | | |__     | |         | |  | |  | |    | | | |    | (___
# |  __/ /\ \  \___ \   | |_   | |  __|    | |         | |  | |  | |    | | | |     \___ \
# | | / ____ \ ____) |  | | |__| | |____   | |   ______| |__| |  | |   _| |_| |____ ____) |
# |_|/_/    \_\_____/   |_|\____/|______|  |_|  |______|\____/   |_|  |_____|______|_____/

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

import numpy as np
import six

from energyflow.utils.particle_utils import phi_fix

__all__ = []

# determine if fastjet can be imported, returns either the fastjet module or false
def _import_fastjet():
    try:
        import fastjet
    except:
        fastjet = False
    return fastjet

fastjet = _import_fastjet()
if fastjet:

    __all__ += [
        'pjs_from_ptyphims',
        'pjs_from_p4s',
        'ptyphims_from_pjs',
        'p4s_from_pjs',
        'jet_def',
        'cluster',
        'softdrop',
    ]

    def pjs_from_ptyphims(ptyphims):
        """Converts particles in hadronic coordinates to FastJet PseudoJets.

        **Arguments**

        - **ptyphims** : _2d numpy.ndarray_
            - An array of particles in hadronic coordinates. The mass is
            optional and will set to zero if not present.

        **Returns**

        - _list_ of _fastjet.PseudoJet_
            - A list of PseudoJets corresponding to the input particles.
        """

        if ptyphims.shape[1] >= 4:
            return [fastjet.PtYPhiM(p[0], p[1], p[2], p[3]) for p in ptyphims]
        else:
            return [fastjet.PtYPhiM(p[0], p[1], p[2]) for p in ptyphims]

    def pjs_from_p4s(p4s):
        """Converts particles in Cartesian coordinates to FastJet PseudoJets.

        **Arguments**

        - **p4s** : _2d numpy.ndarray_
            - An array of particles in Cartesian coordinates, `[E, px, py, pz]`.

        **Returns**

        - _list_ of _fastjet.PseudoJet_
            - A list of PseudoJets corresponding to the input particles.
        """

        return [fastjet.PseudoJet(p4[1], p4[2], p4[3], p4[0]) for p4 in p4s]

    def ptyphims_from_pjs(pjs, phi_ref=None, mass=True):
        """Extracts hadronic four-vectors from FastJet PseudoJets.

        **Arguments**

        - **pjs** : _list_ of _fastjet.PseudoJet_
            - An iterable of PseudoJets.
        - **phi_ref** : _float_ or `None`
            - The reference phi value to use for phi fixing. If `None`, then no
            phi fixing is performed.
        - **mass** : _bool_
            - Whether or not to include the mass in the extracted four-vectors.

        **Returns**

        - _numpy.ndarray_
            - An array of four-vectors corresponding to the given PseudoJets as
            `[pT, y, phi, m]`, where the mass is optional.
        """

        if mass:
            event = np.asarray([[pj.pt(), pj.rap(), pj.phi(), pj.m()] for pj in pjs])
        else:
            event = np.asarray([[pj.pt(), pj.rap(), pj.phi()] for pj in pjs])

        if phi_ref is not None:
            phi_fix(event[:,2], phi_ref, copy=False)

        return event

    def p4s_from_pjs(pjs):
        """Extracts Cartesian four-vectors from FastJet PseudoJets.

        **Arguments**

        - **pjs** : _list_ of _fastjet.PseudoJet_
            - An iterable of PseudoJets.

        **Returns**

        - _numpy.ndarray_
            - An array of four-vectors corresponding to the given PseudoJets as
            `[E, px, py, pz]`.
        """

        return np.asarray([[pj.E(), pj.px(), pj.py(), pj.pz()] for pj in pjs])

    JET_ALGORITHMS = {
        'kt': fastjet.kt_algorithm,
        'eekt': fastjet.ee_kt_algorithm,
        'akt': fastjet.antikt_algorithm,
        'antikt': fastjet.antikt_algorithm,
        'ca': fastjet.cambridge_algorithm,
        'cambridge': fastjet.cambridge_algorithm,
        'cambridge_aachen': fastjet.cambridge_algorithm
    }

    RECOMBINATION_SCHEMES = {
        'escheme': fastjet.E_scheme,
        'etscheme': fastjet.Et_scheme,
        'et2scheme': fastjet.Et2_scheme,
        'ptscheme': fastjet.pt_scheme,
        'pt2scheme': fastjet.pt2_scheme,
        'wtaptscheme': fastjet.WTA_pt_scheme
    }

    def jet_def(algorithm=fastjet.cambridge_algorithm,
                R=fastjet.JetDefinition.max_allowable_R,
                recomb=fastjet.E_scheme):
        """Creates a fastjet JetDefinition from the specified arguments.

        **Arguments**

        - **algorithm** : _str_ or _int_
            - A string such as `'kt'`, `'akt'`, `'antikt'`, `'ca'`, 
            `'cambridge'`, or `'cambridge_aachen'`; or an integer corresponding
            to a fastjet.JetAlgorithm value.
        - **R** : _float_
            - The jet radius. The default value corresponds to
            `max_allowable_R` as defined by the FastJet python package.
        - **recomb** : _int_
            - An integer corresponding to a fastjet RecombinationScheme.

        **Returns**

        - _fastjet.JetDefinition_
            - A JetDefinition instance corresponding to the given arguments.
        """

        if isinstance(algorithm, six.string_types):
            try:
                algorithm = JET_ALGORITHMS[algorithm.lower()]
            except KeyError:
                raise ValueError("algorithm '{}' not understood".format(algorithm))

        if isinstance(recomb, six.string_types):
            try:
                recomb = RECOMBINATION_SCHEMES[recomb.lower()]
            except KeyError:
                raise ValueError("recombination scheme '{}' not understood".format(recomb))

        return fastjet.JetDefinition(algorithm, float(R), recomb)

    def cluster(pjs, jetdef=None, N=None, dcut=None, ptmin=0., **kwargs):
        """Clusters an iterable of PseudoJets. Uses a jet definition that can
        either be provided directly or specified using the same keywords as the
        `jet_def` function. The jets returned can either be includive, the 
        default, or exclusive according to either a maximum number of subjets
        or a particular `dcut`.

        **Arguments**

        - **pjs** : _list_ of _fastjet.PseudoJet_
            - A list of Pseudojets representing particles or other kinematic
            objects that are to be clustered into jets.
        - **jetdef** : _fastjet.JetDefinition_ or `None`
            - The `JetDefinition` used for the clustering. If `None`, the
            keyword arguments are passed on to `jet_def` to create a
            `JetDefinition`.
        - **N** : _int_ or `None`
            - If not `None`, then the `exclusive_jets_up_to` method of the
            `ClusterSequence` class is used to get up to `N` exclusive jets.
        - **dcut** : _float_ or `None`
            - If not `None`, then the `exclusive_jets` method of the
            `ClusterSequence` class is used to get exclusive jets with the
            provided `dcut` parameter.
        - **ptmin** : _float_
            - If both `N` and `dcut` are `None`, then inclusive jets are
            returned using this value as the minimum transverse momentum value.
        - ***kwargs** : keyword arguments
            - If `jetdef` is `None`, then these keyword arguments are passed on
            to the `jet_def` function.

        **Returns**

        - _list_ of _fastjet.PseudoJet_
            - A list of PseudoJets corresponding to the clustered jets.
        """

        if jetdef is None:
            jetdef = jet_def(**kwargs)

        cs = fastjet.ClusterSequence(pjs, jetdef)
        cs.thisown = False

        # specified N means we want exclusive_jets_up_tp
        if N is not None:
            return cs.exclusive_jets_up_to(N)

        # specified dcut means we want exclusive jets
        if dcut is not None:
            cs.exclusive_jets(float(dcut))

        # inclusive jets by default
        return cs.inclusive_jets(ptmin)

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

        parent1, parent2 = fastjet.PseudoJet(), fastjet.PseudoJet()
        if not jet.has_parents(parent1, parent2):
            return jet
        
        pt1, pt2 = parent1.pt(), parent2.pt()
        z = min(pt1, pt2)/(pt1 + pt2)
 
        if z >= (zcut if beta == 0 else zcut * (parent1.delta_R(parent2)/R)**beta):
            return jet
        else:
            return softdrop(parent1 if pt1 >= pt2 else parent2, zcut=zcut, beta=beta, R=R)
