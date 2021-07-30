"""## FastJet Utils

The [FastJet package](http://fastjet.fr/) provides, among other things, fast
jet clustering utilities. Since EnergyFlow 2.0, the [PyFJCore](https://github.
com/pkomiske/PyFJCore) package has been used to provide Python access to
FastJet's classes and algorithms. Keep in mind that if you use these utilities
in EnergyFlow for published research, you are relying on the FastJet library so
please [cite FastJet appropriately](http://fastjet.fr/about.html).

See the [PyFJCore README](https://github.com/pkomiske/PyFJCore/blob/main/
README.md) for more documentation on its functions and classes.
"""

#  ______       _____ _______   _ ______ _______         _    _ _______ _____ _       _____
# |  ____/\    / ____|__   __| | |  ____|__   __|       | |  | |__   __|_   _| |     / ____|
# | |__ /  \  | (___    | |    | | |__     | |          | |  | |  | |    | | | |    | (___
# |  __/ /\ \  \___ \   | |_   | |  __|    | |          | |  | |  | |    | | | |     \___ \
# | | / ____ \ ____) |  | | |__| | |____   | |   ______ | |__| |  | |   _| |_| |____ ____) |
# |_|/_/    \_\_____/   |_|\____/|______|  |_|  |______| \____/   |_|  |_____|______|_____/

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

import numpy as np
import six

from energyflow import fastjet
from energyflow.utils.particle_utils import phi_fix

__all__ = [
    'pjs_from_ptyphims',
    'pjs_from_p4s',
    'ptyphims_from_pjs',
    'p4s_from_pjs',
    'jet_definition',
    'cluster',
    'softdrop',
]

def pjs_from_ptyphims(ptyphims):
    """Converts an array of particles in hadronic coordinates to FastJet
    PseudoJets. See the [`ptyphim_array_to_pseudojets`](https://github.com/
    pkomiske/PyFJCore/blob/main/README.md/#NumPy-conversion-functions) method
    of PyFJCore.

    **Arguments**

    - **ptyphims** : _2d numpy.ndarray_
        - An array of particles in hadronic coordinates, `(pt, y, phi, [mass])`;
        the mass is optional. Any additional features are added as user info for
        each PseudoJet, accessible using the `.python_info()` method.

    **Returns**

    - _tuple_ of _PseudoJet_
        - A Python tuple of `PseudoJet`s corresponding to the input particles.
    """

    return fastjet.ptyphim_array_to_pseudojets(ptyphims)

def pjs_from_p4s(p4s):
    """Converts particles in Cartesian coordinates to FastJet PseudoJets. See
    the [`epxpypz_array_to_pseudojets`](https://github.com/pkomiske/PyFJCore/
    blob/main/README.md/#NumPy-conversion-functions) method of PyFJCore.

    **Arguments**

    - **p4s** : _2d numpy.ndarray_
        - An array of particles in Cartesian coordinates, `(E, px, py, pz)`. Any
        additional features are added as user info for each PseudoJet,
        accessible using the `.python_info()` method.

    **Returns**

    - _tuple_ of _PseudoJet_
        - A Python tuple of `PseudoJet`s corresponding to the input particles.
    """

    return fastjet.epxpypz_array_to_pseudojets(p4s)

def ptyphims_from_pjs(pjs, phi_ref=False, mass=True, phi_std=False, float32=False):
    """Extracts hadronic four-vectors from FastJet PseudoJets. See the
    [`pseudojets_to_ptyphim_array`](https://github.com/pkomiske/PyFJCore/blob/
    main/README.md/#NumPy-conversion-functions) method of PyFJCore.

    **Arguments**

    - **pjs** : iterable of _PseudoJet_
        - An iterable of PseudoJets (list, tuple, array, etc).
    - **phi_ref** : _float_ or `None` or `False`
        - The reference phi value to use for phi fixing. If `False`, then no
        phi fixing is performed. If `None`, then the phi value of the first
        particle is used.
    - **mass** : _bool_
        - Whether or not to include the mass in the extracted four-vectors.

    **Returns**

    - _numpy.ndarray_
        - A 2D array of four-vectors corresponding to the given PseudoJets as
        `(pT, y, phi, [mass])`, where the mass is optional.
    """

    if phi_ref is None and len(pjs):
        phi_ref = pjs[0].phi_std() if phi_std else pjs[0].phi()

    return fastjet.pseudojets_to_ptyphim_array(pjs, mass=mass, phi_std=phi_std,
                                                    phi_ref=phi_ref, float32=float32)

def p4s_from_pjs(pjs, float32=False):
    """Extracts Cartesian four-vectors from FastJet PseudoJets. See the
    [`pseudojets_to_epxpypz_array`](https://github.com/pkomiske/PyFJCore/blob/
    main/README.md/#NumPy-conversion-functions) method of PyFJCore.

    **Arguments**

    - **pjs** : iterable of _PseudoJet_
        - An iterable of PseudoJets (list, tuple, array, etc).

    **Returns**

    - _numpy.ndarray_
        - A 2D array of four-vectors corresponding to the given PseudoJets as
        `(E, px, py, pz)`.
    """

    return fastjet.pseudojets_to_epxpypz_array(pjs, float32=float32)

JET_ALGORITHMS = {
    'kt': fastjet.kt_algorithm,
    'cambridge': fastjet.cambridge_algorithm,
    'antikt': fastjet.antikt_algorithm,
    'genkt': fastjet.genkt_algorithm,
    'ee_kt': fastjet.ee_kt_algorithm,
    'ee_genkt': fastjet.ee_genkt_algorithm,

    # shorthands for the above
    'ca': fastjet.cambridge_algorithm,
    'cambridge_aachen': fastjet.cambridge_algorithm,
    'akt': fastjet.antikt_algorithm,
}

RECOMBINATION_SCHEMES = {
    'E_scheme': fastjet.E_scheme,
    'Et_scheme': fastjet.Et_scheme,
    'Et2_scheme': fastjet.Et2_scheme,
    'pt_scheme': fastjet.pt_scheme,
    'pt2_scheme': fastjet.pt2_scheme,
    'WTA_pt_scheme': fastjet.WTA_pt_scheme,
}

_JET_DEFINITION_KWARGS = {'algorithm', 'R', 'extra', 'recomb'}

# jet_definition(algorithm='ca', R=fastjet.JetDefinition.max_allowable_R, recomb='E_scheme')
def jet_definition(algorithm=fastjet.cambridge_algorithm,
                   R=fastjet.JetDefinition.max_allowable_R,
                   extra=None,
                   recomb=fastjet.E_scheme):
    """Creates a JetDefinition from the specified arguments.

    **Arguments**

    - **algorithm** : _str_ or _int_
        - A string such as `'kt'`, `'akt'`, `'antikt'`, `'ca'`, 
        `'cambridge'`, or `'cambridge_aachen'`; or an integer corresponding to a
        fastjet.JetAlgorithm value.
    - **R** : _float_
        - The jet radius. The default value corresponds to `max_allowable_R` as
        defined by the FastJet package.
    - **extra** : _float_ or `None`
        - Some jet algorithms, like generalized $k_T$, take an extra parameter.
        If not `None`, `extra` can be used to provide that parameter.
    - **recomb** : _str_ or _int_
        - An integer corresponding to a RecombinationScheme, or a string
        specifying a name which is looked up in the PyFJCore module.

    **Returns**

    - _JetDefinition_
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

    if extra is None:
        return fastjet.JetDefinition(algorithm, float(R), recomb)
    else:
        return fastjet.JetDefinition(algorithm, float(R), float(extra), recomb)

# cluster(pjs, jetdef=None, N=None, dcut=None, ptmin=0., return_cs=False, **kwargs)
def cluster(pjs, jetdef=None, N=None, dcut=None, ptmin=0., return_cs=False, **kwargs):
    """Clusters an iterable of PseudoJets. Uses a jet definition that can
    either be provided directly or specified using the same keywords as the
    `jet_def` function. The jets returned can either be includive, the 
    default, or exclusive according to either a maximum number of subjets
    or a particular `dcut`.

    **Arguments**

    - **pjs** : iterable of _PseudoJet_
        - A list of Pseudojets representing particles or other kinematic
        objects that are to be clustered into jets.
    - **jetdef** : _JetDefinition_ or `None`
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

    - _tuple_ of _PseudoJet_
        - A tuple of PseudoJets corresponding to the clustered jets.
    """

    if jetdef is None:
        jetdef = jet_definition(**kwargs)

    cs = fastjet.ClusterSequence(pjs, jetdef)
    if return_cs:
        return cs

    # specified N means we want exclusive_jets_up_tp
    if N is not None:
        jets = cs.exclusive_jets_up_to(N)

    # specified dcut means we want exclusive jets
    elif dcut is not None:
        jets = cs.exclusive_jets(float(dcut))

    # inclusive jets by default
    else:
        jets = cs.inclusive_jets(ptmin)

    # handle lifetime of cs object
    if len(pjs):
        cs.thisown = False
        cs.delete_self_when_unused()

    return jets

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

    - **jet** : _PseudoJet_
        - A PseudoJet that has been obtained from a suitable clustering
        (typically Cambridge/Aachen).
    - **zcut** : _float_
        - The $z_{\rm cut}$ parameter of SoftDrop. Should be between `0.0` and
        `1.0`.
    - **beta** : _int_ or _float_
        - The $\beta$ parameter of SoftDrop.
    - **R** : _float_
        - The jet radius to use for the grooming. Only relevant if `beta != 0.`.

    **Returns**

    - _PseudoJet_
        - The groomed jet. Note that it will not necessarily have all of the
        same associated structure as the original jet, but it is suitable for
        obtaining kinematic quantities, e.g. [$z_g$](/docs/obs/#zg_from_pj).
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
