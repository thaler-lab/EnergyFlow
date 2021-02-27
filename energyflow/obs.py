"""# Observables

Implementations of come collider physics observables. Some observables
require the [FastJet](http://fastjet.fr/) Python interface to be importable;
if it's not, no warnings or errors will be issued, the observables will simply
not be included in this module.
"""

#   ____  ____   _____
#  / __ \|  _ \ / ____|
# | |  | | |_) | (___
# | |  | |  _ < \___ \
# | |__| | |_) |____) |
#  \____/|____/|_____/

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

from abc import abstractmethod

import numpy as np
from numpy.core.multiarray import c_einsum

from energyflow.base import SingleEnergyCorrelatorBase
from energyflow.utils import import_fastjet, transfer
from energyflow.utils.fastjet_utils import *
from energyflow.utils.particle_utils import *

fj = import_fastjet()

__all__ = ['D2', 'C2', 'C3', 'image_activity']

###############################################################################
# D2
###############################################################################

class D2(SingleEnergyCorrelatorBase):

    """Ratio of EFPs (specifically, energy correlation functions) designed to
    tag two prong signals. In graphs, the formula is:

    <img src="https://github.com/pkomiske/EnergyFlow/raw/images/D2.png" 
    class="obs_center" width="20%"/>

    For additional information, see the [original paper](https://arxiv.org/
    abs/1409.6298).
    """

    # line and triangle EFPs
    graphs = [[(0,1)], [(0,1),(1,2),(2,0)]]
    
    # D2(measure='hadr', beta=2, strassen=False, reg=0., kappa=1, normed=True,
    #    coords=None, check_input=True)
    def __init__(self, measure='hadr', beta=2, strassen=False, reg=0., **kwargs):
        r"""Since a `D2` defines and holds a `Measure` instance, all `Measure`
        keywords are accepted.

        **Arguments**

        - **measure** : {`'hadr'`, `'hadrdot'`, `'hadrefm'`, `'ee'`, `'eeefm'`}
            - The choice of measure. See [Measures](../measures) for additional
            info.
        - **beta** : _float_
            - The parameter $\beta$ appearing in the measure. Must be greater
            than zero.
        - **strassen** : _bool_
            - Whether to use matrix multiplication to speed up the evaluation.
            Not recommended when $\beta=2$ since EFMs are faster.
        - **reg** : _float_
            - A regularizing value to be added to the denominator in the event
            that it is zero. Should typically be something less than 1e-30.
        - **kappa** : {_float_, `'pf'`}
            - If a number, the energy weighting parameter $\kappa$. If `'pf'`,
            use $\kappa=v-1$ where $v$ is the valency of the vertex.
        - **normed** : _bool_
            - Controls normalization of the energies in the measure.
        - **coords** : {`'ptyphim'`, `'epxpypz'`, `None`}
            - Controls which coordinates are assumed for the input. See 
            [Measures](../measures) for additional info.
        - **check_input** : _bool_
            - Whether to check the type of the input each time or assume the
            first input type.
        """

        # initialize base class
        super(D2, self).__init__(self.graphs, measure, beta, strassen, kwargs)
        self.reg = reg

    def _strassen_compute(self, event, zs, thetas):

        # evaluate the measure and get the zs and thetas
        zs, thetas = super(D2, self)._strassen_compute(event, zs, thetas)
        zthetas = thetas * zs
        zthetas2 = np.dot(zthetas, zthetas)

        dot = 1. if self.normed else np.sum(zs)
        line = np.sum(zs[:,np.newaxis] * zthetas)
        triangle = np.sum(zthetas2 * zthetas.T)

        return triangle * dot**3/(line**3 + self.reg)

    def _efp_compute(self, event, zs, thetas, nhats):

        # get EFPset results
        results = super(D2, self)._efp_compute(event, zs, thetas, nhats)
        line, triangle = results[:2]
        dot = 1. if self.normed else results[-1]

        # implement D2 formula
        return triangle * dot**3/(line**3 + self.reg)

###############################################################################
# C2
###############################################################################

class C2(SingleEnergyCorrelatorBase):

    """Ratio of Energy Correlation Functions designed to tag two prong signals.
    In graphs, the formula is:

    <img src="https://github.com/pkomiske/EnergyFlow/raw/images/C2.png" 
    class="obs_center" width="20%"/>

    For additional information, see the [original paper](https://arxiv.org/
    abs/1305.0007).
    """

    # line and triangle EFPs
    graphs = [[(0,1)], [(0,1),(1,2),(2,0)]]
    
    # C2(measure='hadr', beta=2, strassen=False, reg=0., kappa=1, normed=True,
    #    coords=None, check_input=True)
    def __init__(self, measure='hadr', beta=2, strassen=False, reg=0., **kwargs):
        r"""Since a `C2` defines and holds a `Measure` instance, all `Measure`
        keywords are accepted.

        **Arguments**

        - **measure** : {`'hadr'`, `'hadrdot'`, `'hadrefm'`, `'ee'`, `'eeefm'`}
            - The choice of measure. See [Measures](../measures) for additional
            info.
        - **beta** : _float_
            - The parameter $\beta$ appearing in the measure. Must be greater
            than zero.
        - **strassen** : _bool_
            - Whether to use matrix multiplication to speed up the evaluation.
            Not recommended when $\beta=2$ since EFMs are faster.
        - **reg** : _float_
            - A regularizing value to be added to the denominator in the event
            that it is zero. Should typically be something less than 1e-30.
        - **kappa** : {_float_, `'pf'`}
            - If a number, the energy weighting parameter $\kappa$. If `'pf'`,
            use $\kappa=v-1$ where $v$ is the valency of the vertex.
        - **normed** : _bool_
            - Controls normalization of the energies in the measure.
        - **coords** : {`'ptyphim'`, `'epxpypz'`, `None`}
            - Controls which coordinates are assumed for the input. See 
            [Measures](../measures) for additional info.
        - **check_input** : _bool_
            - Whether to check the type of the input each time or assume the
            first input type.
        """

        # initialize base class
        super(C2, self).__init__(self.graphs, measure, beta, strassen, kwargs)
        self.reg = reg

    def _strassen_compute(self, event, zs, thetas):

        # evaluate the measure and get the zs and thetas
        zs, thetas = super(C2, self)._strassen_compute(event, zs, thetas)
        zthetas = thetas * zs
        zthetas2 = np.dot(zthetas, zthetas)

        dot = 1. if self.normed else np.sum(zs)
        line = np.sum(zs[:,np.newaxis] * zthetas)
        triangle = np.sum(zthetas2 * zthetas.T)

        return triangle * dot/(line**2 + self.reg)

    def _efp_compute(self, event, zs, thetas, nhats):

        # get EFPset results
        results = super(C2, self)._efp_compute(event, zs, thetas, nhats)
        line, triangle = results[:2]
        dot = 1. if self.normed else results[-1]

        # implement D2 formula
        return triangle * dot/(line**2 + self.reg)

###############################################################################
# C3
###############################################################################

class C3(SingleEnergyCorrelatorBase):

    """Ratio of Energy Correlation Functions designed to tag three prong
    signals. In graphs, the formula is:

    <img src="https://github.com/pkomiske/EnergyFlow/raw/images/C3.png" 
    class="obs_center" width="30%"/>

    For additional information, see the [original paper](https://arxiv.org/
    abs/1305.0007).
    """

    # line, triangle, and kite EFPs
    graphs = [[(0,1)], [(0,1),(1,2),(2,0)], [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]]
    
    # C3(measure='hadr', beta=2, reg=0., kappa=1, normed=True,
    #    coords=None, check_input=True)
    def __init__(self, measure='hadr', beta=2, reg=0., **kwargs):
        r"""Since a `D2` defines and holds a `Measure` instance, all `Measure`
        keywords are accepted.

        **Arguments**

        - **measure** : {`'hadr'`, `'hadrdot'`, `'hadrefm'`, `'ee'`, `'eeefm'`}
            - The choice of measure. See [Measures](../measures) for additional
            info.
        - **beta** : _float_
            - The parameter $\beta$ appearing in the measure. Must be greater
            than zero.
        - **reg** : _float_
            - A regularizing value to be added to the denominator in the event
            that it is zero. Should typically be something less than 1e-30.
        - **kappa** : {_float_, `'pf'`}
            - If a number, the energy weighting parameter $\kappa$. If `'pf'`,
            use $\kappa=v-1$ where $v$ is the valency of the vertex.
        - **normed** : _bool_
            - Controls normalization of the energies in the measure.
        - **coords** : {`'ptyphim'`, `'epxpypz'`, `None`}
            - Controls which coordinates are assumed for the input. See 
            [Measures](../measures) for additional info.
        - **check_input** : _bool_
            - Whether to check the type of the input each time or assume the
            first input type.
        """

        # initialize base class
        super(C3, self).__init__(self.graphs, measure, beta, False, kwargs)
        self.reg = reg

    def _strassen_compute(self, *args, **kwargs):
        raise NotImplementedError('no strassen implementation for C3')

    def _efp_compute(self, event, zs, thetas, nhats):

        # get EFPset results
        results = super(C3, self)._efp_compute(event, zs, thetas, nhats)

        # implement D2 formula
        return results[2]*results[0]/(results[1]**2 + self.reg)

###############################################################################
# Image Activity (a.k.a. N95)
###############################################################################

def image_activity(ptyphis, f=0.95, R=1.0, npix=33, center=None, axis=None):
    """Image activity, also known as $N_f$, is the minimum number of pixels
    in an image that contain a fraction $f$ of the total pT.

    **Arguments**

    - **ptyphis** : _2d numpy.ndarray_
        - Array of particles in hadronic coordinates; the mass is optional
        since it is not used in the computation of this observable.
    - **f** : _float_
        - The fraction $f$ of total pT that is to be contained by the pixels.
    - **R** : _float_
        - Half of the length of one side of the square space to tile with
        pixels when forming the image. For a conical jet, this should typically
        be the jet radius.
    - **npix** : _int_
        - The number of pixels along one dimension of the image, such that the
        image has shape `(npix,npix)`.
    - **center** : _str_ or `None`
        - If not `None`, the centering scheme to use to center the particles
        prior to calculating the image activity. See the option of the same
        name for [`center_ptyphims`](/docs/utils/#center_ptyphims).
    - **axis** : _numpy.ndarray_ or `None`
        - If not `None`, the `[y,phi]` values to use for centering. If `None`,
        the center of the image will be at `(0,0)`.

    **Returns**

    - _int_
        - The image activity defined for the specified image paramters.
    """

    # make bins
    bins = np.linspace(-R, R, npix + 1)

    # center if requested
    if center is not None:
        ptyphis = center_ptyphims(ptyphis, center=center, copy=True)

    # handle passing an axis
    if axis is not None:
        bins = (axis[0] + bins, axis[1] + bins)

    # make 2d image of pt
    ptyphis = np.atleast_2d(ptyphis)
    pixels = np.histogram2d(ptyphis[:,1], ptyphis[:,2], weights=ptyphis[:,0], bins=bins)[0].flatten()

    # calcualte image activity
    nf = np.argmax(np.cumsum(np.sort(pixels/(pixels.sum() + 10**-30))[::-1]) >= f) + 1

    return nf

###############################################################################
# Observables relying on FastJet
###############################################################################

if fj:

    __all__ += ['zg', 'zg_from_pj']

    def zg(ptyphims, zcut=0.1, beta=0, R=1.0, algorithm='ca'):
        r"""Groomed momentum fraction of a jet, as calculated on an array of
        particles in hadronic coordinates. First, the particles are converted
        to FastJet PseudoJets and clustered according to the specified
        algorithm. Second, the jet is groomed according to the specified
        SoftDrop parameters and the momentum fraction of the surviving pair of
        Pseudojets is computed. See the [SoftDrop paper](https://arxiv.org/abs/
        1402.2657) for a complete description of SoftDrop.

        **Arguments**

        - **ptyphims** : _numpy.ndarray_
            - An array of particles in hadronic coordinates that will be
            clustered into a single jet and groomed.
        - **zcut** : _float_
            - The $z_{\rm cut}$ parameter of SoftDrop. Should be between `0`
            and `1`.
        - **beta** : _int_ or _float_
            - The $\beta$ parameter of SoftDrop.
        - **R** : _float_
            - The jet radius to use for the grooming. Only relevant if `beta!=0`.
        - **algorithm** : {'kt', 'ca', 'antikt'}
            - The jet algorithm to use when clustering the particles. Same as
            the argument of the same name of [`cluster`](/docs/utils/#cluster).

        **Returns**

        - _float_
            - The groomed momentum fraction of the given jet."""
        
        return zg_from_pj(cluster(pjs_from_ptyphims(ptyphims), algorithm=algorithm)[0], 
                          zcut=zcut, beta=beta, R=R)

    def zg_from_pj(pseudojet, zcut=0.1, beta=0, R=1.0):
        r"""Groomed momentum fraction $z_g$, as calculated on an ungroomed (but
        already clustered) FastJet PseudoJet object. First, the jet is groomed
        according to the specified SoftDrop parameters and then the momentum
        fraction of the surviving pair of Pseudojets is computed. See the
        [SoftDrop paper](https://arxiv.org/abs/1402.2657) for a complete
        description of SoftDrop. This version of $z_g$ is provided in addition
        to the above function so that a jet does not need to be reclustered if
        multiple grooming parameters are to be used.

        **Arguments**

        - **pseudojet** : _fastjet.PseudoJet_
            - A FastJet PseudoJet that has been obtained from a suitable
            clustering (typically Cambridge/Aachen for SoftDrop).
        - **zcut** : _float_
            - The $z_{\rm cut}$ parameter of SoftDrop. Should be between `0`
            and `1`.
        - **beta** : _int_ or _float_
            - The $\beta$ parameter of SoftDrop.
        - **R** : _float_
            - The jet radius to use for the grooming. Only relevant if `beta!=0`.

        **Returns**

        - _float_
            - The groomed momentum fraction of the given jet.
        """

        sd_jet = softdrop(pseudojet, zcut=zcut, beta=beta, R=R)

        parent1, parent2 = fj.PseudoJet(), fj.PseudoJet()
        if not sd_jet.has_parents(parent1, parent2):
            return 0.
        
        pt1, pt2 = parent1.pt(), parent2.pt()
        ptsum = pt1 + pt2

        return 0. if ptsum == 0. else min(pt1, pt2)/ptsum
