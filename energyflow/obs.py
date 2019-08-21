"""Implementations of some observables that are not covered by other portions
of EnergyFlow. Some observables require the [FastJet](http://fastjet.fr/)
Python interface to be importable; if it's not, no warnings or errors will be
issued, the observables will simply not be included in this module."""
from __future__ import absolute_import, division, print_function

import numpy as np

from energyflow.utils import import_fastjet
from energyflow.utils.fastjet_utils import *
from energyflow.utils.particle_utils import *

fj = import_fastjet()

__all__ = ['image_activity']

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
