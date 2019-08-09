""""""
from __future__ import absolute_import, division, print_function

import numpy as np

from energyflow.utils import import_fastjet
from energyflow.utils.fastjet_utils import *
from energyflow.utils.particle_utils import *

fj = import_fastjet()

__all__ = ['image_activity']

def image_activity(ptyphis, f=0.95, R=1.0, npix=33, center=None, axis=None, reg=10**-30):

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
    nf = np.argmax(np.cumsum(np.sort(pixels/(pixels.sum() + reg))[::-1]) >= f) + 1

    return nf

if fj:

    __all__ += ['zg', 'zg_from_pj']

    def zg_from_pj(pseudojet, zcut=0.1, beta=0, R=1.0):

        sd_jet = softdrop(pseudojet, zcut=zcut, beta=beta, R=R)

        parent1, parent2 = fj.PseudoJet(), fj.PseudoJet()
        if not sd_jet.has_parents(parent1, parent2):
            return 0.
        
        pt1, pt2 = parent1.pt(), parent2.pt()
        ptsum = pt1 + pt2

        return 0. if ptsum == 0. else min(pt1, pt2)/ptsum

    def zg(ptyphims, zcut=0.1, beta=0, R=1.0, algorithm='ca'):
        return zg_from_pj(cluster(pjs_from_ptyphims(ptyphims), algorithm=algorithm)[0], zcut=zcut, beta=beta, R=R)
