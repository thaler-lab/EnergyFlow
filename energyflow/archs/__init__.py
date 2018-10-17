"""
Energy Flow Networks (EFNs) and Particle Flow Networks (PFNs)
are model architectures designed for learning from collider events
as unordered, variable-length sets of particles. Both EFNs and PFNs are
parameterized by a learnable per-particle function $\Phi$ and latent space function $F$.


An EFN takes the following form:
$$\\text{EFN}=F\\left(\\sum_{i=1}^M z_i \Phi(\hat p_i)\\right)$$
where $z_i$ is a measure of the energy of particle $i$, such as $z_i = p_{T,i}$, and $\\hat p_i$ is a measure 
of the angular information of particle $i$, such as $\\hat p_i = (y_i,\\phi_i)$.
Any infrared- and collinear-safe observable can be parameterized in this form.

A PFN takes the following form:
$$\\text{PFN}=F\\left(\\sum_{i=1}^M \Phi(p_i)\\right)$$
where $p_i$ is the information of particle $i$, such as its four-momentum, charge, or flavor.
Any observable can be parameterized in this form.
See the [Deep Sets](https://arxiv.org/abs/1703.06114) framework for additional discussion.

Since these architectures are not
used by the core EnergyFlow code, and require the external 
[Keras](https://keras.io) and [scikit-learn](http://scikit-learn.org/)
libraries, they are not imported by default but must be explicitly 
imported, e.g. `from energyflow.archs import *`.
EnergyFlow also contains several additional model architectures for ease of using
common models that frequently appear in the intersection of 
particle physics and machine learning. 
"""
from __future__ import absolute_import

import warnings

__all__ = []

# requires keras
try:
    from . import cnn
    from . import dnn
    from . import efn
    from .cnn import *
    from .dnn import *
    from .efn import *

    __all__ += cnn.__all__ + dnn.__all__ + efn.__all__

except ImportError as e:
    warnings.warn('could not import some architectures - ' + str(e))

# requires sklearn
try:
    from . import linear
    from .linear import *

    __all__ += linear.__all__

except ImportError as e:
    warnings.warn('could not import some architectures - ' + str(e))
