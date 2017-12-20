"""These are the classes that actually compute EFPs. 
If you plan on studying single graphs, `EFP` would typically be used. 
For studying EFPs in large collections, use `EFPSet` for greater efficiency and simplicity.
"""

from __future__ import absolute_import

# import efps
from . import efp
from . import efpset
from .efp import *
from .efpset import *
__all__ = efp.__all__ + efpset.__all__
