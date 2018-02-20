from __future__ import absolute_import

import numpy as np
import pytest

from energyflow import *

g_def_8 = Generator(dmax=8, filename='default')
g_8 = Generator(dmax=8)

def test_gen_subset_specs():
    assert np.all(g_def_8.specs == g_8.specs)
