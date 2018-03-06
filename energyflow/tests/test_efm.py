from __future__ import absolute_import

import numpy as np
import pytest

import energyflow as ef

def test_has_efm():
    assert ef.EFM

def test_has_EFMSet():
    assert ef.EFMSet

# test subslicing construction
#meas = ef.utils.Measure()
