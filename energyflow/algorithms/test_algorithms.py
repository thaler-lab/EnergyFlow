from __future__ import absolute_import

import pytest

from energyflow.algorithms import *

def test_has_ve():
    assert VariableElimination

def test_has_int_partition_ordered():
    assert int_partition_ordered

def test_has_int_partition_unordered():
    assert int_partition_unordered

# got values from mathematica
@pytest.mark.parametrize('n, ans', [(1,1),(5,7),(10,42),(15,176),(20,627),(25,1958)])
def test_int_partition_ordered_len(n, ans):
    assert len(list(int_partition_unordered(n))) == ans

def test_ve_attrs():
    pass