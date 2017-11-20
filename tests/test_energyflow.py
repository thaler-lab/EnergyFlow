from __future__ import absolute_import

import energyflow as ef

import pytest

def test_trivial():
    assert 1 == 1

def test_nontrivial():
    assert 'EFP' in dir(ef)