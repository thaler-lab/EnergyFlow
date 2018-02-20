"""A subpackage containing utility functions and classes. Not meant to be 
imported directly in energyflow."""

from __future__ import absolute_import

import os
import numpy as np

from . import graph
from . import measure
from .graph import *
from .measure import *

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
default_file = os.path.join(data_dir, 'efps_d_le_10.npz')

def igraph_import():
    """
    Determines if igraph can be imported. 

    Returns
    -------
    output : {igraph, False}
        The igraph module if it was successfully imported, otherwise False.
    """
    
    try:
        import igraph
    except:
        igraph = False
    return igraph

def kwargs_check(name, kwargs, allowed=[]):
    for k in kwargs:
        if k in allowed:
            continue
        message = name + '() got an unexpected keyword argument \'{}\''.format(k)
        raise TypeError(message)

comp_map = {'>':  '__gt__', 
            '<':  '__lt__', 
            '>=': '__ge__', 
            '<=': '__le__',
            '==': '__eq__', 
            '!=': '__ne__'}

def explicit_comp(obj, comp, val):
    return getattr(obj, comp_map[comp])(val)

def concat_specs(c_specs, d_specs):
    if len(d_specs):
        return np.concatenate((c_specs, d_specs), axis=0)
    else:
        return c_specs

def transfer(obj1, obj2, attrs):
    if isinstance(obj2, dict):
        for attr in attrs:
            setattr(obj1, attr, obj2[attr])
    else:
        for attr in attrs:
            setattr(obj1, attr, getattr(obj2, attr))

def unique_dim_nlows(efm_specs, d={}):
    for spec in efm_specs:
        d.setdefault(spec[0], set()).update(spec[1:])
    return d

def nonegen():
    while True:
        yield None

