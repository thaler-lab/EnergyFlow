#   _____ ______ _   _ ______ _____  _____ _____          _    _ _______ _____ _       _____
#  / ____|  ____| \ | |  ____|  __ \|_   _/ ____|        | |  | |__   __|_   _| |     / ____|
# | |  __| |__  |  \| | |__  | |__) | | || |             | |  | |  | |    | | | |    | (___
# | | |_ |  __| | . ` |  __| |  _  /  | || |             | |  | |  | |    | | | |     \___ \
# | |__| | |____| |\  | |____| | \ \ _| || |____  ______ | |__| |  | |   _| |_| |____ ____) |
#  \_____|______|_| \_|______|_|  \_\_____\_____||______| \____/   |_|  |_____|______|_____/

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

import contextlib
import functools
import gzip
import itertools
import json
import multiprocessing
import os
import sys
import time
import warnings

import numpy as np

__all__ = [
    'EF_DATA_DIR',
    'DEFAULT_EFP_FILE',
    'COMP_MAP',
    'REVERSE_COMPS',
    'DROPBOX_URL_PATTERN',
    'ZENODO_URL_PATTERN',
    'concat_specs',
    'create_pool',
    'explicit_comp',
    'iter_or_rep', 
    'kwargs_check',
    'load_efp_file',
    'timing', 
    'transfer'
]

# get access to the data directory of the installed package and the default efp file
EF_DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
DEFAULT_EFP_FILE = os.path.join(EF_DATA_DIR, 'efps_d_le_10.json.gz')

# dictionaries for helping with automated comparisons
COMP_MAP = {
    '>':  '__gt__', 
    '<':  '__lt__', 
    '>=': '__ge__', 
    '<=': '__le__',
    '==': '__eq__', 
    '!=': '__ne__'
}
REVERSE_COMPS = {'>': '<', '<': '>', '<=': '>=', '>=': '<='}

# URL patterns
DROPBOX_URL_PATTERN = 'https://www.dropbox.com/s/{}/{}?dl=1'
ZENODO_URL_PATTERN = 'https://zenodo.org/record/{}/files/{}?download=1'

# concatenates con. and disc. specs along axis 0, handling empty disc. specs
def concat_specs(c_specs, d_specs):
    if len(d_specs):
        return np.concatenate((c_specs, d_specs), axis=0)
    else:
        return c_specs

# handle Pool not being a context manager in Python < 3.4
@contextlib.contextmanager
def create_pool(*args, context=None, **kwargs):

    if context is not None and context not in multiprocessing.get_all_start_methods():
        warnings.warn("'{}' is not available as a multiprocessing start method, ".format(context)
                      + "EnergyFlow multicore functionality may not work properly")
        with multiprocessing.Pool(*args, **kwargs) as pool:
            yield pool

    else:
        with multiprocessing.get_context(context).Pool(*args, **kwargs) as pool:
            yield pool

# applies comparison comp of obj on val
def explicit_comp(obj, comp, val):
    return getattr(obj, COMP_MAP[comp])(val)

# return argument if iterable else make repeat generator
def iter_or_rep(arg):
    if isinstance(arg, (tuple, list)):
        if len(arg) == 1:
            return itertools.repeat(arg[0])
        else:
            return arg
    elif isinstance(arg, itertools.repeat):
        return arg
    else:
        return itertools.repeat(arg)

# raises TypeError if unexpected keyword left in kwargs
def kwargs_check(name, kwargs, allowed=None):
    allowed = frozenset() if allowed is None else frozenset(allowed)
        
    for k in kwargs:
        if k in allowed:
            continue
        raise TypeError(name + '() got an unexpected keyword argument \'{}\''.format(k))

# load EFP file
def load_efp_file(filename):
    if filename is None or filename == 'default':
        filename = DEFAULT_EFP_FILE

    if filename.endswith('.npz'):
        return np.load(filename, allow_pickle=True)

    if '.json' in filename:
        if '.gz' in filename:
            with gzip.open(filename, 'rt') as f:
                return json.load(f)
        else:
            with open(filename, 'rt') as f:
                return json.load(f)

    return None

# timing meta-decorator
def timing(obj, func):
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        ts = time.process_time()
        r = func(*args, **kwargs)
        te = time.process_time()
        obj.times.append(te - ts)
        return r
    return decorated

# transfers attrs from obj2 (dict or object) to obj1
def transfer(obj1, obj2, attrs):
    if isinstance(obj2, dict):
        for attr in attrs:
            setattr(obj1, attr, obj2[attr])
    else:
        for attr in attrs:
            setattr(obj1, attr, getattr(obj2, attr))

# handle pickling methods in python 2
if sys.version_info[0] == 2:
    import copy_reg
    import types

    def pickle_method(method):
        func_name = method.__name__
        obj = method.__self__
        cls = obj.__class__
        return unpickle_method, (func_name, obj, cls)

    def unpickle_method(func_name, obj, cls):
        for cls in cls.mro():
            try:
                func = cls.__dict__[func_name]
            except KeyError:
                pass
            else:
                break
        return func.__get__(obj, cls)

    copy_reg.pickle(types.MethodType, pickle_method, unpickle_method)
