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
from functools import wraps
import gzip
from itertools import repeat
import json
import multiprocessing
import os
import sys
import time
import warnings

import numpy as np
import six

__all__ = [
    'ALL_EXAMPLES',
    'COMP_MAP',
    'DEFAULT_EFP_FILE',
    'EF_DATA_DIR',
    'REVERSE_COMPS',
    'ZENODO_URL_PATTERN',
    'concat_specs',
    'create_pool',
    'explicit_comp',
    'import_fastjet',
    'iter_or_rep', 
    'kwargs_check',
    'load_efp_file',
    'sel_arg_check',
    'timing', 
    'transfer'
]

# list of examples
ALL_EXAMPLES = [
    'efn_example.py',
    'efn_regression_example.py',
    'pfn_example.py',
    'cnn_example.py',
    'dnn_example.py',
    'efp_example.py',
    'animation_example.py'
]

# get access to the data directory of the installed package and the default efp file
EF_DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
DEFAULT_EFP_FILE = os.path.join(EF_DATA_DIR, 'efps_d_le_10.json.gz')
REVERSE_COMPS = {'>': '<', '<': '>', '<=': '>=', '>=': '<='}
COMP_MAP = {
    '>':  '__gt__', 
    '<':  '__lt__', 
    '>=': '__ge__', 
    '<=': '__le__',
    '==': '__eq__', 
    '!=': '__ne__'
}

# zenodo URL pattern
ZENODO_URL_PATTERN = 'https://zenodo.org/record/{}/files/{}?download=1'

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

# concatenates con. and disc. specs along axis 0, handling empty disc. specs
def concat_specs(c_specs, d_specs):
    if len(d_specs):
        return np.concatenate((c_specs, d_specs), axis=0)
    else:
        return c_specs

# handle Pool not being a context manager in Python < 3.4
@contextlib.contextmanager
def create_pool(*args, **kwargs):
    if sys.version_info < (3, 4):
        pool = multiprocessing.Pool(*args, **kwargs)
        yield pool
        pool.terminate()
    else:
        if 'fork' not in multiprocessing.get_all_start_methods():
            warnings.warn("'fork' is not available as a multiprocessing start method, "
                          + "EnergyFlow multicore functionality may not work properly")
            with multiprocessing.Pool(*args, **kwargs) as pool:
                yield pool
        else:
            with multiprocessing.get_context('fork').Pool(*args, **kwargs) as pool:
                yield pool

# applies comprison comp of obj on val
def explicit_comp(obj, comp, val):
    return getattr(obj, COMP_MAP[comp])(val)

# determine if fastjet can be imported, returns either the fastjet module or false
def import_fastjet():
    try:
        import fastjet
    except:
        fastjet = False
    return fastjet

# return argument if iterable else make repeat generator
def iter_or_rep(arg):
    if isinstance(arg, (tuple, list)):
        if len(arg) == 1:
            return repeat(arg[0])
        else:
            return arg
    elif isinstance(arg, repeat):
        return arg
    else:
        return repeat(arg)

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

# check that an argument is well-formed to EFPSet.sel
def sel_arg_check(arg):
    return (isinstance(arg, six.string_types) or 
            (len(arg) == 2 and isinstance(arg[0], six.string_types)))

# timing meta-decorator
def timing(obj, func):
    @wraps(func)
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
