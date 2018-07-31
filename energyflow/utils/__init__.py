from __future__ import absolute_import

from functools import wraps
from itertools import repeat
import os
import sys
import time

import numpy as np

from . import events
from . import graph
from . import particles

from .events import *
from .graph import *
from .particles import *

__all__ = events.__all__ + particles.__all__

py_version = sys.version_info[:2]
sysname = os.uname()[0]

# get access to the data directory of the installed package and the default efp file
ef_data_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
default_efp_file = os.path.join(ef_data_dir, 'efps_d_le_10.npz')

# handle pickling methods in python 2
if py_version[0] == 2:
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

# return argument if iterable else make repeat generator
def iter_or_rep(arg):
    if hasattr(arg, '__getitem__'):
        if len(arg) == 1:
            return repeat(arg[0])
        else:
            return arg
    else:
        return repeat(arg)

# check for sklearn
def sklearn_import():
    try:
        import sklearn
    except:
        sklearn = False
    return sklearn

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
