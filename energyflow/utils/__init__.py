from __future__ import absolute_import

from functools import wraps
import os
import time

import numpy as np

from . import events
from . import graph
from . import particles

from .events import *
from .graph import *
from .particles import *

# concatenates con. and disc. specs along axis 0, handling empty disc. specs
def concat_specs(c_specs, d_specs):
    if len(d_specs):
        return np.concatenate((c_specs, d_specs), axis=0)
    else:
        return c_specs

# timing meta-decorator
def timing(obj, repeat, number):
    def decorator(func):
        @wraps(func)
        def decorated(*args, **kwargs):
            def test():
                func(*args, **kwargs)
            obj.times.append(timeit.repeat(test, repeat=repeat, number=number))
            return func(*args, **kwargs)
        return decorated
    return decorator

# transfers attrs from obj2 (dict or object) to obj1
def transfer(obj1, obj2, attrs):
    if isinstance(obj2, dict):
        for attr in attrs:
            setattr(obj1, attr, obj2[attr])
    else:
        for attr in attrs:
            setattr(obj1, attr, getattr(obj2, attr))

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

# get access to the data directory of the installed package and the default efp file
ef_data_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
default_efp_file = os.path.join(ef_data_dir, 'efps_d_le_10.npz')
default_M_thresh_file = os.path.join(ef_data_dir, 'M_threshs_d_le_10.npy')

# only include events functions in top level module
__all__ = events.__all__ + particles.__all__
