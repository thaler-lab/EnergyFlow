from __future__ import absolute_import

from functools import wraps
from operator import itemgetter
import timeit

import numpy as np

__all__ = [
    'kwargs_check', 
    'explicit_comp', 
    'concat_specs', 
    'transfer', 
    'nonegen',
    'none2inf',
    'timing',
    'pts_from_p4s',
    'p4s_from_ptyphis',
    'thetas2_from_p4s',
    'thetas2_from_yphis',
    'pf_func',
    'kappa_func',
    'flat_metric',
    'find_subslice',
    'find_minimum_rl'
]

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

def nonegen():
    while True:
        yield None

def none2inf(x):
    return np.inf if x is None else x

# timing meta-decorator
def timing(obj, repeat, number):
    def decorator(func):
        @wraps(func)
        def decorated(*args, **kwargs):
            def test():
                func(*args, **kwargs)
            obj.times = timeit.repeat(test, repeat=repeat, number=number)
            return func(*args, **kwargs)
        return decorated
    return decorator

def pts_from_p4s(p4s):
    return np.sqrt(p4s[:,1]**2 + p4s[:,2]**2)

def p4s_from_ptyphis(ptyphis):
    pts, ys, phis = ptyphis[:,0], ptyphis[:,1], ptyphis[:,2]
    return (pts*np.vstack([np.cosh(ys), np.cos(phis), np.sin(phis), np.sinh(ys)])).T

def thetas2_from_p4s(p4s):
    yphis = np.vstack([0.5*np.log((p4s[:,0]+p4s[:,3])/(p4s[:,0]-p4s[:,3])),
                       np.arctan2(p4s[:,2], p4s[:,1])]).T
    return thetas2_from_yphis(yphis)

def thetas2_from_yphis(yphis):
    X = yphis[:,np.newaxis] - yphis[np.newaxis,:]
    X[:,:,0] **= 2
    X[:,:,1] = (np.pi - np.abs(np.abs(X[:,:,1]) - np.pi))**2
    return X[:,:,0] + X[:,:,1]

def pf_func(Es, ps, kappa):
    return np.ones(Es.shape), ps

def kappa_func(Es, ps, kappa):
    return Es**kappa, ps/Es[:,np.newaxis]

long_metric = np.array([1.] + [-1.]*100)
def flat_metric(dim):
    if dim <= 101:
        return long_metric[:dim]
    return np.asarray([1.] + [-1.]*(dim-1))

def find_subslice(sig, efms):
    nlow, nup = sig
    bsigs = list(filter(lambda x: x[0] >= nlow and x[1] >= nup, efms))
    if not len(bsigs):
        return
    return min(bsigs, key=sum)

def find_minimum_rl(sig, efms):
    v = sum(sig)
    vsigs = list(filter(lambda x: sum(x) == v, efms))
    ninds = [(abs(sig[0]-vs[0]), vs) for vs in vsigs]
    return min(ninds, key=itemgetter(0))[1]
