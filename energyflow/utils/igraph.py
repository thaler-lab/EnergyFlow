from __future__ import absolute_import
import sys

__all__ = ['igraph_import']

def igraph_import(file, warn=False):
    try:
        import igraph
    except:
        igraph = False
        if warn:
            sys.stderr.write('WARNING: could not import igraph from {}\n'.format(file))
            sys.stderr.flush()
    return igraph

# eventually we may support everything without igraph (except generation)
