"""Helper functions related to igraph."""

from __future__ import absolute_import
import sys

__all__ = ['igraph_import']

def igraph_import(warn=False, file=None):

    """
    Determines if igraph can be imported. 

    Parameters
    ----------
    warn : bool, optional
        Controls whether or not a warning is printed if igraph cannot be imported.
    file: string, optional
        Filename to use in the optional warning.

    Returns
    -------
    output : {igraph, False}
        The igraph module if it was successfully imported, otherwise False.
    """

    try:
        import igraph
    except:
        igraph = False
        if warn:
            ending = '' if not file else 'from {}'.format(file)
            sys.stderr.write('WARNING: could not import igraph {}\n'.format(ending))
            sys.stderr.flush()
    return igraph

# this file may eventually contain native implementations of igraph functions we use for ve
