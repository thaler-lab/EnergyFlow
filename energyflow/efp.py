r"""# Energy Flow Polynomials

Energy Flow Polynomials (EFPs) are a set of observables, indexed by
non-isomorphic multigraphs, which linearly span the space of infrared and
collinear (IRC) safe observables.

An EFP, indexed by a multigraph $G$, takes the following form:

\[\text{EFP}_G=\sum_{i_1=1}^M\cdots\sum_{i_N=1}^Mz_{i_1}\cdots z_{i_N}
\prod_{(k,\ell)\in G}\theta_{i_ki_\ell}\]

where $z_i$ is a measure of the energy of particle $i$ and $\theta_{ij}$ is a
measure of the angular separation between particles $i$ and $j$. The specific
choices for "energy" and "angular" measure depend on the collider context and
are discussed in the [Measures](../measures) section.
"""

#  ______ ______ _____
# |  ____|  ____|  __ \
# | |__  | |__  | |__) |
# |  __| |  __| |  ___/
# | |____| |    | |
# |______|_|    |_|

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

from collections import Counter
import itertools
import re
import warnings

import numpy as np
import six

from energyflow.algorithms import VariableElimination, einsum_path, einsum
from energyflow.base import EFPBase
from energyflow.efm import EFMSet, efp2efms
from energyflow.measure import PF_MARKER
from energyflow.utils import (concat_specs, create_pool, explicit_comp,
                              kwargs_check, load_efp_file, sel_arg_check)
from energyflow.utils.graph_utils import *

__all__ = ['EFP', 'EFPSet']

###############################################################################
# EFP
###############################################################################

class EFP(EFPBase):

    """A class for representing and computing a single EFP."""

    # EFP(edges, measure='hadr', beta=1, kappa=1, normed=None, coords=None,
    #            check_input=True, np_optimize=True)
    def __init__(self, edges, weights=None, efpset_args=None, np_optimize=True, **kwargs):
        r"""Since a standalone EFP defines and holds a `Measure` instance, all
        `Measure` keywords are accepted.

        **Arguments**

        - **edges** : _list_
            - Edges of the EFP graph specified by pairs of vertices.
        - **weights** : _list_ of _int_ or `None`
            - If not `None`, the multiplicities of each edge.
        - **measure** : {`'hadr'`, `'hadrdot'`, `'hadrefm'`, `'ee'`, `'eeefm'`}
            - The choice of measure. See [Measures](../measures) for additional
            info.
        - **beta** : _float_
            - The parameter $\beta$ appearing in the measure. Must be greater
            than zero.
        - **kappa** : {_float_, `'pf'`}
            - If a number, the energy weighting parameter $\kappa$. If `'pf'`,
            use $\kappa=v-1$ where $v$ is the valency of the vertex.
        - **normed** : _bool_
            - Controls normalization of the energies in the measure.
        - **coords** : {`'ptyphim'`, `'epxpypz'`, `None`}
            - Controls which coordinates are assumed for the input. See 
            [Measures](../measures) for additional info.
        - **check_input** : _bool_
            - Whether to check the type of the input each time or assume the
            first input type.
        - **np_optimize** : {`True`, `False`, `'greedy'`, `'optimal'`}
            - The `optimize` keyword of `numpy.einsum_path`.
        """

        # initialize base class
        super(EFP, self).__init__(kwargs)

        # store options
        self._np_optimize = np_optimize
        self._weights = weights

        # generate our own information from the edges
        if efpset_args is not None:
            (self._einstr, self._einpath, self._spec, self._efm_einstr,
             self._efm_einpath, self._efm_spec) = efpset_args

            # ensure that EFM spec is a list of tuples
            if self.efm_spec is not None:
                self._efm_spec = list(map(tuple, self.efm_spec))

        # process edges
        self._process_edges(edges, self.weights)

        # compute specs if needed
        if efpset_args is None:

            # compute EFM specs
            self._efm_einstr, self._efm_spec = efp2efms(self.graph)

            # only store an EFMSet if this is an external EFP using EFMs
            if self.has_measure and self.use_efms:
                self._efmset = EFMSet(self._efm_spec, subslicing=self.subslicing, no_measure=True)
            args = [np.empty([4]*sum(s)) for s in self._efm_spec]
            self._efm_einpath = einsum_path(self._efm_einstr, *args, optimize=np_optimize)[0]
            
            # setup traditional VE computation
            ve = VariableElimination(self.np_optimize)
            (self._einstr, self._einpath, self._c) = ve.einspecs(self.simple_graph, self.n)

            # compute and store spec information
            vs = valencies(self.graph).values()
            self._e = len(self.simple_graph)
            self._d = sum(self.weights)
            self._v = max(vs) if len(vs) else 0
            self._k = -1
            self._p = len(get_components(self.simple_graph)) if self.d > 0 else 1
            self._h = Counter(vs)[1]
            self._spec = np.array([self.n, self.e, self.d, self.v, self.k, self.c, self.p, self.h])

        # store properties from given spec
        else:
            self._e, self._d, self._v, self._k, self._c, self._p, self._h = self.spec[1:]
            assert self.n == self.spec[0], 'n from spec does not match internally computed n'

    #================
    # PRIVATE METHODS
    #================

    def _process_edges(self, edges, weights):

        # deal with arbitrary vertex labels
        vertex_set = frozenset(v for edge in edges for v in edge)
        vertices = {v: i for i,v in enumerate(vertex_set)}
        
        # determine number of vertices, empty edges are interpretted as graph with one vertex
        self._n = len(vertices) if len(vertices) > 0 else 1

        # construct new edges with remapped vertices
        self._edges = [tuple(vertices[v] for v in sorted(edge)) for edge in edges]

        # handle weights
        if weights is None:
            self._simple_edges = list(frozenset(self._edges))
            counts = Counter(self._edges)
            self._weights = tuple(counts[edge] for edge in self._simple_edges)

            # invalidate einsum quantities because edges got reordered
            self._einstr = self._einpath = None

        else:
            if len(weights) != len(self._edges):
                raise ValueError('length of weights is not number of edges')
            self._simple_edges = self._edges
            self._weights = tuple(weights)

        self._edges = [e for w,e in zip(self._weights, self._simple_edges) for i in range(w)]
        self._weight_set = frozenset(self._weights)

    def _efp_compute(self, zs, thetas_dict):
        einsum_args = [thetas_dict[w] for w in self.weights] + self._n*[zs]
        return einsum(self.einstr, *einsum_args, optimize=self.einpath)

    def _efm_compute(self, efms_dict):
        einsum_args = [efms_dict[sig] for sig in self.efm_spec]
        return einsum(self.efm_einstr, *einsum_args, optimize=self.efm_einpath)

    #===============
    # PUBLIC METHODS
    #===============

    # compute(event=None, zs=None, thetas=None, nhats=None)
    def compute(self, event=None, zs=None, thetas=None, nhats=None, batch_call=None):
        """Computes the value of the EFP on a single event. Note that `EFP`
        also is callable, in which case this method is invoked.

        **Arguments**

        - **event** : 2-d array_like or `fastjet.PseudoJet`
            - The event as an array of particles in the coordinates specified
            by `coords`.
        - **zs** : 1-d array_like
            - If present, `thetas` must also be present, and `zs` is used in place 
            of the energies of an event.
        - **thetas** : 2-d array_like
            - If present, `zs` must also be present, and `thetas` is used in place 
            of the pairwise angles of an event.
        - **nhats** : 2-d array like
            - If present, `zs` must also be present, and `nhats` is used in place
            of the scaled particle momenta. Only applicable when EFMs are being
            used.

        **Returns**

        - _float_
            - The EFP value.
        """

        if self.use_efms:
            return self._efm_compute(self.compute_efms(event, zs, nhats))
        else:
            return self._efp_compute(*self.get_zs_thetas_dict(event, zs, thetas))

    #===========
    # PROPERTIES
    #===========

    @property
    def graph(self):
        """Graph of this EFP represented by a list of edges."""

        return self._edges

    @property
    def simple_graph(self):
        """Simple graph of this EFP (forgetting all multiedges)
        represented by a list of edges."""

        return self._simple_edges

    @property
    def weights(self):
        """Edge weights (counts) for the graph of this EFP."""

        return self._weights

    @property
    def weight_set(self):
        """Set of edge weights (counts) for the graph of this EFP."""

        return self._weight_set

    @property
    def einstr(self):
        """Einstein summation string for the EFP computation."""

        return self._einstr

    @property
    def einpath(self):
        """NumPy einsum path specification for EFP computation."""

        return self._einpath

    @property
    def efm_spec(self):
        """List of EFM signatures corresponding to efm_einstr."""

        return self._efm_spec

    @property
    def efm_einstr(self):
        """Einstein summation string for the EFM computation."""

        return self._efm_einstr

    @property
    def efm_einpath(self):
        """NumPy einsum path specification for EFM computation."""

        return self._efm_einpath

    @property
    def efmset(self):
        """Instance of `EFMSet` help by this EFP if using EFMs."""

        return self._efmset if (self.has_measure and self.use_efms) else None

    @property
    def np_optimize(self):
        """The np_optimize keyword argument that initialized this EFP instance."""

        return self._np_optimize

    @property
    def n(self):
        """Number of vertices in the graph of this EFP."""

        return self._n

    @property
    def e(self):
        """Number of edges in the simple graph of this EFP."""

        return self._e

    @property
    def d(self):
        """Degree, or number of edges, in the graph of this EFP."""

        return self._d

    @property
    def v(self):
        """Maximum valency of any vertex in the graph."""

        return self._v

    @property
    def k(self):
        r"""Index of this EFP. Determined by EFPSet or -1 otherwise."""

        return self._k

    @property
    def c(self):
        r"""VE complexity $\chi$ of this EFP."""

        return self._c

    @property
    def p(self):
        """Number of connected components of this EFP. Note that the empty
        graph conventionally has one connected component."""

        return self._p

    @property
    def h(self):
        """Number of valency 1 vertices ('hanging chads) of this EFP."""

        return self._h

    @property
    def spec(self):
        """Specification array for this EFP."""

        return self._spec

    @property
    def ndk(self):
        """Tuple of `n`, `d`, and `k` values which form a unique identifier of
        this EFP within an `EFPSet`."""

        return (self.n, self.d, self.k)

###############################################################################
# EFPSet
###############################################################################

EFP_FILE_INFO = None
class EFPSet(EFPBase):

    """A class that holds a collection of EFPs and computes their values on
    events. Note that all keyword arguments are stored as properties of the
    `EFPSet` instance.
    """

    # EFPSet(*args, filename=None, measure='hadr', beta=1, kappa=1, normed=None, 
    #               coords=None, check_input=True, verbose=0)
    def __init__(self, *args, **kwargs):
        r"""`EFPSet` can be initialized in one of three ways (in order of
        precedence):

        1. **Graphs** - Pass in graphs as lists of edges, just as for
        individual EFPs.
        2. **Generator** - Pass in a custom `Generator` object as the first
        positional argument.
        3. **Custom File** - Pass in the name of a `.npz` file saved with a
        custom `Generator`.
        4. **Default** - Use the $d\le10$ EFPs that come installed with the
        `EnergFlow` package.

        To control which EFPs are included, `EFPSet` accepts an arbitrary
        number of specifications (see [`sel`](#sel)) and only EFPs meeting each
        specification are included in the set. Note that no specifications
        should be passed in when initializing from explicit graphs.

        Since an EFP defines and holds a `Measure` instance, all `Measure`
        keywords are accepted.

        **Arguments**

        - ***args** : _arbitrary positional arguments_
            - Depending on the method of initialization, these can be either
            1) graphs to store, as lists of edges 2) a Generator instance
            followed by some number of valid arguments to `sel` or 3,4) valid
            arguments to `sel`. When passing in specific graphs, no arguments
            to `sel` should be given.
        - **filename** : _string_
            - Path to a `.npz` file which has been saved by a valid
            `energyflow.Generator`. A value of `None` will use the provided
            graphs, if a file is needed at all.
        - **measure** : {`'hadr'`, `'hadr-dot'`, `'ee'`}
            - See [Measures](../measures) for additional info.
        - **beta** : _float_
            - The parameter $\beta$ appearing in the measure. Must be greater
            than zero.
        - **kappa** : {_float_, `'pf'`}
            - If a number, the energy weighting parameter $\kappa$. If `'pf'`,
            use $\kappa=v-1$ where $v$ is the valency of the vertex.
        - **normed** : _bool_
            - Controls normalization of the energies in the measure.
        - **coords** : {`'ptyphim'`, `'epxpypz'`, `None`}
            - Controls which coordinates are assumed for the input. See 
            [Measures](../measures) for additional info.
        - **check_input** : _bool_
            - Whether to check the type of the input each time or assume the
            first input type.
        - **verbose** : _int_
            - Controls printed output when initializing `EFPSet` from a file or
            `Generator`.
        """

        # process arguments
        for k,v in {'filename': None, 'verbose': 0}.items():
            if k not in kwargs:
                kwargs[k] = v
            setattr(self, k, kwargs.pop(k))

        # initialize EFPBase
        super(EFPSet, self).__init__(kwargs)

        # handle different methods of initialization
        maxs = ['nmax', 'emax', 'dmax', 'cmax', 'vmax', 'comp_dmaxs']
        elemvs = ['edges', 'weights', 'einstrs', 'einpaths']
        efmvs = ['efm_einstrs', 'efm_einpaths', 'efm_specs']
        miscattrs = ['cols', 'gen_efms', 'c_specs', 'disc_specs', 'disc_formulae']
        if len(args) >= 1 and not sel_arg_check(args[0]) and not isinstance(args[0], Generator):
            gen = False
        elif len(args) >= 1 and isinstance(args[0], Generator):
            constructor_attrs = maxs + elemvs + efmvs + miscattrs
            gen = {attr: getattr(args[0], attr) for attr in constructor_attrs}
            args = args[1:]
        else:
            global EFP_FILE_INFO
            if EFP_FILE_INFO is None:
                EFP_FILE_INFO = load_efp_file(self.filename)
            gen = EFP_FILE_INFO

        # compiled regular expression for use in sel()
        self._sel_re = re.compile(r'(\w+)(<|>|==|!=|<=|>=)(\d+)$')
        self._cols = np.array(['n', 'e', 'd', 'v', 'k', 'c', 'p', 'h'])
        self.__dict__.update({col+'_ind': i for i,col in enumerate(self._cols)})

        # initialize from given graphs
        if not gen:
            self._disc_col_inds = None
            self._efps = [EFP(graph, no_measure=True) for graph in args]
            self._cspecs = self._specs = np.asarray([efp.spec for efp in self.efps])

        # initialize from a generator
        else:

            # handle not having efm generation
            if not gen['gen_efms'] and self.use_efms:
                raise ValueError('Cannot use efm measure without providing efm generation.')
            
            # verify columns with generator
            assert np.all(self._cols == gen['cols'])

            # get disc formulae and disc mask
            orig_disc_specs = np.asarray(gen['disc_specs'])
            disc_mask = self.sel(*args, specs=orig_disc_specs)
            disc_formulae = np.asarray(gen['disc_formulae'], dtype='O')[disc_mask]

            # get connected specs and full specs
            orig_c_specs = np.asarray(gen['c_specs'])
            c_mask = self.sel(*args, specs=orig_c_specs)
            self._cspecs = orig_c_specs[c_mask]
            self._specs = concat_specs(self._cspecs, orig_disc_specs[disc_mask])

            # make EFP list
            z = zip(*([gen[v] for v in elemvs] + [orig_c_specs] +
                      [gen[v] if self.use_efms else itertools.repeat(None) for v in efmvs]))
            self._efps = [EFP(args[0], weights=args[1], no_measure=True, efpset_args=args[2:]) 
                          for m,args in enumerate(z) if c_mask[m]]

            # get col indices for disconnected formulae
            connected_ndk = {efp.ndk: i for i,efp in enumerate(self.efps)}
            self._disc_col_inds = []
            for formula in disc_formulae:
                try:
                    self._disc_col_inds.append([connected_ndk[tuple(factor)] for factor in formula])
                except KeyError:
                    warnings.warn('connected efp needed for {} not found'.format(formula))

            # handle printing
            if self.verbose > 0:
                print('Originally Available EFPs:')
                self.print_stats(specs=concat_specs(orig_c_specs, orig_disc_specs), lws=2)
                if len(args) > 0:
                    print('Current Stored EFPs:')
                    self.print_stats(lws=2)

        # setup EFMs
        if self.use_efms:
            efm_specs = set(itertools.chain(*[efp.efm_spec for efp in self.efps]))
            self._efmset = EFMSet(efm_specs, subslicing=self.subslicing)

        # union over all weights needed
        self._weight_set = frozenset(w for efp in self.efps for w in efp.weight_set)

    #================
    # PRIVATE METHODS
    #================

    def _make_graphs(self, connected_graphs):
        disc_comps = [[connected_graphs[i] for i in col_inds] for col_inds in self._disc_col_inds]
        return np.asarray(connected_graphs + [graph_union(*dc) for dc in disc_comps], dtype='O')

    #===============
    # PUBLIC METHODS
    #===============

    def calc_disc(self, X):
        """Computes disconnected EFPs according to the internal 
        specifications using the connected EFPs provided as input. Note that
        this function has no effect if the `EFPSet` was initialized with
        specific graphs.

        **Arguments**

        - **X** : _numpy.ndarray_
            - Array of connected EFPs. Rows are different events, columns are
            the different EFPs. Can handle a single event (a 1-dim array) as
            input. EFPs are assumed to be in the order expected by the instance
            of `EFPSet`; the safest way to ensure this is to use the same
            `EFPSet` to calculate both connected and disconnected EFPs. This
            function is used internally in `compute` and `batch_compute`.

        **Returns**

        - _numpy.ndarray_
            - A concatenated array of the connected and disconnected EFPs.
        """

        if self._disc_col_inds is None or len(self._disc_col_inds) == 0:
            return np.asarray(X)

        X = np.atleast_2d(X)

        results = np.empty((len(X), len(self._disc_col_inds)), dtype=float)
        for i,formula in enumerate(self._disc_col_inds):
            results[:,i] = np.prod(X[:,formula], axis=1)

        return np.squeeze(np.concatenate((X, results), axis=1))

    # compute(event=None, zs=None, thetas=None, nhats=None)
    def compute(self, event=None, zs=None, thetas=None, nhats=None, batch_call=False):
        """Computes the values of the stored EFPs on a single event. Note that
        `EFPSet` also is callable, in which case this method is invoked.

        **Arguments**

        - **event** : 2-d array_like or `fastjet.PseudoJet`
            - The event as an array of particles in the coordinates specified
            by `coords`.
        - **zs** : 1-d array_like
            - If present, `thetas` must also be present, and `zs` is used in place 
            of the energies of an event.
        - **thetas** : 2-d array_like
            - If present, `zs` must also be present, and `thetas` is used in place 
            of the pairwise angles of an event.
        - **nhats** : 2-d array like
            - If present, `zs` must also be present, and `nhats` is used in place
            of the scaled particle momenta. Only applicable when EFMs are being
            used.

        **Returns**

        - _1-d numpy.ndarray_
            - A vector of the EFP values.
        """

        if self.use_efms:
            efms_dict = self.compute_efms(event, zs, nhats)
            results = [efp._efm_compute(efms_dict) for efp in self._efps]
        else:
            zs, thetas_dict = self.get_zs_thetas_dict(event, zs, thetas)
            results = [efp._efp_compute(zs, thetas_dict) for efp in self._efps]

        if batch_call:
            return results
        else:
            return self.calc_disc(results)

    def batch_compute(self, events, n_jobs=None):
        """Computes the value of the stored EFPs on several events.

        **Arguments**

        - **events** : array_like or `fastjet.PseudoJet`
            - The events as an array of arrays of particles in coordinates
            matching those anticipated by `coords`.
        - **n_jobs** : _int_ or `None`
            - The number of worker processes to use. A value of `None` will
            attempt to use as many processes as there are CPUs on the machine.

        **Returns**

        - _2-d numpy.ndarray_
            - An array of the EFP values for each event.
        """

        return self.calc_disc(super(EFPSet, self).batch_compute(events, n_jobs))

    # sel(*args)
    def sel(self, *args, **kwargs):
        """Computes a boolean mask of EFPs matching each of the
        specifications provided by the `args`. 

        **Arguments**

        - ***args** : arbitrary positional arguments
            - Each argument can be either a string or a length-two iterable. If
            the argument is a string, it should consist of three parts: a
            character which is a valid element of `cols`, a comparison
            operator (one of `<`, `>`, `<=`, `>=`, `==`, `!=`), and a number.
            Whitespace between the parts does not matter. If the argument is a
            tuple, the first element should be a string containing a column
            header character and a comparison operator; the second element is
            the value to be compared. The tuple version is useful when the
            value is a variable that changes (such as in a list comprehension).

        **Returns**

        - _1-d numpy.ndarray_
            - A boolean array of length the number of EFPs stored by this object. 
        """

        # ensure only valid keyword args are passed
        specs = kwargs.pop('specs', None)
        kwargs_check('sel', kwargs)

        # use default specs if non provided
        if specs is None:
            specs = self.specs

        # iterate through arguments
        mask = np.ones(len(specs), dtype=bool)
        for arg in args:

            # parse arg
            if isinstance(arg, six.string_types):
                s = arg
            elif hasattr(arg, '__getitem__'):
                if len(arg) == 2:
                    s = arg[0] + str(arg[1])
                else:
                    raise ValueError('{} is not length 2'.format(arg))
            else:
                raise TypeError('invalid argument {}'.format(arg))

            s = s.replace(' ', '')

            # match string to pattern
            match = self._sel_re.match(s)
            if match is None:
                raise ValueError('could not understand \'{}\''.format(arg))

            # get the variable of the selection
            var = match.group(1)
            if var not in self.cols:
                raise ValueError('\'{}\' not in {}'.format(var, self.cols))

            # get the comparison and value
            comp, val = match.group(2, 3)

            # AND the selection with mask
            mask &= explicit_comp(specs[:,getattr(self, var + '_ind')], comp, int(val))
            
        return mask

    # csel(*args)
    def csel(self, *args):
        """Same as `sel` except using `cspecs` to select from."""

        return self.sel(*args, specs=self.cspecs)

    # count(*args)
    def count(self, *args, **kwargs):
        """Counts the number of EFPs meeting the specifications
        of the arguments using `sel`.

        **Arguments** 

        - ***args** : arbitrary positional arguments
            - Valid arguments to be passed to `sel`.

        **Returns**

        - _int_
            - The number of EFPs meeting the specifications provided.
        """

        return np.count_nonzero(self.sel(*args, **kwargs))

    # graphs(*args)
    def graphs(self, *args):
        """Graphs meeting provided specifications.

        **Arguments** 

        - ***args** : arbitrary positional arguments
            - Valid arguments to be passed to `sel`, or, if a single integer, 
            the index of a particular graph.

        **Returns**

        - _list_, if single integer argument is given
            - The list of edges corresponding to the specified graph
        - _1-d numpy.ndarray_, otherwise
            - An array of graphs (as lists of edges) matching the
            specifications.
        """

        # if we haven't extracted the graphs, do it now
        if not hasattr(self, '_graphs'):
            if self._disc_col_inds is None:
                self._graphs = np.asarray([efp.graph for efp in self.efps], dtype='O')
            else:
                self._graphs = self._make_graphs([efp.graph for efp in self.efps])

        # handle case of single graph
        if len(args) and isinstance(args[0], int):
            return self._graphs[args[0]]

        # filter graphs based on mask
        return self._graphs[self.sel(*args)]

    # simple_graphs(*args)
    def simple_graphs(self, *args):
        """Simple graphs meeting provided specifications.

        **Arguments** 

        - ***args** : arbitrary positional arguments
            - Valid arguments to be passed to `sel`, or, if a single integer, 
            the index of particular simple graph.

        **Returns**

        - _list_, if single integer argument is given
            - The list of edges corresponding to the specified simple graph
        - _1-d numpy.ndarray_, otherwise
            - An array of simple graphs (as lists of edges) matching the
            specifications.
        """

        # if we haven't extracted the simple graphs, do it now
        if not hasattr(self, '_simple_graphs'):
            if self._disc_col_inds is None:
                self._simple_graphs = np.asarray([efp.simple_graph for efp in self.efps], dtype='O')
            else:
                self._simple_graphs = self._make_graphs([efp.simple_graph for efp in self.efps])

        # handle case of single graph
        if len(args) and isinstance(args[0], int):
            return self._simple_graphs[args[0]]

        # filter simple graphs based on mask
        return self._simple_graphs[self.sel(*args)]

    def print_stats(self, specs=None, lws=0):
        if specs is None:
            specs = self.specs
        num_prime = self.count('p==1', specs=specs)
        num_composite = self.count('p>1', specs=specs)
        pad = ' '*lws
        print(pad + 'Prime:', num_prime)
        print(pad + 'Composite:', num_composite)
        print(pad + 'Total: ', num_prime+num_composite)

    def set_timers(self):
        if self.use_efms:
            self.efmset.set_timers()
        for efpelem in self.efpelems:
            efpelem.set_timer()

    def get_times(self):
        efp_times = np.asarray([elem.times for elem in self.efpelems])
        if self.use_efms:
            return efp_times, self.efmset.get_times()
        return efp_times

    #===========
    # PROPERTIES
    #===========

    @property
    def efps(self):
        """List of EFPs held by the `EFPSet`."""

        return self._efps

    @property
    def efmset(self):
        """The `EFMSet` held by the `EFPSet`, if using EFMs."""

        return self._efmset if self.use_efms else None

    @property
    def specs(self):
        """An array of EFP specifications. Each row represents an EFP 
        and the columns represent the quantities indicated by `cols`."""

        return self._specs

    @property
    def cspecs(self):
        """Specification array for connected EFPs."""

        return self._cspecs

    @property
    def weight_set(self):
        """The union of all weights needed by the EFPs stored by the 
        `EFPSet`."""

        return self._weight_set

    @property
    def cols(self):
        """Column labels for `specs`. Each EFP has a property corresponding to
        each column.

        - `n` : Number of vertices.
        - `e` : Number of simple edges.
        - `d` : Degree, or number of multiedges.
        - `v` : Maximum valency (number of edges touching a vertex).
        - `k` : Unique identifier within EFPs of this (n,d).
        - `c` : VE complexity $\\chi$.
        - `p` : Number of prime factors (or connected components).
        - `h` : Number of valency 1 vertices (a.k.a. 'hanging chads').
        """

        return self._cols

# put gen import here so it succeeds
from energyflow.gen import Generator
