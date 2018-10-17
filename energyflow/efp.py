"""Energy Flow Polynomials (EFPs) are a set of observables, indexed by non-isomorphic 
multigraphs, which linearly span the space of infrared and collinear safe (IRC-safe) 
observables.

An EFP, indexed by a multigraph $G$, takes the following form:
$$\\text{EFP}_G=\\sum_{i_1=1}^M\\cdots\\sum_{i_N=1}^Mz_{i_1}\\cdots z_{i_N}
\\prod_{(k,\\ell)\\in G}\\theta_{i_ki_\\ell}$$
where $z_i$ is a measure of the energy of particle $i$ and $\\theta_{ij}$ is a measure 
of the angular separation between particles $i$ and $j$. The specific choices for "energy"
and "angular" measure depend on the collider context and are discussed in the 
[Measures](../measures) section.
"""
from __future__ import absolute_import, division, print_function

from  itertools import chain, repeat
import re
import warnings

import numpy as np

from energyflow.algorithms import VariableElimination, einsum_path
from energyflow.efpbase import EFPBase, EFPElem
from energyflow.gen import Generator
from energyflow.utils import concat_specs, default_efp_file
from energyflow.utils.graph_utils import graph_union

__all__ = ['EFP', 'EFPSet']

###############################################################################
# EFP helpers
###############################################################################

comp_map = {
    '>':  '__gt__', 
    '<':  '__lt__', 
    '>=': '__ge__', 
    '<=': '__le__',
    '==': '__eq__', 
    '!=': '__ne__'
}

# applies comprison comp of obj on val
def explicit_comp(obj, comp, val):
    return getattr(obj, comp_map[comp])(val)

# raises TypeError if unexpected keyword left in kwargs
def kwargs_check(name, kwargs, allowed=[]):
    for k in kwargs:
        if k in allowed:
            continue
        raise TypeError(name + '() got an unexpected keyword argument \'{}\''.format(k))

_sel_re = re.compile('(\w+)(<|>|==|!=|<=|>=)(\d+)$')


###############################################################################
# EFP
###############################################################################
class EFP(EFPBase):

    """A class for representing and computing a single EFP. Note that all
    keyword arguments are stored as properties of the `EFP` instance.
    """

    def __init__(self, edges, measure='hadr', beta=1, kappa=1, normed=True, coords=None, 
                              check_input=True, np_optimize='greedy'):
        """
        **Arguments**

        - **edges** : _list_
            - Edges of the EFP graph specified by pairs of vertices.
        - **measure** : {`'hadr'`, `'hadrdot'`, `'ee'`}
            - The choice of measure. See [Measures](../measures) 
            for additional info.
        - **beta** : _float_
            - The parameter $\\beta$ appearing in the measure.
            Must be greater than zero.
        - **kappa** : {_float_, `'pf'`}
            - If a number, the energy weighting parameter $\\kappa$.
            If `'pf'`, use $\\kappa=v-1$ where $v$ is the valency of the vertex.
        - **normed** : _bool_
            - Controls normalization of the energies in the measure.
        - **coords** : {`'ptyphim'`, `'epxpypz'`, `None`}
            - Controls which coordinates are assumed for the input. See 
            [Measures](../measures) for additional info.
        - **check_input** : _bool_
            - Whether to check the type of the input each time or assume
            the first input type.
        - **np_optimize** : {`True`, `False`, `'greedy'`, `'optimal'`}
            - The `optimize` keyword of `numpy.einsum_path`.
        """

        # initialize EFPBase
        super(EFP, self).__init__(measure, beta, kappa, normed, coords, check_input)

        # store these edges as an EFPElem
        self.efpelem = EFPElem(edges)
        self._np_optimize = np_optimize

        # setup ve for standard efp compute
        (self.efpelem.einstr, 
         self.efpelem.einpath, 
         self.chi) = VariableElimination(np_optimize).einspecs(self.simple_graph, self.n)

            
    #===============
    # public methods
    #===============

    # compute(event=None, zs=None, thetas=None)
    def compute(self, event=None, zs=None, thetas=None, batch_call=None):
        """Computes the value of the EFP on a single event.

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

        **Returns**

        - _float_
            - The EFP value.
        """

        zs, thetas_dict = self.get_zs_thetas_dict(event, zs, thetas)
        return self.efpelem.compute(zs, thetas_dict)

    #===========
    # properties
    #===========

    @property
    def _weight_set(self):
        """Set of edge weights for the graph of this EFP."""

        return self.efpelem.weight_set

    @property
    def _einstr(self):
        """Einstein summation string for the EFP computation."""

        return self.efpelem.einstr

    @property
    def _einpath(self):
        """Numpy einsum path specification for EFP computation."""

        return self.efpelem.einpath

    @property
    def np_optimize(self):
        """The np_optimize keyword argument that initialized this EFP instance."""

        return self._np_optimize

    @property
    def graph(self):
        """Graph of this EFP represented by a list of edges."""

        return self.efpelem.edges

    @property
    def simple_graph(self):
        """Simple graph of this EFP (forgetting all multiedges)
        represented by a list of edges."""

        return self.efpelem.simple_edges

    @property
    def n(self):
        """Number of vertices in the graph of this EFP."""

        return self.efpelem.n

    @property
    def d(self):
        """Degree, or number of edges, in the graph of this EFP."""

        return self.efpelem.d

    @property
    def e(self):
        """Number of edges in the simple graph of this EFP."""

        return self.efpelem.e

    @property
    def c(self):
        """VE complexity $\\chi$ of this EFP."""

        return self.chi


###############################################################################
# EFPSet
###############################################################################
class EFPSet(EFPBase):

    """A class that holds a collection of EFPs and computes their values on events.
    Note that all keyword arguments are stored as properties of the `EFPSet` instance.
    """

    # EFPSet(*args, filename=None, measure='hadr', beta=1, kappa=1, normed=True, 
    #        coords='ptyphim', check_input=True, verbose=False)
    def __init__(self, *args, **kwargs):
        """
        EFPSet can be initialized in one of three ways (in order of precedence):

        1. **Default** - Use the ($d\le10$) EFPs that come installed with the
        `EnergFlow` package.
        2. **Generator** - Pass in a custom `Generator` object as the
        first positional argument.
        3. **Custom File** - Pass in the name of a `.npz` file saved
        with a custom `Generator`.

        To control which EFPs are included, `EFPSet` accepts an arbitrary
        number of specifications (see [`sel`](#sel)) and only EFPs meeting each
        specification are included in the set.

        **Arguments**

        - ***args** : _arbitrary positional arguments_
            - If the first positional argument is a `Generator` instance,
            it is used for initialization. The remaining positional
            arguments must be valid arguments to `sel`.
        - **filename** : _string_
            - Path to a `.npz` file which has been saved by a valid
            `energyflow.Generator`.
        - **measure** : {`'hadr'`, `'hadr-dot'`, `'ee'`}
            - See [Measures](../measures) for additional info.
        - **beta** : _float_
            - The parameter $\\beta$ appearing in the measure.
            Must be greater than zero.
        - **kappa** : {_float_, `'pf'`}
            - If a number, the energy weighting parameter $\\kappa$.
            If `'pf'`, use $\\kappa=v-1$ where $v$ is the valency of the vertex.
        - **normed** : _bool_
            - Controls normalization of the energies in the measure.
        - **check_type** : _bool_
            - Whether to check the type of the input each time or use
            the first input type.
        - **verbose** : _bool_
            - Controls printed output when initializing EFPSet.
        """

        default_kwargs = {'filename': None,
                          'measure': 'hadr',
                          'beta': 1,
                          'kappa': 1,
                          'normed': True,
                          'coords': None,
                          'check_input': True,
                          'verbose': False}
        measure_kwargs = ['measure', 'beta', 'kappa', 'normed', 'coords', 'check_input']

        # process arguments
        for k,v in default_kwargs.items():
            if k not in kwargs:
                kwargs[k] = v
            if k not in measure_kwargs:
                setattr(self, k, kwargs.pop(k))

        kwargs_check('__init__', kwargs, allowed=measure_kwargs)

        # initialize EFPBase
        super(EFPSet, self).__init__(*[kwargs[k] for k in measure_kwargs])

        # handle different methods of initialization
        maxs = ['nmax', 'emax', 'dmax', 'cmax', 'vmax', 'comp_dmaxs']
        elemvs = ['edges', 'weights', 'einstrs', 'einpaths']
        if len(args) >= 1 and isinstance(args[0], Generator):
            constructor_attrs = maxs + elemvs + ['cols', 'c_specs', 'disc_specs', 'disc_formulae']
            gen = {attr: getattr(args[0], attr) for attr in constructor_attrs}
            args = args[1:]
        elif self.filename is not None:
            self.filename += '.npz' if not self.filename.endswith('.npz') else ''
            gen = np.load(self.filename)
        else:
            gen = np.load(default_efp_file)

        # compile regular expression for use in sel()
        self._sel_re = _sel_re
        
        # put column headers and indices into namespace
        self._cols = gen['cols']
        self._set_col_inds()

        # put gen maxs into dict
        self.gen_maxs = {m: gen[m] for m in maxs}

        # get disc formulae and disc mask
        orig_disc_specs = gen['disc_specs']
        disc_mask = self.sel(*args, specs=orig_disc_specs)
        self.disc_formulae = gen['disc_formulae'][disc_mask]

        # get connected specs and full specs
        orig_c_specs = gen['c_specs']
        c_mask = self.sel(*args, specs=orig_c_specs)
        self._cspecs = orig_c_specs[c_mask]
        self._specs = concat_specs(self._cspecs, orig_disc_specs[disc_mask])

        # make EFPElem list
        z = zip(*([gen[v] for v in elemvs] + [orig_c_specs[:,self.k_ind]]))
        self.efpelems = [EFPElem(*args) for m,args in enumerate(z) if c_mask[m]]

        # union over all weights needed
        self.__weight_set = frozenset(w for efpelem in self.efpelems for w in efpelem.weight_set)

        # get col indices for disconnected formulae
        connected_ndk = {efpelem.ndk: i for i,efpelem in enumerate(self.efpelems)}
        self.disc_col_inds = []
        for formula in self.disc_formulae:
            try:
                self.disc_col_inds.append([connected_ndk[factor] for factor in formula])
            except KeyError:
                warnings.warn('connected efp needed for {} not found'.format(formula))

        # handle printing
        if self.verbose:
            print('Originally Available EFPs:')
            self.print_stats(specs=concat_specs(orig_c_specs, orig_disc_specs), lws=2)
            if len(args) > 0:
                print('Current Stored EFPs:')
                self.print_stats(lws=2)

    #================
    # PRIVATE METHODS
    #================

    def _set_col_inds(self):
        self.__dict__.update({col+'_ind': i for i,col in enumerate(self.cols)})

    def _make_graphs(self, connected_graphs):
        disc_comps = [[connected_graphs[i] for i in col_inds] for col_inds in self.disc_col_inds]
        return np.asarray(connected_graphs + [graph_union(*dc) for dc in disc_comps])

    #===============
    # PUBLIC METHODS
    #===============

    # calc_disc(X)
    def calc_disc(self, X):
        """Computes disconnected EFPs according to the internal 
        specifications using the connected EFPs provided as input.

        **Arguments**

        - **X** : _numpy.ndarray_
            - Array of connected EFPs. Rows are different events, columns 
            are the different EFPs. Can handle a single event (a 1-dim array) 
            as input. EFPs are assumed to be in the order expected by the 
            instance of `EFPSet`; the safest way to ensure this is to use 
            the same `EFPSet` to calculate both connected and disconnected 
            EFPs. This function is used internally in `compute` and 
            `batch_compute`.

        **Returns**

        - _numpy.ndarray_
            - A concatenated array of the connected and disconnected EFPs.
        """

        if len(self.disc_col_inds) == 0:
            return np.asarray(X)

        X = np.atleast_2d(X)

        results = np.empty((len(X), len(self.disc_col_inds)), dtype=float)
        for i,formula in enumerate(self.disc_col_inds):
            results[:,i] = np.prod(X[:,formula], axis=1)

        return np.squeeze(np.concatenate((X, results), axis=1))

    # compute(event=None, zs=None, thetas=None)
    def compute(self, event=None, zs=None, thetas=None, batch_call=False):
        """Computes the values of the stored EFPs on a single event.

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

        **Returns**

        - _1-d numpy.ndarray_
            - A vector of the EFP values.
        """
        
        zs, thetas_dict = self.get_zs_thetas_dict(event, zs, thetas)
        results = [efpelem.compute(zs, thetas_dict) for efpelem in self.efpelems]

        if batch_call:
            return results
        else:
            return self.calc_disc(results)

    def batch_compute(self, events, n_jobs=-1):
        """Computes the value of the stored EFPs on several events.

        **Arguments**

        - **events** : array_like or `fastjet.PseudoJet`
            - The events as an array of arrays of particles in coordinates
            matching those anticipated by `coords`.
        - **n_jobs** : _int_ 
            - The number of worker processes to use. A value of `-1` will attempt
            to use as many processes as there are CPUs on the machine.

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
            - Each argument can be either a string or a length-two 
            iterable. If the argument is a string, it should consist 
            of three parts: a character which is a valid element of 
            `cols`, a comparison operator (one of `<`, `>`, `<=`, 
            `>=`, `==`, `!=`), and a number. Whitespace between the 
            parts does not matter. If the argument is a tuple, the 
            first element should be a string containing a column 
            header character and a comparison operator; the second 
            element is the value to be compared. The tuple version 
            is useful when the value is a variable that changes 
            (such as in a list comprehension).

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
            if isinstance(arg, str):
                s = arg.replace(' ', '')
            elif hasattr(arg, '__getitem__'):
                if len(arg) == 2:
                    s = arg[0].replace(' ', '') + str(arg[1])
                else:
                    raise ValueError('{} is not length 2'.format(arg))
            else:
                raise TypeError('invalid type for {}'.format(arg))

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
            mask &= explicit_comp(specs[:,getattr(self, var+'_ind')], comp, int(val))
            
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
            - An array of graphs (as lists of edges) matching the specifications.
        """

        # if we haven't extracted the graphs, do it now
        if not hasattr(self, '_graphs'):
            self._graphs = self._make_graphs([elem.edges for elem in self.efpelems])

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
            - An array of simple graphs (as lists of edges) matching the specifications.
        """

        # is we haven't extracted the simple graphs, do it now
        if not hasattr(self, '_simple_graphs'):
            self._simple_graphs = self._make_graphs([elem.simple_edges for elem in self.efpelems])

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
        for efpelem in self.efpelems:
            efpelem.set_timer()

    def get_times(self):
        efp_times = np.asarray([elem.times for elem in self.efpelems])
        return efp_times

    #===========
    # properties
    #===========

    @property
    def _weight_set(self):
        return self.__weight_set

    @property
    def cols(self):
        """Column labels for `specs`. 
        Those of primary interest are listed below.

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

    @property
    def specs(self):
        """An array of EFP specifications. Each row represents an EFP 
        and the columns represent the quantities indicated by `cols`."""

        return self._specs

    @property
    def cspecs(self):
        """Specification array for connected EFPs."""

        return self._cspecs
