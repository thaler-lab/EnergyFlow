"""
Energy Flow Polynomials (EFPs) are a set of observables, indexed by non-isomorphic 
multigraphs, which linearly span the space of infrared and collinear safe (IRC-safe) 
observables.

An EFP index by a multigraph $G$ takes the following form:
$$\\text{EFP}_G=\\sum_{i_1=1}^M\\cdots\\sum_{i_N=1}^Mz_{i_1}\\cdots z_{i_N}
\\prod_{(k,\\ell)\\in G}\\theta_{i_ki_\\ell}$$
where $z_i$ is a measure of the energy of particle $i$ and $\\theta_{ij}$ is a measure 
of the angular separation between particles $i$ and $j$. The specific choices for energy 
and angular measure depend on the collider context and are discussed at length in the 
[Measures](/docs/measure) section.
"""

from __future__ import absolute_import, division, print_function

from  itertools import chain, repeat
import re
import warnings

import numpy as np

from energyflow.algorithms import VariableElimination
from energyflow.efm import EFMSet, efp2efms
from energyflow.efpbase import *
from energyflow.gen import Generator
from energyflow.utils import concat_specs, default_efp_file
from energyflow.utils.graph import graph_union

__all__ = ['EFP', 'EFPSet']

###############################################################################
# EFP helpers
###############################################################################
comp_map = {'>':  '__gt__', 
            '<':  '__lt__', 
            '>=': '__ge__', 
            '<=': '__le__',
            '==': '__eq__', 
            '!=': '__ne__'}

# applies comprison comp of obj on val
def explicit_comp(obj, comp, val):
    return getattr(obj, comp_map[comp])(val)

# raises TypeError if unexpected keyword left in kwargs
def kwargs_check(name, kwargs, allowed=[]):
    for k in kwargs:
        if k in allowed:
            continue
        raise TypeError(name + '() got an unexpected keyword argument \'{}\''.format(k))


###############################################################################
# EFP
###############################################################################
class EFP(EFPBase):

    """A class for representing and computing a single EFP."""

    def __init__(self, edges, measure='hadrdot', beta=2, kappa=1, normed=True, check_input=True, 
                              ve_alg='numpy', np_optimize='greedy'):
        """
        **Arguments**

        - **edges** : _list_
            - Edges of the EFP graph specified by pairs of vertices.
        - **measure** : {`'hadr'`, `'hadrdot'`, `'hadrefm'`, `'ee'`, `'eeefm'`}
            - See [Measures](/docs/measure) for additional info.
        - **beta** : _float_
            - The parameter $\\beta$ appearing in the measure.
            Must be greater than zero.
        - **kappa** : {_float_, `'pf'`}
            - If a number, the energy weighting parameter $\\kappa$.
            If `'pf'`, use $\\kappa=v-1$ where $v$ is the valency of the vertex.
        - **normed** : _bool_
            - Controls normalization of the energies in the measure.
        - **check_input** : _bool_
            - Whether to check the type of the input each time or assume
            the first input type.
        - **ve_alg** : {`'numpy'`, `'ef'`}
            - Which variable elimination algorithm to use.
        - **np_optimize** : {`True`, `False`, `'greedy'`, `'optimal'`}
            - When `ve_alg='numpy'` this is the `optimize` keyword
            of `numpy.einsum_path`.
        """

        # initialize EFPBase
        super(EFP, self).__init__(measure, beta, kappa, normed, check_input)

        # store these edges as an EFPElem
        self.efpelem = EFPElem(edges)

        self.use_efms = 'efm' in self.measure

        if self.use_efms:
            efm_einstr, efm_spec = efp2efms(self.graph)
            self._efmset = EFMSet(efm_spec, subslicing=self.subslicing)
            efm_einpath = np.einsum_path(efm_einstr, 
                                         *[np.empty([4]*sum(s)) for s in efm_spec],
                                         optimize=np_optimize)[0]
            self.pow2 = 2**self.d
            self.efpelem = EFPElem(self.graph, efm_einstr=efm_einstr, 
                                   efm_einpath=efm_einpath, efm_spec=efm_spec)
        else:

            # get ve instance
            self.ve = VariableElimination(ve_alg, np_optimize)

            # set internals of ve to these edges
            self.ve.run(self.simple_graph, self.n)
            self.efpelem.einstr, self.efpelem.einpath = self.ve.einspecs()
            
    #===============
    # public methods
    #===============

    # compute(event=None, zs=None, thetas=None, ps=None)
    def compute(self, event=None, zs=None, thetas=None, ps=None, **kwargs):

        if self.use_efms:
            return self.efpelem.compute(self.construct_efms(event, zs, ps))
        else:

            # get dictionary of thetas to use for event
            zs, thetas_dict = self._get_zs_thetas_dict(event, zs, thetas)

            # call compute on the EFPElem
            return self.efpelem.compute(zs, thetas_dict)

    def batch_compute(self, events, n_jobs=-1):

        return super(EFP, self).batch_compute(events, n_jobs)

    #===========
    # properties
    #===========

    @property
    def _weight_set(self):
        """Set of edge weights for the graph of this EFP."""

        return self.efpelem.weight_set

    @property
    def efmset(self):
        """Get items of EFMs."""

        return self._efmset

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
    def c(self):
        """VE complexity $\\chi$ of this EFP."""

        if hasattr(self.ve, 'chi'):
            return self.ve.chi
        else:
            return None


###############################################################################
# EFPSet
###############################################################################
class EFPSet(EFPBase):

    """A class that holds a collection of EFPs and computes their values on events."""

    # EFPSet(*args, filename=None, measure='hadrdot', beta=2, kappa=1, normed=True, 
    #        check_input=True, verbose=False)
    def __init__(self, *args, **kwargs):
        """
        EFPSet can be initialized in one of three ways (in order of precedence):

        1. **Default** - Use the EFPs that come installed with the
        `EnergFlow` package.
        2. **Generator** - Pass in a custom `Generator` object as the
        first positional argument.
        3. **Custom File** - Pass in the name of a `.npz` file saved
        with a custom `Generator`.

        To control which EFPs are included, `EFPSet` accepts an arbitrary
        number of specifications (see `sel`) and only EFPs meeting each
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
            - See [Measures](/intro/measures) for additional info.
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
                          'measure': 'hadrdot',
                          'beta': 2,
                          'kappa': 1,
                          'normed': True,
                          'check_input': True,
                          'verbose': False}
        measure_kwargs = ['measure', 'beta', 'kappa', 'normed', 'check_input']

        # process arguments
        for k,v in default_kwargs.items():
            if k not in kwargs:
                kwargs[k] = v
            if k not in measure_kwargs:
                setattr(self, k, kwargs.pop(k))
        kwargs_check('__init__', kwargs, allowed=measure_kwargs)

        # initialize EFPBase
        super(EFPSet, self).__init__(*[kwargs[k] for k in measure_kwargs])
        self.use_efms = 'efm' in self.measure

        # handle different methods of initialization
        maxs = ['nmax','emax','dmax','cmax','vmax','comp_dmaxs']
        elemvs = ['edges','weights','einstrs','einpaths']
        efmvs = ['efm_einstrs','efm_einpaths','efm_specs']
        if len(args) >= 1 and isinstance(args[0], Generator):
            constructor_attrs = maxs + elemvs + efmvs + ['cols','gen_efms',
                                                         'c_specs','disc_specs','disc_formulae']
            gen = {attr: getattr(args[0], attr) for attr in constructor_attrs}
            args = args[1:]
        elif self.filename is not None:
            self.filename += '.npz' if not self.filename.endswith('.npz') else ''
            gen = np.load(self.filename)
        else:
            gen = np.load(default_efp_file)

        # handle not having efm generation
        if not gen['gen_efms'] and self.use_efms:
            raise ValueError('cannot use efm measure without providing efm generation')

        # compile regular expression for use in sel()
        self._sel_re = re.compile('(\w+)(<|>|==|!=|<=|>=)(\d+)$')
        
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
        self._specs = concat_specs(orig_c_specs[c_mask], orig_disc_specs[disc_mask])

        # make EFPElem list
        z = zip(*([gen[v] for v in elemvs] + 
                  [gen[v] if self.use_efms else repeat(None) for v in efmvs]))
        ks = orig_c_specs[:,self.k_ind]
        self.efpelems = [EFPElem(edgs, ws, es, ep, ks[m], efm_es, efm_ep, efm_sp) \
                         for m,(edgs,ws,es,ep,efm_es,efm_ep,efm_sp) in enumerate(z) if c_mask[m]]

        # setup EFMs
        if self.use_efms:
            efm_specs = chain(*[elem.efm_spec for elem in self.efpelems])
            self._efmset = EFMSet(efm_specs, subslicing=self.subslicing)

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

    #def _compute_func_thetas(self, args):
    #    return self.compute(zs=args[0], thetas=args[1], batch_call=True)

    #def _compute_func_ps(self, args):
    #    return self.compute(zs=args[0], ps=args[1], batch_call=True)

    def _set_col_inds(self):
        self.__dict__.update({col+'_ind': i for i,col in enumerate(self.cols)})

    def _calc_disc(self, X):

        XX = X
        if not isinstance(X, np.ndarray):
            XX = np.asarray(X)

        if len(self.disc_col_inds) == 0:
            return XX

        l = len(XX.shape) 
        if l == 2:
            results = np.ones((len(X), len(self.disc_col_inds)), dtype=float)
            for i,formula in enumerate(self.disc_col_inds):
                results[:,i] = np.prod(XX[:,formula], axis=1)
            concat_axis = 1
        elif l == 1:
            results = np.ones(len(self.disc_col_inds), dtype=float)
            for i,formula in enumerate(self.disc_col_inds):
                results[i] = np.prod(XX[formula])
            concat_axis = 0
        else:
            raise ValueError('X has the wrong dimensions')

        return np.concatenate([XX, results], axis=concat_axis)

    def _make_graphs(self, connected_graphs):
        disc_comps = [[connected_graphs[i] for i in col_inds] for col_inds in self.disc_col_inds]
        return np.asarray(connected_graphs + [graph_union(*dc) for dc in disc_comps])


    #===============
    # PUBLIC METHODS
    #===============

    # compute(event=None, zs=None, thetas=None, ps=None)
    def compute(self, event=None, zs=None, thetas=None, ps=None, batch_call=False):

        if self.use_efms:
            efms_dict = self.construct_efms(event, zs, ps)
            results = [efpelem.compute(efms_dict) for efpelem in self.efpelems]
        else:
            zs, thetas_dict = self._get_zs_thetas_dict(event, zs, thetas)
            results = [efpelem.compute(zs, thetas_dict) for efpelem in self.efpelems]

        if batch_call:
            return results
        else:
            return self._calc_disc(results)

    def batch_compute(self, events, n_jobs=-1):

        return self._calc_disc(super(EFPSet, self).batch_compute(events, n_jobs))

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

        - _numpy.ndarray_
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
        - _numpy.ndarray_, otherwise
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
        - _numpy.ndarray_, otherwise
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

    def set_timers(self, repeat=5, number=10):
        for efpelem in self.efpelems:
            efpelem.set_timer(repeat, number)

    def get_times(self):
        return np.asarray([elem.times for elem in self.efpelems])


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
    def efmset(self):
        """The `EFMset` held by this object, if using EFMs."""

        return self._efmset
