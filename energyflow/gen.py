"""Implementation of EFP Generator class."""
from __future__ import absolute_import, division, print_function

from collections import Counter
import itertools

import numpy as np

from energyflow.algorithms import *
from energyflow.efpbase import EFPElem
from energyflow.utils import concat_specs, default_efp_file, transfer
from energyflow.utils.graph_utils import *

igraph = igraph_import()

__all__ = ['Generator']

###############################################################################
# Generator helpers
###############################################################################
def none2inf(x):
    return np.inf if x is None else x

###############################################################################
# Generator
###############################################################################
class Generator(object):

    """Generates non-isomorphic multigraphs according to provided specifications."""

    def __init__(self, dmax=None, nmax=None, emax=None, cmax=None, vmax=None, comp_dmaxs=None,
                       filename=None, np_optimize='greedy', verbose=False):
        """Doing a fresh generation of connected multigraphs (`filename=None`) requires
        that `igraph` be installed.

        **Arguments**

        - **dmax** : _int_
            - The maximum number of edges of the generated connected graphs.
        - **nmax** : _int_
            - The maximum number of vertices of the generated connected graphs.
        - **emax** : _int_
            - The maximum number of edges of the generated connected simple graphs.
        - **cmax** : _int_
            - The maximum VE complexity $\\chi$ of the generated connected graphs.
        - **vmax** : _int_
            - The maximum valency of the generated connected graphs.
        - **comp_dmaxs** : {_dict_, _int_}
            - If an integer, the maximum number of edges of the generated disconnected 
            graphs. If a dictionary, the keys are numbers of vertices and the values are
            the maximum number of edges of the generated disconnected graphs with that
            number of vertices.
        - **filename** : _str_
            - If `None`, do a complete generation from scratch. If set to a string, 
            read in connected graphs from the file given, restrict them according to 
            the various 'max' parameters, and do a fresh disconnected generation. The special
            value `filename='default'` means to read in graphs from the default file. This
            is useful when various disconnected graph parameters are to be varied since the 
            generation of large simple graphs is the most computationlly intensive part.
        - **np_optimize** : {`True`, `False`, `'greedy'`, `'optimal'`}
            - The `optimize` keyword of `numpy.einsum_path`.
        - **verbose** : _bool_
            - A flag to control printing.
        """

        # check for new generation
        if dmax is not None and filename is None:

            # set maxs
            self._set_maxs(dmax, nmax, emax, cmax, vmax)

            # set options
            self.np_optimize = np_optimize

            # get prime generator instance
            self.pr_gen = PrimeGenerator(self.dmax, self.nmax, self.emax, self.cmax, self.vmax, 
                                         self.np_optimize)
            self.cols = self.pr_gen.cols
            self._set_col_inds()

            # store lists of important quantities
            transfer(self, self.pr_gen, self._prime_attrs())

        # if filename is set, read in file
        else:
            if filename is None or filename == 'default':
                filename = default_efp_file

            file = np.load(filename + ('' if filename.endswith('.npz') else '.npz'))

            # setup cols and col inds
            self.cols = file['cols']
            self._set_col_inds()

            # get maxs from file and passed in options
            c_specs = file['c_specs']
            local_vars = locals()
            for m in ['dmax','nmax','emax','cmax','vmax']:
                setattr(self, m, min(file[m], none2inf(local_vars[m])))

            # select connected specs based on maxs
            mask = ((c_specs[:,self.d_ind] <= self.dmax) & 
                    (c_specs[:,self.n_ind] <= self.nmax) & 
                    (c_specs[:,self.e_ind] <= self.emax) & 
                    (c_specs[:,self.c_ind] <= self.cmax) & 
                    (c_specs[:,self.v_ind] <= self.vmax))

            # set ve options
            self.np_optimize = file['np_optimize']

            # get lists of important quantities
            for attr in (self._prime_attrs()):
                setattr(self, attr, [x for x,m in zip(file[attr],mask) if m])
            self.c_specs = c_specs[mask]



        # setup generator of disconnected graphs
        self._set_comp_dmaxs(comp_dmaxs)
        self.comp_gen = CompositeGenerator(self.c_specs, self.cols, self.comp_dmaxs)

        # get results and store
        transfer(self, self.comp_gen, self._comp_attrs())


    #################
    # PRIVATE METHODS
    #################

    def _set_col_inds(self):
        self.__dict__.update({col+'_ind': i for i,col in enumerate(self.cols)})

    def _set_maxs(self, dmax, nmax, emax, cmax, vmax):
        self.dmax = dmax
        self.nmax = nmax if nmax is not None else self.dmax + 1
        self.emax = emax if emax is not None else self.dmax
        self.cmax = cmax if cmax is not None else self.nmax
        self.vmax = vmax if vmax is not None else self.dmax

    def _set_comp_dmaxs(self, comp_dmaxs):
        if isinstance(comp_dmaxs, dict):
            self.comp_dmaxs = comp_dmaxs
        else:
            if comp_dmaxs is None:
                comp_dmaxs = self.dmax
            elif not isinstance(comp_dmaxs, int):
                raise TypeError('dmaxs cannot be type {}'.format(type(comp_dmaxs)))

            # implement comp_dmaxs as dict
            self.comp_dmaxs = {}
            if comp_dmaxs >= 2:
                self.comp_dmaxs = {n: comp_dmaxs for n in range(4, 2*comp_dmaxs+1)}

    def _prime_attrs(self, no_global=False):
        attrs = set(['edges', 'weights', 'einstrs', 'einpaths', 'c_specs'])
        return attrs

    def _comp_attrs(self):
        return set(['disc_specs', 'disc_formulae'])


    ################
    # PUBLIC METHODS
    ################

    def save(self, filename):
        """Save the current generator to file.

        **Arguments**

        - **filename** : _str_
            - The path to save the file.
        """
        
        arrs = set(['dmax', 'nmax', 'emax', 'cmax', 'vmax', 'comp_dmaxs',
                    'cols', 'np_optimize'])
        arrs |= self._prime_attrs() | self._comp_attrs()
        np.savez(filename, **{arr: getattr(self, arr) for arr in arrs})

    @property
    def specs(self):
        """An array of EFP specifications. Each row represents an EFP 
        and the columns represent the quantities indicated by `cols`."""
        
        if not hasattr(self, '_specs'):
            self._specs = concat_specs(self.c_specs, self.disc_specs)
        return self._specs

###############################################################################
# PrimeGenerator
###############################################################################
class PrimeGenerator(object):

    """Column descriptions:
    n - number of vertices in graph
    e - number of edges in (underlying) simple graph
    d - number of edges in multigraph
    v - maximum valency of the graph
    k - unique index for graphs with a fixed (n,d)
    c - complexity, with respect to some VE algorithm
    p - number of prime factors for this EFP
    h - number of valency 1 vertices in graph
    """
    cols = ['n','e','d','v','k','c','p','h']

    def __init__(self, dmax, nmax, emax, cmax, vmax, np_optimize):
        """PrimeGenerator __init__."""

        if not igraph:
            raise NotImplementedError('cannot use PrimeGenerator without igraph')
        
        self.ve = VariableElimination(np_optimize)

        # store parameters
        transfer(self, locals(), ['dmax', 'nmax', 'emax', 'cmax', 'vmax',])

        # setup N and e values to be used
        self.ns = list(range(1, self.nmax+1))
        self.emaxs = {n: min(self.emax, int(n/2*(n-1))) for n in self.ns}
        self.esbyn = {n: list(range(n-1, self.emaxs[n]+1)) for n in self.ns}

        # this could be more complicated than the same max for all (n,e)
        self.dmaxs = {(n,e): self.dmax for n in self.ns for e in self.esbyn[n]}

        # setup storage containers
        quantities = ['simple_graphs_d', 'edges_d', 'chis_d', 'einpaths_d',
                      'einstrs_d', 'weights_d']
        for q in quantities:
            setattr(self, q, {(n,e): [] for n in self.ns for e in self.esbyn[n]})

        # get simple connected graphs
        self._generate_simple()

        # get weighted connected graphs
        self._generate_weights()

        # flatten structures
        self._flatten_structures()


    #################
    # PRIVATE METHODS
    #################

    # generates simple graphs subject to constraints
    def _generate_simple(self):

        self.base_edges = {n: list(itertools.combinations(range(n), 2)) for n in self.ns}

        if self.nmax >= 1:
            self._add_if_new(igraph.Graph.Full(1, directed=False), (1,0))

        # iterate over all combinations of n>1 and d
        for n in self.ns[1:]:
            for e in self.esbyn[n]:

                # consider adding new vertex
                if e-1 in self.esbyn[n-1]:

                    # iterate over all graphs with n-1, e-1
                    for seed_graph in self.simple_graphs_d[(n-1,e-1)]:

                        # iterate over vertices to attach to
                        for v in range(n-1):
                            new_graph = seed_graph.copy()
                            new_graph.add_vertices(1)
                            new_graph.add_edges([(v,n-1)])
                            self._add_if_new(new_graph, (n,e))

                # consider adding new edge to existing set of vertices
                if e-1 in self.esbyn[n]:

                    # iterate over all graphs with n, d-1
                    for seed_graph, seed_edges in zip(self.simple_graphs_d[(n,e-1)], 
                                                      self.edges_d[(n,e-1)]):

                        # iterate over edges that don't exist in graph
                        for new_edge in self._edge_filter(n, seed_edges):
                            new_graph = seed_graph.copy()
                            new_graph.add_edges([new_edge])
                            self._add_if_new(new_graph, (n,e))

    # adds simple graph if it is non-isomorphic to existing graphs and has a valid metric
    def _add_if_new(self, new_graph, ne):

        # check for isomorphism with existing graphs
        for graph in self.simple_graphs_d[ne]:
            if new_graph.isomorphic(graph): 
                return

        # check that ve complexity for this graph is valid
        new_edges = new_graph.get_edgelist()
        einstr, einpath, chi = self.ve.einspecs(new_edges, ne[0])
        if chi > self.cmax: 
            return
        
        # append graph and ve complexity to containers
        self.simple_graphs_d[ne].append(new_graph)
        self.edges_d[ne].append(new_edges)
        self.chis_d[ne].append(chi)

        self.einstrs_d[ne].append(einstr)
        self.einpaths_d[ne].append(einpath)

    # generator for edges not already in list
    def _edge_filter(self, n, edges):
        for edge in self.base_edges[n]:
            if edge not in edges:
                yield edge

    # generates non-isomorphic graph weights subject to constraints
    def _generate_weights(self):

        # take care of the n=2 case
        if (2,1) in self.weights_d:
            self.weights_d[(2,1)].append([(d,) for d in range(1, self.dmaxs[(2,1)]+1)])

        # get ordered integer partitions of d of length e for relevant values
        parts = {}
        for n in self.ns[2:]:
            for e in self.esbyn[n]:
                for d in range(e, self.dmaxs[(n,e)]+1):
                    if (d,e) not in parts:
                        parts[(d,e)] = list(int_partition_ordered(d, e))

        # iterate over the rest of ns
        for n in self.ns[2:]:

            # iterate over es for which there are simple graphs
            for e in self.esbyn[n]:

                # iterate over simple graphs
                for graph in self.simple_graphs_d[(n,e)]:
                    weightings = []

                    # iterate over valid d for this graph
                    for d in range(e, self.dmaxs[(n,e)]+1):

                        # iterate over int partitions
                        for part in parts[(d,e)]:

                            # check that maximum valency is not exceeded 
                            if (self.vmax < self.dmax and 
                                max(graph.strength(weights=part)) > self.vmax):
                                continue

                            # check if isomorphic to existing
                            iso = False
                            for weighting in weightings:
                                if graph.isomorphic_vf2(other=graph, 
                                                        edge_color1=weighting, 
                                                        edge_color2=part): 
                                    iso = True
                                    break
                            if not iso: 
                                weightings.append(part)
                    self.weights_d[(n,e)].append(weightings)

    def _flatten_structures(self):
        c_specs, self.edges, self.weights, self.einstrs, self.einpaths = [], [], [], [], []
        ks = {}

        # handle n=1 case specially
        c_specs.append([1,0,0,0,0,1,1,0])
        self.edges.append(())
        self.weights.append(())
        self.einstrs.append(self.einstrs_d[(1,0)][0])
        self.einpaths.append(self.einpaths_d[(1,0)][0])

        for ne in sorted(self.edges_d.keys()):
            n, e = ne
            z = zip(self.edges_d[ne], self.weights_d[ne], self.chis_d[ne],
                    self.einstrs_d[ne], self.einpaths_d[ne])
            for edgs, ws, c, es, ep in z:
                for w in ws:
                    d = sum(w)
                    k = ks.setdefault((n,d), 0)
                    ks[(n,d)] += 1
                    vs = valencies(EFPElem(edgs, weights=w).edges).values()
                    v = max(vs)
                    h = Counter(vs)[1]
                    c_specs.append([n, e, d, v, k, c, 1, h])
                    self.edges.append(edgs)
                    self.weights.append(w)
                    self.einstrs.append(es)
                    self.einpaths.append(ep)
        self.c_specs = np.asarray(c_specs)

###############################################################################
# CompositeGenerator
###############################################################################
class CompositeGenerator(object):

    """CompositeGenerator"""

    def __init__(self, c_specs, cols, comp_dmaxs=None):
        """CompositeGenerator __init__"""

        self.c_specs = c_specs
        self.__dict__.update({col+'_ind': i for i,col in enumerate(cols)})
        self.comp_dmaxs = comp_dmaxs
        
        self.ns = sorted(self.comp_dmaxs.keys())
        self.nmax_avail = np.max(self.c_specs[:,self.n_ind]) if len(self.c_specs) else 0

        self.ks, self.ndk2i = {}, {}
        for i,spec in enumerate(self.c_specs):
            n, d, k = spec[[self.n_ind, self.d_ind, self.k_ind]]
            self.ks.setdefault((n,d), 0)
            self.ks[(n,d)] += 1
            self.ndk2i[(n,d,k)] = i

        self._generate_disconnected()


    #################
    # PRIVATE METHODS
    #################

    def _generate_disconnected(self):
        
        disc_formulae, disc_specs = [], []

        for n in self.ns:

            # partitions with no 1s, no numbers > self.nmax_avail, and not the trivial partition
            good_part = lambda x: (1 not in x and max(x) <= self.nmax_avail and len(x) > 1)
            n_parts = [tuple(x) for x in int_partition_unordered(n) if good_part(x)]
            n_parts.sort(key=len)

            # iterate over all ds
            for d in range(int((n-1)/2)+1, self.comp_dmaxs[n]+1):

                # iterate over all n_parts
                for n_part in n_parts:
                    n_part_len = len(n_part)

                    # get d_parts of the right length
                    d_parts = [x for x in int_partition_unordered(d) if len(x) == n_part_len]

                    # ensure that we found some
                    if len(d_parts) == 0: continue

                    # usage of set and sorting is important to avoid duplicates
                    specs = set()

                    # iterate over all orderings of the n_part
                    for n_part_ord in set(itertools.permutations(n_part)):

                        # iterate over all d_parts
                        for d_part in d_parts:

                            # construct spec. sorting ensures we don't get duplicates in specs
                            spec = tuple(sorted([(npo,dp) for npo,dp in zip(n_part_ord,d_part)]))

                            # check that we have the proper primes to calculate this spec
                            good = True
                            for pair in spec:
                                if pair not in self.ks:
                                    good = False
                                    break
                            if good:
                                specs.add(spec)

                    # iterate over all specs that we found
                    for spec in specs:

                        # keep track of how many we added
                        kcount = 0 if (n,d) not in self.ks else self.ks[(n,d)]

                        # iterate over all possible formula implementations with the different ndk
                        for kspec in itertools.product(*[range(self.ks[factor]) for factor in spec]):

                            # iterate over factors
                            formula = []
                            cmax = e = vmax = h = 0 
                            for (nn,dd),kk in zip(spec,kspec):

                                # add (n,d,k) of factor to formula
                                ndk = (nn,dd,kk)
                                formula.append(ndk)

                                # select original simple graph
                                ind = self.ndk2i[ndk]
                                cmax = max(cmax, self.c_specs[ind, self.c_ind])
                                e += self.c_specs[ind, self.e_ind]
                                vmax = max(vmax, self.c_specs[ind, self.v_ind])
                                h += self.c_specs[ind, self.h_ind]

                            # append to stored array
                            disc_formulae.append(tuple(sorted(formula)))
                            disc_specs.append([n, e, d, vmax, kcount, cmax, len(kspec), h])
                            kcount += 1

        # ensure unique formulae (deals with possible degeneracy in selection of factors)
        disc_form_set = set()
        mask = [not(form in disc_form_set or disc_form_set.add(form)) for form in disc_formulae]

        # store as numpy arrays
        self.disc_formulae = np.asarray(disc_formulae)[mask]
        self.disc_specs = np.asarray(disc_specs)[mask]
