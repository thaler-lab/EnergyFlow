from __future__ import absolute_import, division, print_function

import itertools
import numpy as np
import igraph

from energyflow.efp.efp_base import EFP, EFPSet
from energyflow.efp.integer_partitions import int_partition_ordered, int_partition_unordered
from energyflow.efp.ve import ve_elim_order, ve_einsum_path

__all__ = ['EFPGenerator']

class EFPGenerator(EFPSet, EFP):

    """
    A class that can generate EFPs and save them to file.
    """

    def __init__(self, dmax, Nmax=None, emax=None, cmax=None, verbose=True,
                       ve_alg='numpy', np_optimize='greedy', do_weights=True):

        self.doing_gen = True

        # store parameters in base class
        EFPSet.__init__(dmax=dmax, Nmax=Nmax, emax=emax, cmax=cmax, verbose=verbose)

        assert ve_alg in ['ef', 'numpy'], 've_alg must be either ef or numpy'
        self.ve_alg = ve_alg
        if self.ve_alg == 'numpy':
            EFP._ve_init(np_optimize)
        self.chi_ein_func = {'ef': self._chi_ein_ve, 'numpy': self._chi_ein_numpy}[self.ve_alg]

        # setup N and e values to be used
        self.Ns = list(range(2, self.Nmax+1))
        self.emaxs = {n: min(self.emax, int(n/2*(n-1))) for n in self.Ns}
        self.esbyn = {n: list(range(n-1, self.emaxs[n]+1)) for n in self.Ns}
        self.dmaxs = {(n,e): self.dmax for n in self.Ns for e in self.esbyn[n]}

        # setup storage containers
        self.simple_graphs_d = {(n,e): [] for n in self.Ns for e in self.esbyn[n]}
        self.edges_d         = {(n,e): [] for n in self.Ns for e in self.esbyn[n]}
        self.chis_d          = {(n,e): [] for n in self.Ns for e in self.esbyn[n]}
        self.einpaths_d      = {(n,e): [] for n in self.Ns for e in self.esbyn[n]}
        self.einstrs_d       = {(n,e): [] for n in self.Ns for e in self.esbyn[n]}
        self.weights_d       = {(n,e): [] for n in self.Ns for e in self.esbyn[n]}

        # get simple graphs
        self._generate_simple()

        # get weighted graphs
        if do_weights: 
            self._generate_weights()

            # get disconnected graphs
            self._init_disconnected()
            self._generate_disconnected()

    # generates simple graphs subject to constraints
    def _generate_simple(self):

        self.base_edges = {n: list(itertools.combinations(range(n), 2)) for n in self.Ns}

        self._add_if_new(igraph.Graph.Full(2, directed=False), (2,1))

        # iterate over all combinations of n>2 and d
        for n in self.Ns[1:]:
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

        if self.verbose: 
            print('# of simple graphs by n:', self._count_simple_by_n())
            print('# of simple graphs by e:', self._count_simple_by_e())

    # adds simple graph if it is non-isomorphic to existing graphs and has a valid metric
    def _add_if_new(self, new_graph, ne):

        # check for isomorphism with existing graphs
        for graph in self.simple_graphs_d[ne]:
            if new_graph.isomorphic(graph): return

        # check that ve complexity for this graph is valid
        chiein = self.chi_ein_func(new_graph.get_edgelist(), new_graph.vcount())
        new_chi, new_einstr, new_einpath, new_elim_order = chiein
        if new_chi > self.cmax: return
        
        # append graph and ve complexity to containers
        self.simple_graphs_d[ne].append(new_graph)
        self.edges_d[ne].append(new_graph.get_edgelist())
        self.chis_d[ne].append(new_chi)
        self.einstrs_d[ne].append(new_einstr)

        if self.ve_alg == 'numpy': 
            self.einpaths_d[ne].append(new_einpath)
        else: 
            self.einpaths_d[ne].append(ve_einsum_path(graph, new_elim_order))

    def _chi_ein_ve(self, edges, n):
        einstr = _self.einstr_from_graph(edges, n)[0]
        ve_elim_order, max_min_degree = ve_elim_order(graph)
        return max_min_degree, einstr, None, ve_elim_order

    # generator for edges not already in list
    def _edge_filter(self, n, edges):
        for edge in self.base_edges[n]:
            if edge not in edges:
                yield edge

    # generates non-isomorphic graph weights subject to constraints
    def _generate_weights(self):

        # take care of the n=2 case:
        self.weights_d[(2,1)].append([(d,) for d in range(1, self.dmaxs[(2,1)]+1)])

        # get ordered integer partitions of d of length e for relevant values
        parts = {}
        for n in self.Ns[1:]:
            for e in self.esbyn[n]:
                for d in range(e, self.dmaxs[(n,e)]+1):
                    if (d,e) not in parts:
                        parts[(d,e)] = list(int_partition_ordered(d, e))

        # iterate over the rest of ns
        for n in self.Ns[1:]:

            # iterate over es for which there are simple graphs
            for e in self.esbyn[n]:

                # iterate over simple graphs
                for graph in self.simple_graphs_d[(n,e)]:
                    weightings = []

                    # iterate over valid d for this graph
                    for d in range(e, self.dmaxs[(n,e)]+1):

                        # iterate over int partitions
                        for part in parts[(d,e)]:

                            # check if isomorphic to existing
                            iso = False
                            for weighting in weightings:
                                if graph.isomorphic_vf2(other=graph, 
                                                        edge_color1=weighting, 
                                                        edge_color2=part): 
                                    iso = True
                                    break
                            if not iso: weightings.append(part)
                    self.weights_d[(n,e)].append(weightings)

        if self.verbose: 
            print('# of weightings by n:', self._count_weighted_by_n())
            print('# of weightings by d:', self._count_weighted_by_d())

    def _init_disconnected(self):

        """ 
        Column descriptions:
        n - number of vertices in graph
        e - number of edges in (underlying) simple graph
        d - number of edges in multigraph
        k - unique index for graphs with a fixed (n,d)
        g - index of simple edges in edges
        w - index of weights in weights
        c - complexity, with respect to some VE algorithm
        p - number of prime factors for this EFP
        """

        self.cols = ['n','e','d','k','g','w','c','p']
        self.get_col_inds()
        self.connected_specs = []
        self.edges, self.weights, self.einstrs, self.einpaths = [], [], [], []
        self.ks, self.ndk2i = {}, {}
        g = w = i = 0
        for ne in sorted(self.chis_d.keys()):
            n, e = ne
            z = zip(self.edges_d[ne], self.weights_d[ne], self.chis_d[ne], 
                    self.einstrs_d[ne], self.einpaths_d[ne])
            for edges, weights, chi, es, ep in z:
                for weighting in weights:
                    d = sum(weighting)
                    k = self.ks.setdefault((n,d), 0)
                    self.ks[(n,d)] += 1
                    self.connected_specs.append([n,e,d,k,g,w,chi,1])
                    self.ndk2i[(n,d,k)] = i
                    self.weights.append(weighting)
                    w += 1
                    i += 1
                self.edges.append(edges)
                self.einstrs.append(es)
                self.einpaths.append(ep)
                g += 1
        self.connected_specs = np.asarray(self.connected_specs)

    def _generate_disconnected(self):
        
        disc_formulae, disc_specs = [], []

        # disconnected start at N>=4
        for n in range(4,2*self.dmax+1):

            # partitions with no 1s, no numbers > self.Nmax, and not the trivial partition
            good_part = lambda x: (1 not in x and max(x) <= self.Nmax and len(x) > 1)
            n_parts = [tuple(x) for x in int_partition_unordered(n) if good_part(x)]
            n_parts.sort(key=len)

            # iterate over all ds
            for d in range(int(n/2),self.dmax+1):

                # iterate over all n_parts
                for n_part in n_parts:
                    n_part_len = len(n_part)

                    # get w_parts of the right length
                    d_parts = [x for x in int_partition_unordered(d) if len(x) == n_part_len]

                    # ensure that we found some
                    if len(d_parts) == 0: continue

                    # usage of set and sorting is important to avoid duplicates
                    specs = set()

                    # iterate over all orderings of the n_part
                    for n_part_ord in set([x for x in itertools.permutations(n_part)]):

                        # iterate over all w_parts
                        for d_part in d_parts:

                            # construct spec. sorting ensures we don't get duplicates in specs
                            spec = tuple(sorted([(npo,dp) for npo,dp in zip(n_part_ord,d_part)]))

                            # check that we have the proper primes to calculate this spec
                            good = True
                            for pair in spec:

                                # w needs to be in range n-1,wmaxs[n]
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
                        for kspec in itertools.product(*[range(self.ks[factor]) \
                                                         for factor in spec]):

                            # iterate over factors
                            formula = []
                            cmax = emax = 0 
                            for (nn,dd),kk in zip(spec,kspec):

                                # select original simple graph
                                ind = self.ndk2i[(nn,dd,kk)]

                                # add col index of factor to formula
                                formula.append(ind)
                                cmax = max(cmax, self.connected_specs[ind,self.c_ind])
                                emax = max(emax, self.connected_specs[ind,self.e_ind])

                            # append to stored array
                            disc_formulae.append(tuple(sorted(formula)))
                            disc_specs.append([n,emax,d,kcount,-1,-1,cmax,len(kspec)])
                            kcount += 1

        # ensure unique formulae (deals with possible degeneracy in selection of factors)
        disc_form_set = set()
        mask = np.asarray([not(form in disc_form_set or disc_form_set.add(form)) \
                           for form in disc_formulae])

        # store as numpy arrays
        self.disc_formulae = np.asarray(disc_formulae)[mask]
        self.disc_specs = np.asarray(disc_specs)[mask]

        if len(self.disc_specs.shape) == 2:
            self.specs = np.concatenate((self.connected_specs, self.disc_specs))
        else:
            self.specs = self.connected_specs

    def _count_simple_by_n(self):
        return {n: np.sum([len(self.edges_d[(n,e)]) for e in self.esbyn[n]]) for n in self.Ns}

    def _count_simple_by_e(self):
        return {e: np.sum([len(self.edges_d[(n,e)]) for n in self.Ns if (n,e) in self.edges_d]) \
                           for e in range(1,self.emax+1)}

    def _count_weighted_by_n(self):
        return {n: np.sum([len(weights) for e in self.esbyn[n] \
                           for weights in self.weights_d[(n,e)]]) for n in self.Ns}

    def _count_weighted_by_d(self):
        counts = {d: 0 for d in range(1,self.dmax+1)}
        for n in self.Ns:
            for e in self.esbyn[n]:
                for weights in self.weights_d[(n,e)]:
                    for weighting in weights: counts[sum(weighting)] += 1
        return counts

    def save(self, filename):
        np.savez(filename, **{'ve_alg':        self.ve_alg,
                              'cols':          self.cols,
                              'specs':         self.specs,
                              'disc_formulae': self.disc_formulae,
                              'edges':         self.edges,
                              'einstrs':    self.einstrs,
                              'einpaths':      self.einpaths,
                              'weights':       self.weights})
