from __future__ import absolute_import, division, print_function

import numpy as np

__all__ = ['EFPSet', 'calc_disc']

einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def calc_disc(X, disc_formulae, concat=False):

    results = np.ones((len(X), len(disc_formulae)), dtype=float)
    XX = X if type(X) == np.ndarray else np.asarray(X)

    for i,formula in enumerate(disc_formulae):
        for col_ind in formula:
            results[:,i] *= XX[:,col_ind]

    if concat: 
        return np.concatenate([XX, results], axis=1)
    else: 
        return results

class EFP:

    def __init__(self, edges, np_optimize='greedy'):

        # store the initial graph information
        self.initial_edges = edges

        # deal with arbitrary vertex labels
        vertex_set = set(v for edge in self.initial_edges for v in edge)
        self.vertices = {v: i for i,v in enumerate(sorted(list(vertex_set)))}
        self.N = len(vertex_set)

        # construct new edges
        self.edges = [tuple(self.vertices[v] for v in edge) for edge in self.initial_edges]
        self.d = len(self.edges)

        self._ve_init(np_optimize)
        self.c, self, self.einstr, self.einpath, _ = self._chi_ein_numpy(self.edges, self.N)

    def _ve_init(self, np_optimize):
        self.np_optimize = np_optimize
        self.dummy_dim = 10
        self.X = np.random.rand(self.dummy_dim, self.dummy_dim)
        self.y = np.random.rand(self.dummy_dim)

    def _chi_ein_numpy(self, edges, n):
        einstr, nv, ne = _self.einstr_from_graph(edges, n)
        einpath = np.einsum_path(einstr, *[self.X]*ne, *[self.y]*nv, optimize=self.np_optimize)
        return int(einpath[1].split('\n')[2].split(':')[1]), einstr, einpath[0], None

    def _einstr_from_edges(self, edges, n):
        einstr  = ','.join([einsum_symbols[j] + einsum_symbols[k] for (j, k) in edges]) + ','
        einstr += ','.join([einsum_symbols[v] for v in range(n)])
        return einstr, n, len(edges)

    # compute the energy flow polynomial corresponding to the graph with certain edge weights
    def compute(self, zs, thetas, weights):
        return np.einsum(self.einstr, *[thetas[w] for w in weights], *[zs]*self.N, optimize=self.einpath)

class EFPSet:

    doing_gen = False
    
    def __init__(self, filename=None, dmax=None, Nmax=None, emax=None, cmax=None, verbose=True):

        assert filename is not None or dmax is not None, 'filename and dmax cannot both be None'
        self.verbose = verbose

        if filename is not None:
            self.filename = filename if '.npz' in filename else filename+'.npz'
            self.load_file(dmax, Nmax, emax, cmax)
            self.make_connected_iterable()
        elif dmax is not None:
            if not self.doing_gen:
                print('WARNING: Empty EFPSet being created')
            self.dmax = dmax
            self.Nmax = Nmax if Nmax is not None else dmax+1
            self.emax = emax if emax is not None else dmax
            self.cmax = cmax if cmax is not None else self.Nmax

    def load_file(self, dmax, Nmax, emax, cmax):

        fdict = np.load(self.filename)
        specs = fdict['specs']
        self.cols = fdict['cols']
        self.get_col_inds()

        f_dmax, f_Nmax = np.max(specs[:,self.d_ind]), np.max(specs[:,self.n_ind])
        f_emax, f_cmax = np.max(specs[:,self.e_ind]), np.max(specs[:,self.c_ind])
        num_connected = np.count_nonzero(specs[:,self.p_ind] == 1)

        if self.verbose:
            print('Loading EFPs from {}:'.format(self.filename))
            print('    Total EFPs:', len(specs))
            print('    Total Prime EFPs:', num_connected)
            print('    Maximum d:', f_dmax)
            print('    Maximum N:', f_Nmax)
            print('    Maximum chi:', f_cmax)
            print('    Maximum e:', f_emax)
            print('    ve_alg:', fdict['ve_alg'])

        self.dmax = f_dmax if dmax is None else dmax
        self.Nmax = f_Nmax if Nmax is None else Nmax
        self.emax = f_emax if emax is None else emax
        self.cmax = f_cmax if cmax is None else cmax

        mask = ((specs[:,self.d_ind] <= self.dmax)&(specs[:,self.n_ind] <= self.Nmax)
               &(specs[:,self.e_ind] <= self.emax)&(specs[:,self.c_ind] <= self.cmax))

        connected_inds = np.nonzero(mask & (specs[:,self.p_ind] == 1))[0]
        con_inds_dict = {old_ind: new_ind for new_ind,old_ind in enumerate(connected_inds)}
        disc_inds = np.nonzero(mask & (specs[:,self.p_ind] > 1))[0] - num_connected
        self.disc_formulae = [[con_inds_dict[old_ind] for old_ind in formula] \
                              for formula in fdict['disc_formulae'][disc_inds]]
        
        self.specs = specs[mask]
        gs, ws = set(self.specs[:,self.g_ind]), set(self.specs[:,self.w_ind])
        gs.remove(-1)
        ws.remove(-1)
        self.edges = [edges for g,edges in enumerate(fdict['edges']) if g in gs]
        self.einstrs = [estr for g,estr in enumerate(fdict['einstrs']) if g in gs]
        self.einpaths = [epath for g,epath in enumerate(fdict['einpaths']) if g in gs]
        self.weights = [weights for w,weights in enumerate(fdict['weights']) if w in ws]

        gs_dict = {old_g: new_g for new_g,old_g in enumerate(sorted(list(gs)))}
        ws_dict = {old_w: new_w for new_w,old_w in enumerate(sorted(list(ws)))}
        gw = (self.g_ind, self.w_ind)
        for i in range(len(self.specs)):
            old_g, old_w = self.specs[i,gw]
            if old_g == -1 and old_w == -1: continue
            self.specs[i,gw] = (gs_dict[old_g], ws_dict[old_w])
        self.connected_specs = self.specs[self.specs[:,self.p_ind] == 1]

        if self.verbose:
            if (self.dmax == f_dmax and self.Nmax == f_Nmax and
                self.emax == f_emax and self.cmax == f_cmax):
                print('\nUsing specifications from file')
            else:
                print('\nAfter applying specifications:')
                print('    Total EFPs:', len(self.specs))
                print('    Total Prime EFPs:', np.count_nonzero(self.specs[:,self.p_ind] == 1))

    def get_col_inds(self):
        self.__dict__.update({col+'_ind': i for i,col in enumerate(self.cols)})

    def make_connected_iterable(self):
        self.connected_iterable = [(spec[self.n_ind],
                                    self.weights[spec[self.w_ind]],
                                    self.einstrs[spec[self.g_ind]],
                                    self.einpaths[spec[self.g_ind]]) \
                                   for spec in self.connected_specs]

    def compute(self, pts, yphis, betas=[1]):
        zs = pts/np.sum(pts)
        thetas2 = np.sum((yphis[:,np.newaxis] - yphis[np.newaxis,:])**2, axis=-1)

        results = []
        for beta in betas:
            thetas = {d: thetas2**(beta*d/2) for d in range(1, self.dmax + 1)}
            results.extend([np.einsum(estr, *[thetas[d] for d in ws], *[zs]*n, optimize=epath) \
                            for n,ws,estr,epath in self.connected_iterable])
        return results

    def batch_compute(self, events, pt_ind=0, ang_inds=(1,2), betas=[1], concat=True):
        batch_results = []
        for event in events:
            np_event = event if type(event) == np.ndarray else np.asarray(event)
            results = self.compute(np_event[:,pt_ind], np_event[:,ang_inds], betas=betas)
            batch_results.append(results)
        if concat:
            return calc_disc(batch_results, self.disc_formulae, concat=True)
        else: 
            return np.asarray(batch_results)
