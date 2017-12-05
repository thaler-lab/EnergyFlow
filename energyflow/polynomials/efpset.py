from __future__ import absolute_import, division, print_function
import re
import numpy as np

from energyflow.polynomials.base import EFPBase

__all__ = ['EFPSet']

class EFPSet(EFPBase):
    
    def __init__(self, filename, *args, measure='hadr_yphi', beta=1.0, normed=True, 
                                        check_type=False, verbose=True):

        # initialize base class
        EFPBase.__init__(self, measure, beta, normed, check_type)
        self.verbose = verbose

        # compile regular expression to use for selecting subsets of efps
        self._spec_re = re.compile('(\w+)(<|>|==|!=|<=|>=)(\d+)')

        # handle filename with or without extension
        self.filename = filename if '.npz' == filename[-4:] else filename+'.npz'

        # read in file
        fdict = np.load(self.filename)

        # put column indices into namespace
        self.cols = fdict['cols'].tolist()
        self.__dict__.update({col+'_ind': i for i,col in enumerate(self.cols)})

        # store current specs
        self.specs = fdict['specs']

        # get efp selection mask  from arguments
        mask = self.sel(*args)

        connected_inds = np.nonzero(mask & (specs[:,self.p_ind] == 1))[0]
        con_inds_dict = {old_ind: new_ind for new_ind,old_ind in enumerate(connected_inds)}
        disc_inds = np.nonzero(mask & (specs[:,self.p_ind] > 1))[0] - num_connected
        self.disc_formulae = [[con_inds_dict[old_ind] for old_ind in formula] \
                              for formula in fdict['disc_formulae'][disc_inds]]
        
        self.specs = specs[mask]
        gs, ws = set(self.specs[:,self.g_ind]), set(self.specs[:,self.w_ind])
        gs.discard(-1)
        ws.discard(-1)
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
            if (self.dmax == f_dmax and self.nmax == f_nmax and
                self.emax == f_emax and self.cmax == f_cmax):
                print('\nUsing specifications from file')
            else:
                print('\nAfter applying specifications:')
                print('    Total EFPs:', len(self.specs))
                print('    Total Prime EFPs:', np.count_nonzero(self.specs[:,self.p_ind] == 1))




        self._load_file(dmax, nmax, emax, cmax)
        self._make_connected_iterable()

    def _load_file(self, dmax, nmax, emax, cmax):

        fdict = np.load(self.filename)
        specs = fdict['specs']
        self.cols = fdict['cols'].tolist()
        self.__dict__.update({col+'_ind': i for i,col in enumerate(self.cols)})

        f_dmax, f_nmax = np.max(specs[:,self.d_ind]), np.max(specs[:,self.n_ind])
        f_emax, f_cmax = np.max(specs[:,self.e_ind]), np.max(specs[:,self.c_ind])
        num_connected = np.count_nonzero(specs[:,self.p_ind] == 1)

        if self.verbose:
            print('Loading EFPs from {}:'.format(self.filename))
            print('    Total EFPs:', len(specs))
            print('    Total Prime EFPs:', num_connected)
            print('    Maximum d:', f_dmax)
            print('    Maximum N:', f_nmax)
            print('    Maximum chi:', f_cmax)
            print('    Maximum e:', f_emax)
            print('    ve_alg:', fdict['ve_alg'])

        self.dmax = f_dmax if dmax is None else dmax
        self.nmax = f_nmax if nmax is None else nmax
        self.emax = f_emax if emax is None else emax
        self.cmax = f_cmax if cmax is None else cmax

        mask = ((specs[:,self.d_ind] <= self.dmax)&(specs[:,self.n_ind] <= self.nmax)
               &(specs[:,self.e_ind] <= self.emax)&(specs[:,self.c_ind] <= self.cmax))

        connected_inds = np.nonzero(mask & (specs[:,self.p_ind] == 1))[0]
        con_inds_dict = {old_ind: new_ind for new_ind,old_ind in enumerate(connected_inds)}
        disc_inds = np.nonzero(mask & (specs[:,self.p_ind] > 1))[0] - num_connected
        self.disc_formulae = [[con_inds_dict[old_ind] for old_ind in formula] \
                              for formula in fdict['disc_formulae'][disc_inds]]
        
        self.specs = specs[mask]
        gs, ws = set(self.specs[:,self.g_ind]), set(self.specs[:,self.w_ind])
        gs.discard(-1)
        ws.discard(-1)
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
            if (self.dmax == f_dmax and self.nmax == f_nmax and
                self.emax == f_emax and self.cmax == f_cmax):
                print('\nUsing specifications from file')
            else:
                print('\nAfter applying specifications:')
                print('    Total EFPs:', len(self.specs))
                print('    Total Prime EFPs:', np.count_nonzero(self.specs[:,self.p_ind] == 1))

    def _make_connected_iterable(self):
        self.connected_iterable = [(spec[self.n_ind],
                                    self.weights[spec[self.w_ind]],
                                    self.einstrs[spec[self.g_ind]],
                                    self.einpaths[spec[self.g_ind]]) \
                                   for spec in self.connected_specs]

    def sel(self, *args):
        mask = np.ones(len(self.specs), dtype=bool)
        for arg in args:
            if isinstance(arg, str):
                s = arg.replace(' ', '')
            elif isinstance(arg, (tuple, list)):
                s = arg[0].replace(' ', '') + str(arg[1])
            match = self._spec_re.match(s)
            if match is None:
                raise ValueError('could not understand \'{}\''.format(arg))
            var = match.group(1)
            if var not in self.cols:
                raise ValueError('\'{}\' not in {}'.format(var, self.cols))
            mask &= eval('self.specs[:,self.{}_ind] {} {}'.format(var, *match.group(2,3)))
        return mask

    def count(self, *args):
        return np.count_nonzero(self.sel(*args))

    def compute(self, pts, yphis, beta=1):
        zs = pts/np.sum(pts)
        thetas2 = np.sum((yphis[:,np.newaxis] - yphis[np.newaxis,:])**2, axis=-1)

        thetas = {d: thetas2**(beta*d/2) for d in range(1, self.dmax + 1)}
        return [np.einsum(estr, *([thetas[d] for d in ws]+[zs]*n), optimize=epath) \
                                   for n,ws,estr,epath in self.connected_iterable]

    def _computestar(self, args):
        return self.compute(*args)

    def calc_disc(self, X, concat=False):

        results = np.ones((len(X), len(self.disc_formulae)), dtype=float)
        XX = X if type(X) == np.ndarray else np.asarray(X)

        for i,formula in enumerate(self.disc_formulae):
            for col_ind in formula:
                results[:,i] *= XX[:,col_ind]

        if concat: 
            return np.concatenate([XX, results], axis=1)
        else: 
            return results
