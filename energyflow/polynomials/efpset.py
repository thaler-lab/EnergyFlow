"""Implementation of EFPSet, an efficient collection of EFPs."""

from __future__ import absolute_import, division, print_function
import itertools
import os
import re
import warnings
import numpy as np

from energyflow.multigraphs import Generator
from energyflow.polynomials.base import EFPBase, EFPElem
from energyflow.utils import explicit_comp, graph_union, kwargs_check

__all__ = ['EFPSet']

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
default_file = os.path.join(data_dir, 'multigraphs_d_le_10_numpy.npz')

class EFPSet(EFPBase):

    """A class that holds a collection of EFPs and computes their values on events."""

    # EFPSet(*args, filename=None, measure='hadr', beta=1, normed=True, 
    #        check_type=False, verbose=False)
    def __init__(self, *args, **kwargs):

        """
        EFPSet can be initialized in one of three ways (in order of precedence):

        1. *Generator* - Pass in a custom `Generator` object as the first positional argument.
        2. *Custom File* - Pass in the name of a `.npz` file saved with a custom `Generator`.
        3. *Default* - Use the EFPs that come installed with the `EnergFlow` package.

        To control which EFPs are included, `EFPSet` accepts an arbitrary number of specifications
        (see `sel`) and only EFPs meeting each specification are included in the set. 

        Arguments
        ---------
        *args : arbitrary positional arguments
            - If the first positional argument is a `Generator` instance, it is used for
            initialization. The remaining positional arguments must be valid arguments to `sel`.
        filename : string
            - Path to a `.npz` file which has been saved by a valid `energyflow.Generator`.
        measure : string
            - One of `'hadr'`, `'hadr-dot'`, `'ee'`. 
            See [Measures](/intro/measures) additional info.
        beta : float
            - The parameter $\\beta$ appearing in the measure. Must be greater than zero.
        normed : bool
            - Controls normalization of the energies in the measure.
        check_type : bool
            - Whether to check the type of the input each time or use the first input type.
        verbose : bool
        """

        default_kwargs = {'filename': None, 
                          'measure':'hadr',
                          'beta': 1,
                          'normed': True,
                          'check_type': False, 
                          'verbose': False}

        for k,v in default_kwargs.items():
            setattr(self, k, kwargs.pop(k, v))
        kwargs_check('__init__', kwargs)

        # initialize EFPBase
        super().__init__(self.measure, self.beta, self.normed, self.check_type)

        if len(args) >= 1 and isinstance(args[0], Generator):
            constructor_attrs = ['cols', 'specs', 'disc_formulae', 'edges', 
                                 'einstrs', 'einpaths', 'weights']
            constructor_dict = {attr: getattr(args[0], attr) for attr in constructor_attrs}
            args = args[1:]
        elif self.filename is not None:
            self.filename += '.npz' if not self.filename.endswith('.npz') else ''
            constructor_dict = dict(np.load(self.filename))
        else:
            constructor_dict = dict(np.load(default_file))

        self._sel_re = re.compile('(\w+)(<|>|==|!=|<=|>=)(\d+)$')
        
        # put column headers and indices into namespace
        cols = constructor_dict['cols']
        self.cols = cols if isinstance(cols, list) else cols.tolist()
        self.__dict__.update({col+'_ind': i for i,col in enumerate(self.cols)})

        # get efps which will be kept
        orig_specs = constructor_dict['specs']
        self.stored_specs = orig_specs[self.sel(*args, specs=orig_specs)]

        disc_mask = self.sel(*args, specs=orig_specs[orig_specs[:,self.p_ind]>1])
        self.disc_formulae = constructor_dict['disc_formulae'][disc_mask]

        # make EFPElem list
        edges = constructor_dict['edges']
        einstrs = constructor_dict['einstrs']
        einpaths = constructor_dict['einpaths']
        weights = constructor_dict['weights']
        self.efpelems = []
        for cspec in self.stored_specs[self.stored_specs[:,self.p_ind]==1]:
            k, g, w = cspec[[self.k_ind, self.g_ind, self.w_ind]]
            self.efpelems.append(EFPElem(edges[g], weights=weights[w], einstr=einstrs[g], 
                                                   einpath=einpaths[g], k=k))

        self._set_compute_mask()

        if self.verbose:
            num_prime = self.count('p==1', specs=orig_specs)
            num_composite = self.count('p>1', specs=orig_specs)
            print('Originally Available EFPs:')
            self._print_efp_nums(orig_specs)
            if len(args) > 0:
                print('Current Stored EFPs:')
                self._print_efp_nums()

    def _print_efp_nums(self, specs=None):
        if specs is None:
            specs = self.specs
        num_prime = self.count('p==1', specs=specs)
        num_composite = self.count('p>1', specs=specs)
        print('  Prime:', num_prime)
        print('  Composite:', num_composite)
        print('  Total: ', num_prime+num_composite)

    @property
    def specs(self):
        return self.stored_specs[self.compute_mask]

    def _make_graphs(self, connected_graphs):
        disc_comps = [[connected_graphs[i] for i in col_inds] for col_inds in self.disc_col_inds]
        return np.asarray(connected_graphs + [graph_union(*dc) for dc in disc_comps])

    def graphs(self, *args):
        if not hasattr(self, '_graphs'):
            self._graphs = self._make_graphs([elem.edges for elem in self.efpelems])
        mask = self.sel(*args)
        return [g for g,m in zip(self._graphs, mask) if m]

    def simple_graphs(self, *args):
        if not hasattr(self, '_simple_graphs'):
            self._simple_graphs = self._make_graphs([elem.simple_edges for elem in self.efpelems])
        mask = self.sel(*args)
        return [g for g,m in zip(self._simple_graphs, mask) if m]

    # _set_compute_mask(*args, mask=None)
    def _set_compute_mask(self, *args, **kwargs):
        mask = kwargs.pop('mask', None)
        kwargs_check('set_compute_mask', kwargs)
        if mask is None:
            self.compute_mask = np.ones(len(self.stored_specs), dtype=bool)
            mask = self.sel(*args)
        elif len(mask) != len(self.stored_specs):
            raise IndexError('length of mask does not match internal specs')
        self.compute_mask = mask

        self._weight_set = frozenset(w for efpelem in self.efpelems for w in efpelem.weight_set)
        connected_ndk = [efpelem.ndk for efpelem in self._efpelems_iterator()]

        # get col indices for disconnected formulae
        self.disc_col_inds = []
        for formula in self.disc_formulae[self.compute_mask[len(self.efpelems):]]:
            try:
                self.disc_col_inds.append([connected_ndk.index(factor) for factor in formula])
            except ValueError:
                warnings.warn('connected efp needed for {} not found'.format(formula))

    def _efpelems_iterator(self):
        for cm,elem in zip(self.compute_mask,self.efpelems):
            if cm:
                yield elem

    def _compute_func(self, args):
        return self.compute(zs=args[0], thetas=args[1], batch_call=True)

    @property
    def weight_set(self):
        return self._weight_set

    # compute(event=None, zs=None, thetas=None)
    def compute(self, event=None, zs=None, thetas=None, batch_call=False):
        zs, thetas_dict = self._get_zs_thetas_dict(event, zs, thetas)
        results = [efpelem.compute(zs, thetas_dict) for efpelem in self._efpelems_iterator()]
        if batch_call:
            return results
        else:
            return self.calc_disc(results, concat=True)

    def batch_compute(self, events=None, zs=None, thetas=None, n_jobs=-1, calc_all=True):
        results = super().batch_compute(events, zs, thetas, n_jobs)

        if calc_all:
            return self.calc_disc(results, concat=True)
        else:
            return results

    # sel(*args, specs=None)
    def sel(self, *args, **kwargs):
        specs = kwargs.pop('specs', None)
        kwargs_check('sel', kwargs)
        if specs is None:
            specs = self.specs
        mask = np.ones(len(specs), dtype=bool)
        for arg in args:
            if isinstance(arg, str):
                s = arg.replace(' ', '')
            elif hasattr(arg, '__getitem__'):
                if len(arg) == 2:
                    s = arg[0].replace(' ', '') + str(arg[1])
                else:
                    raise ValueError('{} is not length 2'.format(arg))
            else:
                raise TypeError('invalid type for {}'.format(arg))
            match = self._sel_re.match(s)
            if match is None:
                raise ValueError('could not understand \'{}\''.format(arg))
            var = match.group(1)
            if var not in self.cols:
                raise ValueError('\'{}\' not in {}'.format(var, self.cols))
            comp, val = match.group(2, 3)
            mask &= explicit_comp(specs[:,getattr(self, var+'_ind')], comp, int(val))
        return mask

    # count(*args, specs=None)
    def count(self, *args, **kwargs):
        return np.count_nonzero(self.sel(*args, **kwargs))

    def calc_disc(self, X, concat=False):

        if len(self.disc_col_inds) == 0:
            return X if concat else None

        XX = X
        if not isinstance(X, np.ndarray):
            XX = np.asarray(X)

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

        if concat: 
            return np.concatenate([XX, results], axis=concat_axis)
        else: 
            return results
