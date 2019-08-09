""""""
from __future__ import absolute_import, division, print_function

import gc
import json
import math
import os
import re
import sys
import time
import warnings

import h5py
import numpy as np
import six

from energyflow.utils.data_utils import _get_filepath
from energyflow.utils.generic_utils import COMP_MAP, EF_DATA_DIR, REVERSE_COMPS
from energyflow.utils import create_pool, explicit_comp

__all__ = ['MODDataset', 'load', 'filter_particles']

ZENODO_URL_PATTERN = 'https://zenodo.org/record/{}/files/{}?download=1'

COLLECTIONS = {
    'CMS2011AJets': {
        'cms': {
            'subdatasets': [('CMS_Jet300_pT375-infGeV', 18, '3340205')],
        },

        'sim': {
            'subdatasets': [
                ('SIM170_Jet300_pT375-infGeV', 1, '3341500'),
                ('SIM300_Jet300_pT375-infGeV', 24, '3341498'),
                ('SIM470_Jet300_pT375-infGeV', 73, '3341419'),
                ('SIM600_Jet300_pT375-infGeV', 78, '3364139'),
                ('SIM800_Jet300_pT375-infGeV', 79, '3341413'),
                ('SIM1000_Jet300_pT375-infGeV', 40, '3341502'),
                ('SIM1400_Jet300_pT375-infGeV', 40, '3341770'),
                ('SIM1800_Jet300_pT375-infGeV', 20, '3341772'),
            ]
        },

        'gen': {
            'subdatasets': [
                ('GEN170_pT375-infGeV', 1, '3341500'),
                ('GEN300_pT375-infGeV', 24, '3341498'),
                ('GEN470_pT375-infGeV', 74, '3341419'),
                ('GEN600_pT375-infGeV', 79, '3364139'),
                ('GEN800_pT375-infGeV', 79, '3341413'),
                ('GEN1000_pT375-infGeV', 40, '3341502'),
                ('GEN1400_pT375-infGeV', 40, '3341770'),
                ('GEN1800_pT375-infGeV', 20, '3341772'),
            ]
        }
    }
}

def filter_particles(particles, which='all', pt_cut=None, chs=False, pt_i=0, pid_i=4, vertex_i=5):
    
    mask = np.ones(len(particles), dtype=bool)
    
    # pt cut
    if pt_cut is not None:
        mask &= (particles[:,pt_i] >= pt_cut)
        
    # select specified particles
    if which != 'all':
        chrg_mask = ef.ischrgd(particles[:,pid_i])
        
        if which == 'charged':
            mask &= chrg_mask
        else:
            mask &= ~chrg_mask
            
    # apply chs
    if chs:
        mask &= (particles[:,vertex_i] <= 0)
        
    return mask

def kfactors(dataset, pts, npvs=None, collection='CMS2011AJets', apply_residual_correction=True):

    # verify dataset
    if dataset not in {'sim', 'gen'}:
        raise ValueError("dataset must be one of 'sim' or 'gen'")

    # get info for the specified collection
    info = _get_dataset_info(collection)

    # base kfactors from https://arxiv.org/abs/1309.5311
    base_kfactors = np.interp(pts, info['kfactor_x'], info['kfactor_y'])

    # include npv reweighting if sim
    if dataset == 'sim':

        # verify we have npvs
        if npvs is None:
            raise ValueError("npvs cannot be None when dataset is 'sim'")

        base_kfactors *= info['npv_hist_ratios'][np.asarray(npvs, dtype=int)]

    # apply residual factor if desired
    if apply_residual_correction:
        base_kfactors *= info['residual_factor']

    return base_kfactors

def _get_collection(cname):

    # verify collection
    if cname not in COLLECTIONS:
        raise ValueError("Collection '{}' not recognized".format(cname))

    return COLLECTIONS[cname]

def _get_dataset_info(cname):

    # get collection
    collection = _get_collection(cname)

    # cache info if not already stored
    if 'info' not in collection:

        fpath = os.path.join(EF_DATA_DIR, '{}.json'.format(cname))
        with open(fpath, 'r') as f:
            info = json.load(f)

        collection['info'] = info

    return collection['info']

def _separate_particle_arrays(particles, particles_index, mask, copy=True):
    
    # array to hold particles
    particles_array = np.zeros(np.count_nonzero(mask), dtype='O')
    
    # iterate over indices
    n = 0
    for start, end, m in zip(particles_index[:-1], particles_index[1:], mask):
        if m:
            particles_array[n] = np.array(particles[start:end], copy=copy)
            n += 1
        
    return particles_array

def _make_particles_index(particle_arrays):
    
    # list of indices
    index = [0]

    # iterate over all particles
    for particles in particle_arrays:
        index.append(index[-1] + len(particles))

    # convert to numpy array with proper dtype
    return np.asarray(index, dtype=np.uint32)

def _write_large_object_array_to_h5(hf, name, arr, dtype=None, ncols=None, 
                                                   chunksize=10**5, **compression):

    nrows = sum([len(x) for x in arr])
    ncols = arr[0].shape[1] if ncols is None else ncols
    dtype = arr[0].dtype if dtype is None else dtype

    dataset = hf.create_dataset(name, (nrows, ncols), dtype=dtype, **compression)

    begin = end = ind = 0
    while end < len(arr):
        end = min(len(arr), end + chunksize)

        arr_chunk = np.concatenate(arr[begin:end], axis=0)
        dataset[ind:ind+len(arr_chunk)] = arr_chunk

        begin = end
        ind += len(arr_chunk)
        del arr_chunk

    return dataset

def _process_selections(sel_list):
    sels = []
    for sel in sel_list:
        if isinstance(sel, six.string_types):
            sels.append(sel)
        else:
            sels.append(''.join([str(s) for s in sel]))

    return '&'.join(sels)

def _process_selection_match(groups):
    name = groups[3] + 's'

    specs = []
    for i in [0,4]:
        val, cf, ci = groups[i:i+3]
        if val != '':
            if cf != '':
                c = cf
                val = float(val.replace(cf, ''))
            elif ci != '':
                c = ci
                val = int(val.replace(ci, ''))
            else:
                raise ValueError('Invalid groups from selection: {}'.format(groups))

            # handle reversals
            if i == 0:
                c = REVERSE_COMPS.get(c, c)

            specs.append((name, c, val))

    if len(specs) == 0:
        raise ValueError('Invalid groups from selection: {}'.format(groups))

    return specs

def _moddset_save(arg):
    i, filepath, compression = arg
    moddsets[i].save(filepath, compression=compression, verbose=0)

class MODDataset(object):

    def __init__(self, *args, **kwargs):

        default_kwargs = {
            'num': -1,
            'path': None,
            'shuffle': True,
            'store_gens': True,
            'store_pfcs': True,
        }

        # process kwargs
        for k,v in default_kwargs.items():
            if k not in kwargs:
                kwargs[k] = v
        
        # store options
        self.num = kwargs.pop('num')
        self.shuffle = kwargs.pop('shuffle')
        self.store_pfcs = kwargs.pop('store_pfcs')
        self.store_gens = kwargs.pop('store_gens')

        # check for disallowed kwargs
        other_allowed_kwargs = {'_arrays', '_dataset', 'datasets', 'path'}
        for kw in kwargs:
            if kw not in other_allowed_kwargs:
                raise TypeError("__init__() got an unexpected keyword argument '{}'".format(kw))

        # initialize from explicit arrays (used only when making files initially)
        if len(args) == 0 and '_arrays' in kwargs and '_dataset' in kwargs:
            self._init_from_arrays(kwargs['_dataset'], kwargs['_arrays'])

        # initialize from list of datasets
        elif 'datasets' in kwargs:
            self.selection = _process_selections(args)
            self._init_from_datasets(kwargs['datasets'])

        # initialize from file
        elif len(args) and isinstance(args[0], six.string_types):
            self.selection = _process_selections(args[1:])
            self._init_from_filename(args[0], kwargs['path'])

        else:
            raise RuntimeError('Initialization of MODDataset not understood')

    def _store_dataset_info(self, dataset):

        assert dataset in ['cms', 'sim', 'gen'], "Dataset must be one of ['cms', 'sim', 'gen']"
        self.dataset = dataset

        self.cms = (self.dataset == 'cms')
        self.sim = (self.dataset == 'sim')
        self.gen = (self.dataset == 'gen')

        # update options based on dataset type
        self.store_pfcs &= not self.gen
        self.store_gens &= not self.cms

    def _store_cols(self, dset, cols=None, allow_multiple=False):

        # get cols from file
        if cols is None:
            cols = self.hf[dset].attrs['cols']

        cols = np.asarray(cols, dtype='U')
        setattr(self, dset + '_cols', cols)

        for i,col in enumerate(cols):

            # ensure cols are unique
            if not allow_multiple:
                m = "Repeat instances of col '{}', check file validity".format(col)
                assert not hasattr(self, col), m

            # store column index
            setattr(self, col, i)

    def _store_views_of_jets(self):

        for jets in ['jets_i', 'jets_f']:

            # retrieve array, cols
            arr, cols = getattr(self, jets), getattr(self, jets + '_cols')

            for i,col in enumerate(cols):

                # set attribute + s as view of this column of array
                setattr(self, col + 's', arr[:,i])

        # calculate corrected pts
        self.corr_jet_pts = self.jet_pts*self.jecs if hasattr(self, 'jecs') else self.jet_pts

    def sel(self, *args, **kwargs):

        selection = kwargs.pop('_selection', None)
        for kw in kwargs:
            raise ValueError("Unknown keyword argument '{}'".format(kw))

        return_specs = False
        if selection is None:
            selection = _process_selections(args)
        elif len(args) == 0:
            return_specs = True
        else:
            raise ValueError("args cannot be set when using '_selection'")

        # make mask which is all true
        mask = np.ones(self.jets_i.shape[0], dtype=bool)

        # valid columns to select from
        if not hasattr(self, 'selection_cols'):
            self.selection_cols = self.jets_f_cols.tolist() + self.jets_i_cols.tolist()
            self.selection_cols += ['corr_jet_pt', 'abs_jet_eta', 'abs_jet_y']
            
            # handle special cases for sim
            if 'get_jet_eta' in self.selection_cols:
                self.selection_cols += ['abs_gen_jet_eta', 'abs_gen_jet_y']

            # special cases for gen
            if self.gen:
                self.selection_cols += ['quality']

        if not len(selection):
            return (mask, []) if return_specs else mask

        # regular expression for selection
        if not hasattr(self, '_sel_re'):
            cols_re = '|'.join(self.selection_cols)
            comps_re = '|'.join(COMP_MAP.keys())
            expr = (r'\s*(-?(?:\d*\.\d*|inf)\s*({0})|-?\d+\s*({0}))?'
                    r'\s*({1})s?'
                    r'\s*(({0})\s*-?(?:\d*\.\d*|inf)|({0})\s*-?\d+)?\s*(&\s*|$)').format(comps_re, cols_re)
            self._sel_re = re.compile(expr)
            self._sel_re_check = re.compile('(?:{})+'.format(expr))

        # check that we overall have a valid selection
        if self._sel_re_check.fullmatch(selection) is None:
            raise ValueError("Selection '{}' not understood".format(selection))

        # iterate over selections
        specs = []
        for match in self._sel_re.findall(selection):
            specs.extend(_process_selection_match(match))

        # apply specs
        for spec in specs:
            name = spec[0]
            if 'abs_' in name:
                arr = np.abs(getattr(self, name.replace('abs_', '')))
            elif 'quality' in name and self.gen:
                continue
            else:
                arr = getattr(self, name)

            mask &= explicit_comp(arr, spec[1], spec[2])

        return (mask, specs) if return_specs else mask

    def _set_particles(self):

        if self.store_pfcs and not self.gen:
            self.particles = self.pfcs
            self.particles_cols = self.pfcs_cols

        if self.store_gens and self.gen:
            self.particles = self.gens
            self.particles_cols = self.gens_cols

    def close(self):
        if hasattr(self, 'hf'):
            self.hf.close()

    def __del__(self):

        # close file 
        self.close()

        # delete arrays
        if hasattr(self, 'jets_i'):
            del self.jets_i
        if hasattr(self, 'jets_f'):
            del self.jets_f
        if hasattr(self, 'filenames'):
            del self.filenames

        # delete pfcs if they exist
        if hasattr(self, 'pfcs'):
            del self.pfcs

        # delete gens if they exist
        if hasattr(self, 'gens'):
            del self.gens

        # force garbage collection
        gc.collect()

    def __len__(self):
        return len(self.jets_i)

    def _init_from_arrays(self, dataset, arrays):

        # update options
        self.store_pfcs &= 'pfcs' in arrays
        self.store_gens &= 'gens' in arrays

        # store dataset info by hand
        self._store_dataset_info(dataset)

        # jets arrays
        self.jets_i, self.jets_f = arrays['jets_i'], arrays['jets_f']

        # jets cols
        self._store_cols('jets_i', arrays['jets_i_cols'])
        self._store_cols('jets_f', arrays['jets_f_cols'])

        # store views of jets cols
        self._store_views_of_jets()

        # sum all weights
        self._orig_total_weight = np.sum(self.weights)

        # pfcs
        if self.store_pfcs:
            self.pfcs = arrays['pfcs']
            self._store_cols('pfcs', arrays['pfcs_cols'])

        # gens
        if self.store_gens:
            self.gens = arrays['gens']
            self._store_cols('gens', arrays['gens_cols'], allow_multiple=self.store_pfcs)

        # filenames
        self.filenames = np.asarray(arrays['filenames'], dtype='U')

        # set particles
        self._set_particles()

    def _init_from_filename(self, filename, path):

        # handle suffix
        if not filename.endswith('.h5'):
            filename += '.h5'

        # get filepath
        self.filepath = filename if path is None else os.path.join(path, filename)

        # determine type of dataset
        filename_lower = os.path.basename(self.filepath).lower()
        dataset = ('cms' if 'cms' in filename_lower else 
                  ('sim' if 'sim' in filename_lower else
                  ('gen' if 'gen' in filename_lower else None)))

        # store dataset info
        self._store_dataset_info(dataset)

        # open h5 file
        self.hf = h5py.File(self.filepath, 'r')

        # load selected jets
        self.jets_i = self.hf['jets_i'][:]
        self.jets_f = self.hf['jets_f'][:]

        # update store particles based on availability
        self.store_pfcs &= ('pfcs' in self.hf)
        self.store_gens &= ('gens' in self.hf)

        # store jets cols
        self._store_cols('jets_i')
        self._store_cols('jets_f')

        # store views of jets cols
        self._store_views_of_jets()

        # sum all weights
        self._orig_total_weight = np.sum(self.weights)

        # process selections
        self.mask, self.specs = self.sel(_selection=self.selection)

        # determine weight factor caused by subselecting
        total_weight_after_selections = np.sum(self.weights[self.mask])

        # select the requested number of jets
        if self.num != -1:

            # shuffle if requested
            arange = np.arange(len(self.mask))[self.mask]
            if self.shuffle:
                np.random.shuffle(arange)

            # mask out jets beyond what was requested
            self.mask[arange[self.num:]] = False

            # weight factor
            weight_factor = total_weight_after_selections/np.sum(self.weights[self.mask])

        else:
            weight_factor = 1.0

        # apply mask to jets
        self.jets_i = self.jets_i[self.mask]
        self.jets_f = self.jets_f[self.mask]

        # alter weights due to subselection
        self.jets_f[:,self.weight] *= weight_factor

        # store views of jets cols
        self._store_views_of_jets()

        if self.store_pfcs:

            # read in pfcs_index
            self.pfcs_index = self.hf['pfcs_index'][:]

            # store pfcs as separate arrays
            self.pfcs = _separate_particle_arrays(self.hf['pfcs'][:], self.pfcs_index, self.mask)

            # store pfcs cols
            self._store_cols('pfcs')

        if self.store_gens:

            # read in gens_index
            self.gens_index = self.hf['gens_index'][:]

            # store gens as separate arrays
            self.gens = _separate_particle_arrays(self.hf['gens'][:], self.gens_index, self.mask)

            # store gens cols
            self._store_cols('gens', allow_multiple=self.store_pfcs)

        # store filenames
        self.filenames = self.hf['filenames'][:].astype('U')

        # set particles
        self._set_particles()

    def _init_from_datasets(self, datasets):

        # lists to hold arrays to concatenate
        jets_i, jets_f = [], []
        pfcs, gens = [], []

        # iterate over arguments
        for i,dataset in enumerate(datasets):

            # ensure they are a MODDataset
            td = type(dataset)
            if td != MODDataset:
                m = "Incorrect type '{}' encountered when initializing from list".format(td)
                raise TypeError(m)

            # get info from first dataset
            if i == 0:

                # extract dataset info
                self._store_dataset_info(dataset.dataset)

                # array info
                self.filenames = dataset.filenames
                self.store_pfcs &= dataset.store_pfcs
                self.store_gens &= dataset.store_gens

                # store jets cols
                self._store_cols('jets_i', dataset.jets_i_cols)
                self._store_cols('jets_f', dataset.jets_f_cols)

                # store specs
                self.specs = dataset.specs

                # pfcs cols
                if self.store_pfcs:
                    self._store_cols('pfcs', dataset.pfcs_cols)

                # gens cols
                if self.store_gens:
                    self._store_cols('gens', dataset.gens_cols, allow_multiple=self.store_pfcs)

            # check for consistency
            else:
                m = "Datasets must all be of the same type ('cms', 'sim', 'gen')"
                assert dataset.dataset == self.dataset, m
                assert np.all(dataset.filenames == self.filenames), 'filenames must match'
                assert np.all(dataset.jets_i_cols == self.jets_i_cols), 'jets_i_cols must match'
                assert np.all(dataset.jets_f_cols == self.jets_f_cols), 'jets_f_cols must match'

                if self.store_pfcs:
                    assert np.all(dataset.pfcs_cols == self.pfcs_cols), 'pfcs_cols must match'

                if self.store_gens:
                    assert np.all(dataset.gens_cols == self.gens_cols), 'gen_cols must match'

            # store jets
            jets_i.append(dataset.jets_i)
            jets_f.append(dataset.jets_f)

            # store pfcs
            if self.store_pfcs:
                pfcs.append(dataset.pfcs)

            # store gens
            if self.store_gens:
                gens.append(dataset.gens)

        # concatenate jets
        self.jets_i = np.concatenate(jets_i, axis=0)
        self.jets_f = np.concatenate(jets_f, axis=0)

        # store views of jets cols
        self._store_views_of_jets()

        # sum all weights
        self._orig_total_weight = np.sum(self.weights)

        if self.store_pfcs:
            self.pfcs = np.concatenate(pfcs, axis=0)

        if self.store_gens:
            self.gens = np.concatenate(gens, axis=0)

        # set particles
        self._set_particles()

    def apply_mask(self, mask, preserve_total_weight=False):

        if len(mask) != len(self):
            raise IndexError('Incorrectly sized mask')

        if preserve_total_weight:
            total_weight_before_mask = np.sum(self.weights)

        self.jets_i = self.jets_i[mask]
        self.jets_f = self.jets_f[mask]

        if preserve_total_weight:
            weight_factor = total_weight_before_mask/np.sum(self.jets_f[:,self.weight])
            self.jets_f[:,self.weight] *= weight_factor

        self._store_views_of_jets()

        if self.store_pfcs:
            self.pfcs = self.pfcs[mask]

        if self.store_gens:
            self.gens = self.gens[mask]

        # set particles
        self._set_particles()

    def save(self, filepath, npf=-1, compression=None, verbose=1, n_jobs=1):

        path, name = os.path.split(filepath)
        if name.endswith('.h5'):
            name = '.'.join(name.split('.')[:-1])

        start = time.time()
        if npf != -1:

            i = begin = end = 0
            global moddsets
            args, moddsets = [], []
            while end < len(self.jets_i):
                end = min(end + npf, len(self.jets_i))

                arrays = {'jets_i': self.jets_i[begin:end], 'jets_i_cols': self.jets_i_cols,
                          'jets_f': self.jets_f[begin:end], 'jets_f_cols': self.jets_f_cols,
                          'filenames': self.filenames}

                if self.store_pfcs:
                    arrays['pfcs'] = self.pfcs[begin:end]
                    arrays['pfcs_cols'] = self.pfcs_cols

                if self.store_gens:
                    arrays['gens'] = self.gens[begin:end]
                    arrays['gens_cols'] = self.gens_cols

                filepath = os.path.join(path, '{}_{}'.format(name, i))

                moddset = self.__class__(_dataset=self.dataset, _arrays=arrays)

                if n_jobs == 1:
                    moddset.save(filepath, compression=compression, verbose=verbose)

                else:
                    moddsets.append(moddset)
                    args.append((i, filepath, compression)) 

                begin = end
                i += 1

            if n_jobs != 1:
                if verbose >= 1:
                    l = len(args)
                    pf = (l, 's' if l > 1 else '', time.time() - start)
                    print('Constructed {} temporary MODDataset{} in {:.3f}s'.format(*pf))

                if n_jobs == -1:
                    n_jobs = os.cpu_count()
                    if n_jobs is None:
                        n_jobs = 4

                start = time.time()
                with create_pool(processes=min(n_jobs, len(args))) as pool:
                    for i,_ in enumerate(pool.imap_unordered(_moddset_save, args, chunksize=1)):
                        if verbose >= 1 and ((i+1) % 5 == 0 or i+1 == len(args)):
                            pf = (i+1, (i+1)/len(args)*100, time.time() - start)
                            print('  Saved {} files, {:.2f}% done in {:.3f}s'.format(*pf))

                    del moddsets
                    
            return

        # compression opts
        compression = ({'compression': 'gzip', 'compression_opts': compression} 
                       if compression is not None else {})
        comp_str = '_compressed' if len(compression) else ''

        # ensure directory exists
        if not os.path.exists(path):
            if verbose >= 2:
                print('Creating', path)
            os.mkdir(path)

        if verbose >= 2:
            print('Saving to', path)

        filename = name + comp_str
        hf = h5py.File(os.path.join(path, filename + '.h5'), 'w')

        # jets_i
        jets_i = hf.create_dataset('jets_i', data=self.jets_i, **compression)
        jets_i.attrs.create('cols', np.asarray(self.jets_i_cols, dtype='S'))

        # jets_f
        jets_f = hf.create_dataset('jets_f', data=self.jets_f, **compression)
        jets_f.attrs.create('cols', np.asarray(self.jets_f_cols, dtype='S'))

        # pfcs
        if self.store_pfcs:
            pfcs = _write_large_object_array_to_h5(hf, 'pfcs', self.pfcs, 
                                                   ncols=len(self.pfcs_cols), **compression)
            pfcs.attrs.create('cols', np.asarray(self.pfcs_cols, dtype='S'))
            hf.create_dataset('pfcs_index', data=_make_particles_index(self.pfcs), **compression)

        # gens
        if self.store_gens:
            gens = _write_large_object_array_to_h5(hf, 'gens', self.gens, 
                                                   ncols=len(self.gens_cols), **compression)
            gens.attrs.create('cols', np.asarray(self.gens_cols, dtype='S'))
            hf.create_dataset('gens_index', data=_make_particles_index(self.gens), **compression)

        # filenames
        hf.create_dataset('filenames', data=self.filenames.astype('S'), **compression)

        # close
        hf.close()

        if verbose >= 1:
            args = (filename, len(self.jets_i), time.time() - start)
            print('  Saved {} with {} jets in {:.3f}s'.format(*args))

def load(*args, **kwargs):

    default_kwargs = {
        'amount': 1,
        'cache_dir': '~/.energyflow',
        'validate_files': False,
        'collection': 'CMS2011AJets',
        'dataset': 'cms',
        'subdatasets': None,
        'compressed': True,
        'store_gens': True,
        'store_pfcs': True,
        'verbose': 0,
    }

    # process arguments
    for k,v in default_kwargs.items():
        if k not in kwargs:
            kwargs[k] = v

    # store arguments
    amount = kwargs['amount']
    cache_dir = kwargs['cache_dir']
    compressed = kwargs['compressed']
    validate_files = kwargs['validate_files']
    verbose = kwargs['verbose']
    moddset_kwargs = {kw: kwargs[kw] for kw in ['store_gens', 'store_pfcs']}

    # verify collection
    cname = kwargs['collection']
    collection = _get_collection(cname)

    # verify dataset
    dname = kwargs['dataset']
    assert dname in collection, "Dataset must be one of {}".format(list(collection.keys()))
    dataset = collection[dname]

    # determine subdatasets
    if kwargs['subdatasets'] is None:
        subdatasets = dataset['subdatasets']
    else:
        subdatasets = [sdset for sdset in dataset['subdatasets']
                               if sdset[0] in kwargs['subdatasets']]
        allowed_sds = set([sdset[0] for sdset in dataset['subdatasets']])
        remaining_sds = set(kwargs['subdatasets']) - allowed_sds
        if len(remaining_sds):
            raise ValueError('Did not understand the following subdatasets: {}'.format(remaining_sds)
                             + ', acceptable values are {}'.format(allowed_sds))

    # get file info
    info = _get_dataset_info(cname)
    hashes, total_weights = info['md5_hashes'], info['total_weights']

    # iterate over subdatasets
    moddsets = []
    for subdataset in subdatasets:
        name, nfiles, record = subdataset

        # get path to dataset files
        subdir = os.path.join('datasets', cname, name)

        # determine number of files to read in
        if amount == -1:
            nfiles_load = nfiles
        elif isinstance(amount, float):
            nfiles_load = math.ceil(amount * nfiles)
        elif isinstance(amount, int):
            if amount > nfiles:
                warnings.warn('Requested {} files but only have {}'.format(amount, nfiles))
                nfiles_load = nfiles
            else:
                nfiles_load = amount
        else:
            raise ValueError('Amount {} not understood'.format(amount))

        # iterate over files
        modsubdsets = []
        start_subdset = time.time()
        for i in range(nfiles_load):
            start = time.time()
            filename = '{}_{}{}.h5'.format(name, i, '_compressed' if compressed else '')
            file_hash = hashes[filename] if validate_files else None
            url = ZENODO_URL_PATTERN.format(record, filename)

            filepath = _get_filepath(filename, url, cache_dir, cache_subdir=subdir, 
                                                               file_hash=file_hash)

            moddset_args = (filepath,) + args
            modsubdsets.append(MODDataset(*moddset_args, **moddset_kwargs))

            if verbose >= 2:
                print('  Loaded {} in {:.3f}s'.format(name + '_{}'.format(i), time.time() - start))

        if verbose >= 1:
            print('Loaded {} in {:.3f}s'.format(name, time.time() - start_subdset))

        # set weights appropriately in case we're not using all the files
        subdset_total_weight = sum([dset._orig_total_weight for dset in modsubdsets])
        for dset in modsubdsets:
            dset.jets_f[:,dset.weight] *= total_weights[name]/subdset_total_weight

        moddsets.extend(modsubdsets)

    # return concatenated MODDataset
    return MODDataset(datasets=moddsets)
