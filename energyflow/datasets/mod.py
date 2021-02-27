r"""# Datasets

## CMS Open Data and the MOD HDF5 Format

Starting in 2014, the CMS Collaboration began to release research-grade
recorded and simulated datasets on the [CERN Open Data Portal](http://opendata.
cern.ch/). These fantastic resources provide a unique opportunity for
researchers with diverse connections to experimental particle phyiscs world to
engage with cutting edge particle physics by developing tools and testing novel
strategies on actual LHC data. Our goal in making portions of the CMS Open
Data available in a reprocessed format is to ease as best as possible the
technical complications that have thus far been present when attempting to use
Open Data (see also [recent efforts by the CMS Collaboration](http://opendata.
cern.ch/docs/cms-releases-open-data-for-machine-learning) to make the data more
accessible).

To facilitate access to Open Data, we have developed a format utilizing the
widespread [HDF5 file format](https://www.hdfgroup.org/solutions/hdf5/) that
stores essential information for some particle physics analyses. This "MOD HDF5
Format" is currently optimized for studies based on jets, but may be updated in
the future to support other types of analyses.

To further the goals of Open Data, we have made our reprocessed samples
available on the [Zenodo platform](https://zenodo.org/). Currently, the only
"collection" of datasets that is available is `CMS2011AJets`, which was used in
[Exploring the Space of Jets with CMS Open Data](https://arxiv.org/abs/
1908.08542) for [EMD](/docs/emd)-based studies. More collections may be added
in the future as our research group completes more studies with the Open Data.

For now, this page focuses on the `CMS2011AJets` collection. This collection
includes datasets of jets that are CMS-recorded (CMS), Pythia-generated (GEN),
and detector-simulated (SIM), or in code `'cms'`, `'gen'`, `'sim'`,
respectively. The datasets include all available jets above 375 GeV, which is
where the HLT_Jet300 trigger was found to be fully efficient in both data and
simulation. Note that the pT values referenced in the name of the SIM/GEN
datasets are those of the generator-level hard parton. The DOIs of
`CMS2011AJets` MOD HDF5 datasets are:

[![DOI](/img/zenodo.3340205.svg)](https://
doi.org/10.5281/zenodo.3340205) - CMS 2011A Jets, pT > 375 GeV
<br>
[![DOI](/img/zenodo.3341500.svg)](https://
doi.org/10.5281/zenodo.3341500) - SIM/GEN QCD Jets 170-300 GeV
<br>
[![DOI](/img/zenodo.3341498.svg)](https://
doi.org/10.5281/zenodo.3341498) - SIM/GEN QCD Jets 300-470 GeV
<br>
[![DOI](/img/zenodo.3341419.svg)](https://
doi.org/10.5281/zenodo.3341419) - SIM/GEN QCD Jets 470-600 GeV
<br>
[![DOI](/img/zenodo.3364139.svg)](https://
doi.org/10.5281/zenodo.3364139) - SIM/GEN QCD Jets 600-800 GeV
<br>
[![DOI](/img/zenodo.3341413.svg)](https://
doi.org/10.5281/zenodo.3341413) - SIM/GEN QCD Jets 800-1000 GeV
<br>
[![DOI](/img/zenodo.3341502.svg)](https://
doi.org/10.5281/zenodo.3341502) - SIM/GEN QCD Jets 1000-1400 GeV
<br>
[![DOI](/img/zenodo.3341770.svg)](https://
doi.org/10.5281/zenodo.3341770) - SIM/GEN QCD Jets 1400-1800 GeV
<br>
[![DOI](/img/zenodo.3341772.svg)](https://
doi.org/10.5281/zenodo.3341772) - SIM/GEN QCD Jets 1800-$\infty$ GeV

For more details regarding the creation of these samples, as well as for the
DOIs of the original CMS Open Datasets, see [Exploring the Space of Jets with
CMS Open Data](https://arxiv.org/abs/1908.08542). To get started using the
samples, see the [MOD Jet Demo](/demos/#mod-jet-demo) which makes use of the 
[`load`](#load) function.
"""

#  __  __  ____  _____
# |  \/  |/ __ \|  __ \
# | \  / | |  | | |  | |
# | |\/| | |  | | |  | |
# | |  | | |__| | |__| |
# |_|  |_|\____/|_____/

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

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
from energyflow.utils import (COMP_MAP, EF_DATA_DIR, REVERSE_COMPS, ZENODO_URL_PATTERN,
                              create_pool, explicit_comp, ischrgd, kwargs_check)

__all__ = ['MODDataset', 'load', 'filter_particles', 'kfactors']

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

###############################################################################
# PUBLIC FUNCTIONS
###############################################################################

# load(*args, amount=1, cache_dir='~/.energyflow', collection='CMS2011AJets', 
#             dataset='cms', subdatasets=None, validate_files=False,
#             store_pfcs=True, store_gens=True, verbose=0)
def load(*args, **kwargs):
    r"""Loads samples from the specified MOD dataset. Any file that is needed
    that has not been cached will be automatically downloaded from Zenodo.
    Downloaded files are cached for later use. File checksums are optionally
    validated to ensure dataset fidelity.

    **Arguments**

    - ***args** : _arbitrary positional arguments_
        - Used to specify cuts to be made to the dataset while loading; see
        the detailed description of the positional arguments accepted by
        [`MODDataset`](#moddataset).
    - **amount** : _int_ or _float_
        - Approximate amount of the dataset to load. If an integer, this is the
        number of files to load (a warning is issued if more files are
        requested than are available). If a float, this is the fraction of the
        number of available files to load, rounded up to the nearest whole
        number. Note that since ints and floats are treated different, a value
        of `1` loads one file whereas `1.0` loads the entire dataset. A value
        of `-1` also loads the entire dataset. 
    - **cache_dir** : _str_
        - The directory where to store/look for the files. Note that 
        `'datasets'` is automatically appended to the end of this path, as well
        as the collection name. For example, the default is to download/look
        for files in the directory `'~/.energyflow/datasets/CMS2011AJets'`.
    - **collection** : _str_
        - Name of the collection of datasets to consider. Currently the only
        collection is `'CMS2011AJets'`, though more may be added in the future.
    - **dataset** : _str_
        - Which dataset in the collection to load. Currently the
        `'CMS2011AJets'` collection has `'cms'`, `'sim'`, and `'gen'` datasets.
    - **subdatasets** : {_tuple_, _list_} of _str_ or `None`
        - The names of subdatasets to use. A value of `None` uses all available
        subdatasets. Currently, for the `'CMS2011AJets'` collection, the
        `'cms'` dataset has one subdataset, `'CMS_Jet300_pT375-infGeV'`, the
        `'sim'` dataset has eight subdatasets arrange according to generator
        $\hat p_T$, e.g. `'SIM470_Jet300_pT375-infGeV'`, and the `'gen'`
        dataset also has eight subdatasets arranged similaraly, e.g.
        `'GEN470_pT375-infGeV'`.
    - **validate_files** : _bool_
        - Whether or not to validate files according to their MD5 hashes. It
        is a good idea to set this to `True` when first downloading the files
        from Zenodo in order to ensure they downloaded properly.
    - **store_pfcs** : _bool_
        - Whether or not to store PFCs if they are present in the dataset.
    - **store_gens** : _bool_
        - Whether or not to store gen-level particles (referred to as
        "gens") if they are present in the dataset.
    - **verbose** : _int_
        - Verbosity level to use when loading. `0` is the least verbose, `1`
        is more verbose, and `2` is the most verbose.

    **Returns**

    - _MODDataset_
        - A `MODDataset` object containing the selected events or jets from the
        specified collection, dataset, and subdatasets.
    """

    default_kwargs = {
        'amount': 1,
        'cache_dir': '~/.energyflow',
        'collection': 'CMS2011AJets',
        'dataset': 'cms',
        'subdatasets': None,
        'validate_files': False,
        'store_gens': True,
        'store_pfcs': True,
        'verbose': 0,
    }

    # process arguments
    for k,v in default_kwargs.items():
        if k not in kwargs:
            kwargs[k] = v
    kwargs_check('load', kwargs, default_kwargs.keys())

    # store arguments
    amount = kwargs['amount']
    cache_dir = kwargs['cache_dir']
    validate_files = kwargs['validate_files']
    verbose = kwargs['verbose']
    moddset_kwargs = {kw: kwargs[kw] for kw in ['store_gens', 'store_pfcs']}

    # verify collection
    cname = kwargs['collection']
    collection = _get_collection(cname)

    # verify dataset
    dataset = _get_dataset(collection, kwargs['dataset'])

    # determine subdatasets
    if kwargs['subdatasets'] is None:
        subdatasets = dataset['subdatasets']

    else:

        # filter subdatasets according to specified values
        subdatasets = [sdset for sdset in dataset['subdatasets']
                                if sdset[0] in kwargs['subdatasets']]

        # check that no unrecognized subdatasets were passed in
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

            filename = '{}_{}_compressed.h5'.format(name, i)
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

# filter_particles(particles, which='all', pt_cut=None, chs=False,
#                             pt_i=0, pid_i=4, vertex_i=5)
def filter_particles(particles, which='all', pt_cut=None, chs=False,
                                pt_i=0, pid_i=4, vertex_i=5):
    """Constructs a mask that will select particles according to specified
    properties. Currently supported are selecting particles according to their
    charge, removing particles associated to a pileup vertex, and implementing
    a minimum particle-level pT cut

    **Arguments**

    - **particles** : _numpy.ndarray_
        - Two-dimensional array of particles.
    - **which** : {`'all'`, `'charged'`, `'neutral'`}
        - Selects particles according to their charge.
    - **pt_cut** : _float_ or `None`
        - If not `None`, the minimum transverse momentum a particle can have to
        be selected.
    - **chs** : _bool_
        - Whether or not to implement charged hadron subtraction (CHS), which
        removes particles associated to a non-leading vertex (i.e. with vertex
        ids greater than or equal to 1).
    - **pt_i** : _int_
        - Column index of the transverse momentum values of the particles.
    - **pid_i** : _int_
        - Column index of the particle IDs (used to select by charge).
    - **vertex_i** : _int_
        - Column index of the vertex IDs (used to implement CHS).

    **Returns**

    - _numpy.ndarray_
        - A boolean mask which selects the specified particles, i.e.
        `particles[filter_particles(particles, ...)]` will be an array of only
        those particles passing the specified cuts.
    """
    
    mask = np.ones(len(particles), dtype=bool)
    
    # pt cut
    if pt_cut is not None:
        mask &= (particles[:,pt_i] >= pt_cut)
        
    # select specified particles
    if which != 'all':
        chrg_mask = ischrgd(particles[:,pid_i])
        
        if which == 'charged':
            mask &= chrg_mask
        elif which == 'neutral':
            mask &= ~chrg_mask
        else:
            raise ValueError("'which' must be one of {'all', 'charged', 'neutral}")
            
    # apply chs
    if chs:
        mask &= (particles[:,vertex_i] <= 0)
        
    return mask

# kfactors(dataset, pts, npvs=None, collection='CMS2011AJets',
#                        apply_residual_correction=True)
def kfactors(dataset, pts, npvs=None, collection='CMS2011AJets', apply_residual_correction=True):
    """Evaluates k-factors used by a particular collection. Currently, since
    CMS2011AJets is the only supported collection, some of the arguments are
    specific to the details of this collection (such as the use of jet pTs) and
    may change in future versions of this function.

    **Arguments**

    - **dataset** : {`'sim'`, `'gen'`}
        - Specifies which type of k-factor to use. `'sim'` includes a reweighting
        to match the distribution of the number of primary vertices between
        the simulation dataset and the CMS data whereas `'gen'` does not.
    - **pts** : _numpy.ndarray_
        - The transverse momenta of the jets, used to determine the
        pT-dependent k-factor due to using only leading order matrix elements
        in the event generation. For the CMS2011AJets collection, these are
        derived from Figure 5 of [this reference](https://doi.org/10.1016/j.
        physletb.2014.01.034).
    - **npvs** : _numpy.ndarray_ of integer type or `None`
        - The number of primary vertices of a simulated event, used to
        reweight a simulated event to match the pileup distribution of data.
        Should be the same length as `pts` and correspond to the same events.
        Not used if `dataset` is `'gen'`.
    - **collection** : _str_
        - Name of the collection of datasets to consider. Currently the only
        collection is `'CMS2011AJets'`, though more may be added in the future.
    - **apply_residual_correction** : _bool_
        - Whether or not to apply a residual correction derived from the first
        bin of the pT spectrum that corrects for the remaining difference
        between data and simulation.

    **Returns**

    - _numpy.ndarray_
        - An array of k-factors corresponding to the events specified by the
        `pts` and (optionally) the `npvs` arrays. These should be multiplied
        into any existing weight for the simulated or generated event.
    """

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


###############################################################################
# PRIVATE FUNCTIONS for MODDataset
###############################################################################

def _get_collection(cname):

    # verify collection
    if cname not in COLLECTIONS:
        raise ValueError("Collection '{}' not recognized".format(cname))

    return COLLECTIONS[cname]

def _get_dataset(collection, dname):

    # check for valid dname (special case info since we add that to collection)
    if dname == 'info' or dname not in collection:
        raise ValueError('dataset {} not recognized'.format(dname))

    return collection[dname]

def _get_dataset_info(cname):

    # get collection
    collection = _get_collection(cname)

    # cache info if not already stored
    if 'info' not in collection:

        fpath = os.path.join(EF_DATA_DIR, '{}.json'.format(cname))
        with open(fpath, 'r') as f:
            info = json.load(f)

        # convert to numpy arrays
        for key in ['kfactor_x', 'kfactor_y', 'npv_hist_ratios']:
            info[key] = np.asarray(info[key])

        collection['info'] = info

    return collection['info']

def _cols_str(cols, nspaces=4):
    return str(cols).replace('\n', '\n' + nspaces*' ')

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

def _process_selections(sel_list):
    sels = []
    for sel in sel_list:
        if isinstance(sel, six.string_types):
            sels.append(sel)
        else:
            sels.append(''.join([str(s) for s in sel]))

    return '&'.join(sels)

def _moddset_save(arg):
    i, filepath, compression = arg
    moddsets[i].save(filepath, compression=compression, verbose=0)

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


###############################################################################
# MODDataset
###############################################################################

class MODDataset(object):

    """Loads and provides access to datasets in MOD HDF5 format. Jets can be
    selected when loading from file according to a number of kinematic
    attributes. MOD HDF5 datasets are created via the [`save`](#save) method.

    Currently, the MOD HDF5 format consists of an HDF5 file with the following
    arrays, each of which are stored as properties of the `MODDataset`:

    - `/jets_i` - _int64_
        - An array of integer jet attributes, which are currently:
            - `fn` : The file number of the jet, used to index the
            [`filenames`](#filenames) array.
            - `rn` : The run number of the jet.
            - `lbn` : The lumiblock number (or lumisection) of the jet.
            - `evn` : The event number of the jet.
            - `npv` (CMS/SIM only) : The number of primary vertices of the
            event containing the jet.
            - `quality` (CMS/SIM only) : The quality of the jet, where `0`
            means no quality, `1` is "loose", `2` is "medium", and `3` is
            "tight".
            - `hard_pid` (SIM/GEN only) : The particle ID of the hard parton
            associated to the jet (`0` if not associated).
    - `/jets_f` - _float64_
        - An array of floating point jet attributes, which are currently:
            - `jet_pt` : Transverse momentum of the jet.
            - `jet_y` : Rapidity of the jet.
            - `jet_phi` : Azimuthal angle of the jet.
            - `jet_m` : Mass of the jet.
            - `jet_eta` : Pseudorapidity of the jet.
            - `jec` (CMS/SIM only) : Jet energy correction.
            - `jet_area` (CMS/SIM only) : Area of the jet.
            - `jet_max_nef` (CMS/SIM only) : Maximum of the hadronic and
            electromagnetic energy fractions of the jet.
            - `gen_jet_pt` (SIM only) : Transverse momentum of an associated
            GEN jet. `-1` if not associated.
            - `gen_jet_y` (SIM only) : Rapidity of an associated GEN jet. `-1`
            if not associated.
            - `gen_jet_phi` (SIM only) : Azimuthal angle of an associated GEN
            jet. `-1` if not associated.
            - `gen_jet_m` (SIM only) : Mass of an associated GEN jet. `-1` if
            not associated.
            - `gen_jet_eta` (SIM only) : Pseudorapidity of an associated GEN
            jet. `-1` if not associated.
            - `hard_pt` (SIM/GEN only) : Transverse momentum of an associated
            hard parton. `-1` if not associated.
            - `hard_y` (SIM/GEN only) : Rapidity of an associated hard parton.
            `-1` if not associated.
            - `hard_phi` (SIM/GEN only) : Azimuthal angle of an associated hard
            parton. `-1` if not associated.
            - `weight` :  Contribution of this jet to the cross-section, in
            nanobarns.
    - `/pfcs` - _float64_ (CMS/SIM only)
        - An array of all particle flow candidates, with attributes listed
        below. There is a separate `/pfcs_index` array in the file which
        contains information for `MODDataset` to separate these particles into
        separate jets. The columns of the array are currently:
            - `pt` : Transverse momentum of the PFC.
            - `y` : Rapidity of the PFC.
            - `phi` : Azimuthal angle of the PFC.
            - `m` : Mass of the PFC.
            - `pid` : PDG ID of the PFC.
            - `vertex` : Vertex ID of the PFC. `0` is leading vertex, `>0` is
            a pileup vertex, and `-1` is unknown. Neutral particles are
            assigned to the leading vertex.
    - `/gens` - _float64_ (SIM/GEN only)
        - An array of all generator-level particles, currently with the same 
        columns as the `pfcs` array (the vertex column contains all `0`s). For
        the SIM dataset, these are the particles of jets associated to the SIM
        jets which are described in the `jets_i` and `jets_f` arrays. As with
        `pfcs`, there is a separate `gens_index` array which tells `MODDataset`
        how to separate these gen particles into distinct jets.
    - `/filenames` - _str_
        - An array of strings indexed by the `fn` attribute of each jet. For
        CMS, this array is one dimensional and contains the CMS-provided
        filenames. For SIM/GEN, this array is two dimensional where the first
        column is the pT value that appears in the name of the dataset and the
        second column is the CMS-provided filename. In all cases, indexing this
        array with the `fn` attribute of a jet gives the file information in
        which the event containing that jet is to be found.

    Note that the column names of the `jets_i`, `jets_f`, `pfcs`, and `gens`
    arrays are stored as lists of strings in the attributes `jets_i_cols`,
    `jets_f_cols`, `pfcs_cols`, and `gens_cols`.

    For each of the above arrays, `MODDataset` stores the index of the column
    as an attribute with the same name as the column. For example, for an
    instance called `modds`, `modds.fn` has a value of `0` since it is the
    first column in the `jets_i` array, `modds.jet_phi` has a value of `2`,
    `modds.m` has a value of `3`, etc.

    Even more helpfully, a view of each column of the jets arrays is stored
    as an attribute as well, so that `modds.jet_pts` is the same as
    `modds.jets_f[:,modds.jet_pt]`, `modds.evns` is the same as 
    `modds.jets_i[:,modds.evn]`, etc. Additionally, one special view is stored,
    `corr_jet_pts`, which is equal to the product of the jet pTs and the JECs,
    i.e. `modds.jet_pts*modds.jecs`.

    `MODDataset` supports the builtin `len()` method, which returns the number
    of jets currently stored in the dataset, as well as the `print()` method,
    which prints a summary of the dataset.
    """

    # MODDataset(*args, datasets=None, path=None, num=-1, shuffle=True, 
    #                   store_pfcs=True, store_gens=True)
    def __init__(self, *args, **kwargs):
        """`MODDataset` can be initialized from a MOD HDF5 file or from a list
        of existing `MODDataset`s. In the first case, the filename should be
        given as the first positional argument. In the second case, the 
        `datasets` keyword argument should be set to a list of `MODDataset`
        objects.

        **Arguments**

        - ***args** : _arbitrary positional arguments_
            - Each argument specifies a requirement for an event to be selected
            and kept in the `MODDataset`. All requirements are ANDed together.
            Each specification can be a string or a tuple/list of objects that
            will be converted to strings and concatenated together. Each string
            specifies the name of one of the columns of one of the jets arrays
            (`'corr_jet_pts'` is also accepted, see above, as well as
            `'abs_jet_eta'`, `'abs_gen_jet_y'`, etc, which use the absolute
            values of the [pseudo]rapidities of the jets) as well as one or
            more comparisons to be performed using the values of that column
            and the given values in the string. For example,
            `('corr_jet_pts >', 400)`, which is the same as
            `'corr_jet_pts>400'`, will select jets with a corrected pT above
            400 GeV. Ampersands may be used within one string to indicated
            multiple requirements, e.g. `'corr_jet_pts > 400 & abs_jet_eta'`,
            which has the same effect as using multiple arguements each with a
            single requirement.
        - **datasets** : {_tuple_, _list_} of `MODDataset` instances or `None`
            - `MODDataset`s from which to initialize this dataset. Effectively
            what this does is to concatenate the arrays held by the datasets.
            Should always be `None` when initializing from an existing file.
        - **path** : _str_ or `None`
            - If not `None`, then `path` is prepended to the filename when
            initializing from file. Has no effect when initializing from
            existing datasets.
        - **num** : _int_
            - The number of events or jets to keep after subselections are
            applied. A value of `-1` keeps the entire dataset. The weights
            are properly rescaled to preserve the total cross section of the
            selection.
        - **shuffle** : _bool_
            - When subselecting a fraction of the dataset (i.e. `num!=-1`),
            if `False` the first `num` events passing cuts will be kept, if
            `True` then a random subset of `num` events will be kept. Note that
            this has no effect when `num` is `-1`, and also that this flag only
            affects which events are selected and does not randomize the order
            of the events that are ultimately stored by the `MODDataset` object.
        - **store_pfcs** : _bool_
            - Whether or not to store PFCs if they are present in the dataset.
        - **store_gens** : _bool_
            - Whether or not to store gen-level particles (referred to as
            "gens") if they are present in the dataset.
        """

        default_kwargs = {
            'copy_particles': True,
            'datasets': None,
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
        self.copy_particles = kwargs.pop('copy_particles')
        self.num = kwargs.pop('num')
        self.shuffle = kwargs.pop('shuffle')
        self.store_pfcs = kwargs.pop('store_pfcs')
        self.store_gens = kwargs.pop('store_gens')
        datasets = kwargs.pop('datasets')

        # check for disallowed kwargs
        other_allowed_kwargs = {'_arrays', '_dataset', 'path'}
        for kw in kwargs:
            if kw not in other_allowed_kwargs:
                raise TypeError("__init__() got an unexpected keyword argument '{}'".format(kw))

        # initialize from explicit arrays (used only when making files initially)
        if len(args) == 0 and '_arrays' in kwargs and '_dataset' in kwargs:
            self._init_from_arrays(kwargs['_dataset'], kwargs['_arrays'])

        # initialize from list of datasets
        elif datasets is not None:
            self.selection = _process_selections(args)
            self._init_from_datasets(datasets)

        # initialize from file
        elif len(args) and isinstance(args[0], six.string_types):
            self.selection = _process_selections(args[1:])
            self._init_from_filename(args[0], kwargs['path'])

        else:
            raise RuntimeError('Initialization of MODDataset not understood')

    #################
    # PRIVATE METHODS
    #################

    # close any HDF5 files and try to garbage collect arrays to free memory
    def __del__(self):

        # close file 
        self.close()

        # delete arrays
        if hasattr(self, '_jets_i'):
            del self._jets_i
        if hasattr(self, '_jets_f'):
            del self._jets_f
        if hasattr(self, '_filenames'):
            del self._filenames

        # delete pfcs if they exist
        if hasattr(self, '_pfcs'):
            del self._pfcs

        # delete gens if they exist
        if hasattr(self, '_gens'):
            del self._gens

        # delete particles if the exist
        if hasattr(self, '_particles'):
            del self._particles

        # force garbage collection
        gc.collect()

    # length of this object is the number of jets it's holding
    def __len__(self):
        return len(self.jets_i)

    # makes MODDataset printable
    def __repr__(self):
        s = ('{} MODDataset\n'.format(self.dataset.upper()) + 
             '  Jet Integers - {}\n    {}\n'.format(self.jets_i.shape, _cols_str(self.jets_i_cols)) + 
             '  Jet Floats - {}\n    {}\n'.format(self.jets_f.shape, _cols_str(self.jets_f_cols)))

        if self.store_pfcs:
            s += '  PFCs - {}\n    {}\n'.format(self.pfcs.shape, _cols_str(self.pfcs_cols))

        if self.store_gens:
            s += '  GENs - {}\n    {}\n'.format(self.gens.shape, _cols_str(self.gens_cols))

        s += '  Filenames - {}\n'.format(self.filenames.shape)

        return s

    # determine which type of dataset this object is holding
    def _store_dataset_info(self, dataset):

        assert dataset in ['cms', 'sim', 'gen'], "Dataset must be one of ['cms', 'sim', 'gen']"
        self.dataset = dataset

        self.cms = (self.dataset == 'cms')
        self.sim = (self.dataset == 'sim')
        self.gen = (self.dataset == 'gen')

        # update options based on dataset type
        self.store_pfcs &= not self.gen
        self.store_gens &= not self.cms

    # store column names of the given array
    def _store_cols(self, arr, cols=None, allow_multiple=False):

        # get cols from file
        if cols is None:
            cols = self.hf[arr].attrs['cols']

        cols = np.asarray(cols, dtype='U')
        setattr(self, '_' + arr + '_cols', cols)

        for i,col in enumerate(cols):

            # ensure cols are unique
            if not allow_multiple:
                m = "Repeat instances of col '{}', check file validity".format(col)
                assert not hasattr(self, col), m

            # store column index
            setattr(self, col, i)

    # store views of the columns of the jets_i and jets_f arrays as attributes
    def _store_views_of_jets(self):

        for jets in ['jets_i', 'jets_f']:

            # retrieve array, cols
            arr, cols = getattr(self, jets), getattr(self, jets + '_cols')

            for i,col in enumerate(cols):

                # set attribute + s as view of this column of array
                setattr(self, col + 's', arr[:,i])

        # calculate corrected pts
        self.corr_jet_pts = self.jet_pts*self.jecs if hasattr(self, 'jecs') else self.jet_pts

    # ensure that the particles attribute is set appropriately
    def _set_particles(self):

        if self.store_pfcs and not self.gen:
            self._particles = self.pfcs
            self._particles_cols = self.pfcs_cols

        if self.store_gens and self.gen:
            self._particles = self.gens
            self._particles_cols = self.gens_cols

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
        self._hf = h5py.File(self.filepath, 'r')

        # load selected jets
        self._jets_i = self.hf['jets_i'][:]
        self._jets_f = self.hf['jets_f'][:]

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
        self._mask, self.specs = self.sel(_selection=self.selection)

        # determine weight factor caused by subselecting
        total_weight_after_selections = np.sum(self.weights[self._mask])

        # select the requested number of jets
        if self.num != -1:

            # shuffle if requested
            arange = np.arange(len(self._mask))[self._mask]
            if self.shuffle:
                np.random.shuffle(arange)

            # mask out jets beyond what was requested
            self._mask[arange[self.num:]] = False

            # weight factor
            weight_factor = total_weight_after_selections/np.sum(self.weights[self._mask])

        else:
            weight_factor = 1.0

        # apply mask to jets
        self._jets_i = self.jets_i[self._mask]
        self._jets_f = self.jets_f[self._mask]

        # alter weights due to subselection
        self._jets_f[:,self.weight] *= weight_factor

        # store views of jets cols
        self._store_views_of_jets()

        if self.store_pfcs:

            # read in pfcs_index
            self.pfcs_index = self.hf['pfcs_index'][:]

            # store pfcs as separate arrays
            self._pfcs = _separate_particle_arrays(self.hf['pfcs'][:], self.pfcs_index, self._mask, copy=self.copy_particles)

            # store pfcs cols
            self._store_cols('pfcs')

        if self.store_gens:

            # read in gens_index
            self.gens_index = self.hf['gens_index'][:]

            # store gens as separate arrays
            self._gens = _separate_particle_arrays(self.hf['gens'][:], self.gens_index, self._mask, copy=self.copy_particles)

            # store gens cols
            self._store_cols('gens', allow_multiple=self.store_pfcs)

        # store filenames
        self._filenames = self.hf['filenames'][:].astype('U')

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
                self._filenames = dataset.filenames
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
        self._jets_i = np.concatenate(jets_i, axis=0)
        self._jets_f = np.concatenate(jets_f, axis=0)

        # store views of jets cols
        self._store_views_of_jets()

        # sum all weights
        self._orig_total_weight = np.sum(self.weights)

        if self.store_pfcs:
            self._pfcs = np.concatenate(pfcs, axis=0)

        if self.store_gens:
            self._gens = np.concatenate(gens, axis=0)

        # set particles
        self._set_particles()

    # note that this method of initialization is not publicly supported
    def _init_from_arrays(self, dataset, arrays):

        # update options
        self.store_pfcs &= 'pfcs' in arrays
        self.store_gens &= 'gens' in arrays

        # store dataset info by hand
        self._store_dataset_info(dataset)

        # jets arrays
        self._jets_i, self._jets_f = arrays['jets_i'], arrays['jets_f']

        # jets cols
        self._store_cols('jets_i', arrays['jets_i_cols'])
        self._store_cols('jets_f', arrays['jets_f_cols'])

        # store views of jets cols
        self._store_views_of_jets()

        # sum all weights
        self._orig_total_weight = np.sum(self.weights)

        # pfcs
        if self.store_pfcs:
            self._pfcs = arrays['pfcs']
            self._store_cols('pfcs', arrays['pfcs_cols'])

        # gens
        if self.store_gens:
            self._gens = arrays['gens']
            self._store_cols('gens', arrays['gens_cols'], allow_multiple=self.store_pfcs)

        # filenames
        self._filenames = np.asarray(arrays['filenames'], dtype='U')

        # set particles
        self._set_particles()

    ################
    # PUBLIC METHODS
    ################

    def apply_mask(self, mask, preserve_total_weight=False):
        """Subselects jets held by the `MODDataset` according to a boolean
        mask.

        **Arguments**

        - **mask** : _numpy.ndarray_ or type _bool_
            - A boolean mask used to select which jets are to be kept. Should
            be the same length as the `MODDataset` object.
        - **preserve_total_weight** : _bool_
            - Whether or not to keep the cross section of the `MODDataset`
            fixed after the selection.
        """

        if len(mask) != len(self):
            raise IndexError('Incorrectly sized mask')

        if preserve_total_weight:
            total_weight_before_mask = np.sum(self.weights)

        self._jets_i = self.jets_i[mask]
        self._jets_f = self.jets_f[mask]

        if preserve_total_weight:
            weight_factor = total_weight_before_mask/np.sum(self.jets_f[:,self.weight])
            self._jets_f[:,self.weight] *= weight_factor

        self._store_views_of_jets()

        if self.store_pfcs:
            self._pfcs = self.pfcs[mask]

        if self.store_gens:
            self._gens = self.gens[mask]

        # set particles
        self._set_particles()

    def sel(self, *args, **kwargs):
        """Returns a boolean mask that selects jets according to the specified
        requirements.

        **Arguments**

        - ***args** : _arbitrary positional arguments_
            - Used to specify cuts to be made to the dataset while loading; see
            the detailed description of the positional arguments accepted by
            [`MODDataset`](#moddataset).

        **Returns**

        - _numpy.ndarray_ of type _bool_
            - A boolean mask that will select jets that pass all of the
            specified requirements.
        """

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
        for groups in self._sel_re.findall(selection):
            name = groups[3] + 's'
            nspecs = 0

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
                    nspecs += 1

            if nspecs == 0:
                raise ValueError('Invalid groups from selection: {}'.format(groups))

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

    def save(self, filepath, npf=-1, compression=None, verbose=1, n_jobs=1):
        """Saves a `MODDataset` in the MOD HDF5 format.

        **Arguments**

        - **filepath** : _str_
            - The filepath (with or without the `'.h5'` suffix) where the saved
            file will be located.
        - **npf** : _int_
            - The number of jets per file. If not `-1`, multiple files will be
            created with `npf` jets as the maximum number stored in each file,
            in which case `'_INDEX'`, where `INDEX` is the index of that file,
            will be appended to the filename.
        - **compression** : _int_ or `None`
            - If not `None`, the gzip compression level to use when saving the
            arrays in the HDF5 file. If not `None`, `'_compressed'` will be
            appended to the end of the filename.
        - **verbose** : _int_
            - Verbosity level to use when saving the files.
        - **n_jobs** : _int_
            - The number of processes to use when saving the files; only
            relevant when `npf!=-1`.
        """

        path, name = os.path.split(filepath)
        if name.endswith('.h5'):
            name = '.'.join(name.split('.')[:-1])

        start = time.time()
        if npf != -1:
            
            global moddsets

            i = begin = end = 0
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

    def close(self):
        """Closes the underlying HDF5 file, if one is associated with the
        `MODDataset` object. Note that associated HDF5 files are closed by
        default when the `MODDataset` object is deleted.
        """

        if hasattr(self, '_hf'):
            self._hf.close()

    ############
    # PROPERTIES
    ############

    @property
    def jets_i(self):
        """The `jets_i` array, described under [`MODDataset`](#moddataset)."""

        return self._jets_i

    @property
    def jets_f(self):
        """The `jets_f` array, described under [`MODDataset`](#moddataset)."""

        return self._jets_f

    @property
    def pfcs(self):
        """The `pfcs` array, described under [`MODDataset`](#moddataset)."""

        return self._pfcs if hasattr(self, '_pfcs') else None

    @property
    def gens(self):
        """The `gens` array, described under [`MODDataset`](#moddataset)."""

        return self._gens if hasattr(self, '_gens') else None

    @property
    def particles(self):
        """If this is a CMS or SIM dataset, `particles` is the same as `pfcs`;
        for GEN it is the same as `gens`.
        """

        return self._particles if hasattr(self, '_particles') else None

    @property
    def filenames(self):
        """The `filenames` array, described under [`MODDataset`](#moddataset)."""

        return self._filenames

    @property
    def hf(self):
        """The underlying HDF5 file, if one is associated to the `MODDataset`."""
        
        return self._hf if hasattr(self, '_hf') else None

    @property
    def jets_i_cols(self):
        return self._jets_i_cols

    @property
    def jets_f_cols(self):
        return self._jets_f_cols

    @property
    def pfcs_cols(self):
        return self._pfcs_cols if hasattr(self, '_pfcs_cols') else None

    @property
    def gens_cols(self):
        return self._gens_cols if hasattr(self, '_gens_cols') else None

    @property
    def particles_cols(self):
        return self._particles_cols if hasattr(self, '_particles_cols') else None
