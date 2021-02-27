r"""# Energy Flow Moments

Energy Flow Moments (EFMs) are tensors that can be computed in
$\mathcal O(M)$ where $M$ is the number of particles. They are useful for many
things, including providing a fast way of computing the $\beta=2$ EFPs, which
are the scalar contractions of products of EFMs.

The expression for a (normalized) hadronic EFM in terms of transverse momenta
$\{p_{Ti}\}$ and particle momenta $\{p_i^\mu\}$ is:

\[\mathcal I^{\mu_1\cdots\mu_v} = 2^{v/2}\sum_{i=1}^Mz_in_i^{\mu_1}\cdots n_i^{\mu_v},\]

where

\[z_i=\frac{p_{Ti}}{\sum_jp_{Tj}},\quad\quad n_i^\mu=\frac{p_i^\mu}{p_{Ti}}.\]

Note that for an EFM in an $e^+e^-$ context, transverse momenta are replaced
with energies.


Support for using EFMs to compute $\beta=2$ EFPs is built in to the `EFP` and
`EFPSet` classes using the classes and functions in this module. The `EFM` and
`EFMSet` classes can also be used on their own, as can the `efp2efms` function.
"""

#  ______ ______ __  __
# |  ____|  ____|  \/  |
# | |__  | |__  | \  / |
# |  __| |  __| | |\/| |
# | |____| |    | |  | |
# |______|_|    |_|  |_|

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev


from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from operator import itemgetter
import sys

import numpy as np
from numpy.core.multiarray import c_einsum

from energyflow.algorithms import einsum
from energyflow.base import EFMBase
from energyflow.utils import flat_metric, timing
from energyflow.utils.graph_utils import *

__all__ = ['EFM', 'EFMSet', 'efp2efms']

###############################################################################
# EFM functions
###############################################################################

# allowed einsum symbols
I = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def efp2efms(graph):
    """Translates an EFP formula, specified by its graph, to an expression
    involving EFMs. The input is a graph as a list of edges and the output is a
    tuple where the first argument is a string to be used with einsum and the
    second is a list of EFM signatures (the number of raised indices followed
    by the number of lowered indices).

    **Arguments**

    - **graph** : _list_ of _tuple_
        - The EFP graph given as a list of edges.

    **Returns**

    - (_str_, _list_ of _tuple_)
        - The einstring to be used with einsum for performing the contraction
        of EFMs followed by a list of the EFM specs. If `r` is the result of
        this function, and `efms` is a dictionary containing EFM tensors
        indexed by their signatures, then the value of the EFP is given as
        `np.einsum(r[0], *[efms[sig] for sig in r[1]])`.
    """

    # handle empty graph
    if len(graph) == 0:
        return '', [(0,0)]

    # build convenient data structure to hold graph information
    vds = get_valency_structure(graph)

    # dictionary to hold efm terms
    efms = {}

    # counter to store how to get fresh dummy indices
    ind = 0

    # iterate over vertices sorted by valency in decreasing order
    sorted_verts = sorted(valencies(graph).items(), key=itemgetter(1), reverse=True)
    for vert,valency in sorted_verts:

        # dict holding info for new efm term
        new_efm = {'upper_indices': '', 'lower_indices': ''}

        # iterate over neighboring vertices
        for neighbor,n_shared_edges in vds[vert].items():

            # if a new neighbor, assign fresh I
            if neighbor not in efms:
                new_I = I[ind:ind+n_shared_edges]
                ind += n_shared_edges
                new_efm['upper_indices'] += new_I

                # store I shared with that neighbor
                new_efm[neighbor] = new_I

            # if neighbor already has an efm factor, add already assigned indices to lower_indices
            else:
                new_efm['lower_indices'] += efms[neighbor][vert]

        # store new efm factor
        efms[vert] = new_efm

    einstr_list, efm_specs = [], []
    for vert,valency in sorted_verts:
        efm = efms[vert]

        # conventionally put uppered indices before lower indices
        einstr_list.append(efm['upper_indices'] + efm['lower_indices'])

        # add spec which is (nup, nlow) of efm
        efm_specs.append((len(efm['upper_indices']), len(efm['lower_indices'])))

    # return comma joined einstr and efm_specs
    return ','.join(einstr_list), efm_specs

###############################################################################
# EFM
###############################################################################

class EFM(EFMBase):

    """A class representing and computing a single EFM."""

    # EFM(nup, nlow=0, measure='hadrefm', beta=2, kappa=1, normed=None, 
    #                  coords=None, check_input=True)
    def __init__(self, nup, nlow=0, rl_from=None, subslice_from=None, **kwargs):
        r"""Since EFMs are fully symmetric tensors, they can be specified by
        just two integers: the number of raised and number of lowered indices
        that they carry. Thus we use a tuple of two ints as an EFM "spec" or
        signature throughout EnergyFlow. By convention the raised indices come
        before the lowered indices.

        Since a standalone `EFM` defines and holds a `Measure` instance, all
        `Measure` keywords are accepted. Note that `beta` is ignored as EFMs
        require $\beta=2$.

        **Arguments**

        - **nup** : _int_
            - The number of uppered indices of the EFM.
        - **nlow** : _int_
            - The number of lowered indices of the EFM.
        - **measure** : {`'hadrefm'`, `'eeefm'`}
            - The choice of measure. See [Measures](../measures) for additional
            info. Note that EFMs can only use the `'hadrefm'` and `'eeefm'`
            measures.
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
        """

        # initialize base class
        super(EFM, self).__init__(kwargs)

        # store inputs
        self._nup, self._nlow = nup, nlow
        self._rl_from = rl_from
        self._subslice_from = subslice_from

        # useful derived quantities
        self._v = self.nup + self.nlow
        self._spec = (self.nup, self.nlow)

        # construct by raising/lowering
        if self._rl_from is not None:

            # ensure valid raising/lowering
            if self.v != sum(self._rl_from):
                raise ValueError('cannot raise/lower among different valency EFMs')

            # determine einstr    
            diff = self.nup - self._rl_from[0]
            self._rl_diff = abs(diff)
            i_start, i_end = ((self._rl_from[0], self.nup) if diff > 0 else 
                              (self.nup, self._rl_from[0]))
            self.rl_einstr = ','.join([I[:self.v]] + list(I[i_start:i_end])) + '->' + I[:self.v]

            self._construct = self._rl_construct

        # construct by subslicing
        elif self._subslice_from is not None:

            # get number of subslices
            num_up_subslices = self._subslice_from[0] - self.nup
            num_low_subslices = self._subslice_from[1] - self.nlow
            
            # perform check
            if num_up_subslices < 0 or num_low_subslices < 0:
                m = 'invalid subslicing from {} to {}'.format(self.subslicing_from, self.spec)
                raise ValueError(m)

            # note that python 2 doesn't support pickling ellipsis
            if sys.version_info[0] > 2:
                self.subslice = tuple([0]*num_up_subslices + [Ellipsis] + [0]*num_low_subslices)
            else:
                self.subslice = tuple([0]*num_up_subslices + self.v*[slice(None)] + 
                                      [0]*num_low_subslices)

            self._pow2 = 2**(-(num_up_subslices + num_low_subslices)/2)
            self._construct = self._subslice_construct

        # construct directly
        else:
            self.raw_einstr = (','.join([I[0]] + [I[0] + I[i+1] for i in range(self.v)]) +
                               '->' + I[1:self.v+1])
            self.raw_einpath = ['einsum_path'] + [(0,1)]*self.v
            self._rl_diff = self.nlow
            self.rl_einstr = ','.join([I[:self.v]] + list(I[self.nup:self.v])) + '->' + I[:self.v]
            self._pow2 = 2**(self.v/2)
            self._construct = self._raw_construct

    #================
    # PRIVATE METHODS
    #================

    def _rl_construct(self, tensor):

        # fine to use pure c_einsum here as it's used anyway
        return c_einsum(self.rl_einstr, tensor, *[flat_metric(len(tensor))]*self._rl_diff)

    def _subslice_construct(self, tensor):
        return self._pow2 * tensor[self.subslice]

    def _raw_construct(self, zsnhats):
        zs, nhats = zsnhats
        M, dim = nhats.shape

        # if no lowering is needed
        if self.nlow == 0:
            return self._pow2 * einsum(self.raw_einstr, zs, *[nhats]*self.v, 
                                       optimize=self.raw_einpath)

        # lowering nhats first is better
        elif M*dim < dim**self.v:
            low_nhats = nhats * (flat_metric(dim)[np.newaxis])
            einsum_args = [nhats]*self.nup + [low_nhats]*self.nlow
            return self._pow2 * einsum(self.raw_einstr, zs, *einsum_args, 
                                       optimize=self.raw_einpath)

        # lowering EFM is better    
        else:
            tensor = einsum(self.raw_einstr, zs, *[nhats]*self.v, optimize=self.raw_einpath)
            return self._pow2 * self._rl_construct(tensor)

    #===============
    # PUBLIC METHODS
    #===============

    def compute(self, event=None, zs=None, nhats=None):
        """Evaluates the EFM on a single event. Note that `EFM` also is
        callable, in which case this method is invoked.

        **Arguments**

        - **event** : 2-d array_like or `fastjet.PseudoJet`
            - The event as an array of particles in the coordinates specified
            by `coords`.
        - **zs** : 1-d array_like
            - If present, `nhats` must also be present, and `zs` is used in place 
            of the energies of an event.
        - **nhats** : 2-d array like
            - If present, `zs` must also be present, and `nhats` is used in place
            of the scaled particle momenta.

        **Returns**

        - _numpy.ndarray_ of rank `v`
            - The values of the EFM tensor on the event. The raised indices
            are the first `nup` and the lowered indices are the last `nlow`.
        """

        return self._raw_construct(super(EFM, self).compute(event, zs, nhats))

    def batch_compute(self, events, n_jobs=None):
        """Evaluates the EFM on several events.

        **Arguments**

        - **events** : array_like or `fastjet.PseudoJet`
            - The events as an array of arrays of particles in coordinates
            matching those anticipated by `coords`.
        - **n_jobs** : _int_ or `None`
            - The number of worker processes to use. A value of `None` will
            use as many processes as there are CPUs on the machine.

        **Returns**

        - _numpy.ndarray_ of rank `v+1`
            - Array of EFM tensor values on the events.
        """

        return super(EFM, self).batch_compute(events, n_jobs)

    def set_timer(self):
        self.times = []
        self._construct = timing(self, self._construct)

    #===========
    # PROPERTIES
    #===========

    @property
    def nup(self):
        """The number of uppered indices on the EFM."""

        return self._nup

    @property
    def nlow(self):
        """The number of lowered indices on the EFM."""

        return self._nlow

    @property
    def spec(self):
        """The signature of the EFM as `(nup, nlow)`."""

        return self._spec

    @property
    def v(self):
        """The valency, or total number of indices, of the EFM."""

        return self._v

###############################################################################
# EFMSet
###############################################################################

class EFMSet(EFMBase):

    """A class for holding and efficiently constructing a collection of EFMs."""

    # EFMSet(efm_specs=None, vmax=None, measure='hadrefm', beta=2, kappa=1,
    #        normed=None, coords=None, check_input=True)
    def __init__(self, efm_specs=None, vmax=None, **kwargs):
        r"""An `EFMSet` can be initialized two ways (in order of precedence):

        1. **EFM Specs** - Pass in a list of EFM specs (`nup`, `nlow`).
        2. **Max Valency** - Specify a maximum valency and each EFM with up to
        that many indices will be constructed, with all indices raised.

        Since a standalone `EFMSet` defines and holds a `Measure` instance,
        all `Measure` keywords are accepted. Note that `beta` is ignored as
        EFMs require $\beta=2$.

        **Arguments**

        - **efm_specs** : {_list_, _tuple_, _set_} of _tuple_ or `None`
            - A collection of tuples of length two specifying which EFMs this
            object is to hold. Each spec is of the form `(nup, nlow)` where these
            are the number of upper and lower indices, respectively, that the EFM 
            is to have.
        - **vmax** : _int_
            - Only used if `efm_specs` is None. The maximum EFM valency to
            include in the `EFMSet`. Note that all EFMs will have `nlow=0`.
        - **measure** : {`'hadrefm'`, `'eeefm'`}
            - The choice of measure. See [Measures](../measures) for additional
            info. Note that EFMs can only use the `'hadrefm'` and `'eeefm'`
            measures.
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
        """

        hidden_subslicing = kwargs.pop('subslicing', False)

        # initialize base class
        super(EFMSet, self).__init__(kwargs)

        if efm_specs is None:
            if vmax is not None:
                vmin = 1 if self.normed else 0
                efm_specs = [(v,0) for v in range(vmin, vmax+1)]
            else:
                raise ValueError('efm_specs and vmax cannot both be None.')

        # get unique EFMs 
        self._unique_efms = frozenset(efm_specs)

        # setup EFMs based on whether we can subslice or not
        self.efms, self._args, self.rules = {}, {}, OrderedDict()
        if self.subslicing or hidden_subslicing:
            self._subslicing_setup()
        else:
            self._full_setup()

    #================
    # PRIVATE METHODS
    #================

    def _find_subslice(self, sig):
        """Determine if sig can be subsliced from the currently stored EFMs."""

        nup, nlow = sig
        bsigs = list(filter(lambda x: x[0] >= nup and x[1] >= nlow, self.efms))
        return min(bsigs, key=sum) if len(bsigs) else None

    def _find_minimum_rl(self, sig):
        v = sum(sig)
        vsigs = list(filter(lambda x: sum(x) == v, self.efms))
        return min(vsigs, key=lambda x: abs(sig[0]-x[0]))

    def _subslicing_setup(self):
        """Setup the rules for constructing the EFMs using the fact that
        setting an index to zero "pops" it off, which is referred to as the
        subclicing property. Typically, the EE measures have this property
        whereas the hadronic ones do not.
        """

        # ensure there is at least one EFM of each valency for rl purposes
        maxsig = max(self._unique_efms, key=sum) if len(self._unique_efms) else (0,0)
        self._unique_efms |= set((n,0) for n in range(1, sum(maxsig)+1))

        # sort EFMs to minimize raising/lowering operations
        # EFMs will be ordered first by decreasing v, then decreasing abs difference 
        # between nlow and nup, and then decreasing nup
        self._sorted_efms = sorted(self._unique_efms, key=itemgetter(0), reverse=True)
        self._sorted_efms.sort(key=lambda x: abs(x[0]-x[1]), reverse=True)
        self._sorted_efms.sort(key=sum, reverse=True)

        # take care of empty set
        if not len(self._sorted_efms):
            return

        # the first one must be raw constructed
        sig0 = self._sorted_efms[0]
        self.efms[sig0] = EFM(*sig0, no_measure=True)
        self._args[sig0] = 'r'
        self.rules[sig0] = 'constructing raw'

        for sig in self._sorted_efms[1:]:

            # determine if we can subslice
            big_spec = self._find_subslice(sig)
            if big_spec is not None:
                self.efms[sig] = EFM(*sig, subslice_from=big_spec, no_measure=True)
                self._args[sig] = big_spec
                self.rules[sig] = 'subslicing from {}'.format(big_spec)

            # find best raise/lower available
            else:
                rlsig = self._find_minimum_rl(sig)
                self.efms[sig] = EFM(*sig, rl_from=rlsig, no_measure=True)
                self._args[sig] = rlsig
                rl_n = abs(rlsig[0] - sig[0])
                self.rules[sig] = 'raising/lowering from {}, {}'.format(rlsig, rl_n)

    def _full_setup(self):
        """Setup the rules for constructing the EFMs without the assumption of any
        special properties.
        """

        # sort the EFMs first by increasing v and then by increasing nlow
        self._sorted_efms = sorted(self._unique_efms, key=itemgetter(1))
        self._sorted_efms.sort(key=sum)

        vprev, sigprev = None, None
        for sig in self._sorted_efms:
            v = sum(sig)

            # construct raw (all up) if this is a new valency
            if v != vprev:
                self.efms[sig] = EFM(*sig, no_measure=True)
                self._args[sig] = 'r'
                self.rules[sig] = 'constructing raw'

            # construct from lowering if we have a previous EFM with this v
            else:
                self.efms[sig] = EFM(*sig, rl_from=sigprev, no_measure=True)
                self._args[sig] = sigprev
                self.rules[sig] = 'lowering from {}'.format(sigprev)

            # update prevous values
            vprev, sigprev = v, sig

    #===============
    # PUBLIC METHODS
    #===============

    def compute(self, event=None, zs=None, nhats=None):
        """Evaluates the EFMs held by this `EFMSet` according to the
        predetermined strategy on a single event. Note that `EFMSet` also is
        callable, in which case this method is invoked.

        **Arguments**

        - **event** : 2-d array_like or `fastjet.PseudoJet`
            - The event as an array of particles in the coordinates specified
            by `coords`.
        - **zs** : 1-d array_like
            - If present, `nhats` must also be present, and `zs` is used in place 
            of the energies of an event.
        - **nhats** : 2-d array like
            - If present, `zs` must also be present, and `nhats` is used in place
            of the scaled particle momenta.

        **Returns**

        - _dict_ of _numpy.ndarray_ of rank `v`
            - A dictionary of EFM tensors indexed by their signatures.
        """

        zsnhats = super(EFMSet, self).compute(event, zs, nhats)

        efm_dict = {}
        for sig in self._sorted_efms:
            arg = self._args[sig]
            data_arg = zsnhats if arg == 'r' else efm_dict[arg]
            efm_dict[sig] = self.efms[sig]._construct(data_arg)

        return efm_dict

    def batch_compute(self, events, n_jobs=None):
        """Evaluates the EFMs held by the `EFMSet` on several events.

        **Arguments**

        - **events** : array_like or `fastjet.PseudoJet`
            - The events as an array of arrays of particles in coordinates
            matching those anticipated by `coords`.
        - **n_jobs** : _int_ or `None`
            - The number of worker processes to use. A value of `None` will
            use as many processes as there are CPUs on the machine.

        **Returns**

        - _numpy.ndarray_ of _dict_
            - Object array of dictionaries of EFM tensors indexed by their
            signatures.
        """

        return super(EFMSet, self).batch_compute(events, n_jobs)

    def set_timers(self):
        for efm in self.efms.values():
            efm.set_timer()

    def get_times(self):
        return {sig: np.asarray(efm.times) for sig,efm in self.efms.items()}

    #===========
    # PROPERTIES
    #===========

    @property
    def efms(self):
        """A dictionary of the `EFM` objects held by this `EFMSet` where the
        keys are the signatures of the EFM."""
        
        return self._efms

    @property
    def rules(self):
        """An ordered dictionary of the construction method used for each `EFM`
        where the order is the same as `sorted_efms`."""
        
        return self._rules

    @efms.setter
    def efms(self, value):
        self._efms = value

    @rules.setter
    def rules(self, value):
        self._rules = value
