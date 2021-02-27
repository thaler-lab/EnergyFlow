r"""## Quark and Gluon Jets

Four datasets of quark and gluon jets, each having two million total jets, have
been generated with [Pythia](http://home.thep.lu.se/~torbjorn/Pythia.html) and
[Herwig](https://herwig.hepforge.org/) and are accessible through this
submodule of EnergyFlow. The four datasets are:

- Pythia 8.226 quark (uds) and gluon jets.
- Pythia 8.235 quark (udscb) and gluon jets.
- Herwig 7.1.4 quark (uds) and gluon jets.
- Herwig 7.1.4 quark (udscb) and gluon jets

To avoid downloading unnecessary samples, the datasets are contained in twenty
files with 100k jets each, and only the required files are downloaded. These
are based on the samples used in 
[1810.05165](https://arxiv.org/abs/1810.05165). Splitting the data into 
1.6M/200k/200k train/validation/test sets is recommended for standardized
comparisons.

Each dataset consists of two components:

- `X` : a three-dimensional numpy array of the jets with shape 
`(num_data,max_num_particles,4)`.
- `y` : a numpy array of quark/gluon jet labels (quark=`1` and gluon=`0`).

The jets are padded with zero-particles in order to make a contiguous array.
The particles are given as `(pt,y,phi,pid)` values, where `pid` is the
particle's [PDG id](http://pdg.lbl.gov/2018/reviews/rpp2018-rev-monte
-carlo-numbering.pdf). Quark jets either include or exclude $c$ and $b$
quarks depending on the `with_bc` argument.

The samples are generated from $q\bar q\to Z(\to\nu\bar\nu)+g$ and
$qg\to Z(\to\nu\bar\nu)+(uds[cb])$ processes in $pp$ collisions at
$\sqrt{s}=14$ TeV. Hadronization and multiple parton interactions (i.e.
underlying event) are turned on and the default tunings and shower parameters
are used. Final state non-neutrino particles are clustered into $R=0.4$
anti-$k_T$ jets using FastJet 3.3.0. Jets with transverse momentum
$p_T\in[500,550]$ GeV and rapidity $|y|<1.7$ are kept. Particles are ensured
have to $\phi$ values within $\pi$ of the jet (i.e. no $\phi$-periodicity 
issues). No detector simulation is performed.

The samples are also hosted on Zenodo and we ask that you cite them
appropriately if they are useful to your research. For BibTex entries,
see the [FAQs](/faqs/#how-do-i-cite-the-energyflow-package).

[![DOI](/img/zenodo.3164691.svg)](https://doi.org/10.5281/zenodo.3164691) - Pythia samples
<br>
[![DOI](/img/zenodo.3066475.svg)](https://doi.org/10.5281/zenodo.3066475) - Herwig samples
"""

#   ____   _____               _ ______ _______ _____
#  / __ \ / ____|             | |  ____|__   __/ ____|
# | |  | | |  __              | | |__     | | | (___
# | |  | | | |_ |         _   | |  __|    | |  \___ \
# | |__| | |__| | ______ | |__| | |____   | |  ____) |
#  \___\_\\_____||______||\____/|______|  |_| |_____/

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2021 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

import warnings

import numpy as np

from energyflow.utils.data_utils import _get_filepath, _pad_events_axis1

__all__ = ['load']

NUM_PER_FILE = 100000
MAX_NUM_FILES = 20
URLS = {
    'pythia': {
        'nobc': {
            'dropbox': [
                'https://www.dropbox.com/s/fclsl7pukcpobsb/QG_jets.npz?dl=1',
                'https://www.dropbox.com/s/ztzd1a6lkmgovuy/QG_jets_1.npz?dl=1',
                'https://www.dropbox.com/s/jzgc9e786tbk1m5/QG_jets_2.npz?dl=1',
                'https://www.dropbox.com/s/tiwz2ck3wnzvlcr/QG_jets_3.npz?dl=1',
                'https://www.dropbox.com/s/3miwek1n0brbd2i/QG_jets_4.npz?dl=1',
                'https://www.dropbox.com/s/tsq80wc6ngen9kn/QG_jets_5.npz?dl=1',
                'https://www.dropbox.com/s/5oba2h15ufa57ie/QG_jets_6.npz?dl=1',
                'https://www.dropbox.com/s/npl6b2rts82r1ya/QG_jets_7.npz?dl=1',
                'https://www.dropbox.com/s/7pldxfqdb4n0kaw/QG_jets_8.npz?dl=1',
                'https://www.dropbox.com/s/isw4clv7n370nfb/QG_jets_9.npz?dl=1',
                'https://www.dropbox.com/s/prw7myb889v2y12/QG_jets_10.npz?dl=1',
                'https://www.dropbox.com/s/10r4ydro3e6nsmc/QG_jets_11.npz?dl=1',
                'https://www.dropbox.com/s/42p10sv9jedmtn0/QG_jets_12.npz?dl=1',
                'https://www.dropbox.com/s/crqdeg4arjti7cy/QG_jets_13.npz?dl=1',
                'https://www.dropbox.com/s/1e7ss2quxhkbhwy/QG_jets_14.npz?dl=1',
                'https://www.dropbox.com/s/psje9feje43buc7/QG_jets_15.npz?dl=1',
                'https://www.dropbox.com/s/8qw5bcswgrr9fl1/QG_jets_16.npz?dl=1',
                'https://www.dropbox.com/s/gcdp98bgupfk05x/QG_jets_17.npz?dl=1',
                'https://www.dropbox.com/s/jvgt17z1ufxz1ly/QG_jets_18.npz?dl=1',
                'https://www.dropbox.com/s/gbbfvy2e0slmm8v/QG_jets_19.npz?dl=1',
            ],
            'zenodo': [
                'https://zenodo.org/record/3164691/files/QG_jets.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_1.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_2.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_3.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_4.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_5.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_6.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_7.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_8.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_9.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_10.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_11.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_12.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_13.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_14.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_15.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_16.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_17.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_18.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_19.npz?download=1',
            ],
        },

        'bc': {
            'dropbox': [
                'https://www.dropbox.com/s/hlu497verxb9f4x/QG_jets_withbc_0.npz?dl=1',
                'https://www.dropbox.com/s/fi3knsjwg5dvcu6/QG_jets_withbc_1.npz?dl=1',
                'https://www.dropbox.com/s/cooz6qysvnfsqmr/QG_jets_withbc_2.npz?dl=1',
                'https://www.dropbox.com/s/ej7xteeoyc7meau/QG_jets_withbc_3.npz?dl=1',
                'https://www.dropbox.com/s/j2z30kh5u7t3ppb/QG_jets_withbc_4.npz?dl=1',
                'https://www.dropbox.com/s/d94krcfcn6ca98y/QG_jets_withbc_5.npz?dl=1',
                'https://www.dropbox.com/s/b5rfd0z3na09l99/QG_jets_withbc_6.npz?dl=1',
                'https://www.dropbox.com/s/02gkrs0pbpzxwn2/QG_jets_withbc_7.npz?dl=1',
                'https://www.dropbox.com/s/dlvquskq4fn3oy7/QG_jets_withbc_8.npz?dl=1',
                'https://www.dropbox.com/s/5yny7e4l8bu0ps2/QG_jets_withbc_9.npz?dl=1',
                'https://www.dropbox.com/s/93wu2rnnnf7og9u/QG_jets_withbc_10.npz?dl=1',
                'https://www.dropbox.com/s/1p2whcbhc19rusk/QG_jets_withbc_11.npz?dl=1',
                'https://www.dropbox.com/s/o35w4ds3d0jfkl2/QG_jets_withbc_12.npz?dl=1',
                'https://www.dropbox.com/s/shjbg6mluivyyry/QG_jets_withbc_13.npz?dl=1',
                'https://www.dropbox.com/s/k3phvslpc85qudk/QG_jets_withbc_14.npz?dl=1',
                'https://www.dropbox.com/s/vyif9her9nwstx9/QG_jets_withbc_15.npz?dl=1',
                'https://www.dropbox.com/s/jw6e31c6dmhpk4t/QG_jets_withbc_16.npz?dl=1',
                'https://www.dropbox.com/s/lgxce8v7widxzju/QG_jets_withbc_17.npz?dl=1',
                'https://www.dropbox.com/s/bj43a5a8z3nsb4n/QG_jets_withbc_18.npz?dl=1',
                'https://www.dropbox.com/s/jal2p6o85bnj33d/QG_jets_withbc_19.npz?dl=1',
            ],
            'zenodo': [
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_0.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_1.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_2.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_3.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_3.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_4.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_5.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_6.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_7.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_8.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_9.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_10.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_12.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_13.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_14.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_15.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_16.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_17.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_18.npz?download=1',
                'https://zenodo.org/record/3164691/files/QG_jets_withbc_19.npz?download=1',
            ],
        },
    },

    'herwig': {
        'nobc': {
            'dropbox': [
                'https://www.dropbox.com/s/xizexr2tjq2bm59/QG_jets_herwig_0.npz?dl=1',
                'https://www.dropbox.com/s/ym675q2ui3ik3n9/QG_jets_herwig_1.npz?dl=1',
                'https://www.dropbox.com/s/qic6ejl27y6vpqj/QG_jets_herwig_2.npz?dl=1',
                'https://www.dropbox.com/s/ea5a9wruo7sf3zy/QG_jets_herwig_3.npz?dl=1',
                'https://www.dropbox.com/s/5iz5q2pjcys74tb/QG_jets_herwig_4.npz?dl=1',
                'https://www.dropbox.com/s/6zha7fka0dl7t30/QG_jets_herwig_5.npz?dl=1',
                'https://www.dropbox.com/s/vljp5nhoocv2zmf/QG_jets_herwig_6.npz?dl=1',
                'https://www.dropbox.com/s/vzzl5yv9esro811/QG_jets_herwig_7.npz?dl=1',
                'https://www.dropbox.com/s/74u8y4afe1jqiyw/QG_jets_herwig_8.npz?dl=1',
                'https://www.dropbox.com/s/ra7hdq23qy7lgia/QG_jets_herwig_9.npz?dl=1',
                'https://www.dropbox.com/s/plhupkzt3ap2v6i/QG_jets_herwig_10.npz?dl=1',
                'https://www.dropbox.com/s/jy76a7tk1p7b5mq/QG_jets_herwig_11.npz?dl=1',
                'https://www.dropbox.com/s/cd4bqzk1xhg92tp/QG_jets_herwig_12.npz?dl=1',
                'https://www.dropbox.com/s/5g5rbyowni149y4/QG_jets_herwig_13.npz?dl=1',
                'https://www.dropbox.com/s/uxcgkrz4jhwdnya/QG_jets_herwig_14.npz?dl=1',
                'https://www.dropbox.com/s/brgeiph0rhooffx/QG_jets_herwig_15.npz?dl=1',
                'https://www.dropbox.com/s/jvcw8th5t6ngsk1/QG_jets_herwig_16.npz?dl=1',
                'https://www.dropbox.com/s/hlgksqbpuw3wuo3/QG_jets_herwig_17.npz?dl=1',
                'https://www.dropbox.com/s/yjvnt2z0h1zhvns/QG_jets_herwig_18.npz?dl=1',
                'https://www.dropbox.com/s/8qzs744mx383n2i/QG_jets_herwig_19.npz?dl=1',
            ],
            'zenodo': [
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_0.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_1.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_2.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_3.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_4.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_5.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_6.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_7.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_8.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_9.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_10.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_11.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_12.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_13.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_14.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_15.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_16.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_17.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_18.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_19.npz?download=1',
            ],
        },

        'bc': {
            'dropbox': [
                'https://www.dropbox.com/s/qv5t4171ez82kqr/QG_jets_herwig_withbc_0.npz?dl=1',
                'https://www.dropbox.com/s/mae1hmudq8v0tqr/QG_jets_herwig_withbc_1.npz?dl=1',
                'https://www.dropbox.com/s/6yc14771808mf0w/QG_jets_herwig_withbc_2.npz?dl=1',
                'https://www.dropbox.com/s/ihuffb5nzblw2mr/QG_jets_herwig_withbc_3.npz?dl=1',
                'https://www.dropbox.com/s/nygld5xtxmg7id2/QG_jets_herwig_withbc_4.npz?dl=1',
                'https://www.dropbox.com/s/zn76rajxowk91hn/QG_jets_herwig_withbc_5.npz?dl=1',
                'https://www.dropbox.com/s/uajiizu1k5d24x0/QG_jets_herwig_withbc_6.npz?dl=1',
                'https://www.dropbox.com/s/xcw7nfkr4r7mglf/QG_jets_herwig_withbc_7.npz?dl=1',
                'https://www.dropbox.com/s/hlvgl69hig6nepp/QG_jets_herwig_withbc_8.npz?dl=1',
                'https://www.dropbox.com/s/3cbtd73z0mdop7l/QG_jets_herwig_withbc_9.npz?dl=1',
                'https://www.dropbox.com/s/zadw2vjo71mmfkf/QG_jets_herwig_withbc_10.npz?dl=1',
                'https://www.dropbox.com/s/xivt0q49k0vccmy/QG_jets_herwig_withbc_11.npz?dl=1',
                'https://www.dropbox.com/s/ft0z5eagni71c4v/QG_jets_herwig_withbc_12.npz?dl=1',
                'https://www.dropbox.com/s/4wsui0wc0zueq3l/QG_jets_herwig_withbc_13.npz?dl=1',
                'https://www.dropbox.com/s/73kkum4kfm9jxlk/QG_jets_herwig_withbc_14.npz?dl=1',
                'https://www.dropbox.com/s/i4tflx17r1prr5u/QG_jets_herwig_withbc_15.npz?dl=1',
                'https://www.dropbox.com/s/m0xnoauoghg29zr/QG_jets_herwig_withbc_16.npz?dl=1',
                'https://www.dropbox.com/s/dtgyyflmxa86l8o/QG_jets_herwig_withbc_17.npz?dl=1',
                'https://www.dropbox.com/s/nsm8hj1lolz8qk5/QG_jets_herwig_withbc_18.npz?dl=1',
                'https://www.dropbox.com/s/t2vgtj47jy4o1di/QG_jets_herwig_withbc_19.npz?dl=1',
            ],
            'zenodo': [
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_0.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_1.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_2.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_3.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_4.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_5.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_6.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_7.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_8.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_9.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_10.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_11.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_12.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_13.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_14.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_15.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_16.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_17.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_18.npz?download=1',
                'https://zenodo.org/record/3066475/files/QG_jets_herwig_withbc_19.npz?download=1',
            ],
        },
    },
}

HASHES = {
    'pythia': {
        'nobc': {
            'sha256': [
                '3f27a02eab06e8b83ccc9d25638021e6e24c9361341730961f9d560dee12c257',
                '648e49cd59b5353e0064e7b1a3388d9c2f4a454d3ca67afaa8d0344c836ecb35',
                '09f7b16fa7edb312c0f652bb8504de45f082c4193df65204d693155017272fe9',
                '7dc9a50bb38e9f6fc1f11db18f9bd04f72823c944851746b848dee0bba808537',
                '3e6217aad8e0502f5ce3b6371c61396dfc48a6cf4f26ee377cc7b991b1d2b543',
                'b5b7d742b2599bcbe1d7a639895bca64c28da513dc3620b0e5bbb5801f8c88fd',
                '7d31bc48c15983401e0dbe8fd5ee938c3809d9ee3c909f4adab6daf8b73c14f1',
                'cec0d7b2afa9d955543c597f9b7f3b3767812a68b2401ec870caf3a2ceb98401',
                'e984620f57abe06fc5d0b063f9f84ba54bd3e8c295d2b2419a7b1c6175079ed4',
                '6e3b69196995d6eb3b8e7af874e2b9f93d904624f7a7a73b8ff39f151e3bd189',
                'fa3d386f230b806058ff17e5bd77326ff4bf01d72aa5eb3325c1df2a8825927c',
                'acd49ab7bea8f72ecf699a9a898bccacc8730474259d68406656a5a43d407fb0',
                '2edd55b8bc30c686a0637855e1ba068586eb97041e8114d5540d96db2a7a2e17',
                '7276a8a0e573f9795a47f9d5addc10d2af903c2a0ffa5c848a720ccae93daa90',
                '2068ecfa912e94cd3ce7273b7c77af0bbd5ec57940997e7483b56f03434a6869',
                '41a732ce6321dd593214225b03fb87329607ccae768c705e3896ffecc28bfcca',
                '9d68caeb18f3ccf127b9032f52e63ee011c4381293a3a503f894e5c0741ae215',
                '086053ca611bb04d97fa0b6509b4ffb6955421b067c7b277498f0e5188879331',
                'cdc595f5fedef7db9411a9f93f2786f110073b4d17a523700f625846588b1e44',
                'd07781139320ae134ce4824bc0cefa43fd5003cd97cdf3aed90d4fb12fad8a1d',
            ],
            'md5': [
                'f5d052f10a79c6e8b9382637aca0ef52',
                'f6a1081c76a47386bc11abcf0e499552',
                '2628367c57ba598f4473c870d1381041',
                'dd3ad998b0a1bd9acea2ecf029a8a921',
                'a56d6bb98361b55382aa8c06225e05d8',
                '266c688e9e6ff1cd20840692d45eaaf8',
                '95a9f7e555fb7b1073967056b9030b11',
                '4ae72aaabe121bd489532c99a6bdde95',
                'a2b80bd4199468fde4f302d346a8c9d8',
                '1157cbace488c70c9dcfc250f3345b06',
                '4b424b553e1e7f852e47ea9904bc2dcf',
                'ccd29c9d1abb34dd7cfb48cfc57a9695',
                '1ed1f6f19fb8439c9811dced41d5127d',
                'af45818c361e11ca9b3adaba30db06ad',
                '488ced3ea409d7e2b196da67f7d182ec',
                'c5e083019de6cd6a0ef12bcec1ea566b',
                '48605d55edff665f0c7d2f800b5a622e',
                '8fd47760957b5fd9adec9048b50cd1a9',
                'd43d611484b55391e891ba31c605f792',
                '6753508e34014cc69714a01fca20ec38',
            ],
        },

        'bc': {
            'sha256': [
                '27978b5dfe38f860f9899a4213f115579766ece0f6b3cd1cc043f57483521f9c',
                'bec3a147167ac19f243d74c5c47097a716cee6f6af4edc16fd0b50003ec48bd7',
                '878e415001682fda5493f15f2aaa29bce3d60b5dd882f85d85f66f2e8c5ddf9d',
                'e6f48b8fa5dfb3fa914db5dbcabe6d972d803571aa5586babececa86007d0064',
                'e11f885b97b7e859792b3f0a748f15e89c8e8788d68d1256931c948b745945e1',
                '9598efe81b4d1f5f56049508da23957fd5c90590a242f7e58255aaa44d23c192',
                '1de06fccea886445cc4250aba8e3c10990ccb24d4ae6416c6f1f40811117c13e',
                'df6d44f5c37e5c1bf6a8c7cbfb9633413348c908b5f0c657ec32e8dde781ca95',
                'fa6a01f90cdf3394b77bdec7e1931b2dfa4d6670ad8fbc0266eb14c352456e93',
                'e8c7545edb1bc52a0ea0b6cf8541a3ab825970c337458e5c43402c13939e949d',
                '147a6f80eb191577092c3bc404c90ab5538e2887b23f27d6e91e3643c5a18119',
                'c1068b3ac7d0f94ec928538a451d9e20feb7b0281deb083661f8fc6bffa7c4f1',
                '78545b1517f099b0001f6dafe046a9d39a93d13e29db812f6fa415301bfa590d',
                '8d90ffc18358dfb234359804da6a37bb188fa45c9e24468b0cc91f24ba6e0a1d',
                '4ccdf00ef7948721bfe41cd61cb3eac2025e4869b226c9f22524cf2adb9ed2a0',
                'ca09495a8b5b27435d8523aaca6f3af6795a9a61c17d758be2c1c5560e5423f1',
                '00318c05ef530f7b2e867f4b1b9900e9535add46455f6fc39ad4777ff9c71e00',
                'a23c05633f18b1a047102943f550c8bb5cfc62135a2d543ccc7cdef333996f5c',
                '0883dcd7feebef2f6419ca084b9ecc0c28c5ada68eb56f6062a943d3fe4bad81',
                'e60a6c2bac382f51147b8587119b589b134f39724fec598e28193c9b03f70fa5',
            ],
            'md5': [
                'e9ac4044a07f56a919a96e2b30c15fed',
                '5e0c0a08c5de47b190b514ce49ff4e94',
                '7c4209b14f778bb6c8cb6833a3d47854',
                'cabbd75d4313e07bf8cd8e3479e06c18',
                '87e793e74e5e665e40c1ece764952934',
                'fd0ff359e1e64023b1c1f05e854c7180',
                'fed4dfe45618598c853f8d9d24a40afd',
                '4220bee4b081850e41b3462c06538bda',
                'a7d76ec2ab2ef6d5777de5574e289c26',
                '40b9d803d1579ef77d6eda733e385e22',
                'bd5871239ddadcf080e62ffea07bb122',
                'd7731f5abdf7d3e0c17eca846c25eff3',
                '68514a76032dddb3b9cfd38654a4d433',
                'd3dca7cb617c66e6f58ea248a10b5ccb',
                '625a2d19e9b3ac6907362be3cefc404c',
                'dae5923aba89ad393e1c59c63f2552e2',
                'c611ace22f23518ae20d9e828fb5d0bc',
                '821cff3746d2fa07ad1b9feb056dd88b',
                '5c4775a99d18ac713360d9bbc43bfb43',
                '5fd9bcfa8baafb6b3f6efb5114420976',
            ],
        },
    },

    'herwig': {
        'nobc': {
            'sha256': [
                '0527349778c0ab2f7da268975fb9e7c0705c88d60f2c478401d941b9913f4d44',
                '65ef3b4cced4e2618c2bf8f3c66ef707dbd7a9740825f93549732d64e60d7ea8',
                'f13dab1937e40d0c05b97e9813fb4dda5156a8f6b4e41a89cc13821d02a60f58',
                '7b55e26262f2c156b15014b796d0a7e7a5254a982170f45cf2d9857b1f23b5f7',
                '3a5006da4a05192636a74fc818256fce215970c626719738cae9f82e3f068646',
                '2601564aee41aa5851392d6b3d12021f978fa17199e42dde004e35d1175055ea',
                '2c1fc34e99816a0bb5a84f68fa42f6314252521f6b32384a118cdec872ea97a1',
                '4b05f17acb046ad50232987003b89a91800cc713eefd81142ffeb42259369fb2',
                '150cbe132a2ee3178ba3a93a6b1733b3498b728db91f572295b6213d287ec1f7',
                '7d74c90843c751ade4cac47f6c2505da8bcbaf8645bc3f9870bdca481ff805fd',
                'e2b9072da8436618c602fbcf2409fe9be9a46dea7cff1fcc36f1ba8fefa6842d',
                'c69f499b7ea09029da7e78dcc527feca6b1680685e3c9a481db292d5518e3f1c',
                'db25d85d3a35978c607f9b5b0b52f4140c984eb5a5ab236cbf3e6eb34ead761c',
                '9a51ddd383e32154fc504ddcb138e54f0f1bd35079fe5cfa9139839c229cd78e',
                'be0fa462ea907d36972c8573b9a2f6bcdf5cf66648fa397739d12ecb677948e5',
                '7b17400c6867243e8137bd97e0f9743682a5d8c772685a6654f42f1fa2731960',
                '4c484f6508180c0e4e4a5c90b37d1b15cc67afaa3c5998306e8e633848ce6dfc',
                'd1f9baf3a3a148080d1735130f6b18f0f598991a8d886eff3c427b2e2265fce1',
                'bb70219d78e1d92091efacbf933da632670c8318d00777e4144d9a0c782e5749',
                '94975b3d999868485780d2d9e4330273aa8f0db4b9a7f6094d360f637659a264',
            ],
            'md5': [
                'a9de310c35c5a83ea592ef93070ff2f3',
                'd6ff8cc5c6192309fba915114fdc8358',
                '625bc4a0619b5b2551e273be493c6092',
                '821b293d4e68db8b2bd40a1732d1d865',
                '415dded70fca2ae5e555cdee776724d8',
                '242e23df1b837b9afac880383157a161',
                '068eb955146f773c1b5815dd3424c434',
                'e0212b768f57344ae60df7783ba5ba25',
                'd1a082794c84d2b0cc159034cf4d44b6',
                'f2e1c99033a2ff7d97d9968d394333ba',
                '5eab363df8bdff106f53858e60fe7ed1',
                '2b365047d797207009e3b40f3ec71669',
                '02317fda982357aa3874fd8c6c0e5863',
                'a102b2056cc08e7c7a9312461151e749',
                '628acd4a4e7b39d8b8e3675f4d91a3d0',
                'f19512ae26b5a54930bb57d7b3ab8672',
                'c71796ceae838945710c53ce95a33297',
                'b9e26d977f8b0e638f5db42cd1c3bcee',
                '6339009c2f5e06ac3c7b2ddade3b3e68',
                'efdec3aa9194e835f2773be6fb424054',
            ],
        },

        'bc': {
            'sha256': [
                '173b2ee0c5466772997d6b9f6a8ce25531ae3666ee17e73df1807a707aefbb17',
                'd63f272b7a5be9b75ba26082eb76107e882b756a2285cab0c16ed69d77c16366',
                '59369e725de3f688b231993bc7fbca45c2c1cc1da252aee5160298e83ce303cf',
                'bf5f2e8d6ce306796dc4e3ce9e6a88faa6ddb9b482b714060a11aa257b0fe1e6',
                'db5f5dd682f6f48e1b4900e2f366eddbb4089cb14f745cd2fcc79b23ff9f8104',
                '03833b850fa2c1ad050c753b7be9e08086bbb2b55ae41dfc1921c617cc85a622',
                'a7f94530366d886ab16adbb786671782dc4001e76ffb87937e65852449fe4f9c',
                '0c1f326807a5d5d57398aec6c5816d7c9dc644fd80e56af876fa011b139cb163',
                '1f272ba63ccf9d0af72add26075e9fc6c57e4bb954a2a31a542c06d328061742',
                'd36cd5bbaf84a4b01faa807489133ad5ca283f64f1602e724684f1e1c2996be6',
                'a4e735dcf69de0e635974a5eafb14e7cfa894e2db29a342f04be8660dc8f190f',
                'c3b293be7c5cc45f65a94835d4f8c6920abe9b5c8f4baa3f10afe1dfd0112af0',
                '2a1b54e081692eae967f117f6e8867a46f5d7d7a9d7c9353342d3cb72414d62c',
                '6a7e846baccc8076563ed6bcb5349c2f07c4802ffeefed7fe3350680d49fc9a9',
                'fb203a1273e9ecf93176cb63123f4a5ae39d803322bbd666679651715e1c8617',
                'b9493efda0e3e1c4fdcb684486b1157d73a9ca9f29142125cded06d5a85b5df4',
                '1a8e6fcefb81805d7f22bffa5b45a71d9a0fcb97a490d1866865ea7196d94443',
                '2e961954b1ca4642cc2434789e61cd0e2eb8f18f8e77b120950de1c81ed94a15',
                '34129cd813aa54539837ad57e484206f20f1da0330ae5fab6b378707ea41ca25',
                '3895ad282094f3c248bdee5ff1c57247476f03accafd573b60e0ac0d0463fe0b',
            ],
            'md5': [
                '2acf8751843b18d97fc9d2c5cc1ada6e',
                'd900130f470a16edf95faef844169c98',
                '1ff02a8f99e16645ce1aeea262d48e08',
                'cd8c63195966bec92846fd085e696a24',
                '4d1d284cadc6a2c682f7673611c4d583',
                'd6dccec5c6a7c80d967e2cc2bf5955f6',
                'e4bc0643113a41820895a8911c8b54f8',
                '814d109e673c78fd7f7c15143587c78b',
                '252a31659ebacb17ebd1a41250fa8546',
                'adb64fcc128744fbe8944b865a92a6e6',
                'e7c151d8af531840823c53ac67518e2e',
                'e4535cb57361eb31b51542ced6604626',
                'a752c262f3e0ac1eda48d49496eaf46c',
                'ad486b64b76ba73e93ee0ef1aaf2b3ba',
                '0019446390bd46ed7f8679da57c5ced0',
                '8942d2900be5d924ba5fe859daea6947',
                'c9a01fb2d9bd6a4f18ebc721e2d91b1c',
                'c9d31a2b8baeb44d9b1f821b023669b1',
                '253e7ef9d9edf37306f7d2b769377a95',
                '185902f2aba6c7c1b79dea24d9997146',
            ],
        },
    },
}

GENERATORS = frozenset(URLS.keys())
SOURCES = ['dropbox', 'zenodo']

# load(num_data=100000, pad=True, ncol=4, generator='pythia',
#      with_bc=False, cache_dir='~/.energyflow')
def load(num_data=100000, pad=True, ncol=4, 
         generator='pythia', with_bc=False, cache_dir='~/.energyflow'):
    """Loads samples from the dataset (which in total is contained in twenty 
    files). Any file that is needed that has not been cached will be 
    automatically downloaded. Downloading a file causes it to be cached for
    later use. Basic checksums are performed.

    **Arguments**

    - **num_data** : _int_
        - The number of events to return. A value of `-1` means read in all
        events.
    - **pad** : _bool_
        - Whether to pad the events with zeros to make them the same length.
        Note that if set to `False`, the returned `X` array will be an object
        array and not a 3-d array of floats.
    - **ncol** : _int_
        - Number of columns to keep in each event.
    - **generator** : _str_
        - Specifies which Monte Carlo generator the events should come from.
        Currently, the options are `'pythia'` and `'herwig'`.
    - **with_bc** : _bool_
        - Whether to include jets coming from bottom or charm quarks. Changing
        this flag does not mask out these jets but rather accesses an entirely
        different dataset. The datasets with and without b and c quarks should
        not be combined.
    - **cache_dir** : _str_
        - The directory where to store/look for the files. Note that 
        `'datasets'` is automatically appended to the end of this path.

    **Returns**

    - _3-d numpy.ndarray_, _1-d numpy.ndarray_
        - The `X` and `y` components of the dataset as specified above. If
        `pad` is `False` then these will be object arrays holding the events,
        each of which is a 2-d ndarray.
    """

    # check for valid options
    if generator not in GENERATORS:
        raise ValueError("'generator' must be in " + str(GENERATORS))

    # get number of files we need
    num_files = int(np.ceil(num_data/NUM_PER_FILE)) if num_data > -1 else MAX_NUM_FILES
    if num_files > MAX_NUM_FILES:
        warnings.warn('More data requested than available. Providing the full dataset.')
        num_files = MAX_NUM_FILES
        num_data = -1

    # index into global variables
    bc = 'bc' if with_bc else 'nobc'
    urls = URLS[generator][bc]
    hashes = HASHES[generator][bc]

    # obtain files
    Xs, ys = [], []
    for i in range(num_files):
        for j,source in enumerate(SOURCES):
            try:
                url = urls[source][i]
                filename = url.split('/')[-1].split('?')[0]

                fpath = _get_filepath(filename, url, cache_dir, file_hash=hashes['sha256'][i])

                # we succeeded, so don't continue trying to download this file
                break

            except Exception as e:
                print(str(e))

                # if this was our last source, raise an error
                if j == len(SOURCES) - 1:
                    m = 'Failed to download {} from any source.'.format(filename)
                    raise RuntimeError(m)

                # otherwise indicate we're trying again
                else:
                    print("Failed to download {} from source '{}', trying next source...".format(filename, source))

        # load file and append arrays
        with np.load(fpath) as f:
            Xs.append(f['X'])
            ys.append(f['y'])

    # get X array
    if pad:
        max_len_axis1 = max([X.shape[1] for X in Xs])
        X = np.vstack([_pad_events_axis1(x[...,:ncol], max_len_axis1) for x in Xs])
    else:
        X = np.asarray([x[x[:,0]>0,:ncol] for X in Xs for x in X], dtype='O')

    # get y array
    y = np.concatenate(ys)

    # chop down to specified amount of data
    if num_data > -1:
        X, y = X[:num_data], y[:num_data]

    return X, y
    