import sys

import energyflow as ef

# use a fraction of the full datasets (amount=1.0 uses the full datasets)
amount = float(sys.argv[1])

# load the CMS (cms), Pythia-generated (gen), and detector-simulated (sim) datasets
cms = ef.mod.load(dataset='cms', amount=amount, store_pfcs=False)
sim = ef.mod.load(dataset='sim', amount=amount, store_pfcs=False, store_gens=False)
gen = ef.mod.load(dataset='gen', amount=amount, store_gens=False)
