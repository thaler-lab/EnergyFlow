import sys
import time

import energyflow as ef

dmax = 7 if len(sys.argv) < 2 else int(sys.argv[1])

start = time.time()
ef.Generator(dmax=dmax).save('efps_d_le_{}'.format(dmax))
print('Finished generating EFPs in {:.3f}s'.format(time.time() - start))