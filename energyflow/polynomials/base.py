from __future__ import absolute_import
import multiprocessing as mp

from energyflow.utils import Measure

class EFPBase(Measure):

    def __init__(self, measure, beta, normed, check_type):

        # initialize measure class
        Measure.__init__(self, measure, beta, normed, check_type)

    def batch_compute(self, events, concat_disc=True, n_jobs=None):

        PROCESSES = n_jobs
        if PROCESSES is None:
            try: 
                PROCESSES = mp.cpu_count()
            except:
                PROCESSES = 4 # choose reasonable value

        with mp.Pool(PROCESSES) as pool:
            chunksize = int(len(events)/PROCESSES)
            results = np.asarray(list(pool.imap(self.compute, events, chunksize)))

        if concat_disc:
            return self.calc_disc(results, self.disc_formulae, concat=True)
        else: 
            return results



