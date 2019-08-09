""""""
from __future__ import absolute_import, division, print_function

from energyflow.utils.generic_utils import import_fastjet

fj = import_fastjet()

__all__ = []

if fj:

    __all__ = ['pjs_from_ptyphims', 'cluster', 'softdrop']

    def pjs_from_ptyphims(ptyphims):

        pjs = []
        for ptyphim in ptyphims:
            pj = fj.PseudoJet()
            pj.reset_PtYPhiM(*ptyphim[:4])
            pjs.append(pj)

        return pjs

    def cluster(pjs, algorithm='ca', R=fj.JetDefinition.max_allowable_R):

        algorithm_l = algorithm.lower()
        if algorithm_l  == 'kt':
            jet_alg = fj.kt_algorithm
        elif algorithm_l == 'antikt':
            jet_alg = fj.antikt_algorithm
        elif algorithm_l in {'ca', 'cambridge', 'cambridge_aachen'}:
            jet_alg = fj.cambridge_algorithm
        else:
            raise ValueError("algorithm '{}' not understood".format(algorithm))

        return fj.JetDefinition(jet_alg, R)(pjs)

    def softdrop(jet, zcut=0.1, beta=0, R=1.0):

        parent1, parent2 = fj.PseudoJet(), fj.PseudoJet()
        if not jet.has_parents(parent1, parent2):
            return jet
        
        pt1, pt2 = parent1.pt(), parent2.pt()
        z = min(pt1, pt2)/(pt1 + pt2)
 
        if z >= (zcut if beta == 0 else zcut * (parent1.delta_R(parent2)/R)**beta):
            return jet
        else:
            return softdrop(parent1 if pt1 >= pt2 else parent2, zcut=zcut, beta=beta, R=R)
