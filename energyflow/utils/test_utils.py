from __future__ import absolute_import

from pytest.mark import parameterize

from energyflow.utils import *

def epsilon_close(X, Y, epsilon=10**-14):
    return np.all(np.abs(X - Y) < epsilon)

@parameterize('nparticles', [10,100])
@parameterize('nevents', [20,200])
def test_rambo_phase_space(nparticles, nevents)
    events = gen_massless_phase_space(nparticles, nevents)
    assert epsilon_close(mass2(events), 0)