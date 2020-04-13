  # standard library imports
from __future__ import absolute_import, division, print_function

# standard numerical library imports
import numpy as np

# matplotlib is required for this example
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (4,4)

#############################################################
# NOTE: install ffmpeg and point matplotlib to it
#############################################################
# on linux
plt.rcParams['animation.ffmpeg_path'] = '/home/username/anaconda/envs/env_name/bin/ffmpeg'

# on windows
#plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'

import energyflow as ef
import ot
from matplotlib import animation, rc
from IPython.display import HTML

# helper function to compute the EMD between (potentially unbalanced) jets
def emde(ev0, ev1, R=1, return_flow=False):
    pTs0, pTs1 = np.asarray(ev0[:,0], order='c'), np.asarray(ev1[:,0], order='c')
    thetas = ot.dist(ev0[:,1:3], ev1[:,1:3], metric='euclidean')
    
    # add a fictitious particle to the lower-energy event to balance the energy
    pT0, pT1 = pTs0.sum(), pTs1.sum()       
    pTs0 = np.hstack((pTs0, 0 if pT0 > pT1 else pT1-pT0))
    pTs1 = np.hstack((pTs1, 0 if pT1 > pT0 else pT0-pT1))
    
    # make its distance R to all particles in the other event
    Thetas = R * np.ones((np.shape(thetas)[0]+1,np.shape(thetas)[1]+1))
    Thetas[:-1,:-1] = thetas
    
    return ot.emd2(pTs0, pTs1, Thetas) if not return_flow else ot.emd(pTs0, pTs1, Thetas)

# helper function to interpolate between the optimal transport of two events
def merge(ev0, ev1, R=1, lamb=0.5):    
    G = emde(ev0, ev1, R=R, return_flow=True)
    
    merged = []
    for i in range(len(ev0)):
        for j in range(len(ev1)):
            if G[i, j] > 0:
                merged.append([G[i,j], ((lamb)*ev0[i,1]+(1-lamb)*ev1[j,1]), ((lamb)*ev0[i,2]+(1-lamb)*ev1[j,2])])
            
    for i in range(len(ev0)):
        if G[i,-1] > 0:
            merged.append([G[i,-1]*lamb, ev0[i,1], ev0[i,2]])

    for j in range(len(ev1)):
        if G[-1,j] > 0:
            merged.append([G[-1,j]*(1-lamb), ev1[j,1], ev1[j,2]])            
            
    return np.asarray(merged)


#############################################################
# ANIMATION OPTIONS
#############################################################
zf = 30          # size of points in scatter plot
lw = 1           # linewidth of flow lines
fps = 60         # frames per second
nframes = 10*fps # total number of frames


#############################################################
# LOAD IN JETS
#############################################################
specs = ['375 <= corr_jet_pts <= 425', 'abs_jet_eta < 1.9', 'quality >= 2']
events = ef.mod.load(*specs, dataset='cms', amount=0.01)

# events go here as lists of particle [pT,y,phi]
event0 = events.particles[14930][:,:3]
event1 = events.particles[19751][:,:3]

# center the jets
event0[:,0] /= 100
event0[:,1] -= np.average(event0[:,1], weights=event0[:,0])
event0[:,2] -= np.average(event0[:,2], weights=event0[:,0])

event1[:,0] /= 100
event1[:,1] -= np.average(event1[:,1], weights=event1[:,0])
event1[:,2] -= np.average(event1[:,2], weights=event1[:,0])

ev0 = np.copy(event0)
ev1 = np.copy(event1)


#############################################################
# MAKE ANIMATION
#############################################################

fig, ax = plt.subplots()

merged = merge(ev0, ev1, lamb=0)
pts, ys, phis = merged[:,0], merged[:,1], merged[:,2]

scat = ax.scatter(ys, phis, color='blue', s=pts)

# animation function. This is called sequentially
def animate(i):
    ax.clear()
    
    nstages = 4
    
    # first phase is a static image of event0
    if i < nframes / nstages:
        lamb = nstages*i/(nframes-1)
        ev0  = event0
        ev1  = event0
        color = (1,0,0)
    
    # second phase is a transition from event0 to event1
    elif i < 2 * nframes / nstages:
        lamb = nstages*(i - nframes/nstages)/(nframes-1)
        ev0  = event1
        ev1  = event0
        color = (1-lamb)*np.asarray([1,0,0]) + (lamb)*np.asarray([0,0,1])
    
    # third phase is a static image of event1
    elif i < 3 * nframes / nstages:
        lamb = nstages*(i - 2*nframes/nstages)/(nframes-1)
        ev0  = event1
        ev1  = event1
        color = (0,0,1)
    
    # fourth phase is a transition from event1 to event0
    else:
        lamb = nstages*(i - 3*nframes/nstages)/(nframes-1)
        ev0  = event0
        ev1  = event1
        color = (lamb)*np.asarray([1,0,0]) + (1-lamb)*np.asarray([0,0,1])

    merged = merge(ev0, ev1, lamb=lamb)
    pts, ys, phis = merged[:,0], merged[:,1], merged[:,2]
    scat = ax.scatter(ys, phis, color=color, s=zf*pts, lw=0)
    
    ax.set_xlim(-1,1); ax.set_ylim(-1,1);
    ax.set_axis_off()
    
    return scat,

anim = animation.FuncAnimation(fig, animate, frames=nframes, repeat=True)
anim.save('energyflowanimation.mp4', fps=fps, dpi=200)
HTML(anim.to_html5_video())