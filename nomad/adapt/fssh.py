"""Fewest switches surface hopping algorithm"""

"""Steps in the algorithm: 
    1. Start with one populated state
    2. Propagate tragectory for one time step
    3. determine switching probabilities for each possible switch at that time
    3b. generate a random number, and determine if switch will occur
    4. If no switch, back to step 2
    """

import sys
import random
import numpy as np
import scipy as sp 
import nomad.core.glbl as glbl
import nomad.core.log as log
import nomad.common.constants as constants
import nomad.common.linalg as lalg
import nomad.adapt.utilities as utils

a_cache = None 
Vij     = None
dij     = None
r       = None
r_dot   = None
ns      = None
switch_times = None
# want to propagate amplitudes from wfn0.time to wfn0.time+dt
def adapt(wfn0, wfn, dt):
    """Calculates the probability of a switch between states, tests this against a random number, and switches states accordingly."""
    global a_cache, Vij, dij, r, r_dot, ns

    # this is first time adapt is called, initialize
    if a_cache is None:
        fssh_initialize(wfn0)

    traj0      = wfn0.traj[0]
    traj       = wfn.traj[0]
    t0         = wfn0.time
    local_time = t0
    local_dt   = dt/100.
    current_st = traj0.state
        
    r     = traj0.x()
    r_dot = traj0.velocity()
    if glbl.methods['surface'] == 'diabatic':
        Vij = traj0.pes.get_data('diabat_pot')
    else:
        Vij = np.diag(traj0.pes.get_data('potential'))
        dij = traj0.pes.get_data('derivative')
        for i in range(ns):
            dij[:,i,i] = np.zeros((traj0.dim),dtype='float')

    #Check that we only have one trajectory:
    if wfn0.n_traj() > 1 or wfn.n_traj() > 1:
        sys.exit('fssh algorithm must only have one trajectory')

    #this loop is called for each small time step 
    current_a = a_cache.copy()
    while local_time < t0 + dt:
        flat_a      = current_a.ravel()
        new_a       = glbl.modules['propagator'].propagate(flat_a, a_dot, local_dt)
        current_a   = np.reshape(new_a, (ns,ns))
        local_time += min(local_dt, t0 + dt - local_time)

    a_cache = current_a.copy()
    b = compute_b(a_cache)

    probs = [ max(0, dt * b[current_st][st] / current_a[current_st][current_st])  for st in range(ns)]
    probs[current_st] = 0
    if any([ abs(x.imag) > constants.fpzero for x in probs]):
        sys.exit("Error: complex hopping probability: "+str(probs))
    else:
        probs = [x.real for x in probs]
    probs_copy = probs.copy()
    probs = [sum(probs_copy[i] for i in range(st+1)) for st in range(ns)]

    random_number = np.random.random()
    
    for st in range(glbl.properties['n_states']):

        if st == current_st:
            continue

        switch_prob = probs[st]

        #Check probability against random number, see if there's a switch
        if switch_prob > random_number: 
            log.print_message('general',['Attempting surface hop to state ' + str(st) + ' at time ' + str(local_time)]) 

            target_energy  = traj.classical()
            new_traj       = traj.copy()
            new_traj.state = st 
            adjust_success = utils.adjust_momentum(new_traj, target_energy, dij[:,current_st, st])

            if adjust_success:
                log.print_message('general',['Surface hop successful, switching from state ' + str(current_st) + ' to state ' + str(st)])
                wfn.traj[0] = new_traj.copy()
            else:
                log.print_message('general',['Frustrated hop, momentum adjustment not possible. Remaining on state ' + str(current_st)])                
            break
    # basis has not grown
    return False

############################################################################

#
def fssh_initialize(traj):
    global switch_times, a_cache, Vij, dij, ns

    ns         = glbl.properties['n_states']
    init_state = glbl.properties['init_state']
    Vij        = np.zeros((ns,ns), dtype='float')
    dij        = np.zeros((ns,ns), dtype='float')
    a_cache    = np.zeros((ns,ns), dtype='complex')
    a_cache[init_state, init_state] = complex(1.,0.)
    switch_times = 0
    if (glbl.methods['surface'] == 'diabatic' and 
        'diabat_pot' not in traj.pes.avail_data()):
        sys.exit('Cannot use diabatic potentials: diabat_pot not found')

    return

#
def a_dot(a_flat):
    global Vij, dij, r_dot, ns
    a = np.reshape(a_flat, (ns, ns))   
    im    = complex(0.,1.)
    a_dot = np.array([[sum(a[l,j] * (-im*Vij[k,l] - np.dot(r_dot, dij[:, k,l])) - a[k,l] * (-im*Vij[l,j] - np.dot(r_dot, dij[:, l, j])) 
        for l in range(ns)) 
        for j in range (ns)]
        for k in range(ns)])
    return [a_dot.ravel()] # just returns first derivative

def c_dot(c):
    global Vij, dij, r_dot, ns
    im    = complex(0., 1.)
    c_dot = [sum(c[j]*(-im*Vij[k,j] - np.dot(r_dot, dij[:,k,j])) 
                 for j in range(ns)) 
                 for k in range(ns)]
    return c_dot

#
def compute_b(a):
    global Vij, dij, r, ns

    im = complex(0., 1.)
    b = [[ 2 * (a[k][l].conjugate() * Vij[k,l]).imag - 
        2 * (a[k][l].conjugate() * np.dot(r_dot, dij[:,k,l])).real 
           for k in range(ns)] for l in range(ns)]
    return b

#
def in_coupled_regime(wfn):
    return False
