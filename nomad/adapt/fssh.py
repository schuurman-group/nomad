"""Fewest switches surface hopping algorithm"""

"""Steps in the algorithm: 
    1. Start with one populated state
    2. Propagate tragectory for one time step
    3. determine switching probabilities for each possible switch at that time
    3b. generate a random number, and determine if switch will occur
    4. If no switch, back to step 2
    """


import random
import numpy as np
import scipy as sp 
import nomad.core.glbl as glbl
import nomad.core.log as log

c_cache = None 
Vij     = None
dij     = None
r       = None
r_dot   = None
ns      = None

# want to propagate amplitudes from wfn0.time to wfn0.time+dt
def adapt(wfn0, wfn, dt):
    """Calculates the probability of a switch between states, tests this against a random number, and switches states accordingly."""
    global c_cache, Vij, dij, r_dot, ns

    # this is first time adapt is called, initialize
    if c_cache is None:
        fssh_initialize(wfn0)

    traj       = wfn0.traj[0]
    t0         = wfn0.time
    local_time = t0
    local_dt   = dt/100.
    current_st = traj.state
        
    r     = traj.x()
    r_dot = traj.velocity()
    if glbl.methods['surface'] == 'diabatic':
        Vij = traj.pes.get_data('diabat_pot')
    else:
        Vij = np.diag(traj.get_data('potential'))
        dij = traj.pes.get_data('derivative')
        for i in range(ns):
            dij[i,i] = np.zeros((traj.dim),dtype='float')

    #Check that we only have one trajectory:
    if wfn0.n_traj() > 1 or wfn.n_traj() > 1:
        sys.exit('fssh algorithm must only have one trajectory')

    #this loop is called for each small time step 
    current_c = c_cache.copy()
    while local_time < t0 + dt:
        new_c       = glbl.modules['propagator'].propagate([current_c, c_dot, local_dt)
        current_c   = new_c
        local_time += min(local_dt, t0 + dt - local_time)

    a = compute_a(current_c)
    b = compute_b(a)
    c_cache = current_c.copy()
    
    probs = [dt * b[current_st, st] / a[current_st, current_st] for st in range(ns)]
    probs[current_st] = 0.

    # add scaling/hopping logic below...


#        for st in range(glbl.properties['n_states']):
        #can only switch between states
#            if st == current_st:
#                continue
 
#            state_pop = a(current_st, current_st) 
#            state_flux = (2*np.imag(np.conj(a(current_st, st)) * diabat_pot(current_st, st))) - ( 2* np.real(np.conj(a(current_st, st)) * nac(current_st, st)))
#            switch_prob = state_flux * local_dt / state_pop
#            if switch_prob < 0:     
#                switch_prob = 0

            #add to the previous probability
#            switch_prob = prev_prob + switch_prob

            #Check probability against random number, see if there's a switch
#            if prev_prob < random_number <= switch_prob: 
#                total_E = traj.classical()
#                current_T = traj.kinetic() 
#                log.print_message('general',['Attempting surface hop to state ' + str(st) + ' at time ' + str(local_time)]) 
#                #change the state:
#                traj.state = st
#                new_V = traj.potential()
#                new_T = total_E - new_V
                #is this hop classically possible? If not, go back to previous state:
#                if new_T < 0:
#                    log.print_message('general',['Surface hop failed (not enough T), remaining on state ' + str(current_st)])
#                    traj.state = current_st
#                else:
#                    log.print_message('general',['Surface hop successful, switching from state ' + str(current_st) + ' to state ' + str(st)])
#                    #calculate and set the new momentum:
#                    scale_factor = np.sqrt(new_T/current_T)
#                    current_p = traj.p()
#                    new_p = scale_factor * current_p
#                    traj.update_p(new_p)
#                    #to keep track, for now:
#                    global switch_times
#                    switch_times += 1
#
#                continue      
#
#        if local_time >= glbl.properties['simulation_time'] +glbl.properties['default_time_step'] - local_dt:
#            #plt.show()
#            if traj.x()[0]>0:
#                if switch_times <= 1:
#                    print(current_st)
#                else:
#                    if current_st == 0:
#                        print(2)
#                    if current_st ==1:
#                        print(3)
#            else:
#                print(4)
#
#        propagate_c()
#        #update the local time:
#        local_time = local_time + local_dt


############################################################################

#
def fssh_initialize(traj):
    global a_cache, Vij, dij, ns

    ns         = glbl.properties['n_states']
   # init_state = glbl.properties['init_state']
    Vij        = np.zeros((ns,ns), dtype='float')
    dij        = np.zeros((ns,ns), dtype='float')
    c_cache    = np.zeros((ns), dtype='complex')
    c_cache[init_state] = complex(1.,0.)

    if (glbl.methods['surface'] == 'diabatic' and 
        'diabat_pot' not in traj.pes.avail_data()):
        sys.exit('Cannot use diabatic potentials: diabat_pot not found')

    return

#a test:
#def a_dot(k,j):
#    adq = 0
#    for l in range(n_states):
#        term1 = a(l,j) * np.complex(diabat_pot(k,l), -nac(k,l))
#        term2 = a(k,l) * np.complex(diabat_pot(l,j), -nac(l,j))
#        both_terms = np.complex(0, (term2-term1))
#        adq+= both_terms
#    return adq
def c_dot(c):
    global Vij, dij, r_dot
    ns = glbl.properties['n_states']

    im    = complex(0.,1.)
    c_dot = [ sum([c[j]*(np.dot(r_dot, dij[k,j]) 
                         - im*Vij[k,j]) for j in range(ns)]) 
              for k in range(ns)]

    return [c_dot] # just returns first derivative

#
def compute_a(c):
    return [[c[k] * c[j].conjugate() for k in range(ns)] for j in range(ns)]

#
def compute_b(a):
    global Vij, dij, r, ns

    im = complex(0., 1.)
    b = [[ 2*(a[k,l].conjugate()*V[k,l]).imag - 
           2*(a[k,l]*np.dot(r,dij[k,l]).real 
           for k in range(ns)] for l in range(ns)]

    return b

#
def in_coupled_regime(wfn):
    return False


# don't think you need anything below here...
#########################################
def propagate_c():
    """propagates the values of c"""

    c_copy = current_c.copy()
    n_states = glbl.properties['n_states']
    for k in range(n_states):
        def this_c_dot(ck):
            return [c_dot(c_copy, ck, k)]
        #actually propagate c
        ck =  glbl.modules['propagator'].propagate([c_copy[k]], this_c_dot, local_dt)
        current_c[k] = ck


def c_dot(c, ck, k):
    """returns the first time derivative of ci"""
    c_label = 'c_dot' +str(k) + str(local_time)
    if c_label in c_cache:
        return c_cache[c_label]
    dq1 = 0
    for j in range(n_states):
        if j == k:
            dq1 += ck[0] * np.complex(diabat_pot(k,j), -nac(k,j))
        else:
            dq1 += c[j] * np.complex(diabat_pot(k,j), -nac(k,j))

        c_dot= np.complex(0, -dq1)
    c_cache[c_label] = c_dot

    return c_dot

def diabat_pot(j,k):
    """Returns the specified matrix element of the diabatic potential"""
    return traj.pes.get_data('diabat_pot')[j, k]

def nac(st1, st2):
    """returns the nonadiabatic coupling between two states"""
    nac_label = str(current_time) + str(st1) + str(st2)
    if nac_label in data_cache:
        return data_cache[nac_label]
    vel = traj.velocity()
    deriv = traj.derivative(st1, st2)
    coup = np.dot(vel, deriv)
    data_cache[nac_label] = coup
    return coup

def a(k,j):
    '''returns a = c c*'''
    return current_c[k] * np.conj(current_c[j])
