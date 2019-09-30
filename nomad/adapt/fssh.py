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
import scipy.constants as sp_con
#random.seed(4611)

a_cache = {}
data_cache = {}
#initial state probability (a) matrix:
def init_a():
    init_state = glbl.properties['init_state']
    n_states = glbl.properties['n_states']
    init_a_matrix = np.zeros((n_states, n_states), dtype = 'complex')
    init_a_matrix[init_state,init_state] = 1
    a_cache['init'] = init_a_matrix

init_a()

current_a = a_cache['init']

def adapt(wfn, dt):
    """Calculates the probability of a switch between states, tests this against a random number, and switches states accordingly."""
    global data_cache
    global a_cache
    global current_a
    traj = wfn.traj[0]
    current_time = wfn.time
    #the "local time", or time between larger time steps:
    local_time = current_time
    #the smaller time step:
    local_dt = dt/10
    current_st = traj.state

        #Check that we only have one trajectory:
    if wfn.n_traj() > 1:
        sys.exit('fssh algorithm must only have one trajectory')


    def propagate_a():
        """propagates the matrix of state probabilities and coherences"""
        
        a_copy = current_a.copy()

        n_states = glbl.properties['n_states']
        for k in range(n_states):
            for  j in range(n_states):
                def a_dot(a):
                    """returns the time derivative of the specified a matrix element"""
                    dq0 = 0
                    dq1 = 0
                    for l in range(n_states):
                        term1 = a_copy[l,j] * np.complex(diabat_pot(k,l), -nac(k,l))
                        term2 = a_copy[k,l] * np.complex(diabat_pot(l,j), -nac(l,j))
                        both_terms = (term1 - term2) / (np.complex(0,1))
                        dq1 += both_terms
                    dq = [dq1, 0, 0, 0]
                    return dq
                #actually propagate a
                akj =  glbl.modules['propagator'].propagate([a_copy[k,j]], a_dot, local_dt)
                current_a[k,j] = akj

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



    #this loop is called for each small time step 
    while local_time < current_time + dt:
        random_number = random.random() 
        prev_prob = 0
        for st in range(glbl.properties['n_states']):
        #can only switch between states
            if st == current_st:
                continue
            #Calculate switching probability for this transition:
            state_pop = current_a[current_st, current_st]
            state_flux = (2 * np.imag(np.conj(current_a[current_st, st]) * diabat_pot(current_st, st))) - (2 * np.real(np.conj(current_a[current_st, st]) * nac(current_st, st)))
                
            switch_prob = state_flux * local_dt / np.real(state_pop)
            if switch_prob < 0:     
                switch_prob = 0

            #add to the previous probability
            switch_prob = prev_prob + switch_prob
              
        

            #Check probability against random number, see if there's a switch
            if prev_prob < random_number <= switch_prob: 
                total_E = traj.classical()
                current_T = traj.kinetic() 
                log.print_message('general',['Attempting surface hop to state ' + str(st)]) 
                #change the state:
                traj.state = st
                new_V = traj.potential()
                new_T = total_E - new_V
                #is this hop classically possible? If not, go back to previous state:
                if new_T < 0:
                    log.print_message('general',['Surface hop failed (not enough T), remaining on state ' + str(current_st)])
                    traj.state = current_st
                else:
                    log.print_message('general',['Surface hop successful, switching to state ' + str(st)])
                    #calculate and set the new momentum:
                    scale_factor = np.sqrt(new_T/current_T)
                    current_p = traj.p()
                    new_p = scale_factor * current_p
                    traj.update_p(new_p)
                    #for reflections on st 0:
                    if st == 0:
                        print(2)
        if local_time >= glbl.properties['simulation_time'] +glbl.properties['default_time_step'] - local_dt:
            print(current_st)


        propagate_a()
        #update the local time:
        local_time = local_time + local_dt


def in_coupled_regime(wfn):
    return False
   
