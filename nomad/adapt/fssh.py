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

a_cache = {}

#initial state probability (a) matrix:
def init_a():
    init_state = glbl.properties['init_state']
    n_states = glbl.properties['n_states']
    print('initial number of states' + str(n_states))
    init_a_matrix = np.zeros([n_states, n_states])
    init_a_matrix[init_state,init_state] = 1
    a_cache['init'] = init_a_matrix

init_a()

current_a = a_cache['init']


def adapt(wfn, dt):
    """Calculates the probability of a switch between states, tests this against a random number, and switches states accordingly."""
    
    global a_cache
    global current_a
    #gets the current time    
    current_time = wfn.time

    #generates random number:
    random_number = random.random() 
        

    prev_prob = 0

    #Check that we only have one trajectory:
    if wfn.n_traj() > 1:
        sys.exit('fssh algorithm must only have one trajectory')
    
    traj = wfn.traj[0]
    
    #propagate the state probabilities:
    propagate_a(traj, dt)
        
    for st in range(glbl.properties['n_states']):
    #can only switch between states
        if traj.state == st:
            continue
        print ("current state:"+ str(traj.state), "checking state" + str(st))
        
        #Calculate switching probability for this transition:
        state_pop = current_a[st,st]
        state_flux = 7
        #placeholder:
        switch_prob = 0.01 
        print("switching Probability:" + str(switch_prob), "random:" + str(random_number)) 
        if switch_prob < 0:     
            switch_prob = 0
        #add to the previous probability
        switch_prob = prev_prob + switch_prob
              
        #Check probability against random number, see if there's a switch
        if prev_prob < random_number <= switch_prob:
            print('Surface hop to state', st)
            #change the state:
            traj.state = st

            #If no switch to this state (or any higher state):
        else:
            break 
                

def in_coupled_regime(wfn):
    return False
"""
def a_dot(traj, ordr):
    return [0,5,0]
   """ 
def propagate_a(traj, dt):
    print('-----------------------------------------------------------------------------------------------------------')
    def nac(st1, st2):
        #set up a cache for this!
        "returns the nonadiabatic coupling between two states"""
        vel = traj.velocity()   
        deriv = traj.pes.get_data('derivative')
        coup = np.dot(vel, deriv[:,st1, st2])
        return coup

    global current_a

    
    #a copy for calculations: 
    a_copy = current_a.copy()

    n_states = glbl.properties['n_states']
    for k in range(n_states):
        for  j in range(n_states):
            #function that calculates the derivatives of this a element
            def a_dot(a, ordr):
                diabat_pot = traj.pes.get_data('diabat_pot')
                dq0 = 0
                dq1 = 0
                for l in range(n_states):
                    term1 = a_copy[l,j] * np.complex(diabat_pot[k,l], -sp_con.hbar * nac(k,l))
                    term2 = a_copy[k,l] * np.complex(diabat_pot[l,j], -sp_con.hbar * nac(l,j))
                    both_terms = (term1 - term2) / (np.complex(0, -sp_con.hbar))
                    dq1 += both_terms
                    dq = [0, dq1, 0, 0, 0, 0]
                return dq

            #actually propagate a
            akj =  glbl.modules['propagator'].propagate(a_copy[k,j], a_dot, dt)
            print(str(akj))
            current_a[k,j] = akj
            print("current a matrix:"+ str(current_a))
            print("just to check:" + str(current_a[0,0] + current_a[1,1]))   

