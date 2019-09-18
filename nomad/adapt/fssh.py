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

def adapt(wfn, dt):
    """Calculates the probability of a switch between states, tests this against a random number, and switches states accordingly."""

    #gets the current time    
    current_time = wfn.time
    random_number = random.random() 
        

                

               """ 
def propagate_probabilities(wfn, dt)
Propagates the state probabilities and coherences for a given state
   #make a copy of the population/coherence matrix
   copy_a = a.copy()
   
   n_states = glbl.properties['n_states']
    #need to propagate the entire matrix:
    for k in range(n_states)
        for j in range(n_states)
            
            #need initial values!
            #now propagate:
            propagate(calc_a_derivative(a0, k, j)) 


def calc_a_derivative(a0, k, j)
returns the time derivative for each element in the probability/coherence matrix
    for l in range(n_states)
     #Need to;
     #Set up initial coherences
     #Access trajectory without going through wavefunction?
     q0 = 0 
     dq1 += (a[l][j] * (v[k][l] - (np.complex(0,1) * sp.constants.hbar * wfn.traj.nact(k, l)))) - (a[k][l] * (v[l][j] - (np.complex(0,1) * sp.constants.hbar * wfn.traj.nact[l, j]))))
            dq1  = dq1 / (sp.constants.hbar * np.complex(0, 1))
            
    return [0, dq1]     """

    prev_prob = 0
    # go through all trajectories(but there only should be one):
    for i in range(wfn.n_traj()):
        traj = wfn.traj[i]

        print("calling propagate_general...")
        newx = glbl.modules['propagator'].propagate(traj, traj.x(), a_dot, dt)
        print("original x = "+str(traj.x()))
        print("new x = "+str(newx))
        
        for st in range(glbl.properties['n_states']):
    
            #can only switch between states
            if traj.state == st:
                continue
        
            #Calculate switching probability for this transition:
            coupling = traj.coupling(traj.state, st)

            #placeholder:
            switch_prob = 5 * coupling / dt

            if switch_prob < 0:
                switch_prob = 0

            #add to the previous probability
            switch_prob = prev_prob + switch_prob
              
            #Check probability against random number, see if there's a switch
            if prev_prob < random_number <= switch_prob:
                log.print_message('Surface hop to state', st)
                #change the state:
                traj.state = st

            #If no switch to this state (or any higher state):
        else:
                break 
                

def in_coupled_regime(wfn):
    return False

def a_dot(traj, ordr):
   
    if ordr == 0:
        return 0.
    if ordr == 1:
        return traj.coupling(traj.state, 1)
    else:
        return 0.
    

