"""Fewest switches surface hopping algorithm"""

"""Steps in the algorithm: 
    1. Start with one populated state
    2. Propagate tragectory for one time step
    3. determine switching probabilities for each possible switch at that time
    3b. generate a random number, and determine if switch will occur
    4. If no switch, back to step 2
    """


import numpy as np
import nomad.core.glbl as glbl
import nomad.core.log as log
import nomad.core.stem as step
import nomad.core.surface as evaluate
import nomad.adapt.utilities as utils
import nomad.math.constants as constants

def spawn(wfn, dt):
    """Calculates the probability of a switch between states, tests this against a random number, and switches states accordingly."""

        #gets the current time    
        current_time = wfn.time

        import random
        random_number = random.random() 
        
        prev_prob = 0
        # go through all trajectories(but there only should be one):
        for i in range(wfn.n_traj())
            traj = wfn.traj[i]
            
            for st in range(glbl.properties['n_states'])
        
                #can only switch between states
                if traj.state == st
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
                if prev_prob < random_number <= switch_prob
                    log.print_message('Surface hop to state', st)
                    #change the state:
                    traj.state = st

                #If no switch to this state (or any higher state):
            else
                    break 
                





