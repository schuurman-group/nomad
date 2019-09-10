"""Fewest switches surface hopping algorithm"""

"""Steps in the algorithm: 
    1. Start with one populated state
    2. Propagate tragectory for one time step
    3. determine switching probabilities for each possible switch at that time
    3b. generate a random number, and determine if switch will occur
    4. If no switch, back to step 2
    """


import numpy as np
import nomad.core.global as glbl
import nomad.core.log as log
import nomad.core.stem as step
import nomad.core.surface as evaluate
import nomad.adapt.utilities as utils

def adapt(wfn, dt)
    """Documentation to follow"""

    #gets the current time    
    current_time = wfn.time
    # time_step =

    """Need:
    Size of time step (delta t)
    coupling between states
    """
      #Python's default random number generator:
      import random
      random_number = random.random() 

    for st in range(glbl.properties['n_states'])
        
        #can only switch between states
        if st == #qk current state
            continue
        
        #Calculate switching probability for this transition:



        if switch_prob < 0:
            switch_prob = 0
        #Check probability against random number, see if there's a switch

    
        #if there is a potential switch, 
        if switch_prob > random_number:
            continue
        #If no switch to this state (or any higher state):
        else:
            break #somehow set the new state to switch to - ths depend on how the probabilities actually work.
    #Initiate the switch:





