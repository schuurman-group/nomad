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

random.seed(464)

c_cache = {}
data_cache = {}
n_states = glbl.properties['n_states']
#initial state probability (a) matrix:
def init_c():
    init_state = glbl.properties['init_state']
    init_c = np.zeros((n_states), dtype = 'complex')
    init_c[init_state] = 1
    c_cache['init'] = init_c

init_c()
switch_times = 0
current_c = c_cache['init']

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
    local_dt = dt/100
    current_st = traj.state
        
    #Check that we only have one trajectory:
    if wfn.n_traj() > 1:
        sys.exit('fssh algorithm must only have one trajectory')


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

    
    #this loop is called for each small time step 
    while local_time < current_time + dt:
        current_st = traj.state
        
        random_number = random.random() 
        prev_prob = 0
        for st in range(glbl.properties['n_states']):
        #can only switch between states
            if st == current_st:
                continue
            state_pop = a(current_st,current_st) 
            state_flux = (2*np.imag(np.conj(a(current_st, st)) * diabat_pot(current_st, st))) - ( 2* np.real(np.conj(a(current_st, st)) * nac(current_st, st)))
            switch_prob = state_flux * local_dt / state_pop
            if switch_prob < 0:     
                switch_prob = 0

            #add to the previous probability
            switch_prob = prev_prob + switch_prob


            #a test:
            def a_dot(k,j):
                adq = 0
                for l in range(n_states):
                    term1 = a(l,j) * np.complex(diabat_pot(k,l), -nac(k,l))
                    term2 = a(k,l) * np.complex(diabat_pot(l,j), -nac(l,j))
                    both_terms = np.complex(0, (term2-term1))
                    adq+= both_terms
                return adq
            #-----------------------------------------------------------------------------
            #plt.plot(traj.x()[0], switch_prob, marker='o', markerSize = 3, color = 'brown')              
            #plt.plot(traj.x()[0], current_st, marker='o', markerSize = 3, color = 'blue')              
            #plt.plot(traj.x()[0], diabat_pot(0,1), marker='o', markerSize = 3, color = 'red') 
            #plt.plot(traj.x()[0], diabat_pot(1,0), marker='o', markerSize = 3, color = 'orange') 
            #plt.plot(traj.x()[0], diabat_pot(0,0), marker='o', markerSize = 3, color = 'brown') 
            #plt.plot(traj.x()[0], diabat_pot(1,1), marker='o', markerSize = 3, color = 'green') 
            #plt.plot(local_time,a_dot(0,0, [current_a[0,0]]), marker='x', markerSize = 3, color = 'red') 
            #plt.plot(local_time,np.imag(a_dot(0,0, [current_a[0,0]])), marker='x', markerSize = 3, color = 'green') 
            #plt.plot(local_time,-state_flux + a_dot(0,0, [current_a[0,0]]), marker='x', markerSize = 3, color = 'purple') 
            #plt.plot(traj.x()[0], a_dot(0,1, [current_a[0,1]]), marker='x', markerSize = 3, color = 'orange')
            #plt.plot(local_time, a_dot(1,1), marker='x', markerSize = 3, color = 'blue')
            #plt.plot(local_time, a_dot(0,0), marker='x', markerSize = 3, color = 'green')
            #plt.plot(local_time, nac(0,1), marker='x', markerSize = 3, color = 'red') 
            #plt.plot(local_time, nac(1,0), marker='x', markerSize = 3, color = 'orange') 
            #plt.plot(traj.x()[0], nac(0,0), marker='x', markerSize = 3, color = 'brown') 
            #plt.plot(traj.x()[0], nac(1,1), marker='x', markerSize = 3, color = 'green') 
            #plt.plot(local_time, state_pop, marker='o', markerSize = 3, color = 'green')
            #plt.plot(local_time, state_flux, marker='x', markerSize = 3, color = 'black')
            #plt.plot(traj.x()[0], a(0,0), marker='x', markerSize = 3, color = 'green')
            #plt.plot(traj.x()[0], a(1,1), marker='x', markerSize = 3, color = 'blue')
            #plt.plot(traj.x()[0], a(0,0) + a(1,1), marker='x', markerSize = 3, color = 'pink')
            #plt.plot(traj.x()[0], 1, marker='o', markerSize = 3, color = 'pink')
            ##-----------------------------------------------------------------------------------


            #Check probability against random number, see if there's a switch
            if prev_prob < random_number <= switch_prob: 
                total_E = traj.classical()
                current_T = traj.kinetic() 
                log.print_message('general',['Attempting surface hop to state ' + str(st) + ' at time ' + str(local_time)]) 
                #change the state:
                traj.state = st
                new_V = traj.potential()
                new_T = total_E - new_V
                #is this hop classically possible? If not, go back to previous state:
                if new_T < 0:
                    log.print_message('general',['Surface hop failed (not enough T), remaining on state ' + str(current_st)])
                    traj.state = current_st
                else:
                    log.print_message('general',['Surface hop successful, switching from state ' + str(current_st) + ' to state ' + str(st)])
                    #calculate and set the new momentum:
                    scale_factor = np.sqrt(new_T/current_T)
                    current_p = traj.p()
                    new_p = scale_factor * current_p
                    traj.update_p(new_p)
                    #to keep track, for now:
                    global switch_times
                    switch_times += 1

                continue      

        if local_time >= glbl.properties['simulation_time'] +glbl.properties['default_time_step'] - local_dt:
            #plt.show()
            if traj.x()[0]>0:
                if switch_times <= 1:
                    print(current_st)
                else:
                    if current_st == 0:
                        print(2)
                    if current_st ==1:
                        print(3)
            else:
                print(4)

        propagate_c()
        #update the local time:
        local_time = local_time + local_dt


def in_coupled_regime(wfn):
    return False

