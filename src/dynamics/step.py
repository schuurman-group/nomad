import sys
import numpy as np
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.bundle as bundle

#-----------------------------------------------------------------------------
#
# Public functions
#
#-----------------------------------------------------------------------------
# Set the global "goal" time step. Currently this can take two values
#    - no strongly coupled trajectories / spawning:               default_time_step
#    - two or more trajectories strongly coupled and/or spawning: coupled_time_step
#    
#  coupled time step is currently 0.25 * default_time_step
#
def time_step(master):
    dt = glbl.fms['default_time_step']
    if master.in_coupled_regime():
      dt = glbl.fms['coupled_time_step'] 
      fileio.print_fms_logfile('coupled',[dt])
    return dt
#
#
# Propagate the wave packet using a run-time selected propagator
#
def fms_step_bundle(master,init_time,dt):
    integrator   = __import__('src.propagators.'+glbl.fms['propagator'],fromlist=['a'])
    spawn_method = __import__('src.spawn.'+glbl.fms['spawning'],fromlist=['a'])

    # save the bundle from previous step in case step rejected
    current_time = init_time
    end_time     = init_time + dt
    time_step    = dt
    min_time_step = dt / 2.**5 

    while master.time < end_time:

        # save the bundle from previous step in case step rejected
        master0 = bundle.copy_bundle(master)

        # propagate each trajectory in the bundle
        for i in range(master.n_total()):
            if master.traj[i].alive: 
                integrator.propagate(master.traj[i],time_step)  

        # update current time
        proposed_time = current_time + time_step

        # check time_step is fine, energy/amplitude conserved
        accept = check_step_bundle(master0, master, time_step)

        # if everything is ok..
        if accept:
            # update the current_time
            current_time = proposed_time
            # update the bundle time
            master.time = current_time
            # spawn new basis functions if necessary
            spawn_method.spawn(master,current_time,time_step)
            # set trajectory amplitudes
            master.update_amplitudes(dt,10)
            # kill the dead trajectories
            master.prune()
        else:
            # recall -- this time trying to propagate
            # to the failed step
            time_step  = 0.5 * time_step
            fileio.print_fms_logfile('new_step',[time_step])

            if  time_step < min_time_step:
                fileio.print_fms_logfile('general',
                               ['minimum time step exceeded -- STOPPING.'])
                sys.exit("ERROR: fms_step")

            # reset the beginning of the time step
            master = bundle.copy_bundle(master0)
            # go back to the beginning of the while loop
            continue

#-----------------------------------------------------------------------------
#
# Private functions
#
#-----------------------------------------------------------------------------

#
# check if we should reject a macro step because we're in a coupling region 
#
def check_step_bundle(master0, master, time_step):
    #
    # if we're in the coupled regime and using default time step, reject
    if master.in_coupled_regime() and time_step == glbl.fms['default_time_step']:
        return False
    #
    # ...or if there's a numerical error in the simulation:
    #  norm conservation
    if abs(sum(master.pop()) - sum(master.pop())) > glbl.fms['pop_jump_toler']:
        print("pop jump")
        return False
    #
    #  ... or energy conservation
    #  (only need to check traj which exist in master0. If spawned, will be
    #  last entry(ies) in master
    for i in range(master0.n_total()):
        energy_old = master0.traj[i].potential() +  \
                     master0.traj[i].kinetic()
        energy_new =  master.traj[i].potential() +  \
                      master.traj[i].kinetic()
        if abs(energy_old - energy_new) > glbl.fms['energy_jump_toler']:
            print("de="+str(abs(energy_old - energy_new)))
            print("energy jump")
            return False
    #
    # If we pass all the tests, return 'success'
    return True

