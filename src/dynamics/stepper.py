import sys
import ..io.glbl as glbl
import ..basis.trajectory as trajectory
import ..basis.bundle as bundle
import ..basis.spawn  as spawn
__import__('propagators.'+variable['propagator']) as propagate

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
    dt = variable.fms['default_time_step']
    master.update_bundle()
    if master.in_coupled_regime():
      dt = variable.fms['coupled_time_step'] 
    return dt
#
#
# Propagate the wave packet using a run-time selected propagator
#
def fms_step_bundle(master,init_time,dt):
    
    # save the bundle from previous step in case step rejected
    master0 = bundle.bundle()
    current_time = init_time
    end_time     = init_time + dt
    time_step    = dt

    while current_time < end_time

        # save the bundle from previous step in case step rejected
        master0 = master

        # propagate each trajectory in the bundle
        for i in range(master.nalive) 
            propagate.propagate(master.trajectory[i],time_step)  
 
        # update current time
        proposed_time = current_time + time_step

        # check time_step is fine, energy/amplitude conserved
        accept = check_step_bundle(master0, master, time_step)

        # if everything is ok..
        if accept:
            # remove dead trajectories
            master.prune_bundle()
            #
            spawn.spawn(master,time_step)
            #
            current_time = proposed_time
            # redo time step
        else:
            # recall -- this time trying to propagate
            # to the failed step
            time_step  = 0.5 * time_step

            if  time_step < min_time_step:
                print("minimum step exceed, stopping...")
                sys.exit("ERROR: fms_step")

            # reset the beginning of the time step
            master = master0
            # go back to the beginning of the while loop
            continue           

#
# Propagate a single trajectory
# NOTE: fms_step_bundle and fms_step_trajectory could/should probably
#       be integrated somehow...
#
def fms_step_trajectory(trajectory,init_time,dt):

    current_time = init_time
    end_time     = init_time + dt 
    time_step    = dt
    while current_time < end_time

        # save the bundle from previous step in case step rejected
        traj0 = trajectory

        # propagate single trajectory 
        propagate.propagate(trajectory,time_step)

        # update current time
        proposed_time = current_time + time_step

        # check time_step is fine, energy/amplitude conserved
        accept = check_step_trajectory(traj0, trajectory, time_step)

        # if everything is ok..
        if accept:
            #
            current_time = proposed_time
            # redo time step
        else:
            # recall -- this time trying to propagate
            # to the failed step
            time_step  = 0.5 * time_step

            if  time_step < min_time_step:
                print("minimum step exceed, stopping...")
                sys.exit("ERROR: fms_step")

            # reset the beginning of the time step
            trajectory = traj0
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
    if master.in_coupled_regime() and time_step == glbl.fms['time_step']:
        return False
    #
    # ...or if there's a numerical error in the simulation:
    #  norm conservation
    if abs(sum(master.pop()) - sum(master.pop())) > 0.01:
        return False
    #
    #  ... or energy conservation
    #  (only need to check traj which exist in master0. If spawned, will be
    #  last entry(ies) in master
    for i in range(master0.ntotal())
        energy_old = master0.trajectory[i].potential() +  \
                     master0.trajectory[i].kinetic()
        energy_new =  master.trajectory[i].potential() +  \
                      master.trajectory[i].kinetic()
        if abs(energy_old - energy_new) > 0.005:
            return False
    #
    # If we pass all the tests, return 'success'
    return True

#
# check if we should reject a macro step because we're in a coupling region 
#
def check_step_trajectory(traj0, traj, time_step):
    #
    #  ... or energy conservation
    #  (only need to check traj which exist in master0. If spawned, will be
    #  last entry(ies) in master
    energy_old = master0.trajectory[i].classical()
    energy_new =  master.trajectory[i].classical()
    if abs(energy_old - energy_new) > 0.005:
        return False
     #
     # If we pass all the tests, return 'success'
     return True

