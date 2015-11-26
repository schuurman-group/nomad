import sys
import variable
import basis.bundle as bundle
import basis.spawn  as spawn
__import__('propagators.'+variable['propagator']) as propagate
#
# Set the global "goal" time step. Currently this can take two values
#    - no strongly coupled trajectories / spawning:               default_time_step
#    - two or more trajectories strongly coupled and/or spawning: coupled_time_step
#    
#  coupled time step is currently 0.25 * default_time_step
#
def time_step(master):
    dt = variable.fms['default_time_step']
    master.update_bundle()
    if master.coupled_trajectories():
      dt = variable.fms['coupled_time_step'] 
    return dt
#
#
# Propagate the wave packet using a run-time selected propagator
#
def fms_step(master,init_time,final_time,dt):
    
    # save the bundle from previous step in case step rejected
    master0 = bundle.bundle()
    master0 = master    

    current_time = init_time
    end_time     = final_time
    time_step    = dt
    while current_time < end_time

        # save the bundle from previous step in case step rejected
        master0 = basis.bundle()
        master0 = master

        # propagate each trajectory in the bundle
        for i in range(master.nalive) 
            propagate(master.trajectory[i],time_step)  
 
        # update current time
        current_time = current_time + time_step

        # check time_step is fine, energy/amplitude conserved
        step_reject = check_step(master)

        # if everything is ok..
        if not step_reject:
            # remove dead trajectories
            master.prune_bundle()
            #
            spawn.spawn(master,time_step)
        # redo time step
        else:
            # recall -- this time trying to propagate
            # to the failed step
            redo_final = current_time
            redo_step  = 0.5 * time_step

            if redo_step < min_time_step:
                print("minimum step exceed, stopping...")
                sys.exit("ERROR: fms_step")

            # reset the beginning of the time step
            master = master0
           
            # recall routine
            fms_step(master,init_time,redo_final,redo_step) 

            # if this returns here, all done
            print("all done")

