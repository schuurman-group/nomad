import variable
import bundle
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
def fms_step(master,dt):
    print("input dt="+str(dt))
    nlive = -1
    variable.fms['current_time'] = variable.fms['current_time'] + dt
    return nlive
