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
def fms_step_bundle(master, dt):
    integrator = __import__('src.propagators.'+glbl.fms['propagator'],fromlist=['a'])
    spawning   = __import__('src.spawn.'+glbl.fms['spawning'],fromlist=['a'])

    # save the bundle from previous step in case step rejected
    end_time     = master.time + dt
    time_step    = dt
    min_time_step = dt / 2.**5

    while master.time < end_time:

        # save the bundle from previous step in case step rejected
        try:
            del master0
        except:
            pass
        master0 = bundle.copy_bundle(master)

        # propagate each trajectory in the bundle
        time_step = min(time_step, end_time-master.time)
        integrator.propagate_bundle(master,time_step)

        # check time_step is fine, energy/amplitude conserved
        accept,error_msg = check_step_bundle(master0, master, time_step)

        # if everything is ok..
        if accept:
            # update the bundle time
            master.time += time_step
            # spawn new basis functions if necessary
            basis_grown  = spawning.spawn(master,time_step)
            # kill the dead trajectories
            basis_pruned = master.prune()
            # update the bundle hamiltonian after adding/subtracting trajectories
            if basis_grown or basis_pruned:
                master.update_matrices()
            # update the running log
            fileio.print_fms_logfile('t_step',[master.time,time_step,master.nalive])

        else:
            # recall -- this time trying to propagate
            # to the failed step
            time_step  = 0.5 * time_step
            fileio.print_fms_logfile('new_step',[error_msg,time_step])

            if  time_step < min_time_step:
                fileio.print_fms_logfile('general',
                               ['minimum time step exceeded -- STOPPING.'])
                sys.exit("ERROR: fms_step_bundle")

            # reset the beginning of the time step
            del master
            master = bundle.copy_bundle(master0)
            # go back to the beginning of the while loop
            continue

    return master

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
        return False,' require coupling time step, current step = {:8.4f}'.format(time_step)
    #
    # ...or if there's a numerical error in the simulation:
    #  norm conservation
    dpop = abs(sum(master.pop()) - sum(master.pop()))
    if dpop > glbl.fms['pop_jump_toler']:
        return False,' jump in bundle population, delta[pop] = {:8.4f}'.format(dpop)
    #
    #  ... or energy conservation
    #  (only need to check traj which exist in master0. If spawned, will be
    #  last entry(ies) in master
    for i in range(master0.n_traj()):
        if not master0.traj[i].alive:
            continue
        energy_old = (master0.traj[i].potential() +
                      master0.traj[i].kinetic())
        energy_new = (master.traj[i].potential() +
                      master.traj[i].kinetic())
        dener = abs(energy_old - energy_new)
        if dener > glbl.fms['energy_jump_toler']:
            return False,' jump in trajectory energy, tid = {0:4d}, delta[ener] = {1:10.6f}'.format(i,dener)
    #
    # If we pass all the tests, return 'success'
    return True,' success'
