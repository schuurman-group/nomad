"""
Routines for propagating a bundle forward by a time step.
"""
import sys
import numpy as np
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.bundle as bundle
import src.basis.trajectory as trajectory
import src.dynamics.surface as surface
import src.basis.matching_pursuit as mp


def fms_time_step(master):
    """ Determine time step based on whether in coupling regime"""
    spawning   = __import__('src.spawn.'+glbl.fms['spawning'],
                            fromlist=['a'])

    if spawning.in_coupled_regime(master):
        return float(glbl.fms['coupled_time_step'])
    else:
        # don't change back to default time step unless we're
        # back on a multiple of the default time step, otherwise
        # the log updates get out of sync. This should be fixed more
        # cleanly
        if not fileio.update_logs(master):
            return float(glbl.fms['coupled_time_step'])
        else:
            return float(glbl.fms['default_time_step'])


def fms_step_bundle(master, dt):
    """Propagates the wave packet using a run-time selected propagator."""
    integrator = __import__('src.propagators.'+glbl.fms['propagator'],
                            fromlist=['a'])
    spawning   = __import__('src.spawn.'+glbl.fms['spawning'],
                            fromlist=['a'])

    # save the bundle from previous step in case step rejected
    end_time      = master.time + dt
    time_step     = dt
    min_time_step = dt / 2.**5

    while not step_complete(master.time, end_time, dt):
        # save the bundle from previous step in case step rejected
        #try:
        #    del master0
        #except NameError:
        #    pass
        master0 = master.copy()

        # propagate each trajectory in the bundle
        time_step = min(time_step, end_time-master.time)
        # propagate amplitudes for 1/2 time step using x0
        master.update_amplitudes(0.5*dt, update_ham=False)
        # the propagators update the potential energy surface as need be.
        integrator.propagate_bundle(master, time_step)
        # propagate amplitudes for 1/2 time step using x1
        master.update_amplitudes(0.5*dt)

        # Renormalization
        if glbl.fms['renorm'] == 1:
            master.renormalize()

        # check time_step is fine, energy/amplitude conserved
        accept, error_msg = check_step_bundle(master0, master, time_step)

        # if everything is ok..
        if accept:
            # update the bundle time
            master.time += time_step
            # spawn new basis functions if necessary
            basis_grown  = spawning.spawn(master, time_step)
            # kill the dead trajectories
            basis_pruned = master.prune()

            # if a trajectory has been added, then call update_pes
            # to get the electronic structure information at the associated
            # centroids. This is necessary in order to propagate the amplitudes
            # at the start of the next time step.
            if basis_grown and master.integrals.require_centroids:
                surface.update_pes(master)

            # update the Hamiltonian and associated matrices
            if basis_grown or basis_pruned:
                master.update_matrices()

            # re-expression of the basis using the matching pursuit
            # algorithm
            if glbl.fms['matching_pursuit'] == 1:
                mp.reexpress_basis(master)

            # update the running log
            fileio.print_fms_logfile('t_step',
                                     [master.time, time_step, master.nalive])
        else:
            # recall -- this time trying to propagate to the failed step
            time_step *= 0.5
            fileio.print_fms_logfile('new_step', [error_msg, time_step])

            if  time_step < min_time_step:
                fileio.print_fms_logfile('general',
                                         ['minimum time step exceeded -- STOPPING.'])
                raise ValueError('fms_step_bundle')

            # reset the beginning of the time step and go to beginning of loop
            #del master
            master = master0.copy()

    return master


def fms_step_trajectory(traj, init_time, dt):
    """Propagates a single trajectory.

    Used to backward/forward propagate a trajectory during spawning.
    NOTE: fms_step_bundle and fms_step_trajectory could/should probably
    be integrated somehow...
    """
    integrator = __import__('src.propagators.' + glbl.fms['propagator'],
                            fromlist=['a'])

    current_time = init_time
    end_time     = init_time + dt
    time_step    = dt
    min_time_step = abs(dt / 2.**5)

    while not step_complete(current_time, end_time, time_step):
        # save the bundle from previous step in case step rejected
        traj0 = traj.copy()

        # propagate single trajectory
        integrator.propagate_trajectory(traj, time_step)

        # update current time
        proposed_time = current_time + time_step

        # check time_step is fine, energy/amplitude conserved
        accept = check_step_trajectory(traj0, traj)

        # if everything is ok..
        if accept:
            current_time = proposed_time
        else:
            # redo time step
            # recall -- this time trying to propagate
            # to the failed step
            time_step *= 0.5

            if  abs(time_step) < min_time_step:
                fileio.print_fms_logfile('general',
                                         ['minimum time step exceeded -- STOPPING.'])
                raise ValueError('fms_step_trajectory')

            # reset the beginning of the time step and go to beginning of loop
            traj = traj0.copy()


#-----------------------------------------------------------------------------
#
# Private functions
#
#-----------------------------------------------------------------------------
def step_complete(current_time, final_time, dt):
    """checks if the propagation time has reached the end of the time step.
       Need to allow for negative time steps."""
    if dt > 0:
        return current_time >= final_time
    else:
        return current_time <= final_time


def check_step_bundle(master0, master, time_step):
    """Checks if we should reject a macro step because we're in a
    coupling region."""
    spawning   = __import__('src.spawn.'+glbl.fms['spawning'],
                            fromlist=['a'])

    # if we're in the coupled regime and using default time step, reject
    if spawning.in_coupled_regime(master) and time_step == glbl.fms['default_time_step']:
        return False, ' require coupling time step, current step = {:8.4f}'.format(time_step)
    # ...or if there's a numerical error in the simulation:
    #  norm conservation
    dpop = abs(sum(master0.pop()) - sum(master.pop()))
    if dpop > glbl.fms['pop_jump_toler']:
        return False, ' jump in bundle population, delta[pop] = {:8.4f}'.format(dpop)
    #  ... or energy conservation (only need to check traj which exist in
    # master0. If spawned, will be last entry(ies) in master
    for i in range(master0.n_traj()):
        if master0.traj[i].alive:
            energy_old = (master0.traj[i].potential() +
                          master0.traj[i].kinetic())
            energy_new = (master.traj[i].potential() +
                          master.traj[i].kinetic())
            dener = abs(energy_old - energy_new)
            if dener > glbl.fms['energy_jump_toler']:
                return False, ' jump in trajectory energy, label = {:4d}, delta[ener] = {:10.6f}'.format(i, dener)
    return True, ' success'


def check_step_trajectory(traj0, traj):
    """Checks if we should reject a macro step because we're in a
    coupling region.

    ... or energy conservation
    Only need to check traj which exist in master0. If spawned, will be
    last entry(ies) in master.
    """
    energy_old = traj0.classical()
    energy_new = traj.classical()

    # If we pass all the tests, return 'success'
    return abs(energy_old - energy_new) <= glbl.fms['energy_jump_toler']
