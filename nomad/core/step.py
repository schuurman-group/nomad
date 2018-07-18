"""
Routines for propagating a wavefunction forward by a time step.
"""
import numpy as np
import nomad.core.glbl as glbl
import nomad.core.log as log
import nomad.core.surface as evaluate
#import nomad.core.matching_pursuit as mp


def time_step(master):
    """ Determine time step based on whether in coupling regime"""
    if glbl.adapt.in_coupled_regime(master):
        return float(glbl.propagate['coupled_time_step'])

    else:
        return float(glbl.propagate['default_time_step'])


def step_wavefunction(master, dt):
    """Propagates the wave packet using a run-time selected propagator."""
    # save the wavefunction from previous step in case step rejected
    end_time      = master.time + dt
    time_step     = dt
    min_time_step = dt / 2.**5

    while not step_complete(master.time, end_time, dt):
        # save the wavefunction from previous step in case step rejected
        #try:
        #    del master0
        #except NameError:
        #    pass
        master0 = master.copy()

        # propagate each trajectory in the wavefunction
        time_step = min(time_step, end_time-master.time)

        # propagate amplitudes for 1/2 time step using x0
        master.update_amplitudes(0.5*dt)

        # the propagators update the potential energy surface as need be.
        glbl.integrator.propagate_wfn(master, time_step)

        # update the couplings for all the trajectories
        for i in range(master.n_traj()):
            glbl.interface.evaluate_coupling(master.traj[i])
 
        # propagate amplitudes for 1/2 time step using x1
        glbl.master_mat.build(master, glbl.master_int)
        master.update_matrices(glbl.master_mat)
        master.update_amplitudes(0.5*dt)

        # Renormalization
        if glbl.propagate['renorm'] == 1:
            master.renormalize()

        # check time_step is fine, energy/amplitude conserved
        accept, error_msg = check_step_wfn(master0, master, time_step)

        # if everything is ok..
        if accept:
            # update the wavefunction time
            master.time += time_step
            # spawn new basis functions if necessary
            basis_grown  = glbl.adapt.spawn(master, time_step)
            # kill the dead trajectories
            basis_pruned = master.prune()

            # if a trajectory has been added, then call update_pes
            # to get the electronic structure information at the associated
            # centroids. This is necessary in order to propagate the amplitudes
            # at the start of the next time step.
            if basis_grown and glbl.master_int.require_centroids:
                evaluate.update_pes(master)

            # update the Hamiltonian and associated matrices
            if basis_grown or basis_pruned:
                 glbl.master_mat.build(master, glbl.master_int)
                 master.update_matrices(glbl.master_mat)
                 for i in range(master.n_traj()):
                     glbl.interface.evaluate_coupling(master.traj[i])

            # re-expression of the basis using the matching pursuit
            # algorithm
            #if glbl.propagate['matching_pursuit'] == 1:
            #    mp.reexpress_basis(master)

            # update the running log
            log.print_message('t_step',[master.time, time_step, master.nalive])

        else:
            # recall -- this time trying to propagate to the failed step
            time_step *= 0.5
            log.print_message('new_step', [error_msg, time_step])

            if  time_step < min_time_step:
                log.print_message('general',['minimum time step exceeded -- STOPPING.'])
                raise ValueError('Bundle minimum step exceeded.')

            # reset the beginning of the time step and go to beginning of loop
            #del master
            master = master0.copy()

    return master


def step_trajectory(traj, init_time, dt):
    """Propagates a single trajectory.

    Used to backward/forward propagate a trajectory during spawning.
    NOTE: step_wavefunction and step_trajectory could/should probably
    be integrated somehow...
    """
    current_time = init_time
    end_time     = init_time + dt
    time_step    = dt
    min_time_step = abs(dt / 2.**5)

    while not step_complete(current_time, end_time, time_step):
        # save the wavefunction from previous step in case step rejected
        traj0 = traj.copy()

        # propagate single trajectory
        glbl.integrator.propagate_trajectory(traj, time_step)

        # update the couplings for the trajectory
        glbl.interface.evaluate_coupling(traj)

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
                log.print_message('general',
                                  ['minimum time step exceeded -- STOPPING.'])
                raise ValueError('Trajectory minimum step exceeded.')

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


def check_step_wfn(master0, master, time_step):
    """Checks if we should reject a macro step because we're in a
    coupling region."""

    # if we're in the coupled regime and using default time step, reject
    if glbl.adapt.in_coupled_regime(master) and time_step == glbl.propagate['default_time_step']:
        return False, ' require coupling time step, current step = {:8.4f}'.format(time_step)

    # ...or if there's a numerical error in the simulation:
    #  norm conservation
    dpop = abs(sum(master0.pop()) - sum(master.pop()))
    if dpop > glbl.propagate['pop_jump_toler']:
        return False, ' jump in wavefunction population, delta[pop] = {:8.4f}'.format(dpop)

    # this is largely what the above check is checking -- but is more direct. I would say
    # we should remove the above check...
    dnorm = master.norm()
    if abs(dnorm-1.) > glbl.propagate['norm_thresh']:
        return False, 'Wfn norm threshold exceeded, |norm|-1. = {:8.4f}'.format(dnorm-1.)

    #  ... or energy conservation (only need to check traj which exist in
    # master0. If spawned, will be last entry(ies) in master
    for i in range(master0.n_traj()):
        if master0.traj[i].alive:
            energy_old = (master0.traj[i].potential() +
                          master0.traj[i].kinetic())
            energy_new = (master.traj[i].potential() +
                          master.traj[i].kinetic())
            dener = abs(energy_old - energy_new)
            if dener > glbl.propagate['energy_jump_toler']:
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
    return abs(energy_old - energy_new) <= glbl.propagate['energy_jump_toler']
