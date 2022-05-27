"""
Routines for propagating a wavefunction forward by a time step.
"""
import numpy as np
import scipy.linalg as sp_linalg
import nomad.core.glbl as glbl
import nomad.core.log as log
import nomad.core.checkpoint as checkpoint
import nomad.core.surface as evaluate
import nomad.core.matrices as matrices
import nomad.adapt.utilities as utilities

def time_step(time):
    """ Determine time step based on whether in coupling regime"""
    if glbl.modules['adapt'].in_coupled_regime(glbl.modules['wfn']):
        return float(glbl.properties['coupled_time_step'])

    else:
        # when coming out of coupled regime, may need one more 'coupled'
        # time step to ensure that what follows are multiples of default 
        # time step
        dstep        = 1.e-6
        default_step = float(glbl.properties['default_time_step'])
        t_mod_step   = float(time % default_step)

        # overstepped by epsilon, scale step back a bit
        if t_mod_step < dstep:
            return float(default_step - t_mod_step)
        # understepped by epsilon, scale steup up a bit
        elif t_mod_step > (1. - dstep)*default_step:
            return float(default_step + default_step - t_mod_step)
        # else, just return what we need to get 'back on track'
        else:
            return float(time % glbl.properties['default_time_step'])


def step_wavefunction(dt):
    """Propagates the wave packet using a run-time selected propagator."""
    # save the wavefunction from previous step in case step rejected
    end_time      = min(glbl.modules['wfn'].time + dt,
                        glbl.properties['simulation_time'])
    time_step     = min(dt, glbl.properties['simulation_time'] -
                            glbl.modules['wfn'].time)
    min_time_step = dt / 2.**5

    while not step_complete(glbl.modules['wfn'].time, end_time, time_step):
        # save the wavefunction from previous step in case step rejected
        wfn_start = glbl.modules['wfn'].copy()

        # propagate each trajectory in the wavefunction
        time_step = min(time_step, end_time-glbl.modules['wfn'].time)

        # the propagators update the potential energy surface as need be.
        glbl.modules['propagator'].propagate_wfn(glbl.modules['wfn'], time_step)

        # update the couplings for all the trajectories
        for i in range(glbl.modules['wfn'].n_traj()):
            glbl.modules['interface'].evaluate_coupling(glbl.modules['wfn'].traj[i])

        # build new matrices at time t' = t + dt
        glbl.modules['matrices'].build(glbl.modules['wfn'], 
                                       glbl.modules['wfn'], 
                                       glbl.modules['integrals'],
                                       hermitian=True)
        glbl.modules['wfn'].update_matrices(glbl.modules['matrices'])

        # update the amplitudes
        new_amps = step_amplitudes(wfn_start, glbl.modules['wfn'], time_step)
        glbl.modules['wfn'].set_amplitudes(new_amps)

        # Renormalization
        if glbl.properties['renorm'] == 1:
            glbl.modules['wfn'].renormalize()

        # check time_step is fine, energy/amplitude conserved
        accept, error_msg = check_step_wfn(wfn_start, glbl.modules['wfn'], time_step)

        # if everything is ok..
        if accept:
            # update the wavefunction time
            update_time(glbl.modules['wfn'].time + time_step)
            # spawn new basis functions if necessary
            basis_grown  = glbl.modules['adapt'].adapt(wfn_start, glbl.modules['wfn'], time_step)
            # kill the dead trajectories
            basis_pruned = utilities.prune(glbl.modules['wfn'])

            # if a trajectory has been added, then call update_pes
            # to get the electronic structure information at the associated
            # centroids. This is necessary in order to propagate the amplitudes
            # at the start of the next time step.
            if basis_grown and glbl.modules['integrals'].require_centroids:
                evaluate.update_pes(glbl.modules['wfn'])

            # update the Hamiltonian and associated matrices
            if basis_grown or basis_pruned:
                 glbl.modules['matrices'].build( glbl.modules['wfn'],
                                                 glbl.modules['wfn'],
                                                 glbl.modules['integrals'],
                                                 hermitian=True)
                 glbl.modules['wfn'].update_matrices(glbl.modules['matrices'])
                 for i in range(glbl.modules['wfn'].n_traj()):
                     glbl.modules['interface'].evaluate_coupling(glbl.modules['wfn'].traj[i])

            # udpate populations
            pops = glbl.modules['integrals'].pops(glbl.modules['wfn'])
            glbl.modules['wfn'].update_pop(pops)

            # update the running log
            log.print_message('t_step',[glbl.modules['wfn'].time, time_step, glbl.modules['wfn'].nalive])
            #del wfn_start

            # save the simulation here, so we capture all the
            # intermediate time steps as well 
            if glbl.mpi['rank'] == 0:
                # update the checkpoint, if necessary
                checkpoint.archive_simulation(glbl.modules['wfn'],
                                              glbl.modules['integrals'])


        else:
            # recall -- this time trying to propagate to the failed step
            time_step *= 0.5
            log.print_message('new_step', [error_msg, time_step])

            if  time_step < min_time_step:
                log.print_message('general',['minimum time step exceeded -- STOPPING.'])
                raise ValueError('Bundle minimum step exceeded.')

            # reset the beginning of the time step and go to beginning of loop
            #del master
            glbl.modules['wfn'] = wfn_start.copy()


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
        glbl.modules['propagator'].propagate_trajectory(traj, time_step)

        # update the couplings for the trajectory
        glbl.modules['interface'].evaluate_coupling(traj)

        # update current time
        proposed_time = current_time + time_step

        # check time_step is fine, energy/amplitude conserved
        accept = check_step_trajectory(traj0, traj)

        # if everything is ok..
        if accept:
            current_time = proposed_time
            traj.time    = current_time
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

def step_amplitudes(wfn0, wfn1, dt):
    """step amplitudes using Pade approxmiant"""

    I = complex(0., 1.)
    ct = wfn0.amplitudes()
 
    if glbl.properties['amp_prop'] == 'pade':
        st = wfn0.matrices.matrix['s']
        ht = wfn0.matrices.matrix['h']

        sdt = wfn1.matrices.matrix['s']
        hdt = wfn1.matrices.matrix['h']

        tdt_matrices = matrices.Matrices()
        tdt_matrices.build(wfn0, wfn1, glbl.modules['integrals'],
                                                  hermitian=False)
        stdt = tdt_matrices.matrix['s']
        htdt = tdt_matrices.matrix['h']

        dtt_matrices = matrices.Matrices()
        dtt_matrices.build(wfn1, wfn0, glbl.modules['integrals'],
                                                  hermitian=False)
        sdtt = dtt_matrices.matrix['s']
        hdtt = dtt_matrices.matrix['h']

        a = np.concatenate((stdt + 0.5*I*dt*htdt,
                            sdt  + 0.5*I*dt*hdt))
   
        b = np.concatenate((np.dot(st   - 0.5*I*dt*ht, ct),
                            np.dot(sdtt - 0.5*I*dt*hdtt, ct)))

        cdt, res, rank, sing = np.linalg.lstsq(a, b, rcond=None)
        new_norm = np.conj(cdt).T @ sdt @ cdt
        cdt *= (1./np.sqrt(new_norm)) 

    elif glbl.properties['amp_prop'] == 'expm':
        #Updates the amplitudes of the trajectory in the wfn.
        #Solves d/dt C = -i H C via the computation of
        #exp(-i H(t) dt) C(t).

        # update amplitudes in two steps: 0.5*dt from previous
        # step and 0.5*dt from final step
        B0 = -1j * wfn0.matrices.matrix['heff'] * 0.5 * dt
        u0 = sp_linalg.expm(B0)

        B1 = -1j * wfn1.matrices.matrix['heff'] * 0.5 * dt
        u1 = sp_linalg.expm(B1)

        cdt = np.dot( u1, u0 @ ct)
    else:
        sys.exit('ERROR: amp_prop = '+str(lbl.properties['amp_prop']))

    return cdt 

#-----------------------------------------------------------------------------
#
# Private functions
#
#-----------------------------------------------------------------------------
def update_time(new_time):
    """ update the wfn time, as well as time in all trajectory
        basis functions """

    glbl.modules['wfn'].time = new_time
    for i in range(glbl.modules['wfn'].n_traj()):
        if glbl.modules['wfn'].traj[i].alive:
            glbl.modules['wfn'].traj[i].time = new_time
 
    return

def step_complete(current_time, final_time, dt):
    """checks if the propagation time has reached the end of the time step.
       Need to allow for negative time steps."""
    if dt > 0:
        return current_time >= final_time
    else:
        return current_time <= final_time


def check_step_wfn(wfn_start, wfn, time_step):
    """Checks if we should reject a macro step because we're in a
    coupling region."""

    # if we're in the coupled regime and using default time step, reject
    if glbl.modules['adapt'].in_coupled_regime(wfn) and time_step == glbl.properties['default_time_step']:
        return False, ' require coupling time step, current step = {:8.4f}'.format(time_step)

    # ...or if there's a numerical error in the simulation:
    #  norm conservation
    #dpop = abs(sum(wfn_start.pop()) - sum(wfn.pop()))
    #if dpop > glbl.properties['pop_jump_toler']:
    #    return False, ' jump in wavefunction population, delta[pop] = {:8.4f}'.format(dpop)

    # this is largely what the above check is checking -- but is more direct. I would say
    # we should remove the above check...
    dnorm = wfn.norm()
    if abs(dnorm-1.) > glbl.properties['norm_thresh']:
        return False, 'Wfn norm threshold exceeded, |norm|-1. = {:8.4f}'.format(dnorm-1.)

    #  ... or energy conservation (only need to check traj which exist in
    # master0. If spawned, will be last entry(ies) in master
    for i in range(wfn_start.n_traj()):
        if wfn_start.traj[i].alive:
            energy_old = (wfn_start.traj[i].potential() +
                          wfn_start.traj[i].kinetic())
            energy_new = (wfn.traj[i].potential() +
                          wfn.traj[i].kinetic())
            dener = abs(energy_old - energy_new)
            if dener > glbl.properties['energy_jump_toler']:
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
    return abs(energy_old - energy_new) <= glbl.properties['energy_jump_toler']
