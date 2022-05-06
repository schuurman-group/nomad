"""
Routines for the continuous spawning algorithm.

Schematic:

  start, ti
    |
   \/        spawn_forward
parent(s,ti) -------------> parent(s,ts)
                                 |
             spawn_backward     \/
child(s',ti) <------------- child(s',ts)

1. The spawn routine is called with parent_i at time t0.
2. If parent_i is coupled to another.
"""
import numpy as np
import nomad.core.glbl as glbl
import nomad.core.checkpoint as checkpoint
import nomad.core.log as log
import nomad.core.step as step
import nomad.core.surface as evaluate
import nomad.adapt.utilities as utils


coup_hist = []


def adapt(wfn0, wfn, dt):
    """Propagates to the point of maximum coupling, spawns a new
    basis function, then propagates the function to the current time."""
    global coup_hist

    basis_grown   = False
    current_time  = wfn.time
    # list of added trajectories

    # we want to know the history of the coupling for each trajectory
    # in order to assess spawning criteria -- make sure coup_hist has a slot
    # for every trajectory
    if len(coup_hist) < wfn.n_traj():
        n_add = wfn.n_traj() - len(coup_hist)
        for i in range(n_add):
            coup_hist.append(np.zeros((glbl.properties['n_states'], 3)))

    # iterate over all trajectories in wfn
    for i in range(wfn.n_traj()):
        # only live trajectories can spawn
        if not wfn.traj[i].alive:
            continue

        for st in range(glbl.properties['n_states']):
            # can only spawn to different electronic states
            if wfn.traj[i].state == st:
                continue

            # compute magnitude of coupling to state j
            coup = abs(wfn.traj[i].coupling(wfn.traj[i].state, st))
            coup_hist[i][st,:] = np.roll(coup_hist[i][st,:],1)
            coup_hist[i][st,0] = coup

            # if we satisfy spawning conditions, begin spawn process
            if spawn_trajectory(wfn, i, st, coup_hist[i][st,:],
                                current_time):

                # we're going to mess with this trajectory -- mess with a copy
                parent = wfn.traj[i].copy()

                # propagate the parent forward in time until coupling maximized
                [success, child, parent_spawn, spawn_time,
                 exit_time] = spawn_forward(parent, st, current_time, dt)

                # set the spawn attempt in wfn, even if spawn failed (avoid repeated fails)
                wfn.traj[i].last_spawn[st] = spawn_time
                wfn.traj[i].exit_time[st]  = exit_time

                if success:
                    # at this point, child is at the spawn point. Propagate
                    # backwards in time until we reach the current time
                    # need electronic structure at current geometry -- on correct state
                    evaluate.update_pes_traj(child)
                    spawn_backward(child, spawn_time, current_time, -dt)
                    wfn_overlap = utils.overlap_with_wfn(child, wfn)
                    if not wfn_overlap:
                        basis_grown = True
                        wfn.add_trajectory(child)
                        if glbl.mpi['rank'] == 0:
                            checkpoint.update_adapt(current_time, wfn.traj[i], wfn.traj[-1])
                    else:
                        log.print_message('spawn_bad_step',
                                                 ['overlap with wfn too large'])

    # after spawning, let all processes re-synchronize
    if glbl.mpi['parallel']:
        glbl.mpi['comm'].barrier()

    # let caller known if the basis has been changed
    return basis_grown


def spawn_forward(parent, child_state, initial_time, dt):
    """Propagates the parent forward (into the future) until the coupling
    decreases."""
    parent_state    = parent.state
    current_time    = initial_time
    spawn_time      = initial_time
    exit_time       = initial_time
    child_created   = False
    parent_at_spawn = None
    child_at_spawn  = None

    coup = np.zeros(3)
    log.print_message('spawn_start',
                             [parent.label, parent_state, child_state])

    while True:
        coup                = np.roll(coup,1)
        coup[0]             = abs(parent.coupling(parent_state, child_state))
        child_attempt       = parent.copy()
        child_attempt.state = child_state
        adjust_success      = utils.adjust_child(parent, child_attempt,
                                                 parent.derivative(parent_state,
                                                                   child_state))
        sij = abs(glbl.modules['integrals'].nuc_overlap(parent, child_attempt))

        # if the coupling has already peaked, either we exit with a successful
        # spawn from previous step, or we exit with a fail
        if np.all(coup[0] < coup[1:]):
            sp_str = 'no [decreasing coupling]'
            log.print_message('spawn_step',
                                     [current_time, coup[0], sij, sp_str])

            if child_created:
                log.print_message('spawn_success', [spawn_time])
                child_at_spawn.exit_time[parent_state] = current_time
            else:
                log.print_message('spawn_failure', [current_time])

            # exit, we're done trying to spawn
            exit_time = current_time
            break

        # coupling still increasing
        else:
            # try to set up the child
            if not adjust_success:
                sp_str = 'no [momentum adjust fail]'
            elif sij < glbl.properties['spawn_olap_thresh']:
                sp_str = 'no [overlap too small]'
            elif not np.all(coup[0] > coup[1:]):
                sp_str = 'no [decreasing coupling]'
            else:
                spawn_time                              = current_time
                child_created                           = True
                parent_at_spawn                         = parent.copy()
                child_at_spawn                          = child_attempt.copy()
                child_at_spawn.last_spawn[parent_state] = spawn_time
                child_at_spawn.amplitude                = 0j
                child_at_spawn.parent                   = parent.label
                child_at_spawn.label                    = parent.label
                sp_str                                  = 'yes'

            log.print_message('spawn_step',
                                     [current_time, coup[0], sij, sp_str])

            step.step_trajectory(parent, current_time, dt)
            current_time = current_time + dt

    return child_created, child_at_spawn, parent_at_spawn, spawn_time, exit_time


def spawn_backward(child, current_time, end_time, dt):
    """Propagates the child backwards in time until the current time
    is reached."""
    nstep = int(round( np.abs((current_time-end_time) / dt) ))

    back_time = current_time
    for i in range(nstep):
        step.step_trajectory(child, back_time, dt)
        back_time = back_time + dt
        log.print_message('spawn_back', [back_time])

def spawn_trajectory(wfn, traj_index, spawn_state, coup_h, current_time):
    """Checks if we satisfy all spawning criteria."""

    traj = wfn.traj[traj_index]

    # Return False if:
    # if insufficient population on trajectory to spawn
    if abs(traj.amplitude) < glbl.properties['spawn_pop_thresh']:
        return False

    # we have already spawned to this state
    if current_time <= traj.exit_time[spawn_state]:
        return False

    # there is insufficient coupling
    if abs(traj.coupling(traj.state, spawn_state)) < glbl.properties['spawn_coup_thresh']:
        return False

    # if coupling is decreasing
    if abs(coup_h[0]) < abs(coup_h[1]):
        return False

    # if we already have sufficient overlap with a function on the
    # spawn_state
    if utils.max_nuc_overlap(wfn, traj_index,
                             overlap_state=spawn_state) > glbl.properties['sij_thresh']:
        return False

    return True


def in_coupled_regime(wfn):
    """Checks if we are in spawning regime."""
    for i in range(wfn.n_traj()):
        for st in range(glbl.properties['n_states']):
            if st != wfn.traj[i].state:
                if abs(wfn.traj[i].coupling(wfn.traj[i].state, st)) > glbl.properties['spawn_coup_thresh']:
                    return True

    return False
