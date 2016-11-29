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
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.trajectory as trajectory
import src.spawn.utilities as utilities
nuc_ints = __import__('src.integrals.nuclear_'+glbl.fms['test_function'],
                     fromlist=['NA'])

coup_hist = []


def spawn(master, dt):
    """Propagates to the point of maximum coupling, spawns a new
    basis function, then propagates the function to the current time."""
    global coup_hist

    basis_grown  = False
    current_time = master.time

    # we want to know the history of the coupling for each trajectory
    # in order to assess spawning criteria -- make sure coup_hist has a slot
    # for every trajectory
    if len(coup_hist) < master.n_traj():
        n_add = master.n_traj() - len(coup_hist)
        for i in range(n_add):
            coup_hist.append(np.zeros((master.nstates, 3)))

    for i in range(master.n_traj()):
        # only live trajectories can spawn
        if not master.traj[i].alive:
            continue

        for st in range(master.nstates):
            # can only spawn to different electornic states
            if master.traj[i].state == st:
                continue

            # check overlap with other trajectories
            max_sij = 0.
            for j in range(master.n_traj()):
                if master.traj[j].alive and master.traj[j].state == st:
                    sij = abs(nuc_ints(master.traj[i],master.traj[j]))
                    if sij > max_sij:
                        max_sij = sij
                        if max_sij > glbl.fms['sij_thresh']:
                            break
            if max_sij > glbl.fms['sij_thresh']:
                fileio.print_fms_logfile('general',
                                         ['trajectory overlap with bundle too large, cannot spawn'])
                continue

            # compute magnitude of coupling to state j
            coup = master.traj[i].eff_coup(st)
            coup_hist[i][st,:] = np.roll(coup_hist[i][st,:],1)
            coup_hist[i][st,0] = coup

            # if we satisfy spawning conditions, begin spawn process
            if spawn_trajectory(master.traj[i], st, coup_hist[i][st,:],
                                current_time):
                parent = trajectory.copy_traj(master.traj[i])
                child  = trajectory.copy_traj(parent)
                child.amplitude = complex(0.,0.)
                child.state     = st
                child.parent    = parent.tid
                # the child and parent share an id before the child is added
                # to the bundle this helps the interface use electronic
                # structure information that is reasonable, but not sure it
                # matters for anything else
                child.tid       = parent.tid

                # propagate the parent forward in time until coupling maximized
                spawn_time, exit_time, success = spawn_forward(parent, child,
                                                               current_time, dt)
                master.traj[i].last_spawn[st] = spawn_time
                master.traj[i].exit_time[st]  = exit_time

                if success:
                    # at this point, child is at the spawn point. Propagate
                    # backwards in time until we reach the current time
                    spawn_backward(child, spawn_time, current_time, -dt)
                    bundle_overlap = utilities.overlap_with_bundle(child, master)
                    if not bundle_overlap:
                        basis_grown = True
                        master.add_trajectory(child)
                        if master.ints.require_centroids:
                            master.update_centroids()
                    else:
                        fileio.print_fms_logfile('spawn_bad_step',
                                                 ['overlap with bundle too large'])

                utilities.write_spawn_log(current_time, spawn_time, exit_time,
                                          master.traj[i], master.traj[-1])

    # let caller known if the basis has been changed
    return basis_grown


def spawn_forward(parent, child, initial_time, dt):
    """Propagates the parent forward (into the future) until the coupling
    decreases."""
    parent_state = parent.state
    child_state  = child.state
    current_time = initial_time
    spawn_time   = initial_time
    exit_time    = initial_time
    child_created = False

    coup = np.zeros(3)
    fileio.print_fms_logfile('spawn_start',
                             [parent.tid, parent_state, child_state])

    while True:
        coup = np.roll(coup,1)
        coup[0] = abs(parent.eff_coup(child_state))
        child_attempt       = trajectory.copy_traj(parent)
        child_attempt.state = child_state
        adjust_success      = utilities.adjust_child(parent, child_attempt,
                                                     parent.derivative(child_state))
        sij = abs(nuc_ints(parent,child_attemp))

        # if the coupling has already peaked, either we exit with a successful
        # spawn from previous step, or we exit with a fail
        if np.all(coup[0] < coup[1:]):
            sp_str = 'no [decreasing coupling]'
            fileio.print_fms_logfile('spawn_step',
                                     [current_time, coup[0], sij, sp_str])
            if child_created:
                fileio.print_fms_logfile('spawn_success', [spawn_time])
            else:
                fileio.print_fms_logfile('spawn_failure', [current_time])
                parent.last_spawn[child_state] = current_time
                child.last_spawn[parent_state] = current_time
            exit_time                     = current_time
            parent.exit_time[child_state] = exit_time
            child.exit_time[parent_state] = exit_time
            break

        # coupling still increasing
        else:
            # try to set up the child
            if not adjust_success:
                sp_str = 'no [momentum adjust fail]'
            elif sij < glbl.fms['spawn_olap_thresh']:
                sp_str = 'no [overlap too small]'
            elif not np.all(coup[0] > coup[1:]):
                sp_str = 'no [decreasing coupling]'
            else:
                child = trajectory.copy_traj(child_attempt)
                child_created                  = True
                spawn_time                     = current_time
                parent.last_spawn[child_state] = spawn_time
                child.last_spawn[parent_state] = spawn_time
                sp_str                         = 'yes'

            fileio.print_fms_logfile('spawn_step',
                                     [current_time, coup[0], sij, sp_str])

            utilities.fms_step_trajectory(parent, current_time, dt)
            current_time = current_time + dt

    return spawn_time, exit_time, child_created


def spawn_backward(child, current_time, end_time, dt):
    """Propagates the child backwards in time until the current time
    is reached."""
    nstep = int(round( np.abs((current_time-end_time) / dt) ))

    back_time = current_time
    for i in range(nstep):
        utilities.fms_step_trajectory(child, back_time, dt)
        back_time = back_time + dt
        fileio.print_fms_logfile('spawn_back', [back_time])


def spawn_trajectory(traj, spawn_state, coup_h, current_time):
    """Checks if we satisfy all spawning criteria."""

    # Return False if:

    # if insufficient population on trajectory to spawn
    if abs(traj.amplitude) < glbl.fms['spawn_pop_thresh']:
        return False

    # we have already spawned to this state
    if current_time <= traj.last_spawn[spawn_state]:
        return False

    # there is insufficient coupling
    if abs(traj.eff_coup(spawn_state)) < glbl.fms['spawn_coup_thresh']:
        return False

    # if coupling is decreasing
    if abs(coup_h[0]) < abs(coup_h[1]):
        return False

    return True
