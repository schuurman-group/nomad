#
# this spawning procedure propagates to the point of maximum coupling, 
# creates a new function, that back propagates to the current time
#
import numpy as np
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.trajectory as trajectory
import src.basis.bundle as bundle
import src.spawn.utilities as utilities 

coup_hist = []

# This is the top-level routine for spawning
#
# Schematic
#
#   start, ti
#     |
#    \/        spawn_forward 
# parent(s,ti) -------------> parent(s,ts)
#                                  |
#              spawn_backward     \/
# child(s',ti) <------------- child(s',ts)
#
# 1. The spawn routine is called with parent_i at time t0.
# 2. If parent_i is coupled to another 
#

def spawn(master,dt):
    global coup_hist

    current_time = master.time

    # we want to have know the history of the coupling for each trajectory
    # in order to assess spawning criteria -- make sure coup_hist has a slot
    # for every trajectory
    if len(coup_hist) < master.n_total():
        n_add = master.n_total() - len(coup_hist)
        for i in range(n_add):
            coup_hist.append(np.zeros((master.nstates,3),dtype=np.float))

    for i in range(master.n_total()):

        # only live trajectories can spawn
        if not master.traj[i].alive:
            continue

        for st in range(master.nstates):

            # can only spawn to different electornic states
            if master.traj[i].state == st:
                continue

            # check overlap with other trajectories
            max_sij = 0.
            for j in range(master.n_total()):
                if master.traj[j].alive and master.traj[j].state == st:
                    sij = abs(master.traj[i].overlap(master.traj[j]))
                    if sij > max_sij:
                        max_sij = sij 
                        if max_sij > glbl.fms['sij_thresh']:
                            break
            if max_sij > glbl.fms['sij_thresh']:
                fileio.print_fms_logfile('general',
                        ['trajectory overlap with bundle too large, cannot spawn'])
                continue

            # compute magnitude of coupling to state j
            coup = master.traj[i].coup_dot_vel(st)
            coup_hist[i][st,:] = np.roll(coup_hist[i][st,:],1)
            coup_hist[i][st,0] = coup

            # if we satisfy spawning conditions, begin spawn process                    
            if spawn_trajectory(master.traj[i], st, coup_hist[i][st,:], current_time):
                parent       = trajectory.copy_traj(master.traj[i])
                child        = trajectory.copy_traj(parent)
                child.amplitude = complex(0.,0.)
                child.state     = st
                child.parent    = parent.tid
                # the child and parent share an id before the child is added to the bundle
                # this helps the interface use electronic structure information that is reasonable,
                # but not sure it matters for anything else
                child.tid       = parent.tid

                # propagate the parent forward in time until coupling maximized
                (spawn_time, exit_time, success) = spawn_forward(parent, child, current_time, dt)
                master.traj[i].last_spawn[st] = spawn_time
                master.traj[i].exit_time[st]  = exit_time

                if success:
                    # at this point, child is at the spawn point. Propagate backwards in time
                    # until we reach the current time
                    spawn_backward(child, exit_time, current_time, -dt)
                    bundle_overlap = overlap_with_bundle(child,master)
                    if not bundle_overlap:
                        master.add_trajectory(child)
                    else:
                        fileio.print_fms_logfile('spawn_bad_step',['overlap with bundle too large'])

                utilities.write_spawn_log(current_time, spawn_time, exit_time, master.traj[i], master.traj[-1])

    # update matrices
    master.update_matrices()

#
# propagate the parent forward (into the future) until the coupling decreases
#
def spawn_forward(parent, child, initial_time, dt):

    parent_state = parent.state
    child_state  = child.state
    current_time = initial_time
    spawn_time   = initial_time
    exit_time    = initial_time

    coup_hist = np.zeros(3,dtype=np.float)
    fileio.print_fms_logfile('spawn_start',[parent.tid, parent_state, child_state])

    while True:

        child_create = False
        coup_hist = np.roll(coup_hist,1)
        coup_hist[0] = parent.coup_dot_vel(child.state)

        # if the coupling has already peaked, either we exit with a successful
        # spawn from previous step, or we exit with a fail
        if np.all(coup_hist[0] < coup_hist[1:]):
            if child_created:
                fileio.print_fms_logfile('spawn_success',[current_time])
            else:
                fileio.print_fms_logfile('spawn_failure',[current_time])
                parent.last_spawn[child_state] = current_time
                child.last_spawn[parent_state] = current_time
            exit_time = current_time
            parent.exit_time[child_state] = exit_time
            child.exit_time[parent_state] = exit_time
            break
        # coupling still increasing
        else:
            child = trajectory.copy_traj(parent)
            child.state = child_state
            adjust_success = utilities.adjust_child(parent, child, parent.derivative(child_state))
            sij = parent.overlap(child)

            # try to set up the child
            if not adjust_success:
                fileio.print_fms_logfile('spawn_bad_step',
                                        ['could not adjust child momentum'])
                sp_str = 'no'
            elif abs(sij) < glbl.fms['spawn_olap_thresh']:
                fileio.print_fms_logfile('spawn_bad_step',
                                        ['child-parent overlap too small'])
                sp_str = 'no'
            else:
                child_created = True
                spawn_time = current_time
                parent.last_spawn[child_state] = spawn_time
                child.last_spawn[parent_state] = spawn_time
                coup_max = coup_hist[0]
                sp_str   = 'yes'
            fileio.print_fms_logfile('spawn_step',[current_time,coup_max,abs(sij),sp_str])

            utilities.fms_step_trajectory(parent, current_time, dt)
            current_time = current_time + dt

    return spawn_time, exit_time, child_created

#
# propagate the child backwards in time until we reach the current time
#
def spawn_backward(child, current_time, end_time, dt):
    nstep = int( np.absolute( (current_time-end_time) / dt) )

    back_time = current_time
    while back_time > end_time:
        utilities.fms_step_trajectory(child,back_time,dt)
        back_time = back_time + dt
        fileio.print_fms_logfile('spawn_back',[back_time])

#
# check if we satisfy all spawning criteria
#
def spawn_trajectory(traj, spawn_state, coup_hist, current_time):

    # Return False if:
    # if insufficient population on trajectory to spawn
    if abs(traj.amplitude) < glbl.fms['spawn_pop_thresh']:
        return False

    # we have already spawned to this state 
    if current_time <= traj.last_spawn[spawn_state]:
        return False

    # there is insufficient coupling
    if traj.coup_dot_vel(spawn_state) < glbl.fms['spawn_coup_thresh']:
        return False

    # if coupling is decreasing
    if coup_hist[0] < coup_hist[1]:
        return False

    return True


