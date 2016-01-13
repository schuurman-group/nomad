import sys
import copy
import numpy as np
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.trajectory as trajectory
import src.basis.bundle as bundle
integrator = __import__('src.propagators.'+glbl.fms['propagator'],fromlist=['a'])

coup_hist = []

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
      fileio.print_fms_logfile('coup_time_step',[dt])
    return dt
#
#
# Propagate the wave packet using a run-time selected propagator
#
def fms_step_bundle(master,init_time,dt):

    # save the bundle from previous step in case step rejected
    current_time = init_time
    end_time     = init_time + dt
    time_step    = dt

    while master.time < end_time:

        # save the bundle from previous step in case step rejected
        master0 = bundle.copy_bundle(master)

        # propagate each trajectory in the bundle
        for i in range(master.n_total()):
            if master.traj[i].alive: 
                integrator.propagate(master.traj[i],time_step)  

        # update current time
        proposed_time = current_time + time_step

        # check time_step is fine, energy/amplitude conserved
        accept = check_step_bundle(master0, master, time_step)

        # if everything is ok..
        if accept:
            # update the current_time
            current_time = proposed_time
            # update the bundle time
            master.time = current_time
            # spawn new basis functions if necessary
            spawn(master,current_time,time_step)
            # set trajectory amplitudes
            master.update_amplitudes(dt,10)
            # kill the dead trajectories
            master.prune()
        else:
            # recall -- this time trying to propagate
            # to the failed step
            time_step  = 0.5 * time_step

            if  time_step < min_time_step:
                fileio.print_fms_logfile('general',
                               ['minimum time step exceeded -- STOPPING.'])
                sys.exit("ERROR: fms_step")

            # reset the beginning of the time step
            master = bundle.copy_bundle(master0)
            # go back to the beginning of the while loop
            continue

#-----------------------------------------------------------------------------
#
# Private functions
#
#-----------------------------------------------------------------------------

#
# Propagate a single trajectory -- used to backward/forward propagate a trajectory
#                                  during spawning
# NOTE: fms_step_bundle and fms_step_trajectory could/should probably
#       be integrated somehow...
#
def fms_step_trajectory(traj, init_time, dt):

    current_time = init_time
    end_time     = init_time + dt 
    time_step    = dt

    while current_time < end_time:

        # save the bundle from previous step in case step rejected
        traj0 = trajectory.copy_traj(traj)

        # propagate single trajectory 
        integrator.propagate(traj,time_step)

        # update current time
        proposed_time = current_time + time_step

        # check time_step is fine, energy/amplitude conserved
        accept = check_step_trajectory(traj0, traj, time_step)

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
                fileio.print_fms_logfile('general',
                               ['minimum time step exceeded -- STOPPING.'])
                sys.exit("ERROR: fms_step")

            # reset the beginning of the time step
            traj = trajectory.copy_traj(traj0)
            # go back to the beginning of the while loop
            continue

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
    if abs(sum(master.pop()) - sum(master.pop())) > glbl.fms['pop_jump_toler']:
        return False
    #
    #  ... or energy conservation
    #  (only need to check traj which exist in master0. If spawned, will be
    #  last entry(ies) in master
    for i in range(master0.n_total()):
        energy_old = master0.traj[i].potential() +  \
                     master0.traj[i].kinetic()
        energy_new =  master.traj[i].potential() +  \
                      master.traj[i].kinetic()
        if abs(energy_old - energy_new) > glbl.fms['energy_jump_toler']:
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
    energy_old = traj0.classical()
    energy_new = traj.classical()
    if abs(energy_old - energy_new) > glbl.fms['energy_jump_toler']:
        return False

    # If we pass all the tests, return 'success'
    return True

#
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
 
def spawn(master,current_time,dt):
    global coup_hist

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
                if master.traj[j].state == st:
                    if abs(master.traj[i].overlap(master.traj[j])) > max_sij:
                        fileio.print_fms_logfile('general',
                                     ['trajectory overlap with bundle too large, cannot spawn'])
                        max_sij = abs(master.traj[i].overlap(master.traj[j]))
                        if max_sij > glbl.sij_thresh:
                            break
            if max_sij > glbl.sij_thresh:
                continue

            # compute magnitude of coupling to state j
            coup = master.traj[i].coup_dot_vel(st)
            coup_hist[i][st,:] = np.roll(coup_hist[i][st,:],1)
            coup_hist[i][st,0] = coup

            # if we satisfy spawning conditions, begin spawn process                    
            if spawn_trajectory(master.traj[i], st, coup_hist[i][st,:], current_time):
                parent       = trajectory.copy_traj(master.traj[i])
                child        = trajectory.copy_traj(parent)
                child.state  = st
                child.parent = parent.tid
                # the child and parent share an id before the child is added to the bundle
                # this helps the interface use electronic structure information that is reasonable,
                # but not sure it matters for anything else
                child.tid    = parent.tid
 
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
                        master.addtrajectory(child)
                    else:
                        fileio.print_fms_logfile('spawn_bad_step',['overlap with bundle too large'])

                write_spawn_log(current_time, spawn_time, exit_time, master.traj[i], master.traj[-1])

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
    print_fms_logfile('spawn_start',[parent.tid, parent_state, child_state])

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
            child = trajectory.copy_tray(parent)
            child.state = child_state
            adjust_success = adjust_child(parent, child, parent.derivative(child_state))
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

            fms_step_trajectory(parent, current_time, dt)
            current_time = current_time + dt

    return spawn_time, exit_time, child_created

#
# propagate the child backwards in time until we reach the current time
#
def spawn_backward(child, current_time, end_time, dt):
    nstep = int( np.absolute( (current_time-end_time) / dt) )

    back_time = current_time
    while back_time > end_time:
        fms_step_trajectory(child,back_time,dt)
        back_time = back_time + dt

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

#
# adjust the momentum of the child to so that energy of parent and child
# have the same energy
# 
#   1. First try to scale the momentum along the NAD vector direction
#   2. If that fails, scale the momentum uniformly
#
def adjust_child(parent, child, scale_dir):
    e_parent = parent.classical()
    e_child  = child.classical()    

    # determine the magnitude of the KE correction
    ke_goal  = e_parent - child.potential()
    ke_child = child.kinetic()
    if ke_goal < 0:
        fileio.print_fms_logfile('spawn_bad_step',
                                ['cannot adjust kinetic energy of child'])
        return False

    # try to scale the momentum along the scale direction
    scale_vec  = scale_dir
    scale_norm = np.linalg.norm(scale_vec)
    if scale_norm > glbl.fpzero:
        scale_vec = scale_vec / scale_norm
    else:
        # if scale_dir is zero, scale momentum uniformly
        scale_vec = np.ones(len(scale_dir))
        scale_vec = scale_vec / np.linalg.norm(scale_vec)

    p_child = child.momentum()

    # scale the momentum along the scale_vec direction
    p_para = np.dot(p_child,scale_vec) * scale_vec
    p_perp = p_child - p_parallel   

    # the kinetic energy is given by:
    # KE = (P . P) / 2m
    #    = (p_para + p_perp).(p_para + p_perp) / 2m
    #    = (p_para.p_para)/2m + (p_para.p_perp)/m + (p_perp.p_perp)/2m 
    #    = KE_para_para + KE_para_perp + KE_perp_perp
    mass = p_child.masses
    ke_para_para = np.dot(p_para,p_perp) / (2*mass)
    ke_perp_perp = np.dot(p_perp,p_perp) / (2*mass)
    ke_para_perp = np.dot(p_para,p_perp) / mass

    # scale p_para by x so that KE == ke_goal
    # (ke_para_para)*x^2 + (ke_para_perp)*x + (ke_perp_perp - ke_goal) = 0
    # solve quadratic equation
    a = ke_para_para
    b = ke_para_perp
    c = ke_perp_perp

    discrim = b**2 - 4.*a*c 
    if discrim < 0:
        fileio.print_fms_logfile('spawn_bad_step',
                                ['cannot scale momentum of child'])
        return False

    if np.absolute(a) > glbl.fpzero:
        x = (-b + math.sqrt(discrim)) / (2.*a)
    elif np.absolute(b) > glbl.fpzero:
        x = -c / b
    else:
        x = 0.

    p_new = x*p_para + p_perp

    child.update_p(p_new)
    return True

#
# check to see if trajectory has significant overlap with any of the
# trajectories already in the bundle
#
def overlap_with_bundle(trajectory,bundle):

    t_overlap_bundle = False

    for i in range(bundle.n_total()):
        if bundel.traj[i].alive:

            sij = trajectory.overlap(bundle.traj[i])
            if abs(sij) > glbl.sij_thresh:
                t_overlap_bundle = True
                break

    return t_overlap_bundle

#
# package up the data to print to the spawn log
#
def write_spawn_log(entry_time, spawn_time, exit_time, parent, child):
 
    # add a line entry to the spawn log 
    data = [entry_time, spawn_time, exit_time]
    data.extend([parent.tid, parent.state, child.tid, child.state])
    data.extend([parent.kinetic(), parent.potential(), child.kinetic(), child.potential()])
    fileio.print_bund_row(4,data) 
    return

