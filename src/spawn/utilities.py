"""
General routines for all spawning algorithms.
"""
import numpy as np
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.trajectory as trajectory


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
    min_time_step = dt / 2.**5

    while current_time < end_time:
        # save the bundle from previous step in case step rejected
        traj0 = trajectory.copy_traj(traj)

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
            time_step  = 0.5 * time_step

            if  time_step < min_time_step:
                fileio.print_fms_logfile('general',
                                         ['minimum time step exceeded -- STOPPING.'])
                raise ValueError('fms_step_trajectory')

            # reset the beginning of the time step
            traj = trajectory.copy_traj(traj0)
            # go back to the beginning of the while loop
            continue


# check if we should reject a macro step because we're in a coupling region
#
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
    return not abs(energy_old - energy_new) > glbl.fms['energy_jump_toler']


def adjust_child(parent, child, scale_dir):
    """Adjust the child momentum so that child and parent have the same
    energy

    1. First try to scale the momentum along the NAD vector direction
    2. If that fails, scale the momentum uniformly
    """
    e_parent = parent.classical()
    e_child  = child.classical()

    # determine the magnitude of the KE correction
    ke_goal  = e_parent - child.potential()
    ke_child = child.kinetic()
    if ke_goal < 0:
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

    p_child = child.p()
    # scale the momentum along the scale_vec direction
    p_para = np.dot(p_child, scale_vec) * scale_vec
    p_perp = p_child - p_para

    # the kinetic energy is given by:
    # KE = (P . P) * kecoeff
    #    = (p_para + p_perp).(p_para + p_perp) * kecoeff
    #    = (p_para.p_para)*kecoeff + (p_para.p_perp)*2*kecoeff + (p_perp.p_perp)*kecoeff
    #    = KE_para_para + KE_para_perp + KE_perp_perp
    kecoeff=child.interface.kecoeff
    ke_para_para = np.dot( p_para, p_para * kecoeff )
    ke_para_perp = np.dot( p_para, p_perp * kecoeff )
    ke_perp_perp = np.dot( p_perp, p_perp * kecoeff )

    # scale p_para by x so that KE == ke_goal
    # (ke_para_para)*x^2 + (ke_para_perp)*x + (ke_perp_perp - ke_goal) = 0
    # solve quadratic equation
    a = ke_para_para
    b = ke_para_perp
    c = ke_perp_perp - ke_goal

    discrim = b**2 - 4.*a*c
    if discrim < 0:
        return False

    if abs(a) > glbl.fpzero:
        x = (-b + np.sqrt(discrim)) / (2.*a)
    elif abs(b) > glbl.fpzero:
        x = -c / b
    else:
        x = 0.

    p_new = x*p_para + p_perp

    child.update_p(p_new)
    return True


def overlap_with_bundle(traj, bundle):
    """Checks if trajectory has significant overlap with any trajectories
    already in the bundle."""
    t_overlap_bundle = False

    for i in range(bundle.n_traj()):
        if bundle.traj[i].alive:

            sij = traj.overlap(bundle.traj[i], st_orthog=True)
            if abs(sij) > glbl.fms['sij_thresh']:
                t_overlap_bundle = True
                break

    return t_overlap_bundle


def write_spawn_log(entry_time, spawn_time, exit_time, parent, child):
    """Packages data to print to the spawn log."""
    # add a line entry to the spawn log
    data = [entry_time, spawn_time, exit_time]
    data.extend([parent.tid, parent.state, child.tid, child.state])
    data.extend([parent.kinetic(), child.kinetic(), parent.potential(),
                 child.potential()])
    data.extend([parent.classical(), child.classical()])
    fileio.print_bund_row(2, data)
