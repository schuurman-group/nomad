"""
Routines for the continuous spawning algorithm.

Schematic:

  start, ti
    |           if Min(parent.overlap(traj(st',ti))) < Omin
parent(r,p,st,ti) ------------------------------------> child(r,p',st',ti)

1. if the minimum overlap between the "parent" trajectory on state st with
   trajectories on state st' drops below a user-defined threshold, spawn a
   new function on st' at the same position, with a scaled momentum to
   enforce constraint that classical energies be equal.
"""
import numpy as np
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.dynamics.timings as timings
import src.basis.trajectory as trajectory
import src.spawn.utilities as utilities


@timings.timed
def spawn(master, dt):
    """Spawns a basis function if the minimum overlap drops below a given
    threshold."""
    basis_grown  = False
    current_time = master.time
    for i in range(master.n_traj()):
        if not master.traj[i].alive:
            continue

        parent = master.traj[i]
        for st in range(master.nstates):
            # don't check overlap with basis functions on same state
            if st == parent.state:
                continue

            s_array = [abs(parent.overlap(master.traj[j], st_orthog=False))
                       if master.traj[j].state == st
                       and master.traj[j].alive else 0.
                       for j in range(master.n_traj())]
            if max(s_array) < glbl.fms['continuous_min_overlap']:
                child           = trajectory.copy_traj(parent)
                child.amplitude = complex(0.,0.)
                child.state     = st
                child.parent    = parent.tid

                success = utilities.adjust_child(parent, child,
                                                 parent.derivative(st))
                sij = parent.overlap(child)

                # try to set up the child
                if not success:
                    fileio.print_fms_logfile('spawn_bad_step',
                                             ['cannot adjust kinetic energy of child'])
                elif abs(sij) < glbl.fms['spawn_olap_thresh']:
                    fileio.print_fms_logfile('spawn_bad_step',
                                             ['child-parent overlap too small'])
                else:
                    child_created = True
                    spawn_time = current_time
                    parent.last_spawn[child.state] = spawn_time
                    child.last_spawn[parent.state] = spawn_time

                    bundle_overlap = utilities.overlap_with_bundle(child, master)
                    if not bundle_overlap:
                        basis_grown = True
                        master.add_trajectory(child)
                        fileio.print_fms_logfile('spawn_success',
                                                 [current_time, parent.tid, st])
                        utilities.write_spawn_log(current_time, current_time,
                                                  current_time, parent,
                                                  master.traj[-1])
                    else:
                        err_msg = ('Traj ' + str(parent.tid) + ' from state ' +
                                   str(parent.state) + ' to state ' + str(st) +
                                   ': ' + 'overlap with bundle too large,' +
                                   ' s_max=' + str(glbl.fms['sij_thresh']))
                        fileio.print_fms_logfile('spawn_bad_step', [err_msg])
    return basis_grown
