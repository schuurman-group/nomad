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
import nomad.core.glbl as glbl
import nomad.core.log as log
import nomad.core.timings as timings
import nomad.adapt.utilities as utilities


@timings.timed
def adapt(wfn0, wfn, dt):
    """Spawns a basis function if the minimum overlap drops below a given
    threshold."""
    basis_grown  = False
    current_time = wfn.time
    for i in range(wfn.n_traj()):
        if not wfn.traj[i].alive:
            continue

        parent = wfn.traj[i]
        for st in range(glbl.properties['n_states']):
            # don't check overlap with basis functions on same state
            if st == parent.state:
                continue

            s_array = [abs(glbl.modules['integrals'].nuc_overlap(parent, wfn.traj[j]))
                       if wfn.traj[j].state == st
                       and wfn.traj[j].alive else 0.
                       for j in range(wfn.n_traj())]
            if max(s_array, key=abs) < glbl.properties['continuous_min_overlap']:
                child           = parent.copy()
                child.amplitude = 0j
                child.state     = st
                child.parent    = parent.label

                success = utilities.adjust_momentum(child, parent.classical()
                                                    parent.derivative(parent.state,
                                                                   child.state))
                sij = glbl.modules['integrals'].nuc_overlap(parent, child)

                # try to set up the child
                if not success:
                    pass
                    #log.print_message('spawn_bad_step',
                    #                         ['cannot adjust kinetic energy of child'])
                elif abs(sij) < glbl.properties['spawn_olap_thresh']:
                    pass
                    #log.print_message('spawn_bad_step',
                    #                         ['child-parent overlap too small'])
                else:
                    child_created = True
                    spawn_time = current_time
                    parent.last_spawn[child.state] = spawn_time
                    child.last_spawn[parent.state] = spawn_time

                    bundle_overlap = utilities.overlap_with_bundle(child, wfn)
                    if not bundle_overlap:
                        basis_grown = True
                        wfn.add_trajectory(child)
                        log.print_message('spawn_success',
                                          [current_time, parent.label, st])
                        utilities.write_spawn_log(current_time, current_time,
                                                  current_time, parent,
                                                  wfn.traj[-1])
                    else:
                        err_msg = ('Traj ' + str(parent.label) + ' from state ' +
                                   str(parent.state) + ' to state ' + str(st) +
                                   ': ' + 'overlap with bundle too large,' +
                                   ' s_max=' + str(glbl.properties['sij_thresh']))
                        log.print_message('spawn_bad_step', [err_msg])
    return basis_grown


def in_coupled_regime(bundle):
    """Checks if we are in spawning regime.
    
    Since we are always spawning, this function always returns False for
    continuous spawning -- no need to change timestep to spawn.
    """
    return False
