"""
Sample a specific geometry or set of geometries.
"""
import numpy as np
import nomad.core.glbl as glbl
import nomad.core.trajectory as trajectory
import nomad.core.log as log


def gen_initial_coords(cnt):
    """Takes initial position and momentum from geometry specified in input"""
    coords = glbl.properties['init_coords']
    ndim   = coords.shape[-1]

    log.print_message('string',[' Initial coordinates taken from input file(s).\n'])

    coord    = coords[cnt]
    new_traj = trajectory.Trajectory(glbl.properties['n_states'], ndim,
                                      width=glbl.properties['crd_widths'],
                                      mass=glbl.properties['crd_masses'],
                                      parent=0, kecoef=glbl.modules['integrals'].kecoef) 

    # set position and momentum
    new_traj.update_x(np.array(coord[0]))
    new_traj.update_p(np.array(coord[1]))

    return new_traj

