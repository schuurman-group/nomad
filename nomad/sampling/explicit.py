"""
Sample a specific geometry or set of geometries.
"""
import numpy as np
import nomad.parse.glbl as glbl
import nomad.basis.trajectory as trajectory


def set_initial_coords(master):
    """Takes initial position and momentum from geometry specified in input"""

    ngeoms  = len(glbl.nuclear_basis['geometries'])
    ndim    = len(glbl.nuclear_basis['geometries'][0])

    for i in range(ngeoms):

        itraj = trajectory.Trajectory(glbl.propagate['n_states'], ndim,
                                      width  = glbl.nuclear_basis['widths'],
                                      mass   = glbl.nuclear_basis['masses'],
                                      parent = 0,
                                      kecoef = glbl.kecoef)

        # set position and momentum
        itraj.update_x(np.array(glbl.nuclear_basis['geometries'][i]))
        itraj.update_p(np.array(glbl.nuclear_basis['momenta'][i]))

        # add a single trajectory specified by geometry.dat
        master.add_trajectory(itraj)
