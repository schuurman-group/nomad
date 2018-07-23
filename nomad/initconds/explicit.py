"""
Sample a specific geometry or set of geometries.
"""
import numpy as np
import nomad.core.glbl as glbl
import nomad.core.trajectory as trajectory


def set_initial_coords(master):
    """Takes initial position and momentum from geometry specified in input"""

    ngeoms  = len(glbl.properties['geometries'])
    ndim    = len(glbl.properties['geometries'][0])

    for i in range(ngeoms):

        itraj = trajectory.Trajectory(glbl.properties['n_states'], ndim,
                                      width=glbl.properties['widths'],
                                      mass=glbl.properties['masses'],
                                      parent=0, kecoef=glbl.kecoef)

        # set position and momentum
        itraj.update_x(np.array(glbl.properties['geometries'][i]))
        itraj.update_p(np.array(glbl.properties['momenta'][i]))

        # add a single trajectory specified by geometry.dat
        master.add_trajectory(itraj)
