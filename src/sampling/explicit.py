"""
Sample a distribution of geometries.
"""
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.trajectory as trajectory

def set_initial_coords(masses, widths, geoms, momenta, master):
    """Takes initial position and momentum from geometry specified in input"""

    ngeoms  = len(geoms)
    ndim    = int( len(geoms[0]) )

    for i in range(ngeoms):
       
        itraj = trajectory.Trajectory(glbl.propagate['n_states'], ndim, 
                                      width=widths, mass=masses, 
                                      parent=0)

        # set position and momentum
        itraj.update_x(geoms[i,:])
        itraj.update_p(momenta[i,:])

        # add a single trajectory specified by geometry.dat
        master.add_trajectory(itraj)

    return
