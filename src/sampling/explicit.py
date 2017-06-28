"""
Sample a distribution of geometries.
"""
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.trajectory as trajectory

def set_initial_coords(widths, geoms, momenta, master):
    """Takes initial position and momentum from geometry specified in input"""

    ngeoms  = len(geoms)
    crd_dim = int(len(geoms[0][0]))
    ndim    = int(len(geoms[0]) * crd_dim )

    x_vec = np.array()
    p_vec = np.array()
    w_vec = np.array([widths[i] for j in range(crd_dim) for i in range(len(widths))])  

    for i in range(ngeoms):
       
        itraj = trajectory.Trajectory(glbl.propagate['n_states'],
                                      ndim,
                                      width=widths,
                                      crd_dim=crd_dim,
                                      parent=0)

        # set position and momentum
        itraj.update_x(x_vec[i,:])
        itraj.update_p(p_vec[i,:])

        # add a single trajectory specified by geometry.dat
        master.add_trajectory(itraj)

    return
