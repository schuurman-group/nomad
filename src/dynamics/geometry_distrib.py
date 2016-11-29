"""
Sample a distribution of geometries.
"""
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.trajectory as trajectory

def sample_distribution(master):
    """Takes initial position and momentum from geometry.dat file."""
    crd_dim, amps, lbls, geoms, moms, width, mass = fileio.read_geometry()
    ngeoms  = len(amps)
    ndim    = int(len(geoms) / ngeoms)

    for i in range(ngeoms):
        
        # add a single trajectory specified by geometry.dat
        master.add_trajectory(trajectory.Trajectory(
                                     glbl.fms['n_states'],
                                     ndim,
                                     widths=width[i*ndim:(i+1)*ndim],
                                     masses=mass[i*ndim:(i+1)*ndim],
                                      labels=lbls[i*ndim:(i+1)*ndim],
                                     crd_dim=crd_dim,
                                     parent=0))

        # set position and momentum
        master.traj[i].update_x(geoms[i*ndim:(i+1)*ndim])
        master.traj[i].update_p(moms[i*ndim:(i+1)*ndim])    

        # and initial amplitude
        master.traj[i].amplitude = amps[i]

    # state of trajectory not set, return false
    return False
