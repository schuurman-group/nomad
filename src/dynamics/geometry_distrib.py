"""
Sample a distribution of geometries.
"""
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.trajectory as trajectory

def sample_distribution(master):
    """Takes initial position and momentum from geometry.dat file."""
    (ncrd, crd_dim, amps, label, 
                 geoms, moms, width, mass, states) = fileio.read_geometry()
    ngeoms  = len(amps)
    ndim    = int(len(geoms) / ngeoms)
    state_set = True

    print("states="+str(states))
    for i in range(ngeoms):
       
        itraj = trajectory.Trajectory(glbl.fms['n_states'],
                                     ndim,
                                     width=width[i*ndim:(i+1)*ndim],
                                     mass=mass[i*ndim:(i+1)*ndim],
                                     crd_dim=crd_dim,
                                     parent=0)

        # set position and momentum
        itraj.update_x(geoms[i*ndim:(i+1)*ndim])
        itraj.update_p(moms[i*ndim:(i+1)*ndim])

        # set the initial amplitude
        itraj.update_amplitude(complex(amps[i]))

        # set the initial state
        if states[i] != -1:
            itraj.state = int(states[i])
        else:
            state_set = False

        # add a single trajectory specified by geometry.dat
        master.add_trajectory(itraj)

    # if all states aren't set, return False
    return state_set
