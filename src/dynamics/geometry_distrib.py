"""
Sample a distribution of geometries.
"""
import src.fmsio.glbl as glbl
import src.dynamics.utilities as utils
import src.basis.trajectory as trajectory


def sample_distribution(master):
    """Takes initial position and momentum from geometry.dat file."""
    amps, geoms = utils.load_geometry()
    ngeoms = len(amps)
    natms  = int(len(geoms)/ngeoms)

    for i in range(ngeoms):
        amp  = amps[i]
        geom = []

        # load particles in geometry array
        for j in range(natms):
            geom.append(geoms[i*natms + j])

        # add a single trajectory specified by geometry.dat
        master.add_trajectory(trajectory.Trajectory(glbl.fms['n_states'],
                                                    particles=geom,
                                                    parent=0))
        # ...with unit amplitude
        master.traj[i].amplitude = amp

    # state of trajectory not set, return false
    return False
