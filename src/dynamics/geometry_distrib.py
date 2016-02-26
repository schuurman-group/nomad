#
#  Take initial position and momentum from geometry.dat file
#
def sample_distribution(master):
    geom = load_geometry()
    amp  = complex(1.,0.)
    # add a single trajectory specified by geometry.dat
    master.add_trajectory(trajectory.trajectory(
                          glbl.fms['n_states'],
                          particles=geom,
                          parent=0))
    # ...with unit amplitude
    master.traj[i].amplitude = amp

    # state of trajectory not set, return false
    return False

