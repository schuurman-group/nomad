"""
Compute first-order saddle point integrals over trajectories traveling on
adiabataic potentials
"""
import numpy as np


# Let propagator know if we need data at centroids to propagate
require_centroids = True

# Determines the basis set
basis = 'gaussian'


def v_integral(traj1, traj2=None, centroid=None, S_ij=None):
    """Returns potential coupling matrix element between two trajectories."""
    # if we are passed a single trajectory, this is a diagonal
    # matrix element -- simply return potential energy of trajectory
    if traj2 is None:
        return traj1.energy(traj1.state)

    # off-diagonal matrix element, between trajectories on the same
    # state [this also requires the centroid be present
    elif traj1.state == traj2.state:
        return centroid.energy(traj1.state) * traj1.overlap(traj2)

    # [necessarily] off-diagonal matrix element between trajectories
    # on different electronic states
    elif not traj1.state != traj2.state:
        fij = centroid.derivative(traj2.state)
        return np.vdot( fij, traj1.deldx_m(traj2) )
    else:
        print('ERROR in v_integral -- argument disagreement')
        return 0.


def ke_integral(traj1, traj2, S_ij=None):
    """Returns kinetic energy integral over trajectories."""
    ke = complex(0.,0.)
    if traj1.state == traj2.state:
        for i in range(traj1.n_particle):
            ke -= (traj1.particles[i].deld2x(traj2.particles[i]) /
                   (2.*traj1.particles[i].mass))
        return ke * traj1.overlap(traj2)
    else:
        return ke


def sdot_integral(traj1, traj2, S_ij=None):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    sdot = (-np.dot( traj2.velocity(), traj1.deldx(traj2) ) +
            np.dot( traj2.force()   , traj1.deldp(traj2) ) +
            complex(0.,1.) * traj2.phase_dot() * traj1.overlap(traj2))
    return sdot
