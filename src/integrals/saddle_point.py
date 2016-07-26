"""
Compute saddle-point integrals over trajectories traveling on adiabataic
potentials

This currently uses first-order saddle point.
"""
import sys
import numpy as np
import src.fmsio.glbl as glbl
import src.interfaces.vcham.hampar as ham

# Let propagator know if we need data at centroids to propagate
require_centroids = True

# Determines the basis set
basis = 'gaussian'

def v_integral(traj1, traj2=None, centroid=None, S_ij=None):
    """Returns potential coupling matrix element between two trajectories."""
    # if we are passed a single trajectory, this is a diagonal
    # matrix element -- simply return potential energy of trajectory
    if traj2 is None:
        # Adiabatic energy
        v = traj1.energy(traj1.state)
        # DBOC
        if glbl.fms['coupling_order'] == 3:
            v += traj1.scalar_coup(traj1.state)
        return v
    # off-diagonal matrix element, between trajectories on the same
    # state [this also requires the centroid be present
    elif traj1.state == traj2.state:
        # Adiabatic energy
        v = centroid.energy(traj1.state) * traj1.overlap(traj2)
        # DBOC
        if glbl.fms['coupling_order'] == 3:
            v += centroid.scalar_coup(traj1.state) * traj1.overlap(traj2)
        return v
    # [necessarily] off-diagonal matrix element between trajectories
    # on different electronic states
    elif traj1.state != traj2.state:
        # Derivative coupling
        fij = centroid.derivative(traj2.state)
        v = np.vdot( fij, traj1.deldx_m(traj2) )
        # Scalar coupling
        if glbl.fms['coupling_order'] > 1:
            v += traj1.scalar_coup(traj2.state) * traj1.overlap(traj2)
        return v

    else:
        print('ERROR in v_integral -- argument disagreement')
        return 0.


def ke_integral(traj1, traj2, S_ij=None):
    """Returns kinetic energy integral over trajectories."""
    ke = complex(0.,0.)
    if traj1.state == traj2.state:
        if glbl.fms['interface'] == 'vibronic':
            for i in range(traj1.n_particle):
                ke -= traj1.particles[i].deld2x(traj2.particles[i]) \
                      * ham.freq[i]/2.0
        else:
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
