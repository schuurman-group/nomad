"""
Compute integrals over trajectories traveling on the boson model potential.
"""
import numpy as np
import src.interfaces.boson_model_diabatic as boson


# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the basis set
basis = 'gaussian'

# Determines the Hamiltonian symmetry
hamsym = 'hermitian'

def v_integral(traj1, traj2=None, S_ij=None):
    """Returns potential coupling matrix element between two
    trajectories."""
    if traj2 is None:
        return traj1.energy(traj1.state)

    if S_ij is None:
        S_ij = traj1.h_overlap(traj2)

    if traj1.state == traj2.state:
        sgn   = -1. + 2.*traj1.state
        pos1  = traj1.x()
        mom1  = traj1.p()
        pos2  = traj2.x()
        mom2  = traj2.p()
        v_int = complex(0.,0.)
        for k in range(boson.ncrd):
            a = (1. + 1.)
            b = (2. * 1. * (pos1[k] + pos2[k]) +
                 complex(0.,1.)*(mom2[k] - mom1[k]))
            v_int += (0.5 * boson.omega[k] * (2*a + b**2)/(4 * a**2) +
                      sgn * boson.C[k] * b/(2*a))
        return v_int * S_ij
    else:
        return boson.delta * S_ij


def ke_integral(traj1, traj2, S_ij=None):
    """Returns kinetic energy integral over trajectories."""
    ke_int = complex(0.,0.)
    if traj1.state == traj2.state:
        if S_ij is None:
            S_ij = traj1.h_overlap(traj2)
        for k in range(boson.ncrd):
            #ke_int -= (boson.omega[k] *
            #           traj1.particles[k].deld2x(traj2.particles[k]))
            ke_int -= traj1.particles[k].deld2x(traj2.particles[k])
        return 0.5 * ke_int * S_ij
    else:
        return ke_int


def sdot_integral(traj1, traj2, S_ij=None):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if S_ij is None:
        S_ij = traj1.h_overlap(traj2, st_orthog=True)

    sdot = (-np.dot( traj2.velocity(), traj1.deldx(traj2, S_ij) ) +
            np.dot( traj2.force()   , traj1.deldp(traj2, S_ij) ) +
            complex(0.,1.) * traj2.phase_dot() * S_ij)
    return sdot
