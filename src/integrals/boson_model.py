"""
Compute integrals over trajectories traveling on the boson model potential.
"""
import numpy as np
import src.interfaces.boson_model_diabatic as boson
import src.dynamics.timings as timings


# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the basis set
basis = 'gaussian'

# Determines the Hamiltonian symmetry
hamsym = 'hermitian'

def v_integral(traj1, traj2=None, S_ij=None):
    """Returns potential coupling matrix element between two
    trajectories.

    This is the analytical solution for Gaussian functions at positions
    pos1, pos2, momenta mom1, mom2 and with widths a1, a2. The product
    of Gaussians is written such that
    g1 g2 = N^2 exp(-ax^2 - bx - c),
    where N is a constant prefactor and the variables a, b and c depend
    on positions, momenta and widths.

    If the overlap of two Gaussian functions is S_12, it can be shown
    that the first and second moments in x are
    int( dx x g1 g2 )  = (-b / 2a) S_12
    int( dx x^2 g1 g2 ) = ((2a + b^2) / 4a^2) S_12.
    """
    if traj2 is None:
        sgn  = -1. + 2.*traj1.state
        pos1 = traj1.x()
        a1 = traj1.widths()
        a = 2. * a1
        b = -4. * a1*pos1
        v_int = sum(boson.omega * (2.*a + b**2)/(8. * a**2) -
                    sgn * boson.C * b/(2.*a))
        return v_int

    if S_ij is None:
        S_ij = traj1.h_overlap(traj2)

    if traj1.state == traj2.state:
        sgn  = -1. + 2.*traj1.state
        pos1 = traj1.x()
        mom1 = traj1.p()
        a1 = traj1.widths()
        pos2 = traj2.x()
        mom2 = traj2.p()
        a2 = traj2.widths()
        a = a1 + a2
        b = -2. * (a1*pos1 + a2*pos2) + 1j * (mom1 - mom2)
        v_int = sum(boson.omega * (2.*a + b**2)/(8. * a**2) -
                    sgn * boson.C * b/(2.*a))
        return v_int * S_ij
    else:
        return boson.delta * S_ij


def ke_integral(traj1, traj2, S_ij=None):
    """Returns kinetic energy integral over trajectories."""
    if traj1.state == traj2.state:
        if S_ij is None:
            S_ij = traj1.h_overlap(traj2)
        ke_int = complex(0.,0.)
        for k in range(boson.ncrd):
            ke_int -= (0.5 * boson.omega[k] *
                       traj1.particles[k].deld2x(traj2.particles[k]))
        return ke_int * S_ij
    else:
        return complex(0.,0.)


def sdot_integral(traj1, traj2, S_ij=None):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if S_ij is None:
        S_ij = traj1.h_overlap(traj2, st_orthog=True)

    sdot = (-np.dot( traj2.velocity(), traj1.deldx(traj2, S_ij) ) +
            np.dot( traj2.force(), traj1.deldp(traj2, S_ij) ) +
            1j * traj2.phase_dot()*S_ij)
    return sdot
