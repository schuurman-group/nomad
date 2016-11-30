"""
Compute integrals over trajectories traveling on vibronic potentials
"""
import math
import numpy as np
nuc_ints  = __import__('src.integrals.nuclear_'+glbl.fms['test_function'],
                     fromlist=['NA'])


# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the Hamiltonian symmetry
hermitian = True

# returns total overlap of trajectory basis function
def s_integral(traj1, traj2, Snuc=None):
    """ Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""
    if traj1.state != traj2.state:
        return complex(0.,0.)
    else:
        if Snuc is None:
            return nuc_ints.overlap(traj1,traj2)
        else:
            return Snuc

def v_integral(traj1, traj2, centroid=None, Snuc=None):
    """Returns potential coupling matrix element between two trajectories.

    This will depend on how the operator is stored, and
    what type of coordinates are employed, etc.
    """
    pass


def prim_v_integral(n, a1, x1, p1, a2, x2, p2):
    """Returns the matrix element <cmplx_gaus(q,p)| q^N |cmplx_gaus(q,p)>

    Takes the arguments as particles.
    """
    n_2 = np.floor(0.5 * n)
    a   = p1.width + p2.width
    b   = np.fromiter((complex(2.*(a1*x1 + a2*x2),-(p1-p2)) 
                                         for i in range(1)),dtype=complex)

    # generally these should be 1D harmonic oscillators. If
    # multi-dimensional, the final result is a direct product of
    # each dimension
    v_total  = complex(1.,0.)
    for d in range(1):
        v = complex(0.,0.)
        for i in range(n_2):
            v = (v + a**(i-N) * b**(N-2*i) /
                 (np.math.factorial(i) * np.math.factorial(N-2*i)))
        v_total = v_total * v

    # refer to appendix for derivation of these relations
    return v_total * np.math.factorial(N) / 2.**N


def ke_integral(traj1, traj2, Snuc=None):
    """Returns kinetic energy integral over trajectories."""
    if traj1.state != traj2.state:
        return complex(0.,0.)
    else:
        if Snuc is None:
            Snuc = nuc_ints.overlap(traj1, traj2)
        ke = nuc_ints.deld2x(traj1,traj2, S=Snuc)
        return -sum( ke * interface.kecoeff)

def sdot_integral(traj1, traj2, Snuc=None):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if traj1.state != traj2.state:
        return complex(0.,0.)
    else:
        if Snuc is None:
            Snuc = nuc_ints.overlap(traj1, traj2)
        sdot = (-np.dot( traj2.velocity(), nuc_ints.deldx(traj1,traj2,S=Snuc) )+
                 np.dot( traj2.force(),    nuc_ints.deldp(traj1,traj2,S=Snuc) )+
                 1.j * traj2.phase_dot() * Snuc)
        return sdot
