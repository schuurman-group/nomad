"""
Compute integrals over trajectories traveling on vibronic potentials
"""
import math
import numpy as np
import src.integrals.nuclear_gaussian as nuclear

# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the Hamiltonian symmetry
hermitian = True

# Returns functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'

# returns the overlap between two trajectories (differs from s_integral in that
# the bra and ket functions for the s_integral may be different
# (i.e. pseudospectral/collocation methods). 
def traj_overlap(traj1, traj2, nuc_only=False, Snuc=None):
    """ Returns < Psi | Psi' >, the overlap integral of two trajectories"""
    return s_integral(traj1, traj2, nuc_only, Snuc)

# returns total overlap of trajectory basis function
def s_integral(traj1, traj2, nuc_only=False, Snuc=None):
    """ Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""
    if traj1.state != traj2.state and not nuc_only:
        return complex(0.,0.)

    else:
        if Snuc is None:
            return nuclear.overlap(traj1, traj2)
        else:
            return Snuc

#
def v_integral(traj1, traj2, centroid=None, Snuc=None):
    """Returns potential coupling matrix element between two trajectories."""
    # evaluate just the nuclear component (for re-use)
    if Snuc is None:
        Snuc = nuclear.overlap(traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                               traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    states = np.sort(np.array([traj1.state, traj2.state]))
    v_total = complex(0.,0.)

    # roll through terms in the hamiltonian
    for i in range(ham.nterms):

        if states == ham.stalbl[i,:]-1: 
            # adiabatic states in diabatic basis -- cross terms between orthogonal
            # diabatic states are zero
            v_term = complex(1.,0.) * ham.coe[i]
            for q in range(ham.nmode_active):
                if ham.order[i,q] > 0:
                    v_term *=  nuclear.prim_v_integral(ham.order[i,q],
                               traj1.widths()[q],traj1.x()[q],traj1.p()[q],
                               traj2.widths()[q],traj2.x()[q],traj2.p()[q])

            v_total += v_term

    return v_total * Snuc 


def ke_integral(traj1, traj2, Snuc=None):
    """Returns kinetic energy integral over trajectories."""
    if traj1.state != traj2.state:
        return 0j

    else:

        if Snuc is None:
            Snuc = nuclear.overlap(traj1, traj2)

        ke = nuclear.deld2x(traj1, traj2, S=Snuc)

        return -sum( ke * interface.kecoeff)

def sdot_integral(traj1, traj2, Snuc=None):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if traj1.state != traj2.state:
        return complex(0.,0.)

    else:

        if Snuc is None:
            Snuc = nuclear.overlap(traj1, traj2)

        sdot = (-np.dot( traj2.velocity(), nuclear.deldx(traj1,traj2,S=Snuc) )+
                 np.dot( traj2.force(),    nuclear.deldp(traj1,traj2,S=Snuc) )+
                 1.j * traj2.phase_dot() * Snuc)

        return sdot
