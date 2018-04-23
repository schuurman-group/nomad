"""
Compute integrals over trajectories traveling on vibronic potentials
"""
import math
import numpy as np
import src.integrals.nuclear_gaussian as nuclear

# Let FMS know if overlap matrix elements require PES info
overlap_requires_pes = False

# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the Hamiltonian symmetry
hermitian = True

# Returns functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'

# returns the overlap between two trajectories (differs from s_integral in that
# the bra and ket functions for the s_integral may be different
# (i.e. pseudospectral/collocation methods). 
def traj_overlap(t1, t2, nuc_only=False, Snuc=None):
    """ Returns < Psi | Psi' >, the overlap integral of two trajectories"""
    return s_integral(t1, t2, nuc_only, Snuc)

# returns total overlap of trajectory basis function
def s_integral(t1, t2, nuc_only=False, Snuc=None):
    """ Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""
    if t1.state != t2.state and not nuc_only:
        return complex(0.,0.)
    else:
        if Snuc is None:
            return nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())
        else:
            return Snuc

#
def v_integral(t1, t2, Snuc=None):
    """Returns potential coupling matrix element between two trajectories."""
    # evaluate just the nuclear component (for re-use)
    if Snuc is None:
        Snuc = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                               t2.phase(),t2.widths(),t2.x(),t2.p())

    states = np.sort(np.array([t1.state, t2.state]))
    v_total = complex(0.,0.)

    # roll through terms in the hamiltonian
    for i in range(glbl.interface.ham.nterms):

        if np.array_equal(states,glbl.interface.ham.stalbl[i,:]-1): 
            # adiabatic states in diabatic basis -- cross terms between orthogonal
            # diabatic states are zero
            [s1,s2] = glbl.interface.ham.stalbl[i,:]-1
            v_term = complex(1.,0.) * glbl.interface.ham.coe[i]
            for q in range(len(glbl.interface.ham.order[i])):
                qi      =  glbl.interface.ham.mode[i][q]
                v_term *=  nuclear.prim_v_integral(glbl.interface.ham.order[i][q],
                           t1.widths()[qi],t1.x()[qi],t1.p()[qi],
                           t2.widths()[qi],t2.x()[qi],t2.p()[qi])
            v_total += v_term

    return v_total * Snuc 


def ke_integral(t1, t2, Snuc=None):
    """Returns kinetic energy integral over trajectories."""
    if t1.state != t2.state:
        return complex(0.,0.) 
    else:
        if Snuc is None:
            Snuc = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p()) 

        ke = nuclear.deld2x(Snuc,t1.widths(),t1.x(),t1.p(),
                                 t2.widths(),t2.x(),t2.p())

        return -sum( ke * glbl.interface.kecoeff)

def sdot_integral(t1, t2, Snuc=None):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if t1.state != t2.state:
        return complex(0.,0.)
    else:
        if Snuc is None:
            Snuc = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())
        t1_dx_t2 = nuclear.deldx(Snuc,t1.widths(),t1.x(),t1.p(),
                                      t2.widths(),t2.x(),t2.p())
        t1_dp_t2 = nuclear.deldp(Snuc,t1.widths(),t1.x(),t1.p(),
                                      t2.widths(),t2.x(),t2.p())
        sdot = ( np.dot( t2.velocity(), t1_dx_t2) + 
                 np.dot( t2.force(),    t1_dp_t2) + 
                 1.j*t2.phase_dot()*Snuc )

        return sdot
