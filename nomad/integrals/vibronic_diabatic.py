"""
Compute integrals over trajectories traveling on vibronic potentials
"""
import numpy as np
import nomad.integrals.nuclear_gaussian as nuclear

# Let FMS know if overlap matrix elements require PES info
overlap_requires_pes = False

# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the Hamiltonian symmetry
hermitian = True

# functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'

def nuc_overlap(t1, t2):
    """ Returns the just the nuclear component of the overlap integral 
        between two trajectories"""
    return nuclear.overlap(t1.phase(), t1.widths(), t1.x(), t1.p(),
                           t2.phase(), t2.widths(), t2.x(), t2.p())

def elec_overlap(t1, t2):
    """ Returns the overlap between two electronic wavefunctions in diabatic
        basis (i.e. KroneckerDelta)"""
    if t1.state != t2.state:
        return complex(0.,0.)
    else:
        return complex(1.,0.)

def traj_overlap(t1, t2, nuc_ovrlp=None):
    """ Returns < Psi | Psi' >, the overlap integral of two trajectories"""
    return s_integral(t1, t2, nuc_ovrlp=nuc_ovrlp)

def s_integral(t1, t2, nuc_ovrlp=None):
    """ Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""
    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1,t2)

    return elec_overlap(t1,t2) * nuc_ovrlp


def v_integral(t1, t2, nuc_ovrlp=None):
    """Returns potential coupling matrix element between two trajectories."""
    # evaluate just the nuclear component (for re-use)
    if nuc_ovrlp is None:
        nuc_ovrlp = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
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

    return v_total * nuc_ovrlp


def ke_integral(t1, t2, nuc_ovrlp=None):
    """Returns kinetic energy integral over trajectories."""
    if t1.state != t2.state:
        return complex(0.,0.)
    else:
        if nuc_ovrlp is None:
            nuc_ovrlp = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())

        ke = nuclear.deld2x(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                 t2.widths(),t2.x(),t2.p())

        return -sum( ke * t1.kecoef)


def sdot_integral(t1, t2, nuc_ovrlp=None):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if t1.state != t2.state:
        return complex(0.,0.)
    else:
        if nuc_ovrlp is None:
            nuc_ovrlp = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())
        t1_dx_t2 = nuclear.deldx(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                      t2.widths(),t2.x(),t2.p())
        t1_dp_t2 = nuclear.deldp(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                      t2.widths(),t2.x(),t2.p())
        sdot = ( np.dot( t2.velocity(), t1_dx_t2) +
                 np.dot( t2.force(),    t1_dp_t2) +
                 1.j*t2.phase_dot()*nuc_ovrlp )

        return sdot
