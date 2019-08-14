"""
Compute integrals over trajectories traveling on vibronic potentials
"""
import numpy as np
import nomad.core.glbl as glbl
import nomad.compiled.nuclear_gaussian as nuclear

# Let FMS know if overlap matrix elements require PES info
overlap_requires_pes = False

# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the Hamiltonian symmetry
hermitian = True

# functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'

def v_integral(t1, t2, kecoef, nuc_ovrlp, elec_ovrlp):
    """Returns potential coupling matrix element between two trajectories."""
    # evaluate just the nuclear component (for re-use)
    if elec_ovrlp == 0.:
        return 0j 

    states = np.sort(np.array([t1.state, t2.state]))
    v_total = complex(0.,0.)

    # roll through terms in the hamiltonian
    for i in range(glbl.modules['interface'].ham.nterms):
        if np.array_equal(states,glbl.modules['interface'].ham.stalbl[i,:]-1):
            # adiabatic states in diabatic basis -- cross terms between orthogonal
            # diabatic states are zero
            [s1,s2] = glbl.modules['interface'].ham.stalbl[i,:]-1
            v_term = complex(1.,0.) * glbl.modules['interface'].ham.coe[i]
            for q in range(len(glbl.modules['interface'].ham.order[i])):
                qi      =  glbl.modules['interface'].ham.mode[i][q]
                v_term *=  nuclear.qn_integral(glbl.modules['interface'].ham.order[i][q],
                                                t1.widths()[qi], t1.x()[qi], t1.p()[qi],
                                                t2.widths()[qi], t2.x()[qi], t2.p()[qi])
            v_total += v_term

    return v_total * nuc_ovrlp

