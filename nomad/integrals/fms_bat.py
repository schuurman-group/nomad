"""
Compute Bra-ket averaged Taylor expansion integrals over trajectories
traveling on adiabataic potentials
"""
import numpy as np
import nomad.core.glbl as glbl
import nomad.compiled.nuclear_gaussian as nuclear


# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the Hamiltonian symmetry
hermitian = True

# Returns functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'


def v_integral(t1, t2, kecoef, nuc_ovrlp, elec_ovrlp):
    """Returns potential coupling matrix element between two trajectories
    using the bra-ket averaged approach.

    If we are passed a single trajectory, this is a diagonal matrix
    element -- simply return potential energy of trajectory.
    """
    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    Sij = nuc_ovrlp
    Sji = Sij.conjugate()

    if glbl.properties['integral_order'] > 2:
        raise ValueError('Integral_order > 2 not implemented for bra_ket_averaged')

    if t1.state == t2.state:
        state = t1.state
        # Adiabatic energy
        vij = t1.energy(state) * Sij
        vji = t2.energy(state) * Sji

        if glbl.properties['integral_order'] > 0:

            o1_ij = nuclear.qn_vector(1, Sij, t1.widths(), t1.x(), t1.p(),
                                              t2.widths(), t2.x(), t2.p())
            o1_ji = nuclear.qn_vector(1, Sji, t2.widths(), t2.x(), t2.p(),
                                              t1.widths(), t1.x(), t1.p())
            vij += np.dot(o1_ij - t1.x()*Sij, t1.derivative(state,state))
            vji += np.dot(o1_ji - t2.x()*Sji, t2.derivative(state,state))

        if glbl.properties['integral_order'] > 1:
            xcen  = (t1.widths()*t1.x() + t2.widths()*t2.x()) / (t1.widths()+t2.widths())
            o2_ij = nuclear.qn_vector(2, Sij, t1.widths(), t1.x(), t1.p(),
                                              t2.widths(), t2.x(), t2.p())
            o2_ji = nuclear.qn_vector(2, Sji, t2.widths(), t2.x(), t2.p(),
                                              t1.widths(), t1.x(), t1.p())

            for k in range(t1.dim):
                vij += 0.5*o2_ij[k]*t1.hessian(state)[k,k]
                vji += 0.5*o2_ji[k]*t2.hessian(state)[k,k]
                for l in range(k):
                    vij += 0.5 * ((2.*o1_ij[k]*o1_ij[l] -
                                   xcen[k]*o1_ij[l] - xcen[l]*o1_ij[k] -
                                   o1_ij[k]*t1.x()[l] - o1_ij[l]*t1.x()[k] +
                                   (t1.x()[k]*xcen[l] + t1.x()[l]*xcen[k])*Sij) *
                                  t1.hessian(state)[k,l])
                    vji += 0.5 * ((2.*o1_ji[k]*o1_ji[l] -
                                   xcen[k]*o1_ji[l] - xcen[l]*o1_ji[k] -
                                   o1_ji[k]*t2.x()[l] - o1_ji[l]*t2.x()[k] +
                                   (t2.x()[k]*xcen[l] + t2.x()[l]*xcen[k])*Sji) *
                                  t2.hessian(state)[k,l])

        return 0.5*(vij + vji.conjugate())

    # [necessarily] off-diagonal matrix element between trajectories
    # on different electronic states
    else:
        # Derivative coupling
        fij = t1.derivative(t1.state, t2.state)
        fji = t2.derivative(t1.state, t2.state)
        vij = 2.*np.vdot(fij, kecoef *
                         nuclear.deldx(Sij,t1.widths(),t1.x(),t1.p(),
                                           t2.widths(),t2.x(),t2.p()))
        vji = 2.*np.vdot(fji, kecoef *
                         nuclear.deldx(Sji,t2.widths(),t2.x(),t2.p(),
                                           t1.widths(),t1.x(),t1.p()))
#        vij = np.vdot(t1.velocity(), fij)
#        vji = np.vdot(t2.velocity(), fji)
#        print("t1.v="+str(t1.velocity()))
#        print("t2.v="+str(t2.velocity()))
        print("del/dx t1="+str(nuclear.deldx(Sij,t1.widths(),t1.x(),t1.p(),
                                           t2.widths(),t2.x(),t2.p())))
        print("del/dx t2="+str(nuclear.deldx(Sji,t2.widths(),t2.x(),t2.p(),
                                           t1.widths(),t1.x(),t1.p())))
        print("fij="+str(fij))
        print("fji="+str(fji))
        print("vij="+str(vij))
        print("vji="+str(vji))
        print("Sij="+str(Sij)) 
#        return complex(0,1.)*0.5*Sij*(vij + vji)
        return 0.5*(vij + vji)
   
