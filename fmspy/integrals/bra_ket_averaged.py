"""
Compute Bra-ket averaged Taylor expansion integrals over trajectories
traveling on adiabataic potentials
"""
import sys
import math
import numpy as np
import src.fmsio.glbl as glbl
import src.integrals.nuclear_gaussian as nuclear

# Let FMS know if overlap matrix elements require PES info
overlap_requires_pes = False

# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the Hamiltonian symmetry
hermitian = True

# Returns functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'


def nuc_overlap(t1, t2):
    """ Returns < Chi | Chi' >, the nuclear overlap integral of two trajectories"""
    return nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                           t2.phase(),t2.widths(),t2.x(),t2.p())


def traj_overlap(t1, t2, nuc_only=False, Snuc=None):
    """Returns < Psi | Psi' >, the overlap integral of two trajectories.

    The bra and ket functions for the s_integral may be different
    (i.e. pseudospectral/collocation methods).
    """
    return s_integral(t1, t2, nuc_only=nuc_only, Snuc=Snuc)


def s_integral(t1, t2, nuc_only=False, Snuc=None):
    """ Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""
    if t1.state != t2.state and not nuc_only:
        return 0j
    else:
        if Snuc is None:
            return nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())
        else:
            return Snuc


def v_integral(t1, t2, centroid=None, Snuc=None):
    """Returns potential coupling matrix element between two trajectories.

    If we are passed a single trajectory, this is a diagonal matrix
    element -- simply return potential energy of trajectory.
    """

    if Snuc is None:
        Sij = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                              t2.phase(),t2.widths(),t2.x(),t2.p())
    else:
        Sij = Snuc
    Sji = Sij.conjugate()

    if glbl.propagate['integral_order'] > 2:
        raise ValueError('Integral_order > 2 not implemented for bra_ket_averaged')

    if t1.state == t2.state:
        state = t1.state
        # Adiabatic energy
        vij = t1.energy(state) * Sij
        vji = t2.energy(state) * Sji

        if glbl.propagate['integral_order'] > 0:
            o1_ij = nuclear.ordr1_vec(t1.widths(),t1.x(),t1.p(),
                                      t2.widths(),t2.x(),t2.p())
            o1_ji = nuclear.ordr1_vec(t2.widths(),t2.x(),t2.p(),
                                      t1.widths(),t1.x(),t1.p())
            vij += np.dot(o1_ij - t1.x()*Sij, t1.derivative(state,state))
            vji += np.dot(o1_ji - t2.x()*Sji, t2.derivative(state,state))

        if glbl.propagate['integral_order'] > 1:
            xcen  = (t1.widths()*t1.x() + t2.widths()*t2.x()) / (t1.widths()+t2.widths())
            o2_ij = nuclear.ordr2_vec(t1.widths(),t1.x(),t1.p(),
                                      t2.widths(),t2.x(),t2.p())
            o2_ji = nuclear.ordr2_vec(t2.widths(),t2.x(),t2.p(),
                                      t1.widths(),t1.x(),t1.p())

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

    # [necessarily] off-diagonal matrix element between trajectories
    # on different electronic states
    else:
        # Derivative coupling
        fij = t1.derivative(t1.state, t2.state)

        vij = 2.*np.vdot(t1.derivative(t1.state,t2.state), glbl.pes.kecoeff *
                         nuclear.deldx(Sij,t1.widths(),t1.x(),t1.p(),
                                           t2.widths(),t2.x(),t2.p()))
        vji = 2.*np.vdot(t2.derivative(t2.state,t1.state), glbl.pes.kecoeff *
                         nuclear.deldx(Sji,t2.widths(),t2.x(),t2.p(),
                                           t1.widths(),t1.x(),t1.p()))
    return 0.5*(vij + vji.conjugate())


def ke_integral(t1, t2, Snuc=None):
    """Returns kinetic energy integral over trajectories."""
    if t1.state != t2.state:
        return 0j

    else:
        if Snuc is None:
            Snuc = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())

        ke = nuclear.deld2x(Snuc,t1.widths(),t1.x(),t1.p(),
                                 t2.widths(),t2.x(),t2.p())

        return -np.dot(ke, glbl.pes.kecoeff)


def sdot_integral(t1, t2, Snuc=None, e_only=False, nuc_only=False):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if t1.state != t2.state:
        return 0j

    else:
        if Snuc is None:
            Snuc = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())

        deldx = nuclear.deldx(Snuc,t1.widths(),t1.x(),t1.p(),
                                   t2.widths(),t2.x(),t2.p())
        deldp = nuclear.deldp(Snuc,t1.widths(),t1.x(),t1.p(),
                                   t2.widths(),t2.x(),t2.p())

        sdot = (np.dot(deldx,t2.velocity()) + np.dot(deldp,t2.force()) +
                1j * t2.phase_dot() * Snuc)

        return sdot
