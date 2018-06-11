"""
Compute saddle-point integrals over trajectories traveling on adiabataic
potentials

This currently uses first-order saddle point.
"""
import sys
import numpy as np
import nomad.parse.glbl as glbl
import nomad.integrals.nuclear_gaussian as nuclear

# Let propagator know if we need data at centroids to propagate
require_centroids = True

# Determines the Hamiltonian symmetry
hermitian = False

# Returns functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'


def elec_overlap(t1, t2):
    """ Returns < phi | phi' >, the electronic overlap integral of two trajectories"""
    t1.state == t2.state:
        return 1.
    else:
        return 0.


def nuc_overlap(t1, t2):
    """ Returns < Chi | Chi' >, the nuclear overlap integral of two trajectories"""
    return nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                           t2.phase(),t2.widths(),t2.x(),t2.p())


def traj_overlap(t1, t2, nuc_ovrlp=None):
    """ Returns < Psi | Psi' >, the overlap integral of two trajectories"""
    return s_integral(t1, t2, nuc_ovrlp=nuc_ovrlp)


def s_integral(t1, t2, nuc_ovrlp=None):
    """ Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""
    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    return elec_overlap(t1,t2) * nuc_ovrlp


def v_integral(t1, t2, centroid, nuc_ovrlp=None):
    """Returns potential coupling matrix element between two trajectories."""
    # if we are passed a single trajectory, this is a diagonal
    # matrix element -- simply return potential energy of trajectory
    if t1.label == t2.label:
        # Adiabatic energy
        v = t1.energy(t1.state)
        # DBOC
        if glbl.iface_params['coupling_order'] == 3:
            v += t1.scalar_coup(t1.state, t2.state)
        return v

    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    # off-diagonal matrix element, between trajectories on the same
    # state (this also requires the centroid be present)
    elif t1.state == t2.state:
        # Adiabatic energy
        v = centroid.energy(t1.state) * nuc_ovrlp
        # DBOC
        if glbl.iface_params['coupling_order'] == 3:
            v += centroid.scalar_coup(t1.state, t2.state) * Snuc
        return v

    # [necessarily] off-diagonal matrix element between trajectories
    # on different electronic states
    elif t1.state != t2.state:
        # Derivative coupling
        fij = centroid.derivative(t1.state, t2.state)
        v = 2.*np.vdot(fij, t1.kecoef *
                       nuclear.deldx(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                               t2.widths(),t2.x(),t2.p()))
        # Scalar coupling
        if glbl.iface_params['coupling_order'] > 1:
            v += centroid.scalar_coup(t1.state, t2.state) * Snuc
        return v

    else:
        print('ERROR in v_integral -- argument disagreement')
        return 0j


def t_integral(t1, t2, nuc_ovrlp=None):
    """Returns kinetic energy integral over trajectories."""
    if t1.state != t2.state:
        return 0j

    else:
        if nuc_ovrlp is None:
            nuc_ovrlp = nuc_overlap(t1, t2)

        ke = nuclear.deld2x(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                      t2.widths(),t2.x(),t2.p())

        return -np.dot(ke, t1.kecoef)


def sdot_integral(t1, t2, nuc_ovrlp=None):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if t1.state != t2.state:
        return 0j

    else:
        if nuc_ovrlp is None:
            nuc_ovrlp = nuc_overlap(t1, t2)

        deldx = nuclear.deldx(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                        t2.widths(),t2.x(),t2.p())
        deldp = nuclear.deldp(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                        t2.widths(),t2.x(),t2.p())

        sdot = (np.dot(deldx,t2.velocity()) + np.dot(deldp,t2.force())
                +1j * t2.phase_dot() * Snuc)

        return sdot