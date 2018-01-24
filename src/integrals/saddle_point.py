"""
Compute saddle-point integrals over trajectories traveling on adiabataic
potentials

This currently uses first-order saddle point.
"""
import sys
import math
import numpy as np
import src.fmsio.glbl as glbl
import src.integrals.nuclear_gaussian as nuclear
interface = __import__('src.interfaces.' + glbl.interface['interface'],
                       fromlist = ['a'])

# Let FMS know if overlap matrix elements require PES info
overlap_requires_pes = False

# Let propagator know if we need data at centroids to propagate
require_centroids = True

# Determines the Hamiltonian symmetry
hermitian = False 

# Returns functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'

def nuc_overlap(t1, t2):
    """ Returns < Chi | Chi' >, the nuclear overlap integral of two trajectories"""
    return nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                           t2.phase(),t2.widths(),t2.x(),t2.p())

# the bra and ket functions for the s_integral may be different
# (i.e. pseudospectral/collocation methods).
def traj_overlap(t1, t2, nuc_only=False, Snuc=None):
    """ Returns < Psi | Psi' >, the overlap integral of two trajectories"""
    return s_integral(t1, t2, nuc_only=nuc_only, Snuc=Snuc)

# returns total overlap of trajectory basis function
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
    """Returns potential coupling matrix element between two trajectories."""
    # if we are passed a single trajectory, this is a diagonal
    # matrix element -- simply return potential energy of trajectory
    if t1.label == t2.label:
        # Adiabatic energy
        v = t1.energy(t1.state)
        # DBOC
        if glbl.interface['coupling_order'] == 3:
            v += t1.scalar_coup(t1.state, t2.state)
        return v

    if Snuc is None:
        Snuc = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                               t2.phase(),t2.widths(),t2.x(),t2.p())
#        Snuc = nuclear.overlap(t1,t2)

    # off-diagonal matrix element, between trajectories on the same
    # state (this also requires the centroid be present)
    elif t1.state == t2.state:
        # Adiabatic energy
        v = centroid.energy(t1.state) * Snuc
        # DBOC
        if glbl.interface['coupling_order'] == 3:
            v += centroid.scalar_coup(t1.state, t2.state) * Snuc
        return v

    # [necessarily] off-diagonal matrix element between trajectories
    # on different electronic states
    elif t1.state != t2.state:
        # Derivative coupling
        fij = centroid.derivative(t1.state, t2.state)
        v = 2.*np.vdot(fij, interface.kecoeff *
                       nuclear.deldx(Snuc,t1.phase(),t1.widths(),t1.x(),t1.p(),
                                          t2.phase(),t2.widths(),t2.x(),t2.p()))
#                       nuclear.deldx(t1, t2, S=Snuc))
        # Scalar coupling
        if glbl.interface['coupling_order'] > 1:
            v += centroid.scalar_coup(t1.state, t2.state) * Snuc
        return v

    else:
        print('ERROR in v_integral -- argument disagreement')
        return 0j

# kinetic energy integral
def ke_integral(t1, t2, Snuc=None):
    """Returns kinetic energy integral over trajectories."""
    if t1.state != t2.state:
        return 0j

    else:
        if Snuc is None:
            Snuc = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())

        ke = nuclear.deld2x(Snuc,t1.phase(),t1.widths(),t1.x(),t1.p(),
                                 t2.phase(),t2.widths(),t2.x(),t2.p())

        return -np.dot(ke, interface.kecoeff)

# time derivative of the overlap
def sdot_integral(t1, t2, Snuc=None, e_only=False, nuc_only=False):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if t1.state != t2.state:
        return 0j

    else:
        if Snuc is None:
            Snuc = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())

        deldx = nuclear.deldx(Snuc,t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())
        deldp = nuclear.deldp(Snuc,t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())

        sdot = (np.dot(deldx,t2.velocity()) + np.dot(deldp,t2.force())
                +1j * t2.phase_dot() * Snuc)

        return sdot
