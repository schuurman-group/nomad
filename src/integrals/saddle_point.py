"""
Compute saddle-point integrals over trajectories traveling on adiabataic
potentials

This currently uses first-order saddle point.
"""
import sys
import math
import numpy as np
import src.fmsio.glbl as glbl
import src.interfaces.vcham.hampar as ham
import src.integrals.nuclear_gaussian as nuclear
interface = __import__('src.interfaces.' + glbl.fms['interface'],
                       fromlist = ['a'])

# Let propagator know if we need data at centroids to propagate
require_centroids = True

# Determines the Hamiltonian symmetry
hermitian = True

# Returns functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'

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

def v_integral(t1, t2, centroid=None, Snuc=None):
    """Returns potential coupling matrix element between two trajectories."""
    # if we are passed a single trajectory, this is a diagonal
    # matrix element -- simply return potential energy of trajectory
    if t1.tid == t2.tid:
        # Adiabatic energy
        v = t1.energy(t1.state)
        # DBOC
        if glbl.fms['coupling_order'] == 3:
            v += t1.scalar_coup(t1.state)
        return v

    if Snuc is None:
        Snuc = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                               t2.phase(),t2.widths(),t2.x(),t2.p())

    # off-diagonal matrix element, between trajectories on the same
    # state [this also requires the centroid be present
    elif t1.state == t2.state:
        # Adiabatic energy
        v = centroid.energy(t1.state) * Snuc
        # DBOC
        if glbl.fms['coupling_order'] == 3:
            v += centroid.scalar_coup() * Snuc
        return v

    # [necessarily] off-diagonal matrix element between trajectories
    # on different electronic states
    elif t1.state != t2.state:
        # Derivative coupling
        fij = centroid.derivative()
        v = np.vdot(fij, 2.* interface.kecoeff *
                    nuclear.deldx(Snuc,t1.phase(),t1.widths(),t1.x(),t1.p(),
                                       t2.phase(),t2.widths(),t2.x(),t2.p()))
        # Scalar coupling
        if glbl.fms['coupling_order'] > 1:
            v += t1.scalar_coup(t2.state) * Snuc
        return v

    else:
        print('ERROR in v_integral -- argument disagreement')
        return complex(0.,0.)

# kinetic energy integral
def ke_integral(t1, t2, Snuc=None):
    """Returns kinetic energy integral over trajectories."""
    if t1.state != t2.state:
        return complex(0.,0.)

    else:
        if Snuc is None:
            Snuc = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())

        ke = nuclear.deld2x(Snuc,t1.phase(),t1.widths(),t1.x(),t1.p(),
                                 t2.phase(),t2.widths(),t2.x(),t2.p())
        return -sum( ke * interface.kecoeff)

# time derivative of the overlap
def sdot_integral(t1, t2, Snuc=None):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if t1.state != t2.state:
        return complex(0.,0.)

    else:
        if Snuc is None:
            Snuc = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())

        deldx = nuclear.deldx(Snuc,t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())
        deldp = nuclear.deldp(Snuc,t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())

        sdot = (-np.dot(t2.velocity(),deldx) + np.dot(t2.force(), deldp) 
                +1j * t2.phase_dot() * Snuc)
        return sdot
