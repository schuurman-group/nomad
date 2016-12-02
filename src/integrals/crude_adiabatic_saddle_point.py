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
import src.integrals.nuclear_gaussian as ints
interface = __import__('src.interfaces.' + glbl.fms['interface'],
                       fromlist = ['a'])

# Let propagator know if we need data at centroids to propagate
require_centroids = True

# Determines the Hamiltonian symmetry
hermitian = True

# returns basis in which matrix elements are evaluated
basis = 'gaussian'

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
    """Returns potential coupling matrix element between two trajectories."""
    # if we are passed a single trajectory, this is a diagonal
    # matrix element -- simply return potential energy of trajectory
    if traj1.tid == traj2.tid:
        # Adiabatic energy
        v = traj1.energy(traj1.state)
        # DBOC
        if glbl.fms['coupling_order'] == 3:
            v += traj1.scalar_coup(traj1.state)
        return v

    if Snuc is None:
        Snuc = nuc_ints.overlap(traj1, traj2)

    # off-diagonal matrix element, between trajectories on the same
    # state [this also requires the centroid be present
    elif traj1.state == traj2.state:
        # Adiabatic energy
        v = centroid.energy(traj1.state) * Snuc
        # DBOC
        if glbl.fms['coupling_order'] == 3:
            v += centroid.scalar_coup(traj1.state) * Snuc
        return v

    # [necessarily] off-diagonal matrix element between trajectories
    # on different electronic states
    elif traj1.state != traj2.state:
        # Derivative coupling
        fij = centroid.derivative(traj1.state)
        v = np.vdot(fij, 
                    nuc_ints.deldx(traj1,traj2, S=Snuc)* 2.* interface.kecoeff)
        # Scalar coupling
        if glbl.fms['coupling_order'] > 1:
            v += traj1.scalar_coup(traj2.state) * Snuc
        return v

    else:
        print('ERROR in v_integral -- argument disagreement')
        return complex(0.,0.)

# kinetic energy integral
def ke_integral(traj1, traj2, Snuc=None):
    """Returns kinetic energy integral over trajectories."""
    if traj1.state != traj2.state:
        return complex(0.,0.)

    else:
        if Snuc is None:
            Snuc = nuc_ints.overlap(traj1, traj2)
        ke = nuc_ints.deld2x(traj1, traj2, S = Snuc)
        return -sum( ke * interface.kecoeff)

# time derivative of the overlap
def sdot_integral(traj1, traj2, Snuc=None):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if traj1.state != traj2.state:
        return complex(0.,0.)

    else:
        if Snuc is None:
            Snuc = nuc_ints.overlap(traj1, traj2)
        sdot = (-np.dot( traj2.velocity(), nuc_ints.deldx(traj1,traj2,S=Snuc) ) +
                 np.dot( traj2.force()   , nuc_ints.deldp(traj1,traj2,S=Snuc) ) +
                 1j * traj2.phase_dot() * Snuc)
        return sdot
