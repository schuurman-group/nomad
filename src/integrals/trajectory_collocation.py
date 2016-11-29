"""
Computes matrix elements over a dirac delta function at the centre of
trajectory 1 and trajectory 2.
"""
import sys
import math
import numpy as np
import src.fmsio.glbl as glbl
import src.interfaces.vcham.hampar as ham
nuc_ints  = __import__('src.integrals.nuclear_'+glbl.fms['test_function'],
                     fromlist=['NA'])
interface = __import__('src.interfaces.' + glbl.fms['interface'],
                       fromlist = ['a'])

# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the Hamiltonian symmetry
hermitian = False

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
    """ Returns < delta(R-R1) | V | g2 > if state1 = state2, else
    returns < delta(R-R1) | F . d/dR | g2 > """
    if Snuc is None:
        Snuc = nuc_ints.overlap(traj1,traj2)

    # Off-diagonal element between trajectories on the same adiabatic
    # state
    if  traj1.state == traj2.state:
        # Adiabatic energy
        return traj1.energy(traj1.state) * Snuc

    # off-diagonal matrix elements between trajectories on different
    # elecronic states
    elif traj1.state != traj2.state:
        # Derivative coupling
        fij = traj1.derivative(traj2.state)
        v = np.dot(fij, nuc_ints.deldx(traj1, traj2, S=Snuc)* 
                        2.*interface.kecoeff[i*traj1.d_particle])
        return v * Snuc
    else:
        print('ERROR in v_integral -- argument disagreement')
        return complex(0.,0.) 

# evaluate the kinetic energy integral
def ke_integral(traj1, traj2, Snuc=None):
    """ Returns < delta(R-R1) | T | g2 > """
    if traj1.state != traj2.state:
        return complex(0.,0.)
    else:
        if Snuc is None:
            Snuc = nuc_ints.overlap(traj1,traj2)
        ke = nuc_ints.deld2x(traj1, traj2, S=Snuc)
        return -sum(ke * interface.kecoeff)

#evaulate the time derivative of the overlap
def sdot_integral(traj1, traj2, Snuc=None):
    """ Returns < delta(R-R1) | d/dt g2 >

    Note that for now we have to pass Snuc=1 to deldx and deldp so
    that these functions do not multiply the result by the overlap
    of traj1 and traj2. This isn't ideal, but will do for now.
    """
    if traj1.state != traj2.state:
        return complex(0.,0.) 
    else:
        if Snuc is None:
            Snuc = nuc_ints.overlap(traj1,traj2)
        sdot = (np.dot(traj2.velocity(), nuc_ints.deldx(traj1, traj2, S=Snuc)) +
                np.dot(traj2.force()   , nuc_ints.deldp(traj1, traj2, S=Snuc)) +
                1j * traj2.phase_dot() * Snuc)
        return sdot
