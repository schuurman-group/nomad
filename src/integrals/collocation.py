"""
Computes matrix elements over a dirac delta function at the centre of
trajectory 1 and trajectory 2.
""" 

import sys
import numpy as np
import src.fmsio.glbl as glbl
import src.interfaces.vcham.hampar as ham

interface = __import__('src.interfaces.' + glbl.fms['interface'],
                       fromlist = ['a'])

# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the basis set
basis = 'dirac_delta'

# Determines the Hamiltonian symmetry
hamsym = 'nosym'

#
# potential coupling matrix element between two trajectories
#
def v_integral(traj1, traj2=None, centroid=None, S_ij=None):
    """ Returns < delta(R-R1) | V | g2 > if state1 = state2, else
    returns < delta(R-R1) | F . d/dR | g2 > """
    #
    # On-diagonal element
    if traj2 is None:
        # Adiabatic energy
        return traj1.energy(traj1.state) * traj1.h_overlap(traj1)

    #
    # off-diagonal matrix element, between trajectories on the same
    # state
    elif traj1.state == traj2.state:
        return traj1.energy(traj1.state) * traj1.h_overlap(traj2)

    #
    # off-diagonal matrix elements between trajectories on different 
    # elecronic states
    elif traj1.state != traj2.state:
        # Derivative coupling
        # Note that for now we have to pass S_ij=1 to deldx_m so
        # that this function does not multiply the result by the overlap
        # of traj1 and traj2. This isn't ideal, but will do for now.
        fij = traj1.derivative(traj2.state)
        v = np.dot(fij, traj1.deldx_m(traj2, S_ij=complex(1.0,0.0)))
        return v * traj1.h_overlap(traj2)
        
    else:
        print('ERROR in v_integral -- argument disagreement')
        return 0.
        
#
# kinetic energy integral over trajectories
#
def ke_integral(traj1, traj2, S_ij=None):
    """ Returns < delta(R-R1) | T | g2 > """
    ke = complex(0.,0.)
    if traj1.state == traj2.state:
        for i in range(traj1.n_particle):
            ke -= (traj1.particles[i].deld2x(traj2.particles[i]) *
                   interface.kecoeff[i*traj1.d_particle])
        return ke * traj1.h_overlap(traj2)
    else:
        return ke
    
#
# return the matrix element <Psi_1 | d/dt | Psi_2> 
#
def sdot_integral(traj1, traj2, S_ij=None):
    """ Returns < delta(R-R1) | d/dt g2 > """
    # Note that for now we have to pass S_ij=1 to deldx and deldp so
    # that these functions do not multiply the result by the overlap
    # of traj1 and traj2. This isn't ideal, but will do for now.
    sdot =  (np.dot( traj2.velocity(), traj1.deldx(traj2, S_ij=complex(1.,0.)) ) + \
             np.dot( traj2.force()   , traj1.deldp(traj2, S_ij=complex(1.,0.)) ) + \
             complex(0.,1.) * traj2.phase_dot()) * traj1.h_overlap(traj2)
    return sdot

