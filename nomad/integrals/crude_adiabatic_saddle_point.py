"""
Compute saddle-point integrals over trajectories traveling on adiabataic
potentials

This currently uses first-order saddle point.
"""
import numpy as np
import nomad.integrals.nuclear_gaussian as nuclear
import nomad.interfaces.vibronic as vibronic
import nomad.interfaces.vcham.hampar as ham

# Let FMS know if overlap matrix elements require PES info
overlap_requires_pes = True

# Let propagator know if we need data at centroids to propagate
require_centroids = True 

# Determines the Hamiltonian symmetry
hermitian = True 

# returns basis in which matrix elements are evaluated
basis = 'gaussian'

def elec_overlap(traj1, traj2, centroid):
    """ Returns < Psi | Psi' >, the overlap integral of two trajectories"""

    sij = complex(0.,0.)

    if traj1.tid == traj2.tid:
        sij += 1.        

    sij += np.dot( centroid.derivative(traj1.state,traj2.state),
                   traj2.x()-traj1.x())

    return sij 

# returns the overlap between two trajectories (differs from s_integral in that
# the bra and ket functions for the s_integral may be different
# (i.e. pseudospectral/collocation methods). 
def traj_overlap(traj1, traj2, centroid, nuc_only=False):
    """ Returns < Psi | Psi' >, the overlap integral of two trajectories"""
    nuc_ovrlp = nuclear.overlap(traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                                traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    if nuc_only:
        return nuc_ovrlp
    else:
        return nuc_ovrlp * elec_overlap(traj1, traj2, centroid)

# total overlap of trajectory basis function
def s_integral(traj1, traj2, centroid=None, nuc_only=False, Snuc=None):
    """ Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""
    if Snuc is None:
        Snuc = traj_overlap(traj1, traj2, centroid=centroid, nuc_only=True)

    if nuc_only:
        return Snuc
    else:
        return Snuc * elec_overlap(traj1, traj2, centroid) 

#
def v_integral(traj1, traj2, centroid=None, Snuc=None):
    """Returns potential coupling matrix element between two trajectories."""
    # evaluate just the nuclear component (for re-use)
    if Snuc is None:
        Snuc = nuclear.overlap(traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                               traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    dx_1 = traj1.x() - centroid.x()
    dx_2 = traj2.x() - centroid.x()
    dp   = traj1.p() - traj2.p()
    if traj1.tid == traj2.tid:
        vint = Snuc * (traj1.energy(traj1.state) + 
                   0.5*np.dot(centroid.derivative(traj1.state,traj1.state),
                              dx_1 + dx_2 - 1j*dp))
    else:
        ave_ener = 0.5 * (centroid.energy(traj1.state) + 
                          centroid.energy(traj2.state))
        dif_ener = 0.5 * (centroid.energy(traj1.state) - 
                          centroid.energy(traj2.state))
        vint = Snuc * np.dot(centroid.derivative(traj1.state,traj2.state),
                       ave_ener * dx_2 - ave_ener * dx_1 + 1j * dif_ener * dp)
        
    return vint

# kinetic energy integral
def ke_integral(traj1, traj2, centroid=None, Snuc=None):
    """Returns kinetic energy integral over trajectories."""
    # evaluate just the nuclear component (for re-use)
    if Snuc is None:
        Snuc = nuclear.overlap(traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                               traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    # overlap of electronic functions
    Selec = elec_overlap(traj1, traj2)

    # < chi | del^2 / dx^2 | chi'> 
    ke = nuclear.deld2x(Snuc,traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                             traj2.phase(),traj2.widths(),traj2.x(),traj2.p())
   
    return -sum( ke * vibronic.kecoeff) * Selec

    
# time derivative of the overlap
def sdot_integral(traj1, traj2, centroid=None, Snuc=None, e_only=False, nuc_only=False):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if Snuc is None:
        Snuc = nuclear.overlap(traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                               traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    # overlap of electronic functions
    Selec = elec_overlap(traj1, traj2, centroid)

    # < chi | d / dx | chi'>
    deldx = nuclear.deldx(Snuc,traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                               traj2.phase(),traj2.widths(),traj2.x(),traj2.p())
    # < chi | d / dp | chi'>
    deldp = nuclear.deldp(Snuc,traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                               traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    # the nuclear contribution to the sdot matrix
    sdot = ( np.dot(traj2.velocity(), deldx) + np.dot(traj2.force(), deldp) 
            + 1j * traj2.phase_dot() * Snuc) * Selec

    # time-derivative of the electronic component
    
    return sdot
 
