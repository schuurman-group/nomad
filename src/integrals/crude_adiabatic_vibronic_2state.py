"""
Compute saddle-point integrals over trajectories traveling on adiabataic
potentials

This currently uses first-order saddle point.
"""
import numpy as np
import src.fmsio.glbl as glbl
import src.integrals.nuclear_gaussian as nuclear
import src.interfaces.vibronic as vibronic 
import src.interfaces.vcham.hampar as ham

# Let propagator know if we need data at centroids to propagate
require_centroids = False 

# Determines the Hamiltonian symmetry
hermitian = True 

# returns basis in which matrix elements are evaluated
basis = 'gaussian'

# cache of theta values -- to ensure theta is smoothly varying
adt_cache = dict()

# adiabatic-diabatic rotation matrix
def rot_mat(theta):
    """ Returns the adiabatic-diabatic rotation matrix for a given value of
        theta"""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

# derivative of the adiabatic-diabatic rotation matrix
def drot_mat(theta):
    """ Returns the derivative adiabatic-diabatic rotation matrix with respect
        to theta"""
    return np.array([[-np.sin(theta), -np.cos(theta)],
                     [ np.cos(theta), -np.sin(theta)]])

# compute the diabatic-adiabatic rotation angle theta
def theta(traj):
    """ Returns to the adiabatic-diabatic rotation angle theta. Choose theta
        to be consistent with diabatic-adiabatic transformation matrix, which
        itself is chosen to have a phase resulting in a slowly varying value of
        of theta."""
 
    # can also run the trivial case of a single state
    if traj.nstates == 1:
        return 0.

    hmat    = traj.pes_data.diabat_pot
    dat_mat = traj.pes_data.dat_mat
    h12     = hmat[0,1]
    de      = hmat[1,1]-hmat[0,0]
    ang     = 0.5*np.arctan(2.*h12/de)
    adia_st = rot_mat(ang)

    # confirm that this rotation results in correct state ordering
    adia_eners = np.diag(np.dot(np.dot(adia_st, hmat),
                                       np.transpose(adia_st)))
    if np.linalg.norm(adia_eners-np.sort(adia_eners)) > glbl.fpzero:
        # if we've swapped adiabatic states, rotate by pi/2
        ang -= 0.5 * np.pi
        adia_eners = np.diag(np.dot(np.dot(rot_mat(ang), hmat), 
                                           np.transpose(rot_mat(ang))))

    # ensure theta agrees with dat matrix
    pi_mult  = [-2.,1.,0.,1.,2.]
    dif_vec  = np.array([np.linalg.norm(dat_mat - rot_mat(ang+i*np.pi))
                         for i in pi_mult])

    if np.min(dif_vec) < glbl.fpzero:
        ang += pi_mult[np.argmin(dif_vec)]*np.pi

    return ang

# compute the derivative of the diabatic-adiabatic rotation angle theta
def dtheta(traj):
    """ Returns to the derivative adiabatic-diabatic rotation angle theta with
        respect to the internal coordinates."""

    # can also run the trivial case of a single state
    if traj.nstates == 1:
        return np.zeros((traj.dim), dtype=float)

    hmat      = traj.pes_data.diabat_pot
    dhmat     = traj.pes_data.diabat_deriv 
    h12       = hmat[0,1]
    de        = hmat[1,1] - hmat[0,0]
    arg       = 2. * h12 / de

    dtheta_dq = np.array([(dhmat[q,0,1] / de - 
                        h12*(dhmat[q,1,1]-dhmat[q,0,0])/de**2)/(1+arg**2)
                        for q in range(traj.dim)])

    return dtheta_dq

# return the diabatic-adiabatic transformation matrix
def phi(traj):
    """Returns the transformation matrix using the rotation angle. 
       Should be indentical to the dat_mat in the vibronic interface"""

    # can also run the trivial case of a single state
    if traj.nstates == 1:
        return np.array([1.], dtype=float)

    angle   = theta(traj)
    phi_mat = rot_mat(angle)

    return phi_mat[:,traj.state]

# return the derivatie of the diabatic-adiabatic transformation matrix
def dphi(traj):
    """Returns the derivative transformation matrix using the rotation angle. 
       Should be indentical to the dat_mat in the vibronic interface"""

    # can also run the trivial case of a single state
    if traj.nstates == 1:
        return np.zeros(traj.dim, dtype=float)

    angle    = theta(traj)
    dangle   = dtheta(traj)
    dphi_mat = drot_mat(angle)
 
    dphi_dq = np.array([dphi_mat[i,traj.state]*dangle for i in range(traj.dim)])

    return dphi_dq

def elec_overlap(traj1, traj2):
    """ Returns < Psi | Psi' >, the overlap integral of two trajectories"""

    return complex( np.dot(phi(traj1),phi(traj2)), 0.)

# returns the overlap between two trajectories (differs from s_integral in that
# the bra and ket functions for the s_integral may be different
# (i.e. pseudospectral/collocation methods). 
def traj_overlap(traj1, traj2, nuc_only=False):
    """ Returns < Psi | Psi' >, the overlap integral of two trajectories"""
    nuc_ovrlp = nuclear.overlap(traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                                traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    if nuc_only:
        return nuc_ovrlp
    else:
        return nuc_ovrlp * elec_overlap(traj1, traj2)

# total overlap of trajectory basis function
def s_integral(traj1, traj2, centroid=None, nuc_only=False, Snuc=None):
    """ Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""
    if Snuc is None:
        return traj_overlap(traj1, traj2, nuc_only=nuc_only)

    if nuc_only:
        return Snuc
    else:
        return Snuc * elec_overlap(traj1, traj2) 

#
def v_integral(traj1, traj2, centroid=None, Snuc=None):
    """Returns potential coupling matrix element between two trajectories."""
    # evaluate just the nuclear component (for re-use)
    if Snuc is None:
        Snuc = nuclear.overlap(traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                               traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    # get the linear combinations corresponding to the adiabatic states
    nst   = traj1.nstates
    h_mat = np.zeros((nst,nst),dtype=complex)

    # roll through terms in the hamiltonian
    for i in range(ham.nterms):
        s1    = ham.stalbl[i,0] - 1
        s2    = ham.stalbl[i,1] - 1
      
        # adiabatic states in diabatic basis -- cross terms between orthogonal
        # diabatic states are zero
        v_term = complex(1.,0.) * ham.coe[i]
        for q in range(ham.nmode_active):
            if ham.order[i,q] > 0:
                v_term *=  nuclear.prim_v_integral(ham.order[i,q],
                           traj1.widths()[q],traj1.x()[q],traj1.p()[q],
                           traj2.widths()[q],traj2.x()[q],traj2.p()[q])            
        h_mat[s1,s2] += v_term
        if s1 != s2:
            h_mat[s2,s1] += v_term 
      
    return np.dot(np.dot(phi(traj1), h_mat*Snuc), phi(traj2))

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
   
    return -sum( ke * vibronic.kecoeff ) * Selec

    
# time derivative of the overlap
def sdot_integral(traj1, traj2, centroid=None, Snuc=None, e_only=False, nuc_only=False):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if Snuc is None:
        Snuc = nuclear.overlap(traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                               traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    # overlap of electronic functions
    Selec = elec_overlap(traj1, traj2)

    # < chi | d / dx | chi'>
    deldx = nuclear.deldx(Snuc,traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                               traj2.phase(),traj2.widths(),traj2.x(),traj2.p())
    # < chi | d / dp | chi'>
    deldp = nuclear.deldp(Snuc,traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                               traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    # the nuclear contribution to the sdot matrix
    sdot = Selec * (np.dot(deldx, traj2.velocity()) 
                  + np.dot(deldp, traj2.force()) 
                  + 1j * traj2.phase_dot() * Snuc)

    if nuc_only:
        return sdot

    phi1  = phi(traj1)
    dphi2 = dphi(traj2)

    # the derivative coupling
    deriv_coup = np.array([np.dot(phi1, dphi2[:,q]) for q in range(traj2.dim)])
    e_coup     = np.dot(deriv_coup, traj2.velocity()) * Snuc

    if e_only:
        return e_coup

    sdot += e_coup

    return sdot

