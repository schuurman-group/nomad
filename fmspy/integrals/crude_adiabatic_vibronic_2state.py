"""
Compute saddle-point integrals over trajectories traveling on adiabataic
potentials

This currently uses first-order saddle point.
"""
import numpy as np
import fmspy.fmsio.glbl as glbl
import fmspy.integrals.nuclear_gaussian_ccs as nuclear

# Let FMS know if overlap matrix elements require PES info
overlap_requires_pes = True

# Let propagator know if we need data at centroids to propagate
require_centroids = False 

# Determines the Hamiltonian symmetry
hermitian = True 

# returns basis in which matrix elements are evaluated
basis = 'gaussian'

#cache previous values of theta, ensure continuity
theta_cache = dict()

# this is mostly to allow for consistent conventions: which
# diabat is the "excited" state, which is "ground" state.
gs = 0 
es = 1 

def elec_overlap(traj1, traj2):
    """ Returns < Psi | Psi' >, the overlap integral of two trajectories"""

    return complex( np.dot(phi(traj1),phi(traj2)), 0.)

# returns the overlap between two trajectories (differs from s_integral in that
# the bra and ket functions for the s_integral may be different
# (i.e. pseudospectral/collocation methods). 
def traj_overlap(traj1, traj2, nuc_only=False, Snuc=None):
    """ Returns < Psi | Psi' >, the overlap integral of two trajectories"""
    
    if Snuc is None:
        Snuc = nuclear.overlap(traj1.widths(),traj1.x(),traj1.p(),
                               traj2.widths(),traj2.x(),traj2.p())
    if nuc_only:
        return Snuc 
    else:
        return Snuc * elec_overlap(traj1, traj2)

# total overlap of trajectory basis function
def s_integral(traj1, traj2, centroid=None, nuc_only=False, Snuc=None):
    """ Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""
    if Snuc is None:
        Snuc = traj_overlap(traj1, traj2, nuc_only=nuc_only)

    if nuc_only:
        return Snuc
    else:
        return Snuc * elec_overlap(traj1, traj2) 

#
def v_integral(traj1, traj2, centroid=None, Snuc=None):
    """Returns potential coupling matrix element between two trajectories."""
    # evaluate just the nuclear component (for re-use)
    if Snuc is None:
        Snuc = nuclear.overlap(traj1.widths(),traj1.x(),traj1.p(),
                               traj2.widths(),traj2.x(),traj2.p())

    # get the linear combinations corresponding to the adiabatic states
    nst   = traj1.nstates
    v_mat = np.zeros((nst,nst),dtype=complex)

    # adiabatic states in diabatic basis -- cross terms between orthogonal
    # diabatic states are zero
    for i in range(glbl.pes.ham.nterms):
        [s1,s2] = glbl.pes.ham.stalbl[i,:]-1
        v_term = complex(1.,0.) * glbl.pes.ham.coe[i]
        for q in range(len(glbl.pes.ham.order[i])):
            qi      =  glbl.pes.ham.mode[i][q]
            v_term *=  nuclear.prim_v_integral(glbl.pes.ham.order[i][q],
                       traj1.widths()[qi],traj1.x()[qi],traj1.p()[qi],
                       traj2.widths()[qi],traj2.x()[qi],traj2.p()[qi])
        v_mat[s1,s2] += v_term

    # Fill in the upper-triangle
    v_mat += (v_mat.T - np.diag(v_mat.diagonal()))

    return np.dot(np.dot(phi(traj1), v_mat), phi(traj2)) * Snuc

# kinetic energy integral
def ke_integral(traj1, traj2, centroid=None, Snuc=None):
    """Returns kinetic energy integral over trajectories."""
    # evaluate just the nuclear component (for re-use)
    if Snuc is None:
        Snuc = nuclear.overlap(traj1.widths(),traj1.x(),traj1.p(),
                               traj2.widths(),traj2.x(),traj2.p())

    # < chi | del^2 / dx^2 | chi'> 
    ke = nuclear.deld2x(Snuc,traj1.widths(),traj1.x(),traj1.p(),
                             traj2.widths(),traj2.x(),traj2.p())

    return -np.dot(glbl.pes.kecoeff,ke)*elec_overlap(traj1,traj2)

    
# time derivative of the overlap
def sdot_integral(traj1, traj2, centroid=None, Snuc=None, e_only=False, nuc_only=False):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if Snuc is None:
        Snuc = nuclear.overlap(traj1.widths(),traj1.x(),traj1.p(),
                               traj2.widths(),traj2.x(),traj2.p())

    # overlap of electronic functions
    Selec = elec_overlap(traj1, traj2)

    # < chi | d / dx | chi'>
    deldx = nuclear.deldx(Snuc,traj1.widths(),traj1.x(),traj1.p(),
                               traj2.widths(),traj2.x(),traj2.p())
    # < chi | d / dp | chi'>
    deldp = nuclear.deldp(Snuc,traj1.widths(),traj1.x(),traj1.p(),
                               traj2.widths(),traj2.x(),traj2.p())

    # the nuclear contribution to the sdot matrix
    sdot = ( np.dot(deldx, traj2.velocity()) 
           + np.dot(deldp, traj2.force())) * Selec
								
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


# adiabatic-diabatic rotation matrix
def rot_mat(theta):
    """ Returns the adiabatic-diabatic rotation matrix for a given value of
        theta"""
    global gs, es

    if gs == 0:
        return np.array([[ np.cos(theta), -np.sin(theta)],
                         [ np.sin(theta),  np.cos(theta)]])
    else:
        return np.array([[-np.sin(theta), np.cos(theta)],
                         [ np.cos(theta), np.sin(theta)]])

# derivative of the adiabatic-diabatic rotation matrix
def drot_mat(theta):
    """ Returns the derivative adiabatic-diabatic rotation matrix with respect
        to theta"""
    global gs, es

    if gs == 0:
        return np.array([[-np.sin(theta), -np.cos(theta)],
                         [ np.cos(theta), -np.sin(theta)]])
    else:
        return np.array([[-np.cos(theta), -np.sin(theta)],
                         [-np.sin(theta),  np.cos(theta)]])

# compute the diabatic-adiabatic rotation angle theta
def theta(traj):
    """ Returns to the adiabatic-diabatic rotation angle theta. Choose theta
        to be consistent with diabatic-adiabatic transformation matrix, which
        itself is chosen to have a phase resulting in a slowly varying value of
        of theta."""
    global theta_cache, gs, es

    # can also run the trivial case of a single state
    if traj.nstates == 1:
        return 0.

    hmat    = traj.pes_data.diabat_pot
    h12     = hmat[0,1]
    de      = hmat[es,es]-hmat[gs,gs]

    if abs(de) < glbl.constants['fpzero']:
        sgn = np.sign(de)
        if sgn == 0.:
            sgn = 1
        de = sgn * glbl.constants['fpzero']
    ang     = 0.5*np.arctan2(2.*h12,de)

    # check the cached value and shift if necessary.
    pi_mult  = [0,-1.,1.]
    # if not in cache, return current value
    if traj.label in theta_cache:
        dif_vec  = [abs(ang + pi_mult[i]*np.pi - theta_cache[traj.label])
                    for i in range(len(pi_mult))]
        shft = dif_vec.index(min(dif_vec))
        if shft != 0:
            ang += pi_mult[shft]*np.pi

    theta_cache[traj.label] = ang

    return ang

# compute the derivative of the diabatic-adiabatic rotation angle theta
def dtheta(traj):
    """ Returns to the derivative adiabatic-diabatic rotation angle theta with
        respect to the internal coordinates."""
    global gs, es

    # can also run the trivial case of a single state
    if traj.nstates == 1:
        return np.zeros((traj.dim), dtype=float)

    hmat      = traj.pes_data.diabat_pot
    dhmat     = traj.pes_data.diabat_deriv
    h12       = hmat[0,1]
    de        = hmat[es,es] - hmat[gs,gs]
    if abs(de) < glbl.constants['fpzero']:
        sgn = np.sign(de)
        if sgn == 0.:
            sgn = 1
        de = sgn * glbl.constants['fpzero']

    arg       = 2. * h12 / de
    if abs(arg) < glbl.constants['fpzero']:
        sgn = np.sign(arg)
        if sgn == 0.:
            sgn = 1
        arg = sgn * glbl.constants['fpzero']

    dtheta_dq = np.array([((dhmat[q,0,1]*de - h12*(dhmat[q,es,es]-dhmat[q,gs,gs]))/de**2)/(1+arg**2)
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
    """Returns the derivative transformation matrix using the rotation angle."""

    # can also run the trivial case of a single state
    if traj.nstates == 1:
        return np.zeros(traj.dim, dtype=float)

    angle    = theta(traj)
    dangle   = dtheta(traj)
    dphi_mat = drot_mat(angle)

    dphi_dq = np.array([dphi_mat[i,traj.state]*dangle for i in range(traj.nstates)])

    return dphi_dq

