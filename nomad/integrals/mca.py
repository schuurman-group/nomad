"""
Compute saddle-point integrals over trajectories traveling on adiabataic
potentials

This currently uses first-order saddle point.
"""
import numpy as np
import nomad.math.constants as constants
import nomad.core.glbl as glbl
import nomad.compiled.nuclear_gaussian_ccs as nuclear


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
    """Returns < Psi | Psi' >, the electronic overlap integral of two trajectories"""
    return complex( np.dot(phi(traj1),phi(traj2)), 0.)


def nuc_overlap(traj1, traj2):
    """Returns < chi| chi' >, the nuclear overlap integral of two trajectories"""
    return nuclear.overlap(traj1.widths(),traj1.x(),traj1.p(),
                           traj2.widths(),traj2.x(),traj2.p())


def traj_overlap(traj1, traj2):
    """Returns < chi| chi' >, the nuclear overlap integral of two trajectories"""
    return elec_overlap(traj1, traj2) * nuc_overlap(traj1, traj2)


def s_integral(traj1, traj2, nuc_ovrlp, elec_ovrlp):
    """Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""
    return nuc_ovrlp * elec_ovrlp


def t_integral(traj1, traj2, kecoef, nuc_ovrlp, elec_ovrlp):
    """Returns kinetic energy integral over trajectories."""
    # evaluate just the nuclear component (for re-use)

    # < chi | del^2 / dx^2 | chi'>
    ke = nuclear.deld2x(nuc_ovrlp,traj1.widths(),traj1.x(),traj1.p(),
                                  traj2.widths(),traj2.x(),traj2.p())

    return -np.dot(kecoef,ke) * elec_ovrlp


def sdot_integral(traj1, traj2, nuc_ovrlp, elec_ovrlp):
    """Returns kinetic energy integral over trajectories."""
    # evaluate just the nuclear component (for re-use)

    # < chi | d / dx | chi'>
    deldx = nuclear.deldx(nuc_ovrlp,traj1.widths(),traj1.x(),traj1.p(),
                                    traj2.widths(),traj2.x(),traj2.p())
    # < chi | d / dp | chi'>
    deldp = nuclear.deldp(nuc_ovrlp,traj1.widths(),traj1.x(),traj1.p(),
                                    traj2.widths(),traj2.x(),traj2.p())

    # the nuclear contribution to the sdot matrix
    sdot = ( np.dot(deldx, traj2.velocity())
           + np.dot(deldp, traj2.force())) * elec_ovrlp

    phi1  = phi(traj1)
    dphi2 = dphi(traj2)

    # the derivative coupling
    deriv_coup = np.array([np.dot(phi1, dphi2[:,q]) for q in range(traj2.dim)])
    e_coup     = np.dot(deriv_coup, traj2.velocity()) * nuc_ovrlp

    return sdot + e_coup


def rot_mat(theta):
    """Returns the adiabatic-diabatic rotation matrix for a given value of
    theta"""
    global gs, es

    if gs == 0:
        return np.array([[ np.cos(theta), -np.sin(theta)],
                         [ np.sin(theta),  np.cos(theta)]])
    else:
        return np.array([[-np.sin(theta), np.cos(theta)],
                         [ np.cos(theta), np.sin(theta)]])


def drot_mat(theta):
    """Returns the derivative adiabatic-diabatic rotation matrix with respect
    to theta"""
    global gs, es

    if gs == 0:
        return np.array([[-np.sin(theta), -np.cos(theta)],
                         [ np.cos(theta), -np.sin(theta)]])
    else:
        return np.array([[-np.cos(theta), -np.sin(theta)],
                         [-np.sin(theta),  np.cos(theta)]])


def theta(traj):
    """Returns to the adiabatic-diabatic rotation angle theta.
    
    Choose theta to be consistent with diabatic-adiabatic transformation
    matrix, which itself is chosen to have a phase resulting in a slowly
    varying value of of theta.
    """
    global theta_cache, gs, es

    # can also run the trivial case of a single state
    if traj.nstates == 1:
        return 0.

    hmat    = traj.pes.get_data('diabat_pot')
    h12     = hmat[0,1]
    de      = hmat[es,es]-hmat[gs,gs]

    if abs(de) < constants.fpzero:
        sgn = np.sign(de)
        if sgn == 0.:
            sgn = 1
        de = sgn * constants.fpzero
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

    #print("traj="+str(traj.label)+" theta="+str(ang)+"\n")

    return ang


def dtheta(traj):
    """Returns to the derivative adiabatic-diabatic rotation angle theta with
    respect to the internal coordinates."""
    global gs, es

    # can also run the trivial case of a single state
    if traj.nstates == 1:
        return np.zeros((traj.dim), dtype=float)

    hmat      = traj.pes.get_data('diabat_pot')
    dhmat     = traj.pes.get_data('diabat_deriv')
    h12       = hmat[0,1]
    de        = hmat[es,es] - hmat[gs,gs]
    if abs(de) < constants.fpzero:
        sgn = np.sign(de)
        if sgn == 0.:
            sgn = 1
        de = sgn * constants.fpzero

    arg       = 2. * h12 / de
    if abs(arg) < constants.fpzero:
        sgn = np.sign(arg)
        if sgn == 0.:
            sgn = 1
        arg = sgn * constants.fpzero

    dtheta_dq = np.array([((dhmat[q,0,1]*de - h12*(dhmat[q,es,es]-dhmat[q,gs,gs]))/de**2)/(1+arg**2)
                        for q in range(traj.dim)])

    return dtheta_dq


def phi(traj):
    """Returns the transformation matrix using the rotation angle.
    Should be indentical to the dat_mat in the vibronic interface"""
    # can also run the trivial case of a single state
    if traj.nstates == 1:
        return np.array([1.], dtype=float)

    angle   = theta(traj)
    phi_mat = rot_mat(angle)

    return phi_mat[:,traj.state]


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
