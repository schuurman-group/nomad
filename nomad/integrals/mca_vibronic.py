"""
Compute saddle-point integrals over trajectories traveling on adiabataic
potentials

This currently uses first-order saddle point.
"""
import numpy as np
import nomad.math.constants as constants
import nomad.core.glbl as glbl
import nomad.compiled.nuclear_gaussian_ccs as nuclear
import nomad.compiled.vibronic_gaussian as vibronic

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

def v_integral(traj1, traj2, kecoef, nuc_ovrlp, elec_ovrlp):
    """Returns potential coupling matrix element between two trajectories."""

    # get the linear combinations corresponding to the adiabatic states
    nst   = traj1.nstates
    v_mat = np.zeros((nst,nst),dtype=complex)

    # adiabatic states in diabatic basis -- cross terms between orthogonal
    # diabatic states are zero
    for i in range(glbl.interface.ham.nterms):
        [s1,s2] = glbl.interface.ham.stalbl[i,:]-1
        v_term = complex(1.,0.) * glbl.interface.ham.coe[i]
        for q in range(len(glbl.interface.ham.order[i])):
            qi      =  glbl.interface.ham.mode[i][q]
            v_term *=  vibronic.qn_integral(glbl.interface.ham.order[i][q],
                       traj1.widths()[qi],traj1.x()[qi],traj1.p()[qi],
                       traj2.widths()[qi],traj2.x()[qi],traj2.p()[qi])
        v_mat[s1,s2] += v_term

    # Fill in the upper-triangle
    v_mat += (v_mat.T - np.diag(v_mat.diagonal()))

    return np.dot(np.dot(phi(traj1), v_mat), phi(traj2)) * nuc_ovrlp


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

def theta(traj):
    """ Returns to the adiabatic-diabatic rotation angle theta. Choose theta
        to be consistent with diabatic-adiabatic transformation matrix, which
        itself is chosen to have a phase resulting in a slowly varying value of
        of theta."""
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

#    print("traj="+str(traj.label)+" theta="+str(ang)+"\n")

    return ang

def phi(traj):
    """Returns the transformation matrix using the rotation angle.
       Should be indentical to the dat_mat in the vibronic interface"""

    # can also run the trivial case of a single state
    if traj.nstates == 1:
        return np.array([1.], dtype=float)

    angle   = theta(traj)
    phi_mat = rot_mat(angle)

    return phi_mat[:,traj.state]

