"""
Compute saddle-point integrals over trajectories traveling on adiabataic
potentials

This currently uses first-order saddle point.
"""
import numpy as np
import scipy as scipy
import scipy.integrate as integrate
import nomad.common.constants as constants
import nomad.core.glbl as glbl
import nomad.compiled.nuclear_gaussian_ccs as nuclear
import nomad.interfaces.vibronic as vibronic

# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the Hamiltonian symmetry
hermitian  = False

# returns basis in which matrix elements are evaluated
basis = 'gaussian'

#cache previous values of theta, ensure continuity
theta_cache = dict()

# position of the CI
qci   = None
d_alpha  = None
At    = None

# hamiltonian parameters 
b_alpha = None
w_mat = None
w_vec = None
z_vec = None
x_vec = None
ew    = None
ez    = None
ex    = None

# guassian parameters
m_mat = None
alpha = None

def initialize():
    """
    Returns 
    -------
    None

    Initialize the time-independent variables just once and save
    as global variables
    """
    global qci, d_alpha, b_alpha, At
    global w_mat, w_vec, z_vec, x_vec, ew, ez, ex
    global m_mat, alpha

    # initialize coordinate system variables    
    nc      = len(vibronic.ham.freq)
    # limit oursevles to 2 states, for now...
    nstates = 2

    # initialize variables related to LVC parameters
    kappa = np.zeros((nstates,nc), dtype=float)
    s_con = np.zeros(nstates, dtype=float)

    w_mat = np.zeros((nc,nc), dtype=float)
    w_vec = np.zeros(nc, dtype=float)
    z_vec = np.zeros(nc, dtype=float)
    x_vec = np.zeros(nc, dtype=float)
    ew = 0.
    ez = 0.
    ex = 0.

    for i in range(vibronic.ham.nterms):
        cf     = vibronic.ham.coe[i]
        # state labels run from 0
        states = vibronic.ham.stalbl[i] - 1
        modes  = vibronic.ham.mode[i]
        exp    = vibronic.ham.order[i]

        # get order of term
        ordr = sum(exp)

        if ordr == 0:
            if states[0] == states[1]:
                s_con[states[0]] = cf
            else:
                ex += cf

        elif ordr == 1:
            if states[0] == states[1]:
                kappa[states[0],modes[0]] = cf
            else:
                x_vec[modes[0]] = cf

        elif ordr == 2:
            if states[0] == states[1]:
                if(len(modes) > 1):
                    sys.exit('Error in LVC_exact: cannot currently'+
                             ' handle bi-linear second order terms')
                # we expect w_mat to equal omega, cf is 0.5*omega
                w_mat[modes[0], modes[0]] = 2.*cf
            else:
                sys.exit('Error in lvc_exact.py: exactly integration'+
                         ' currently limited to LVC models')

        else:
            sys.exit('Error in lvc_exact.py, ordr > 2: '+str(ordr))

    w_vec = 0.5*(kappa[0,:] + kappa[1,:])
    z_vec = 0.5*(kappa[0,:] - kappa[1,:])
    ew    = 0.5*(s_con[0]   + s_con[1])
    ez    = 0.5*(s_con[0]   - s_con[1])

    # shift the LVC parameters to the MECI coordinates
    qci = ci_coord_shift(w_mat, w_vec, z_vec, x_vec, ez, ex)

    w_vec, ew = shift_lvc_params(qci, mat=0.5*w_mat, vec=w_vec, con=ew)
    z_vec, ez = shift_lvc_params(qci, mat=None,  vec=z_vec, con=ez)
    x_vec, ex = shift_lvc_params(qci, mat=None,  vec=x_vec, con=ex)

    # confirm that ez and ex are now zero:
    if abs(ez)>1.e-16 or abs(ex)>1.e-16:
        sys.exit('ERROR: CI shift process failed: qci='+str(qci))

    # initialize variables related to Gaussian basis functions
    # coefficient on kinetic energy contribution
    m_mat = np.identity(nc, dtype=float)
    # gaussian widths 
    alpha = np.diag(2 * glbl.properties['crd_widths'])

    # determine shifted branching space parameters
    d_alpha, At = bspace_transform(z_vec, x_vec, alpha)
    b_alpha     = np.dot(At, np.column_stack((z_vec, x_vec)))

    return

def elec_overlap(t1, t2):
    """Returns < Psi | Psi' >, the electronic overlap integral of two trajectories"""
    return complex( np.dot(phi(t1), phi(t2)), 0.)

def nuc_overlap(t1, t2):
    """
    Returns
    -------
    float
        The Gaussian basis function overlap.
    """
    global alpha 

    bk, ck = extract_gauss_params(t1)
    bl, cl = extract_gauss_params(t2)

    bkl = bk.conjugate() + bl
    ckl = ck.conjugate() + cl

    abk  = scipy.linalg.solve(0.5*alpha.real, bk.real)
    abl  = scipy.linalg.solve(0.5*alpha.real, bl.real)
    abkl = scipy.linalg.solve(alpha, bkl)
    ainv = scipy.linalg.inv(alpha)

    oarg = np.dot(bkl, abkl)/4. + ckl

    Nk  = scipy.linalg.det( alpha.real / np.pi )**(0.25)
    Nk *= np.exp(-ck.real - np.dot(bk.real, abk)/4.)

    Nl  = scipy.linalg.det( alpha.real / np.pi )**(0.25)
    Nl *= np.exp(-cl.real - np.dot(bl.real, abl)/4.)

    olap = Nk*Nl * np.sqrt(scipy.linalg.det(ainv*np.pi)) * np.exp(oarg)

    return olap

def traj_overlap(traj1, traj2):
    """Returns < chi| chi' >, the nuclear overlap integral of two trajectories"""
    return elec_overlap(traj1, traj2) * nuc_overlap(traj1, traj2)


def s_integral(traj1, traj2, nuc_ovrlp=None, elec_ovrlp=None):
    """Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""
    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(traj1, traj2)

    if elec_ovrlp is None:
        elec_ovrlp = elec_overlap(traj1, traj2)

    return nuc_ovrlp * elec_ovrlp


def t_integral(t1, t2, nuc_ovrlp=None, elec_ovrlp=None):
    """Returns kinetic energy integral over trajectories."""
    # evaluate just the nuclear component (for re-use)
    global alpha, qci, w_mat

    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    if elec_ovrlp is None:
        elec_ovrlp = elec_overlap(t1, t2)

    bk, ck = extract_gauss_params(t1)
    bk, ck = shift_gauss_params(qci, 0.5*alpha, bk, ck)

    bl, cl = extract_gauss_params(t2)
    bl, cl = shift_gauss_params(qci, 0.5*alpha, bl, cl)

    bkc = bk.conjugate()

    # nuclear kinetic energy
    t_nuc = 0.5 * nuc_ovrlp * (
              exact_poly(alpha, bk, bl,
                 aa =  np.dot(np.dot(alpha, w_mat), alpha),
                 a  = -np.dot(np.dot(alpha, w_mat), bkc+bl),
                 ea =  np.dot(np.dot(bkc,   w_mat), bl)))

    t_int = np.dot( np.dot( phi(t1), t_nuc*np.identity(2),), phi(t2))

    return t_int


def sdot_integral(t1, t2, nuc_ovrlp=None, elec_ovrlp=None):
    """Returns kinetic energy integral over trajectories."""
    # evaluate just the nuclear component (for re-use)

    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    if elec_ovrlp is None:
        elec_ovrlp = elec_overlap(t1, t2)

    tderiv_nuc  = nuc_sdot(t1, t2, nuc_ovrlp, elec_ovrlp)
    tderiv_elec = elec_sdot(t1, t2, nuc_ovrlp)

    return tderiv_nuc + tderiv_elec


def nuc_sdot(t1, t2, nuc_ovrlp, elec_ovrlp):
    """Returns kinetic energy integral over trajectories."""
    # evaluate just the nuclear component (for re-use)

    # < chi | d / dx | chi'>
    deldx = nuclear.deldx(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                    t2.widths(),t2.x(),t2.p())
    # < chi | d / dp | chi'>
    deldp = nuclear.deldp(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                    t2.widths(),t2.x(),t2.p())

    vel   = np.diagonal(w_mat)*t2.p()*complex(1.,0.)
    force = lvc_force(t2)*complex(1.,0.)
    #print('t2, force='+str(t2.label)+' '+str(force))
    #if t1.label == 1 and t2.label == 0:
    #    deldp = np.conj(deldp)
    #print(' delp='+str(deldp))

    # the nuclear contribution to the sdot matrix
    sdot = ( np.dot(deldx, vel) +
             np.dot(deldp, force) +
             1j*t2.phase_dot()*nuc_ovrlp )*elec_ovrlp

    #print('t1, t2, deldx, deldp, phase_dot='+str(t1.label)+' '+str(t2.label)+' '+str(np.dot(deldx, vel))+' '+str(np.dot(deldp, force))+' '+str(1j*t2.phase_dot()*nuc_ovrlp))
    return sdot

def elec_sdot(t1, t2, nuc_ovrlp):
    """Returns the time derivative of the electronic functions"""

    vel   = np.diagonal(w_mat)*t2.p()*complex(1.,0.)

    # the electronic wave function in diabatic basis
    phi1  = phi(t1)
    # derivative of the mixing angle wrt coordinates
    dphi2 = dphi(t2)

    # the derivative coupling, dphi2.T (nstate,ncrd)
    tderiv_coup = np.dot(phi1, dphi2.T)

    #print('tderiv_coup, vel='+str(tderiv_coup)+','+str(vel))
    e_coup = np.dot(tderiv_coup, vel) * nuc_ovrlp

    #print('e_coup='+str(e_coup))
    return e_coup


def lvc_force(traj):
    """return the LVC gradient at the current trajectory geometry"""
    global w_vec, w_vec, z_vec, x_vec, ew
    global qci, w_mat

    qt = traj.x() - qci

    ave_grad  = np.dot(w_mat, qt) + w_vec

    # state specific component of the force
    z1   = np.dot(qt, z_vec)
    x1   = np.dot(qt, x_vec)

    denom   = np.sqrt( z1**2 + x1**2 )
    if abs(denom) == 0.:
        ss_grad = 0.
    else:
        ss_grad = (z_vec*z1 + x_vec*x1) / denom

    sgn = int(-1 + 2*traj.state)
    return -(ave_grad + sgn*ss_grad)


def v_integral(t1, t2, nuc_ovrlp=None, elec_ovrlp=None):
    """Returns potential coupling matrix element between two trajectories."""
    global qci, alpha, At
    global w_mat, w_vec, z_vec, x_vec, ew, ez, ex

    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    if elec_ovrlp is None:
        elec_ovrlp = elec_overlap(t1, t2)

    # potential matrix elements are diagonal, 
    # t1.state != t2.state yields 0
    if t1.state != t2.state:
        return 0.j 

    bk, ck = extract_gauss_params(t1)
    bk, ck = shift_gauss_params(qci, 0.5*alpha, bk, ck)

    bl, cl = extract_gauss_params(t2)
    bl, cl = shift_gauss_params(qci, 0.5*alpha, bl, cl)

    bkc            = bk.conj()
    beta           = np.dot(At, bkc + bl)

    # value using rotated potential matrix
    #w_me     = exact_poly(alpha, bk, bl, aa=0.5*w_mat, a=w_vec, ea=ew)
    #delta_me = exact_delta(beta)
    #v_int    = w_me + int(-1 + 2*t1.state) * delta_me
    #v_int   *= elec_ovrlp * nuc_ovrlp
    #print('v_int, rotated = '+str(v_int))

    # value using diabatic Hamiltonian
    v_mat = np.zeros((2,2), dtype=complex)
    wme   = exact_poly(alpha, bk, bl, aa=0.5*w_mat, a=w_vec, ea=ew)
    zme   = exact_poly(alpha, bk, bl, aa=None, a=z_vec, ea=ez)
    xme   = exact_poly(alpha, bk, bl, aa=None, a=x_vec, ea=ex)

    v_mat[0,0] = wme + zme
    v_mat[0,1] = xme
    v_mat[1,0] = xme
    v_mat[1,1] = wme - zme
    v_int      = np.dot(np.dot( phi(t1).conjugate(), v_mat ), phi(t2))

    return v_int

def popwt(t1, t2, nuc_ovrlp=None):
    """returns the population weight for each adiabatic state"""

    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    bk, ck = extract_gauss_params(t1)
    bk, ck = shift_gauss_params(qci, 0.5*alpha, bk, ck)

    bl, cl = extract_gauss_params(t2)
    bl, cl = shift_gauss_params(qci, 0.5*alpha, bl, cl)

    bkc   = bk.conj()
    beta  = np.dot(At, bkc + bl)
    #print('bk,bl='+str(bk)+','+str(bl))

    zme, xme = exact_pop(beta)    
    sigmaz   = nuc_ovrlp * np.array([[zme, -xme], [-xme, -zme]], dtype=complex)
    za = np.dot( np.conj(phi(t1)).dot(sigmaz), phi(t2) )
    return za 

#---------------------------------------------------------------------------------

def eff_overlap(u, beta):
    """evaluate the effective overlap for change-of-variable
       integration. 'grd' is a numpy array of length n. 
       If n == 1, a scalar is returned, else a numpy of length n

    Parameters
    ----------
    grd: (*,) array_like
        The points at which to evaluate the overlap
    dp : (2,) array_like
        The gradient terms of the integral transformed into the branching
        space.
    gp : (2,) array_like
        Branching space magnitudes.
    bp : (2,) array_like
        Linear terms of the Gaussian basis functions transformed into the
        branching space.
    """
    global d_alpha

    delta_a = d_alpha[1] - d_alpha[0]
    denom   = 1 + delta_a * u**2

    fac  = (1 - d_alpha[0] * u**2) / np.sqrt(denom)
    earg = -0.25 * u**2 * (d_alpha[0]*beta[0]**2 +
                           d_alpha[1]*beta[1]**2 / denom)

    return fac * np.exp(earg)


def exact_poly(alpha, bk, bl, aa=None, a=None, ea=0):
    """Evaluates analytic integrals of 2nd order polynomials.

    Returns
    -------
    float
        The integration result divided by the overlap.
    """
    bkl = bk.conjugate() + bl
    bf = scipy.linalg.solve(alpha, bkl)
    poly = ea

    #print('a-1 B='+str(bf))
    #print('Bt a-1='+str(np.dot(bkl.T, scipy.linalg.inv(alpha))))

    if aa is not None:
        ainv_aa = scipy.linalg.solve(alpha, aa)
        poly += np.trace(ainv_aa) / 2.
        poly += np.dot(bf, np.dot(aa, bf)) / 4.
    if a is not None:
        poly += np.dot(a, bf) / 2.

    return poly


def exact_pop(beta):
    """
    Parameters
    ----------
    """
    global d_alpha, b_alpha

    etol = 1.e-10

    # if dx_alpha == 0, integral is 0
    if d_alpha[0] == 0.:
        return 0.j

    #print('beta='+str(beta))
    #print('d_alpha=',d_alpha)
    #print('b_alpha(:,1)=',b_alpha[:,0])
    #print('b_alpha(:,2)=',b_alpha[:,1])

    delta_a = d_alpha[1] - d_alpha[0]

    def integrand(u, zx):
        d1    = 1 + delta_a * u**2
        d2    = 1 - d_alpha[0]  * u**2
        dint  = eff_overlap(u, beta) / np.sqrt(d2)
        dint *= (beta[0]*b_alpha[0,zx] + beta[1]*b_alpha[1,zx]/d1)
        return dint

    def integrand_real(u, zx):
        return scipy.real(integrand(u, zx))

    def integrand_imag(u, zx):
        return scipy.imag(integrand(u, zx))

    ul = 1./np.sqrt(d_alpha[0])

    pop_z_real = integrate.quad(integrand_real, 0, ul,
                          args=(0,), epsabs=etol)
    pop_z_imag = integrate.quad(integrand_imag, 0, ul,
                          args=(0,), epsabs=etol)
    pop_z = (pop_z_real[0] + pop_z_imag[0]*1.j) / np.sqrt(np.pi)
    pop_err = 0.5*(np.abs(pop_z_real[1]) + np.abs(pop_z_imag[1]))

    pop_x_real = integrate.quad(integrand_real, 0, ul,
                          args=(1,), epsabs=etol)
    pop_x_imag = integrate.quad(integrand_imag, 0, ul,
                          args=(1,), epsabs=etol)
    pop_x = (pop_x_real[0] + pop_x_imag[0]*1.j) / np.sqrt(np.pi)
    pop_err += 0.5*(np.abs(pop_x_real[1]) + np.abs(pop_x_imag[1]))

    # only spout error message if tolerance exceed by factor of 10 
    if pop_err > 100*etol:
        print('WARNING: integral error tolerance exceeded in exact_pop:'
               +str(pop_err) +'>'+str(etol))

    return pop_z, pop_x


def rot_mat(theta):
    """Returns the adiabatic-diabatic rotation matrix for a given value of
    theta"""
    return np.array([[ np.cos(theta),  -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]], dtype=float)

def drot_mat(theta):
    """Returns the derivative adiabatic-diabatic rotation matrix with respect
    to theta"""
    return np.array([[-np.sin(theta),  -np.cos(theta)],
                     [np.cos(theta), -np.sin(theta)]], dtype=float)


def theta(traj):
    """Returns to the adiabatic-diabatic rotation angle theta.
    
    Choose theta to be consistent with diabatic-adiabatic transformation
    matrix, which itself is chosen to have a phase resulting in a slowly
    varying value of of theta.
    """
    global theta_cache
    global qci, z_vec, x_vec 

    # can also run the trivial case of a single state
    if traj.nstates == 1:
        return 0.

    # shifted position
    qshft = traj.x() - qci
    X = np.dot(x_vec, qshft)
    Z = np.dot(z_vec, qshft)

    ang  = 0.5 * np.arctan2( X, Z )

    # check the cached value and shift if necessary.
    pi_mult  = np.array([0,-np.pi,np.pi])
    # if not in cache, return current value
    if traj.label in theta_cache:
        dif_vec  = np.absolute(ang + pi_mult - theta_cache[traj.label])
        shft     = np.argmin(dif_vec)
        if shft != 0:
            ang += pi_mult[shft]

    theta_cache[traj.label] = ang

    #print("traj="+str(traj.label)+" X/Z="+str(X/Z)+" theta="+str(ang)+"\n")

    return ang


def dtheta(traj):
    """Returns to the derivative adiabatic-diabatic rotation angle theta with
    respect to the internal coordinates."""
    global gs, es
    global qci, alpha, x_vec, z_vec

    # can also run the trivial case of a single state
    if traj.nstates == 1:
        return np.zeros((traj.dim), dtype=float)

    qshft  = traj.x() - qci

    X = np.dot(x_vec, qshft)
    Z = np.dot(z_vec, qshft)
    if Z == 0:
        Z = constants.fpzero

    arg = X / Z
    d_ang = 0.5 * (x_vec/Z - X*z_vec/Z**2) / (1 + arg**2)

    return d_ang


def phi(traj):
    """Returns the adiabatic state in a basis of diabatic states""" 
    # can also run the trivial case of a single state
    if traj.nstates == 1:
        return np.array([1.], dtype=float)

    angle  = theta(traj)
    rot    = rot_mat(angle)

    return rot[:, traj.state]


def dphi(traj):
    """Returns the derivative transformation matrix using the rotation angle."""
    # can also run the trivial case of a single state
    if traj.nstates == 1:
        return np.zeros(traj.dim, dtype=float)

    angle    = theta(traj)
    drot     = drot_mat(angle)
    dangle   = np.column_stack([dtheta(traj)]*traj.nstates)

    #print('dangle='+str(dangle))
    #print('dphi_mat = '+str(drot[:, traj.state]))
    #print('dphii= '+str(drot[:, traj.state]*dangle))

    # returns result that is ncrd x nstates
    return drot[:, traj.state]*dangle


#######################################################################################

def extract_gauss_params(t1):
    """extract the parameters from the guassian basis functions"""

    bk = 2*t1.widths()*t1.x() + 1j*t1.p()
    # just assume the phase, gamma, is complex
    ck = -np.dot(t1.x(), t1.widths()*t1.x()) + (t1.phase()*t1.time -
                                         np.dot(t1.x(), t1.p()))*1.j

    return bk, ck 

def bspace_transform(z, x, alpha):
    """Calculates the branching space magnitudes as well as scaling
    and rotation matrices.

    Returns
    ------
    (N,) ndarray
        The branching space magnitudes.
    (N, N) ndarray
        The transformation matrix for scaling and rotation.
    """
    ainv = scipy.linalg.inv(alpha)
    B    = np.column_stack((z, x))
    gh,U = scipy.linalg.eigh(np.dot(np.dot(B.T, ainv),B))

    # convention will be that dy > dx
    if gh[0] > gh[1]:
      gh[[0,1]]  = gh[[1,0]]
      U[:,[0,1]] = U[:,[1,0]]

    #print('Bmat='+str(B))
    #print('Uorig='+str(np.dot(np.dot(B.T, ainv),B)))
    #print('gh='+str(gh))
    #print('evecs='+str(U))

    dm12 = np.diag([1./np.sqrt(d_alpha)
                    if np.abs(d_alpha) > 1.e-16 else 0.
                    for d_alpha in gh])

    At   = np.dot(np.dot(dm12, U.T), np.dot(B.T, ainv))
    return gh, At

def ci_coord_shift(ww, w, z, x, ez, ex):
    """Find the coordinate shift to move the CI to the origin.

    Returns
    -------
    (N,) ndarray
        The position of the CI.
    """
    n  = len(ww)
    nt = n
    v  = w

    if np.linalg.norm(z) > 0.:
        nt += 1
        v  = np.hstack((v, ez))
    if np.linalg.norm(x) > 0.:
        nt += 1
        v  = np.hstack((v, ex))

    T_ci = np.zeros((nt, nt), dtype=float)
    T_ci[:n,:n] = ww

    nt = n-1
    if np.linalg.norm(z) > 0.:
        nt += 1
        T_ci[:n,nt] = z
        T_ci[nt,:n] = z.conjugate()

    if np.linalg.norm(x) > 0.:
        nt += 1
        T_ci[:n,nt] = x
        T_ci[nt,:n] = x.conjugate()

    # we assume the second order coefficients are 0.5*
    #print('T_ci='+str(T_ci),flush=True)

    return -scipy.linalg.solve(T_ci, v)[:n]

def shift_lvc_params(q_ci, mat=None, vec=None, con=None):
    """Shifts a set of LVC parameters such that the CI is at the origin.

    These values are constant and only need to be determined once.

    Parameters
    ----------
    q_ci : (N,) array_like
        The position of the CI to be shifted to the origin.

    Returns
    -------
    new_w : (N,) ndarray
        The shifted value of 1st order w contribution.
    new_ew : float
        The shifted value of ew.
    """
    new_vec, new_con = shift_poly(q_ci, mat, vec, con)
    return new_vec, new_con

#--------------------------------------------------------------------------
# 
# these routines involve simply transformations of time-dependent quantities,
# are called each time integrals are requested.
#
#-----------------------------------------------------------------------------
def shift_poly(q_shift, mat=None, vec=None, con=None):
    """Shifts position-dependent parameters to new values.

    Parameters
    ----------
    q_shift : (N,) array_like
        The shift in position.
    mat : (N, N) array_like
        The matrix of 2nd order expansion coefficients.
    vec : (N,) array_like
        The vector of 1st order expansion coefficients.
    sca : float
        The scalar expansion coefficient.

    Returns
    -------
    (N,) ndarray
        The shifted vector values, vec + 2*q_shift*mat.
    float
        The shifted scalar value, sca + q_shift*mat*q_shift + q_shift*vec.
    """

    new_con = con
    if mat is not None:
        new_vec = vec + 2*np.dot(mat, q_shift)
        new_con += np.dot(q_shift, np.dot(mat, q_shift)) + np.dot(q_shift, vec)
    else:
        new_vec = vec
        new_con += np.dot(q_shift, vec)

    return new_vec, new_con

def shift_gauss_params(q_ci, f, bk, ck):
    """Shifts a set of Gaussian parameters such that the CI is at the origin.

    These values need to be updated throughout a simulation.

    Parameters
    ----------
    q_ci : (N,) array_like
        The position of the CI to be shifted to the origin.

    Returns
    -------
    new_bk : (N,) ndarray
        The shifted value of bk.
    new_bl : (N,) ndarray
        The shifted value of bl.
    new_ck : float
        The shifted value of ck.
    new_cl : float
        The shifted value of cl.
    """
    new_bk, new_ck = shift_poly(q_ci, mat=-f, vec=bk, con=ck)
    return new_bk, new_ck

