"""
Script for testing and evaluating exact integration of Gaussian basis
functions on LVC model two-state surfaces.

LVC Hamiltonian:

    H = (W - D'*m*D)*I - Z*sigma_z + X*sigma_x 

    D = (d/dq_1, d/dq_2, ...)
    W = 0.5 * q' * ww * q + q' * w + ew
    Z = q' * z + ez
    X = q' * x + ex

Gaussian basis functions:

    g_k = exp(-q' * alpha * q + q' * b_k + i*c_k)

Hamiltonian Parameters
-----------------------
m : (N, N) array_like
    The scaling matrix of kinetic energy, related to the frequencies.
ww : (N, N) array_like
    The scaling factors for I elements of H that depend on q**2.
w : (N,) array_like
    The scaling factors for I elements of H that depend on q.
ew : float
    The scalar portion for I elements of H.
z : (N,) array_like
    The scaling factors for sigma_z elements of H that depend on q.
ez : float
    The scalar portion for sigma_z elements of H.
x : (N,) array_like
    The scaling factors for sigma_x elements of H that depend on q.
ex : float
    The scalar portion for sigma_x elements of H.

Gaussian Parameters
--------------------
alpha : (N, N) array_like
    The scaling matrix of q**2 elements of Gaussians, related to the
    widths (assumed independent of basis function).
bk : (N,) array_like
    The scaling factors of elements of Gaussian basis function k
    proportional to q.
bl : (N,) array_like
    The scaling factors of elements of Gaussian basis function l
    proportional to q.
ck : (N,) array_like
    The scalar portion of Gaussian basis function k.
cl : (N,) array_like
    The scalar portion of Gaussian basis function l.

-----------------------------------------

These parameters will be unused for standard LVC formulations,
but are included to maintain generality.

aa : (N, N) array_like
    Polynomial expansion term proportional to q**2.
a : (N,) array_like
    Polynomial expansion term proportional to q.
ea : float
    Scalar polynomial expansion term.
"""
import sys as sys
import math as math
import numpy as np
import os.path as path
import scipy as scipy
import scipy.integrate as integrate
import nomad.core.glbl as glbl
import nomad.core.log as log
import nomad.common.constants as constants
import nomad.interfaces.vibronic as vibronic
import nomad.compiled.nuclear_gaussian as nuclear

# Determines the Hamiltonian symmetry
hermitian = False 

# Returns functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'

# Let propagator know if we need data at centroids to propagate
require_centroids = False

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

# generalized second order parameters for
# diagonal elements (currently not used)
a_mat = None
a_vec = None
ea    = None

# guassian parameters
m_mat = None
alpha = None

# regularization parameter delta
reg_delta = 0.

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
    global a_mat, a_vec, ea
    global m_mat, alpha
    global reg_delta

    # set the regularization parameter
    reg_delta = float(glbl.vibronic['dboc_delta']) 

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

    # don't need these right now, but may in the future (general
    # off-diagonal coefficients). Just initialize to zero for time-
    # being
    a_mat = np.zeros((nc, nc), dtype=float)
    a_vec = np.zeros(nc, dtype=float)
    ea    = 0. 

    # shift the LVC parameters to the MECI coordinates
    qci = ci_coord_shift(w_mat, w_vec, z_vec, x_vec, ez, ex)
    #print('qci='+str(qci))

    w_vec, ew = shift_lvc_params(qci, mat=0.5*w_mat, vec=w_vec, con=ew) 
    a_vec, ea = shift_lvc_params(qci, mat=a_mat, vec=a_vec, con=ea)      
    z_vec, ez = shift_lvc_params(qci, mat=None,  vec=z_vec, con=ez)
    x_vec, ex = shift_lvc_params(qci, mat=None,  vec=x_vec, con=ex)
    #print('shifted w_vec='+str(w_vec))
    #print('x_vec='+str(x_vec))
    #print('z_vec='+str(z_vec))

    # confirm that ez and ex are now zero:
    if abs(ez)>1.e-16 or abs(ex)>1.e-16:
        sys.exit('ERROR: CI shift process failed: qci='+str(qci))

    # initialize variables related to Gaussian basis functions
    # coefficient on kinetic energy contribution
    #m_mat = np.diag(vibronic.ham.freq)
    m_mat = np.identity(nc, dtype=float)
    # gaussian widths 
    alpha = np.diag(2 * glbl.properties['crd_widths'])    
    #print('alpha='+str(alpha))

    # determine shifted branching space parameters
    d_alpha, At = bspace_transform(z_vec, x_vec, alpha)
    b_alpha     = np.dot(At, np.column_stack((z_vec, x_vec)))

    pstr = '\n DBOC parameter should be << than sqrt(gh_alpha):\n' + \
           ' DBOC integral parameter: {0:.3e}\n' + \
           ' sqrt(gh_alpha):          {1:.3e}\n'
    log.print_message('string', [pstr.format(reg_delta, 
                                             np.sqrt(d_alpha[0]))])

    return

def nuc_overlap(t1, t2):
    """
    Returns
    -------
    float
        The Gaussian basis function overlap.
    """
    global qci, alpha

    bk, bl, ck, cl = extract_gauss_params(t1, t2)
    bk, bl, ck, cl = shift_gauss_params(qci, 0.5*alpha, bk, bl, ck, cl)

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

    #print('<'+str(t1.label)+'|'+str(t1.label)+'>='+str(olap))
    return olap

def elec_overlap(t1, t2):
    """return overlap of electronic functions. 1 if same state, 
       0 otherwise"""

    if t1.state == t2.state:
        return 1.+0.j
    else:
        return 0.+0.j

def traj_overlap(t1, t2, nuc_ovrlp=None, elec_ovrlp=None):
    """return the overlap of two trajectory functions -- a product
       of the nuclear and electronic factors"""

    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    if elec_ovrlp is None:
        elec_ovrlp = elec_overlap(t1, t2)

    return nuc_ovrlp * elec_ovrlp

def s_integral(t1, t2, nuc_ovrlp=None, elec_ovrlp=None):
    """the overlap matrix element -- this identical to the trajectory
       overlap"""

    return traj_overlap(t1, t2, nuc_ovrlp, elec_ovrlp)


def t_integral(t1, t2, nuc_ovrlp=None, elec_ovrlp=None):
    """the kinetic energy integral"""
    global qci, d_alpha, At
    global m_mat, w_mat, alpha
    global x_vec, z_vec

    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    # evaluate integrals and print results
    bk, bl, ck, cl = extract_gauss_params(t1, t2)
    bk, bl, ck, cl = shift_gauss_params(qci, 0.5*alpha, bk, bl, ck, cl)

    bkc   = bk.conj()
    db    = bkc - bl
    beta  = np.dot(At, bkc + bl)

    # nonadiabatic coupling matrix element
    # ------------------------------------
    J       = 0.5*(np.outer( z_vec, x_vec.conjugate())
                 - np.outer( x_vec, z_vec.conjugate()))
    D       =     (np.outer( x_vec, x_vec.conjugate())
                 + np.outer( z_vec, z_vec.conjugate()))

    p_alpha = np.dot(np.dot( At, J.T), np.dot(w_mat, db))

    pkl = 1.j * db / 2.
    qkl = scipy.linalg.solve(np.real(alpha), np.real(bkc+bl)) / 2
    qkk = scipy.linalg.solve(np.real(0.5*alpha), np.real(bk)) / 2
    qll = scipy.linalg.solve(np.real(0.5*alpha), np.real(bl)) / 2

    if glbl.vibronic['exact_spa']:
        nacme = eval_nac(qkl, pkl, D, w_mat @ J)    
    elif glbl.vibronic['exact_bat']:
        nacme = 0.5*(eval_nac(qkk, pkl, D, w_mat @ J) +
                     eval_nac(qll, pkl, D, w_mat @ J))
    else:
        nacme = exact_nac(beta, p_alpha)
    nacme *= nuc_ovrlp

    # DBOC matrix element
    # -------------------
    K       = np.dot(np.dot(J, w_mat), J.T)
    k_alpha = np.dot(np.dot(At, K), At.T.conj())

    if glbl.vibronic['inc_dboc']:
        if glbl.vibronic['exact_spa']:
            dbocme = eval_dboc(qkl, D, K)
        elif glbl.vibronic['exact_bat']:
            dbocme = 0.5 * (eval_dboc(qkk, D, K) + 
                            eval_dboc(qll, D, K))
        else:
            dbocme = exact_dboc(beta, k_alpha)
        dbocme *= nuc_ovrlp

    # kinetic energy only contributes to the diagonal
    if t1.state == t2.state:
        # nuclear kinetic energy
        t_int = 0.5 * nuc_ovrlp * (
           exact_poly(alpha, bk, bl, 
                      aa =  np.dot(np.dot(alpha, w_mat), alpha),
                      a  = -np.dot(np.dot(alpha, w_mat), bkc+bl),
                      ea =  np.dot(np.dot(bkc,   w_mat), bl)))
       
        # add diagonal component of NAC and DBOC
        if glbl.vibronic['inc_nuc_phase']:
            t_int += 0.5 * 1.j * nacme

        if glbl.vibronic['inc_dboc']:
            t_int += dbocme

    else:
        # add off-diagonal component of NAC and DBOC
        sigy = (t1.state - t2.state)*1.j

        # derivative coupling contribution
        t_int = sigy * nacme * 1.j

        if glbl.vibronic['inc_nuc_phase']:
            t_int *= 0.5
            
            if glbl.vibronic['inc_dboc']:
                t_int += sigy * dbocme

    return t_int


def sdot_integral(t1, t2, nuc_ovrlp=None, elec_ovrlp=None):
    """the time-derivative of the overlap"""
    global w_mat

    if elec_ovrlp is None:
        elec_ovrlp = elec_overlap(t1, t2)

    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    if abs(elec_ovrlp) == 0.:
        return elec_ovrlp

    deldx = nuclear.deldx(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                    t2.widths(),t2.x(),t2.p())
    deldp = nuclear.deldp(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                    t2.widths(),t2.x(),t2.p())

    vel   = np.diagonal(w_mat)*t2.p()*complex(1.,0.)
    force = lvc_force(t2)*complex(1.,0.)

    #print('t1 t2 deldx='+str(t1.label)+' '+str(t2.label)+' '+str(vel)+' '+str(deldx)+' '+str(np.dot(deldx,vel)))
    #print('t1 t2 deldp='+str(t1.label)+' '+str(t2.label)+' '+str(force)+' '+str(deldp)+' '+str(np.dot(deldp,force)))

    sdot = (np.dot(deldx, vel) +
            np.dot(deldp, force) +
            1j * t2.phase_dot() * nuc_ovrlp)

    return sdot

def lvc_energy(q, state):
    """evaluat the lvc energy at point q"""
    global w_mat, x_vec, z_vec

    ave_ener   = 0.5 * q @ w_mat @ q
    delta_ener = np.sqrt( (q @ x_vec)**2 + (q @ z_vec)**2 )
    sgn        = 2*state - 1

    return ave_ener + sgn * delta_ener


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
    """return the non-kinetic energy components of the Hamiltonian
       matrix"""
    global qci, d_alpha, At 
    global w_mat, w_vec, z_vec, x_vec, ew, ez, ex
    global a_mat, a_vec, ea
    global m_mat, alpha

    if elec_ovrlp is None:
        elec_ovrlp = elec_overlap(t1, t2)

    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    # potential matrix elements are diagonal, 
    # t1.state != t2.state yields 0
    olap  = nuc_ovrlp * elec_ovrlp
    if abs(olap) == 0.:
        return 0.j

    bk, bl, ck, cl = extract_gauss_params(t1, t2)
    bk, bl, ck, cl = shift_gauss_params(qci, 0.5*alpha, bk, bl, ck, cl)
    bkc            = bk.conj()
    beta           = np.dot(At, bkc + bl)

    qkl = scipy.linalg.solve(np.real(alpha), np.real(bkc+bl)) / 2
    qkk = scipy.linalg.solve(np.real(0.5*alpha), np.real(bk)) / 2
    qll = scipy.linalg.solve(np.real(0.5*alpha), np.real(bl)) / 2

    if glbl.vibronic['exact_spa']:
        v_int = lvc_energy(qkl, t1.state)
    elif glbl.vibronic['exact_bat']:
        v_int = 0.5*(lvc_energy(qkk, t1.state) + 
                     lvc_energy(qll, t1.state))

    else:
        # average potential energy
        w_me  = olap * exact_poly(alpha, bk, bl, 
                                  aa=0.5*w_mat, a=w_vec, ea=ew) 
        # energy difference term
        delta_me  = olap * exact_delta(beta)
        # the average energy contribution is state independent
        v_int = w_me

        # the energy shift contribution is -sigma_z
        sgn = int(-1 + 2 * t1.state)
        v_int += sgn * delta_me

    return v_int

def popwt(t1, t2, nuc_ovrlp=None):
    """returns the population weight in adiabatic basis"""

    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    pop = np.zeros(t1.nstates, dtype=complex)

    if t1.state == t2.state:
      pop[t1.state] = nuc_ovrlp

    return pop

def popwt_diabatic(t1, t2, nuc_ovrlp=None):
    """returns the population weights in diabatic basis"""

    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    bra = np.zeros(2, dtype=float)
    ket = np.zeros(2, dtype=float)
    bra[t1.state] = 1.
    ket[t2.state] = 1.

    bk, bl, ck, cl = extract_gauss_params(t1, t2)
    bk, bl, ck, cl = shift_gauss_params(qci, 0.5*alpha, bk, bl, ck, bl)

    bkc            = bk.conj()
    beta           = np.dot(At, bkc + bl)

    zme, xme = exact_pop(beta)
    sigmaz   = nuc_ovrlp * np.array([[zme, -xme], [-xme, -zme]], dtype=complex)
    za = np.dot( np.dot(bra, sigmaz), ket )

    return za 


#--------------------------------------------------------------------------
#
# Numerical integral routines
#
#--------------------------------------------------------------------------
def f_real(f, *args):
    """Returns the real component of a function with set parameters."""
    def func(u):
        return np.real(f(u, *args))

    return func


def f_imag(f, *args):
    """Returns the real component of a function with set parameters."""
    def func(u):
        return np.imag(f(u, *args))

    return func


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

def extract_gauss_params(t1, t2):
    """extract the parameters from the guassian basis functions"""

    bk = 2*t1.widths()*t1.x() + 1j*t1.p()
    # just assume the phase, gamma, is complex
    ck = -np.dot(t1.x(), t1.widths()*t1.x()) + (t1.phase()*t1.time - 
                                         np.dot(t1.x(), t1.p()))*1.j
 
    bl = 2*t2.widths()*t2.x() + 1j*t2.p()
    cl = -np.dot(t2.x(), t2.widths()*t2.x()) + (t2.phase()*t2.time - 
                                         np.dot(t2.x(), t2.p()))*1.j

    return bk, bl, ck, cl

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

def eval_nac(q, p, D, wJ):
    """
    evaluate the NAC at point (q,p)
    """
    return 2. * p @ wJ @ q / (q @ D @ q)


def exact_nac(beta, p):
    """
    Parameters
    ----------
    """
    global d_alpha

    etol = 1.e-10
    # if dx_alpha == 0, integral is 0
    if d_alpha[0] == 0.:
        return 0.j

    delta_a = d_alpha[1] - d_alpha[0]

    def integrand(u):
        d1    = 1 - d_alpha[0] * u**2
        d2    = 1 + delta_a * u**2
        dint  = u * eff_overlap(u, beta) / d1
        dint *=  p[0]*beta[0] + p[1]*beta[1] / d2
        return dint 

    def integrand_real(u):
        return scipy.real(integrand(u))

    def integrand_imag(u):
        return scipy.imag(integrand(u))

    ul = 1./np.sqrt(d_alpha[0])
    nac_int_real = integrate.quad(integrand_real, 0, ul, epsabs=etol)
    nac_int_imag = integrate.quad(integrand_imag, 0, ul, epsabs=etol)

    nac_int = nac_int_real[0] + nac_int_imag[0]*1.j
    nac_err = 0.5*(np.abs(nac_int_real[1]) + np.abs(nac_int_imag[1]))

    #npts=1000;
    #if printf:
    #    with open('/globalhome/schuurm/nac.dat', 'w') as f:
    #        for i in range(npts):
    #            u = i * ul / (npts-1)
    #            f.write('{:f} {:f} {:f}\n'.format(u, integrand_real(u), integrand_imag(u)))

    # only spout error message if tolerance exceed by factor of 10 
    if nac_err > 10*etol:
        print('WARNING: integral error tolerance exceeded in exact_nac:'
               +str(nac_err) +'>'+str(etol))

    return nac_int

#
def eval_dboc(q, D, K):
    """
    evaluate the DBOC at position q
    """

    return q @ K @ q / (q @ D @ q)**2


#
def exact_dboc(beta, k):
    """
    exact dboc integral
    """
    global d_alpha
    global reg_delta

    if d_alpha[0] == 0.:
        return 0.j

    etol     = 1.e-8
    [dx, dy] = d_alpha
    [bx, by] = beta
    kxx      = k[0, 0]
    kyy      = k[1, 1]
    kxy      = k[0, 1]
    dxy      = d_alpha[1] - d_alpha[0]
    bxy      = bx**2 + by**2

    # asymptotic limit
    Pcf  = kxx / ( dx*np.sqrt(dx) ) + kyy / ( dy**1.5 )
    Pexp = -(bx**2 + by**2) / 4.
    Plim = Pcf * np.exp( Pexp )

    # divergent integral
    dboc_div = 0.5 * np.exp(-bxy/4.) 
    dboc_div *= (kxx / np.sqrt(dy*dx**3) + kyy / np.sqrt(dx*dy**3)) 
    dboc_div *= (np.log(4*dx) - 2.*np.log(reg_delta) - np.euler_gamma)

    # convergent integral
    ul = 1./np.sqrt(dx)
    def term1(u):
        if np.isclose(u, ul):
            t1 = kxx * ((by**2 - 2)*dx + (bx**2 - 4)*dy) / (dx*np.sqrt(dy**3))
            t2 = kyy * ((by**2 - 6)*dx + dy*bx**2) / np.sqrt(dy)**5
            return -0.25*(t1 + t2)*np.exp(-bxy / 4)

        d1 = (1 - dx*u**2)
        d2 = (1 + dxy*u**2)
        t2 = (kxx + kyy / d2) * u**3 * eff_overlap(u, beta) / d1
        return (Plim - t2) / d1

    # second term
    def term2(u):
        if np.isclose(u, ul):
            t1 = kxx*bx**2 / (dx * np.sqrt(dy))
            t2 = dx*kyy*by**2 / np.sqrt(dy)**5
            t3 = 2*kxy*bx*by / np.sqrt(dy)**3
            return 0.5 * (t1 + t2 + t3) * np.exp(-bxy / 4)

        d1 = (1 - dx*u**2)
        d2 = (1 + dxy*u**2)
        t1 = kxx*bx**2 + kyy*by**2 / d2**2 + 2*kxy*bx*by / d2
        return 0.5 * t1 * u**3 * eff_overlap(u, beta) / d1

    term1_int_r = integrate.quad(f_real(term1), 0, ul, epsabs=etol)
    term1_int_i = integrate.quad(f_imag(term1), 0, ul, epsabs=etol)

    term2_int_r = integrate.quad(f_real(term2), 0, ul, epsabs=etol)
    term2_int_i = integrate.quad(f_imag(term2), 0, ul, epsabs=etol)

    dboc_int = term1_int_r[0] + term2_int_r[0] + \
              (term1_int_i[0] + term1_int_i[0])*1.j + \
              dboc_div

    dboc_err = 0.25 * (term1_int_r[1] + term2_int_r[1] +
                       term1_int_i[1] + term1_int_i[1])

    # only spout error message if tolerance exceed by factor of 10 
    if dboc_err > 10*etol:
        print('WARNING: integral error tolerance exceeded in exact_dboc:'
               +str(dboc_err) +'>'+str(etol))

    #print('dboc, div='+str(dboc_int)+','+str(dboc_div))
    return dboc_int


def exact_delta(beta):
    """
    Parameters
    ----------
    """
    global d_alpha

    etol = 1.e-10
    # if dx_alpha == 0, integral is 0
    if d_alpha[0] == 0.:
        return 0.j

    delta_a = d_alpha[1] - d_alpha[0]

    def integrand(u):
        d1    = 1 + delta_a * u**2
        d2    = 1 - d_alpha[0]  * u**2
        dint  = eff_overlap(u, beta)
        dint *= (d_alpha[0] + 
                 d_alpha[1]/d1 +
                (d_alpha[0]*beta[0]**2 + 
                 d_alpha[1]*beta[1]**2 / d1**2) * d2 / 2.)
        return dint / np.sqrt(d2)

    def integrand_real(u):
        return scipy.real(integrand(u))

    def integrand_imag(u):
        return scipy.imag(integrand(u))

    ul = 1./np.sqrt(d_alpha[0])
    delta_int_real = integrate.quad(integrand_real, 0, ul, epsabs=etol)
    delta_int_imag = integrate.quad(integrand_imag, 0, ul, epsabs=etol)

    delta_int = (delta_int_real[0] + delta_int_imag[0]*1.j) / np.sqrt(np.pi)
    delta_err = 0.5*(np.abs(delta_int_real[1]) + np.abs(delta_int_imag[1]))

    # only spout error message if tolerance exceed by factor of 10 
    if delta_err > 10*etol:
        print('WARNING: integral error tolerance exceeded in exact_delta:'
               +str(delta_err) +'>'+str(etol))

    return delta_int

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

    delta_a = d_alpha[1] - d_alpha[0]

    def integrand(u, zx):
        d1    = 1 + delta_a * u**2
        d2    = 1 - d_alpha[0]  * u**2
        dint  = eff_overlap(u, beta) / np.sqrt(d2)
        dint *= beta[0]*b_alpha[0,zx] + beta[1]*b_alpha[1,zx]/d1
        return dint

    def integrand_real(u, zx):
        return scipy.real(integrand(u, zx))

    def integrand_imag(u, zx):
        return scipy.imag(integrand(u,zx))

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
        print('WARNING: integral error tolerance exceeded in exact_delta:'
               +str(pop_err) +'>'+str(etol))

    return pop_z, pop_x


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

def shift_gauss_params(q_ci, f, bk, bl, ck, cl):
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
    new_bl, new_cl = shift_poly(q_ci, mat=-f, vec=bl, con=cl)
    return new_bk, new_bl, new_ck, new_cl



#----------------------------------------------------------------------------
#
# these routines involve time-independent quantities and are called one
# during initialization.
#
#----------------------------------------------------------------------------
#
# these routines involve time-independent quantities and are called one
# during initialization.
#
#-----------------------------------------------------------------------------
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

