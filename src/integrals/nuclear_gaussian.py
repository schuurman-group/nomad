"""
Computes matrix elements over the nuclear component of the trajectory
basis function, assumes bra is also product of frozen gaussians
"""
import operator, functools
import numpy as np
import utils.constants

print('WARNING: Using uncompiled Python module for Gaussian integrals. '
      'For optimal performance, compile Cython module using '
      '\'python setup.py build_ext --inplace\'.')

#@timings.timed
def overlap(gamma1, a1, x1, p1, gamma2, a2, x2, p2):
    """Returns overlap of the nuclear component between two trajectories."""
    S  = np.exp( 1j * (gamma2 - gamma1) )
    dx        = x1 - x2
    dp        = p1 - p2
    prefactor = np.sqrt(2. * np.sqrt(a1 * a2) / (a1 + a2))
    x_center  = (a1 * x1 + a2 * x2) / (a1 + a2)
    real_part = (a1*a2*dx**2 + 0.25*dp**2) / (a1 + a2)
    imag_part = (p1 * x1 - p2 * x2) - x_center * dp
    S *= (functools.reduce(operator.mul,prefactor) *
         np.exp(sum(-real_part + 1j*imag_part)))
    return S

#@timings.timed
def deldp(S, gamma1, a1, x1, p1, gamma2, a2, x2, p2):
    """Returns the del/dp matrix element between the nuclear component
       of two trajectories for each componet of 'p' (does not sum over terms)"""
    if S is None:
        S = overlap(gamma1, a1, x1, p1, gamma2, a2, x2, p2)
    dx    = x1 - x2
    dp    = p1 - p2
    dpval = (dp + 2. * 1j * a1 * dx) / (2. * (a1 + a2))
    return dpval * S

#@timings.timed
def deldx(S, gamma1, a1, x1, p1, gamma2, a2, x2, p2):
    """Returns the del/dx matrix element between the nuclear component
       of two trajectories for each componet of 'x' (does not sum over terms)"""
    if S is None:
        S = overlap(gamma1, a1, x1, p1, gamma2, a2, x2, p2)
    dx    = x1 - x2
    psum  = a1*p2 + a2*p1
    dxval = (2. * a1 * a2 * dx - 1j * psum) / (a1 + a2)
    return dxval * S

#@timings.timed
def deld2x(S, gamma1, a1, x1, p1, gamma2, a2, x2, p2):
    """Returns the del^2/d^2x matrix element between the nuclear component
       of two trajectories for each componet of 'x' (does not sum over terms)"""
    if S is None:
        S  = overlap(gamma1, a1, x1, p1, gamma2, a2, x2, p2)
    dx     = x1 - x2
    psum   = a1*p2 + a2*p1
    d2xval = -(1j * 4. * a1 * a2 * dx * psum + 2. * a1 * a2 * (a1 + a2) -
               4. * dx**2 * a1**2 * a2**2 + psum**2) / (a1 + a2)**2
    return d2xval * S

def prim_v_integral(N, a1, x1, p1, a2, x2, p2):
    """Returns the matrix element <cmplx_gaus(q,p)| q^N |cmplx_gaus(q,p)>
     -- up to an overlap integral --
    """
    # since range(N) runs up to N-1, add "1" to result of floor
    n_2 = int(np.floor(0.5 * N) + 1)
    a   = a1 + a2
    b   = complex(2.*(a1*x1 + a2*x2),-(p1-p2))

    if np.absolute(b) < constants.fpzero:
        if N % 2 != 0:
            return 0.
        else:
            n_2 -= 1 # get rid of extra "1" for range value
            v_int = (a**(-n_2))/np.math.factorial(n_2)
            return v_int * np.math.factorial(N) / 2.**N

    # generally these should be 1D harmonic oscillators. If
    # multi-dimensional, the final result is a direct product of
    # each dimension
    v_int = complex(0.,0.)
    for i in range(n_2):
        v_int += (a**(i-N) * b**(N-2*i) /
                 (np.math.factorial(i) * np.math.factorial(N-2*i)))

    # refer to appendix for derivation of these relations
    return v_int * np.math.factorial(N) / 2.**N

def ordr1_vec(a1, x1, p1, a2, x2, p2):
    a   = a1 + a2
    b   = complex(2.*(a1*x1 + a2*x2),-(p1-p2))
    v_int = b / (2 * a)
    
    return v_int

def ordr2_vec(a1, x1, p1, a2, x2, p2):
    a   = a1 + a2
    b   = complex(2.*(a1*x1 + a2*x2),-(p1-p2))
    v_int = 0.5 * (b**2 / (2 * a**2) + (1/a))

    return v_int


