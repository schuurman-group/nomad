"""
Computes matrix elements over the nuclear component of the trajectory
basis function, assumes bra is also product of frozen gaussians
"""
cdef extern from "math.h":
    double sqrt(double)
cdef extern from "complex.h":
    double complex cexp(double complex)
cdef extern from "complex.h":
    double complex I

import cython
import numpy as np

#@timings.timed
@cython.boundscheck(False)
def overlap(double g1,
            double[::1] a1, double[::1] x1, double[::1] p1,
            double g2,
            double[::1] a2, double[::1] x2, double[::1] p2):
    """Returns overlap of the nuclear component between two trajectories."""
    cdef double dx, dp, x_center, real_part, imag_part
    cdef double complex S = 1.
    cdef int i
    cdef int n = a1.shape[0]

    real_part = 0.
    imag_part = 0.
    for i in range(n):
        dx        = x1[i] - x2[i]
        dp        = p1[i] - p2[i]
        x_center  = (a1[i] * x1[i] + a2[i] * x2[i]) / (a1[i] + a2[i])
        real_part += (a1[i]*a2[i]*dx**2 + 0.25*dp**2) / (a1[i] + a2[i])
        imag_part += (p1[i]*x1[i] - p2[i]*x2[i]) - x_center * dp
        S         *= sqrt(2. * sqrt(a1[i] * a2[i]) / (a1[i] + a2[i]))

    S *= cexp( -real_part + I*imag_part +I*(g2-g1) )
    return S

#@timings.timed
@cython.boundscheck(False)
def deldp(double complex S,
          double g1,
          double[::1] a1, double[::1] x1, double[::1] p1,
          double g2,
          double[::1] a2, double[::1] x2, double[::1] p2):
    """Returns the del/dp matrix element between the nuclear component
       of two trajectories for each componet of 'p' (does not sum over terms)"""
    cdef int i
    cdef int n = a1.shape[0]
    cdef double dx, dp
    cdef double complex[::1] delpv = np.empty((n),dtype=np.complex)

    for i in range(n):
        dx       = x1[i] - x2[i]
        dp       = p1[i] - p2[i]
        delpv[i] = (dp + 2. * I * a1[i] * dx) / (2. * (a1[i] + a2[i])) * S

    return delpv

#@timings.timed
@cython.boundscheck(False)
def deldx(double complex S,
          double g1,
          double[::1] a1, double[::1] x1, double[::1] p1,
          double g2,
          double[::1] a2, double[::1] x2, double[::1] p2):
    """Returns the del/dx matrix element between the nuclear component
       of two trajectories for each componet of 'x' (does not sum over terms)"""
    cdef int i
    cdef int n = a1.shape[0]
    cdef double dx, psum
    cdef double complex[::1] delxv = np.empty((n),dtype=np.complex)

    for i in range(n):
        dx       = x1[i] - x2[i]
        psum     = a1[i] * p2[i] + a2[i] * p1[i]
        delxv[i] = (2. * a1[i] * a2[i] * dx - I * psum) / (a1[i] + a2[i]) * S

    return delxv

#@timings.timed
@cython.boundscheck(False)
def deld2x(double complex S,
           double g1,
           double[::1] a1, double[::1] x1, double[::1] p1,
           double g2,
           double[::1] a2, double[::1] x2, double[::1] p2):
    """Returns the del^2/d^2x matrix element between the nuclear component
       of two trajectories for each componet of 'x' (does not sum over terms)"""
    cdef int i
    cdef int n = a1.shape[0]
    cdef double dx, psum
    cdef double complex[::1] delx2v = np.empty((n),dtype=np.complex)

    for i in range(n):
        dx        = x1[i] - x2[i]
        psum      = a1[i] * p2[i] + a2[i] * p1[i]
        delx2v[i] = -S * (I*4.*a1[i]*a2[i]*dx*psum +
                          2.*a1[i]*a2[i]*(a1[i] + a2[i]) -
                          4.* dx**2 * a1[i]**2 * a2[i]**2 +
                          psum**2) / (a1[i] + a2[i])**2
    return delx2v

