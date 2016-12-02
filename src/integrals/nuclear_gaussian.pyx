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
import operator, functools

#@timings.timed
cdef double complex overlap(double g1, 
                           double[:] a1, double[:] x1, double[:] p1,
                           double g2, 
                           double[:] a2, double[:] x2, double[:] p2):
    """Returns overlap of the nuclear component between two trajectories."""
    cdef double dx, dp, x_center, prefactor, real_part, imag_part
    cdef double complex S
    dx        = x1 - x2
    dp        = p1 - p2
    x_center  = (a1 * x1 + a2 * x2) / (a1 + a2)
    prefactor = sqrt(2. * sqrt(a1 * a2) / (a1 + a2))
    real_part = (a1*a2*dx**2 + 0.25*dp**2) / (a1 + a2)
    imag_part = (p1 * x1 - p2 * x2) - x_center * dp
    S         = (functools.reduce(operator.mul,prefactor) * 
                 cexp(sum(-real_part + I*imag_part) + I*(g2-g1)))
    return S

#@timings.timed
cdef double complex[:] deldp(double g1, 
                            double[:] a1, double[:] x1, double[:] p1,
                            double g2, 
                            double[:] a2, double[:] x2, double[:] p2,
                            double complex S=None):
    """Returns the del/dp matrix element between the nuclear component
       of two trajectories for each componet of 'p' (does not sum over terms)"""
    cdef double dx, dp
    if S is None:
        S = overlap(g1, a1, x1, p1, g2, a2, x2, p2)
    dx    = x1 - x2
    dp    = p1 - p2
    return (dp + 2. * I * a1 * dx) / (2. * (a1 + a2)) * S

#@timings.timed
cdef double complex[:] deldx(double g1, 
                            double[:] a1, double[:] x1, double[:] p1,
                            double g2, 
                            double[:] a2, double[:] x2, double[:] p2,
                            double complex S=None):
    """Returns the del/dx matrix element between the nuclear component
       of two trajectories for each componet of 'x' (does not sum over terms)"""
    cdef double dx, psum 
    if S is None:
        S = overlap(g1, a1, x1, p1, g2, a2, x2, p2)
    dx    = x1 - x2
    psum  = a1 * p2 + a2 * p1
    return (I * psum - 2. * a1 * a2 * dx) / (a1 + a2) * S

#@timings.timed
cdef double complex[:] deld2x(double g1, 
                             double[:] a1, double[:] x1, double[:] p1,
                             double g2, 
                             double[:] a2, double[:] x2, double[:] p2,
                             double complex S=None):
    """Returns the del^2/d^2x matrix element between the nuclear component
       of two trajectories for each componet of 'x' (does not sum over terms)"""
    cdef double dx, psum
    if S is None:
        S = overlap(g1, a1, x1, p1, g2, a2, x2, p2)
    dx     = x1 - x2
    psum   = a1 * p2 + a2 * p1
    d2xval = -(1j * 4. * a1 * a2 * dx * psum + 2. * a1 * a2 * (a1 + a2) -
               4. * dx**2 * a1**2 * a2**2 + psum**2) / (a1 + a2)**2
    return d2xval * S

