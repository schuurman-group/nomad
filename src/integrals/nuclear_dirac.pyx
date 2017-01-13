"""
Mathematical functions for matrix elements between a Dirac delta
function and a primitive Gaussian function.

This module is also written in Cython. For optimal performance, compile
the Cython module using the directions in the README.
"""
cdef extern from "math.h":
    double sqrt(double)
cdef extern from "math.h":
    double M_PI
cdef extern from "complex.h":
    double complex cexp(double complex)
cdef extern from "complex.h":
    double complex I

cimport cython
import numpy as np

#@timings.timed
@cython.boundscheck(False)
def overlap(double[::1] x1, 
            double g2, 
            double[::1] a2, double[::1] x2, double[::1] p2):
    """Returns the overlap of a Dirac delta function and a primitive
    Gaussian function."""
    cdef double dx, real_part, imag_part
    cdef double complex S = 1.
    cdef int i
    cdef int n = x1.shape[0]

    real_part = 0.
    imag_part = 0.
    for i in range(n):
        real_part += a2[i] * (x1[i] - x2[i])**2
        imag_part += p2[i] * (x1[i] - x2[i])
        S         *= sqrt(sqrt((2. * a2[i] / M_PI)))
    S *= cexp(-real_part + I*imag_part +I*g2 )
    return S

#@timings.timed
@cython.boundscheck(False)
def deldp(double complex S,
          double[::1] x1,
          double g2, 
          double[::1] a2, double[::1] x2, double[::1] p2):
    """Returns the d/dp[i] matrix element between a Dirac delta function
    and a primitive Gaussian. Returns a vector for each p[i]"""
    cdef int i
    cdef int n = x1.shape[0]
    cdef double complex[::1] delpv = np.empty((n),dtype=np.complex)
   
    for i in range(n):
        delpv[i] = (x1[i] - x2[i]) * I * S
    return delpv 

#@timings.timed
@cython.boundscheck(False)
def deldx(double complex S,
          double[::1] x1,
          double g2, 
          double[::1] a2, double[::1] x2, double[::1] p2):
    """Returns the d/dx[i] matrix element between a Dirac delta function
    and a primitive Gaussian. Returns a vector for each x[i]"""
    cdef int i
    cdef int n = x1.shape[0]
    cdef double complex[::1] delxv = np.empty((n),dtype=np.complex)

    for i in range(n):
        delxv[i] = (2. * a2[i] * (x1[i] - x2[i]) -  I * p2[i]) * S
    return delxv 

#@timings.timed
@cython.boundscheck(False)
def deld2x(double complex S,
           double[::1] x1,
           double g2, 
           double[::1] a2, double[::1] x2, double[::1] p2):
    """Returns the d^2/dx[i]^2 matrix element between a Dirac delta function
    and a primitive Gaussian. Returns a vector for each x[i]"""
    cdef int i
    cdef int n = x1.shape[0]
    cdef double complex[::1] delx2v = np.empty((n),dtype=np.complex)

    for i in range(n):
        delx2v[i] = (-2. * a2[i] + 
                    (-2. * a2[i] * (x1[i] - x2[i]) + I * p2[i])**2) * S
    return delx2v

