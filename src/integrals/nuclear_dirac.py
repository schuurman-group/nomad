"""
Mathematical functions for matrix elements between a Dirac delta
function and a primitive Gaussian function.

This module is also written in Cython. For optimal performance, compile
the Cython module using the directions in the README.
"""
import operator
import functools
import numpy as np

print('WARNING: Using uncompiled Python module for Dirac integrals. '
      'For optimal performance, compile Cython module using '
      '\'python setup.py build_ext --inplace\'.')

#@timings.timed
def overlap(gamma1, a1, x1, p1, gamma2, a2, x2, p2):
    """Returns the overlap of a Dirac delta function and a primitive
    Gaussian function."""
    S         = np.exp( 1j * gamma2 )
    dx        = x1 - x2
    real_part = a2 * dx**2
    imag_part = p2 * dx
    prefactor = (2. * a2 / np.pi)**(1./4.)
    S *= (functools.reduce(operator.mul, prefactor) *
          np.exp(sum(-real_part + 1j*imag_part)))
    return S

#@timings.timed
def deldp(S, gamma1, a1, x1, p1, gamma2, a2, x2, p2):
    """Returns the d/dp[i] matrix element between a Dirac delta function
    and a primitive Gaussian. Returns a vector for each p[i]"""
    dx    = x1 - x2
    dpval = 1j * dx
    return dpval * S

#@timings.timed
def deldx(S, gamma1, a1, x1, p1, gamma2, a2, x2, p2):
    """Returns the d/dx[i] matrix element between a Dirac delta function
    and a primitive Gaussian. Returns a vector for each x[i]"""
    dx    = x1 - x2
    dxval = (-2. * a2 * dx - 1j * p2)
    return dxval * S

#@timings.timed
def deld2x(S, gamma1, a1, x1, p1, gamma2, a2, x2, p2):
    """Returns the d^2/dx[i]^2 matrix element between a Dirac delta function
    and a primitive Gaussian. Returns a vector for each x[i]"""
    dx     = x1 - x2
    d2xval = (2. * a2 + (2. * a2 * dx + 1j * p2)**2)
    return d2xval * S
