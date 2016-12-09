"""
Mathematical functions for matrix elements between a Dirac delta
function and a primitive Gaussian function.

This module is also written in Cython. For optimal performance, compile
the Cython module using the directions in the README.
"""
import math
import operator, functools
import numpy as np

#@timings.timed
def overlap(t1, t2):
    """Returns the overlap of a Dirac delta function and a primitive
    Gaussian function."""
    S         = np.exp( 1j * t2.gamma )
    x1        = t1.x()
    a2        = t2.widths()
    x2        = t2.x()
    p2        = t2.p()
    dx        = x1 - x2
    real_part = a2 * dx**2
    imag_part = p2 * dx
    prefactor = (2. * a2 / math.pi)**(1./4.)
    S *= (functools.reduce(operator.mul,prefactor) *
         np.exp(sum(-real_part + 1j*imag_part)))
    return S

#@timings.timed
def deldp(t1, t2, S=None):
    """Returns the d/dp[i] matrix element between a Dirac delta function
    and a primitive Gaussian. Returns a vector for each p[i]"""
    if S is None:
        S = overlap(t1,t2)
    x1    = t1.x()
    x2    = t2.x()
    dx    = x1 - x2
    dpval = 1j * dx
    return dpval * S

#@timings.timed
def deldx(t1, t2, S=None):
    """Returns the d/dx[i] matrix element between a Dirac delta function
    and a primitive Gaussian. Returns a vector for each x[i]"""
    if S is None:
        S = overlap(t1,t2)
    x1    = t1.x()
    a2    = t2.widths()
    x2    = t2.x()
    p2    = t2.p()
    dx    = x1 - x2
    dxval = (-2. * a2 * dx - 1j * p2)
    return dxval * S

#@timings.timed
def deld2x(t1, t2, S=None):
    """Returns the d^2/dx[i]^2 matrix element between a Dirac delta function
    and a primitive Gaussian. Returns a vector for each x[i]"""
    if S is None:
        S = overlap(t1,t2)
    x1    = t1.x()
    a2    = t2.widths()
    x2    = t2.x()
    p2    = t2.p()
    dx     = x1 - x2
    d2xval = (2. * a2 + (2. * a2 * dx + 1j * p2)**2)
    return d2xval * S

