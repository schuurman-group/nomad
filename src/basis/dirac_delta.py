"""
Mathematical functions for matrix elements between a Dirac delta
function and a primitive Gaussian function.

This module is also written in Cython. For optimal performance, compile
the Cython module using the directions in the README.
"""
import math
import cmath


def overlap(x1, p1, a1, x2, p2, a2):
    """Returns the overlap of a Dirac delta function and a primitive
    Gaussian function."""
    dx        = x1 - x2
    real_part = a2 * dx**2
    imag_part = p2 * dx
    prefactor = (2. * a2 / math.pi)**(1./4.)
    S         = prefactor * cmath.exp(-real_part + 1j * imag_part)
    return S

def overlap_prefactor(g1, g2):
    """Returns the prefactor value for the overlap between a Dirac delta
    function and a primitive Gaussian function."""
    prefactor = cmath.exp(1j * g2)
    return prefactor

def deldp(x1, p1, a1, x2, p2, a2):
    """Returns the d/dp matrix element between a Dirac delta function
    and a primitive Gaussian divided by the overlap."""
    dx    = x1 - x2
    dpval = 1j * dx
    return dpval

def deldx(x1, p1, a1, x2, p2, a2):
    """Returns the d/dx matrix element between a Dirac delta function
    and a primitive Gaussian divided by the overlap."""
    dx    = x1 - x2
    dxval = (2. * a2 * dx - 1j * p2)
    return dxval

def deld2x(x1, p1, a1, x2, p2, a2):
    """Returns the d^2/dx^2 matrix element between a Dirac delta function
    and a primitive Gaussian divided by the overlap."""
    dx     = x1 - x2
    d2xval = (-2. * a2 + (-2. * a2 * dx + 1j * p2)**2)
    return d2xval
