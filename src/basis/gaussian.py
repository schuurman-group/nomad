"""
Mathematical functions for treating Gaussian basis functions.

FMS Gaussian basis functions are of the form
g1(x) = (2a1 / pi)^(1/4) exp[-a1 (x - x1)^2 + i p1 (x - x1)]
where a1 is the width, x1 centre of the Gaussian position and p1 is
the centre of the Gaussian momentum.
"""
import math
import cmath


def overlap(x1, p1, a1, x2, p2, a2):
    """Returns the overlap of two primitive Gaussian functions."""
    dx        = x1 - x2
    dp        = p1 - p2
    real_part = (a1*a2*dx**2 + 0.25*dp**2) / (a1 + a2)
    prefactor = math.sqrt(2. * math.sqrt(a1 * a2) / (a1 + a2))
    x_center  = (a1 * x1 + a2 * x2) / (a1 + a2)
    imag_part = (p1 * x1 - p2 * x2) - x_center * dp
    S         = prefactor * cmath.exp(-real_part + 1j * imag_part)
    return S


def overlap_prefactor(g1, g2):
    """Returns the prefactor value for the overlap between two primitive
    Gaussian functions."""
    prefactor = cmath.exp(1j * (g2 - g1))
    return prefactor


def deldp(x1, p1, a1, x2, p2, a2):
    """Returns the d/dp matrix element between two primitive Gaussians
    divided by the overlap."""
    dx    = x1 - x2
    dp    = p1 - p2
    dpval = (dp + 2. * 1j * a1 * dx) / (2. * (a1 + a2))
    return dpval


def deldx(x1, p1, a1, x2, p2, a2):
    """Returns the d/dx matrix element between two primitive Gaussians
    divided by the overlap."""
    dx    = x1 - x2
    psum  = a1 * p2 + a2 * p1
    dxval = (1j * psum - 2. * a1 * a2 * dx) / (a1 + a2)
    return dxval


def deld2x(x1, p1, a1, x2, p2, a2):
    """Returns the d^2/dx^2 matrix element between two primitive
    Gaussians divided by the overlap."""
    dx     = x1 - x2
    psum   = a1 * p2 + a2 * p1
    d2xval = -(1j * 4. * a1 * a2 * dx * psum + 2. * a1 * a2 * (a1 + a2) -
               4. * dx**2 * a1**2 * a2**2 + psum**2) / (a1 + a2)**2
    return d2xval
