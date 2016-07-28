"""
Mathematical functions for treating Gaussian basis functions using dirac delta integration.
"""
import math
import cmath


def overlap(x1,p1,a1,x2,p2,a2):
    """Returns the overlap of two gaussian basis functions by evaluating 
       the jth basis function at the center of the ith basis function."""
    dx          = x2 - x1
    real_part   = a1 * dx**2
    imag_part   = p2 * dx
    prefactor   = (2.0 * a2 / math.pi)**(1./4.)
    S           = prefactor * cmath.exp(-real_part + complex(0.,1.) * imag_part)
    return S


def deldp(x1,p1,a1,x2,p2,a2):
    """Returns the del/dp matrix element between two primitive gaussians."""
    dx          = x2 - x1
    dpval       = complex(0.,1.) * dx
    return dpval


def deldx(x1,p1,a1,x2,p2,a2):
    """Returns the del/dx matrix element between two primitive gaussians."""
    dx          = x2 - x1
    dxval       = (2.* a2 * dx - complex(0.,1.) * p2)
    return dxval 
     

def deld2x(x1,p1,a1,x2,p2,a2):
    """Returns the del^2/dx^2 matrix element between two primitive
    gaussians."""
    dx          = x2 - x1
    d2xval      = (-2. * a2 + (-2. * a2 * dx + complex(0.,1.) * p2)**2)
    return d2xval

    
