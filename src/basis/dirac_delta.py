"""
Mathematical functions for matrix elements between a Dirac delta
function and a primitive gaussian.
"""
import math
import cmath


def overlap(x1,p1,a1,x2,p2,a2):
    """Returns the overlap of a Dirac delta function and a primitive
    gaussian.

    """
    dx          = x1 - x2    
    real_part   = a2 * dx**2
    imag_part   = p2 * dx
    prefactor   = (2.0 * a2 / math.pi)**(1./4.)
    S           = prefactor * cmath.exp(-real_part + complex(0.,1.) * imag_part)
    return S


def overlap_prefactor(g1, g2):

    prefactor = cmath.exp( complex(0., 1.) * g2 )

    return prefactor


def deldp(x1,p1,a1,x2,p2,a2):
    """Returns the del/dp matrix element between a Dirac delta
    function and a primitive gaussian."""
    dx          = x1 - x2
    dpval       = complex(0.,1.) * dx
    return dpval


def deldx(x1,p1,a1,x2,p2,a2):
    """Returns the del/dx matrix element between a Dirac delta
    function and a primitive gaussian."""
    dx          = x1 - x2
    dxval       = (2.* a2 * dx - complex(0.,1.) * p2)
    return dxval 
     

def deld2x(x1,p1,a1,x2,p2,a2):
    """Returns the del^2/dx^2 matrix element between a Dirac delta
    function and a primitive gaussian."""
    dx          = x1 - x2
    d2xval      = (-2. * a2 + (-2. * a2 * dx + complex(0.,1.) * p2)**2)
    return d2xval

    
