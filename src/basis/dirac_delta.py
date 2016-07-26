import math
import cmath
import numpy as np
#
# overlap of two gaussian functions
#
def overlap(x1,p1,a1,x2,p2,a2):
    dx          = x1 - x2
    real_part   = a1 * dx**2
    imag_part   = p2 * dx
    prefactor   = (2.0 * a2 / np.pi)**(1./4.)
    S           = prefactor * np.exp(-real_part + np.complex(0.,1.) * imag_part)
    return S
#
# del/dp matrix element between two primitive gaussians
#
def deldp(x1,p1,a1,x2,p2,a2):
    dx          = x2 - x1
    dpval       = np.complex(0.,1.) * dx
    return dpval

# del/dx matrix element between two primitive gaussians
#
def deldx(x1,p1,a1,x2,p2,a2):
    dx          = x2 - x1
    dxval       = (2.* a2 * dx - np.complex(0.,1.) * p2)
    return dxval 
     
#
# del^2/dx^2 matrix element between two primitive gaussians
#
def deld2x(x1,p1,a1,x2,p2,a2):
    dx          = x2 - x1
    d2xval      = (-2. * a2 + (-2. * a2 * dx + np.complex(0.,1.) * p2)**2)
    return d2xval

    
