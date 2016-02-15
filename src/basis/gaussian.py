import math
import cmath
import numpy as np
#
# overlap of two gaussian functions
#
def overlap(x1,p1,a1,x2,p2,a2):
    dx        = x1 - x2
    dp        = p1 - p2
    real_part = (a1*a2*dx**2 + 0.25*dp**2) / (a1 + a2)
    prefactor = math.sqrt(2.0 * math.sqrt(a1 * a2) / (a1 + a2))
    x_center  = (a1 * x1 + a2 * x2) / (a1 + a2)
    imag_part = (p1 * x1 - p2 * x2) - x_center * dp
    S         = prefactor * cmath.exp(-real_part + np.complex(0.,1.) * imag_part)
    return S

#
# del/dp matrix element between two primitive gaussians
#
def deldp(x1,p1,a1,x2,p2,a2):
    dx        = x1 - x2
    dp        = p1 - p2
    dpval     = (dp + 2.*np.complex(0.,1.) * a1 * dx) / (2.0 * (a1 + a2))
    return dpval

#
# del/dx matrix element between two primitive gaussians
#
def deldx(x1,p1,a1,x2,p2,a2):
     dx    = x1 - x2
     psum  = a1 * p2 + a2 * p1
     dxval = (np.complex(0.,1.) * psum - 2.0 * a1 * a2 * dx) / (a1 + a2)
     return dxval

#
# del^2/dx^2 matrix element between two primitive gaussians
#
def deld2x(x1,p1,a1,x2,p2,a2):
     dx     = x1 - x2
     psum   = a1 * p2 + a2 * p1
     d2xval = -(+4. * a1 * a2 * dx * psum * np.complex(0.,1.) \
                +2. * a1 * a2 * (a1 + a2)                     \
                -4. * dx**2 * a1**2 * a2**2                   \
                +psum**2)                                     \
                / (a1 + a2)**2
     return d2xval

