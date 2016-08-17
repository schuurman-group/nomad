cdef extern from "math.h":
    double sqrt(double)
cdef extern from "complex.h":
    double complex cexp(double complex)
cdef extern from "complex.h":
    double complex I

#
# Returns the overlap of two primitive Gaussian functions.
#
def overlap(double x1, double p1, double a1,
            double x2, double p2, double a2):
    cdef double dx, dp, real_part, prefactor, x_center, imag_part
    cdef double complex S
    dx        = x1 - x2
    dp        = p1 - p2
    real_part = (a1*a2*dx**2 + 0.25*dp**2) / (a1 + a2)
    prefactor = sqrt(2. * sqrt(a1 * a2) / (a1 + a2))
    x_center  = (a1 * x1 + a2 * x2) / (a1 + a2)
    imag_part = (p1 * x1 - p2 * x2) - x_center * dp
    S         = prefactor * cexp(-real_part + I * imag_part)
    return S

#
# Returns the prefactor value for the overlap between two primitive
# Gaussian functions.
#
def overlap_prefactor(double g1, double g2):
    cdef double complex prefactor
    prefactor = cexp(I * (g2 - g1))
    return prefactor

#
# Returns the d/dp matrix element between two primitive Gaussians
# divided by the overlap.
#
def deldp(double x1, double p1, double a1,
          double x2, double p2, double a2):
    cdef double dx, dp
    cdef double complex dpval
    dx        = x1 - x2
    dp        = p1 - p2
    dpval     = (dp + 2. * I * a1 * dx) / (2. * (a1 + a2))
    return dpval

#
# Returns the d/dx matrix element between two primitive Gaussians
# divided by the overlap.
#
def deldx(double x1, double p1, double a1,
          double x2, double p2, double a2):
    cdef double dx, psum
    cdef double complex dxval
    dx    = x1 - x2
    psum  = a1 * p2 + a2 * p1
    dxval = (I * psum - 2. * a1 * a2 * dx) / (a1 + a2)
    return dxval

#
# Returns the d^2/dx^2 matrix element between two primitive
# Gaussians divided by the overlap.
#
def deld2x(double x1, double p1, double a1,
           double x2, double p2, double a2):
     cdef double dx, psum
     cdef double complex d2xval
     dx     = x1 - x2
     psum   = a1 * p2 + a2 * p1
     d2xval = -(I * 4. * a1 * a2 * dx * psum + 2. * a1 * a2 * (a1 + a2) - \
                4. * dx**2 * a1**2 * a2**2 + psum**2) / (a1 + a2)**2
     return d2xval

