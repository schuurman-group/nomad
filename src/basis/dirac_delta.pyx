cdef extern from "math.h":
    double sqrt(double)
cdef extern from "math.h":
    double M_PI
cdef extern from "complex.h":
    double complex cexp(double complex)
cdef extern from "complex.h":
    double complex I

#
# Returns the overlap of a Dirac delta function and a primitive
# Gaussian function.
#
def overlap(double x1, double p1, double a1,
            double x2, double p2, double a2):
    cdef double dx, real_part, prefactor, imag_part
    cdef double complex S
    dx        = x1 - x2
    real_part = a2 * dx**2
    imag_part = p2 * dx
    prefactor = (2. * a2 / M_PI)**(1./4.)
    S         = prefactor * cexp(-real_part + I * imag_part)
    return S

#
# Returns the prefactor value for the overlap between a Dirac delta
# function and a primitive Gaussian function.
#
def overlap_prefactor(double g1, double g2):
    cdef double complex prefactor
    prefactor = cexp(I * g2)
    return prefactor

#
# Returns the d/dp matrix element between a Dirac delta function
# and a primitive Gaussian divided by the overlap.
#
def deldp(double x1, double p1, double a1,
          double x2, double p2, double a2):
    cdef double dx
    cdef double complex dpval
    dx    = x1 - x2
    dpval = I * dx
    return dpval

#
# Returns the d/dx matrix element between a Dirac delta function
# and a primitive Gaussian divided by the overlap.
#
def deldx(double x1, double p1, double a1,
          double x2, double p2, double a2):
    cdef double dx
    cdef double complex dxval
    dx    = x1 - x2
    dxval = (2. * a2 * dx - I * p2)
    return dxval

#
# Returns the d^2/dx^2 matrix element between a Dirac delta function
# and a primitive Gaussian divided by the overlap.
#
def deld2x(double x1, double p1, double a1,
          double x2, double p2, double a2):
    cdef double dx
    cdef double complex d2xval
    dx     = x1 - x2
    d2xval = (-2. * a2 + (-2. * a2 * dx + I * p2)**2)
    return d2xval
