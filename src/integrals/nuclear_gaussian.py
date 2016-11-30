"""
Computes matrix elements over the nuclear component of the trajectory
basis function, assumes bra is also product of frozen gaussians
"""
import operator, functools
import numpy as np

#@timings.timed
def overlap(t1, t2):
    """Returns overlap of the nuclear component between two trajectories."""
    S  = np.exp( 1j * (t2.gamma - t1.gamma) )
    a1 = t1.widths()
    a2 = t2.widths()
    x1 = t1.x()
    x2 = t2.x()
    p1 = t1.p()
    p2 = t2.p()
    dx        = x1 - x2
    dp        = p1 - p2
    prefactor = np.sqrt(2. * np.sqrt(a1 * a2) / (a1 + a2))
    x_center  = (a1 * x1 + a2 * x2) / (a1 + a2)
    real_part = (a1*a2*dx**2 + 0.25*dp**2) / (a1 + a2)
    imag_part = (p1 * x1 - p2 * x2) - x_center * dp
    S *= (functools.reduce(operator.mul,prefactor) * 
         np.exp(sum(-real_part + 1j*imag_part)))
    return S

#@timings.timed
def deldp(t1, t2, S=None):
    """Returns the del/dp matrix element between the nuclear component
       of two trajectories for each componet of 'p' (does not sum over terms)"""
    if S is None:
        S = overlap(t1,t2)
    a1    = t1.widths()
    a2    = t2.widths()
    x1    = t1.x()
    x2    = t2.x()
    p1    = t1.p()
    p2    = t2.p()
    dx    = x1 - x2
    dp    = p1 - p2
    dpval = (dp + 2. * 1j * a1 * dx) / (2. * (a1 + a2))
    return dpval * S

#@timings.timed
def deldx(t1, t2, S=None):
    """Returns the del/dx matrix element between the nuclear component
       of two trajectories for each componet of 'x' (does not sum over terms)"""
    if S is None:
        S = overlap(t1,t2)
    a1 = t1.widths()
    a2 = t2.widths()
    x1 = t1.x()
    x2 = t2.x()
    p1 = t1.p()
    p2 = t2.p()
    dx    = x1 - x2
    psum  = a1*p2 + a2*p1
    dxval = (1j * psum - 2. * a1 * a2 * dx) / (a1 + a2)
    return dxval * S

#@timings.timed
def deld2x(t1, t2, S=None):
    """Returns the del^2/d^2x matrix element between the nuclear component
       of two trajectories for each componet of 'x' (does not sum over terms)"""
    if S is None:
        S  = overlap(t1,t2)
    a1 = t1.widths()
    a2 = t2.widths()
    x1 = t1.x()
    x2 = t2.x()
    p1 = t1.p()
    p2 = t2.p()
    dx     = x1 - x2
    psum   = a1*p2 + a2*p1
    d2xval = -(1j * 4. * a1 * a2 * dx * psum + 2. * a1 * a2 * (a1 + a2) -
               4. * dx**2 * a1**2 * a2**2 + psum**2) / (a1 + a2)**2
    return d2xval * S

