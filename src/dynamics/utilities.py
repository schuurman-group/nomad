"""
Utilities for dynamics calculations.
"""
import numpy as np

def mode_overlap(alpha, dx, dp):
    """Returns the overlap of Gaussian primitives

    Given a displacement along a set of x, p coordiantes (dx, dp), return
    the overlap of the resultant gaussian primitive with the gaussian primitive
    centered at (x0,p0) (integrate over x, independent of x0).
    """
    return abs(np.exp((-4.*alpha*dx**2 + 4.*1j*dx*dp -
                       (1./alpha)*dp**2) / 8.))

