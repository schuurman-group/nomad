"""
Utilities for dynamics calculations.
"""
import numpy as np
import src.fmsio.fileio as fileio
import src.basis.particle as particle


def mode_overlap(alpha, dx, dp):
    """Returns the overlap of Gaussian primitives

    Given a displacement along a set of x, p coordiantes (dx, dp), return
    the overlap of the resultant gaussian primitive with the gaussian primitive
    centered at (x0,p0) (integrate over x, independent of x0).
    """
    return abs(np.exp((-4.*alpha*dx**2 + 4.*complex(0.,1.)*dx*dp -
                       (1./alpha)*dp**2) / 8.))


def load_geometry():
    """Reads in geometry file and Hessian and sample about v=0 distribution

    Assumes that Hessian and geometry/momentum are in the same
    basis (i.e. atom centered cartesians vs. normal modes).
    """
    a_list                       = []
    p_list                       = []
    a_data, g_data, p_data, w_data  = fileio.read_geometry()

    for a_item in a_data:
        a_list.append(a_item)

    for i in range(len(g_data)):
        dim = len(p_data[i])
        p_list.append(particle.Particle(dim, i))
        p_list[i].name = g_data[i][0]
        particle.load_particle(p_list[i])
        p_list[i].x = np.fromiter((float(g_data[i][j])
                                   for j in range(1, dim+1)), dtype=float)
        p_list[i].p = np.fromiter((float(p_data[i][j])
                                   for j in range(0, dim)), dtype=float)
        if len(w_data) > i:
            p_list[i].width = w_data[i]

    return a_list, p_list


def load_hessian():
    """Does error checking on the Hessian file."""
    hessian = fileio.read_hessian()
    return hessian
