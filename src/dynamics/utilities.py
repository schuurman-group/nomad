import sys
import cmath
import numpy as np
import src.fmsio.fileio as fileio
import src.basis.particle as particle
#-------------------------------------------------------------------
#
# Utilities
#
#-------------------------------------------------------------------

#
# Given a displacement along a set of x, p coordiantes (dx, dp), return
# the overlap of the resultant gaussian primitive with the gaussian primitive
# centered at (x0,p0) (integrate over x, independent of x0)
#
def mode_overlap(alpha, dx, dp):
    return abs(cmath.exp( (-4*alpha*dx**2 + 4*np.complex(0.,1.)*dx*dp - (1/alpha)*dp**2) / 8.))

#
# Read in geometry file and hessian and sample about v=0 distribution
#    NOTE: assumes that hessian and geometry/momentum are in the same
#          basis (i.e. atom centered cartesians vs. normal modes)
#
def load_geometry():

    a_list                       = []
    p_list                       = []
    a_data,g_data,p_data,w_data  = fileio.read_geometry()

    for i in range(len(a_data)):
        a_list.append(a_data[i])

    for i in range(len(g_data)):
        dim = len(p_data[i])
        p_list.append(particle.particle(dim,i))
        p_list[i].name = g_data[i][0]
        particle.load_particle(p_list[i])
        p_list[i].x = np.fromiter((float(g_data[i][j]) for j in range(1,dim+1)),dtype=np.float)
        p_list[i].p = np.fromiter((float(p_data[i][j]) for j in range(0,dim)),dtype=np.float)
        if len(w_data) > i:
            p_list[i].width = w_data[i]

    return a_list,p_list

#
# do some error checking on the hessian file
#
def load_hessian():
    hessian = fileio.read_hessian()
    return hessian
