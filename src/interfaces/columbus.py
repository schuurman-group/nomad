#
# routines for running a columbus computation
#
import sys
import numpy as np
import particle
import variable
#----------------------------------------------------------------
# 
# Functions called from interface object
#
#----------------------------------------------------------------
#
# returns the energy at the specified geometry. If value on file 
#  not current, or we don't care about saving data -- recompute
#
# geom is a list of particles
def energy(geom,nstates):
    if not in_cache(geom):
        run_single_point(geom)
    try:
        return fetch_energy(cache_index(geom),nstates)
    except:
        print("ERROR in fetch_energy")       
        sys.exit("ERROR in columbus module fetching energy")

#
# returns the MOs as an numpy array
#
def orbitals(geom):
    if not in_cache(geom):
        run_single_point(geom)
    try:
        return fetch_orbitals(cache_index(geom))
    except:
        print("ERROR in fetch_orbitals")     
        sys.exit("ERROR in columbus module fetching orbitals")

#
# return gradient. If lstate == rstate, gradient on state lstate. Else
#   returns non-adiabatic coupling vector
#
def derivative(geom,lstate,rstate):
    if not in_cache(geom):
        run_single_point(geom)
    try:
        return fetch_gradients(cache_index(geom),lstate,rstate)
    except:
        print("ERROR in fetch_gradients")     
        sys.exit("ERROR in columbus module fetching gradients")

#
# if lstate != rstate, corresponds to transition dipole
#
def dipole(geom,lstate,rstate):
    if not in_cache(geom):
        run_single_point(geom)
    try:
        return fetch_dipole(cache_index(geom),lstate,rstate)
    except:
        print("ERROR in fetch_dipole")     
        sys.exit("ERROR in columbus module fetching dipoles")

#
# return second moment tensor for state=state
#
def quadrupole(geom,state):
    if not in_cache(geom):
        run_single_point(geom)
    try:
        return fetch_quadpole(cache_index(geom),lstate)
    except:
        print("ERROR in fetch_quadpole")     
        sys.exit("ERROR in columbus module fetching quadpole")

#
#
#
def charges(geom,state):
    if not in_cache(geom):
        run_single_point(geom,lstate)
    try:
        return fetch_charges(cache_index(geom),lstate)
    except:
        print("ERROR in fetch_charges")     
        sys.exit("ERROR in columbus module fetching charges")

#----------------------------------------------------------------
#
#  "Private" functions
#
#----------------------------------------------------------------
def in_cache(geom):
    g = geom
    for i in range(len(self.cache))
        if np.linalg.norm(g-self.cache[i]) <= variable.fpzero:
              

#
# For the columbus module, since gradients are not particularly
# time consuming, it's easier (and probably faster) to compute
# EVERYTHING at once (i.e. all energies, all gradients, all properties)
# Thus, if electronic structure information is not up2date, all methods
# call the same routine: run_single_point.
#
#  This routine will:
#    1. Compute an MCSCF/MRCI energy
#    2. Compute all couplings
#
def run_single_point(geom):



