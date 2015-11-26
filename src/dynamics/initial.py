import variable
import fileio
import particle
import trajectory
import bundle
import interface
#--------------------------------------------------------------------------
#
# Externally visible routines
#
#--------------------------------------------------------------------------
#
#
def init_restart(master):
    if variable['restart_time'] == 0.:
        return
    elif variable['restart_time'] == -1.:
        chkpt = open('last_step.dat','r',encoding='utf-8')
        master.read_bundle(chkpt)
    else:
        chkpt = open('checkpoint.dat','r',encoding='utf-8')
        master.read_chkpt(chkpt,variable['restart_time'])
    chkpt.close()
    return

#
#  initialize the t=0 set of trajectories
#
def initial_conditions(master):
 
    # reads in geometry and hessian and samples v=0 distribution
    if variable.fms['init_sampling'] == 'gs_wigner':
        gs_wigner(master)   
    # reads in geometry, hessian, and dist.inp -- an arbitrary
    #  distribution expressed as a linear combination of h.o. wfns.
    elif variable.fms['init_sampling'] == 'distribution':
        ho_distribution(master)
    # reads in geometry file and employs specified geometry and
    # and momentum from this file alone
    elif variable.fms['init_sampling'] == 'user_specified':   
        user_specified(master)
    else:
        print("{:s} not a recognized initial condition".
                          format(variable.fms['init_sampling']))

#----------------------------------------------------------------------------
#
# Private routines
#
#----------------------------------------------------------------------------
#
# Read in geometry file and hessian and sample about v=0 distribution
#    NOTE: assumes that hessian and geometry/momentum are in the same
#          basis (i.e. atom centered cartesians vs. normal modes)
#
def gs_wigner(master):
    phase_gm   = fileio.read_geometry()
    hessian    = fileio.read_hessian()
    return

#
# 
#
def ho_distribution(master):
    print("ho_distribution not implemented yet")
    return

#
#
#
def user_specified(master):
    phase_gm   = fileio.read_geometry()
    return
