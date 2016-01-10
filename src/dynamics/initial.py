import sys 
import numpy as np
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.particle as particle
import src.basis.trajectory as trajectory
import src.basis.bundle as bundle
pes = __import__("src.interfaces."+glbl.fms['interface'],fromlist=['NA'])
#--------------------------------------------------------------------------
#
# Externally visible routines
#
#--------------------------------------------------------------------------
#
# initalize the trajectories
#
def init_bundle(master):

    #
    # initialize the trajectory and bundle output files
    #
    fileio.init_fms_output()

    #
    # initialize the interface we'll be using the determine the 
    #   the PES. There are some details here that trajectories
    #    will want to know about
    pes.init_interface()

    # 
    # now load the initial trajectories into the bundle
    #
    if(glbl.fms['restart']):
        init_restart(master)
    else:
        init_trajectories(master)

    master.update_matrices()
    master.update_logs()
    fileio.print_fms_logfile('t_step',[master.time,glbl.fms['default_time_step'],master.nalive])

    return master.time

############################################################################

#---------------------------------------------------------------------------
#
# Private routines
#
#----------------------------------------------------------------------------

#
# initialize a restart
#
def init_restart(master):
    
    if glbl.fms['restart_time'] == -1.:
        fname = fileio.home_path+'/last_step.dat'
    else:
        fname = fileio.home_path+'/checkpoint.dat'

    master.read_bundle(fname,glbl.fms['restart_time'])

    return

#
#  initialize the t=0 set of trajectories
#
def init_trajectories(master):

    # reads in geometry and hessian and samples v=0 distribution
    if glbl.fms['init_sampling'] == "gs_wigner":
        gs_wigner(master)   
    # reads in geometry, hessian, and dist.inp -- an arbitrary
    #  distribution expressed as a linear combination of h.o. wfns.
    elif glbl.fms['init_sampling'] == "distribution":
        ho_distribution(master)
    # reads in geometry file and employs specified geometry and
    # and momentum from this file alone
    elif glbl.fms['init_sampling'] == 'user_specified':   
        user_specified(master)
    else:
        print("{:s} not a recognized initial condition".
                          format(glbl.fms['init_sampling']))

#
# set the initial state of the trajectories in the bundle
#
def set_initial_state(master):
    if glbl.fms['init_state'] != 0:
        for i in range(master.n_total()):
            master.traj[i].state = glbl.fms['init_state']
    elif glbl.fms['init_brightest']:
        for i in range(master.n_total()):
            master.traj[i].state = 1
            tdip =(np.linalg.norm(master.traj[i].dipole(j)) for j
                                    in range(2,glbl.fms['n_states']+1))
            master.traj[i].state = np.argmax(tdip)+2
    else:
        print("ERROR: Ambiguous initial state assignment")
        sys.exit()

#----------------------------------------------------------------
#
# Initial condition routines -- wigner, general, user, etc.
#
#-----------------------------------------------------------------

#
# sample a v=0 wigner distribution
# 
def gs_wigner(master):
    phase_gm   = load_geometry()
    hessian    = load_hessian()
    return

#
#  Sample a generic distribution
#
def ho_distribution(master):
    print("ho_distribution not implemented yet")
    return

#
#  Take initial position and momentum from geometry.dat file
#
def user_specified(master):
    geom_list = [load_geometry()]
    amp_list  = [complex(1.,0.)]
    pes.populate_bundle(master,geom_list,amp_list)
    set_initial_state(master)
    return

#-------------------------------------------------------------------
#
# Utilities
#
#-------------------------------------------------------------------

#
# Read in geometry file and hessian and sample about v=0 distribution
#    NOTE: assumes that hessian and geometry/momentum are in the same
#          basis (i.e. atom centered cartesians vs. normal modes)
#
def load_geometry():
    p_list      = []
    geom_data   = fileio.read_geometry()
    
    for i in range(len(geom_data)):
        dim = int((len(geom_data[i])-1)/2)
        p_list.append(particle.particle(dim,i))
        p_list[i].name = geom_data[i][0]
        particle.load_particle(p_list[i])
        p_list[i].x = np.fromiter((float(geom_data[i][j]) for j in range(1,4)),dtype=np.float)
        p_list[i].p = np.fromiter((float(geom_data[i][j]) for j in range(4,7)),dtype=np.float)
    return p_list

#
# do some error checking on the hessian file
#
def load_hessian():
    hessian_data = file.read_hessian()

