from src.fmsio import glbl
from src.fmsio import fileio
from src.basis import particle
from src.basis import trajectory
from src.basis import bundle
pes = __import__("src.interfaces."+glbl.fms['interface'],fromlist=['NA'])
#--------------------------------------------------------------------------
#
# Externally visible routines
#
#--------------------------------------------------------------------------
#
# initalize the trajectories
#
def init_trajectories(master):
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
        init_conditions(master)
    return

#
# initialize a restart
#
def init_restart(master):
    if glbl.fms['restart_time'] == 0.:
        return
    elif glbl.fms['restart_time'] == -1.:
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
def init_conditions(master):

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
#  Sample a generic distribution
#
def ho_distribution(master):
    print("ho_distribution not implemented yet")
    return

#
#  Take initial position and momentum from geometry.dat file
#
def user_specified(master):
    phase_gm   = fileio.read_geometry()
    print("n_orbs = "+str(glbl.pes['n_orbs']))
    master.add_trajectory(trajectory.trajectory(
                          phase_gm,
                          glbl.fms['interface'],
                          glbl.fms['n_states'],
                          tid=0,
                          parent=0,
                          n_basis=int(glbl.pes['n_orbs'])))
    print("geom = "+str(master.traj[0].x))
    master.traj[0].amplitude = complex(1.,0.)
    set_initial_state(master)
    return

#
# set the initial state of the trajectories in the bundle
#
def set_initial_state(master):
    if glbl.fms['init_state'] !=0:
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

