import sys 
import numpy as np
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.trajectory as trajectory
#--------------------------------------------------------------------------
#
# Externally visible routines
#
#--------------------------------------------------------------------------
#
# initalize the trajectories
#
def init_bundle(master):
    pes = __import__("src.interfaces."+glbl.fms['interface'],fromlist=['NA'])

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

    #
    # print out the time t=0 summaries
    #
    master.update_matrices()
    master.update_logs()
    fileio.print_fms_logfile('t_step',[master.time,glbl.fms['default_time_step'],master.nalive])

    return master.time

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
    distrib = __import__("src.dynamics."+glbl.fms['init_sampling'],fromlist=['NA'])

    #
    # sample the requested phase distribution
    #
    distrib.sample_distribution(master)

    #
    # determine the initial state of each trajectory
    #
    set_initial_state(master)
    return

#
# set the initial state of the trajectories in the bundle
#
def set_initial_state(master):
 
    #
    # determine
    # 
    if glbl.fms['init_state'] != -1:
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

    #
    # if mirror basis, put zero amplitude functions on all other states
    #
    if glbl.fms['mirror_basis']: 
        for i in range(master.n_total()):
            for j in range(master.nstates):

                if j == master.traj[i].state:
                    continue
       
                new_traj = trajectory.copy_traj(master.traj[i]) 
                new_traj.amplitude = np.complex(0.,0.)
                new_traj.state     = j
                master.add_trajectory(new_traj)
  
