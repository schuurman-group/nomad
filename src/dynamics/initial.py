import sys 
import random
import cmath
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
    print("init_bundle -- update logs")
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

#----------------------------------------------------------------
#
# Initial condition routines -- wigner, general, user, etc.
#
#-----------------------------------------------------------------

#
# sample a v=0 wigner distribution
# 
def gs_wigner(master):
    phase_gm    = load_geometry()
    hessian     = load_hessian()

    origin_traj = trajectory.trajectory(
                          glbl.fms['interface'],
                          glbl.fms['n_states'],
                          particles=phase_gm,
                          parent=0)

    dim     = phase_gm[0].dim
    masses  = np.asarray([phase_gm[i].mass for i in range(len(phase_gm)) for j in range(dim)],dtype=np.float)
    invmass = np.asarray([1./ np.sqrt(masses[i]) if masses[i] != 0. else 0 for i in range(len(masses))],dtype=np.float)
    mw_hess = invmass * hessian * invmass[:,np.newaxis]

    # set up nodes and frequencies
    evals, evecs = np.linalg.eigh(mw_hess)
    f_cutoff = 0.0001
    freq_list = []
    mode_list = []
    for i in range(len(evals)):
        if evals[i] < 0:
            continue
        if np.sqrt(evals[i]) < f_cutoff:
            continue
        freq_list.append(np.sqrt(evals[i]))
        mode_list.append(evecs[:,i].tolist())       
    n_modes = len(freq_list)
    freqs = np.asarray(freq_list)
    print("freqs="+str(freqs))
    modes = np.asarray(mode_list).transpose()

    # confirm that modes * tr(modes) = 1
    m_chk = np.dot(modes,np.transpose(modes))
    
    # loop over the number of initial trajectories
    max_try   = 1000
    for i in range(glbl.fms['n_init_traj']):
        delta_x = np.zeros(n_modes,dtype=np.float)
        delta_p = np.zeros(n_modes,dtype=np.float)
        disp_gm = [particle.copy_part(phase_gm[j]) for j in range(len(phase_gm))]

        for j in range(n_modes):
            alpha   = 0.5 * freqs[j]
            sigma_x = np.sqrt(0.25 / alpha)
            sigma_p = np.sqrt(alpha)
            itry = 0
            while 0 <= itry <= max_try:    
                dx = random.gauss(0.,sigma_x)
                dp = random.gauss(0.,sigma_p)
                itry += 1
                if mode_overlap(alpha, dx, dp) > glbl.fms['init_mode_min_olap']:
                    break
            if mode_overlap(alpha, dx, dp) < glbl.fms['init_mode_min_olap']:
                print("Cannot get mode overlap > "
                       +str(glbl.fms['init_mode_min_olap'])
                       +" within "+str(max_try)+" attempts. Exiting...")
            delta_x[j] = dx
            delta_p[j] = dp
 
        # now displace along each normal mode to generate the final geometry
        disp_x = np.dot(modes,delta_x) / np.sqrt(masses)
        disp_p = np.dot(modes,delta_p) / np.sqrt(masses)

        for j in range(len(disp_gm)):
            disp_gm[j].x[:] += disp_x[j*dim:(j+1)*dim] 
            disp_gm[j].p[:] += disp_p[j*dim:(j+1)*dim]

        print(' '.join([str(disp_x[k]) for k in range(len(disp_x))])+' '.join([str(disp_p[k]) for k in range(len(disp_p))]))
        new_traj = trajectory.trajectory(
                          glbl.fms['interface'],
                          glbl.fms['n_states'],
                          particles=disp_gm,
                          parent=0)
        new_traj.amplitude = new_traj.overlap(origin_traj)
        master.add_trajectory(new_traj)
 
    # after all trajectories have been added, renormalize the total population
    # in the bundle to unity
    master.renormalize()
    set_initial_state(master)

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
    geom = load_geometry()
    amp  = complex(1.,0.)
    # add a single trajectory specified by geometry.dat
    master.add_trajectory(trajectory.trajectory(
                          glbl.fms['interface'],
                          glbl.fms['n_states'],
                          particles=geom,
                          parent=0))
    master.traj[i].amplitude = amp
    set_initial_state(master)
    return

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
    return abs(cmath.exp( (-4*alpha*dx**2 + 4*complex(0.,1.)*dx*dp - (1/alpha)*dp**2) / 8.))

#
# Read in geometry file and hessian and sample about v=0 distribution
#    NOTE: assumes that hessian and geometry/momentum are in the same
#          basis (i.e. atom centered cartesians vs. normal modes)
#
def load_geometry():
    p_list                = []
    g_data,p_data,w_data  = fileio.read_geometry()

    for i in range(len(g_data)):
        dim = len(p_data[i])
        p_list.append(particle.particle(dim,i))
        p_list[i].name = g_data[i][0]
        particle.load_particle(p_list[i])
        p_list[i].x = np.fromiter((float(g_data[i][j]) for j in range(1,dim+1)),dtype=np.float)
        p_list[i].p = np.fromiter((float(p_data[i][j]) for j in range(0,dim)),dtype=np.float)
        if len(w_data) > i:
            p_list[i].width = w_data[i]

    # debug
    for i in range(len(p_list)):
        p_list[i].write_particle(sys.stdout)

    return p_list

#
# do some error checking on the hessian file
#
def load_hessian():
    hessian = fileio.read_hessian()
    return hessian

