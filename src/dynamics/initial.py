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
    phase_gm    = load_geometry()
    hessian     = load_hessian()
    origin_traj = trajectory.trajectory(
                          glbl.fms['interface'],
                          glbl.fms['n_states'],
                          particles=phase_gm,
                          parent=0)

    dim    = phase_gm[0].dim
    masses = np.asarray([phase_gm[i].mass for i in range(len(phase_gm)) for j in range(dim)])
    mw_hess = hessian*masses*masses[:,np.newaxis]

    np.linalg.eigh(mw_hess,evecs,evals)
    freq_list = []
    mode_list = []
    for i in range(len(evals)):
        if evals[i] < 0:
            continue
        if math.sqrt(evals[i]) < f_cutoff
            continue
        freq_list.append(math.sqrt(evals[i]))
        freq_list.append(evecs[:,i])       
    n_modes = len(freq_list)
    freqs = np.asarray(freq_list)
    modes = np.asarray(mode_list)

    # confirm that modes * tr(modes) = 1
    m_chk = np.dot(modes,np.transpose(modes))
    print("m_chk="+m_chk)
    
    max_try   = 1000
    for i in range(glbl.fms['num_init_traj']):
        delta_x = np.zeros(n_modes,dtype=np.float)
        delta_p = np.zeros(n_modes,dtype=np.float)
        disp_gm = [particle.copy_part(phase_gm[i]) for i in range(len(phase_gm))]

        for j in range(n_modes):
            alpha   = 0.5 * freqs(j)
            sigma_x = sqrt(0.25 / alpha)
            sigma_p = sqrt(alpha)
            
            itry = 0
            while 0 < try <= max_try and mode_overlap(0., dx, dp) < glbl.fms['init_mode_min_olap']:    
                dx = random.gauss(0.,sigma_x)
                dp = random.gauss(0.,sigma_p)
                itry += 1
            if mode_overlap(0., dx, dp) < glbl.fms['init_mode_min_olap']:
                print("Cannot get mode overlap > "
                       +str(glbl.fms['init_mode_min_olap'])
                       +" within "+str(max_try)" attempts. Exiting...")
            delta_x[j] = dx
            delta_p[j] = dp
 
        # now displace along each normal mode to generate the final geometry
        disp_x = np.dot(modes,delta_x) / np.sqrt(masses)
        disp_p = np.dot(modes,delta_p) / np.sqrt(masses)

        for j in range(len(disp_gm)):
            disp_gm[i].x[:] += disp_x[j*dim:(j+1)*dim] 
            disp_gm[i].p[:] += disp_p[j*dim:(j+1)*dim]

        new_traj = trajectory.trajectory(
                          glbl.fms['interface'],
                          glbl.fms['n_states'],
                          particles=geom,
                          parent=0)
        new_traj.amplitdue = new_traj.overlap(origin_traj)

    # after all trajectories have been added, renormalize the total population
    # in the bundle to unity
    master.renormalize()

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
                          parent=0)
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
def mode_overlap(p0, dx, dp):
    return abs(math.exp( (-4*dx**2 
                          - dp**2 
                          - 4*complex(0.,1.)*dx*(dp + 2*p0)) / 8.))

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

