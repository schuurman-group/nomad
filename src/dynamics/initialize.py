"""
Routines for initializing dynamics calculations.
"""
import sys
import numpy as np
import scipy.linalg as sp_linalg
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.trajectory as trajectory
import src.basis.bundle as bundle
import src.dynamics.surface as surface

# the dynamically loaded libraries (done once by init_bundle)
pes       = None
sampling  = None 
integrals = None
#--------------------------------------------------------------------------
#
# Externally visible routines
#
#--------------------------------------------------------------------------
def init_bundle(master):
    """Initializes the trajectories."""
    global pes, sampling, integrals
   
    try:
        pes      = __import__('src.interfaces.'+glbl.interface['interface'],
                         fromlist=['NA'])
    except ImportError:
        print('Cannot import pes: src.interfaces.'+
               str(glbl.interface['interface']))

    try:
        print("samp string="+'src.sampling.'+glbl.sampling['init_sampling'])
        sampling = __import__('src.sampling.'+glbl.sampling['init_sampling'],
                         fromlist=['NA'])
        print("getting here?")
    except ImportError:
        print('Cannot import sampling: src.sampling.'+
               str(glbl.sampling['init_sampling']))

    try:
        integrals = __import__('src.integrals.'+glbl.propagate['integrals'],
                         fromlist=['NA'])
    except ImportError:
        print('Cannot import sampling: src.integrals.'+
               str(glbl.propagate['integrals']))


    # initialize the trajectory and bundle output files
    fileio.init_fms_output()

    # initialize the interface we'll be using the determine the
    # the PES. There are some details here that trajectories
    # will want to know about
    pes.init_interface()

    # initialize the surface module -- caller of the pes interface
    surface.init_surface(glbl.interface['interface'])

    # now load the initial trajectories into the bundle
    if glbl.sampling['restart']:
        init_restart(master)

    else:

        # first generate the initial nuclear coordinates and momenta
        # and add the resulting trajectories to the bundle
        sampling.set_initial_coords(master)

        # set the initial state of the trajectories in bundle. This may
        # require evaluation of electronic structure
        set_initial_state(master)

        # set the initial amplitudes of the basis functions
        set_initial_amplitudes(master)
 
        # add virtual basis functions, if desired (i.e. virtual basis = true)
        if glbl.sampling['virtual_basis']:
            virtual_basis(master)

    # update all pes info for all trajectories and centroids (where necessary)
    surface.update_pes(master)
    # compute the hamiltonian matrix...
    master.update_matrices()
    # so that we may appropriately renormalize to unity
    master.renormalize()

    # this is the bundle at time t=0.  Save in order to compute auto
    # correlation function
    glbl.bundle0 = master.copy()

    # write to the log files
    if glbl.mpi['rank'] == 0:
        master.update_logs()

    fileio.print_fms_logfile('t_step', [master.time, glbl.propagate['default_time_step'],
                                        master.nalive])

    return master.time


#---------------------------------------------------------------------------
#
# Private routines
#
#----------------------------------------------------------------------------
def init_restart(master):
    """Initializes a restart."""
    if glbl.sampling['restart_time'] == -1.:
        fname = fileio.home_path+'/last_step.dat'
    else:
        fname = fileio.home_path+'/checkpoint.dat'

    master.read_bundle(fname, glbl.sampling['restart_time'])

#
#
def set_initial_state(master):
    """Sets the initial state of the trajectories in the bundle."""

    # initialize to the state with largest transition dipole moment
    if glbl.sampling['init_brightest']:

        # set all states to the ground state
        for i in range(master.n_traj()):
            master.traj[i].state = 0
            # compute transition dipoles
            surface.update_pes_traj(master.traj[i])

        # set the initial state to the one with largest t. dip.
        for i in range(master.n_traj()):
            if 'tr_dipole' not in master.traj[i].pes_data.data_keys:
                raise KeyError('ERROR, trajectory '+str(i)+
                               ': Cannot set state by transition moments - '+
                               'tr_dipole not in pes_data.data_keys')
            tdip = np.array([np.linalg.norm(master.traj[i].pes_data.dipoles[:,0,j])
                             for j in range(1, glbl.propagate['n_states'])])
            fileio.print_fms_logfile('general',
                                    ['Initializing trajectory '+str(i)+
                                     ' to state '+str(np.argmax(tdip)+1)+
                                     ' | tr. dipople array='+np.array2string(tdip, \
                                       formatter={'float_kind':lambda x: "%.4f" % x})])
            master.traj[i].state = np.argmax(tdip)+1

    # use "init_state" to set the initial state
    elif len(glbl.sampling['init_state']) == master.n_traj():
        for i in range(master.n_traj()):
            istate = glbl.sampling['init_state'][i]
            if istate == -1:
                raise ValueError('Invalid state assignment traj '+str(i)+' state=-1')
            master.traj[i].state = glbl.sampling['init_state'][i]
        
    else:
        raise ValueError('Ambiguous initial state assignment.')

    return

#
#
def set_initial_amplitudes(master):
    global integrals
 
    # if init_amp_overlap is set, overwrite 'amplitudes' that was 
    # set in fms.input
    if glbl.nuclear_basis['init_amp_overlap']:

        origin = make_origin_traj()

        # update all pes info for all trajectories and centroids (where necessary)
        if integrals.overlap_requires_pes:
            surface.update_pes(master)

        # Calculate the initial expansion coefficients via projection onto
        # the initial wavefunction that we are sampling
        ovec = np.zeros(master.n_traj(), dtype=complex)
        for i in range(master.n_traj()):
            ovec[i] = integrals.traj_overlap(master.traj[i], origin, nuc_only=True)
        print("ovec="+str(ovec))
        smat = np.zeros((master.n_traj(), master.n_traj()), dtype=complex)
        for i in range(master.n_traj()):
            for j in range(i+1):
                smat[i,j] = master.integrals.traj_overlap(master.traj[i], 
                                                          master.traj[j])
                if i != j:
                    smat[j,i] = smat[i,j].conjugate()
        print(smat)
        sinv = sp_linalg.pinvh(smat)
        glbl.nuclear_basis['amplitudes'] = np.dot(sinv, ovec)

    # if we don't have a sufficient number of amplitudes, append
    # amplitudes with "zeros" as necesary
    if len(glbl.nuclear_basis['amplitudes']) < master.n_traj():
        dif = master.n_traj() - len(glbl.nuclear_basis['amplitudes'])
        fileio.print_fms_logfile('warning',['appending '+str(dif)+
                                 ' values of 0+0j to amplitudes'])
        glbl.nuclear_basis['amplitudes'].extend([0+0j for i in range(dif)])    

    # finally -- update amplitudes in the bundle
    for i in range(master.n_traj()):
        master.traj[i].update_amplitude(glbl.nuclear_basis['amplitudes'][i])

    return

def virtual_basis(master):
    """Add virtual basis funcions.

    If additional virtual basis functions requested, for each trajectory
    in bundle, add aditional basis functions on other states with zero
    amplitude.
    """
    for i in range(master.n_traj()):
        for j in range(master.nstates):
            if j == master.traj[i].state:
                continue

            new_traj = master.traj[i].copy()
            new_traj.amplitude = 0j
            new_traj.state = j
            master.add_trajectory(new_traj)

#
# construct a trajectory basis function at the origin specified in the 
# input.
#
def make_origin_traj():
    """construct a trajectory basis function at the origin
       specified in the input files"""

    ndim = len(glbl.nuclear_basis['geometries'][0])
    m_vec = np.array(glbl.nuclear_basis['masses'])
    w_vec = np.array(glbl.nuclear_basis['widths'])
    x_vec = np.array(glbl.nuclear_basis['geometries'][0])
    p_vec = np.array(glbl.nuclear_basis['momenta'][0])

    origin = trajectory.Trajectory(glbl.propagate['n_states'], ndim,
                                   width=w_vec, mass=m_vec, parent=0)

    origin.update_x(x_vec)
    origin.update_p(p_vec)
    origin.state = 0
    # if we need pes data to evaluate overlaps, determine that now
    if integrals.overlap_requires_pes:
        surface.update_pes_traj(origin)

    return origin
