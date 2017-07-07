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
    global pes, sampling, integrals
    
    """Initializes the trajectories."""

    try:
        pes      = __import__('src.interfaces.'+glbl.interface['interface'],
                         fromlist=['NA'])
    except ImportError:
        print('Cannot import pes: src.interfaces.'+
               str(glbl.interface['interface']))

    try:
        sampling = __import__('src.sampling.'+glbl.sampling['init_sampling'],
                         fromlist=['NA'])
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

        # load the width, coordinates and momenta from input
        # this is an edit
        (masses, widths, coords, momenta) = read_coord_info()

        # first generate the initial nuclear coordinates and momenta
        # and add the resulting trajectories to the bundle
        sampling.set_initial_coords(masses, widths, coords, momenta, master)

        # set widths and masses
        set_masses(masses, master)

        # set the initial state of the trajectories in bundle. This may
        # require evaluation of electronic structure
        set_initial_state(master)

        # set the initial amplitudes of the basis functions
        origin = None
        if glbl.nuclear_basis['init_amp_overlap']:
            origin = make_origin_traj(master, masses, widths, coords, momenta)
        set_initial_amplitudes(master, origin)
 
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
# read the geometries, momenta, widths from input
#
def read_coord_info():

    if glbl.nuclear_basis['geomfile'] is not '':
        (labels, xlst, plst) = fileio.read_geometry(glbl.nuclear_basis['geomfile'])
        geoms = np.array(xlst)
        moms  = np.array(plst)

    elif len(glbl.nuclear_basis['geometries']) != 0:
        labels = glbl.nuclear_basis['labels']
        geoms  = np.array(glbl.nuclear_basis['geometries'])
        moms   = np.array(glbl.nuclear_basis['momenta'])
        ngeoms = len(glbl.nuclear_basis['geometries'])

    else:
        sys.exit('sampling.explicit: No geometry specified')

    if glbl.nuclear_basis['use_atom_lib']:
        ncart = 3
        wlst  = []
        mlst  = []
        for i in len(labels):
            (mass, wid, num) = basis.atom_lib(labels[i])
            mlst.extend([mass for i in range(ncart)])
            wlst.extend([wid for i in range(ncart)])
        masses = np.array(mlst, dtype=float)
        widths = np.array(wlst, dtype=float)
                
    elif (len(glbl.nuclear_basis['masses']) != 0 and 
          len(glbl.nuclear_basis['widths']) != 0):
        masses = np.array(glbl.nuclear_basis['masses'])
        widths = np.array(glbl.nuclear_basis['widths'])

    else:
        sys.exit('sampling.explicit: No masses/widths specified') 

    return (masses, widths, geoms, moms)

#
# set the mass and width of trajectory basis functions
#
def set_masses(masses, master):

    # if vibronic, assume freq/mass weighted coordinates
    if glbl.interface['interface'] is 'vibronic':
        m_vec = pes.kecoeff

    # use the masses from atom lib 
    else:
        m_vec = masses

    for i in range(master.n_traj()):
        master.traj[i].mass = m_vec

    return

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
def set_initial_amplitudes(master, origin):

    if glbl.nuclear_basis['init_amp_overlap']:

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
        cvec = np.dot(sinv, ovec)
        for i in range(master.n_traj()):
            master.traj[i].update_amplitude(cvec[i])

    elif len(glbl.nuclear_basis['amplitudes']) == master.n_traj():
        amps = np.array(glbl.nuclear_basis['amplitudes'],dtype=complex)
        for i in range(master.n_traj()):
            master.traj[i].update_amplitude(amps[i])

    else:
        for i in range(master.n_traj()):
            master.traj[i].update_amplitude(1.+0j)

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
def make_origin_traj(master, masses, widths, pos, mom):
    """construct a trajectory basis function at the origin
       specified in the input files"""

    ndim = len(pos[0])
    m_vec = np.array(masses)
    w_vec = np.array(widths)
    x_vec = np.array(pos[0])
    p_vec = np.array(mom[0])

    origin = trajectory.Trajectory(glbl.propagate['n_states'], ndim,
                                   width=w_vec, mass=m_vec, parent=0)

    origin.update_x(x_vec)
    origin.update_p(p_vec)
    origin.state = 0
    # if we need pes data to evaluate overlaps, determine that now
    if integrals.overlap_requires_pes:
        surface.update_pes_traj(origin)

    return origin
