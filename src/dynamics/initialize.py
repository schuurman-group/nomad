"""
Routines for initializing dynamics calculations.
"""
import sys
import numpy as np
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.trajectory as trajectory
import src.basis.bundle as bundle
import src.dynamics.surface as surface


#--------------------------------------------------------------------------
#
# Externally visible routines
#
#--------------------------------------------------------------------------
def init_bundle(master):
    """Initializes the trajectories."""
    pes       = __import__('src.interfaces.'+glbl.interface['interface'],
                         fromlist=['NA'])
    sampling  = __import__('src.sampling.'+glbl.sampling['init_sampling'],
                         fromlist=['NA'])

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
        (labels, widths, coords, momenta) = read_coord_info()

        # first generate the initial nuclear coordinates and momenta
        # and add the resulting trajectories to the bundle
        sampling.set_initial_coords(widths, coords, momenta, master)

        # set widths and masses
        set_masses(labels, master)

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
    if glbl.mpi_rank == 0:
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
        (labels, geoms, moms) = fileio.read_geometry(glbl.nuclear_basis['geomfile'])
    elif len(glbl.nuclear_basis['geometry']) != 0:
        labels = glbl.nuclear_basis['labels']
        ngeoms = len(glbl.nuclear_basis['geometry'])
        for i in range(ngeoms):
            geoms  = np.asarray(glbl.nuclear_basis['geometry'][i])
            moms   = np.asarray(glbl.nuclear_basis['momenta'][i])
    else:
        sys.exit('sampling.explicit: No geometry specified')

    if glbl.nuclear_basis['use_atom_lib']:
        widths = np.zeros(len(geoms[0]))
        for i in len(labels):
            (mass, widths[i], num) = basis.atom_lib(labels[i])
    else:
        mass  = np.asarray(glbl.nuclear_basis['masses'])
        width = np.asarray(glbl.nuclear_basis['width'])

    return (labels, widths, geoms, moms)

#
# set the mass and width of trajectory basis functions
#
def set_masses(labels, master):

    if glbl.nuclear_basis['use_atom_lib']:
        masses = np.zeros(len(labels))
        for i in len(labels):
            (masses[i], widths, num) = basis.atom_lib(labels[i])

        crd_dim = master.traj[0].crd_dim

        m_vec = np.array([masses[i] for j in range(crd_dim) 
                                    for i in range(len(masses))])

        for i in range(master.n_traj()):
            master.traj[i].mass = m_vec
        
    # if vibronic, assume freq/mass weighted coordinates
    elif glbl.interface['interface'] is 'vibronic':
        m_vec = pes.kecoeff

        for i in range(master.n_traj()):
            master.traj[i].mass = m_vec

    # don't know how to set masses
    else:
        sys.exit('No masses found')

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
def set_initial_amplitude(master):

    if glbl.nuclear_basis['init_amp_overlap']:

        # Calculate the initial expansion coefficients via projection onto
        # the initial wavefunction that we are sampling
        ovec = np.zeros(ntraj, dtype=complex)
        for i in range(ntraj):
            ovec[i] = integrals.traj_overlap(master.traj[i],origin_traj)
        smat = np.zeros((ntraj, ntraj), dtype=complex)
        for i in range(ntraj):
            for j in range(i+1):
                smat[i,j] = integrals.traj_overlap(master.traj[i],master.traj[j])
                if i != j:
                    smat[j,i] = smat[i,j].conjugate()
        #print(smat)
        sinv = sp_linalg.pinvh(smat)
        cvec = np.dot(sinv, ovec)
        for i in range(ntraj):
            master.traj[i].update_amplitude(cvec[i])

    elif len(glbl.sampling['amplitudes']) == master.n_traj():
        amps = np.array(glbl.sampling['amplitudes'],dtype=complex)
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
