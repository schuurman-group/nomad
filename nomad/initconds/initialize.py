"""
Routines for initializing dynamics calculations.
"""
import numpy as np
import scipy.linalg as sp_linalg
import nomad.core.glbl as glbl
import nomad.core.log as log
import nomad.core.trajectory as trajectory
import nomad.core.wavefunction as wavefunction
import nomad.core.surface as evaluate
import nomad.core.checkpoint as checkpoint
import nomad.core.matrices as matrices
import nomad.integrals.integral as integral

def init_wavefunction(master):
    """Initializes the trajectories."""
    # initialize the interface we'll be using the determine the
    # the PES. There are some details here that trajectories
    # will want to know about

    glbl.modules['interface']  = __import__('nomad.interfaces.' + glbl.methods['interface'],
                                 fromlist=['NA'])
    glbl.modules['interface'].init_interface()

    # create glbl modules
    glbl.modules['matrices']   = matrices.Matrices()

    glbl.modules['init_conds'] = __import__('nomad.initconds.' + glbl.methods['init_conds'],
                                 fromlist=['NA'])
    glbl.modules['adapt']      = __import__('nomad.adapt.' + glbl.methods['adapt_basis'],
                                 fromlist=['a'])
    glbl.modules['propagator'] = __import__('nomad.propagators.' + glbl.methods['propagator'],
                                 fromlist=['a'])

    # WARNING: this should be done more eloquently
    # need to determine form of the coefficients on the kinetic energy
    # operator, i.e. cartesian vs. normal mode coordinate basis
    if glbl.methods['interface'] == 'vibronic':
        # if normal modes, read frequencies from vibronic interface
        kecoef = 0.5 * glbl.modules['interface'].ham.freq
    else:
        kecoef = 0.5 / glbl.properties['crd_masses']
    glbl.modules['integrals']  = integral.Integral(kecoef, 
                                                   glbl.methods['ansatz'],
                                                   glbl.methods['integral_eval'])

    # now load the initial trajectories into the bundle
    if glbl.properties['restart']:

        # retrieve current wave function
        checkpoint.retrieve_simulation(master, integrals=glbl.modules['integrals'],
                                       file_name=glbl.paths['chkpt_file'])
        # build the necessary matrices
        glbl.modules['matrices'].build(master, glbl.modules['integrals'])
        master.update_matrices(glbl.modules['matrices'])

        # retrieve time t=0 wave function
        master0 = wavefunction.Wavefunction()
        checkpoint.retrieve_simulation(master0, integrals=None, time=0.,
                                       file_name=glbl.paths['chkpt_file'])
        save_initial_wavefunction(master0)

    else:
        # first generate the initial nuclear coordinates and momenta
        # and add the resulting trajectories to the bundle
        glbl.modules['init_conds'].set_initial_coords(master)

        # set the initial state of the trajectories in bundle. This may
        # require evaluation of electronic structure
        set_initial_state(master)

        # set the initial amplitudes of the basis functions
        set_initial_amplitudes(master)

        # add virtual basis functions, if desired
        if glbl.properties['virtual_basis']:
            virtual_basis(master)

        # update the integrals
        glbl.modules['integrals'].update(master)

        # once phase space position and states of the basis functions
        # are set, update the potential information
        evaluate.update_pes(master)

        # update the couplings for all the trajectories
        for i in range(master.n_traj()):
            glbl.modules['interface'].evaluate_coupling(master.traj[i])

        # compute the hamiltonian matrix...
        glbl.modules['matrices'].build(master, glbl.modules['integrals'])
        master.update_matrices(glbl.modules['matrices'])

        # so that we may appropriately renormalize to unity
        master.renormalize()

        # this is the bundle at time t=0.  Save in order to compute auto
        # correlation function
        save_initial_wavefunction(master)

    # write the wavefunction to the archive
    if glbl.mpi['rank'] == 0:
        checkpoint.archive_simulation(master, integrals=glbl.modules['integrals'],
                                      time=master.time, file_name=glbl.paths['chkpt_file'])

    log.print_message('t_step', [master.time, glbl.properties['default_time_step'],
                                      master.nalive])

    return master.time


#---------------------------------------------------------------------------
#
# Private routines
#
#----------------------------------------------------------------------------
def set_initial_state(master):
    """Sets the initial state of the trajectories in the bundle."""
    if glbl.properties['init_brightest']:
        # initialize to the state with largest transition dipole moment
        # set all states to the ground state
        for i in range(master.n_traj()):
            master.traj[i].state = 0
            # compute transition dipoles
            evaluate.update_pes_traj(master.traj[i])

        # set the initial state to the one with largest t. dip.
        for i in range(master.n_traj()):
            if 'dipole' not in master.traj[i].pes.avail_data():
                raise KeyError('trajectory '+str(i)+
                               ': Cannot set state by transition moments - '+
                               'dipole not in pes.avail_data()')

            tr_dipole = master.traj[i].pes.get_data('dipole')
            tdip = np.array([np.linalg.norm(tr_dipole[:,0,j])
                             for j in range(1, glbl.properties['n_states'])])
            log.print_message('general',
                             ['Initializing trajectory '+str(i)+
                              ' to state '+str(np.argmax(tdip)+1)+
                              ' | tr. dipople array='+np.array2string(tdip, \
                              formatter={'float_kind':lambda x: "%.4f" % x})])
            master.traj[i].state = np.argmax(tdip)+1
    elif len(glbl.properties['init_state']) == master.n_traj():
        # use "init_state" to set the initial state
        for i in range(master.n_traj()):
            istate = glbl.properties['init_state'][i]
            if istate < 0:
                master.traj[i].state = glbl.properties['n_states'] + istate
            else:
                master.traj[i].state = istate
    else:
        raise ValueError('Ambiguous initial state assignment.')


def set_initial_amplitudes(master):
    """Sets the initial amplitudes."""
    # if init_amp_overlap is set, overwrite 'amplitudes' that was
    # set in nomad.input
    if glbl.properties['init_amp_overlap']:
        origin = make_origin_traj()

        # Calculate the initial expansion coefficients via projection onto
        # the initial wavefunction that we are sampling
        ovec = np.zeros(master.n_traj(), dtype=complex)
        for i in range(master.n_traj()):
            ovec[i] = glbl.modules['integrals'].nuc_overlap(master.traj[i], origin)
        smat = np.zeros((master.n_traj(), master.n_traj()), dtype=complex)
        for i in range(master.n_traj()):
            for j in range(i+1):
                smat[i,j] = glbl.modules['integrals'].traj_overlap(master.traj[i],
                                                                   master.traj[j])
                if i != j:
                    smat[j,i] = smat[i,j].conjugate()
        sinv = sp_linalg.pinvh(smat)
        glbl.properties['init_amps'] = np.dot(sinv, ovec)

    # if we didn't set any amplitudes, set them all equal -- normalization
    # will occur later
    elif len(glbl.properties['init_amps']) == 0:
        glbl.properties['init_amps'] = np.ones(master.n_traj(),dtype=complex)

    # if we don't have a sufficient number of amplitudes, append
    # amplitudes with "zeros" as necesary
    elif len(glbl.properties['init_amps']) < master.n_traj():
        dif = master.n_traj() - len(glbl.properties['init_amps'])
        glbl.properties['init_amps'].extend([0+0j for i in range(dif)])

    # finally -- update amplitudes in the bundle
    for i in range(master.n_traj()):
        master.traj[i].update_amplitude(glbl.properties['init_amps'][i])


def save_initial_wavefunction(master):
    """Sets the intial t=0 bundle in order to compute the autocorrelation
    function for subsequent time steps"""
    glbl.modules['wfn0'] = master.copy()
    # change the trajectory labels in this bundle to differentiate
    # them from trajctory labels in the master bundle. This avoids
    # cache collisions between trajetories in 'bundle0' and trajectories
    # in 'master'
    for i in range(glbl.modules['wfn0'].n_traj()):
        new_label = str(glbl.modules['wfn0'].traj[i].label)+'_0'
        glbl.modules['wfn0'].traj[i].label = new_label


def virtual_basis(master):
    """Add virtual basis funcions.

    If additional virtual basis functions requested, for each trajectory
    in bundle, add aditional basis functions on other states with zero
    amplitude.
    """
    for i in range(master.n_traj()):
        for j in range(glbl.properties['n_states']):
            if j != master.traj[i].state:
                new_traj = master.traj[i].copy()
                new_traj.amplitude = 0j
                new_traj.state = j
                master.add_trajectory(new_traj)


def make_origin_traj():
    """Construct a trajectory basis function at the origin
    specified in the input files"""
    coords = glbl.properties['init_coords']
    ndim = coords.shape[-1]
    m_vec = glbl.properties['crd_masses']
    w_vec = glbl.properties['crd_widths']
    x_vec = coords[0,0]
    p_vec = coords[0,1]

    origin = trajectory.Trajectory(glbl.properties['n_states'], ndim,
                                   width=w_vec, mass=m_vec, parent=0,
                                   kecoef=glbl.modules['integrals'].kecoef)

    origin.update_x(x_vec)
    origin.update_p(p_vec)
    origin.state = 0
    # if we need pes data to evaluate overlaps, determine that now
    if glbl.modules['integrals'].overlap_requires_pes:
        evaluate.update_pes_traj(origin)

    return origin
