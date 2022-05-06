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


def init_wavefunction():
    """Initializes the trajectories."""
    # initialize the interface we'll be using the determine the
    # the PES. There are some details here that trajectories
    # will want to know about

    log.print_message('string',['\n **************\n'+
                                  ' initialization\n'+
                                  ' **************\n'])

    # creat the wave function instance
    glbl.modules['wfn']        = wavefunction.Wavefunction()

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
    # this is messy -- should be re-worked mvoing forward...
    if glbl.methods['interface'] == 'vibronic':
        # if normal modes, read frequencies from vibronic interface
        kecoef = 0.5 * glbl.modules['interface'].ham.freq
    else:
        kecoef = 0.5 / glbl.properties['crd_masses']
    glbl.modules['integrals']  = integral.Integral(kecoef,
                                                   glbl.methods['ansatz'],
                                                   glbl.methods['integral_eval'])

    # this is a temporary hack
    if glbl.methods['integral_eval'] == 'lvc_exact':
        glbl.modules['integrals'].ints.init_parameters(
                                   glbl.properties['crd_widths'])


    # now load the initial trajectories into the bundle
    if glbl.properties['restart']:

        # print to logfile that we're restarting simulation
        log.print_message('string',[' restarting simulation from checkpoint file: '
                                     +str(glbl.paths['chkpt_file'])+'\n'])

        # retrieve current wave function, no arguments defaults to most recent simulation
        wfn0  = None
        ints0 = None
        if glbl.mpi['rank'] == 0:
            [glbl.modules['wfn'], chkpt_ints] = checkpoint.retrieve_simulation(time=glbl.properties['restart_time'])
            [wfn0, ints0]                     = checkpoint.retrieve_simulation(time=0.)
            if chkpt_ints is not None:
                glbl.modules['integrals'] = chkpt_ints

        # only root reads checkpoint file -- then broadcasts contents to other proceses
        if glbl.mpi['parallel']:
            # synchronize tasks
            glbl.mpi['comm'].barrier()
            glbl.modules['wfn']       = glbl.mpi['comm'].bcast(glbl.modules['wfn'], root=0)
            wfn0                      = glbl.mpi['comm'].bcast(wfn0, root=0)
            if glbl.modules['integrals'].require_centroids:
                glbl.modules['integrals'].centroid_required = glbl.mpi['comm'].bcast(glbl.modules['integrals'].centroid_required, root=0)
                glbl.modules['integrals'].centroids         = glbl.mpi['comm'].bcast(glbl.modules['integrals'].centroids, root=0)

        # save copy of t=0 wfn for autocorrelation function
        save_initial_wavefunction(wfn0)

        # check that we have all the data we need to propagate the first step -- otherwise, we
        # need to update the potential
        update_surface = False
        for i in range(glbl.modules['wfn'].n_traj()):
            if glbl.modules['wfn'].traj[i].alive and glbl.modules['wfn'].traj[i].active:
                pes_data = glbl.modules['wfn'].traj[i].pes
                if ('potential' not in pes_data.avail_data() or
                    'derivative' not in pes_data.avail_data()):
                    update_suface = True
            if glbl.modules['integrals'].require_centroids:
                for j in range(i):
                    pes_data = glbl.modules['integrals'].centroids[i][j].pes
                    if glbl.modules['integrals'].centroid_required[i][j]:
                        if ('potential' not in pes_data.avail_data() and
                            glbl.modules['wfn'].traj[i].state ==
                            glbl.modules['wfn'].traj[j].state):
                            update_surface = True
                        if ('derivative' not in pes_data.avail_data() and
                            glbl.modules['wfn'].traj[i].state !=
                            glbl.modules['wfn'].traj[j].state):
                            update_surface = True
        if update_surface:
            evaluate.update_pes(glbl.modules['wfn'])
            # update the couplings for all the trajectories
            for i in range(glbl.modules['wfn'].n_traj()):
                glbl.modules['interface'].evaluate_coupling(glbl.modules['wfn'].traj[i])

        # build the necessary matrices
        glbl.modules['matrices'].build(glbl.modules['wfn'], glbl.modules['integrals'])
        glbl.modules['wfn'].update_matrices(glbl.modules['matrices'])

    else:

        # iterate through initial trajectories and set gm/momt and initial
        # state, and then check that any constraints are satisfied
        for icond in range(glbl.properties['n_init_traj']):

            accept = False
            while not accept:
                # first generate the initial nuclear coordinates and momenta
                # and add the resulting trajectories to the bundle
                trial_traj = glbl.modules['init_conds'].gen_initial_coords(icond)
                # set the initial state of the trajectories in bundle. This may
                # require evaluation of electronic structure
                trial_traj = set_initial_state(trial_traj, icond)
                # check if this a valid starting geometry and state
                accept = check_initial_condition(trial_traj)

            # if accepted, add to wfn
            glbl.modules['wfn'].add_trajectory(trial_traj)

        # set the initial amplitudes of the basis functions
        set_initial_amplitudes(glbl.modules['wfn'])

        # add virtual basis functions, if desired
        if glbl.properties['virtual_basis']:
            virtual_basis(glbl.modules['wfn'])

        # update the integrals
        glbl.modules['integrals'].update(glbl.modules['wfn'])

        # once phase space position and states of the basis functions
        # are set, update the potential information
        evaluate.update_pes(glbl.modules['wfn'])

        # update the couplings for all the trajectories
        for i in range(glbl.modules['wfn'].n_traj()):
            glbl.modules['interface'].evaluate_coupling(glbl.modules['wfn'].traj[i])

        # compute the hamiltonian matrix...
        glbl.modules['matrices'].build(glbl.modules['wfn'], glbl.modules['integrals'])
        glbl.modules['wfn'].update_matrices(glbl.modules['matrices'])

        # so that we may appropriately renormalize to unity
        glbl.modules['wfn'].renormalize()

        # this is the bundle at time t=0.  Save in order to compute auto
        # correlation function
        save_initial_wavefunction(glbl.modules['wfn'])

    # write the wavefunction to the archive
    if glbl.mpi['rank'] == 0:
        checkpoint.archive_simulation(glbl.modules['wfn'], glbl.modules['integrals'])

    log.print_message('string',['\n ***********\n'+
                                  ' propagation\n'+
                                  ' ***********\n\n'])

    log.print_message('t_step', [glbl.modules['wfn'].time, glbl.properties['default_time_step'],
                                      glbl.modules['wfn'].nalive])

    return glbl.modules['wfn'].time


#---------------------------------------------------------------------------
#
# Private routines
#
#----------------------------------------------------------------------------
def check_initial_condition(trial_traj):
    """checks to see if all constraints on an initial condition 
       is satisified"""

    passed     = True
    chk_funcs  = ['square']
    chk_params = [2]

    func       = glbl.properties['filter_func']
    params     = glbl.properties['filter_params']

    if func is not None: 
        if func in chk_funcs:
           f_name = chk_funcs.index(func)
           if len(params) == chk_params[f_name]:

               # if square function
               if f_name == 0:
                   istate = trial_traj.state
                   e_low   = float(params[0] - 0.5*params[1])
                   e_high  = float(params[0] + 0.5*params[1])
                   delta_e = float(trial_traj.energy(istate) - 
                                   trial_traj.energy(0))
                   passed = delta_e >= e_low and delta_e <= e_high
                   # if condition rejected, print message
                   if not passed:
                       log.print_message('general',
                                [str(delta_e)+' not within ' + 
                                str([e_low,e_high])+', retrying...'])
                   else:
                       log.print_message('general',
                                [str(delta_e)+' is within ' + 
                                str([e_low,e_high])+', accpeted'])

           # else number of params is wrong
           else:
               raise ValueError('number of filter params wrong'+
                             str(len(params))+
                             '!='+ str(len(chk_params[f_name])))

        #else the function function is not recognized
        else:
           raise ValueError('filter function not recognized: ' +
                            str(func)+' not in '+str(chk_funcs))

    return passed


def set_initial_state(trial_traj, icond):
    """Sets the initial state of the trajectories in the bundle."""

    if glbl.properties['init_brightest']:
        # initialize to the state with largest transition dipole moment
        # set all states to the ground state
        trial_traj.state = 0
        # compute transition dipoles
        evaluate.update_pes_traj(trial_traj)

        # set the initial state to the one with largest t. dip.
        if 'dipole' not in trial_traj.pes.avail_data():
            raise KeyError('trajectory '+str(i)+
                           ': Cannot set state by transition moments - '+
                           'dipole not in pes.avail_data()')

        # load transition dipoles
        tr_dipole = trial_traj.pes.get_data('dipole')

        # if "bright_states" is set, only consider states in the array,
        # else, consider transition dipoles to all states in simulation.
        if glbl.properties['bright_states'] is None:
            br_states = [i for i in range(1,glbl.properties['n_states'])]
        else:
            br_states = glbl.properties['bright_states']
        tdip = np.array([np.linalg.norm(tr_dipole[:,0,j])
                         for j in br_states])
        trial_traj.state = br_states[np.argmax(tdip)]
        log.print_message('general',
                         ['Initializing trajectory '+str(icond)+
                          ' to state '+str(trial_traj.state)+
                          ' | tr. dipople array='+np.array2string(tdip, \
                          formatter={'float_kind':lambda x: "%.4f" % x})])

    elif len(glbl.properties['init_state']) > icond:
        # use "init_state" to set the initial state
        istate = glbl.properties['init_state'][icond]
        # if istate is less than zero, determine state number 'from the
        # right'. I forget why this would be desirable...
        if istate < 0:
            trial_traj.state = glbl.properties['n_states'] + istate
        else:
            trial_traj.state = istate
    else:
        raise ValueError('Ambiguous initial state assignment.')

    return trial_traj


def set_initial_amplitudes(wfn):
    """Sets the initial amplitudes."""
    # if init_amp_overlap is set, overwrite 'amplitudes' that was
    # set in nomad.input

    # initial conditions are explicitly specified, ignore init_amp_overlap
    # as the geometry is not the "origin"
    init_mod  = glbl.modules['init_conds'].__name__
    init_type = init_mod[init_mod.rfind('.')+1:]
    if glbl.properties['init_amp_overlap'] and init_type != 'explicit':
        origin = make_origin_traj()

        # Calculate the initial expansion coefficients via projection onto
        # the initial wavefunction that we are sampling
        ovec = np.zeros(wfn.n_traj(), dtype=complex)
        for i in range(wfn.n_traj()):
            ovec[i] = glbl.modules['integrals'].nuc_overlap(wfn.traj[i], origin)
        smat = np.zeros((wfn.n_traj(), wfn.n_traj()), dtype=complex)
        for i in range(wfn.n_traj()):
            for j in range(i+1):
                smat[i,j] = glbl.modules['integrals'].traj_overlap(wfn.traj[i],
                                                                   wfn.traj[j])
                if i != j:
                    smat[j,i] = smat[i,j].conjugate()
        sinv = sp_linalg.pinvh(smat)
        glbl.properties['init_amps'] = np.dot(sinv, ovec)

    # if we didn't set any amplitudes, set them all equal, normalization
    # will occur later
    elif not isinstance(glbl.properties['init_amps'],
                                               (list, np.ndarray)):
        glbl.properties['init_amps'] = np.ones(wfn.n_traj(),
                                               dtype=complex)

    # if we don't have a sufficient number of amplitudes, append
    # amplitudes with "zeros" as necesary
    elif len(glbl.properties['init_amps']) < wfn.n_traj():
        dif = wfn.n_traj() - len(glbl.properties['init_amps'])
        glbl.properties['init_amps'].extend([0+0j for i in range(dif)])

    # finally -- update amplitudes in the bundle
    for i in range(wfn.n_traj()):
        wfn.traj[i].update_amplitude(glbl.properties['init_amps'][i])


def save_initial_wavefunction(wfn):
    """Sets the intial t=0 bundle in order to compute the autocorrelation
    function for subsequent time steps"""
    glbl.modules['wfn0'] = wfn.copy()
    # change the trajectory labels in this bundle to differentiate
    # them from trajctory labels in the wfn bundle. This avoids
    # cache collisions between trajetories in 'bundle0' and trajectories
    # in 'wfn'
    for i in range(glbl.modules['wfn0'].n_traj()):
        new_label = str(glbl.modules['wfn0'].traj[i].label)+'_0'
        glbl.modules['wfn0'].traj[i].label = new_label


def virtual_basis(wfn):
    """Add virtual basis funcions.

    If additional virtual basis functions requested, for each trajectory
    in bundle, add aditional basis functions on other states with zero
    amplitude.
    """
    for i in range(wfn.n_traj()):
        for j in range(glbl.properties['n_states']):
            if j != wfn.traj[i].state:
                new_traj = wfn.traj[i].copy()
                new_traj.amplitude = 0j
                new_traj.state = j
                wfn.add_trajectory(new_traj)


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
