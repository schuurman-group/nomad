"""
Routines for generating and sampling a Wigner vibrational distribution.
"""
import random
import numpy as np
import src.fmsio.glbl as glbl
import src.dynamics.utilities as utils
import src.basis.particle as particle
import src.basis.trajectory as trajectory
import src.interfaces.vcham.hampar as ham


def sample_distribution(master):
    """Samples a v=0 Wigner distribution

    Modified to allow for the sampling of mass- and frequency-scaled
    normal mode coordinates in the case that a vibronic coupling
    Hamiltonian is being used. (SN 15/06/2016)
    """
    # Set the coordinate type: Cartesian or normal mode coordinates
    if glbl.fms['interface'] == 'vibronic':
        coordtype = 'normal'
    else:
        coordtype = 'cart'

    # Read the geometry.dat file
    amps, phase_gm = utils.load_geometry()

    # if multiple geometries in geometry.dat -- just take the first one
    natm = int(len(phase_gm)/len(amps))
    geom = [phase_gm[i] for i in range(natm)]

    # Read the hessian.dat file (Cartesian coordinates only)
    if coordtype == 'cart':
        hessian = utils.load_hessian()

    origin_traj = trajectory.Trajectory(glbl.fms['n_states'],
                                        particles=phase_gm,
                                        parent=0)

    dim = phase_gm[0].dim

    # If Cartesian coordinates are being used, then set up the
    # mass-weighted Hessian and diagonalise to obtain the normal modes
    # and frequencies
    if coordtype == 'cart':
        masses  = np.asarray([phase_gm[i].mass for i in range(natm)
                              for j in range(dim)], dtype=float)
        invmass = np.asarray([1./ np.sqrt(masses[i]) if masses[i] != 0.
                              else 0 for i in range(len(masses))], dtype=float)
        mw_hess = invmass * hessian * invmass[:,np.newaxis]
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
        modes = np.asarray(mode_list).transpose()
        # confirm that modes * tr(modes) = 1
        m_chk = np.dot(modes, np.transpose(modes))

    # If normal modes are being used, set the no. modes
    # equal to the number of active modes of the model
    # Hamiltonian and load the associated frequencies
    if coordtype == 'normal':
        n_modes = ham.nmode_active
        freqs = ham.freq

    # loop over the number of initial trajectories
    max_try = 1000
    for i in range(glbl.fms['n_init_traj']):
        delta_x = np.zeros(n_modes)
        delta_p = np.zeros(n_modes)
        disp_gm = [particle.copy_part(phase_gm[j]) for j in range(natm)]
        for j in range(n_modes):
            alpha   = 0.5 * freqs[j]
            sigma_x = np.sqrt(0.25 / alpha)
            sigma_p = np.sqrt(alpha)
            itry = 0
            while itry <= max_try:
                dx = random.gauss(0., sigma_x)
                dp = random.gauss(0., sigma_p)
                itry += 1
                if utils.mode_overlap(alpha, dx, dp) > glbl.fms['init_mode_min_olap']:
                    break
            if utils.mode_overlap(alpha, dx, dp) < glbl.fms['init_mode_min_olap']:
                print('Cannot get mode overlap > ' +
                      str(glbl.fms['init_mode_min_olap']) +
                      ' within ' + str(max_try) + ' attempts. Exiting...')
            delta_x[j] = dx
            delta_p[j] = dp

        # If Cartesian coordinates are being used, displace along each
        # normal mode to generate the final geometry...
        if coordtype == 'cart':
            disp_x = np.dot(modes, delta_x) / np.sqrt(masses)
            disp_p = np.dot(modes, delta_p) / np.sqrt(masses)

            for j in range(len(disp_gm)):
                disp_gm[j].x[:] += disp_x[j*dim:(j+1)*dim]
                disp_gm[j].p[:] += disp_p[j*dim:(j+1)*dim]

        # ... else if mass- and frequency-scaled normal modes are
        # being used, then take the frequency-scaled normal mode
        # displacements and momenta as the inital point in phase
        # space
        elif coordtype == 'normal':
            disp_x = delta_x * np.sqrt(freqs)
            disp_p = delta_p * np.sqrt(freqs)

            for j in range(natm):
                disp_gm[j].x[:] += disp_x[j*dim:(j+1)*dim]
                disp_gm[j].p[:] += disp_p[j*dim:(j+1)*dim]

        new_traj = trajectory.Trajectory(glbl.fms['n_states'],
                                         particles=disp_gm,
                                         parent=0)
        # with unit amplitude
        new_traj.amplitude = new_traj.overlap(origin_traj)
        master.add_trajectory(new_traj)

    # state of trajectory not set, return False
    return False
