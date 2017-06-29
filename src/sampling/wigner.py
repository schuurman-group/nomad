"""
Routines for generating and sampling a Wigner vibrational distribution.
"""
import sys
import random
import numpy as np
import scipy.linalg as sp_linalg
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.dynamics.surface as surface
import src.basis.trajectory as trajectory
integrals = __import__('src.integrals.'+glbl.propagate['integrals'],fromlist=['a'])

def set_initial_coords(widths, geoms, momenta, master):
    """Samples a v=0 Wigner distribution
    """

    # Compression parameter
    beta = glbl.sampling['distrib_compression']

    # Set the coordinate type: Cartesian or normal mode coordinates
    if glbl.interface['interface'] == 'vibronic':
        import src.interfaces.vibronic as interface
        coordtype = 'normal'
        ham = interface.ham
    else:
        coordtype = 'cart'

    # if multiple geometries in geometry.dat -- just take the first one
    ngeoms  = len(geoms)
    crd_dim = int(len(geoms[0][0]))
    ndim    = int(len(geoms[0]) * crd_dim )

    x_ref = np.array()
    p_ref = np.array()
    w_vec = np.array([widths[i] for j in range(crd_dim) for i in range(len(widths))])

    # Read the hessian.dat file (Cartesian coordinates only)
    if coordtype == 'cart':
        hessian = fileio.read_hessian()

    origin_traj = trajectory.Trajectory(glbl.propagate['n_states'],
                                        ndim,
                                        width=w_vec,
                                        crd_dim=crd_dim,
                                        parent=0)

    origin_traj.update_x(geom_ref)
    origin_traj.update_p(mom_ref)
    # if we need pes data to evaluate overlaps, determine that now
    if integrals.overlap_requires_pes:
        surface.update_pes_traj(origin_traj)

    # If Cartesian coordinates are being used, then set up the
    # mass-weighted Hessian and diagonalise to obtain the normal modes
    # and frequencies
    if coordtype == 'cart':
        masses  = np.asarray([mass[i] for i in range(ndim)], dtype=float)
        invmass = np.asarray([1./ np.sqrt(masses[i]) if masses[i] != 0.
                              else 0 for i in range(len(masses))], dtype=float)
        mw_hess = invmass * hessian * invmass[:,np.newaxis]
        evals, evecs = sp_linalg.eigh(mw_hess)
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

    ## If normal modes are being used, set the no. modes
    ## equal to the number of active modes of the model
    ## Hamiltonian and load the associated frequencies
    #if coordtype == 'normal':
    #    n_modes = ham.nmode_active
    #    freqs = ham.freq

    # If normal modes are being used, set the no. modes
    # equal to the total number of modes of the model
    # Hamiltonian and load only the active frequencies
    if coordtype == 'normal':
        n_modes = ham.nmode_total
        freqs = np.zeros(n_modes)
        freqs[ham.mrange] = ham.freq

    # loop over the number of initial trajectories
    max_try = 1000
    ntraj  = glbl.sampling['n_init_traj']
    for i in range(ntraj):
        delta_x   = np.zeros(n_modes)
        delta_p   = np.zeros(n_modes)
        x_sample  = np.zeros(n_modes)
        p_sample  = np.zeros(n_modes)
        for j in range(n_modes):
            alpha   = 0.5 * freqs[j]
            if alpha > glbl.fpzero:
                sigma_x = beta*np.sqrt(0.25 / alpha)
                sigma_p = beta*np.sqrt(alpha)
                itry = 0
                while itry <= max_try:
                    dx = random.gauss(0., sigma_x)
                    dp = random.gauss(0., sigma_p)
                    itry += 1
                    if mode_overlap(alpha, dx, dp) > glbl.sampling['init_mode_min_olap']:
                        break
                if mode_overlap(alpha, dx, dp) < glbl.sampling['init_mode_min_olap']:
                    print('Cannot get mode overlap > ' +
                      str(glbl.sampling['init_mode_min_olap']) +
                      ' within ' + str(max_try) + ' attempts. Exiting...')
                delta_x[j] = dx
                delta_p[j] = dp

        # If Cartesian coordinates are being used, displace along each
        # normal mode to generate the final geometry...
        if coordtype == 'cart':
            disp_x = np.dot(modes, delta_x) / np.sqrt(masses)
            disp_p = np.dot(modes, delta_p) / np.sqrt(masses)

        # ... else if mass- and frequency-scaled normal modes are
        # being used, then take the frequency-scaled normal mode
        # displacements and momenta as the inital point in phase
        # space
        elif coordtype == 'normal':
            disp_x = delta_x * np.sqrt(freqs)
            disp_p = delta_p * np.sqrt(freqs)

        x_sample = geom_ref + disp_x
        p_sample = mom_ref  + disp_p

        # add new trajectory to the bundle
        new_traj = origin_traj.copy()
        new_traj.update_x(x_sample)
        new_traj.update_p(p_sample)

        # if we need pes data to evaluate overlaps, determine that now
        if integrals.overlap_requires_pes:
            surface.update_pes_traj(new_traj)

        # Add the trajectory to the bundle
        master.add_trajectory(new_traj)

    return

def mode_overlap(alpha, dx, dp):
    """Returns the overlap of Gaussian primitives

    Given a displacement along a set of x, p coordiantes (dx, dp), return
    the overlap of the resultant gaussian primitive with the gaussian primitive
    centered at (x0,p0) (integrate over x, independent of x0).
    """
    return abs(np.exp((-4.*alpha*dx**2 + 4.*1j*dx*dp -
                       (1./alpha)*dp**2) / 8.))

