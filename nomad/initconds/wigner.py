"""
Routines for generating and sampling a Wigner vibrational distribution.
"""
import numpy as np
import scipy.linalg as sp_linalg
import nomad.common.constants as constants
import nomad.core.glbl as glbl
import nomad.core.log as log
import nomad.core.trajectory as trajectory


def gen_initial_coords(cnt):
    """Samples a v=0 Wigner distribution."""
    # Set the coordinate type: Cartesian or normal mode coordinates
    if glbl.methods['interface'] == 'vibronic':
        coordtype = 'normal'
        ham       = glbl.modules['interface'].ham
    else:
        coordtype = 'cart'

    log.print_message('string',[' sampling from v=0 Wigner distribution.\n'])

    # if multiple geometries, just take the first one
    coords = glbl.properties['init_coords']
    ndim = coords.shape[-1]
    x_ref = coords[0,0]
    p_ref = coords[0,1]
    w_vec = glbl.properties['crd_widths']
    m_vec = glbl.properties['crd_masses']

    # create template trajectory basis function
    template = trajectory.Trajectory(glbl.properties['n_states'], ndim,
                                     width=w_vec,
                                     mass=m_vec,
                                     parent=0,
                                     kecoef=glbl.modules['integrals'].kecoef)
    template.update_x(x_ref)
    template.update_p(p_ref)

    # If Cartesian coordinates are being used, then set up the
    # mass-weighted Hessian and diagonalise to obtain the normal modes
    # and frequencies
    if coordtype == 'cart':
        iface_dict = glbl.sections[glbl.methods['interface']]
        hessian = iface_dict['hessian']
        invmass = np.asarray([1./ np.sqrt(m_vec[i]) if m_vec[i] != 0.
                              else 0 for i in range(len(m_vec))], dtype=float)
        mw_hess      = invmass * hessian * invmass[:,np.newaxis]
        evals, evecs = sp_linalg.eigh(mw_hess)
        f_cutoff     = 0.0001
        freq_list    = []
        mode_list    = []
        for i in range(len(evals)):
            if evals[i] >= 0 and np.sqrt(evals[i]) >= f_cutoff:
                freq_list.append(np.sqrt(evals[i]))
                mode_list.append(evecs[:,i].tolist())
        n_modes = len(freq_list)
        freqs = np.asarray(freq_list)
        modes = np.asarray(mode_list).transpose()
        # confirm that modes * tr(modes) = 1
        m_chk = np.dot(modes.T, modes)
        if not np.allclose(m_chk, np.eye(len(m_chk))):
            raise ValueError('Internal coordinates not orthonormal.')
        log.print_message('string',[' -- frequencies from hessian.dat --\n'])

    # If normal modes are being used, set the no. modes
    # equal to the total number of modes of the model
    # Hamiltonian and load only the active frequencies
    if coordtype == 'normal':
        n_modes = len(w_vec)
        # we multiply by 0.5 below -- multiply by 2 here (i.e. default width for
        # vibronic hamiltonans is 1/2, assuming frequency weighted coords
        freqs   = 2. * w_vec
        log.print_message('string',['\n -- widths employed in coordinate sampling --\n'])

    # write out frequencies
    fstr = '\n'.join(['{0:.5f} au == {1:10.1f} cm^-1'.format(freqs[j],freqs[j]*constants.au2cm)
                                                       for j in range(n_modes)])
    log.print_message('string',[fstr+'\n'])

    # loop over the number of initial trajectories
    max_try = 1000

    delta_x   = np.zeros(n_modes)
    delta_p   = np.zeros(n_modes)
    x_sample  = np.zeros(n_modes)
    p_sample  = np.zeros(n_modes)
    for j in range(n_modes):
        alpha   = 0.5 * freqs[j]
        if alpha > constants.fpzero:
            sigma_x = (glbl.properties['distrib_compression'] *
                       np.sqrt(0.25 / alpha))
            sigma_p = (glbl.properties['distrib_compression'] *
                       np.sqrt(alpha))
            itry = 0
            while itry <= max_try:
                dx = np.random.normal(0., sigma_x)
                dp = np.random.normal(0., sigma_p)
                itry += 1
                if mode_overlap(alpha, dx, dp) > glbl.properties['init_mode_min_olap']:
                    break
            if mode_overlap(alpha, dx, dp) < glbl.properties['init_mode_min_olap']:
                raise ValueError('Cannot get mode overlap > ' +
                                 str(glbl.properties['init_mode_min_olap']) +
                                 ' within ' + str(max_try) + ' attempts. Exiting...')
            delta_x[j] = dx
            delta_p[j] = dp

    # If Cartesian coordinates are being used, displace along each
    # normal mode to generate the final geometry...
    if coordtype == 'cart':
        disp_x = np.dot(modes, delta_x) / np.sqrt(m_vec)
        disp_p = np.dot(modes, delta_p) * np.sqrt(m_vec)

    # ... else if mass- and frequency-scaled normal modes are
    # being used, then take the frequency-scaled normal mode
    # displacements and momenta as the inital point in phase
    # space
    elif coordtype == 'normal':
        disp_x = delta_x
        disp_p = delta_p
        #disp_x = delta_x * np.sqrt(freqs)
        #disp_p = delta_p * np.sqrt(freqs)

    x_sample = x_ref + disp_x
    p_sample = p_ref + disp_p

    # add new trajectory to the bundle
    new_traj = template.copy()
    new_traj.update_x(x_sample)
    new_traj.update_p(p_sample)

    return new_traj


def mode_overlap(alpha, dx, dp):
    """Returns the overlap of Gaussian primitives

    Given a displacement along a set of x, p coordiantes (dx, dp), return
    the overlap of the resultant gaussian primitive with the gaussian primitive
    centered at (x0,p0) (integrate over x, independent of x0).
    """
    return abs(np.exp((-4.*alpha*dx**2 + 4.*1j*dx*dp -
                       (1./alpha)*dp**2) / 8.))
