"""
Module for reading and interpreting adiabatic populations, including
finding transferred population and fitting population curves.
"""
import os
import numpy as np
from scipy.optimize import curve_fit
import nomad.analysis.fileio
import nomad.analysis.fitting


def error_amps(stamps, nboot=1000, bthrsh=1e-3):
    """Calculates the amplitude errors using the bootstrap method.
    A random set of seeds are sampled until the bootstrap average
    converges to the true average or the maximum number of iterations
    is reached.
    """
    nseed = len(stamps)
    bootsamp = np.random.randint(nseed, size=(nboot, nseed))
    tavg = np.average(stamps, axis=0)
    bavg = np.average(stamps[bootsamp[0]], axis=0)
    bdel = np.zeros_like(stamps[0])
    for i in range(1, nboot):
        bstamp = np.average(stamps[bootsamp[i]], axis=0)
        ei = bstamp - bavg
        bavg += ei / (i + 1)
        bdel += ei * (bstamp - bavg)
        if np.all(np.abs(bavg - tavg) < bthrsh):
            break
    #print('n_boot = {:d}'.format(i+1))
    if i+1 == nboot:
        print('Warning: bootstrap not converged.')
        print('max absolute error = '
              '{:7.4e}'.format(np.max(np.abs(bavg - tavg))))
    return bavg, np.sqrt(bdel / i)


def get_spawn_info(fname, first_lbl=1, noparent=False):
    """Reads the trajectory labels and states for parent and child of
    a given trajectory as well as the seed."""
    dsplit = fname.split('/')
    seed = int(dsplit[-2].split('.')[1])
    cid = int(dsplit[-1].split('.')[1])
    if cid > first_lbl and not noparent:
        try:
            spawnf = fname.replace('TrajDump', 'Spawn').replace('trajectory',
                                                                'spawn')
            with open(spawnf, 'r') as f:
                f.readline()
                comment = f.readline().split()
            cstate = int(comment[3]) - 1
            pid = int(comment[6])
            pstate = int(comment[9]) - 1
        except FileNotFoundError:
            # assume it is also a parent
            with open(fname, 'r') as f:
                f.readline()
                alldat = f.readline().split()
            cstate = int(alldat[-1].split('.')[0]) - 1
            pid = -1
            pstate = first_lbl
    else:
        with open(fname, 'r') as f:
            f.readline()
            alldat = f.readline().split()
        cstate = int(alldat[-1].split('.')[0]) - 1
        pid = -1
        pstate = first_lbl

    return np.array([seed, cid, cstate, pid, pstate], dtype=int)


def state_mask(traj_info, statei=None, statef=None):
    """Returns a boolean mask for states that have trajectory info
    matching given state indices and aren't initial trajectories."""
    ntraj = len(traj_info)
    mask0 = traj_info[:,3] >= 0
    if statei is not None:
        mask1 = traj_info[:,4] == statei
    else:
        mask1 = np.ones(ntraj, dtype=bool)
    if statef is not None:
        mask2 = traj_info[:,2] == statef
    else:
        mask2 = np.ones(ntraj, dtype=bool)
    mask12 = np.logical_and(mask1, mask2)
    return np.logical_and(mask0, mask12)


def integ_spawn_pop(amps, times, traj_info, statei=None, statef=None, maxt=400):
    """Returns the population transferred at each spawning event using
    the integration method.
    The change in population of the child is the negative of the change
    in population of the parent (assuming no effects from other
    trajectories). Thus, taking the negative product of the two
    derivatives gives a correlated function,
        (dp/dt)^2 = -(dp_c/dt)*(dp_p/dt),
    where the subscripts c and p correspond to child and parent. If
    negative values of (dp/dt)^2 are neglected, the total transferred
    population can be found by:
        p_trans = int[ dt * sgn(dp_c/dt) * sqrt((dp/dt)^2) ],
    where int represents integration over all time and sgn represents
    the sign function.
    """
    ntraj = len(amps)
    pops = np.zeros(ntraj)
    mask = state_mask(traj_info, statei=statei, statef=statef)
    irange = np.arange(ntraj)[mask]

    for i in irange:
        # get parent trajectory index and match times
        pmask = np.logical_and(traj_info[:,0] == traj_info[i,0],
                               traj_info[:,1] == traj_info[i,3])
        if not np.any(pmask):
            raise ValueError('Parent with index {:d}'.format(traj_info[i,3]) +
                             ' not found for trajectory ' +
                             '{:d} in seed {:d}.'.format(traj_info[i,1],
                                                         traj_info[i,0]))
        j = np.argmax(pmask)
        tshft = np.argmin(np.abs(times[j] - times[i][0]))
        lm = min(len(times[i]), len(times[j])-tshft, maxt)

        # get the child and parent populations
        pc = amps[i][:lm]
        pp = amps[j][tshft:lm+tshft]

        # take the derivatives
        dpc = pc[1:] - pc[:-1]
        dpp = pp[1:] - pp[:-1]

        # multiply the derivatives, take the sqrt and unwrap the sign
        dp2 = -dpc*dpp
        dp2[dp2 < 0] = 0
        dp = np.sign(dpc) * np.sqrt(dp2)

        # integrate from the inital child population
        pops[i] = pc[0] + np.sum(dp)

    return pops, mask


def thrsh_spawn_pop(amps, times, traj_info, inthrsh=5e-4, fithrsh=1e-4, nbuf=4):
    """Returns the population tranferred at each spawning event using
    the threshold method.
    A population event generally corresponds to a nearly stepwise
    change in the population, and thus a spike in its derivative. When
    dp/dt goes above a specified value, that is marked at the inital time
    and a final time is found where dp/dt falls below another given
    threshold.
    To correct for possible turning points, a buffer window
    can be chosen such that all points in the buffer must fall below the
    threshold before the final time is chosen.
    """
    ntraj = len(amps)
    pops = np.zeros(ntraj)

    for i in range(ntraj):
        if traj_info[i,3] >= 0:
            p = amps[i]
            t = times[i]

            # get the derivative, population range and number of time steps
            dp = (p[1:] - p[:-1]) / (t[1:] - t[:-1])
            rng = np.ptp(p)
            nt = len(dp)

            # find the time where dp exceeds inthrsh
            for iin in range(nt):
                if dp[iin] > inthrsh*rng:
                    break

            # find the time where dp falls below fithrsh for nbuf steps
            for iout in range(iin, nt-nbuf):
                comp = dp[iout:iout+nbuf]
                if max(comp) < fithrsh*rng:
                    iout += 1
                    break

            pops[i] = p[iout] - p[iin]

    return pops


def import_func(funcname):
    """Returns a fitting function from the function name."""
    if 'delay_' in funcname:
        basename = funcname.replace('delay_', '')
        return fitting.add_delay(getattr(fitting, basename))
    else:
        return getattr(fitting, funcname)


def fit_function(func, times, amps, p0, err=None, ethrsh=1e-5):
    """Fits amplitudes to a given exponential decay function.
    The type of fit depends on the function specified. If a 'single_'
    function is used, only the ground state is fit. For any larger function,
    if the number of adiabatic states is greater than the number of fit
    states, the highest adiabatic state populations are summed for the
    fit.
    """
    abssig = err is not None
    if not abssig:
        fit_err = None

    nr = len(func(0, *p0))
    t = np.tile(times, nr)

    if len(amps) == nr:
        fit_amps = amps.ravel()
        if abssig:
            fit_err = err.ravel() + ethrsh
    elif len(amps) > nr:
        # combine amplitudes for highest states
        if nr == 1:
            fit_amps = amps[0]
            if abssig:
                fit_err = err[0] + ethrsh
        else:
            fit_amps = np.vstack((amps[:nr-1], np.sum(amps[nr-1:], axis=0))).ravel()
            if abssig:
                fit_err = np.vstack((err[:nr-1], np.sqrt(np.sum(err[nr-1:]**2, axis=0))))
                fit_err = fit_err.ravel() + ethrsh
    else:
        raise ValueError('less states in amplitudes than in fitting function')

    popt, pcov = curve_fit(fitting.ravelf(func), t, fit_amps,
                           p0=p0, sigma=fit_err, absolute_sigma=abssig)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def write_fit(funcname, popt, perr, outfname):
    """Writes fit information to an output file.
    This should be generalized to accept more than one set of fit values.
    """
    lbls = fitting.get_labels(funcname)

    with open(outfname, 'w') as f:
        f.write('       ')
        f.write(''.join(['{:>12s}'.format(fv) for fv in lbls]) + '\n')
        f.write('Fit    ')
        f.write(''.join(['{:12.4e}'.format(p) for p in popt]) + '\n')
        f.write('Error  ')
        f.write(''.join(['{:12.4e}'.format(p) for p in perr]) + '\n')
