"""
Routines for propagation with the 4th order Runge-Kutta-Fehlberg algorithm
with 5th order error estimator.

Adaptive Runge-Kutta-Fehlberg 4-5:
  x4(t+dt) = x(t) + dt*[(25/216)*kx1 + (1405/2565)*kx3 + (2197/4104)*kx4 -
                        (1/5)*kx5]
  p4(t+dt) = p(t) + dt*[(25/216)*kp1 + (1405/2565)*kp3 + (2197/4104)*kp4 -
                        (1/5)*kp5]
  x5(t+dt) = x(t) + dt*[(16/135)*kx1 + (6656/12825)*kx3 + (28561/56430)*kx4 -
                        (9/50)*kx5 + (2/55)*kx6]
  p5(t+dt) = p(t) + dt*[(16/135)*kp1 + (6656/12825)*kp3 + (28561/56430)*kp4 -
                        (9/50)*kp5 + (2/55)*kp6]

  ky1 = f[t, y(t)] = dy(t)/dt
  ky2 = f[t + (1/4)dt, y(t) + (1/4)ky1*dt]
  ky3 = f[t + (3/8)dt, y(t) + (3/32)ky1*dt + (9/32)ky2*dt]
  ky4 = f[t + (12/13)dt, y(t) + (1932/2197)ky1*dt - (7200/2197)ky2*dt +
                                (7296/2197)ky3*dt]
  ky5 = f[t + dt, y(t) + (439/216)ky1*dt - 8*ky2*dt + (3680/513)ky3*dt -
                         (845/4104)ky4*dt]
  ky6 = f[t + (1/2)dt, y(t) - (8/27)ky1*dt + 2*ky2*dt - (3544/2565)ky3*dt +
                              (1859/4104)ky4*dt - (11/40)ky5*dt]

  error = max(|x5-x4|, |p5-p4|)
  sfac  = S * min((tolerance/error)^(1/5), (tolerance/error)^(1/4))

S is a safety factor (~0.9). If error > tolerance, repeat with new
step size sfac*dt. Otherwise, use sfac*dt for next step. Update
position, momentum using x4, p4.
"""
import numpy as np
import src.fmsio.glbl as glbl
import src.dynamics.timings as timings
import src.dynamics.surface as surface


rk_ordr = 6
coeff = np.array([[1./4., 0., 0., 0., 0., 0.],
                  [3./32., 9./32., 0., 0., 0., 0.],
                  [1932./2197., -7200./2197., 7296./2197., 0., 0., 0.],
                  [439./216., -8., 3680./513., -845./4104., 0., 0.],
                  [-8./27., 2., -3544./2565., 1859./4104., -11./40., 0.]])
wgt_lo = np.array([25./216., 0., 1408./2565., 2197./4104., -1./5., 0.])
wgt_hi = np.array([16./135., 0., 6656./12825., 28561./56430., -9./50., 2./55.])

propphase = glbl.fms['phase_prop'] != 0
safety = 0.9
tol = 1e-6
h = None
h_traj = None


@timings.timed
def propagate_bundle(master, dt):
    """Propagates the Bundle object with RKF45."""
    global h
    ncrd = master.traj[0].dim
    kx = np.zeros((master.nalive, rk_ordr, ncrd)) # should use nactive, but it is set to 0 when entering coupling regions
    kp = np.zeros((master.nalive, rk_ordr, ncrd))
    kg = np.zeros((master.nalive, rk_ordr, ncrd))

    t = 0.
    if h is None:
        h = dt
    while t < dt:
        hstep = min(h, dt - t)
        for rk in range(rk_ordr):
            tmpbundle = master.copy()
            for i in range(tmpbundle.n_traj()):
                if tmpbundle.traj[i].active:
                    propagate_rk(tmpbundle.traj[i], hstep, rk,
                                 kx[i], kp[i], kg[i])

            # update the PES to evaluate new gradients
            if rk < rk_ordr - 1:
                surface.update_pes(tmpbundle, update_centroids=False)

        # calculate the 4th and 5th order changes and the error
        dx_lo = np.zeros((master.nalive, ncrd))
        dx_hi = np.zeros((master.nalive, ncrd))
        dp_lo = np.zeros((master.nalive, ncrd))
        dp_hi = np.zeros((master.nalive, ncrd))
        dg_lo = np.zeros((master.nalive, ncrd))
        dg_hi = np.zeros((master.nalive, ncrd))
        for i in range(master.n_traj()):
            if master.traj[i].active:
                dx_lo[i] = np.sum(wgt_lo[:,np.newaxis] * kx[i], axis=0)
                dx_hi[i] = np.sum(wgt_hi[:,np.newaxis] * kx[i], axis=0)
                dp_lo[i] = np.sum(wgt_lo[:,np.newaxis] * kp[i], axis=0)
                dp_hi[i] = np.sum(wgt_hi[:,np.newaxis] * kp[i], axis=0)

        if propphase:
            for i in range(master.n_traj()):
                if master.traj[i].active:
                    dg_lo[i] = np.sum(wgt_lo[:,np.newaxis] * kg[i], axis=0)
                    dg_hi[i] = np.sum(wgt_hi[:,np.newaxis] * kg[i], axis=0)

            err = np.max((np.abs(dx_hi-dx_lo), np.abs(dp_hi-dp_lo),
                          np.abs(dg_hi-dg_lo)))
        else:
            err = np.max((np.abs(dx_hi-dx_lo), np.abs(dp_hi-dp_lo)))

        if err > tol:
            # scale the time step and try again
            h = hstep * max(safety*(tol/err)**0.25, 0.1)
        else:
            # update the position and scale the time step
            for i in range(master.n_traj()):
                if master.traj[i].active:
                    master.traj[i].update_x(master.traj[i].x() + dx_lo[i])
                    master.traj[i].update_p(master.traj[i].p() + dp_lo[i])
                    if propphase:
                        master.traj[i].update_phase(master.traj[i].phase() +
                                                    dg_lo[i])
            surface.update_pes(master)
            t += h
            h *= min(safety*(tol/err)**0.2, 5.)


@timings.timed
def propagate_trajectory(traj, dt):
    """Propagates a single trajectory with RKF45."""
    global h_traj
    ncrd = traj.dim
    kx = np.zeros((rk_ordr, ncrd))
    kp = np.zeros((rk_ordr, ncrd))
    kg = np.zeros((rk_ordr, ncrd))

    t = 0.
    if h_traj is None:
        h_traj = dt
    while t < dt:
        hstep = min(h_traj, dt - t)
        for rk in range(rk_ordr):
            tmptraj = traj.copy()
            propagate_rk(tmptraj, hstep, rk, kx, kp, kg)

            # update the PES to evaluate new gradients
            if rk < rk_ordr - 1:
                surface.update_pes_traj(tmptraj)

        # calculate the 4th and 5th order changes and the error
        dx_lo = np.sum(wgt_lo[:,np.newaxis] * kx, axis=0)
        dx_hi = np.sum(wgt_hi[:,np.newaxis] * kx, axis=0)
        dp_lo = np.sum(wgt_lo[:,np.newaxis] * kp, axis=0)
        dp_hi = np.sum(wgt_hi[:,np.newaxis] * kp, axis=0)

        if propphase:
            dg_lo = np.sum(wgt_lo[:,np.newaxis] * kg, axis=0)
            dg_hi = np.sum(wgt_hi[:,np.newaxis] * kg, axis=0)
            err = np.max((np.abs(dx_hi-dx_lo), np.abs(dp_hi-dp_lo),
                          np.abs(dg_hi-dg_lo)))
        else:
            err = np.max((np.abs(dx_hi-dx_lo), np.abs(dp_hi-dp_lo)))

        if err > tol:
            # scale the time step and try again
            h_traj = hstep * max(safety*(tol/err)**0.25, 0.1)
        else:
            # update the position and scale the time step
            traj.update_x(traj.x() + dx_lo)
            traj.update_p(traj.p() + dp_lo)
            if propphase:
                traj.update_phase(traj.phase() + dg_lo)
            surface.update_pes_traj(traj)
            t += h_traj
            h_traj *= min(safety*(tol/err)**0.2, 5.)


def propagate_rk(traj, dt, rk, kxi, kpi, kgi):
    """Gets k values and updates the position and momentum by
    a single rk step."""
    # calculate the k values at this point
    kxi[rk] = dt * traj.velocity()
    kpi[rk] = dt * traj.force()
    if propphase:
        kgi[rk] = dt * traj.phase_dot()

    # update the position using previous k values, except for k6
    if rk < rk_ordr - 1:
        traj.update_x(traj.x() + np.sum(coeff[rk,:,np.newaxis]*kxi, axis=0))
        traj.update_p(traj.p() + np.sum(coeff[rk,:,np.newaxis]*kpi, axis=0))
        if propphase:
            traj.update_phase(traj.phase() +
                              np.sum(coeff[rk,:,np.newaxis]*kgi, axis=0))
