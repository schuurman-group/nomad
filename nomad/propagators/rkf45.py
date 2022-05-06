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
import nomad.core.glbl as glbl
import nomad.core.timings as timings
import nomad.core.surface as evaluate


rk_ordr = 6
coeff = np.array([[1./4., 0., 0., 0., 0., 0.],
                  [3./32., 9./32., 0., 0., 0., 0.],
                  [1932./2197., -7200./2197., 7296./2197., 0., 0., 0.],
                  [439./216., -8., 3680./513., -845./4104., 0., 0.],
                  [-8./27., 2., -3544./2565., 1859./4104., -11./40., 0.]])
wgt_lo = np.array([25./216., 0., 1408./2565., 2197./4104., -1./5., 0.])
wgt_hi = np.array([16./135., 0., 6656./12825., 28561./56430., -9./50., 2./55.])

propphase = glbl.properties['phase_prop']
safety = 0.9
tol = 1e-6

h      = None
h_wfn  = None
h_traj = None

@timings.timed
def propagate(q0, t_deriv, dt):
    """Propagates from q0 = q(t0) to q(t0+dt) using 4th-order Runge-Kutta
       method with error estimation. A reference to a function that
       evaluates time derivative at any point qi is also given"""
    global h
 
    ndim = len(q0)
    k    = np.zeros((rk_ordr, ndim), dtype = np.dtype(q0[0]))
    
    t    = 0.
    qt   = q0
    if h is None:
        h = dt
    while abs(t) < abs(dt):
        h_step = np.sign(dt) * min(abs(h), abs(dt - t))
        qk = qt.copy()
        for rk in range(rk_ordr):
            k[rk] = h_step * t_deriv(qk)[0]
            if rk < rk_ordr - 1:
                qk += np.sum(coeff[rk,:,np.newaxis]*k, axis=0)
        # calculate the 4th and 5th order changes and the error
        dq_lo = np.sum(wgt_lo[:,np.newaxis] * k, axis=0)
        dq_hi = np.sum(wgt_hi[:,np.newaxis] * k, axis=0)
        err = np.max(np.abs(dq_hi-dq_lo))

        if err > tol:
            # scale the time step and try again
            h = h_step * max(safety*(tol/err)**0.25, 0.1)
        else:
            # scale the time step and update the position
            t  += h
            err = max(err, tol*1e-5)
            h   = min(dt, h*safety*(tol/err)**0.2)
            qt += dq_lo

    return qt

@timings.timed
def propagate_wfn(wfn, dt):
    """Propagates the Bundle object with RKF45."""

    global h_wfn
    ncrd = wfn.traj[0].dim
    ntraj = wfn.n_traj()
    kx = np.zeros((ntraj, rk_ordr, ncrd))
    kp = np.zeros((ntraj, rk_ordr, ncrd))
    kg = np.zeros((ntraj, rk_ordr))

    t = 0.
    if h_wfn is None:
        h_wfn = dt
    while abs(t) < abs(dt):
        hstep = np.sign(dt) * min(abs(h_wfn), abs(dt - t))
        for rk in range(rk_ordr):
            tmp_wfn = wfn.copy()
            for i in range(ntraj):
                if tmp_wfn.traj[i].active:
                    propagate_rk(tmp_wfn.traj[i], hstep, rk,
                                 kx[i], kp[i], kg[i])

            # update the PES to evaluate new gradients
            if rk < rk_ordr - 1:
                evaluate.update_pes(tmp_wfn, update_integrals=False)

        # calculate the 4th and 5th order changes and the error
        dx_lo = np.zeros((wfn.nalive, ncrd))
        dx_hi = np.zeros((wfn.nalive, ncrd))
        dp_lo = np.zeros((wfn.nalive, ncrd))
        dp_hi = np.zeros((wfn.nalive, ncrd))
        dg_lo = np.zeros((wfn.nalive))
        dg_hi = np.zeros((wfn.nalive))
        for i in range(ntraj):
            if wfn.traj[i].active:
                dx_lo[i] = np.sum(wgt_lo[:,np.newaxis] * kx[i], axis=0)
                dx_hi[i] = np.sum(wgt_hi[:,np.newaxis] * kx[i], axis=0)
                dp_lo[i] = np.sum(wgt_lo[:,np.newaxis] * kp[i], axis=0)
                dp_hi[i] = np.sum(wgt_hi[:,np.newaxis] * kp[i], axis=0)

        if propphase:
            for i in range(ntraj):
                if wfn.traj[i].active:
                    dg_lo[i] = np.sum(wgt_lo * kg[i])
                    dg_hi[i] = np.sum(wgt_hi * kg[i])

            #print("a="+str(np.abs(dx_hi-dx_lo).flatten()))
            #print("b="+str(np.abs(dp_hi-dp_lo).flatten()))
            #print("c="+str(np.abs(dg_hi-dg_lo)))

            err = np.max(np.max(np.abs(dg_hi-dg_lo)), np.max((np.abs(dx_hi-dx_lo).flatten(), np.abs(dp_hi-dp_lo).flatten())))
        else:
            err = np.max((np.abs(dx_hi-dx_lo), np.abs(dp_hi-dp_lo)))

        if err > tol:
            # scale the time step and try again
            h_wfn = hstep * max(safety*(tol/err)**0.25, 0.1)
        else:
            # scale the time step and update the position
            t     += h_wfn
            err   = max(err, tol*1e-5)
            h_wfn = min(dt, h_wfn*safety*(tol/err)**0.2)
            for i in range(ntraj):
                if wfn.traj[i].active:
                    wfn.traj[i].update_x(wfn.traj[i].x() + dx_lo[i])
                    wfn.traj[i].update_p(wfn.traj[i].p() + dp_lo[i])
                    if propphase:
                        wfn.traj[i].update_phase(wfn.traj[i].phase() +
                                                    dg_lo[i])
            evaluate.update_pes(wfn, update_integrals=(abs(t)>=abs(dt)))


@timings.timed
def propagate_trajectory(traj, dt):
    """Propagates a single trajectory with RKF45."""
    global h_traj
    ncrd = traj.dim
    kx = np.zeros((rk_ordr, ncrd))
    kp = np.zeros((rk_ordr, ncrd))
    kg = np.zeros((rk_ordr))

    t = 0.
    if h_traj is None:
        h_traj = dt
    while abs(t) < abs(dt):
        hstep = np.sign(dt) * min(abs(h_traj), abs(dt - t))
        for rk in range(rk_ordr):
            tmptraj = traj.copy()
            propagate_rk(tmptraj, hstep, rk, kx, kp, kg)

            # update the PES to evaluate new gradients
            if rk < rk_ordr - 1:
                evaluate.update_pes_traj(tmptraj)

        # calculate the 4th and 5th order changes and the error
        dx_lo = np.sum(wgt_lo[:,np.newaxis] * kx, axis=0)
        dx_hi = np.sum(wgt_hi[:,np.newaxis] * kx, axis=0)
        dp_lo = np.sum(wgt_lo[:,np.newaxis] * kp, axis=0)
        dp_hi = np.sum(wgt_hi[:,np.newaxis] * kp, axis=0)

        if propphase:
            dg_lo = np.sum(wgt_lo * kg)
            dg_hi = np.sum(wgt_hi * kg)
            #err = np.max((np.abs(dx_hi-dx_lo), np.abs(dp_hi-dp_lo),
            #              np.abs(dg_hi-dg_lo)))
            err = np.max(np.abs(dg_hi-dg_lo),np.max((np.abs(dx_hi-dx_lo).flatten(), np.abs(dp_hi-dp_lo).flatten())))

        else:
            err = np.max((np.abs(dx_hi-dx_lo), np.abs(dp_hi-dp_lo)))

        if err > tol:
            # scale the time step and try again
            h_traj = hstep * max(safety*(tol/err)**0.25, 0.1)
        else:
            # scale the time step and update the position
            t     += h_traj
            err    = max(err, tol*1e-5)
            h_traj = min(dt, h_traj*safety*(tol/err)**0.2)
            traj.update_x(traj.x() + dx_lo)
            traj.update_p(traj.p() + dp_lo)
            if propphase:
                traj.update_phase(traj.phase() + dg_lo)
            evaluate.update_pes_traj(traj)


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
                              np.sum(coeff[rk]*kgi))
