"""
Routines for propagation with the Bulirsch-Stoer algorithm.

Bulirsch-Stoer (Modified Midpoint):
  x(t+dt) = 0.5*(x{n} + x{n-1} + h*f[t+H,x{n}])
  p(t+dt) = 0.5*(p{n} + p{n-1} + h*f[t+H,p{n}])

  h = H / n, n = 2, 4, 6, 8, ...
  f[t, y(t)] = dy(t)/dt
  y{0} = y(t)
  y{1} = y(t) + h*f[t, y(t)]
  y{m} = y{m-2} + 2*h*f[t+(m-1)*h, y{m-1}], m = 2, 3, ..., n

The number of steps n is increased in order to extrapolate the final
values y{m} for each n and estimate the error. The total step size H can
then be adjusted based on the error and the work involved in greater
values of n.

For more details, see http://apps.nrbook.com/fortran/index.html.
"""
import numpy as np
import nomad.simulation.glbl as glbl
import nomad.simulation.timings as timings
import nomad.simulation.evaluate as evaluate

propphase = glbl.propagate['phase_prop']

maxattempt = 8
tol = 1e-6
h = None
h_traj = None

# setup for work coefficients, correction factors and optimal column number
nstep = 2*np.arange(1, maxattempt + 2)
afac = np.cumsum(nstep) + 1
alpha = np.zeros((maxattempt, maxattempt))
for i in range(1, maxattempt):
    for k in range(i):
        alpha[k, i] = (0.25*tol) ** ((afac[k+1] - afac[i+1]) /
                                     ((2*k + 1)*(afac[i+1] - afac[1] + 1)))
for kopt in range(2, maxattempt):
    if afac[kopt] > afac[kopt-1]*alpha[kopt-2,kopt-1]:
        break
kmax = np.copy(kopt)
kopt_traj = np.copy(kopt)
kmax_traj = np.copy(kopt)


@timings.timed
def propagate_wfn(master, dt):
    """Propagates the Bundle object with BS."""
    global h
    ncrd  = master.traj[0].dim
    ntraj = master.n_traj()

    t = 0.
    if h is None:
        h = dt
    while abs(t) < abs(dt):
        hstep = np.sign(dt) * min(abs(h), abs(dt - t))
        reduced = False
        tsav = np.zeros(kmax)
        err = np.zeros(kmax)
        Tx = np.zeros((kmax, ntraj, ncrd))
        Tp = np.zeros((kmax, ntraj, ncrd))
        Tg = np.zeros((kmax, ntraj, ncrd))

        for k in range(kmax):
            tmp_wfn = master.copy()

            x0 = np.zeros((ntraj, ncrd))
            p0 = np.zeros((ntraj, ncrd))
            g0 = np.zeros((ntraj, ncrd))
            x1 = np.zeros((ntraj, ncrd))
            p1 = np.zeros((ntraj, ncrd))
            g1 = np.zeros((ntraj, ncrd))

            # step through n modified midpoint steps
            for n in range(nstep[k]):
                for i in range(ntraj):
                    if tmp_wfn.traj[i].active:
                        mm_step(tmp_wfn.traj[i], hstep/nstep[k], x0[i],
                                x1[i], p0[i], p1[i], g0[i], g1[i], n)
                evaluate.update_pes(tmp_wfn, update_integrals=False)

            # compute the modified midpoint estimate
            for i in range(ntraj):
                if master.traj[i].active:
                    x1[i] = 0.5*(x1[i] + x0[i] +
                                 hstep/nstep[k]*tmp_wfn.traj[i].velocity())
                    p1[i] = 0.5*(p1[i] + p0[i] +
                                 hstep/nstep[k]*tmp_wfn.traj[i].force())
                    if propphase:
                        g1[i] = 0.5*(g1[i] + g0[i] +
                                     hstep/nstep[k]*tmp_wfn.traj[i].phase_dot())

            # extrapolate from modified midpoint results
            poly_extrapolate(k, (hstep/nstep[k])**2, tsav, x1, p1, g1, Tx, Tp, Tg)
            if k > 0:
                errmax = np.amax((abs(Tx[k]), abs(Tp[k]), abs(Tg[k]))) / tol
                err[k-1] = (errmax / 0.25) ** (1. / (2.*k + 1.))
                if k >= kopt - 2:
                    if errmax > 1:
                        # scale the time step and try again
                        sfac, reduced = reduce_tstep(k, err)
                        if reduced:
                            red = max(1e-5, min(reduced, 0.7))
                            h = sfac * hstep
                            break
                    else:
                        # scale the time step if possible
                        t += h
                        h = increase_tstep(k, err, reduced) * hstep

                        # update to the final position
                        xnew = np.sum(Tx, axis=0)
                        pnew = np.sum(Tp, axis=0)
                        gnew = np.sum(Tg, axis=0)
                        for i in range(ntraj):
                            if master.traj[i].active:
                                master.traj[i].update_x(xnew[i])
                                master.traj[i].update_p(pnew[i])
                                if propphase:
                                    master.traj[i].update_phase(gnew[i])
                        evaluate.update_pes(master,
                                           update_integrals=(abs(t)>=abs(dt)))
                        break


@timings.timed
def propagate_trajectory(traj, dt):
    """Propagates a single trajectory with BS."""
    global h_traj
    ncrd = traj.dim

    t = 0.
    if h_traj is None:
        h_traj = dt
    while abs(t) < abs(dt):
        hstep = np.sign(dt) * min(abs(h_traj), abs(dt - t))
        reduced = False
        tsav = np.zeros(kmax_traj)
        err = np.zeros(kmax_traj)
        Tx = np.zeros((kmax_traj, ncrd))
        Tp = np.zeros((kmax_traj, ncrd))
        Tg = np.zeros((kmax_traj, ncrd))

        for k in range(kmax):
            tmptraj = traj.copy()

            x0 = np.zeros(ncrd)
            p0 = np.zeros(ncrd)
            g0 = np.zeros(ncrd)
            x1 = np.zeros(ncrd)
            p1 = np.zeros(ncrd)
            g1 = np.zeros(ncrd)

            # step through n modified midpoint steps
            for n in range(nstep[k]):
                mm_step(tmptraj, hstep/nstep[k], x0, x1, p0, p1, g0, g1, n)
                evaluate.update_pes_traj(tmptraj)

            # compute the modified midpoint estimate
            x1 = 0.5*(x1 + x0 + hstep/nstep[k]*tmptraj.velocity())
            p1 = 0.5*(p1 + p0 + hstep/nstep[k]*tmptraj.force())
            if propphase:
                g1 = 0.5*(g1 + g0 + hstep/nstep[k]*tmptraj.phase_dot())

            # extrapolate from modified midpoint results
            poly_extrapolate(k, (hstep/nstep[k])**2, tsav, x1, p1, g1, Tx, Tp, Tg)
            if k > 0:
                errmax = np.amax((abs(Tx[k]), abs(Tp[k]), abs(Tg[k]))) / tol
                err[k-1] = (errmax / 0.25) ** (1. / (2*k + 1))
                if k >= kopt - 2:
                    if errmax > 1:
                        # scale the time step and try again
                        sfac, reduced = reduce_tstep(k, err)
                        if reduced:
                            red = max(1e-5, min(red, 0.7))
                            h_traj = sfac * hstep
                            break
                    else:
                        # scale the time step if possible
                        t += h_traj
                        h_traj = increase_tstep(k, err, reduced) * hstep

                        # update to the final position
                        xnew = np.sum(Tx, axis=0)
                        pnew = np.sum(Tp, axis=0)
                        traj.update_x(xnew)
                        traj.update_p(pnew)
                        if propphase:
                            gnew = np.sum(Tg, axis=0)
                            traj.update_phase(gnew)
                        evaluate.update_pes_traj(traj)
                        break


def mm_step(traj, dt, x0, x1, p0, p1, g0, g1, n):
    """Steps forward by a single modified midpoint step.

    Positions of the previous two steps are stored in y0 and y1. In each
    step, y0 := y1 and y1 := the new position.
    """
    if n == 0:
        x0[:] = traj.x()
        p0[:] = traj.p()
        x1[:] = x0 + dt*traj.velocity()
        p1[:] = p0 + dt*traj.force()
        if propphase:
            g0[:] = traj.phase()
            g1[:] = g0 + dt*traj.phase_dot()
    else:
        x1[:] = x0 + 2*dt*traj.velocity()
        p1[:] = p0 + 2*dt*traj.force()
        x0[:] = traj.x()
        p0[:] = traj.p()
        if propphase:
            g1[:] = g0 + 2*dt*traj.phase_dot()
            g0[:] = traj.phase()

    traj.update_x(x1)
    traj.update_p(p1)
    if propphase:
        traj.update_phase(g1)


def poly_extrapolate(ki, t0, tsav, x0, p0, g0, Tx, Tp, Tg):
    """Extrapolates a set of modified midpoint estimates using
    polynomial extrapolation."""
    tsav[ki] = t0
    dx = x0
    dp = p0
    dg = g0
    if ki > 0:
        dx1 = x0
        dp1 = p0
        dg1 = g0
        for k in range(ki):
            qx = np.copy(Tx[k])
            qp = np.copy(Tp[k])
            qg = np.copy(Tg[k])
            Tx[k] = dx
            Tp[k] = dp
            Tg[k] = dg
            fac1 = t0 / (tsav[ki-k-1] - t0)
            dx = fac1 * (dx1 - qx)
            dp = fac1 * (dp1 - qp)
            dg = fac1 * (dg1 - qg)
            fac2 = tsav[ki-k-1] / (tsav[ki-k-1] - t0)
            dx1 = fac2 * (dx1 - qx)
            dp1 = fac2 * (dp1 - qp)
            dg1 = fac2 * (dg1 - qg)

    Tx[ki] = dx
    Tp[ki] = dp
    Tg[ki] = dg


def rati_extrapolate(ki, t0, tsav, x0, p0, g0, Tx, Tp, Tg):
    """Extrapolates a set of modified midpoint estimates using
    rational function extrapolation."""
    tsav[ki] = t0
    if ki > 0:
        vx = np.copy(Tx[0])
        vp = np.copy(Tp[0])
        vg = np.copy(Tg[0])

    Tx[0] = np.copy(x0)
    Tp[0] = np.copy(p0)
    Tg[0] = np.copy(g0)
    if ki > 0:
        cx = Tx[0]
        cp = Tp[0]
        cg = Tg[0]
        for k in range(1, ki+1):
            fac = tsav[ki-k] / t0
            b1x = fac*vx
            b1p = fac*vp
            b1g = fac*vg
            bx = b1x - cx
            bp = b1p - cp
            bg = b1g - cg
            bx[bx != 0] = (cx[bx != 0] - vx[bx != 0]) / bx[bx != 0]
            bp[bp != 0] = (cp[bp != 0] - vp[bp != 0]) / bp[bp != 0]
            bg[bg != 0] = (cg[bg != 0] - vg[bg != 0]) / bg[bg != 0]
            ddx = cx*bx
            ddp = cp*bp
            ddg = cg*bg
            ddx[bx == 0] = vx[bx == 0]
            ddp[bp == 0] = vp[bp == 0]
            ddg[bg == 0] = vg[bg == 0]
            cx = b1x*bx
            cp = b1p*bp
            cg = b1g*bg

            if k != ki:
                vx = Tx[k]
                vp = Tp[k]
                vg = Tg[k]

            Tx[k] = ddx
            Tp[k] = ddp
            Tg[k] = ddg


def reduce_tstep(ki, error):
    """Calculates the timestep scale factor based on the error."""
    redfac = 1.
    reduced = False

    if ki == kmax - 1 or ki == kopt:
        redfac = 0.7 / error[ki-1]
        reduced = True
    elif ki == kopt - 1:
        if alpha[kopt-2, kopt-1] < error[ki-1]:
            redfac = 1. / error[ki-1]
            reduced = True
    elif kopt == kmax:
        if alpha[ki-1, kmax-2] < error[ki-1]:
            redfac = alpha[ki-1, kmax-1] * 0.7 / error[ki-1]
            reduced = True
    elif alpha[ki-1, kopt-1] < error[ki-1]:
        redfac = alpha[ki-1, kopt-1] / error[ki-1]
        reduced = True

    return redfac, reduced


def increase_tstep(ki, error, reduced):
    """Calculates the timestep scale factor and determine the optimal
    row number for extrapolation."""
    global kopt

    for j in range(ki):
        fact = max(error[j], 0.1)
        work = fact * afac[j+1]
        workmin = 1e16
        if work < workmin:
            scale = fact
            workmin = work
            kopt = j + 1

    if ki < kopt and kopt != kmax and not reduced:
        fact = max(scale / alpha[kopt-2, kopt-1], 0.1)
        if afac[kopt] * fact < workmin:
            scale = fact
            kopt += 1

    return 1./scale