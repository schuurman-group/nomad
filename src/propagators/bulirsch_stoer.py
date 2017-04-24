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

The number of steps n is based on...
"""
import numpy as np
import src.fmsio.glbl as glbl
import src.dynamics.timings as timings
import src.dynamics.surface as surface


propphase = glbl.fms['phase_prop'] != 0


@timings.timed
def propagate_bundle(master, dt):
    """Propagates the Bundle object with BS."""
    ncrd = master.traj[0].dim
    ntraj = master.n_traj()

    nstep = 2 # should adjust with error
    h = dt / nstep
    tmpbundle = master.copy()

    x0 = np.zeros((ntraj, ncrd))
    p0 = np.zeros((ntraj, ncrd))
    g0 = np.zeros((ntraj, ncrd))
    x1 = np.zeros((ntraj, ncrd))
    p1 = np.zeros((ntraj, ncrd))
    g1 = np.zeros((ntraj, ncrd))

    # step through n modified midpoint steps
    for n in range(nstep):
        for i in range(ntraj):
            if tmpbundle.traj[i].active:
                mm_step(tmpbundle.traj[i], h, x0[i], x1[i], p0[i], p1[i],
                        g0[i], g1[i], n)
                if i == 0:
                    print(tmpbundle.traj[i].x(), x0[i])
        surface.update_pes(tmpbundle, update_centroids=False)

    # update to the final position
    for i in range(ntraj):
        if master.traj[i].active:
            master.traj[i].update_x(0.5*(x1[i] + x0[i] +
                                         h*tmpbundle.traj[i].velocity()))
            master.traj[i].update_p(0.5*(p1[i] + p0[i] +
                                         h*tmpbundle.traj[i].force()))
            if propphase:
                master.traj[i].update_phase(0.5*(g1[i] + g0[i] +
                                                 h*tmpbundle.traj[i].phase_dot()))
    surface.update_pes(master)


@timings.timed
def propagate_trajectory(traj, dt):
    """Propagates a single trajectory with BS."""
    ncrd = traj.dim

    nstep = 2 # should adjust with error
    h = dt / nstep
    tmptraj = traj.copy()

    x0 = np.zeros(ncrd)
    p0 = np.zeros(ncrd)
    g0 = np.zeros(ncrd)
    x1 = np.zeros(ncrd)
    p1 = np.zeros(ncrd)
    g1 = np.zeros(ncrd)

    # step through n modified midpoint steps
    for n in range(nstep):
        mm_step(tmptraj, h, x0, x1, p0, p1, g0, g1, n)
        surface.update_pes_traj(tmptraj)

    # update to the final position
    traj.update_x(0.5*(x1 + x0 + h*tmptraj.velocity()))
    traj.update_p(0.5*(p1 + p0 + h*tmptraj.force()))
    if propphase:
        traj.update_phase(0.5*(p1 + p0 + h*tmptraj.phase_dot()))
    surface.update_pes_traj(traj)


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
