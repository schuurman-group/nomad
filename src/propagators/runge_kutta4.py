"""
Routines for propagation with the 4th order Runge-Kutta algorithm.

4th order Runge-Kutta:
  x(t+dt) = x(t) + dt*[(1/6)kx1 + (1/3)kx2 + (1/3)kx3 + (1/6)kx4]
  p(t+dt) = p(t) + dt*[(1/6)kp1 + (1/3)kp2 + (1/3)kp3 + (1/6)kp4]
  ky1 = f[t, y(t)] = dy(t)/dt
  ky2 = f[t+(1/2)dt, y(t)+(1/2)ky1]
  ky3 = f[t+(1/2)dt, y(t)+(1/2)ky2*dt]
  ky4 = f[t+dt, y(t)+ky3*dt]
"""
import numpy as np
import src.fmsio.glbl as glbl
import src.dynamics.timings as timings
import src.dynamics.surface as surface


rk_ordr = 4
coeff = np.array([[0., 0., 0., 0.], [0.5, 0., 0., 0.],
                  [0., 0.5, 0., 0.], [0., 0., 1., 0.]])
wgt = np.array([1./6., 1./3., 1./3., 1./6.])
propphase = glbl.fms['phase_prop'] != 0


@timings.timed
def propagate_bundle(master, dt):
    """Propagates the Bundle object with RK4."""
    ncrd = master.traj[0].dim
    kx = np.zeros((master.nalive, rk_ordr, ncrd)) # should use nactive, but it is set to 0 when entering coupling regions
    kp = np.zeros((master.nalive, rk_ordr, ncrd))
    kg = np.zeros((master.nalive, rk_ordr, ncrd))

    # propagate amplitudes for 1/2 time step using x0
    master.update_amplitudes(0.5*dt)

    for rk in range(rk_ordr):
        tmpbundle = master.copy()
        for i in range(tmpbundle.n_traj()):
            if tmpbundle.traj[i].active:
                propagate_rk(tmpbundle.traj[i], dt, rk, kx[i], kp[i], kg[i])

        if rk < rk_ordr - 1:
            surface.update_pes(tmpbundle)

    for i in range(master.n_traj()):
        if master.traj[i].active:
            x0 = master.traj[i].x()
            p0 = master.traj[i].p()
            master.traj[i].update_x(x0 + np.sum(wgt[:,np.newaxis] *
                                                kx[i], axis=0))
            master.traj[i].update_p(p0 + np.sum(wgt[:,np.newaxis] *
                                                kp[i], axis=0))
            if propphase:
                g0 = master.traj[i].phase()
                master.traj[i].update_phase(g0 + np.sum(wgt[:,np.newaxis] *
                                                        kg[i], axis=0))
    surface.update_pes(master)

    # propagate amplitudes for 1/2 time step using x1
    master.update_amplitudes(0.5*dt)


@timings.timed
def propagate_trajectory(traj, dt):
    """Propagates a single trajectory with RK4."""
    ncrd = traj.dim
    kx = np.zeros((rk_ordr, ncrd))
    kp = np.zeros((rk_ordr, ncrd))
    kg = np.zeros((rk_ordr, ncrd))

    for rk in range(rk_ordr):
        tmptraj = traj.copy()
        propagate_rk(tmptraj, dt, rk, kx, kp, kg)

        if rk < rk_ordr - 1:
            surface.update_pes_traj(tmptraj)

    x0 = traj.x()
    p0 = traj.p()
    traj.update_x(x0 + np.sum(wgt[:,np.newaxis] * kx, axis=0))
    traj.update_p(p0 + np.sum(wgt[:,np.newaxis] * kp, axis=0))
    if propphase:
        g0 = traj.phase()
        traj.update_phase(g0 + np.sum(wgt[:,np.newaxis] * kg, axis=0))
    surface.update_pes_traj(traj)


def propagate_rk(traj, dt, rk, kxi, kpi, kgi):
    """Gets k values and updates the position and momentum by
    a single rk step."""
    x0 = traj.x()
    p0 = traj.p()

    kxi[rk] = dt * traj.velocity()
    kpi[rk] = dt * traj.force()
    kgi[rk] = dt * traj.phase_dot()

    if rk < rk_ordr - 1:
        traj.update_x(x0 + np.sum(coeff[rk,:,np.newaxis]*kxi, axis=0))
        traj.update_p(p0 + np.sum(coeff[rk,:,np.newaxis]*kpi, axis=0))
        if propphase:
            traj.update_phase(g0 + np.sum(coeff[rk,:,np.newaxis]*kgi, axis=0))
