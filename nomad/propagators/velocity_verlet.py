"""
Routines for propagation with the velocity verlet algorithm.

Velocity Verlet:
  x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*t^2
  p(t+dt) = p(t) + 0.5*m*(a(t) + a(t+dt))*dt
"""
import numpy as np
import nomad.core.glbl as glbl
import nomad.core.timings as timings
import nomad.core.surface as evaluate

propphase = glbl.properties['phase_prop']

@timings.timed
def propagate(q0, t_deriv, dt):
    """the fundamental routine for propagating a quantity using velocity-verlet
       algorithm"""

    # we're going to limit this to 2-nd order DE for time being...
    q_deriv = t_deriv(q0)
    qt      = q0 + q_deriv[0] * dt + 0.5 * q_deriv[1] * dt**2

    return qt

@timings.timed
def propagate_traj(q0, cf, dq, dt):
    """the fundamental routine for propagating a quantity using velocity-verlet
       algorithm"""

    # we're going to limit this to 2-nd order DE for time being...
    q1 = q0
    for i in range(len(dq)):
        q1 += cf[i] * dq[i] * (dt**i)

    return q1

@timings.timed
def propagate_wfn(wfn, dt):
    """Propagates the Bundle object with VV."""
    # update position
    for i in range(wfn.n_traj()):
        if wfn.traj[i].active:
            propagate_half1(wfn.traj[i], dt)

    # update electronic structure for all trajectories
    # and centroids (where necessary)
    evaluate.update_pes(wfn)

    # finish update of momentum and phase
    for i in range(wfn.n_traj()):
        if wfn.traj[i].active:
            propagate_half2(wfn.traj[i], dt)

@timings.timed
def propagate_trajectory(traj, dt):
    """Propagates a single trajectory with VV."""

    # position update
    propagate_half1(traj, dt)

    # update electronic structure
    evaluate.update_pes_traj(traj)

    # momentum/phase update
    propagate_half2(traj, dt)


def propagate_half1(traj, dt):
    """Updates the position to end of time step and half-propagate the
    momentum and phase."""
    # set the t=t values
    x0 = traj.x()
    p0 = traj.p()
    g0 = traj.phase()
    v0 = traj.velocity()
    f0 = traj.force()
    m  = traj.masses()

    # phase_dot needs to be called before position update to avoid errors that
    # (correctly) state that surface information does not correspond
    # to current geometry [since phase_dot depends on the value of the
    # potential energy]
    if propphase:
        # half update phase
        g_dot = traj.phase_dot()
        g_cur = np.dot(f0, v0)
        g1    = propagate_traj(g0, [0, 0.5, -0.25], [0, g_dot, g_cur], dt)
        traj.update_phase(g1)

    # update position and momentum
    #   x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    #   p(t+dt) = p(t) + 0.5*m*(a(t) + a(t+dt))*dt
    # --> need to compute forces at new geometry
    x1 = propagate_traj(x0, [0, 1., 0.5], [0, v0, f0/m], dt)
    p1 = propagate_traj(p0, [0, 0.5], [0, f0], dt)

    # update x, and half-update p
    traj.update_x(x1)
    traj.update_p(p1)

def propagate_half2(traj, dt):
    """Finish the phase and momentum update using forces and velocities at
    the final position."""
    # half update p
    f1 = traj.force()
    p1 = traj.p()
    v1 = traj.velocity()
    g1 = traj.phase()

    if propphase:
        # update the nuclear phase
        g_dot = traj.phase_dot()
        g_cur = np.dot(f1, v1)
        #delta_gamma = (gdot1 + gdot2) * dt / 2.0 - (gcur1 - gcur2) * dt**2 / 4.
        g2    = propagate_traj(g1, [0, 0.5, 0.25], [0, g_dot, g_cur], dt)
        traj.update_phase(g2)

    p2 = propagate_traj(p1, [0, 0.5], [0, f1], dt)
    traj.update_p(p2)
