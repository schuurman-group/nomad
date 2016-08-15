"""
Routines for propagation with the velocity verlet algorithm.

All propagators have to define a function "propagate" that takes
a trajectory argument and a time step argument

Velocity Verlet:
  x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*t^2
  p(t+dt) = p(t) + 0.5*m*(a(t) + a(t+dt))*dt
"""
import numpy as np
import src.dynamics.timings as timings
import src.dynamics.surface as surface


@timings.timed_func
def propagate_bundle(master, dt):
    """Propagates the Bundle object with VV."""
    # propagate amplitudes for 1/2 time step using x0
    master.update_amplitudes(0.5*dt, 10)

    # update position
    for i in range(master.n_traj()):
        if not master.traj[i].alive:
            continue
        propagate_position(master.traj[i], dt)

    # update electronic structure for all trajectories
    # and centroids (where necessary)
    surface.update_pes(master)

    # finish update of momentum and phase
    for i in range(master.n_traj()):
        if not master.traj[i].alive:
            continue
        propagate_momentum(master.traj[i], dt)

    # propagate amplitudes for 1/2 time step using x1
    master.update_amplitudes(0.5*dt, 10)


@timings.timed_func
def propagate_trajectory(traj, dt):
    """Propagates a single trajectory with VV."""
    # position update
    propagate_position(traj, dt)

    # update electronic structure
    surface.update_pes_traj(traj)

    # momentum/phase update
    propagate_momentum(traj, dt)


def propagate_position(traj, dt):
    """Updates the position to end of time step and half-propagate the
    momentum and phase."""
    # set the t=t values
    x0   = traj.x()
    p0   = traj.p()
    v0   = traj.velocity()
    f0   = traj.force()
    mass = traj.masses()

    # update the nuclear phase
    #  gamma = gamma + dt * phase_dot / 2.0
    g1_0 = traj.phase_dot()
    g2_0 = 2. * np.dot(f0, v0)

    # update position and momentum
    #   x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    #   p(t+dt) = p(t) + 0.5*m*(a(t) + a(t+dt))*dt
    # --> need to compute forces at new geometry
    x1 = x0 + v0*dt + 0.5 * (f0/mass) * dt**2

    # update x
    traj.update_x(x1)

    # half update p and phase
    traj.update_p(p0 + 0.5 * f0 * dt)
    dgamma = g1_0 * dt / 2. - g2_0 * dt**2 / 8.
    traj.update_phase(traj.phase() + dgamma)


def propagate_momentum(traj, dt):
    """Finish the phase and momentum update using forces and velocities at
    the final position."""
    # half update p
    f1 = traj.force()
    p1 = traj.p() + 0.5 * f1 * dt
    traj.update_p(p1)
    v1 = traj.velocity()

    # update the nuclear phase
    g1_1 = traj.phase_dot()
    g2_1 = 2. * np.dot(f1, v1)

    # solve for the phases
    #a = 0.5 * dt**2
    #b = (1./6.) * dt**3
    #c = dt
    #d = 0.5 * dt**2

    #vec   = np.array([g1_1 - g1_0 - g2_0 * dt, g2_1 - g2_1])
    #alpha =( d*vec[0] - b*vec[1]) / (a*d - b*c)
    #beta  =(-c*vec[0] - a*vec[1]) / (a*d - b*c)

    #dgamma = (g1_0 + g1_1) * dt / 2.0 - (g2_0 - g2_1) * dt**2 / 8.
    dgamma = g1_1 * dt / 2. + g2_1 * dt**2 / 8.
    traj.update_phase(traj.phase() + dgamma)
