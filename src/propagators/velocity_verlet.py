#
import numpy as np
from ..basis import trajectory
#
# all propagators have to define a function "propagate" that takes
# a trajectory argument and a time step argument
#
# Velocity Verlet
#   x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*t^2
#   p(t+dt) = p(t) + 0.5*m*(a(t) + a(t+dt))*dt
#
def propagate(trajectory,dt):
    #
    # set the t=t values 
    #
    x0   = trajectory.x()
    p0   = trajectory.p()
    v0   = trajectory.velocity()
    f0   = trajectory.force()
    mass = trajectory.masses()

    #------------------------------------------------
    # update the nuclear phase
    #  gamma = gamma + dt * phase_dot / 2.0
    g1_0 = trajectory.phase_dot()
    g2_0 = 2. * np.dot(f0, v0)

    #------------------------------------------------
    # update position and momentum
    #   x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    #   p(t+dt) = p(t) + 0.5*m*(a(t) + a(t+dt))*dt
    # --> need to compute forces at new geometry
    x1 = x0 + v0*dt + 0.5 * (f0/mass) * dt**2

    #-------------------------------------------
    # update x
    trajectory.update_x(x1)

    #------------------------------------------
    # update p
    f1 = trajectory.force()
    p1 = p0 + 0.5 * (f0 + f1) * dt 
    trajectory.update_p(p1)
    v1 = trajectory.velocity()

    #---------------------------------------------
    # update the nuclear phase
    #
    g1_1 = trajectory.phase_dot()
    g2_1 = 2. * np.dot(f1, v1)

    #--------------------------------------------
    # solve for the phases
    a = (1./2.) * dt**2
    b = (1./6.) * dt**3
    c = dt
    d = (1./2.) * dt**2

    vec   = np.array([g1_1 - g1_0 - g2_0 * dt, g2_1 - g2_1])
    alpha =( d*vec[0] - b*vec[1]) / (a*d - b*c)
    beta  =(-c*vec[0] - a*vec[1]) / (a*d - b*c)

    dgamma = (g1_0 + g1_1) * dt / 2.0 - (g2_0 - g2_1) * dt**2 / 8.0
    trajectory.update_phase(trajectory.phase + dgamma)

