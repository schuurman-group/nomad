import numpy as np
import src.dynamics.timings as timings
#
# all propagators have to define a function "propagate" that takes
# a trajectory argument and a time step argument
#
# Velocity Verlet
#   x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*t^2
#   p(t+dt) = p(t) + 0.5*m*(a(t) + a(t+dt))*dt
#
def propagate(master,dt):

    timings.start('propagators.velocity_verlet')

    #-----------------------------------------------
    # propagate amplitudes for 1/2 time step using x0
    #
    master.update_amplitudes(0.5*dt,10) 

    for i in range(master.n_total()):
        
        if not master.traj[i].alive:
            continue

        traj = master.traj[i]
        #
        # set the t=t values 
        #
        x0   = traj.x()
        p0   = traj.p()
        v0   = traj.velocity()
        f0   = traj.force()
        mass = traj.masses()

        #------------------------------------------------
        # update the nuclear phase
        #  gamma = gamma + dt * phase_dot / 2.0
        g1_0 = traj.phase_dot()
        g2_0 = 2. * np.dot(f0, v0)

        #------------------------------------------------
        # update position and momentum
        #   x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        #   p(t+dt) = p(t) + 0.5*m*(a(t) + a(t+dt))*dt
        # --> need to compute forces at new geometry
        x1 = x0 + v0*dt + 0.5 * (f0/mass) * dt**2

        #-------------------------------------------
        # update x
        traj.update_x(x1)

        #------------------------------------------
        # update p
        f1 = traj.force()
        p1 = p0 + 0.5 * (f0 + f1) * dt 
        traj.update_p(p1)
        v1 = traj.velocity()

        #---------------------------------------------
        # update the nuclear phase
        #
        g1_1 = traj.phase_dot()
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
        traj.update_phase(traj.phase + dgamma)

    #-------------------------------------------------
    # propagate amplitudes for 1/2 time step using x1
    #
    master.update_amplitudes(0.5*dt,10)

    timings.stop('propagators.velocity_verlet')
    return
