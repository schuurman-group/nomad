import sys
import numpy as np
import src.fmsio.glbl as glbl
import src.dynamics.timings as timings
import src.dynamics.surface as surface
#
# all propagators have to define a function "propagate_bundle" that
# takes a trajectory argument and a time step argument
#
# 4th order Runge-Kutta
#   x(t+dt) = x(t) + dt/6 * (kv1 + 2*kv2 + 2*kv3 + kv4)
#   p(t+dt) = p(t) + dt/6 * (ka1 + 2*ka2 + 2*ka3 + ka4)
#   ky1 = f(t, y(t)) = dy(t)/dt
#   ky2 = f(t+dt/2, y(t)+ky1*dt/2)
#   ky3 = f(t+dt/2, y(t)+ky2*dt/2)
#   ky4 = f(t+dt, y(t)+ky3*dt)
#
def propagate_bundle(master, dt):

    timings.start('propagators.propagate_bundle')
    dt_2 = dt / 2.
    dt_6 = dt / 6.
    ncrd = glbl.fms['num_particles'] * glbl.fms['dim_particles']
    mass = master.traj[0].masses()

    amp0 = master.amplitudes()
    x0   = np.zeros((master.nalive,ncrd),dtype=np.float)

    xnew = np.zeros((master.nalive,ncrd),dtype=np.float)
    pnew = np.zeros((master.nalive,ncrd),dtype=np.float)
    gnew = np.zeros(master.nalive,dtype=np.float)

    # initialize position,momentum,phase
    for i in range(master.nalive):
        ii = master.alive[i]
        x0[i,:]   = master.traj[ii].x()
        xnew[i,:] = master.traj[ii].x()
        pnew[i,:] = master.traj[ii].p()
        gnew[i]   = master.traj[ii].phase

    # 
    # determine k1, k2, k3, k4
    #
    t_step = [dt_2, dt_2, dt]
    k_mult = [1., 2., 2., 1.]
    r_ordr = 4

    #
    # Do 4th order RK
    #    
    H_list = []
    a_list = [amp0]
    for rk in range(r_ordr):
    
        # determine x,p,gamma at f(t,x)
        H_list.append(master.Heff)
        for i in range(master.nalive):
            ii = master.alive[i]

            vk = master.traj[ii].velocity()
            fk = master.traj[ii].force()
            gk = master.traj[ii].phase_dot()        

            xnew[i,:] += dt_6 * k_mult[rk] * vk
            pnew[i,:] += dt_6 * k_mult[rk] * fk/mass
            gnew[i]   += dt_6 * k_mult[rk] * gk
            if rk != (r_ordr-1):
                master.traj[ii].update_x(x0[i,:] + t_step[rk]*vk)
            else:
                master.traj[ii].update_x(xnew[i,:])
                master.traj[ii].update_p(pnew[i,:])
                master.traj[ii].update_phase(gnew[i])
        
        # update potential energy surfaces and Heff        
        surface.update_pes(master)
        if rk != (r_ordr-1):
            master.update_amplitudes(t_step[rk],10)
            a_list.append(master.amplitudes())
            master.set_amplitudes(amp0)
        else:
            ampn    = amp0
            for i in range(r_ordr):
                ampn += dt_6 * k_mult[i] * -np.complex(0.,1.) * np.dot(H_list[i], a_list[i])
            master.set_amplitudes(ampn)

    #
    timings.stop('propagators.propagate_bundle')
    return

#
# propagate a single trajectory
#
def propagate_traj(traj, dt):
    
    timings.start('propagators.propagate_trajectory')

    dt_2 = dt / 2.
    dt_6 = dt / 6.
    ncrd = glbl.fms['num_particles'] * glbl.fms['dim_particles']

    # 
    # determine k1, k2, k3, k4
    #
    t_step = [dt_2, dt_2, dt]
    k_mult = [1.,2.,2.,1.]
    r_ordr = 4

    # 
    # initialize values
    #
    x0   = traj.x()
    xnew = traj.x()
    pnew = traj.p()
    gnew = traj.phase

    #
    # Do 4th order RK
    #    
    for rk in range(r_ordr):

        vk = traj.velocity()
        fk = traj.force()
        gk = traj.phase_dot()

        xnew += dt_6 * k_mult[rk] * vk
        pnew += dt_6 * k_mult[rk] * fk/mass
        gnew += dt_6 * k_mult[rk] * gk
        if rk != (r_ordr-1):
            traj.update_x(x0 + t_step[rk]*vk)
        else:
            traj.update_x(xnew)
            traj.update_p(pnew)
            traj.update_phase(gnew)

    timings.stop('propagators.propagate_trajectory')
    return

