import sys
import copy
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
    ncrd = glbl.fms['num_particles'] * glbl.fms['dim_particles']
    mass = master.traj[0].masses()

    x0   = np.zeros((master.nalive,ncrd),dtype=np.float)
    p0   = np.zeros((master.nalive,ncrd),dtype=np.float)
    g0   = np.zeros(master.nalive,dtype=np.float)
    xnew = np.zeros((master.nalive,ncrd),dtype=np.float)
    pnew = np.zeros((master.nalive,ncrd),dtype=np.float)
    gnew = np.zeros(master.nalive,dtype=np.float)

    # initialize position,momentum,phase
    amp0 = master.amplitudes()
    for i in range(master.nalive):
        ii = master.alive[i]
        x0[i,:]   = master.traj[ii].x()
        p0[i,:]   = master.traj[ii].p()
        g0[i]     = master.traj[ii].gamma()
        xnew[i,:] = master.traj[ii].x()
        pnew[i,:] = master.traj[ii].p()
        gnew[i]   = master.traj[ii].gamma()

    # 
    # determine k1, k2, k3, k4
    #
    rk_ordr = 4
    k_mult  = [1, 2, 2, 1]
    t_step  = [dt / k_mult[i] for i in range(1,len(k_mult))]
    dt_seg = dt / sum(k_mult)

    #
    # Do 4th order RK
    #    
    H_list = []
    for rk in range(rk_ordr):

        # determine x,p,gamma at f(t,x)
        H_list.append(master.Heff)
        for i in range(master.nalive):
            ii = master.alive[i]

            xdot = master.traj[ii].velocity()
            pdot = master.traj[ii].force()/mass
            gdot = master.traj[ii].phase_dot()

            xnew[i,:] += dt_seg * k_mult[rk] * xdot
            pnew[i,:] += dt_seg * k_mult[rk] * pdot
            gnew[i]   += dt_seg * k_mult[rk] * gdot

            if rk != (rk_ordr-1):
                master.traj[ii].update_x( x0[i,:]   + t_step[rk] * xdot)
                master.traj[ii].update_p( p0[i,:]   + t_step[rk] * pdot)
                master.traj[ii].update_phase( g0[i] + t_step[rk] * gdot)
            else:
                master.traj[ii].update_x( xnew[i,:] )
                master.traj[ii].update_p( pnew[i,:] )
                master.traj[ii].update_phase( gnew[i] )
       
        # update potential energy surfaces and Heff        
        surface.update_pes(master)

    amp_sum = 0
    for i in range(len(k_mult)):
        master.update_amplitudes(dt,10,H_list[i],amp0)
        amp_sum += k_mult[i] * master.amplitudes()
    master.set_amplitudes(amp_sum/sum(k_mult))
   
    #
    timings.stop('propagators.propagate_bundle')
    return

#
# propagate a single trajectory
#
def propagate_trajectory(traj, dt):
    
    timings.start('propagators.propagate_trajectory')

    ncrd = glbl.fms['num_particles'] * glbl.fms['dim_particles']
    mass = master.traj[0].masses()

    # 
    # determine k1, k2, k3, k4
    #
    rk_ordr = 4
    k_mult  = [1, 2, 2, 1]
    t_step  = [dt / k_mult[i] for i in range(1,len(k_mult))]
    dt_seg  = dt / sum(k_mult)

    # 
    # initialize values
    #
    x0   = traj.x()
    xnew = traj.x()
    pnew = traj.p()
    gnew = traj.gamma()

    #
    # Do 4th order RK
    #    
    for rk in range(rk_ordr):

        xdot = traj.velocity()
        pdot = traj.force()/mass
        gdot = traj.phase_dot()

        xnew += dt_seg * k_mult[rk] * xdot 
        pnew += dt_seg * k_mult[rk] * pdot
        gnew += dt_seg * k_mult[rk] * gdot
        if rk != (rk_ordr-1):
            traj.update_x( x0 + t_step[rk]*xdot )
        else:
            traj.update_x( xnew )
            traj.update_p( pnew )
            traj.update_phase( gnew )

    timings.stop('propagators.propagate_trajectory')
    return

