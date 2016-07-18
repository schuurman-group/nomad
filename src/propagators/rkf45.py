import sys
import numpy as np
import src.fmsio.glbl as glbl
import src.dynamics.timings as timings
import src.dynamics.surface as surface
import src.basis.bundle as bundle
import src.basis.trajectory as trajectory

########################################################################
# Working RK4 code
########################################################################
#def propagate_bundle(master, dt):
#
#    #-------------------------------------------------------------------
#    # Propagate amplitudes for 1/2 time step using Heff(t0)
#    #-------------------------------------------------------------------
#    master.update_amplitudes(0.5*dt, 10)
#
#    #-------------------------------------------------------------------
#    # Initialisation
#    #-------------------------------------------------------------------
#    # Masses
#    mass = master.traj[0].masses()
#
#    # No. coordinates
#    ncrd = glbl.fms['num_particles'] * glbl.fms['dim_particles']
#
#    # Work arrays
#    x0   = np.zeros((master.nalive, ncrd))
#    p0   = np.zeros((master.nalive, ncrd))
#    g0   = np.zeros(master.nalive)
#    xnew = np.zeros((master.nalive, ncrd))
#    pnew = np.zeros((master.nalive, ncrd))
#    gnew = np.zeros(master.nalive)
#    k1_x = np.zeros((master.nalive, ncrd))
#    k1_p = np.zeros((master.nalive, ncrd))
#    k1_g = np.zeros(master.nalive)
#    k2_x = np.zeros((master.nalive, ncrd))
#    k2_p = np.zeros((master.nalive, ncrd))
#    k2_g = np.zeros(master.nalive)
#    k3_x = np.zeros((master.nalive, ncrd))
#    k3_p = np.zeros((master.nalive, ncrd))
#    k3_g = np.zeros(master.nalive)
#    k4_x = np.zeros((master.nalive, ncrd))
#    k4_p = np.zeros((master.nalive, ncrd))
#    k4_g = np.zeros(master.nalive)
#    
#    #-------------------------------------------------------------------
#    # k1
#    #-------------------------------------------------------------------
#    # Temporary bundle copy
#    tmpbundle = bundle.copy_bundle(master)
#
#    # Initial phase space centres
#    for i in range(master.nalive):
#        ii = master.alive[i]
#        x0[i,:] = tmpbundle.traj[ii].x()
#        p0[i,:] = tmpbundle.traj[ii].p()
#        g0[i]   = tmpbundle.traj[ii].phase()
#
#    # Calculate the time-derivatives at the new phase space centres
#    for i in range(master.nalive):
#        ii = master.alive[i]
#        xdot = tmpbundle.traj[ii].velocity()
#        pdot = tmpbundle.traj[ii].force() / mass
#        gdot = tmpbundle.traj[ii].phase_dot()
#        k1_x[i,:] = xdot
#        k1_p[i,:] = pdot
#        k1_g[i]   = gdot
#
#    #-------------------------------------------------------------------
#    # k2
#    #-------------------------------------------------------------------
#    # Update the phase space centres
#    for i in range(master.nalive):
#        ii = master.alive[i]
#        tmpbundle.traj[ii].update_x(x0[i,:]   + 0.5*dt*k1_x[i,:])
#        tmpbundle.traj[ii].update_p(p0[i,:]   + 0.5*dt*k1_p[i,:])
#        tmpbundle.traj[ii].update_phase(g0[i] + 0.5*dt*k1_g[i])
#
#    # Calculate the potentials at the new phase space centres
#    surface.update_pes(tmpbundle)
#
#    # Calculate the time-derivatives at the new phase space centres
#    for i in range(master.nalive):
#        ii = master.alive[i]
#        xdot = tmpbundle.traj[ii].velocity()
#        pdot = tmpbundle.traj[ii].force() / mass
#        gdot = tmpbundle.traj[ii].phase_dot()
#        k2_x[i,:] = xdot
#        k2_p[i,:] = pdot
#        k2_g[i]   = gdot
#    
#    #-------------------------------------------------------------------
#    # k3
#    #-------------------------------------------------------------------
#    # Update the phase space centres
#    tmpbundle = bundle.copy_bundle(master)
#    for i in range(master.nalive):
#        ii = master.alive[i]
#        tmpbundle.traj[ii].update_x(x0[i,:] + 0.5*dt*k2_x[i,:])
#        tmpbundle.traj[ii].update_p(p0[i,:] + 0.5*dt*k2_p[i,:])
#        tmpbundle.traj[ii].update_phase(g0[i] + 0.5*dt*k2_g[i])
#        
#    # Calculate the potentials at the new phase space centres
#    surface.update_pes(tmpbundle)
#
#    # Calculate the time-derivatives at the new phase space centres
#    for i in range(master.nalive):
#        ii = master.alive[i]
#        xdot = tmpbundle.traj[ii].velocity()
#        pdot = tmpbundle.traj[ii].force() / mass
#        gdot = tmpbundle.traj[ii].phase_dot()
#        k3_x[i,:] = xdot
#        k3_p[i,:] = pdot
#        k3_g[i]   = gdot
#
#    #-------------------------------------------------------------------
#    # k4
#    #-------------------------------------------------------------------
#    # Update the phase space centres
#    tmpbundle = bundle.copy_bundle(master)
#    for i in range(master.nalive):
#        ii = master.alive[i]
#        tmpbundle.traj[ii].update_x(x0[i,:] + dt*k3_x[i,:])
#        tmpbundle.traj[ii].update_p(p0[i,:] + dt*k3_p[i,:])
#        tmpbundle.traj[ii].update_phase(g0[i] +dt*k3_g[i])
#
#    # Calculate the potentials at the new phase space centres
#    surface.update_pes(tmpbundle)
#
#    # Calculate the time-derivatives at the new phase space centres
#    for i in range(master.nalive):
#        ii = master.alive[i]
#        xdot = tmpbundle.traj[ii].velocity()
#        pdot = tmpbundle.traj[ii].force() / mass
#        gdot = tmpbundle.traj[ii].phase_dot()
#        k4_x[i,:] = xdot
#        k4_p[i,:] = pdot
#        k4_g[i]   = gdot
#
#    #-------------------------------------------------------------------
#    # Calculate the approximate solutions at time t+dt
#    #-------------------------------------------------------------------
#    for i in range(master.nalive):
#        ii = master.alive[i]
#        xnew[i,:] = x0[i,:] + (dt/6.0) * (k1_x[i,:] + 2.0*k2_x[i,:] + 2.0*k3_x[i,:] +k4_x[i,:])
#        pnew[i,:] = p0[i,:] + (dt/6.0) * (k1_p[i,:] + 2.0*k2_p[i,:] + 2.0*k3_p[i,:] +k4_p[i,:])
#        gnew[i] = g0[i] + (dt/6.0) * (k1_g[i] + 2.0*k2_g[i] + 2.0*k3_g[i] +k4_g[i])
#        master.traj[ii].update_x(xnew[i,:])
#        master.traj[ii].update_p(pnew[i,:])
#        master.traj[ii].update_phase(gnew[i])
#
#    #-------------------------------------------------------------------
#    # Calculate the potentials at the propagated phase space centres
#    #-------------------------------------------------------------------
#    surface.update_pes(master)
#
#    #-------------------------------------------------------------------
#    # Propagate amplitudes for 1/2 time step using Heff(t0+dt)
#    #-------------------------------------------------------------------
#    master.update_amplitudes(0.5*dt, 10)
#
#    return
#
#########################################################################
#
#def propagate_trajectory(traj, dt):
#
#    #-------------------------------------------------------------------
#    # Initialisation
#    #-------------------------------------------------------------------
#    # Masses
#    mass = traj.masses()
#
#    # No. coordinates
#    ncrd = glbl.fms['num_particles'] * glbl.fms['dim_particles']
#
#    # Work arrays
#    x0   = np.zeros((ncrd))
#    p0   = np.zeros((ncrd))
#    g0   = 0.0
#    xnew = np.zeros((ncrd))
#    pnew = np.zeros((ncrd))
#    gnew = 0.0
#    k1_x = np.zeros((ncrd))
#    k1_p = np.zeros((ncrd))
#    k1_g = 0.0
#    k2_x = np.zeros((ncrd))
#    k2_p = np.zeros((ncrd))
#    k2_g = 0.0
#    k3_x = np.zeros((ncrd))
#    k3_p = np.zeros((ncrd))
#    k3_g = 0.0
#    k4_x = np.zeros((ncrd))
#    k4_p = np.zeros((ncrd))
#    k4_g = 0.0
#
#    #-------------------------------------------------------------------
#    # k1
#    #-------------------------------------------------------------------
#    # Temporary trajectory copy
#    tmptraj = trajectory.copy_traj(traj)
#
#    # Initial phase space centres
#    x0[:] = tmptraj.x()
#    p0[:] = tmptraj.p()
#    g0    = tmptraj.phase()
#
#    # Calculate the time-derivatives at the new phase space centres
#    xdot = tmptraj.velocity()
#    pdot = tmptraj.force() / mass
#    gdot = tmptraj.phase_dot()
#    k1_x[:] = xdot
#    k1_p[:] = pdot
#    k1_g    = gdot
#
#    #-------------------------------------------------------------------
#    # k2
#    #-------------------------------------------------------------------
#    # Update the phase space centres
#    tmptraj.update_x(x0[:]  + 0.5*dt*k1_x[:])
#    tmptraj.update_p(p0[:]  + 0.5*dt*k1_p[:])
#    tmptraj.update_phase(g0 + 0.5*dt*k1_g)
#
#    # Calculate the potentials at the new phase space centres
#    surface.update_pes_traj(tmptraj)
#
#    # Calculate the time-derivatives at the new phase space centres
#    xdot = tmptraj.velocity()
#    pdot = tmptraj.force() / mass
#    gdot = tmptraj.phase_dot()
#    k2_x[:] = xdot
#    k2_p[:] = pdot
#    k2_g    = gdot
#
#    #-------------------------------------------------------------------
#    # k3
#    #-------------------------------------------------------------------
#    # Update the phase space centres
#    tmptraj = trajectory.copy_traj(traj)
#    tmptraj.update_x(x0[:]  + 0.5*dt*k2_x[:])
#    tmptraj.update_p(p0[:]  + 0.5*dt*k2_p[:])
#    tmptraj.update_phase(g0 + 0.5*dt*k2_g)
#        
#    # Calculate the potentials at the new phase space centres
#    surface.update_pes_traj(tmptraj)
#
#    # Calculate the time-derivatives at the new phase space centres
#    xdot = tmptraj.velocity()
#    pdot = tmptraj.force() / mass
#    gdot = tmptraj.phase_dot()
#    k3_x[:] = xdot
#    k3_p[:] = pdot
#    k3_g    = gdot
#
#    #-------------------------------------------------------------------
#    # k4
#    #-------------------------------------------------------------------
#    # Update the phase space centres
#    tmptraj = trajectory.copy_traj(traj)
#    tmptraj.update_x(x0[:]  + dt*k3_x[:])
#    tmptraj.update_p(p0[:]  + dt*k3_p[:])
#    tmptraj.update_phase(g0 + dt*k3_g)
#
#    # Calculate the potentials at the new phase space centres
#    surface.update_pes_traj(tmptraj)
#
#    # Calculate the time-derivatives at the new phase space centres
#    xdot = tmptraj.velocity()
#    pdot = tmptraj.force() / mass
#    gdot = tmptraj.phase_dot()
#    k4_x[:] = xdot
#    k4_p[:] = pdot
#    k4_g    = gdot
#
#    #-------------------------------------------------------------------
#    # Calculate the approximate solutions at time t+dt
#    #-------------------------------------------------------------------
#    xnew[:] = x0[:] + (dt/6.0) * (k1_x[:] + 2.0*k2_x[:] + 2.0*k3_x[:] +k4_x[:])
#    pnew[:] = p0[:] + (dt/6.0) * (k1_p[:] + 2.0*k2_p[:] + 2.0*k3_p[:] +k4_p[:])
#    gnew = g0 + (dt/6.0) * (k1_g + 2.0*k2_g + 2.0*k3_g +k4_g)
#    traj.update_x(xnew[:])
#    traj.update_p(pnew[:])
#    traj.update_phase(gnew)
#
#    #-------------------------------------------------------------------
#    # Calculate the potentials at the propagated phase space centres
#    #-------------------------------------------------------------------
#    surface.update_pes_traj(traj)
#
#    return


########################################################################
# RKF45 code
########################################################################
def propagate_bundle(master, dt):

    #-------------------------------------------------------------------
    # Propagate amplitudes for 1/2 time step using Heff(t0)
    #-------------------------------------------------------------------
    master.update_amplitudes(0.5*dt, 10)

    #-------------------------------------------------------------------
    # Initialisation
    #-------------------------------------------------------------------
    # Masses
    mass = master.traj[0].masses()

    # No. coordinates
    ncrd = glbl.fms['num_particles'] * glbl.fms['dim_particles']

    # Work arrays
    x0   = np.zeros((master.nalive, ncrd))
    p0   = np.zeros((master.nalive, ncrd))
    g0   = np.zeros(master.nalive)
    xnew = np.zeros((master.nalive, ncrd))
    pnew = np.zeros((master.nalive, ncrd))
    gnew = np.zeros(master.nalive)
    xnew4 = np.zeros((master.nalive, ncrd))
    pnew4 = np.zeros((master.nalive, ncrd))
    gnew4 = np.zeros(master.nalive)
    xnew5 = np.zeros((master.nalive, ncrd))
    pnew5 = np.zeros((master.nalive, ncrd))
    gnew5 = np.zeros(master.nalive)
    k1_x = np.zeros((master.nalive, ncrd))
    k1_p = np.zeros((master.nalive, ncrd))
    k1_g = np.zeros(master.nalive)
    k2_x = np.zeros((master.nalive, ncrd))
    k2_p = np.zeros((master.nalive, ncrd))
    k2_g = np.zeros(master.nalive)
    k3_x = np.zeros((master.nalive, ncrd))
    k3_p = np.zeros((master.nalive, ncrd))
    k3_g = np.zeros(master.nalive)
    k4_x = np.zeros((master.nalive, ncrd))
    k4_p = np.zeros((master.nalive, ncrd))
    k4_g = np.zeros(master.nalive)
    k5_x = np.zeros((master.nalive, ncrd))
    k5_p = np.zeros((master.nalive, ncrd))
    k5_g = np.zeros(master.nalive)
    k6_x = np.zeros((master.nalive, ncrd))
    k6_p = np.zeros((master.nalive, ncrd))
    k6_g = np.zeros(master.nalive)

    #-------------------------------------------------------------------
    # k1
    #-------------------------------------------------------------------
    # Temporary bundle copy
    tmpbundle = bundle.copy_bundle(master)

    # Initial phase space centres
    for i in range(master.nalive):
        ii = master.alive[i]
        x0[i,:] = tmpbundle.traj[ii].x()
        p0[i,:] = tmpbundle.traj[ii].p()
        g0[i]   = tmpbundle.traj[ii].phase()

    # Calculate the time-derivatives at the new phase space centres
    for i in range(master.nalive):
        ii = master.alive[i]
        xdot = tmpbundle.traj[ii].velocity()
        pdot = tmpbundle.traj[ii].force() / mass
        gdot = tmpbundle.traj[ii].phase_dot()
        k1_x[i,:] = dt*xdot
        k1_p[i,:] = dt*pdot
        k1_g[i]   = dt*gdot

    #-------------------------------------------------------------------
    # k2
    #-------------------------------------------------------------------
    ## Update the phase space centres
    #for i in range(master.nalive):
    #    ii = master.alive[i]
    #    tmpbundle.traj[ii].update_x(x0[i,:]   + 0.25*k1_x[i,:])
    #    tmpbundle.traj[ii].update_p(p0[i,:]   + 0.25*k1_p[i,:])
    #    tmpbundle.traj[ii].update_phase(g0[i] + 0.25*k1_g[i])
    #
    ## Calculate the potentials at the new phase space centres
    #surface.update_pes(tmpbundle)
    #
    ## Calculate the time-derivatives at the new phase space centres
    #for i in range(master.nalive):
    #    ii = master.alive[i]
    #    xdot = tmpbundle.traj[ii].velocity()
    #    pdot = tmpbundle.traj[ii].force() / mass
    #    gdot = tmpbundle.traj[ii].phase_dot()
    #    k2_x[i,:] = dt*xdot
    #    k2_p[i,:] = dt*pdot
    #    k2_g[i]   = dt*gdot
    
    #-------------------------------------------------------------------
    # k3
    #-------------------------------------------------------------------
    # Update the phase space centres
    tmpbundle = bundle.copy_bundle(master)
    for i in range(master.nalive):
        ii = master.alive[i]
        tmpbundle.traj[ii].update_x(x0[i,:]   + (3./32.)*k1_x[i,:] +
                                    (9./32.)*k2_x[i,:]) 
        tmpbundle.traj[ii].update_p(p0[i,:]   + (3./32.)*k1_p[i,:] +
                                    (9./32.)*k2_p[i,:]) 
        tmpbundle.traj[ii].update_phase(g0[i] + (3./32.)*k1_g[i]   +
                                        (9./32.)*k2_g[i]) 
        
    # Calculate the potentials at the new phase space centres
    surface.update_pes(tmpbundle)

    # Calculate the time-derivatives at the new phase space centres
    for i in range(master.nalive):
        ii = master.alive[i]
        xdot = tmpbundle.traj[ii].velocity()
        pdot = tmpbundle.traj[ii].force() / mass
        gdot = tmpbundle.traj[ii].phase_dot()
        k3_x[i,:] = dt*xdot
        k3_p[i,:] = dt*pdot
        k3_g[i]   = dt*gdot

    #-------------------------------------------------------------------
    # k4
    #-------------------------------------------------------------------
    # Update the phase space centres
    tmpbundle = bundle.copy_bundle(master)
    for i in range(master.nalive):
        ii = master.alive[i]
        tmpbundle.traj[ii].update_x(x0[i,:]   +
                                    (1932./2197.)*k1_x[i,:]-
                                    (7200./2197.)*k2_x[i,:]+
                                    (7296./2197.)*k3_x[i,:]) 
        tmpbundle.traj[ii].update_p(p0[i,:]   +
                                    (1932./2197.)*k1_p[i,:]-
                                    (7200./2197.)*k2_p[i,:]+
                                    (7296./2197.)*k3_p[i,:]) 
        tmpbundle.traj[ii].update_phase(g0[i] + (1932./2197.)*k1_g[i]-
                                        (7200./2197.)*k2_g[i]+
                                        (7296./2197.)*k3_g[i]) 

    # Calculate the potentials at the new phase space centres
    surface.update_pes(tmpbundle)

    # Calculate the time-derivatives at the new phase space centres
    for i in range(master.nalive):
        ii = master.alive[i]
        xdot = tmpbundle.traj[ii].velocity()
        pdot = tmpbundle.traj[ii].force() / mass
        gdot = tmpbundle.traj[ii].phase_dot()
        k4_x[i,:] = dt*xdot
        k4_p[i,:] = dt*pdot
        k4_g[i]   = dt*gdot

    #-------------------------------------------------------------------
    # k5
    #-------------------------------------------------------------------
    # Update the phase space centres
    tmpbundle = bundle.copy_bundle(master)
    for i in range(master.nalive):
        ii = master.alive[i]
        tmpbundle.traj[ii].update_x(x0[i,:] + (439./216.)*k1_x[i,:]
                                    -8.*k2_x[i,:]
                                    +(3680./513.)*k3_x[i,:] -
                                    (845./4104.)*k3_x[i,:]) 
        tmpbundle.traj[ii].update_p(p0[i,:] + (439./216.)*k1_p[i,:]
                                    -8.*k2_p[i,:]
                                    +(3680./513.)*k3_p[i,:] -
                                    (845./4104.)*k3_p[i,:]) 
        tmpbundle.traj[ii].update_phase(g0[i] + (439./216.)*k1_g[i]
                                    -8.*k2_g[i]
                                    +(3680./513.)*k3_g[i] -
                                    (845./4104.)*k3_g[i]) 

    # Calculate the potentials at the new phase space centres
    surface.update_pes(tmpbundle)

    # Calculate the time-derivatives at the new phase space centres
    for i in range(master.nalive):
        ii = master.alive[i]
        xdot = tmpbundle.traj[ii].velocity()
        pdot = tmpbundle.traj[ii].force() / mass
        gdot = tmpbundle.traj[ii].phase_dot()
        k5_x[i,:] = dt*xdot
        k5_p[i,:] = dt*pdot
        k5_g[i]   = dt*gdot
    
    #-------------------------------------------------------------------
    # k6
    #-------------------------------------------------------------------
    # Update the phase space centres
    tmpbundle = bundle.copy_bundle(master)
    for i in range(master.nalive):
        ii = master.alive[i]
        tmpbundle.traj[ii].update_x(x0[i,:] - (8./27.)*k1_x[i,:] +
                                    2.*k2_x[i,:] -
                                    (3544./2565.)*k3_x[i,:] +
                                    (1859./4104.)*k4_x[i,:] -
                                    (11./40.)*k5_x[i,:])
        tmpbundle.traj[ii].update_p(p0[i,:] - (8./27.)*k1_p[i,:] +
                                    2.*k2_p[i,:] -
                                    (3544./2565.)*k3_p[i,:] +
                                    (1859./4104.)*k4_p[i,:] -
                                    (11./40.)*k5_p[i,:])
        tmpbundle.traj[ii].update_phase(g0[i] - (8./27.)*k1_g[i] +
                                        2.*k2_g[i] -
                                        (3544./2565.)*k3_g[i] +
                                        (1859./4104.)*k4_g[i] -
                                        (11./40.)*k5_g[i])

    # Calculate the potentials at the new phase space centres
    surface.update_pes(tmpbundle)

    # Calculate the time-derivatives at the new phase space centres
    for i in range(master.nalive):
        ii = master.alive[i]
        xdot = tmpbundle.traj[ii].velocity()
        pdot = tmpbundle.traj[ii].force() / mass
        gdot = tmpbundle.traj[ii].phase_dot()
        k6_x[i,:] = dt*xdot
        k6_p[i,:] = dt*pdot
        k6_g[i]   = dt*gdot

    #-------------------------------------------------------------------
    # Calculate the RK4 solutions at time t+dt
    #-------------------------------------------------------------------
    for i in range(master.nalive):
        ii = master.alive[i]
        xnew4[i,:] = x0[i,:] + ((25./216.)*k1_x[i,:] +
                               (1408./2565.)*k3_x[i,:] +
                               (2197./4101.)*k4_x[i,:] -
                               (1./5.)*k5_x[i,:])
        pnew4[i,:] = p0[i,:] + ((25./216.)*k1_p[i,:] +
                               (1408./2565.)*k3_p[i,:] +
                               (2197./4101.)*k4_p[i,:] -
                               (1./5.)*k5_p[i,:])
        gnew4[i] = g0[i] + ((25./216.)*k1_g[i] + (1408./2565.)*k3_g[i]
                            + (2197./4101.)*k4_g[i] - (1./5.)*k5_g[i])
        #master.traj[ii].update_x(xnew4[i,:])
        #master.traj[ii].update_p(pnew4[i,:])
        #master.traj[ii].update_phase(gnew4[i])

    #-------------------------------------------------------------------
    # Calculate the RK5 solutions at time t+dt
    #-------------------------------------------------------------------
    for i in range(master.nalive):
        ii = master.alive[i]
        xnew5[i,:] = x0[i,:] + ( (16./135.)*k1_x[i,:] +
                                 (6656./12825.)*k3_x[i,:] +
                                 (28561./56430.)*k4_x[i,:] -
                                 (9./50.)*k5_x[i,:] +
                                 (2./55.)*k6_x[i,:])
        pnew5[i,:] = p0[i,:] + ( (16./135.)*k1_p[i,:] +
                                 (6656./12825.)*k3_p[i,:] +
                                 (28561./56430.)*k4_p[i,:] -
                                 (9./50.)*k5_p[i,:] +
                                 (2./55.)*k6_p[i,:])
        gnew5[i] = g0[i] + ( (16./135.)*k1_g[i] +
                             (6656./12825.)*k3_g[i] +
                             (28561./56430.)*k4_g[i] -
                             (9./50.)*k5_g[i] + (2./55.)*k6_g[i])
        #master.traj[ii].update_x(xnew5[i,:])
        #master.traj[ii].update_p(pnew5[i,:])
        #master.traj[ii].update_phase(gnew5[i])

    #-------------------------------------------------------------------
    # Calculate the smallest optimal step size
    #-------------------------------------------------------------------
    tol=1e-6
    
    sfac=1e+10
    for i in range(master.nalive):
        for j in range(ncrd):
            s = (tol*dt/(2.0*abs(xnew5[i,j]-xnew4[i,j])))**0.25
            if s < sfac:
                sfac = s
            s = (tol*dt/(2.0*abs(pnew5[i,j]-pnew4[i,j])))**0.25
            if s < sfac:
                sfac = s
        
        s = (tol*dt/(2.0*abs(gnew5[i]-gnew4[i])))**0.25
        if s < sfac:
            sfac = s

    print(sfac)
    sys.exit()

    #-------------------------------------------------------------------
    # Calculate the potentials at the propagated phase space centres
    #-------------------------------------------------------------------
    surface.update_pes(master)

    #-------------------------------------------------------------------
    # Propagate amplitudes for 1/2 time step using Heff(t0+dt)
    #-------------------------------------------------------------------
    master.update_amplitudes(0.5*dt, 10)

    return

########################################################################

def propagate_trajectory(traj, dt):

    print("WRITE THE RKF45 propagate_trajectory CODE!")
    sys.exit()

    #-------------------------------------------------------------------
    # Initialisation
    #-------------------------------------------------------------------
    # Masses
    mass = traj.masses()

    # No. coordinates
    ncrd = glbl.fms['num_particles'] * glbl.fms['dim_particles']

    # Work arrays
    x0   = np.zeros((ncrd))
    p0   = np.zeros((ncrd))
    g0   = 0.0
    xnew = np.zeros((ncrd))
    pnew = np.zeros((ncrd))
    gnew = 0.0
    k1_x = np.zeros((ncrd))
    k1_p = np.zeros((ncrd))
    k1_g = 0.0
    k2_x = np.zeros((ncrd))
    k2_p = np.zeros((ncrd))
    k2_g = 0.0
    k3_x = np.zeros((ncrd))
    k3_p = np.zeros((ncrd))
    k3_g = 0.0
    k4_x = np.zeros((ncrd))
    k4_p = np.zeros((ncrd))
    k4_g = 0.0

    #-------------------------------------------------------------------
    # k1
    #-------------------------------------------------------------------
    # Temporary trajectory copy
    tmptraj = trajectory.copy_traj(traj)

    # Initial phase space centres
    x0[:] = tmptraj.x()
    p0[:] = tmptraj.p()
    g0    = tmptraj.phase()

    # Calculate the time-derivatives at the new phase space centres
    xdot = tmptraj.velocity()
    pdot = tmptraj.force() / mass
    gdot = tmptraj.phase_dot()
    k1_x[:] = xdot
    k1_p[:] = pdot
    k1_g    = gdot

    #-------------------------------------------------------------------
    # k2
    #-------------------------------------------------------------------
    # Update the phase space centres
    tmptraj.update_x(x0[:]  + 0.5*dt*k1_x[:])
    tmptraj.update_p(p0[:]  + 0.5*dt*k1_p[:])
    tmptraj.update_phase(g0 + 0.5*dt*k1_g)

    # Calculate the potentials at the new phase space centres
    surface.update_pes_traj(tmptraj)

    # Calculate the time-derivatives at the new phase space centres
    xdot = tmptraj.velocity()
    pdot = tmptraj.force() / mass
    gdot = tmptraj.phase_dot()
    k2_x[:] = xdot
    k2_p[:] = pdot
    k2_g    = gdot

    #-------------------------------------------------------------------
    # k3
    #-------------------------------------------------------------------
    # Update the phase space centres
    tmptraj = trajectory.copy_traj(traj)
    tmptraj.update_x(x0[:]  + 0.5*dt*k2_x[:])
    tmptraj.update_p(p0[:]  + 0.5*dt*k2_p[:])
    tmptraj.update_phase(g0 + 0.5*dt*k2_g)
        
    # Calculate the potentials at the new phase space centres
    surface.update_pes_traj(tmptraj)

    # Calculate the time-derivatives at the new phase space centres
    xdot = tmptraj.velocity()
    pdot = tmptraj.force() / mass
    gdot = tmptraj.phase_dot()
    k3_x[:] = xdot
    k3_p[:] = pdot
    k3_g    = gdot

    #-------------------------------------------------------------------
    # k4
    #-------------------------------------------------------------------
    # Update the phase space centres
    tmptraj = trajectory.copy_traj(traj)
    tmptraj.update_x(x0[:]  + dt*k3_x[:])
    tmptraj.update_p(p0[:]  + dt*k3_p[:])
    tmptraj.update_phase(g0 + dt*k3_g)

    # Calculate the potentials at the new phase space centres
    surface.update_pes_traj(tmptraj)

    # Calculate the time-derivatives at the new phase space centres
    xdot = tmptraj.velocity()
    pdot = tmptraj.force() / mass
    gdot = tmptraj.phase_dot()
    k4_x[:] = xdot
    k4_p[:] = pdot
    k4_g    = gdot

    #-------------------------------------------------------------------
    # Calculate the approximate solutions at time t+dt
    #-------------------------------------------------------------------
    xnew[:] = x0[:] + (dt/6.0) * (k1_x[:] + 2.0*k2_x[:] + 2.0*k3_x[:] +k4_x[:])
    pnew[:] = p0[:] + (dt/6.0) * (k1_p[:] + 2.0*k2_p[:] + 2.0*k3_p[:] +k4_p[:])
    gnew = g0 + (dt/6.0) * (k1_g + 2.0*k2_g + 2.0*k3_g +k4_g)
    traj.update_x(xnew[:])
    traj.update_p(pnew[:])
    traj.update_phase(gnew)

    #-------------------------------------------------------------------
    # Calculate the potentials at the propagated phase space centres
    #-------------------------------------------------------------------
    surface.update_pes_traj(traj)

    return
