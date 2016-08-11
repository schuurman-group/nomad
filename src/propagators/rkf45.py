import sys
import numpy as np
import src.fmsio.glbl as glbl
import src.dynamics.timings as timings
import src.dynamics.surface as surface
import src.basis.bundle as bundle
import src.basis.trajectory as trajectory

tol=1e-6

err=0.
sfac=0.
xnew4=None
pnew4=None
gnew4=None
hlast=None

xnew4_traj=None
pnew4_traj=None
gnew4_traj=None
hlast_traj=None

########################################################################
# RKF45 code
########################################################################

def propagate_bundle(master, dt):

    global err
    global sfac
    global tol
    global xnew4
    global pnew4
    global gnew4
    global hlast

    # Set the initial step size
    if hlast is None:
        h=dt
    else:
        h=hlast
    
    # propagate amplitudes for 1/2 time step using x0
    #master.update_amplitudes(0.5*dt, 10)
    
    # Propagate the bundle of trajectories
    t=0.
    while t < dt:

        # Save the last time step value
        hlast=h

        # Make sure that we hit time dt
        if t+h > dt:
            h=dt-t

        # Propagate forwards one step, adapting the timestep
        # to keep the error estimate below tolerance
        success=False
        while not success:
            rkf45_bundle(master, h)
            if err > tol:
                h = h/2.
            else:
                hsucc=h
                t+=h
                success = True
                if t < dt and sfac >2.0:
                    h=h*2.

        # propagate amplitudes for 1/2 time step using x0
        master.update_amplitudes(0.5*hsucc, 10)

        # Update the Gaussian parameters and PESs
        for i in range(master.nalive):
            ii = master.alive[i]
            master.traj[ii].update_x(xnew4[i,:])
            master.traj[ii].update_p(pnew4[i,:])
            master.traj[ii].update_phase(gnew4[i])
        surface.update_pes(master)
        
        # propagate amplitudes for 1/2 time step using x1
        master.update_amplitudes(0.5*hsucc, 10)

    # propagate amplitudes for 1/2 time step using x1
    #master.update_amplitudes(0.5*dt, 10)

########################################################################
        
def rkf45_bundle(master, dt):

    global err
    global sfac
    global tol
    global xnew4
    global pnew4
    global gnew4
    
    #-------------------------------------------------------------------
    # Initialisation
    #-------------------------------------------------------------------
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
        pdot = tmpbundle.traj[ii].force()
        gdot = tmpbundle.traj[ii].phase_dot()
        k1_x[i,:] = dt*xdot
        k1_p[i,:] = dt*pdot
        k1_g[i]   = dt*gdot

    #-------------------------------------------------------------------
    # k2
    #-------------------------------------------------------------------
    # Update the phase space centres
    for i in range(master.nalive):
        ii = master.alive[i]
        tmpbundle.traj[ii].update_x(x0[i,:]   + 0.25*k1_x[i,:])
        tmpbundle.traj[ii].update_p(p0[i,:]   + 0.25*k1_p[i,:])
        tmpbundle.traj[ii].update_phase(g0[i] + 0.25*k1_g[i])
    
    # Calculate the potentials at the new phase space centres
    surface.update_pes(tmpbundle)
    
    # Calculate the time-derivatives at the new phase space centres
    for i in range(master.nalive):
        ii = master.alive[i]
        xdot = tmpbundle.traj[ii].velocity()
        pdot = tmpbundle.traj[ii].force()
        gdot = tmpbundle.traj[ii].phase_dot()
        k2_x[i,:] = dt*xdot
        k2_p[i,:] = dt*pdot
        k2_g[i]   = dt*gdot
    
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
        pdot = tmpbundle.traj[ii].force()
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
        pdot = tmpbundle.traj[ii].force()
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
                                    (845./4104.)*k4_x[i,:]) 
        tmpbundle.traj[ii].update_p(p0[i,:] + (439./216.)*k1_p[i,:]
                                    -8.*k2_p[i,:]
                                    +(3680./513.)*k3_p[i,:] -
                                    (845./4104.)*k4_p[i,:]) 
        tmpbundle.traj[ii].update_phase(g0[i] + (439./216.)*k1_g[i]
                                    -8.*k2_g[i]
                                    +(3680./513.)*k3_g[i] -
                                    (845./4104.)*k4_g[i]) 

    # Calculate the potentials at the new phase space centres
    surface.update_pes(tmpbundle)

    # Calculate the time-derivatives at the new phase space centres
    for i in range(master.nalive):
        ii = master.alive[i]
        xdot = tmpbundle.traj[ii].velocity()
        pdot = tmpbundle.traj[ii].force()
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
        pdot = tmpbundle.traj[ii].force()
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

    #-------------------------------------------------------------------
    # Calculate the error estimates
    #-------------------------------------------------------------------
    err=0.
    for i in range(master.nalive):
        for j in range(ncrd):
            tmp = abs(xnew5[i,j]-xnew4[i,j])
            if tmp > err:
                err = tmp
            tmp = abs(pnew5[i,j]-pnew4[i,j])
            if tmp > err:       
                err = tmp
        tmp=abs(gnew5[i]-gnew4[i])
        if tmp > err:
            err = tmp

    if err == 0.0:
        sfac = 1.0
    else:
        sfac = 0.9*(tol/err)**0.2

    
        
########################################################################

def propagate_trajectory(traj, dt):

    global err
    global sfac
    global tol
    global xnew4_traj
    global pnew4_traj
    global gnew4_traj
    global hlast_traj

    # Set the initial step size
    if hlast_traj is None:
        h=dt
    else:
        h=hlast

    # Propagate the trajectory
    t=0.
    while t < dt:

        # Make sure that we hit time dt
        if t+h > dt:
            h=dt-t

        # Propagate forwards one step, adapting the timestep
        # to keep the error estimate below tolerance
        success=False        
        while not success:
            rkf45_trajectory(traj, h)
            if err > tol:
                h = h/2.
            else:
                t+=h
                success = True
                if t < dt and sfac >2.0:
                    h=h*2.
        traj.update_x(xnew4_traj[:])
        traj.update_p(pnew4_traj[:])
        traj.update_phase(gnew4_traj)
        surface.update_pes_traj(traj)

########################################################################
           
def rkf45_trajectory(traj,dt):

    global err
    global sfac
    global tol
    global xnew4_traj
    global pnew4_traj
    global gnew4_traj
    
    #-------------------------------------------------------------------
    # Initialisation
    #-------------------------------------------------------------------
    # No. coordinates
    ncrd = glbl.fms['num_particles'] * glbl.fms['dim_particles']

    # Work arrays
    x0   = np.zeros((ncrd))
    p0   = np.zeros((ncrd))
    g0   = 0.0
    xnew4_traj = np.zeros((ncrd))
    pnew4_traj = np.zeros((ncrd))
    gnew4_traj = 0.0
    xnew5_traj = np.zeros((ncrd))
    pnew5_traj = np.zeros((ncrd))
    gnew5_traj = 0.0
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
    k5_x = np.zeros((ncrd))
    k5_p = np.zeros((ncrd))
    k5_g = 0.0
    k6_x = np.zeros((ncrd))
    k6_p = np.zeros((ncrd))
    k6_g = 0.0
    
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
    pdot = tmptraj.force()
    gdot = tmptraj.phase_dot()
    k1_x[:] = dt*xdot
    k1_p[:] = dt*pdot
    k1_g    = dt*gdot

    #-------------------------------------------------------------------
    # k2
    #-------------------------------------------------------------------
    # Update the phase space centres
    tmptraj.update_x(x0[:]  + 0.25*k1_x[:])
    tmptraj.update_p(p0[:]  + 0.25*k1_p[:])
    tmptraj.update_phase(g0 + 0.25*k1_g)

    # Calculate the potentials at the new phase space centres
    surface.update_pes_traj(tmptraj)

    # Calculate the time-derivatives at the new phase space centres
    xdot = tmptraj.velocity()
    pdot = tmptraj.force()
    gdot = tmptraj.phase_dot()
    k2_x[:] = dt*xdot
    k2_p[:] = dt*pdot
    k2_g    = dt*gdot

    #-------------------------------------------------------------------
    # k3
    #-------------------------------------------------------------------
    # Update the phase space centres
    tmptraj = trajectory.copy_traj(traj)
    tmptraj.update_x(x0[:]   + (3./32.)*k1_x[:] + (9./32.)*k2_x[:]) 
    tmptraj.update_p(p0[:]   + (3./32.)*k1_p[:] + (9./32.)*k2_p[:]) 
    tmptraj.update_phase(g0  + (3./32.)*k1_g    + (9./32.)*k2_g) 
    
    # Calculate the potentials at the new phase space centres
    surface.update_pes_traj(tmptraj)

    # Calculate the time-derivatives at the new phase space centres
    xdot = tmptraj.velocity()
    pdot = tmptraj.force()
    gdot = tmptraj.phase_dot()
    k3_x[:] = dt*xdot
    k3_p[:] = dt*pdot
    k3_g    = dt*gdot

    #-------------------------------------------------------------------
    # k4
    #-------------------------------------------------------------------
    # Update the phase space centres
    tmptraj = trajectory.copy_traj(traj)
    tmptraj.update_x(x0[:] + (1932./2197.)*k1_x[:] - (7200./2197.)*k2_x[:]
                     + (7296./2197.)*k3_x[:])
    tmptraj.update_p(p0[:] + (1932./2197.)*k1_p[:] - (7200./2197.)*k2_p[:]
                     + (7296./2197.)*k3_p[:])
    tmptraj.update_phase(g0 + (1932./2197.)*k1_g - (7200./2197.)*k2_g
                     + (7296./2197.)*k3_g)


    # Calculate the potentials at the new phase space centres
    surface.update_pes_traj(tmptraj)

    # Calculate the time-derivatives at the new phase space centres
    xdot = tmptraj.velocity()
    pdot = tmptraj.force()
    gdot = tmptraj.phase_dot()
    k4_x[:] = dt*xdot
    k4_p[:] = dt*pdot
    k4_g    = dt*gdot

    #-------------------------------------------------------------------
    # k5
    #-------------------------------------------------------------------
    # Update the phase space centres
    tmptraj = trajectory.copy_traj(traj)
    tmptraj.update_x(x0[:] + (439./216.)*k1_x[:] -8.*k2_x[:]
                     + (3680./513.)*k3_x[:]
                     - (845./4104.)*k4_x[:]) 
    tmptraj.update_p(p0[:] + (439./216.)*k1_p[:] -8.*k2_p[:]
                     + (3680./513.)*k3_p[:]
                     - (845./4104.)*k4_p[:]) 
    tmptraj.update_phase(g0 + (439./216.)*k1_g -8.*k2_g
                     + (3680./513.)*k3_g
                     - (845./4104.)*k4_g) 
    
    # Calculate the potentials at the new phase space centres
    surface.update_pes_traj(tmptraj)

    # Calculate the time-derivatives at the new phase space centres
    xdot = tmptraj.velocity()
    pdot = tmptraj.force()
    gdot = tmptraj.phase_dot()
    k5_x[:] = dt*xdot
    k5_p[:] = dt*pdot
    k5_g    = dt*gdot

    #-------------------------------------------------------------------
    # k6
    #-------------------------------------------------------------------
    # Update the phase space centres
    tmptraj = trajectory.copy_traj(traj)
    tmptraj.update_x(x0[:] - (8./27.)*k1_x[:] + 2.*k2_x[:]
                     - (3544./2565.)*k3_x[:]
                     + (1859./4104.)*k4_x[:]
                     - (11./40.)*k5_x[:])
    tmptraj.update_p(p0[:] - (8./27.)*k1_p[:] + 2.*k2_p[:]
                     - (3544./2565.)*k3_p[:]
                     + (1859./4104.)*k4_p[:]
                     - (11./40.)*k5_p[:])
    tmptraj.update_phase(g0 - (8./27.)*k1_g + 2.*k2_g
                     - (3544./2565.)*k3_g
                     + (1859./4104.)*k4_g
                     - (11./40.)*k5_g)
    
    # Calculate the potentials at the new phase space centres
    surface.update_pes_traj(tmptraj)

    # Calculate the time-derivatives at the new phase space centres
    xdot = tmptraj.velocity()
    pdot = tmptraj.force()
    gdot = tmptraj.phase_dot()
    k6_x[:] = dt*xdot
    k6_p[:] = dt*pdot
    k6_g    = dt*gdot
    
    #-------------------------------------------------------------------
    # Calculate the RK4 solutions at time t+dt
    #-------------------------------------------------------------------
    xnew4_traj[:] = x0[:] + ((25./216.)*k1_x[:] + (1408./2565.)*k3_x[:]
                        + (2197./4101.)*k4_x[:] - (1./5.)*k5_x[:])

    pnew4_traj[:] = p0[:] + ((25./216.)*k1_p[:] + (1408./2565.)*k3_p[:]
                        + (2197./4101.)*k4_p[:] - (1./5.)*k5_p[:])
    
    gnew4_traj = g0 + ((25./216.)*k1_g + (1408./2565.)*k3_g
                        + (2197./4101.)*k4_g - (1./5.)*k5_g)

    #-------------------------------------------------------------------
    # Calculate the RK5 solutions at time t+dt
    #-------------------------------------------------------------------
    xnew5_traj[:] = x0[:] + ( (16./135.)*k1_x[:]
                              + (6656./12825.)*k3_x[:]
                              + (28561./56430.)*k4_x[:]
                              - (9./50.)*k5_x[:]
                              + (2./55.)*k6_x[:])

    pnew5_traj[:] = p0[:] + ( (16./135.)*k1_p[:]
                              + (6656./12825.)*k3_p[:]
                              + (28561./56430.)*k4_p[:]
                              - (9./50.)*k5_p[:]
                              + (2./55.)*k6_p[:])

    gnew5_traj = g0 + ( (16./135.)*k1_g
                              + (6656./12825.)*k3_g
                              + (28561./56430.)*k4_g
                              - (9./50.)*k5_g
                              + (2./55.)*k6_g)

    #-------------------------------------------------------------------
    # Calculate the error estimates
    #-------------------------------------------------------------------
    err=0.
    for j in range(ncrd):
        tmp = abs(xnew5_traj[j]-xnew4_traj[j])
        if tmp > err:
            err = tmp
        tmp = abs(pnew5_traj[j]-pnew4_traj[j])
        if tmp > err:
            err = tmp
    tmp=abs(gnew5_traj-gnew4_traj)
    if tmp > err:
        err = tmp

    sfac = 0.9*(tol/err)**0.2
    
