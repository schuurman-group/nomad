import numpy as np
from ..basis import trajectory
#
# all propagators have to define a function "propagate" that takes
# a trajectory argument and a time step argument
#
def propagate(master,dt):

    dt_2 = dt / 2.
    dt_6 = dt / 6.

    for i in range(master.n_total()):
   
        if not master.traj[i].alive:
            continue

        traj = master.traj[i]

        x0 = traj.x()
        p0 = traj.p()
        v0 = traj.velocity()
        f0 = traj.force()
        print("origin force: "+str(f0))
        g0 = traj.phase
        d_g0 = traj.phase_dot()

        g1 = g0 + dt_2 * d_g0
        x1 = x0 + dt_2 * v0
        p1 = p0 + dt_2 * f0
        traj.update_x(x1)
        traj.update_p(p1)
        traj.update_phase(g1)
        v1 = traj.velocity()
        f1 = traj.force()
        d_g1 = traj.phase_dot()

        g2 = g0 + dt_2 * d_g1
        x2 = x0 + dt_2 * v1
        p2 = x0 + dt_2 * f1
        traj.update_x(x2)
        traj.update_p(p2)
        traj.update_phase(g2)
        v2 = traj.velocity()
        f2 = traj.force()
        d_g2 = traj.phase_dot()

        g3 = g0 + dt * d_g2
        x3 = x0 + dt * v2
        p3 = x0 + dt * f2
        traj.update_x(x3)
        traj.update_p(p3)
        traj.update_phase(g3)
        v3 = traj.velocity()
        f3 = traj.force()
        d_g3 = traj.phase_dot()

        traj.update_phase( g0 + dt_6 * (d_g0 + 2*d_g1 + 2*d_g2 + d_g3) ) 
        traj.update_x(     x0 + dt_6 * (v0 + 2*v1 + 2*v2 + v3) )
        traj.update_p(     p0 + dt_6 * (f0 + 2*f1 + 2*f2 + f3) )  

