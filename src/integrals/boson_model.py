#
# Compute integrals over trajectories traveling on adiabataic potentials
#  This currently uses first-order saddle point.
#
import numpy as np
import src.interfaces.boson_model_diabatic as boson
#
# Let propagator know if we need data at centroids to propagate
#
require_centroids = False

#
# potential coupling matrix element between two trajectories
#
def v_integral(traj1,traj2):

    if traj1.state == traj2.state:
        sgn   = -1. + 2.*traj1.state
        pos1  = traj1.x()
        mom1  = traj1.p()
        pos2  = traj2.x()
        mom2  = traj2.p()
        v_int = np.complex(0.,0.)
        for k in range(boson.ncrd):
            a = (1. + 1.)
            b = 2. * 1. * (pos1[k] + pos2[k]) + np.complex(0.,1)*(mom2[k] - mom1[k])
            v_int += 0.5 * boson.omega[k] * (2*a + b**2)/(4 * a**2) + sgn * boson.C[k] * b/(2*a)
        return v_int * traj1.overlap(traj2)
    else:
        return boson.delta * traj1.overlap(traj2)

#
# kinetic energy integral over trajectories
#
def ke_integral(traj1,traj2):
    ke_int = np.complex(0.,0.)
    if traj1.state == traj2.state:    
        for k in range(boson.ncrd):
            ke_int -= boson.omega[k] * traj1.particles[k].deld2x(traj2.particles[k])
        return 0.5 * ke_int * traj1.overlap(traj2)
    else:
        return ke_int

#
# return the matrix element <Psi_1 | d/dt | Psi_2> 
#
def sdot_integral(traj1,traj2):
    sdot =  -np.dot( traj2.velocity(), traj1.deldx(traj2) ) \
            +np.dot( traj2.force()   , traj1.deldp(traj2) ) \
            +np.complex(0.,1.) * traj2.phase_dot() * traj1.overlap(traj2,st_orthog=True)
    return sdot

