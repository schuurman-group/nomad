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

    sgn   = (-1)**(traj1.state+1)
    if traj1.state == traj2.state:
        pos1  = traj1.x()
        mom1  = traj1.p()
        pos2  = traj2.x()
        mom2  = traj2.p()
        v_int = complex(0.,0.)
        for k in range(boson.ncrd):
            a = (boson.omega[k] + boson.omega[k])
            b = 2. * boson.omega[k] * (pos1[k] + pos2[k]) + complex(0.,1)*(mom1[k] - mom2[k])
            v_int += (0.5*boson.omega[k]*( (2*a+b**2)/(4*a**2) ) + sgn * boson.C[k]*(b/(2*a)))
        return v_int * traj1.overlap(traj2)
    else:
        return boson.delta * traj1.overlap(traj2)

#
# kinetic energy integral over trajectories
#
def ke_integral(traj1,traj2):
    ke = complex(0.,0.)
    if traj1.state == traj2.state:
        for i in range(boson.ncrd):
            ke -= 0.5 * boson.omega[i] * traj1.particles[i].deld2x(traj2.particles[i])
        return ke * traj1.overlap(traj2)
    else:
        return ke

#
# return the matrix element <Psi_1 | d/dt | Psi_2> 
#
def sdot_integral(traj1,traj2):
    sdot =  -np.dot( traj2.velocity(), traj1.deldx(traj2) ) \
            +np.dot( traj2.force()   , traj1.deldp(traj2) ) \
            +complex(0.,1.) * traj2.phase_dot() * traj1.overlap(traj2,st_orthog=True)
    return sdot

