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
def v_integral(traj1,traj2=None):

    if not traj2:
        v_int = complex(1.,0.)
        pos   = traj1.x()
        mom   = traj1.p()
        for k in range(boson.ncrd):
            a = (boson.omega[k] + boson.omega[k])
            b = 2. * (boson.omega[k]*pos[k] + boson.omega[k]*pos[k])-complex(0.,1)*(mom[k]-mom[k])
            v_int *= 0.5*boson.omega[k]*( (2*a+b**2)/(4*a**2)) - boson.C[k]*(b/(2*a))
        return v_int
    elif traj1.state == traj2.state:
        sgn   = (-1)**(traj1.state+1)
        v_int = complex(1.,0.)
        pos1   = traj1.x()
        mom1   = traj1.p()
        pos2   = traj2.x()
        mom2   = traj2.p()
        for k in range(boson.ncrd):
            a = (boson.omega[k] + boson.omega[k])
            b = 2. * (boson.omega[k]*pos1[k] + boson.omega[k]*pos2[k])-complex(0.,1)*(mom1[k]-mom2[k])
            v_int *= 0.5*boson.omega[k]*( (2*a+b**2)/(4*a**2)) + sgn * boson.C[k]*(b/(2*a))
        return v_int * traj1.overlap(traj2)

    else:
        return traj1.overlap(traj2)*boson.delta

#
# kinetic energy integral over trajectories
#
def ke_integral(traj1,traj2):
    ke = complex(0.,0.)
    if traj1.state == traj2.state:
        for i in range(traj1.n_particle):
            ke = ke - traj1.particles[i].deld2x(traj2.particles[i]) /  \
                      (2.0*traj1.particles[i].mass)
        return ke * traj1.overlap(traj2)
    else:
        return ke

#
# return the matrix element <Psi_1 | d/dt | Psi_2> 
#
def sdot_integral(traj1,traj2):
    sdot =  -np.dot( traj2.velocity(), traj1.deldx(traj2) ) +  \
             np.dot( traj2.force()   , traj1.deldp(traj2) ) +  \
             complex(0.,1.) * traj2.phase_dot() * traj1.overlap(traj2)
    return sdot

