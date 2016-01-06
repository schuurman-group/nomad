#
# Compute integrals over trajectories traveling on adiabataic potentials
#  This currently uses first-order saddle point.
#
import numpy as np
import src.basis.trajectory as trajectory
import src.interfaces.vibronic as vibronic
#
# potential coupling matrix element between two trajectories
#
def v_integral(traj1,traj2=None,centroid=None):
    # this will depend on how the operator is stored, and
    # what type of coordinates are employed, etc.
    pass

#
# This routine returns the matrix element <cmplx_gaus(q,p)| q^N |cmplx_gaus(q,p)>
#  -- takes the arguments as particles
def prim_v_integral(n, p1, p2):
    n_2 = math.floor(0.5 * n)
    a   = p1.width + p2.width
    b   = np.fromiter( (complex( 2.0*(p1.width*p1.x(i) + p2.width*p2.x(i)), 
                       -(p1.p(i)-p2.p(i))) for i in range(p1.dim)), np.complex)

    # generally these should be 1D harmonic oscillators. If multi-dimensional, the 
    # final result is a direct product of each dimension
    v_total  = complex(1.,0.)
    for d in range(p1.dim):
        v = complex(0.,0.)
        for i in range(n_2):
            v = v + a**(i-N) * b**(N-2*i) /  \
                (math.factorial(i)*math.factorial(N-2*i))
        v_total = v_total * v

    # refer to appendix for derivation of these relations
    return v_total * math.factorial(N) / 2**N

#
# kinetic energy integral over trajectories
#
def ke_integral(traj1,traj2):
    ke = complex(0.,0.)
    if traj1.state == traj2.state:
        for i in range(traj1.nparticles):
            ke = ke - traj1.particles[i].deld2x(traj2.particles[i]) /  \
                      (2.0*traj1.particles[i].mass)
        return ke * traj1.overlap(traj2)
    else:
        return ke

#
# return the matrix element <Psi_1 | d/dt | Psi_2> 
#
def sdot_integral(traj1,traj2):
     sdot = (-np.dot( traj2.velocity(), traj1.deldx(traj2) )   \
             +np.dot( traj2.force()   , traj1.deldp(traj2) )   \
             +complex(0.,1.) * traj2.phase_dot() * traj1.overlap(traj2)
     return sdot


