import math
import numpy as np
import src.fmsio.glbl as glbl
import src.basis.trajectory as trajectory
import src.basis.bundle as bundle

ncrd    = 1 
C       = np.zeros(ncrd,dtype=np.float)
omega   = np.zeros(ncrd,dtype=np.float)
omega_c = 0.
delta   = 0.
alpha   = 0.
#
#
#
def init_interface():
   global C, omega, omega_c, delta, alpha

#   omega   = np.asarray([0.01, 1.34, 2.67, 4.00], dtype=np.float)
   omega   = np.asarray([4.00], dtype=np.float)
   delta   = 1.
   omega_c = 2.5 * delta 
   d_omega = 1.33
   alpha   = glbl.boson['coupling'] 
   C = np.asarray([math.sqrt(d_omega * alpha * omega[i] * math.exp(-omega[i]/omega_c)) for i in range(ncrd)])
   print("Carray="+str(C))
   print("alpha="+str(alpha))

#
#
#
def energy(tid, geom, t_state, rstate):
    global ncrd, omega, C
    h0 = 0.
    for i in range(ncrd):   
        h0 += 0.5 * omega[i] * ( geom[i].x[0]**2 )

    hk = 0.
    for i in range(ncrd):
        hk += C[i] * geom[i].x[0]
    
    return h0 + (-1 + 2*rstate) * hk

#
#
#
def derivative(tid,geom,t_state,rstate):
    global ncrd, omega, C, delta

    ncrd = len(geom)
    if t_state == rstate:
        sgn = -1 + 2*rstate
        deriv = np.fromiter((omega[i]*geom[i].x[0] + sgn*C[i] for i in range(ncrd)),dtype=np.float)
    else:
        deriv = abs(np.fromiter((delta / sum(2 * C[i] * geom[i].x[0] for i in range(ncrd)) 
                                                               for j in range(ncrd)),dtype=np.float))
    return deriv

#
#
#
def orbitals(tid,geom, t_state):
    pass

#
#
#
def dipole(tid,geom, t_state, lstate, rstate):
    pass
#
#
#
def sec_mom(tid,geom,t_state, rstate):
    pass
#
#
#
def atom_pop(tid,geom,t_state, rstate):
    pass


