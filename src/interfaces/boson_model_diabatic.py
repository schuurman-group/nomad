import math
import numpy as np
import src.fmsio.glbl as glbl

comp_properties = False

ncrd    = 4 
C       = np.zeros(ncrd,dtype=np.float)
omega   = np.zeros(ncrd,dtype=np.float)
omega_c = 0.
delta   = 0.
#
#
#
def init_interface():
   global C, omega, omega_c, delta, ncrd

   omega   = np.asarray([0.01, 1.34, 2.67, 4.00], dtype=np.float)
   delta   = 1.
   omega_c = 2.5 * delta 
   d_omega = 1.33
   alpha   = glbl.boson['coupling'] 
   C = np.asarray([math.sqrt(d_omega * alpha * omega[i] * math.exp(-omega[i]/omega_c)) for i in range(ncrd)])

def evaluate_trajectory(tid, geom, t_state):
   global ncrd

   gm = np.array([geom[i].x[j] for i in range(ncrd) 
                               for j in range(geom[i].dim)],dtype=np.float)

   eners = energy(gm)
   grads = derivative(gm, t_state)

   return[gm,eners,grads]

#
#
#
def energy(geom):
    global ncrd, omega, C

    h0  = np.zeros(2,dtype=np.float)
    sgn = np.ones(2,dtype=np.float)
    sgn[0] = -1.

    for i in range(ncrd):   
        h0 += 0.5 * omega[i] * geom[i]**2

    hk = 0.
    for i in range(ncrd):
        hk += C[i] * geom[i]
   
    return h0 + sgn * hk

#
#
#
def derivative(geom, t_state):
    global ncrd, omega, C, delta
    grads = np.zeros((2,ncrd),dtype=np.float)

    sgn = -1 + 2*t_state
    for i in range(2):
        if t_state == i:
            grads[i,:] = np.array([omega[i]*geom[i] + sgn*C[i] for i in range(ncrd)],dtype=np.float)
        else:
            coup = delta / abs(sum(2 * C[i] * geom[i] for i in range(ncrd)))
            grads[i,:] = np.array([coup for j in range(ncrd)],dtype=np.float)      
    return grads 

