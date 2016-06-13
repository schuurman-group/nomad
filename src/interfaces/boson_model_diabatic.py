import math
import numpy as np
from src.fmsio import glbl

comp_properties = False

#
#
def init_interface():
   global C, omega, omega_c, delta, ncrd

   ncrd    = 1 
#   omega   = np.asarray([0.01, 1.34, 2.67, 4.00], dtype=np.float)
   omega   = np.asarray([1.34],dtype=np.float)  
#   delta   = 1.
   delta   = 0.
#   omega_c = 2.5 * delta
   omega_c = 2.5
   d_omega = 1.33
   alpha   = glbl.boson['coupling'] 
#   C = np.array([math.sqrt(d_omega * alpha * omega[i] * math.exp(-omega[i]/omega_c)) for i in range(ncrd)])
   C = np.array([0. for i in range(ncrd)])

#
# evaluate trajectory energy and gradients
#
def evaluate_trajectory(tid, geom, t_state):
   global ncrd

   gm = np.array([geom[i].x[j] for i in range(ncrd) 
                               for j in range(geom[i].dim)],dtype=np.float)

   eners = energy(gm)
   grads = derivative(gm, t_state)

   return[gm,eners,grads]

#
# evaluate energy in boson model
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
#            grads[i,:] = np.array([omega[i]*geom[i] + sgn*C[i] for i in range(ncrd)],dtype=np.float)
            grads[i,:] = np.array([omega[i]*geom[i] for i in range(ncrd)],dtype=np.float)
        else:
#            coup = delta / abs(sum(2 * C[i] * geom[i] for i in range(ncrd)))
            coup = 0.
            grads[i,:] = np.array([coup for j in range(ncrd)],dtype=np.float)      
    return grads 


#
# evaluate worker for parallel job (global variables passed as parameters)
#
def evaluate_worker(packet, global_var):
   global ncrd, omega, C, delta

   tid = packet[0]
   geom = packet[1]
   t_state = packet[2]

   set_global_vars(global_var)

   xval = [geom[i].x[0] for i in range(len(geom))]
   dims = [geom[i].dim for i in range(len(geom))]

   gm = np.array([geom[i].x[j] for i in range(ncrd)
                               for j in range(geom[i].dim)],dtype=np.float)

   eners = energy(gm)
   grads = derivative(gm, t_state)

   return[gm,eners,grads]

#
# return the value of global variables 
#
def set_global_vars(gvars):
    global ncrd,omega,C,delta

    ncrd  = gvars[0]
    omega = gvars[1]
    delta = gvars[2]
    C     = gvars[3]
 
    return

#
# package global variables into a list
#
def get_global_vars():
    global ncrd,omega,C,delta

    return [ncrd, omega, delta, C]


