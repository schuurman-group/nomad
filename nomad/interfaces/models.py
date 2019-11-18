# file: models.py
# 
# A library of model potentials for testing 
#
import sys
import math
import numpy as np
import nomad.core.glbl as glbl
import nomad.core.surface as surface
import nomad.math.adt as adt
import nomad.math.constants as constants

#---------------------------------------------------------------------
#
# Functions called from interface object
#
#---------------------------------------------------------------------
model_potentials = {}
data_cache = {}

#
# init_interface: none of the existing models require initialization
#
def init_interface():
    global model_potentials

    model_names      = ['tully_avoided', 'tully_dual', 'tully_extended']

    if glbl.models['model_name'] not in model_names:
        sys.exit('Model: '+str(glbl.models['model_name'])+' not implemented')

    model_potentials = {
                        'tully_avoided' : tully_avoided,
                        'tully_dual'    : tully_dual,
                        'tully_extended': tully_extended
                       }
    return

#
# evaluate_trajectory: evaluate all reaqusted electronic structure
# information for a single trajectory
#
def evaluate_trajectory(traj, t=None):
    global model_potentials, data_cache

    label = traj.label
    geom  = traj.x()
    nd    = len(geom)
    ns   = traj.nstates

    # Calculation of the diabatic potential matrix and derivatives
    [diabpot, diabderiv1, diabderiv2, diablap] =\
     model_potentials[glbl.models['model_name']](geom)

    if ns != diabpot.shape[0]:
        sys.exit('Wrong number of states expected in model: '
                 +str(ns)+' != '+str(diabpot.shape[0]))

    if nd != diabderiv1.shape[0]:
        sys.exit('Wrong number of coordinates expected in model: '
                 +str(nd)+' != '+str(diabderiv1.shape[0]))

    # load the data into the pes object to pass to trajectory
    t_data = surface.Surface()
    t_data.add_data('geom',geom)

    t_data.add_data('diabat_pot',diabpot)
    t_data.add_data('diabat_deriv',diabderiv1)
    t_data.add_data('diabat_hessian',diabderiv2)

    if glbl.methods['surface'] == 'adiabatic':

        datmat_prev = None
        if traj.label in data_cache:
            datmat_prev = data_cache[traj.label].get_data('dat_mat')

        # Calculation of the adiabatic potential vector and ADT matrix
        adiabpot, datmat = adt.calc_dat(label, diabpot, 
                                        previous_datmat=datmat_prev)

        # Calculation of the NACT matrix
        nactmat = adt.calc_nacts(adiabpot, datmat, diabderiv1)

        # Calculation of the gradients (for diagonal elements) and derivative
        # couplings (off-diagonal elements)
        adiabderiv1 = adt.calc_adiabderiv1(datmat, diabderiv1)

        # Calculation of the hessians (for diagonal elements) and derivative of
        # derivative couplings (off-diagonal elements)
        adiabderiv2 = adt.calc_adiabderiv2(datmat, diabderiv2)

        t_data.add_data('potential',adiabpot)
        t_data.add_data('derivative', np.array([np.diag(adiabderiv1[m]) for m in
                                      range(nd)] + nactmat))
        t_data.add_data('hessian',adiabderiv2)

        # non-standard items
        t_data.add_data('nac',nactmat)
        t_data.add_data('dat_mat',datmat)
        t_data.add_data('adt_mat',datmat.T)

    else:
        # determine the effective diabatic coupling: Hij / (Ei - Ej)
        diab_effcoup     = adt.calc_diabeffcoup(diabpot)

        t_data.add_data('potential',np.array([diabpot[i,i] for i in range(ns)]))
        t_data.add_data('derivative',diabderiv1)
        t_data.add_data('hessian',diabderiv2)

    data_cache[traj.label] = t_data

    return t_data

#
# evaluate_centroid: evaluate all requested electronic structure 
# information at a centroid
#
def evaluate_centroid(cent, t=None):

    return evaluate_trajectory(cent, t=None) 

#
# evaluate the coupling between electronic states
# 1. for adiabatic basis, this will just be the derivative coupling dotted into
#    the velocity
# 2. for diabatic basis, it will be the potential coupling, or something else
#
def evaluate_coupling(traj):
    """Updates the coupling to the other states"""

    ns = traj.nstates 

    if glbl.methods['surface'] == 'adiabatic':
        vel   = traj.velocity()
        deriv = traj.pes.get_data('derivative')

        coup = np.array([[np.dot(vel, deriv[:,i,j]) for i in range(ns)]
                                                    for j in range(ns)])
        coup -= np.diag(coup.diagonal())
        traj.pes.add_data('coupling',coup)


    else:
        diabpot          = traj.pes.get_data('diabat_pot')
        diab_effcoup     = adt.calc_diabeffcoup(diabpot)
        traj.pes.add_data('coupling',diab_effcoup)


#---------------------------------------------------------------------
#
# MODELS
#
#--------------------------------------------------------------------
#
# The Tully simple avoided crossing model, taken from JCP, 93, 1061 (1990)
#
def tully_avoided(geom):
    """The Tully simple avoided crossing model, taken from 
       JCP, 93, 1061 (1990)"""
    x = geom[0]
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0

    v11 = np.sign(x) * A * (1 - math.exp(-B*abs(x)))
    v22 = -v11
    v12 = C * math.exp(-D * x**2)

    dv11 = A * B * math.exp( -B*abs(x))
    dv22 = -dv11
    dv12 = - 2 * C * D * x * math.exp(-D * x**2)

    d2v11 = -np.sign(x) * A * B**2 * math.exp( -B*abs(x))
    d2v22 = -d2v11
    d2v12 = 4 * C * D**2 * x**2 * math.exp(-D * x**2)

    diabpot    = np.array([[v11, v12], [v12, v22]], dtype=float)
    diabderiv1 = np.array([[[dv11, dv12], [dv12, dv22]]], dtype=float)    
    diabderiv2 = np.array([[[[d2v11, d2v12], [d2v12, d2v22]]]], dtype=float)
    diablap    = diabderiv2[0,0,:,:]

    return [diabpot, diabderiv1, diabderiv2, diablap]

#
def tully_dual(geom):
    """The Tully simple dual crossing model, taken from 
       JCP, 93, 1061 (1990)"""
    x  = geom[0]
    A  = 0.1
    B  = 0.28
    C  = 0.015
    D  = 0.06
    E0 = 0.05

    v11 = 0. 
    v22 = -A * math.exp(-B * x**2) + E0
    v12 = C * math.exp(-D * x**2)

    dv11 = 0. 
    dv22 =  2 * A * B * x * math.exp(-B * x**2)
    dv12 = -2 * C * D * x * math.exp(-D * x**2) 

    d2v11 = 0. 
    d2v22 = -4 * A * B**2 * x**2 * math.exp(-B * x**2)
    d2v12 =  4 * C * D**2 * x**2 * math.exp(-D * x**2)

    diabpot    = np.array([[v11, v12], [v12, v22]])
    diabderiv1 = np.array([[[dv11, dv12], [dv12, dv22]]])
    diabderiv2 = np.array([[[[d2v11, d2v12], [d2v12, d2v22]]]])
    diablap    = diabderiv2[0,0,:,:]

    return [diabpot, diabderiv1, diabderiv2, diablap]

#
def tully_extended(geom):
    """The Tully extended coupling model, taken from 
       JCP, 93, 1061 (1990)"""
    x = geom[0]
    A = 0.0006 
    B = 0.1
    C = 0.9

    v11 = A 
    v22 = -A
    v12 = B * (1.+ np.sign(x) + (np.sign(-x) * math.exp(np.sign(-x)*C*x)))

    dv11 = 0. 
    dv22 = 0.
    dv12 = B * C * math.exp(np.sign(-x)*C*x)

    d2v11 = 0. 
    d2v22 = 0.
    d2v12 = np.sign(-x) * B * C**2 * math.exp(np.sign(-x)*C*x)

    diabpot    = np.array([[v11, v12], [v12, v22]])
    diabderiv1 = np.array([[[dv11, dv12], [dv12, dv22]]])
    diabderiv2 = np.array([[[[d2v11, d2v12], [d2v12, d2v22]]]])
    diablap    = diabderiv2[0,0,:,:]

    return [diabpot, diabderiv1, diabderiv2, diablap]

