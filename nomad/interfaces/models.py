# file: models.py
# 
# A library of model potentials for testing 
#
import sys
import math
import numpy as np
import nomad.core.glbl as glbl
import nomad.core.surface as surface
import nomad.math.constants as constants

#---------------------------------------------------------------------
#
# Functions called from interface object
#
#---------------------------------------------------------------------
model_potentials = {}
model_names      = ['tully_avoided', 'tully_dual', 'tully_extended']

#
# init_interface: none of the existing models require initialization
#
def init_interface():
    global model_potentials, model_names

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
    global model_potentials

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
        # Calculation of the adiabatic potential vector and ADT matrix
        adiabpot, datmat = calc_dat(label, diabpot)

        # Calculation of the NACT matrix
        nactmat = calc_nacts(adiabpot, datmat, diabderiv1)

        # Calculation of the gradients (for diagonal elements) and derivative
        # couplings (off-diagonal elements)
        adiabderiv1 = calc_adiabderiv1(datmat, diabderiv1)

        # Calculation of the hessians (for diagonal elements) and derivative of
        # derivative couplings (off-diagonal elements)
        adiabderiv2 = calc_adiabderiv2(datmat, diabderiv2)

        # Calculation of the scalar couplings terms (SCTs)
        # Note that in order to calculate the SCTs, we get the gradients of the
        # diagonal Born-Oppenheimer corrections (DBOCs). Consequently, we
        # save these as a matter of course.
        sctmat, dbocderiv1 = calc_scts(adiabpot, datmat,      diabderiv1,
                                       nactmat,  adiabderiv1, diablap)

        t_data.add_data('potential',adiabpot)
        t_data.add_data('derivative', np.array([np.diag(adiabderiv1[m]) for m in
                                      range(nd)] + nactmat))
        t_data.add_data('hessian',adiabderiv2)

        # non-standard items
        t_data.add_data('nac',nactmat)
        t_data.add_data('scalar_coup',0.5*sctmat) #account for the 1/2 prefactor in the EOMs
        t_data.add_data('dat_mat',datmat)
        t_data.add_data('adt_mat',datmat.T)

    else:
        # determine the effective diabatic coupling: Hij / (Ei - Ej)
        diab_effcoup     = calc_diabeffcoup(diabpot)

        t_data.add_data('potential',np.array([diabpot[i,i] for i in range(ns)]))
        t_data.add_data('derivative',diabderiv1)
        t_data.add_data('hessian',diabderiv2)

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
        diab_effcoup     = calc_diabeffcoup(diabpot)
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

def tully_dual(geom):

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



def tully_extended(geom):

    x = geom[0]
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0

    v11 = A 
    v22 = -A
    v12 = B * ((1.+np.sign(x)) - math.exp(-C*abs(x)))

    dv11 = 0. 
    dv22 = 0.
    dv12 = B * C * math.exp(-C*abs(x))

    d2v11 = 0. 
    d2v22 = 0.
    d2v12 = -np.sign(x) * B * C**2 * math.exp(-C*abs(x))

    diabpot    = np.array([[v11, v12], [v12, v22]])
    diabderiv1 = np.array([[[dv11, dv12], [dv12, dv22]]])
    diabderiv2 = np.array([[[[d2v11, d2v12], [d2v12, d2v22]]]])
    diablap    = diabderiv2[0,0,:,:]

    return [diabpot, diabderiv1, diabderiv2, diablap]


#---------------------------------------------------------------------------
#
# The following functions are replicated from vibronic.py -- we should
# consoliate these into a set of adiabatic <--> diabatic transformation
# routines.
#
#----------------------------------------------------------------------------

def calc_diabeffcoup(diabpot):
    """Calculates the effective diabatic coupling between diabatic states i, j via
       eff_coup = Hij / (H[i,i] - H[j,j])
    """
    nst = diabpot.shape[0]

    demat = np.array([[max([constants.fpzero, diabpot[i,i]-diabpot[j,j]],key=abs)
                       for i in range(nst)] for j in range(nst)])
    eff_coup = np.divide(diabpot - np.diag(diabpot.diagonal()), demat)

    return eff_coup

#
def calc_dat(label, diabpot):
    """Diagonalises the diabatic potential matrix to yield the adiabatic
    potentials and the adiabatic-to-diabatic transformation matrix."""
    adiabpot, datmat = sp_linalg.eigh(diabpot)

    if label in data_cache:
        # Ensure phase continuity from geometry to another
        datmat *= np.sign(np.dot(datmat.T, data_cache[label].get_data('dat_mat').diagonal()))
    else:
        # Set phase convention that the greatest abs element in dat column
        # vector is positive
        datmat *= np.sign(datmat[range(len(adiabpot)),
                                 np.argmax(np.abs(datmat), axis=0)])
    return adiabpot, datmat

#
def calc_nacts(adiabpot, datmat, diabderiv1):
    """Calculates the matrix of non-adiabatic coupling terms from the
    adiabatic potentials, the ADT matrix and the nuclear derivatives of
    the diabatic potentials.

    Equation used: F_aij = [S (d/dX_a W) S^T]_ij / (V_j - V_i)
    F_aij = < psi_i | d/dX_a psi_j >: non-adiabatic coupling terms
    W: diabatic potential matrix
    S^T: matrix of eigenvectors of W
    V: vector of adiabatic energies
    """
    # Fill in the matrix
    nactmat = np.zeros((ham.nmode_total, nsta, nsta))
    fac = -np.subtract.outer(adiabpot, adiabpot) + np.eye(nsta)
    for m in ham.mrange:
        nactmat[m] = np.dot(np.dot(datmat.T, diabderiv1[m]), datmat) / fac

    # Subtract the diagonal to make sure it is zero
    nactmat -= [np.diag(nactmat[m].diagonal()) for m in range(ham.nmode_total)]
    return nactmat

#
def calc_adiabderiv1(datmat, diabderiv1):
    """Calculates the gradients of the adiabatic potentials.

    Equation used: d/dX V_ii = (S{d/dX W}S^T)_ii
    """
    # Get the diagonal elements of the matrix
    adiabderiv1 = np.zeros((ham.nmode_total, nsta))
    for m in ham.mrange:
        adiabderiv1[m] = np.dot(np.dot(datmat.T, diabderiv1[m]),
                                datmat).diagonal()
    return adiabderiv1

#
def calc_adiabderiv2(datmat, diabderiv2):
    """Calculates the hessians of the adiabatic potentials.

    Equation used: d^2/dXidXj V_ii = (S{d^2/dXidXj W}S^T)_ii
    """
    # Get the diagonal elements of the matrix
    adiabderiv2 = np.zeros((ham.nmode_total, ham.nmode_total, nsta))

    for m in ham.mrange:
        for n in range(m+1):
            adiabderiv2[m,n] = np.dot(np.dot(datmat.T, diabderiv2[m,n]),
                                             datmat).diagonal()
            if m != n:
                adiabderiv2[n,m] = adiabderiv2[m,n]

    return adiabderiv2

#
def calc_scts(adiabpot, datmat, diabderiv1, nactmat, adiabderiv1, diablap):
    """Calculates the scalar coupling terms.

    Uses the equation:
    G = (d/d_X F) - F.F

    d/d_X F_ij = d/dX (xi_ij*T_ij), where, xi_ij = (V_j-V_i)^-1
                                           T = S {d/dX W} S^T

    Additionally, the gradients of the on-diagonal SCTs (DBOCs)
    are calculated

    In the following:

    ximat      <-> xi
    tmat       <-> T
    deltmat    <-> d/dX T
    delximat   <-> d/dX xi
    delnactmat <-> d/dX F
    fdotf      <-> F.F
    """
    #-------------------------------------------------------------------
    # (1) Construct d/dX F (delnactmat)
    #-------------------------------------------------------------------

    # (a) deltmat = S {Del^2 W} S^T + -FS{d/dX W}S^T + S{d/dX W}S^TF
    # tmp1 <-> S {Del^2 W} S^T
    tmp1 = np.dot(np.dot(datmat.T, diablap), datmat)

    # tmp2 <-> -F S{d/dX W}S^T
    mat2 = [np.dot(np.dot(np.dot(nactmat[m], datmat.T), diabderiv1[m]), datmat)
            for m in ham.mrange]
    tmp2 = -np.sum(ham.freq[:,np.newaxis,np.newaxis] * mat2, axis=0)

    # tmp3 <-> S{d/dX W}S^T F
    mat3 = [np.dot(np.dot(np.dot(datmat.T, diabderiv1[m]), datmat), nactmat[m])
            for m in ham.mrange]
    tmp3 = np.sum(ham.freq[:,np.newaxis,np.newaxis] * mat3, axis=0)

    # deltmat
    deltmat = tmp1 + tmp2 + tmp3

    # (b) delximat
    ## This is slightly slower for nsta == 2
    #delximat = np.zeros((ham.nmode_total, nsta, nsta))
    #for m in ham.mrange:
    #    delximat[m] = -(np.subtract.outer(adiabderiv1[m], adiabderiv1[m]) /
    #                    (np.subtract.outer(adiabpot, adiabpot) ** 2 +
    #                     np.eye(nsta)))
    delximat = np.zeros((ham.nmode_total, nsta, nsta))
    for m in ham.mrange:
        for i in range(nsta):
            for j in range(nsta):
                if i != j:
                    delximat[m,i,j] = ((adiabderiv1[m,i] - adiabderiv1[m,j]) /
                                       (adiabpot[j] - adiabpot[i])**2)

    # (c) tmat
    tmat = np.zeros((ham.nmode_total, nsta, nsta))
    for m in ham.mrange:
        tmat[m] = np.dot(np.dot(datmat.T, diabderiv1[m]), datmat)

    # (d) ximat
    ## This is slightly slower for nsta == 2
    #ximat = 1. / (-np.subtract.outer(adiabpot, adiabpot) +
    #              np.eye(nsta)) - np.eye(nsta)
    ximat = np.zeros((nsta, nsta))
    for i in range(nsta):
        for j in range(nsta):
            if i != j:
                ximat[i,j] = 1. / (adiabpot[j] - adiabpot[i])

    # (f) delnactmat_ij = delximat_ij*tmat_ij + ximat_ij*deltmat_ij (i.ne.j)
    delnactmat = np.sum(delximat * tmat, axis=0) + ximat * deltmat

    #-------------------------------------------------------------------
    # (2) Construct F.F (fdotf)
    #-------------------------------------------------------------------
    matf = [np.dot(nactmat[m], nactmat[m]) for m in ham.mrange]
    fdotf = np.sum(ham.freq[:,np.newaxis,np.newaxis] * matf, axis=0)

    #-------------------------------------------------------------------
    # (3) Calculate the scalar coupling terms G = (d/dX F) - F.F
    #-------------------------------------------------------------------
    sctmat = delnactmat - fdotf

    #-------------------------------------------------------------------
    # (4) Calculation of the 1st derivatives of the DBOCs
    #-------------------------------------------------------------------
    dbocderiv1 = np.zeros((ham.nmode_total, nsta))
    for m in ham.mrange:
        dbocderiv1[m] = -2.*np.sum(delnactmat * nactmat[m], axis=0)

    return sctmat, dbocderiv1
