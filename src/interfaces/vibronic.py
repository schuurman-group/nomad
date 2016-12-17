"""
Routines for running a vibronic coupling calculation.

Much of this could benefit from changing for loops to numpy array operations.
(But this is so computationally cheap that it really doesn't matter...)
"""
import sys
import copy
import numpy as np
import src.interfaces.vcham.hampar as ham
import src.interfaces.vcham.rdoper as rdoper
import src.interfaces.vcham.rdfreq as rdfreq
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio

# KE operator coefficients a_i:
# T = sum_i a_i p_i^2,
# where p_i is the momentum operator
kecoeff = None

diabpot = None
adiabpot = None
adtmat = None
diabderiv1 = None
nactmat = None
adiabderiv1 = None
diablap = None
sctmat = None
dbocderiv1 = None
nsta = 0

class surface_data:
    def __init__(self, n_states, t_dim, crd_dim):

        # necessary for array allocation
        self.n_states     = n_states
        self.t_dim        = t_dim
        self.crd_dim      = crd_dim

        # these are the standard quantities ALL interface_data objects return
        self.data_keys    = []
        self.geom         = np.zeros(t_dim)
        self.energies     = np.zeros(n_states)
        self.grads        = np.zeros((n_states, t_dim)) 
    
        # these are interface-specific quantities
        self.scalar_coup  = np.zeros(n_states)
        self.adt_mat      = np.zeros((n_states, n_states))    
        self.dat_mat      = np.zeros((n_states, n_states))
        self.ddat_mat     = np.zeros((t_dim,n_states, n_states))
        self.diabat_pot   = np.zeros((n_states, n_states))
        self.diabat_deriv = np.zeros((t_dim, n_states, n_states))
        self.adiabat_pot   = np.zeros((n_states, n_states))
        self.adiabat_deriv = np.zeros((t_dim, n_states, n_states))

# 
def copy_data(orig_info):
    new_info = surface_data(orig_info.n_states,
                            orig_info.t_dim,
                            orig_info.crd_dim)

    new_info.data_keys    = copy.copy(orig_info.data_keys)
    new_info.geom         = copy.deepcopy(orig_info.geom)
    new_info.energies     = copy.deepcopy(orig_info.energies)
    new_info.grads        = copy.deepcopy(orig_info.grads)
    new_info.scalar_coup  = copy.deepcopy(orig_info.scalar_coup)
    new_info.adt_mat      = copy.deepcopy(orig_info.adt_mat)
    new_info.dat_mat      = copy.deepcopy(orig_info.dat_mat)
    new_info.ddat_mat      = copy.deepcopy(orig_info.ddat_mat)
    new_info.diabat_pot   = copy.deepcopy(orig_info.diabat_pot)
    new_info.diabat_deriv = copy.deepcopy(orig_info.diabat_deriv)
    new_info.adiabat_pot   = copy.deepcopy(orig_info.adiabat_pot)
    new_info.adiabat_deriv = copy.deepcopy(orig_info.adiabat_deriv)
    return new_info 

#
def init_interface():
    """Read the freq.dat file

    Note that, at least for now, the no. active modes and their
    labels are determined from the freq.dat file.
    As such, we must read the freq.dat file BEFORE reading the
    operator file.
    """

    global kecoeff

    rdfreq.rdfreqfile()

    # Open the operator file
    opfile = open(glbl.fms['opfile'], 'r')

    # Read the operator file
    rdoper.rdoperfile(opfile)

    # Close the operator file
    opfile.close()

    # KE operator coefficients, mass- and frequency-scaled normal mode
    # coordinates, a_i = 0.5*omega_i
    kecoeff = np.zeros((ham.nmode_active))
#    kecoeff = 0.5*ham.freq[:ham.nmode_active]
    kecoeff = 0.5*np.ones((ham.nmode_active)) 
   
    # Ouput some information about the Hamiltonian
    fileio.print_fms_logfile('string', ['*'*72])
    fileio.print_fms_logfile('string',
                             ['* Vibronic Coupling Hamiltonian Information'])
    fileio.print_fms_logfile('string', ['*'*72])
    fileio.print_fms_logfile('string',
                             ['Operator file: ' + glbl.fms['opfile']])
    fileio.print_fms_logfile('string',
                             ['Number of Hamiltonian terms: ' + str(ham.nterms)])
    string = 'Total no. modes: ' + str(ham.nmode_total)
    fileio.print_fms_logfile('string', [string])

    string = 'No. active modes: ' + str(ham.nmode_active)
    fileio.print_fms_logfile('string', [string])

    fileio.print_fms_logfile('string', ['Active mode labels:'])
    for i in range(ham.nmode_active):
        string = str(i+1) + ' ' + ham.mlbl_active[i]
        fileio.print_fms_logfile('string', [string])

    fileio.print_fms_logfile('string', ['Active mode frequencies (a.u.):'])
    for i in range(ham.nmode_active):
        string = str(i+1) + ' ' + str(ham.freq[i])
        fileio.print_fms_logfile('string', [string])


def evaluate_trajectory(tid, geom, stateindx):
    """Evaluates the trajectory."""
    global diabpot, adiabpot, adtmat, diabderiv1, nactmat, adiabderiv1
    global diablap, sctmat, dbocderiv1, nsta

    # System dimensions
    ncoo   = len(geom)
    nsta   = glbl.fms['n_states']
    t_data = surface_data(nsta,ncoo,1)

    # Initialisation of arrays
    diabpot = np.zeros((nsta, nsta))
    ener = np.zeros(nsta)
    grad = np.zeros((nsta, ncoo))
    diabderiv1 = np.zeros((ncoo, nsta, nsta))
    nactmat = np.zeros((ncoo, nsta, nsta))
    adiabderiv1 = np.zeros((ncoo, nsta))
    diablap = np.zeros((nsta, nsta))
    sctmat = np.zeros((nsta, nsta))
    dbocderiv1 = np.zeros((ncoo, nsta))
    sct_return = np.zeros((nsta))

    # Set the current normal mode coordinates
    qcoo = geom

    # Calculation of the diabatic potential matrix
    calc_diabpot(qcoo)

    # Calculation of the adiabatic potential vector and ADT matrix
    calc_adt()

    # Calculation of the nuclear derivatives of the diabatic potential
    calc_diabderiv1(qcoo)

    # Calculation of the NACT matrix
    calc_nacts()

    # Calculation of the gradients of the adiabatic potential
    calc_adiabderiv1()

    # Calculation of the Laplacian of the diabatic potential wrt the
    # nuclear DOFs
    calc_diablap(qcoo)

    # Calculation of the scalar couplings terms (SCTs)
    #
    # Note that in order to calculate the SCTs, the gradients of the
    # diagonal Born-Oppenheimer corrections (DBOCs). Consequently, we
    # save these as a matter of course.
    calc_scts()

    # Package up the energies, gradients and NACTs
    ener = adiabpot

    for i in range(nsta):
        for m in range(ncoo):
            if i == stateindx:
                grad[i,m] = adiabderiv1[m,i]
            else:
                grad[i,m]=nactmat[m,stateindx,i]

    # Package the SCTs: here we account for the 1/2 prefactor
    # in the EOMs
    for i in range(nsta):
        sct_return[i]=0.5*sctmat[stateindx,i]

#    print("adt="+str(adtmat))
#    print("dat="+str(np.linalg.inv(adtmat)))
    de    = diabpot[1,1]-diabpot[0,0]
    v12   = diabpot[0,1]
    argt  = 2*v12/de
    theta = 0.5*np.arctan(argt)
#    print("dat2="+str([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]))
    dderiv = np.array([(diabderiv1[q,0,1]/de - v12*(diabderiv1[q,1,1]- diabderiv1[q,0,0])/de**2)/(1+argt**2) for q in range(len(qcoo))])
    ddat2  = np.array([[[-np.sin(theta)*dderiv[i],np.cos(theta)*dderiv[i]],[-np.cos(theta)*dderiv[i],np.sin(theta)*dderiv[i]]] for i in range(len(qcoo))])
#    print("ddat2="+str(ddat2)) 

    t_data.geom          = qcoo
    t_data.energies      = ener
    t_data.grads         = grad
    t_data.scalar_coup   = sct_return
    t_data.adt_mat       = adtmat
    t_data.dat_mat       = np.linalg.inv(adtmat)
    t_data.ddat_mat      = ddat2
    t_data.diabat_pot    = diabpot
    t_data.diabat_deriv  = diabderiv1
    t_data.adiabat_pot   = adiabpot
    t_data.adiabat_deriv = adiabderiv1
    t_data.data_keys    = ['geom','poten','deriv',
                           'scalar_coup','adt_mat','dat_mat','ddat_mat',
                           'diabat_pot','diabat_deriv',
                           'adiabat_pot','adiabat_deriv']
    return t_data

def evaluate_centroid(tid, geom, stateindx, stateindx2):
    """Evaluates the centroid.

    Note that because energies, gradients and couplings are so cheap
    to extract from a vibronic coupling Hamiltonian, we return
    all quantities, even if they are not actually needed.
    """
    global diabpot, adiabpot, adtmat, diabderiv1, nactmat, adiabderiv1
    global diablap, sctmat, dbocderiv1, nsta

    # System dimensions
    ncoo = len(geom) 
    nsta = glbl.fms['n_states']
    t_data = surface_data(nsta,ncoo,1)

    # Initialisation of arrays
    diabpot=np.zeros((nsta,nsta), dtype=np.float)
    ener=np.zeros((nsta), dtype=np.float)
    grad=np.zeros((nsta,ncoo), dtype=np.float)
    diabderiv1=np.zeros((ncoo,nsta,nsta), dtype=np.float)
    nactmat=np.zeros((ncoo,nsta,nsta), dtype=np.float)
    adiabderiv1=np.zeros((ncoo,nsta), dtype=np.float)
    diablap=np.zeros((nsta,nsta), dtype=np.float)
    sctmat=np.zeros((nsta,nsta), dtype=np.float)
    dbocderiv1=np.zeros((ncoo,nsta), dtype=np.float)
    sct_return=np.zeros((nsta), dtype=np.float)
    diabpot = np.zeros((nsta, nsta))
    ener = np.zeros(nsta)
    grad = np.zeros((nsta, ncoo))
    diabderiv1 = np.zeros((ncoo, nsta, nsta))
    nactmat = np.zeros((ncoo, nsta, nsta))
    adiabderiv1 = np.zeros((ncoo, nsta))
    diablap = np.zeros((nsta, nsta))
    sctmat = np.zeros((nsta, nsta))
    dbocderiv1 = np.zeros((ncoo, nsta))
    sct_return = np.zeros((nsta))

    # Set the current normal mode coordinates
    qcoo = geom 

    # Calculation of the diabatic potential matrix
    calc_diabpot(qcoo)

    # Calculation of the adiabatic potential vector and ADT matrix
    calc_adt()

    # Calculation of the nuclear derivatives of the diabatic potential
    calc_diabderiv1(qcoo)

    # Calculation of the NACT matrix
    calc_nacts()

    # Calculation of the gradients of the adiabatic potential
    calc_adiabderiv1()

    # Calculation of the Laplacian of the diabatic potential wrt the
    # nuclear DOFs
    calc_diablap(qcoo)

    # Calculation of the scalar couplings terms (SCTs)
    #
    # Note that in order to calculate the SCTs, the gradients of the
    # diagonal Born-Oppenheimer corrections (DBOCs). Consequently, we
    # save these as a matter of course.
    calc_scts()

    # Package up the energies, gradients and NACTs
    # N.B. we need to include here the option to send back either
    # the adiabatic or diabatic quantities...
    ener = adiabpot

    for i in range(nsta):
        for m in range(ncoo):
            if i == stateindx:
                grad[i,m] = adiabderiv1[m,i]
            else:
                grad[i,m]=nactmat[m,stateindx,i]

    # Package the SCTs: here we account for the 1/2 prefactor
    # in the EOMs
    for i in range(nsta):
        sct_return[i]=0.5*sctmat[stateindx,i]

    t_data.geom         = qcoo
    t_data.energies     = ener
    t_data.grads        = grad
    t_data.scalar_coup  = sct_return
    t_data.adt_mat      = adtmat
    t_data.dat_mat      = np.linalg.inv(adtmat)
    t_data.diabat_pot   = diabpot
    t_data.diabat_deriv = diabderiv1
    t_data.adiabat_pot   = adiabpot
    t_data.adiabat_deriv = adiabderiv1
    t_data.data_keys    = ['geom','poten','deriv',
                           'scalar_coup','adt_mat','dat_mat',
                           'diabat_pot','diabat_deriv',
                           'adiabat_pot','adiabat_deriv']
    return t_data

#--------------------------------------------------------------------
#*****PRIVATE FUNCTIONS (should not be called outside interface)*****
#--------------------------------------------------------------------

def calc_diabpot(q):
    """Constructs the diabatic potential matrix for a given nuclear
    geometry q."""
    global diabpot, nsta

    #-------------------------------------------------------------------
    # Build the diabatic potential
    #
    # N.B. the terms held in the coe, ord and stalbl contribute only
    # to the lower-triangle of the diabatic potential matrix
    #-------------------------------------------------------------------
    # Fill in the lower-triangle
    for i in range(ham.nterms):
        s1 = ham.stalbl[i,0] - 1
        s2 = ham.stalbl[i,1] - 1
        fac = ham.coe[i]
        for m in range(ham.nmode_active):
            fac = fac * q[m]**ham.order[i,m]

        diabpot[s1,s2] = diabpot[s1,s2] + fac

    # Fill in the upper-triangle
    for s1 in range(nsta-1):
        for s2 in range(s1+1, nsta):
            diabpot[s2,s1] = diabpot[s1,s2]


def calc_adt():
    """Diagonalises the diabatic potential matrix to yield the adiabatic
    potentials and the adiabatic-to-diabatic transformation matrix."""
    global adiabpot, adtmat

    adiabpot, adtmat = np.linalg.eigh(diabpot)
    
    # set phase convention that largest element in adt column vector is 
    # positive
    # array index of maximum values for each column
    mxvals = np.argmax(np.abs(adtmat),axis=0)
    for i in range(len(mxvals)):
        if adtmat[mxvals[i],i] < 0:
            adtmat[:,i] *= -1.

def calc_diabderiv1(q):
    """Calculates the 1st derivatives of the elements of the diabatic
    potential matrix wrt the nuclear DOFs."""
    global diabderiv1, nsta

    #-------------------------------------------------------------------
    # Build the tensor of nuclear derviatives of the diabatic potential
    #-------------------------------------------------------------------
    # Fill in the lower-triangle
    for i in range(ham.nterms):
        s1 = ham.stalbl[i,0] - 1
        s2 = ham.stalbl[i,1] - 1
        for m in range(ham.nmode_active):
            fac = 0.0
            if ham.order[i,m] != 0:
                fac = ham.coe[i]
                for n in range(ham.nmode_active):
                    p = ham.order[i,n]
                    if n == m:
                        fac *= p * q[n]**(p-1)
                    else:
                        fac *= q[n]**p

            diabderiv1[m,s1,s2] += fac

    # Fill in the upper-triangle
    for m in range(ham.nmode_active):
        for s1 in range(nsta-1):
            for s2 in range(s1+1, nsta):
                diabderiv1[m,s2,s1] = diabderiv1[m,s1,s2]


def calc_nacts():
    """Calculates the matrix of non-adiabatic coupling terms from the
    adiabatic potentials, the ADT matrix and the nuclear derivatives of
    the diabatic potentials.

    Equation used: F_aij = [S (d/dX_a W) S^T]_ij / (V_j - V_i)
    F_aij = < psi_i | d/dX_a psi_j >: non-adiabatic coupling terms
    W: diabatic potential matrix
    S^T: matrix of eigenvectors of W
    V: vector of adiabatic energies
    """
    global adtmat, diabderiv1, nactmat, nsta

    #-------------------------------------------------------------------
    # (1) Calculation of S{d/dX W}S^T (tmpmat)
    #-------------------------------------------------------------------
    tmpmat = np.zeros((ham.nmode_active, nsta, nsta))
    for m in range(ham.nmode_active):
        for i in range(nsta):
            for j in range(nsta):
                for k in range(nsta):
                    for l in range(nsta):
                        tmpmat[m,i,j] += (adtmat[k,i] *
                                            diabderiv1[m,k,l] * adtmat[l,j])

    #-------------------------------------------------------------------
    # (2) Calculation of the non-adiabatic coupling terms
    #-------------------------------------------------------------------
    # Fill in the lower-triangle minus the on-diagonal terms (which
    # are zero by symmetry)
    for m in range(ham.nmode_active):
        for i in range(nsta-1):
            for j in range(i+1, nsta):
                nactmat[m,i,j] = tmpmat[m,i,j] / (adiabpot[j]-adiabpot[i])

    # Fill in the upper-triangle using the anti-symmetry of the NACTs
    for m in range(ham.nmode_active):
        for i in range(nsta-1):
            for j in range(i+1, nsta):
                nactmat[m,j,i] = -nactmat[m,i,j]


def calc_adiabderiv1():
    """Calculates the gradients of the adiabatic potentials."""
    global adiabderiv1, adtmat, nsta

    #-------------------------------------------------------------------
    # d/dX V_ii = (S{d/dX W}S^T)_ii
    #-------------------------------------------------------------------
    for m in range(ham.nmode_active):
        for i in range(nsta):
            for k in range(nsta):
                for l in range(nsta):
                    adiabderiv1[m,i] += (adtmat[k,i] *
                                          diabderiv1[m,k,l] * adtmat[l,i])


def calc_diablap(q):
    """Calculates the Laplacian of the diabatic potential matrix wrt
    the nuclear DOFs at the point q."""
    global diablap, nsta

    # Initialise arrays
    der2 = np.zeros((ham.nmode_active, nsta, nsta))

    #-----------------------------------------------------------------------
    # Build the Laplacian of the diabatic potential
    #-----------------------------------------------------------------------
    # Fill in the lower-triangle
    for i in range(ham.nterms):
        s1 = ham.stalbl[i,0] - 1
        s2 = ham.stalbl[i,1] - 1
        for m in range(ham.nmode_active):
            fac = 0.0
            if ham.order[i,m] > 1:
                fac = ham.coe[i]
                for n in range(ham.nmode_active):
                    p = ham.order[i,n]
                    if n == m:
                        fac *= p * (p-1) * q[n]**(p-2)
                    else:
                        fac *= q[n]**p
            der2[m,s1,s2] += fac

    for i in range(nsta):
        for j in range(nsta):
            for m in range(ham.nmode_active):
                diablap[i,j] += der2[m,i,j]

    # Fill in the upper-triangle
    for i in range(nsta-1):
        for j in range(i+1, nsta):
            diablap[j,i] = diablap[i,j]


def calc_scts():
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
    global nsta, sctmat, dbocderiv1, adtmat, diablap, nactmat

    #-------------------------------------------------------------------
    # Initialise arrays
    #-------------------------------------------------------------------
    tmp1 = np.zeros((nsta, nsta))
    tmp2 = np.zeros((nsta, nsta))
    tmp3 = np.zeros((nsta, nsta))
    ximat = np.zeros((nsta, nsta))
    deltmat = np.zeros((nsta, nsta))
    delnactmat = np.zeros((nsta, nsta))
    fdotf = np.zeros((nsta, nsta))
    tmat = np.zeros((ham.nmode_active, nsta, nsta))
    delximat = np.zeros((ham.nmode_active, nsta, nsta))

    #-------------------------------------------------------------------
    # (1) Construct d/dX F (delnactmat)
    #-------------------------------------------------------------------

    # (a) deltmat = S {Del^2 W} S^T + -FS{d/dX W}S^T + S{d/dX W}S^TF

    # tmp1 <-> S {Del^2 W} S^T
    for i in range(nsta):
        for j in range(nsta):
            for k in range(nsta):
                for l in range(nsta):
                    tmp1[i,j] += adtmat[k,i] * diablap[k,l] * adtmat[l,j]

    # tmp2 <-> -FS{d/dX W}S^T
    for i in range(nsta):
        for j in range(nsta):
            for k in range(nsta):
                for l in range(nsta):
                    for m in range(nsta):
                        dp = 0.0
                        for n in range(ham.nmode_active):
                            dp+=ham.freq[n]*nactmat[n,i,k]*diabderiv1[n,l,m]
                        tmp2[i,j]-=adtmat[l,k]*adtmat[m,j]*dp

    # tmp3 <-> S{d/dX W}S^TF
    for i in range(nsta):
        for j in range(nsta):
            for k in range(nsta):
                for l in range(nsta):
                    for m in range(nsta):
                        dp = 0.0
                        for n in range(ham.nmode_active):
                            dp+=ham.freq[n]*nactmat[n,m,j]*diabderiv1[n,k,l]
                        tmp3[i,j]+=adtmat[k,i]*adtmat[l,m]*dp

    # deltmat
    for i in range(nsta):
        for j in range(nsta):
            deltmat[i,j] = tmp1[i,j] + tmp2[i,j] + tmp3[i,j]


    # (b) delximat
    for m in range(ham.nmode_active):
        for i in range(nsta):
            for j in range(nsta):
                if i != j:
                    delximat[m,i,j] = -(adiabderiv1[m,j] - adiabderiv1[m,i])
                    delximat[m,i,j] /= (adiabpot[j] - adiabpot[i])**2

    # (c) tmat
    for m in range(ham.nmode_active):
        for i in range(nsta):
            for j in range(nsta):
                for k in range(nsta):
                    for l in range(nsta):
                        tmat[m,i,j] += (adtmat[k,i] *
                                          diabderiv1[m,k,l] * adtmat[l,j])

    # (d) ximat
    for i in range(nsta):
        for j in range(nsta):
            if i != j:
                ximat[i,j] = 1. / (adiabpot[j] - adiabpot[i])

    # (f) delnactmat_ij = delximat_ij*tmat_ij + ximat_ij*deltmat_ij (i.ne.j)
    # tmp1 = delximat_ij*tmat_ij
    tmp1 = np.zeros((nsta, nsta))
    for i in range(nsta):
        for j in range(nsta):
            if i != j:
                for m in range(ham.nmode_active):
                    tmp1[i,j] += delximat[m,i,j] * tmat[m,i,j]

    # tmp2 = ximat_ij*deltmat_ij
    tmp2 = np.zeros((nsta, nsta))
    for i in range(nsta):
        for j in range(nsta):
            if i != j:
                tmp2[i,j] = ximat[i,j] * deltmat[i,j]

    # delnactmat
    for i in range(nsta):
        for j in range(nsta):
            delnactmat[i,j] = tmp1[i,j] + tmp2[i,j]

    #-------------------------------------------------------------------
    # (2) Construct F.F (fdotf)
    #-------------------------------------------------------------------
    for i in range(nsta):
        for j in range(nsta):
            for k in range(nsta):
                for m in range(ham.nmode_active):
                    fdotf[i,j]+=ham.freq[m]*nactmat[m,i,k]*nactmat[m,k,j]

    #-------------------------------------------------------------------
    # (3) Calculate the scalar coupling terms G = (d/dX F) - F.F
    #-------------------------------------------------------------------
    for i in range(nsta):
        for j in range(nsta):
            sctmat[i,j] = delnactmat[i,j] - fdotf[i,j]

    #-------------------------------------------------------------------
    # (4) Calculation of the 1st derivatives of the DBOCs
    #
    # CHECK THIS CAREFULLY!
    #-------------------------------------------------------------------
    for i in range(nsta):
        for m in range(ham.nmode_active):
            for k in range(nsta):
                dbocderiv1[m,i] -= 2. * delnactmat[i,k] * nactmat[m,i,k]

