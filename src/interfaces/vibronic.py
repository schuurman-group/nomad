import sys
import numpy            as np
import src.vcham.hampar as ham
import src.vcham.rdoper as rdoper
import src.vcham.rdfreq as rdfreq
import src.vcham.rdfreq as rdfreq
import src.fmsio.glbl   as glbl
import src.fmsio.fileio as fileio

diabpot=None
adiabpot=None
adtmat=None
diabderiv1=None
nactmat=None
adiabderiv1=None
diablap=None
sctmat=None
dbocderiv1=None
nsta=0

########################################################################

def init_interface():

    # Read the freq.dat file
    #
    # Note that, at least for now, the no. active modes and their
    # labels are determined from the freq.dat file.
    # As such, we must read the freq.dat file BEFORE reading the
    # operator file.
    rdfreq.rdfreqfile()

    # Open the operator file
    opfile=open(glbl.fms['opfile'],'r')

    # Read the operator file
    rdoper.rdoperfile(opfile)

    # Close the operator file
    opfile.close()    

    # Ouput some information about the Hamiltonian    
    fileio.print_fms_logfile('string',['*'*72])

    fileio.print_fms_logfile('string',['* Vibronic Coupling Hamiltonian Information'])

    fileio.print_fms_logfile('string',['*'*72])

    fileio.print_fms_logfile('string',['Operator file: '+glbl.fms['opfile']])

    fileio.print_fms_logfile('string',\
                             ['Number of Hamiltonian terms: '+str(ham.nterms)])
    
    string='Total no. modes: '+str(ham.nmode_total)
    fileio.print_fms_logfile('string',[string])

    string='No. active modes: '+str(ham.nmode_active)
    fileio.print_fms_logfile('string',[string])
    
    fileio.print_fms_logfile('string',['Active mode labels:'])
    for i in range(ham.nmode_active):
        string=str(i+1)+' '+ham.mlbl_active[i]
        fileio.print_fms_logfile('string',[string])

    fileio.print_fms_logfile('string',['Active mode frequencies (a.u.):'])
    for i in range(ham.nmode_active):
        string=str(i+1)+' '+str(ham.freq[i])
        fileio.print_fms_logfile('string',[string])

    return

########################################################################

def evaluate_trajectory(tid, geom, stateindx):

    global diabpot
    global adiabpot
    global adtmat
    global diabderiv1
    global nactmat
    global adiabderiv1
    global diablap
    global sctmat
    global dbocderiv1
    global nsta

    # System dimensions
    ncoo=glbl.fms['num_particles']
    nsta=glbl.fms['n_states']

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

    # Set the current normal mode coordinates
    qcoo=np.zeros(ncoo)
    for i in range(ncoo):
        qcoo[i]=geom[i].x
    
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

    ener=adiabpot

    for i in range(nsta):
        for m in range(ncoo):
            if i==(stateindx):
                grad[i][m]=adiabderiv1[m][i]
            else:
                grad[i][m]=nactmat[m][stateindx][i]

    return[qcoo,ener,grad]

########################################################################

def evaluate_centroid(tid, geom, stateindx, stateindx2):

    # Note that because energies, gradients and couplings are so cheap 
    # to extract from a vibronic coupling Hamiltonian, we return
    # all quantities, even if they are not actually needed

    global diabpot
    global adiabpot
    global adtmat
    global diabderiv1
    global nactmat
    global adiabderiv1
    global diablap
    global sctmat
    global dbocderiv1
    global nsta

    # System dimensions
    ncoo=glbl.fms['num_particles']
    nsta=glbl.fms['n_states']

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

    # Set the current normal mode coordinates
    qcoo=np.zeros(ncoo)
    for i in range(ncoo):
        qcoo[i]=geom[i].x
    
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

    ener=adiabpot

    for i in range(nsta):
        for m in range(ncoo):
            if i==(stateindx):
                grad[i][m]=adiabderiv1[m][i]
            else:
                grad[i][m]=nactmat[m][stateindx][i]

    return[qcoo,ener,grad]

########################################################################
# calc_diabpot: constructs the diabatic potential matrix for a given
#               nuclear geometry q
########################################################################

def calc_diabpot(q):

    global diabpot
    global nsta
    
    #-------------------------------------------------------------------
    # Build the diabatic potential
    #
    # N.B. the terms held in the coe, ord and stalbl contribute only
    # to the lower-triangle of the diabatic potential matrix
    #-------------------------------------------------------------------
    # Fill in the lower-triangle
    for i in range(ham.nterms):
        s1=ham.stalbl[i][0]-1
        s2=ham.stalbl[i][1]-1
        fac=ham.coe[i]
        for m in range(ham.nmode_active):
            fac=fac*q[m]**ham.order[i][m]            

        diabpot[s1][s2]=diabpot[s1][s2]+fac

    # Fill in the upper-triangle    
    for s1 in range(nsta-1):
        for s2 in range(s1+1,nsta):
            diabpot[s2][s1]=diabpot[s1][s2]

    return

########################################################################
# calc_adt: diagonalises the diabatic potential matrix to yield the
#           adiabatic potentials and the adiabatic-to-diabatic
#           transformation matrix
########################################################################

def calc_adt():

    global adiabpot
    global adtmat

    adiabpot, adtmat = np.linalg.eigh(diabpot)

    return

########################################################################
# calc_diabderiv1: calculates the 1st derivatives of the elements of the
#                  diabatic potential matrix wrt the nuclear DOFs
########################################################################

def calc_diabderiv1(q):

    global diabderiv1
    global nsta

    #-------------------------------------------------------------------
    # Build the tensor of nuclear derviatives of the diabatic potential
    #-------------------------------------------------------------------
    # Fill in the lower-triangle
    for i in range(ham.nterms):
        s1=ham.stalbl[i][0]-1
        s2=ham.stalbl[i][1]-1
        for m in range(ham.nmode_active):
            fac=0.0
            if (ham.order[i][m]!=0):
                fac=ham.coe[i]
                for n in range(ham.nmode_active):
                    p=ham.order[i][n]
                    if n==m:
                        fac=fac*p*q[n]**(p-1)
                    else:
                        fac=fac*q[n]**p
                    
            diabderiv1[m][s1][s2]+=fac
    
    # Fill in the upper-triangle
    for m in range(ham.nmode_active):
        for s1 in range(nsta-1):
            for s2 in range(s1+1,nsta):
                diabderiv1[m][s2][s1]=diabderiv1[m][s1][s2]

    return

########################################################################
# calc_nacts: calculates the matrix of non-adiabatic coupling terms from
#             the adiabatic potentials, the ADT matrix and the nuclear
#             derivatives of the diabatic potentials
#
# Equation used: F_aij = [S (d/dX_a W) S^T]_ij / (V_j - V_i)
#
# F_aij = < psi_i | d/dX_a psi_j >: non-adiabatic coupling terms
#
# W: diabatic potential matrix
#
# S^T: matrix of eigenvectors of W
#
# V: vector of adiabatic energies
########################################################################

def calc_nacts():

    global adtmat
    global diabderiv1
    global nactmat
    global nsta

    #-------------------------------------------------------------------
    # (1) Calculation of S{d/dX W}S^T (tmpmat)
    #-------------------------------------------------------------------
    tmpmat=np.zeros((ham.nmode_active,nsta,nsta), dtype=np.float)
    for m in range(ham.nmode_active):
        for i in range(nsta):
            for j in range(nsta):
                for k in range(nsta):
                    for l in range(nsta):
                        tmpmat[m][i][j]+=adtmat[k][i]*diabderiv1[m][k][l]*adtmat[l][j]

    #-------------------------------------------------------------------
    # (2) Calculation of the non-adiabatic coupling terms
    #-------------------------------------------------------------------
    # Fill in the lower-triangle minus the on-diagonal terms (which
    # are zero by symmetry)
    for m in range(ham.nmode_active):
        for i in range(nsta-1):
            for j in range(i+1,nsta):
                nactmat[m][i][j]=tmpmat[m][i][j]/(adiabpot[j]-adiabpot[i])
   
    # Fill in the upper-triangle using the anti-symmetry of the NACTs
    for m in range(ham.nmode_active):
        for i in range(nsta-1):
            for j in range(i+1,nsta):
                nactmat[m][j][i]=-nactmat[m][i][j]

    return

########################################################################
# calc_adiabderiv1: calculates the gradients of the adiabatic potentials
########################################################################

def calc_adiabderiv1():

    global adiabderiv1
    global adtmat
    global nsta

    #-------------------------------------------------------------------
    # d/dX V_ii = (S{d/dX W}S^T)_ii
    #-------------------------------------------------------------------
    for m in range(ham.nmode_active):
        for i in range(nsta):
            for k in range(nsta):
                for l in range(nsta):
                    adiabderiv1[m][i]+=adtmat[k][i]*diabderiv1[m][k][l]*adtmat[l][i]

    return

########################################################################
# calc_diablap: calculates the Laplacian of the diabatic potential
#               matrix wrt the nuclear DOFs at the point q
########################################################################

def calc_diablap(q):

    global diablap
    global nsta

    #-------------------------------------------------------------------
    # Initialise arrays
    #-------------------------------------------------------------------
    der2=np.zeros((ham.nmode_active,nsta,nsta), dtype=np.float)

    #-----------------------------------------------------------------------
    # Build the Laplacian of the diabatic potential
    #-----------------------------------------------------------------------
    # Fill in the lower-triangle
    for i in range(ham.nterms):
        s1=ham.stalbl[i][0]-1
        s2=ham.stalbl[i][1]-1
        for m in range(ham.nmode_active):
            fac=0.0
            if ham.order[i][m]>1:
                fac=ham.coe[i]
                for n in range(ham.nmode_active):
                    p=ham.order[i][n]
                    if n==m:
                        fac=fac*p*(p-1)*q[n]**(p-2)
                    else:
                        fac=fac*q[n]**p
            der2[m][s1][s2]+=fac

    for i in range(nsta):
        for j in range(nsta):
            for m in range(ham.nmode_active):
                diablap[i][j]+=der2[m][i][j]

    # Fill in the upper-triangle
    for i in range(nsta-1):
        for j in range(i+1,nsta):
            diablap[j][i]=diablap[i][j]

    return

########################################################################
# calc_scts: calculates the scalar coupling terms using the equation
#
#            G = (d/d_X F) - F.F
#
#            d/d_X F_ij = d/dX (xi_ij*T_ij), where, xi_ij = (V_j-V_i)^-1
#                                                   T = S {d/dX W} S^T
#
#            Additionally, the gradients of the on-diagonal SCTs (DBOCs)
#            are calculated
#
# In the following:
#
# ximat      <-> xi
# tmat       <-> T
# deltmat    <-> d/dX T
# delximat   <-> d/dX xi
# delnactmat <-> d/dX F
# fdotf      <-> F.F
########################################################################

def calc_scts():
    
    global nsta
    global sctmat
    global dbocderiv1
    global adtmat
    global diablap
    global nactmat

    #-------------------------------------------------------------------
    # Initialise arrays
    #-------------------------------------------------------------------
    tmp1=np.zeros((nsta,nsta), dtype=np.float)
    tmp2=np.zeros((nsta,nsta), dtype=np.float)
    tmp3=np.zeros((nsta,nsta), dtype=np.float)
    ximat=np.zeros((nsta,nsta), dtype=np.float)
    deltmat=np.zeros((nsta,nsta), dtype=np.float)
    delnactmat=np.zeros((nsta,nsta), dtype=np.float)
    fdotf=np.zeros((nsta,nsta), dtype=np.float)
    tmat=np.zeros((ham.nmode_active,nsta,nsta), dtype=np.float)
    delximat=np.zeros((ham.nmode_active,nsta,nsta), dtype=np.float)

    #-------------------------------------------------------------------
    # (1) Construct d/dX F (delnactmat)
    #-------------------------------------------------------------------

    # (a) deltmat = S {Del^2 W} S^T + -FS{d/dX W}S^T + S{d/dX W}S^TF
      
    # tmp1 <-> S {Del^2 W} S^T
    for i in range(nsta):
        for j in range(nsta):
            for k in range(nsta):
                for l in range(nsta):
                    tmp1[i][j]+=adtmat[k][i]*diablap[k][l]*adtmat[l][j]

    # tmp2 <-> -FS{d/dX W}S^T
    for i in range(nsta):
        for j in range(nsta):
            for k in range(nsta):
                for l in range(nsta):
                    for m in range(nsta):
                        dp=0.0
                        for n in range(ham.nmode_active):
                            dp+=nactmat[n][i][k]*diabderiv1[n][l][m]
                        tmp2[i][j]-=adtmat[l][k]*adtmat[m][j]*dp

    # tmp3 <-> S{d/dX W}S^TF
    for i in range(nsta):
        for j in range(nsta):
            for k in range(nsta):
                for l in range(nsta):
                    for m in range(nsta):
                        dp=0.0
                        for n in range(ham.nmode_active):
                            dp+=nactmat[n][m][j]*diabderiv1[n][k][l]
                        tmp3[i][j]+=adtmat[k][i]*adtmat[l][m]*dp

    # deltmat
    for i in range(nsta):
        for j in range(nsta):
            deltmat[i][j]=tmp1[i][j]+tmp2[i][j]+tmp3[i][j]


    # (b) delximat
    for m in range(ham.nmode_active):
        for i in range(nsta):
            for j in range(nsta):
                if i!=j:
                  delximat[m][i][j]=-(adiabderiv1[m][j]-adiabderiv1[m][i])
                  delximat[m][i][j]=delximat[m][i][j]/(adiabpot[j]-adiabpot[i])**2
            
    # (c) tmat
    for m in range(ham.nmode_active):
        for i in range(nsta):
            for j in range(nsta):
                for k in range(nsta):
                    for l in range(nsta):
                        tmat[m][i][j]+=adtmat[k][i]*diabderiv1[m][k][l]*adtmat[l][j]

    # (d) ximat
    for i in range(nsta):
        for j in range(nsta):
            if i!=j:
                ximat[i][j]=1.0/(adiabpot[j]-adiabpot[i])

    # (f) delnactmat_ij = delximat_ij*tmat_ij + ximat_ij*deltmat_ij (i.ne.j)
    # tmp1 = delximat_ij*tmat_ij
    tmp1=np.zeros((nsta,nsta), dtype=np.float)
    for i in range(nsta):
        for j in range(nsta):
            if i!=j:
                for m in range(ham.nmode_active):
                    tmp1[i][j]+=delximat[m][i][j]*tmat[m][i][j]

    # tmp2 = ximat_ij*deltmat_ij
    tmp2=np.zeros((nsta,nsta), dtype=np.float)
    for i in range(nsta):
        for j in range(nsta):
            if i!=j:
                tmp2[i][j]=ximat[i][j]*deltmat[i][j]

    # delnactmat
    for i in range(nsta):
        for j in range(nsta):
            delnactmat[i][j]=tmp1[i][j]+tmp2[i][j]

    #-------------------------------------------------------------------
    # (2) Construct F.F (fdotf)
    #-------------------------------------------------------------------
    for i in range(nsta):
        for j in range(nsta):
            for k in range(nsta):
                for m in range(ham.nmode_active):
                  fdotf[i][j]+=nactmat[m][i][k]*nactmat[m][k][j]

    #-------------------------------------------------------------------
    # (3) Calculate the scalar coupling terms G = (d/dX F) - F.F
    #-------------------------------------------------------------------
    for i in range(nsta):
        for j in range(nsta):
            sctmat[i][j]=delnactmat[i][j]-fdotf[i][j]

    #-------------------------------------------------------------------
    # (4) Calculation of the 1st derivatives of the DBOCs
    #
    # CHECK THIS CAREFULLY!
    #-------------------------------------------------------------------
    for i in range(nsta):
        for m in range(ham.nmode_active):
            for k in range(nsta):
               dbocderiv1[m][i]-=2.0*delnactmat[i][k]*nactmat[m][i][k]
            
    return

########################################################################

def orbitals(tid,geom,t_state):

    return

########################################################################

def derivative(tid,geom,t_state,lstate,rstate):

    return

########################################################################

def dipole(tid,geom,t_state,lstate,rstate):

    return

########################################################################

def sec_mom(tid,geom,t_state,rstate):

    return

########################################################################

def atom_pop(tid,geom,t_state, rstate):

    return
