"""
Linear algebra library routines.
"""
import numpy as np
import scipy.linalg as sp_linalg
import nomad.core.glbl as glbl
import nomad.math.constants as constants

#
# Routines for diabatic <-> adiabatic transformations

def calc_diabeffcoup(diabpot):
    """Calculates the effective diabatic coupling between diabatic states i, j via
       eff_coup = Hij / (H[i,i] - H[j,j])
    """
    nst = diabpot.shape[0]

    demat = np.array([[max([constants.fpzero, diabpot[i,i]-diabpot[j,j]],key=abs)
                       for i in range(nst)] for j in range(nst)])
    eff_coup = np.divide(diabpot - np.diag(diabpot.diagonal()), demat)

    return eff_coup

def calc_dat(label, diabpot, previous_datmat=None):
    """Diagonalises the diabatic potential matrix to yield the adiabatic
    potentials and the adiabatic-to-diabatic transformation matrix."""
    adiabpot, datmat = sp_linalg.eigh(diabpot)

    if previous_datmat is not None:
        # Ensure phase continuity from geometry to another
        datmat *= np.sign(np.dot(datmat.T, previous_datmat).diagonal())
    else:
        # Set phase convention that the greatest abs element in dat column
        # vector is positive
        datmat *= np.sign(datmat[range(len(adiabpot)),
                                 np.argmax(np.abs(datmat), axis=0)])
    return adiabpot, datmat

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
    nd      = diabderiv1.shape[0]
    ns      = diabderiv1.shape[1]
    nactmat = np.zeros((nd, ns, ns))
    fac = -np.subtract.outer(adiabpot, adiabpot) + np.eye(ns)
    for m in range(nd):
        nactmat[m] = np.dot(np.dot(datmat.T, diabderiv1[m]), datmat) / fac

    # Subtract the diagonal to make sure it is zero
    nactmat -= [np.diag(nactmat[m].diagonal()) for m in range(nd)]
    return nactmat

#
def calc_adiabderiv1(datmat, diabderiv1):
    """Calculates the gradients of the adiabatic potentials.

    Equation used: d/dX V_ii = (S{d/dX W}S^T)_ii
    """
    nd      = diabderiv1.shape[0]
    ns      = diabderiv1.shape[1]

    # Get the diagonal elements of the matrix
    adiabderiv1 = np.zeros((nd, ns))
    for m in range(nd):
        adiabderiv1[m] = np.dot(np.dot(datmat.T, diabderiv1[m]),
                                datmat).diagonal()
    return adiabderiv1

#
def calc_adiabderiv2(datmat, diabderiv2):
    """Calculates the hessians of the adiabatic potentials.

    Equation used: d^2/dXidXj V_ii = (S{d^2/dXidXj W}S^T)_ii
    """
    # Get the diagonal elements of the matrix
    nd      = diabderiv2.shape[0]
    ns      = diabderiv2.shape[2]

    adiabderiv2 = np.zeros((nd, nd, ns))

    for m in range(nd):
        for n in range(m+1):
            adiabderiv2[m,n] = np.dot(np.dot(datmat.T, diabderiv2[m,n]),
                                             datmat).diagonal()
            if m != n:
                adiabderiv2[n,m] = adiabderiv2[m,n]

    return adiabderiv2

def calc_scts(adiabpot, datmat, diabderiv1, nactmat, adiabderiv1, diablap, freq):
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
    nd      = diabderiv1.shape[0]
    ns      = diabderiv1.shape[1]

    # (a) deltmat = S {Del^2 W} S^T + -FS{d/dX W}S^T + S{d/dX W}S^TF
    # tmp1 <-> S {Del^2 W} S^T
    tmp1 = np.dot(np.dot(datmat.T, diablap), datmat)

    # tmp2 <-> -F S{d/dX W}S^T
    mat2 = [np.dot(np.dot(np.dot(nactmat[m], datmat.T), diabderiv1[m]), datmat)
            for m in range(nd)]
    tmp2 = -np.sum(freq[:,np.newaxis,np.newaxis] * mat2, axis=0)

    # tmp3 <-> S{d/dX W}S^T F
    mat3 = [np.dot(np.dot(np.dot(datmat.T, diabderiv1[m]), datmat), nactmat[m])
            for m in range(nd)]
    tmp3 = np.sum(freq[:,np.newaxis,np.newaxis] * mat3, axis=0)

    # deltmat
    deltmat = tmp1 + tmp2 + tmp3
    # (b) delximat
    ## This is slightly slower for nsta == 2
    #delximat = np.zeros((ham.nmode_total, nsta, nsta))
    #for m in ham.mrange:
    #    delximat[m] = -(np.subtract.outer(adiabderiv1[m], adiabderiv1[m]) /
    #                    (np.subtract.outer(adiabpot, adiabpot) ** 2 +
    #                     np.eye(nsta)))
    delximat = np.zeros((nd, ns, ns))
    for m in range(nd):
        for i in range(ns):
            for j in range(ns):
                if i != j:
                    delximat[m,i,j] = ((adiabderiv1[m,i] - adiabderiv1[m,j]) /
                                       (adiabpot[j] - adiabpot[i])**2)

    # (c) tmat
    tmat = np.zeros((nd, ns, ns))
    for m in range(nd):
        tmat[m] = np.dot(np.dot(datmat.T, diabderiv1[m]), datmat)

    # (d) ximat
    ## This is slightly slower for nsta == 2
    #ximat = 1. / (-np.subtract.outer(adiabpot, adiabpot) +
    #              np.eye(nsta)) - np.eye(nsta)
    ximat = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(ns):
            if i != j:
                ximat[i,j] = 1. / (adiabpot[j] - adiabpot[i])

    # (f) delnactmat_ij = delximat_ij*tmat_ij + ximat_ij*deltmat_ij (i.ne.j)
    delnactmat = np.sum(delximat * tmat, axis=0) + ximat * deltmat

    #-------------------------------------------------------------------
    # (2) Construct F.F (fdotf)
    #-------------------------------------------------------------------
    matf = [np.dot(nactmat[m], nactmat[m]) for m in range(nd)]
    fdotf = np.sum(freq[:,np.newaxis,np.newaxis] * matf, axis=0)

    #-------------------------------------------------------------------
    # (3) Calculate the scalar coupling terms G = (d/dX F) - F.F
    #-------------------------------------------------------------------
    sctmat = delnactmat - fdotf

    #-------------------------------------------------------------------
    # (4) Calculation of the 1st derivatives of the DBOCs
    #-------------------------------------------------------------------
    dbocderiv1 = np.zeros((nd, ns))
    for m in range(nd):
        dbocderiv1[m] = -2.*np.sum(delnactmat * nactmat[m], axis=0)

    return sctmat, dbocderiv1
