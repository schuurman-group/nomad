"""
Form the Hermitian Hamiltonian matrix in the basis of the FMS trajectories.

This will necessarily involve a set of additional matrices. For ab initio
propagation, this is never the rate determining step. For numerical
propagation, however, it is THE bottleneck. Thus, this function is
compiled in Cython in order to make use of openMP parallelization.

As a matter of course, this function also builds:
   - the S matrix
   - the time derivative of the S matrix (sdot)
   - the effective Hamiltonian (i.e. i * S^-1 [ S * H - Sdot])
     --> this is the matrix employed to solve for the time
         dependent amplitudes
"""
import sys
import numpy as np
from scipy import linalg
import src.dynamics.timings as timings
import src.fmsio.glbl as glbl

def c_ind(i, j):
    """Returns the index in the cent array of the centroid between
    trajectories i and j."""
    if i == j:
        return -1
    else:
        a = max(i, j)
        b = min(i, j)
        return int(a*(a-1)/2 + b)

def ut_ind(index):
    """Gets the (i,j) index of an upper triangular matrix from the
    sequential matrix index 'index'"""
    i = 0
    while i*(i+1)/2 - 1 < index:
        i += 1
    return int(index-i*(i-1)/2), int(i-1)

def sq_ind(index, n):
    """Gets the (i,j) index of a square matrix from the 
    sequential matrix index 'index'"""
    return int(index / n), index - int(index / n) * n

def pseudo_inverse(mat, dim):
    """ Modified version of the scipy pinv function. Altered such that
    the the cutoff for singular values can be set to a hard
    value. Note that by default the scipy cutoff of 1e-15*sigma_max is
    taken."""

    invmat = np.zeros((dim, dim), dtype=complex)
    mat=np.conjugate(mat)
    
    # SVD of the overlap matrix
    u, s, vt = np.linalg.svd(mat, full_matrices=True)

    #print("\n",s,"\n")

    # Condition number
    if s[dim-1] < 1e-90:
        cond = 1e+90
    else:
        cond = s[0]/s[dim-1]

    # Moore-Penrose pseudo-inverse
    if glbl.fms['sinv_thrsh'] == -1.0:
        cutoff = glbl.fms['sinv_thrsh']*np.maximum.reduce(s)
    else:
        cutoff = glbl.fms['sinv_thrsh']
    for i in range(dim):
        if s[i] > cutoff:
            s[i] = 1./s[i]
        else:
            s[i] = 0.
    invmat = np.dot(np.transpose(vt), np.multiply(s[:, np.newaxis],
                                                  np.transpose(u)))    

    return invmat, cond

@timings.timed
def build_hamiltonian(intlib, traj_list, traj_alive, cent_list=None):
    """Builds the Hamiltonian matrix from a list of trajectories."""
    integrals = __import__('src.integrals.' + intlib, fromlist=['a'])

    n_alive = len(traj_alive)
    if integrals.hermitian:
        n_elem  = int(n_alive * (n_alive + 1) / 2)
    else:
        n_elem  = n_alive * n_alive

    T        = np.zeros((n_alive, n_alive), dtype=complex)
    V        = np.zeros((n_alive, n_alive), dtype=complex)
    H        = np.zeros((n_alive, n_alive), dtype=complex)
    Snuc     = np.zeros((n_alive, n_alive), dtype=complex)
    Stotal   = np.zeros((n_alive, n_alive), dtype=complex)
    Sinv     = np.zeros((n_alive, n_alive), dtype=complex)
    Sdot     = np.zeros((n_alive, n_alive), dtype=complex)
    Heff     = np.zeros((n_alive, n_alive), dtype=complex)

    for ij in range(n_elem):
        if integrals.hermitian:
            i, j = ut_ind(ij)
        else:
            i, j = sq_ind(ij, n_alive)

        ii = traj_alive[i]
        jj = traj_alive[j]

        # overlap matrix (excluding electronic component)
        Snuc[i,j] = integrals.snuc_integral(traj_list[ii],traj_list[jj])

        # overlap matrix (including electronic component)
        Stotal[i,j] = integrals.stotal_integral(traj_list[ii], 
                                        traj_list[jj], Snuc=Snuc[i,j])

        # time-derivative of the overlap matrix (not hermitian in general)
        Sdot[i,j]   = integrals.sdot_integral(traj_list[ii], 
                                        traj_list[jj], Snuc=Snuc[i,j])
        Sdot[j,i]   = integrals.sdot_integral(traj_list[ii], 
                                        traj_list[jj], Snuc=Snuc[j,i])

        # kinetic energy matrix
        T[i,j] = integrals.ke_integral(traj_list[ii], 
                                       traj_list[jj], Snuc=Snuc[i,j])

        # potential energy matrix
        if integrals.require_centroids:
            V[i,j] = integrals.v_integral(traj_list[ii], traj_list[jj], 
                              centroid=cent_list[c_ind(ii,jj)], Snuc=Snuc[i,j])
        else:
            V[i,j] = integrals.v_integral(traj_list[ii], 
                                          traj_list[jj], Snuc=Snuc[i,j])

        # Hamiltonian matrix in non-orthogonal basis
        H[i,j] = T[i,j] + V[i,j]

        # if hermitian matrix, set (j,i) indices
        if integrals.hermitian:
            Snuc[j,i]   = Snuc[i,j].conjugate()
            Stotal[j,i] = Stotal[i,j].conjugate()
            T[j,i]      = T[i,j].conjugate()
            V[j,i]      = V[i,j].conjugate()
            H[j,i]      = H[i,j].conjugate()


    if integrals.hermitian:
        # compute the S^-1, needed to compute Heff
        timings.start('linalg.pinvh')
        Sinv = linalg.pinvh(Stotal)
        timings.stop('linalg.pinvh')
    else:
        # compute the S^-1, needed to compute Heff
        timings.start('build_hamiltonian.pseudo_inverse')
        Sinv, cond = pseudo_inverse(Stotal, n_alive)
        timings.stop('build_hamiltonian.pseudo_inverse')

    Heff = np.dot( Sinv, H - 1j * Sdot )

    return T, V, Snuc, Sdot, Heff
