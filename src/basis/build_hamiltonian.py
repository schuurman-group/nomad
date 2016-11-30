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
import scipy.linalg as sp_linalg
import src.dynamics.timings as timings
import src.fmsio.glbl as glbl
import src.utils.linalg as fms_linalg

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

@timings.timed
def build_hamiltonian(nucint, trajint, traj_list, traj_alive, cent_list=None):
    """Builds the Hamiltonian matrix from a list of trajectories."""
    nuc_int  = __import__('src.integrals.nuclear_'+ nucint,  fromlist=['a'])
    traj_int = __import__('src.integrals.trajectory_'+ trajint, fromlist=['a'])

    n_alive = len(traj_alive)
    if traj_int.hermitian:
        n_elem  = int(n_alive * (n_alive + 1) / 2)
    else:
        n_elem  = n_alive * n_alive

    T    = np.zeros((n_alive, n_alive), dtype=complex)
    V    = np.zeros((n_alive, n_alive), dtype=complex)
    H    = np.zeros((n_alive, n_alive), dtype=complex)
    Snuc = np.zeros((n_alive, n_alive), dtype=complex)
    S    = np.zeros((n_alive, n_alive), dtype=complex)
    Sinv = np.zeros((n_alive, n_alive), dtype=complex)
    Sdot = np.zeros((n_alive, n_alive), dtype=complex)
    Heff = np.zeros((n_alive, n_alive), dtype=complex)

    for ij in range(n_elem):
        if traj_int.hermitian:
            i, j = ut_ind(ij)
        else:
            i, j = sq_ind(ij, n_alive)

        ii = traj_alive[i]
        jj = traj_alive[j]

        # overlap matrix (excluding electronic component)
        Snuc[i,j] = nuc_int.overlap(traj_list[ii],traj_list[jj])

        # overlap matrix (including electronic component)
        S[i,j]    = traj_int.s_integral(traj_list[ii], 
                                        traj_list[jj], Snuc=Snuc[i,j])

        # time-derivative of the overlap matrix (not hermitian in general)
        Sdot[i,j] = traj_int.sdot_integral(traj_list[ii], 
                                           traj_list[jj], Snuc=Snuc[i,j])

        # kinetic energy matrix
        T[i,j]    = traj_int.ke_integral(traj_list[ii], 
                                         traj_list[jj], Snuc=Snuc[i,j])

        # potential energy matrix
        if traj_int.require_centroids:
            V[i,j] = traj_int.v_integral(traj_list[ii], traj_list[jj], 
                       centroid=cent_list[c_ind(ii,jj)], Snuc=Snuc[i,j])
        else:
            V[i,j] = traj_int.v_integral(traj_list[ii], 
                                         traj_list[jj], Snuc=Snuc[i,j])

        # Hamiltonian matrix in non-orthogonal basis
        H[i,j] = T[i,j] + V[i,j]

        # if hermitian matrix, set (j,i) indices
        if traj_int.hermitian:
            Snuc[j,i] = Snuc[i,j].conjugate()
            S[j,i]    = S[i,j].conjugate()
            Sdot[j,i] = traj_int.sdot_integral(traj_list[jj],
                                               traj_list[ii], Snuc=Snuc[j,i])
            T[j,i]      = T[i,j].conjugate()
            V[j,i]      = V[i,j].conjugate()
            H[j,i]      = H[i,j].conjugate()


    if traj_int.hermitian:
        # compute the S^-1, needed to compute Heff
        timings.start('linalg.pinvh')
        Sinv = sp_linalg.pinvh(S)
        timings.stop('linalg.pinvh')
    else:
        # compute the S^-1, needed to compute Heff
        timings.start('build_hamiltonian.pseudo_inverse')
        Sinv, cond = fms_linalg.pseudo_inverse(S)
        timings.stop('build_hamiltonian.pseudo_inverse')

    Heff = np.dot( Sinv, H - 1j * Sdot )

    return T, V, Snuc, Sdot, Heff
