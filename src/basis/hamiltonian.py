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

def ut_ind(index):
    """Gets the (i,j) index of an upper triangular matrix from the
    sequential matrix index 'index'"""
    i = 0
    while i*(i+1)/2 - 1 < index:
        i += 1
    return index - i*(i-1)//2, i-1

def sq_ind(index, n):
    """Gets the (i,j) index of a square matrix from the
    sequential matrix index 'index'"""
    return index // n, index % n

@timings.timed
def hamiltonian(traj_list, traj_alive, cent_list=None):
    """Builds the Hamiltonian matrix from a list of trajectories."""
    ints = __import__('src.integrals.'+glbl.propagate['integrals'], fromlist=['a'])

    n_alive = len(traj_alive)
    if ints.hermitian:
        n_elem  = int(n_alive * (n_alive + 1) / 2)
    else:
        n_elem  = n_alive * n_alive

    T       = np.zeros((n_alive, n_alive), dtype=complex)
    V       = np.zeros((n_alive, n_alive), dtype=complex)
    H       = np.zeros((n_alive, n_alive), dtype=complex)
    S       = np.zeros((n_alive, n_alive), dtype=complex)
    Sinv    = np.zeros((n_alive, n_alive), dtype=complex)
    Sdot    = np.zeros((n_alive, n_alive), dtype=complex)
    Heff    = np.zeros((n_alive, n_alive), dtype=complex)
    t_ovrlp = np.zeros((n_alive, n_alive), dtype=complex)
    sig     = np.zeros((n_alive, n_alive), dtype=complex)
    tau     = np.zeros((n_alive, n_alive), dtype=complex)

    # now evaluate the hamiltonian matrix
    for ij in range(n_elem):
        if ints.hermitian:
            i, j = ut_ind(ij)
        else:
            i, j = sq_ind(ij, n_alive)

        ii = traj_alive[i]
        jj = traj_alive[j]

#        print("hmat="+str(ii)+' '+str(traj_list[ii].pes_data.diabat_pot))
#        print("theta="+str(ii)+' '+str(jj)+' '+str(ints.theta(traj_list[ii]))+' '+str(ints.theta(traj_list[jj])))
#        print("eovr="+str(ii)+' '+str(jj)+' '+str(ints.elec_overlap(traj_list[ii],traj_list[jj])))

        # compute overlap of trajectories (different from S, which may or may
        # not involve integration in a gaussian basis
        t_ovrlp[i,j] = ints.traj_overlap(traj_list[ii],traj_list[jj])

        # overlap matrix (excluding electronic component)
        Snuc      = ints.s_integral(traj_list[ii],traj_list[jj],nuc_only=True)
#        print("novr="+str(ii)+' '+str(jj)+' '+str(Snuc))

        # overlap matrix (including electronic component)
        S[i,j]    = ints.s_integral(traj_list[ii],traj_list[jj],Snuc=Snuc)

        # time-derivative of the overlap matrix (not hermitian in general)
        Sdot[i,j] = ints.sdot_integral(traj_list[ii], 
                                       traj_list[jj], Snuc=Snuc)

        sig[i,j]     = ints.sdot_integral(traj_list[ii],
                                          traj_list[jj], Snuc=Snuc,e_only=True)
        tau[i,j]     = ints.sdot_integral(traj_list[ii],
                                          traj_list[jj], Snuc=Snuc,nuc_only=True)

        # kinetic energy matrix
        T[i,j]    = ints.ke_integral(traj_list[ii], 
                                     traj_list[jj], Snuc=Snuc)

        # potential energy matrix
        if ints.require_centroids:
            V[i,j] = ints.v_integral(traj_list[ii], traj_list[jj], 
                            centroid=cent_list[ii][jj], Snuc=Snuc)
        else:
            V[i,j] = ints.v_integral(traj_list[ii], 
                                     traj_list[jj], Snuc=Snuc)

        # Hamiltonian matrix in non-orthogonal basis
        H[i,j] = T[i,j] + V[i,j]

        # if hermitian matrix, set (j,i) indices
        if ints.hermitian:
            Snuc         = Snuc.conjugate()
            S[j,i]       = S[i,j].conjugate()
            t_ovrlp[j,i] = t_ovrlp[i,j].conjugate()
            Sdot[j,i]    = ints.sdot_integral(traj_list[jj],
                                              traj_list[ii], Snuc=Snuc)
            sig[j,i]     = ints.sdot_integral(traj_list[jj],
                                              traj_list[ii], Snuc=Snuc,e_only=True)
            tau[j,i]     = ints.sdot_integral(traj_list[jj],
                                              traj_list[ii], Snuc=Snuc,nuc_only=True)

            T[j,i]       = T[i,j].conjugate()
            V[j,i]       = V[i,j].conjugate()
            H[j,i]       = H[i,j].conjugate()

    if ints.hermitian:
        # compute the S^-1, needed to compute Heff
        timings.start('linalg.pinvh')
        Sinv = sp_linalg.pinvh(S)
        timings.stop('linalg.pinvh')
    else:
        # compute the S^-1, needed to compute Heff
        timings.start('hamiltonian.pseudo_inverse')
        Sinv, cond = fms_linalg.pseudo_inverse(S)
        timings.stop('hamiltonian.pseudo_inverse')

    Heff = np.dot( Sinv, H - 1j * Sdot )

    sig[abs(sig)< 1e-10]=0
#    print("tau="+str(tau))

    return t_ovrlp, T, V, S, Sdot, Heff
