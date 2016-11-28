#
# Form the Hamiltonian matrix in the basis of the FMS trajectories.
# This will necessarily involve a set of additional matrices. For ab initio
# propagation, this is never the rate determining step. For numerical 
# propagation, however, it is THE bottleneck. Thus, this function is
# compiled in Cython in order to make use of openMP parallelization. 
#
# As a matter of course, this function also builds:
#    - the S matrix
#    - the time derivative of the S matrix (sdot)
#    - the effective Hamiltonian (i.e. i * S^-1 [ S * H - Sdot])
#      --> this is the matrix employed to solve for the time
#          dependent amplitudes 
import sys
import numpy as np
from cython.parallel import prange,parallel
import src.dynamics.timings as timings


#
#
def build_hamiltonian(intlib,traj_list,int[:] traj_alive):

    timings.start('build_hamiltonian')

    try:
        integrals = __import__('src.integrals.'+intlib,fromlist=['a'])
        sdot_int      = integrals.sdot_integral_c
        v_int         = integrals.v_integral_c
        ke_int        = integrals.ke_integral_c
    except ImportError:
        print('build_hamiltonian cannot import: src.integrals.'+intlib)
        sys.exit(1)

    cdef int                      a,b,i,j,ii,jj,ij
    cdef int            n_crd   = traj_list[0].n_particle * traj_list[0].d_particle
    cdef int            n_alive = len(traj_alive)
    cdef int            n_elem  = int(n_alive * (n_alive + 1) / 2)
    cdef double complex c_zero  = np.complex(0.,0.)
    cdef double complex c_imag  = np.complex(0.,1.)

    cdef int[:]              ind      = np.zeros(2,dtype=np.int)
    cdef double complex[:,:] T        = np.zeros((n_alive,n_alive),dtype=np.complex)
    cdef double complex[:,:] V        = np.zeros((n_alive,n_alive),dtype=np.complex)
    cdef double complex[:,:] H        = np.zeros((n_alive,n_alive),dtype=np.complex)
    cdef double complex[:,:] S        = np.zeros((n_alive,n_alive),dtype=np.complex)
    cdef double complex[:,:] S_orthog = np.zeros((n_alive,n_alive),dtype=np.complex)
    cdef double complex[:,:] Sinv     = np.zeros((n_alive,n_alive),dtype=np.complex)
    cdef double complex[:,:] Sdot     = np.zeros((n_alive,n_alive),dtype=np.complex)
    cdef double complex[:,:] iSdot    = np.zeros((n_alive,n_alive),dtype=np.complex)
    cdef double complex[:,:] Heff     = np.zeros((n_alive,n_alive),dtype=np.complex)
    cdef double[:,:,:] traj_data      = np.zeros((n_crd,3,n_alive),dtype=np.float)
    cdef double[:]     phas_data      = np.zeros(n_alive,dtype=np.float)
    cdef int[:]        stat_data      = np.zeros(n_alive,dtype=np.int)

    #
    # return the index in the cent array of the centroid between
    # trajectories i and j
    #
    def c_ind(int ii, int jj):
        cdef int a,b
        if ii == jj:
            return -1
        else:
            a = max(ii,jj)
            b = min(ii,jj)
        return int(a*(a-1)/2 + b)

    #
    # return pair of indices given sequential index
    #
    def ij_ind(int ij,int *arr):
        while arr[1]*(arr[1]+1)/2-1 < ij:
            arr[1] += 1
        arr[1] -= 1
        arr[0]  = int(ij - arr[1]*(arr[1]-1)/2)
        return arr

    traj_data = pack_trajectories(traj_list,n_alive)
    for i in range(n_alive):
        ii = traj_alive[i]
        phas_data[i] = traj_list[i].phase
        stat_data[i] = traj_list[i].state

    with nogil, parallel():
        for ij in prange(n_elem,schedule='static'):
            ij_ind(ij,ind)
            i   = ind[0]
            j   = ind[1]

            # overlap matrix (excluding electronic component)
            S[i,j] = t_overlap(n_crd, phas_data[i], phas_data[j], traj_data[:,:,i],traj_data[:,:,j])
            S[j,i] = conj(S[i,j])

            # overlap matrix (including electronic component)               
            if stat_data[i] == stat_data[j]:
                S_orthog[i,j] = S[i,j]
                S_orthog[j,i] = S[j,i]
    
                # time derivative of the overlap matrix
                Sdot[i,j]  = sdot_int(traj_list[ii], traj_list[jj], S_ij=S[i,j])
                Sdot[j,i]  = sdot_int(traj_list[jj], traj_list[ii], S_ij=S[j,i])

                # kinetic energy matrix
                T[i,j]     = ke_int(traj_list[ii], traj_list[jj], S_ij=S[i,j])
                T[j,i]     = conj(T[i,j])

            else:
                S_orthog[i,j] = c_zero
                S_orthog[j,i] = c_zero

            # potential energy matrix
            V[i,j]     = v_int(traj_list[ii], traj_list[jj], S_ij=S[i,j])
            V[j,i]     = conj(V[i,j])

            # Hamiltonian matrix in non-orthongonal basis
            H[i,j]          = T[i,j] + V[i,j]
            H[j,i]          = conj(H[i,j])

    for i in range(n_alive):
        iSdot[i,i] = Sdot[i,i]*c_imag
        H[i,i]     -= iSdot[i,i]
        for j in range(i):
            iSdot[i,j] = Sdot[i,j]*c_imag
            iSdot[j,i] = Sdot[j,i]*c_imag
            H[i,j]     -= iSdot[i,j]
            H[j,i]     -= iSdot[j,i]

    # compute the S^-1, needed to compute Heff
    Sinv = np.linalg.pinv(S_orthog)
    Heff = np.dot( Sinv, H )

    timings.stop('build_hamiltonian')

    return T,V,S,Sdot,Heff

    #
    # pack trajectory info into c array structures
    #
    def pack_trajectory(traj_list, traj_alive, double *packed_traj):

        for i in range(len(traj_alive)):
            ii = traj_alive[i]
            packed_traj[:,0,i] = traj_list[ii].x()
            packed_traj[:,1,i] = traj_list[ii].p()
            packed_traj[:,2,i] = traj_list[ii].widths() 
        return

    #
    #
    #
    cdef double complex t_overlap(int n_crd, double pi, double pj, double *ti, double *tj):
      
        double complex sij = cexp( I * (pj - pi) )
        for i in range(n_crd):
            sij *= gaussian.overlap(ti[0,i],ti[1,i],ti[2,i],tj[0,i],tj[1,i],tj[2,i])

        return sij
