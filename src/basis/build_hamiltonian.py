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
import src.dynamics.timings as timings
import src.fmsio.glbl as glbl

#
# return the index in the cent array of the centroid between
# trajectories i and j
#
def c_ind(i, j):
    if i == j:
        return -1
    else:
        a = max(i,j)
        b = min(i,j)
        return int(a*(a-1)/2 + b)

#
# get the (i,j) index of an upper triangular matrix from the 
# sequential matrix index 'index'
#
def ij_ind(index):
    i = 0
    while i*(i+1)/2-1 < index:
        i += 1
    return int(index-i*(i-1)/2),int(i-1) 

#
#
#
def build_hamiltonian(intlib,traj_list,traj_alive,cent_list=None):

    timings.start('build_hamiltonian')

    try:
        integrals = __import__('src.integrals.'+intlib,fromlist=['a'])
        sdot_int      = integrals.sdot_integral
        v_int         = integrals.v_integral
        ke_int        = integrals.ke_integral
        req_centroids = integrals.require_centroids
    except ImportError:
        print('build_hamiltonian cannot import: src.integrals.'+intlib)
        sys.exit(1)

    n_alive = len(traj_alive)
    n_elem  = int(n_alive * (n_alive + 1) / 2)
    c_zero  = np.complex(0.,0.)
    c_imag  = np.complex(0.,1.)

    T        = np.zeros((n_alive,n_alive),dtype=np.complex)
    V        = np.zeros((n_alive,n_alive),dtype=np.complex)
    H        = np.zeros((n_alive,n_alive),dtype=np.complex)
    S        = np.zeros((n_alive,n_alive),dtype=np.complex)
    S_orthog = np.zeros((n_alive,n_alive),dtype=np.complex)
    Sinv     = np.zeros((n_alive,n_alive),dtype=np.complex)
    Sdot     = np.zeros((n_alive,n_alive),dtype=np.complex)
    Heff     = np.zeros((n_alive,n_alive),dtype=np.complex)

    for ij in range(n_elem):
        i,j = ij_ind(ij)
        ii  = traj_alive[i]
        jj  = traj_alive[j]

        # overlap matrix (excluding electronic component)
        S[i,j] = traj_list[ii].overlap(traj_list[jj])
        S[j,i] = S[i,j].conjugate()

        # overlap matrix (including electronic component)               
        if traj_list[ii].state == traj_list[jj].state:
            S_orthog[i,j] = S[i,j]
            S_orthog[j,i] = S[j,i]

            # time derivative of the overlap matrix
            Sdot[i,j]  = sdot_int(traj_list[ii], traj_list[jj], S_ij=S[i,j])
            Sdot[j,i]  = sdot_int(traj_list[jj], traj_list[ii], S_ij=S[j,i])

            # kinetic energy matrix
            T[i,j]     = ke_int(traj_list[ii], traj_list[jj], S_ij=S[i,j])
            T[j,i]     = T[i,j].conjugate()

        else:
            S_orthog[i,j] = c_zero
            S_orthog[j,i] = c_zero

        # potential energy matrix
        if req_centroids:
            V[i,j]  =  v_int(traj_list[ii], traj_list[jj],
                            cent_list[c_ind(ii,jj)],S_ij=S[i,j])
        else:
            V[i,j] =  v_int(traj_list[ii], traj_list[jj], S_ij=S[i,j])
        V[j,i]     = V[i,j].conjugate()

        # Hamiltonian matrix in non-orthongonal basis
        H[i,j]          = T[i,j] + V[i,j]
        H[j,i]          = H[i,j].conjugate()

    # compute the S^-1, needed to compute Heff
    Sinv = np.linalg.pinv(S_orthog)
    Heff = np.dot( Sinv, H - c_imag * Sdot )

    timings.stop('build_hamiltonian')

    return T,V,S,Sdot,Heff

