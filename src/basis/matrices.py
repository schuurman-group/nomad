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
import src.utils.linalg as fms_linalg
import src.utils.timings as timings

class Matrices:
    """Builds the Hamiltonian and associated matrices given a wavefunction and
       an Integral object"""
      
    self.T      = np.empty(shape=(0, 0)) 
    self.V      = np.empty(shape=(0, 0))
    self.H      = np.empty(shape=(0, 0))
    self.S_traj = np.empty(shape=(0, 0))
    self.S_nuc  = np.empty(shape=(0, 0))
    self.S      = np.empty(shape=(0, 0))
    self.Sinv   = np.empty(shape=(0, 0))
    self.Sdot   = np.empty(shape=(0, 0))
    self.Heff   = np.empty(shape=(0, 0))

    # temporary debugging business
    self.Sdnuc = np.empty(shape=(0, 0))
    self.Sdele = np.empty(shape=(0, 0))

    # provide this dictionary to point to member matrices
    self.mat_lst = {"T":self.T, "V":self.V, "H":self.H, "S_traj":self.S_traj,
                    "S_nuc":self.S_nuc, "S":self.S, "Sinv":self.Sinv, "Sdot", self.Sdot,
                    "Heff":self.Heff, "Sdnuc":self.Sdnuc, "Sdele", self.Sdele}

    #
    #
    #
    def set(self, key, value):
        """set the matrices in the current matrix object equal
           to the passed matrix object"""
        
        self.mat_lst[key] = None
        self.mat_lst[key] = copy.deepcopy(value)        
     
        return

    #
    #
    #
    def copy(self):
        new_matrices = Matrices()

        for key,value in self.items()
            new_matrices.set(key,value)

        return new_matrices
   
    #
    #
    #
    @timings.timed
    def build(self, wfn, integrals):
    """Builds the Hamiltonian matrix from a list of trajectories."""

    n_alive = wfn.nalive

    if integrals.hermitian:
        n_elem  = int(n_alive * (n_alive + 1) / 2)
    else:
        n_elem  = n_alive * n_alive

    if self.Heff.shape != (n_alive, n_alive):
        self.T       = np.zeros((n_alive, n_alive), dtype=complex)
        self.V       = np.zeros((n_alive, n_alive), dtype=complex)
        self.H       = np.zeros((n_alive, n_alive), dtype=complex)
        self.S_traj  = np.zeros((n_alive, n_alive), dtype=complex)
        self.S_nuc   = np.zeros((n_alive, n_alive), dtype=complex)
        self.S       = np.zeros((n_alive, n_alive), dtype=complex)
        self.Sinv    = np.zeros((n_alive, n_alive), dtype=complex)
        self.Sdot    = np.zeros((n_alive, n_alive), dtype=complex)
        self.Heff    = np.zeros((n_alive, n_alive), dtype=complex)
        self.Sdnuc   = np.zeros((n_alive, n_alive), dtype=complex)
        self.Sdele   = np.zeros((n_alive, n_alive), dtype=complex)

    # now evaluate the hamiltonian matrix
    for ij in range(n_elem):
        if integrals.hermitian:
            i, j = self.ut_ind(ij)
        else:
            i, j = self.sq_ind(ij, n_alive)

        ii = traj_alive[i]
        jj = traj_alive[j]

        # nuclear overlap matrix (excluding electronic component)
        S_nuc[i,j]  = integrals.nuc_overlap(traj_list[ii],traj_list[jj])

        # compute overlap of trajectories (different from S, which may or may
        # not involve integration in a gaussian basis
        S_traj[i,j] = integrals.traj_overlap(traj_list[ii],traj_list[jj],Snuc=Snuc[i,j])

        # overlap matrix (including electronic component)
        S[i,j]      = integrals.s_integral(traj_list[ii],traj_list[jj],Snuc=Snuc[i,j])

        # time-derivative of the overlap matrix (not hermitian in general)
        Sdot[i,j]   = integrals.sdot_integral(traj_list[ii],traj_list[jj],Snuc=Snuc[i,j])

        # debugging stuff: temporary
        Sdnuc[i,j]  = integrals.sdot_integral(traj_list[ii],
                                       traj_list[jj], Snuc=Snuc[i,j],nuc_only=True)
        Sdele[i,j]  = integrals.sdot_integral(traj_list[ii],
                                       traj_list[jj], Snuc=Snuc[i,j],e_only=True)

        # kinetic energy matrix
        T[i,j]      = integrals.ke_integral(traj_list[ii],traj_list[jj], Snuc=Snuc[i,j])

        # potential energy matrix
        V[i,j]      = integrals.v_integral(traj_list[ii],traj_list[jj], Snuc=Snuc[i,j])

        # Hamiltonian matrix in non-orthogonal basis
        H[i,j] = T[i,j] + V[i,j]

        # if hermitian matrix, set (j,i) indices
        if integrals.hermitian and i!=j:
            S_nuc[j,i]   = S_nuc[i,j].conjugate()
            S_traj[j,i]  = S_traj[i,j].conjugate()
            S[j,i]       = S[i,j].conjugate()
            Sdot[j,i]    = integrals.sdot_integral(traj_list[jj],
                                                   traj_list[ii], Snuc=Snuc[j,i])
            T[j,i]       = T[i,j].conjugate()
            V[j,i]       = V[i,j].conjugate()
            H[j,i]       = H[i,j].conjugate()

            # TEMPORARY 
            Sdnuc[j,i] = integrals.sdot_integral(traj_list[jj],
                                                traj_list[ii], Snuc=Snuc[j,i],nuc_only=True)
            Sdele[j,i] = ntegrals.sdot_integral(traj_list[jj],
                                                traj_list[ii], Snuc=Snuc[j,i],e_only=True)

    if integrals.hermitian:
        # compute the S^-1, needed to compute Heff
        timings.start('linalg.pinvh')
        Sinv = sp_linalg.pinvh(S)
#        Sinv, cond = fms_linalg.pseudo_inverse2(S)
        timings.stop('linalg.pinvh')
    else:
        # compute the S^-1, needed to compute Heff
        timings.start('hamiltonian.pseudo_inverse')
        Sinv, cond = fms_linalg.pseudo_inverse(S)
        timings.stop('hamiltonian.pseudo_inverse')

    Heff = np.dot( Sinv, H - 1j * Sdot )
    
    #
    #
    #
    def ut_ind(index):
        """Gets the (i,j) index of an upper triangular matrix from the
        sequential matrix index 'index'"""
        i = 0
        while i*(i+1)/2 - 1 < index:
            i += 1
        return index - i*(i-1)//2, i-1

    #
    #
    #
    def sq_ind(index, n):
        """Gets the (i,j) index of a square matrix from the
        sequential matrix index 'index'"""
        return index // n, index % n


