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
import copy as copy
import scipy.linalg as sp_linalg
import src.utils.linalg as fms_linalg
import src.utils.timings as timings

class Matrices:
    """Builds the Hamiltonian and associated matrices given a wavefunction and
       an Integral object"""
    def __init__(self):      
        self.T      = np.empty(shape=(0, 0)) 
        self.V      = np.empty(shape=(0, 0))
        self.H      = np.empty(shape=(0, 0))
        self.S_traj = np.empty(shape=(0, 0))
        self.S_nuc  = np.empty(shape=(0, 0))
        self.S      = np.empty(shape=(0, 0))
        self.Sinv   = np.empty(shape=(0, 0))
        self.Sdot   = np.empty(shape=(0, 0))
        self.Heff   = np.empty(shape=(0, 0))

        # provide this dictionary to point to member matrices
        self.mat_lst = {"T":self.T, "V":self.V, "H":self.H, "S_traj":self.S_traj,
                        "S_nuc":self.S_nuc, "S":self.S, "Sinv":self.Sinv, "Sdot":self.Sdot,
                        "Heff":self.Heff}

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
        """Documentation to come"""
        new_matrices = Matrices()

        for key,value in self.mat_lst.items():
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

        # now evaluate the hamiltonian matrix
        for ij in range(n_elem):
            if integrals.hermitian:
                i, j = self.ut_ind(ij)
            else:
                i, j = self.sq_ind(ij, n_alive)
    
            ii = wfn.alive[i]
            jj = wfn.alive[j]

            # nuclear overlap matrix (excluding electronic component)
            self.S_nuc[i,j]  = integrals.nuc_overlap(wfn.traj[ii],wfn.traj[jj])
    
            # compute overlap of trajectories (different from S, which may or may
            # not involve integration in a gaussian basis
            self.S_traj[i,j] = integrals.traj_overlap(wfn.traj[ii],wfn.traj[jj],nuc_ovrlp=self.S_nuc[i,j])
    
            # overlap matrix (including electronic component)
            self.S[i,j]      = integrals.s_integral(wfn.traj[ii],wfn.traj[jj],nuc_ovrlp=self.S_nuc[i,j])

            # time-derivative of the overlap matrix (not hermitian in general)
            self.Sdot[i,j]   = integrals.sdot_integral(wfn.traj[ii],wfn.traj[jj],nuc_ovrlp=self.S_nuc[i,j])

            # kinetic energy matrix
            self.T[i,j]      = integrals.t_integral(wfn.traj[ii],wfn.traj[jj], nuc_ovrlp=self.S_nuc[i,j])

            # potential energy matrix
            self.V[i,j]      = integrals.v_integral(wfn.traj[ii],wfn.traj[jj], nuc_ovrlp=self.S_nuc[i,j])

            # Hamiltonian matrix in non-orthogonal basis
            self.H[i,j]      = self.T[i,j] + self.V[i,j]

            # if hermitian matrix, set (j,i) indices
            if integrals.hermitian and i!=j:
                self.S_nuc[j,i]   = self.S_nuc[i,j].conjugate()
                self.S_traj[j,i]  = self.S_traj[i,j].conjugate()
                self.S[j,i]       = self.S[i,j].conjugate()
                self.Sdot[j,i]    = integrals.sdot_integral(wfn.traj[jj],
                                                   wfn.traj[ii], nuc_ovrlp=self.S_nuc[j,i])
                self.T[j,i]       = self.T[i,j].conjugate()
                self.V[j,i]       = self.V[i,j].conjugate()
                self.H[j,i]       = self.H[i,j].conjugate()

        if integrals.hermitian:
            # compute the S^-1, needed to compute Heff
            timings.start('linalg.pinvh')
            self.Sinv = sp_linalg.pinvh(self.S)
#            Sinv, cond = fms_linalg.pseudo_inverse2(S)
            timings.stop('linalg.pinvh')
        else:
            # compute the S^-1, needed to compute Heff
            timings.start('hamiltonian.pseudo_inverse')
            self.Sinv, cond = fms_linalg.pseudo_inverse(S)
            timings.stop('hamiltonian.pseudo_inverse')

        self.Heff = np.dot( self.Sinv, self.H - 1j * self.Sdot )
    
    #
    #
    #
    def ut_ind(self,index):
        """Gets the (i,j) index of an upper triangular matrix from the
        sequential matrix index 'index'"""
        i = 0
        while i*(i+1)/2 - 1 < index:
            i += 1
        return index - i*(i-1)//2, i-1

    #
    #
    #
    def sq_ind(self,index, n):
        """Gets the (i,j) index of a square matrix from the
        sequential matrix index 'index'"""
        return index // n, index % n


