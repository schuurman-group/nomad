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
import copy
import numpy as np
import scipy.linalg as sp_linalg
import nomad.math.linalg as linalg
import nomad.core.timings as timings


class Matrices:
    """Object containing the Hamiltonian and associated matrices."""
    def __init__(self):
        self.mat_dict = dict(
            t      = np.empty((0, 0)),
            v      = np.empty((0, 0)),
            h      = np.empty((0, 0)),
            s_traj = np.empty((0, 0)),
            s_nuc  = np.empty((0, 0)),
            s_elec = np.empty((0, 0)),
            s      = np.empty((0, 0)),
            sinv   = np.empty((0, 0)),
            sdot   = np.empty((0, 0)),
            heff   = np.empty((0, 0))
                             )

    def set(self, name, matrix):
        """Sets the matrices in the current matrix object equal
        to the passed matrix object"""
        self.mat_dict[name] = copy.deepcopy(matrix)

    def copy(self):
        """Documentation to come"""
        new_matrices = Matrices()

        for typ,matrix in self.mat_dict.items():
            new_matrices.set(typ, matrix)

        return new_matrices

    @timings.timed
    def build(self, wfn, integrals):
        """Builds the Hamiltonian matrix from a list of trajectories."""
        n_alive = wfn.nalive

        if integrals.hermitian:
            n_elem  = int(n_alive * (n_alive + 1) / 2)
        else:
            n_elem  = n_alive * n_alive

        if self.mat_dict['heff'].shape != (n_alive, n_alive):
            self.mat_dict['t']       = np.zeros((n_alive, n_alive), dtype=complex)
            self.mat_dict['v']       = np.zeros((n_alive, n_alive), dtype=complex)
            self.mat_dict['h']       = np.zeros((n_alive, n_alive), dtype=complex)
            self.mat_dict['s_traj']  = np.zeros((n_alive, n_alive), dtype=complex)
            self.mat_dict['s_nuc']   = np.zeros((n_alive, n_alive), dtype=complex)
            self.mat_dict['s_elec']  = np.zeros((n_alive, n_alive), dtype=complex)
            self.mat_dict['s']       = np.zeros((n_alive, n_alive), dtype=complex)
            self.mat_dict['sinv']    = np.zeros((n_alive, n_alive), dtype=complex)
            self.mat_dict['sdot']    = np.zeros((n_alive, n_alive), dtype=complex)
            self.mat_dict['heff']    = np.zeros((n_alive, n_alive), dtype=complex)

        # now evaluate the hamiltonian matrix
        for ij in range(n_elem):
            if integrals.hermitian:
                i, j = self.ut_ind(ij)
            else:
                i, j = self.sq_ind(ij, n_alive)

            ii = wfn.alive[i]
            jj = wfn.alive[j]

            s_nuc  = integrals.nuc_overlap(wfn.traj[ii],wfn.traj[jj])
            s_elec = integrals.elec_overlap(wfn.traj[ii],wfn.traj[jj])

            # nuclear overlap matrix (excluding electronic component)
            self.mat_dict['s_nuc'][i,j]  = s_nuc

            # nuclear overlap matrix (excluding electronic component)
            self.mat_dict['s_elec'][i,j] = s_elec

            # compute overlap of trajectories (different from S, which may or may
            # not involve integration in a gaussian basis
            self.mat_dict['s_traj'][i,j] = integrals.traj_overlap(wfn.traj[ii],wfn.traj[jj])

            # overlap matrix (including electronic component)
            self.mat_dict['s'][i,j]      = integrals.s_integral(wfn.traj[ii],wfn.traj[jj],
                                                                nuc_ovrlp=s_nuc, elec_ovrlp=s_elec)

            # time-derivative of the overlap matrix (not hermitian in general)
            self.mat_dict['sdot'][i,j]   = integrals.sdot_integral(wfn.traj[ii],wfn.traj[jj],
                                                                   nuc_ovrlp=s_nuc, elec_ovrlp=s_elec)

            # kinetic energy matrix
            self.mat_dict['t'][i,j]      = integrals.t_integral(wfn.traj[ii],wfn.traj[jj],
                                                                nuc_ovrlp=s_nuc, elec_ovrlp=s_elec)

            # potential energy matrix
            self.mat_dict['v'][i,j]      = integrals.v_integral(wfn.traj[ii],wfn.traj[jj],
                                                                nuc_ovrlp=s_nuc, elec_ovrlp=s_elec)

            # Hamiltonian matrix in non-orthogonal basis
            self.mat_dict['h'][i,j]      = self.mat_dict['t'][i,j] + self.mat_dict['v'][i,j]

            # if hermitian matrix, set (j,i) indices
            if integrals.hermitian and i!=j:
                self.mat_dict['s_nuc'][j,i]   = self.mat_dict['s_nuc'][i,j].conjugate()
                self.mat_dict['s_elec'][j,i]  = self.mat_dict['s_elec'][i,j].conjugate()
                self.mat_dict['s_traj'][j,i]  = self.mat_dict['s_traj'][i,j].conjugate()
                self.mat_dict['s'][j,i]       = self.mat_dict['s'][i,j].conjugate()
                self.mat_dict['sdot'][j,i]    = integrals.sdot_integral(wfn.traj[jj], wfn.traj[ii],
                                                nuc_ovrlp=self.mat_dict['s_nuc'][j,i],
                                                elec_ovrlp=self.mat_dict['s_elec'][j,i])
                self.mat_dict['t'][j,i]       = self.mat_dict['t'][i,j].conjugate()
                self.mat_dict['v'][j,i]       = self.mat_dict['v'][i,j].conjugate()
                self.mat_dict['h'][j,i]       = self.mat_dict['h'][i,j].conjugate()

        if integrals.hermitian:
            # compute the S^-1, needed to compute Heff
            timings.start('linalg.pinvh')
            self.mat_dict['sinv'] = sp_linalg.pinvh(self.mat_dict['s'])
            #Sinv, cond = linalg.pseudo_inverse2(S)
            timings.stop('linalg.pinvh')
        else:
            # compute the S^-1, needed to compute Heff
            timings.start('hamiltonian.pseudo_inverse')
            self.mat_dict['sinv'], cond = linalg.pseudo_inverse(self.mat_dict['s'])
            timings.stop('hamiltonian.pseudo_inverse')

        self.mat_dict['heff'] = np.dot( self.mat_dict['sinv'], self.mat_dict['h'] - 1j * self.mat_dict['sdot'] )

    def ut_ind(self,index):
        """Gets the (i,j) index of an upper triangular matrix from the
        sequential matrix index 'index'"""
        i = 0
        while i*(i+1)/2 - 1 < index:
            i += 1
        return index - i*(i-1)//2, i-1

    def sq_ind(self,index, n):
        """Gets the (i,j) index of a square matrix from the
        sequential matrix index 'index'"""
        return index // n, index % n
