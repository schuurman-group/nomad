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
import nomad.common.linalg as linalg
import nomad.core.timings as timings


class Matrices:
    """Object containing the Hamiltonian and associated matrices."""
    def __init__(self):
        self.mat_list = ['s','sinv','s_traj','s_nuc','s_elec',
                         't',   'v',     'h', 'sdot',  'heff']
        self.matrix = dict()

    def set(self, name, matrix):
        """Sets the matrices in the current matrix object equal
        to the passed matrix object"""
        if name in self.mat_list:
            self.matrix[name] = copy.deepcopy(matrix)

    def avail(self):
        """returns the list of matrices included in matrix"""
        return self.matrix.keys()

    def copy(self):
        """Documentation to come"""
        new_matrices = Matrices()

        for typ,mat in self.matrix.items():
            new_matrices.set(typ, mat)

        return new_matrices

    @timings.timed
    def build(self, bra, ket, integrals, hermitian=False):
        """Builds the Hamiltonian matrix from a list of trajectories."""

        if (bra.nalive != ket.nalive or 
            bra.traj[0].nstates != ket.traj[0].nstates):
            sys.exit('error buiding matrices - '+
                     'bra/ket wfns not consistent')

        n_alive = bra.nalive
        n_st    = bra.traj[0].nstates
        m_shape = (n_alive, n_alive)

        if integrals.hermitian and hermitian:
            n_elem  = int(n_alive * (n_alive + 1) / 2)
        else:
            n_elem  = n_alive * n_alive

        if (any([mat.shape != m_shape for typ,mat in self.matrix.items()]) or
            len(self.avail()) != len(self.mat_list)):
            self.matrix['t']       = np.zeros(m_shape, dtype=complex)
            self.matrix['v']       = np.zeros(m_shape, dtype=complex)
            self.matrix['h']       = np.zeros(m_shape, dtype=complex)
            self.matrix['s_traj']  = np.zeros(m_shape, dtype=complex)
            self.matrix['s_nuc']   = np.zeros(m_shape, dtype=complex)
            self.matrix['s_elec']  = np.zeros(m_shape, dtype=complex)
            self.matrix['s']       = np.zeros(m_shape, dtype=complex)
            self.matrix['sinv']    = np.zeros(m_shape, dtype=complex)
            self.matrix['sdot']    = np.zeros(m_shape, dtype=complex)
            self.matrix['heff']    = np.zeros(m_shape, dtype=complex)

        # now evaluate the hamiltonian matrix
        for ij in range(n_elem):
            if integrals.hermitian and hermitian:
                i, j = self.ut_ind(ij)
            else:
                i, j = self.sq_ind(ij, n_alive)

            ii = bra.alive[i]
            jj = ket.alive[j]

            s_nuc  = integrals.nuc_overlap(bra.traj[ii],ket.traj[jj])
            s_elec = integrals.elec_overlap(bra.traj[ii],ket.traj[jj])

            # nuclear overlap matrix (excluding electronic component)
            self.matrix['s_nuc'][i,j]  = s_nuc

            # nuclear overlap matrix (excluding electronic component)
            self.matrix['s_elec'][i,j] = s_elec

            # compute overlap of trajectories (different from S, which may or may
            # not involve integration in a gaussian basis
            self.matrix['s_traj'][i,j] = integrals.traj_overlap(bra.traj[ii], ket.traj[jj])

            # overlap matrix (including electronic component)
            self.matrix['s'][i,j]      = integrals.s_integral(bra.traj[ii], ket.traj[jj],
                                                                nuc_ovrlp=s_nuc, elec_ovrlp=s_elec)

            # time-derivative of the overlap matrix (not hermitian in general)
            self.matrix['sdot'][i,j]   = integrals.sdot_integral(bra.traj[ii], ket.traj[jj],
                                                                nuc_ovrlp=s_nuc, elec_ovrlp=s_elec)

            # kinetic energy matrix
            self.matrix['t'][i,j]      = integrals.t_integral(bra.traj[ii], ket.traj[jj],
                                                                nuc_ovrlp=s_nuc, elec_ovrlp=s_elec)

            # potential energy matrix
            self.matrix['v'][i,j]      = integrals.v_integral(bra.traj[ii], ket.traj[jj],
                                                                nuc_ovrlp=s_nuc, elec_ovrlp=s_elec)

            # Hamiltonian matrix in non-orthogonal basis
            self.matrix['h'][i,j]      = self.matrix['t'][i,j] + self.matrix['v'][i,j]


            # if hermitian matrix, set (j,i) indices
            if integrals.hermitian and hermitian and i!=j:
                self.matrix['s_nuc'][j,i]   = self.matrix['s_nuc'][i,j].conjugate()
                self.matrix['s_elec'][j,i]  = self.matrix['s_elec'][i,j].conjugate()
                self.matrix['s_traj'][j,i]  = self.matrix['s_traj'][i,j].conjugate()
                self.matrix['s'][j,i]       = self.matrix['s'][i,j].conjugate()
                self.matrix['sdot'][j,i]    = integrals.sdot_integral(bra.traj[jj], ket.traj[ii],
                                                nuc_ovrlp=self.matrix['s_nuc'][j,i],
                                                elec_ovrlp=self.matrix['s_elec'][j,i])
                self.matrix['t'][j,i]       = self.matrix['t'][i,j].conjugate()
                self.matrix['v'][j,i]       = self.matrix['v'][i,j].conjugate()
                self.matrix['h'][j,i]       = self.matrix['h'][i,j].conjugate()

        # TEMP *********************************************************
        #if hermitian:
        #    for i in range(bra.n_traj()):
        #        with open('theta'+str(bra.traj[i].label)+'.dat', 'a') as f:
        #            f.write(str(bra.time)+' '+str(integrals.ints.theta(bra.traj[i]))+'\n')
        #
        #        with open('dtheta'+str(bra.traj[i].label)+'.dat','a') as f:
        #            dtheta = integrals.ints.dtheta(bra.traj[i])
        #            f.write(str(bra.time)+' '+str(dtheta[0])+' '+str(dtheta[1])+'\n')
        #
        #        with open('phi'+str(bra.traj[i].label)+'.dat', 'a') as f:
        #            phi = integrals.ints.phi(bra.traj[i])
        #            f.write(str(bra.time)+' '+str(phi[0])+' '+str(phi[1])+'\n')
        #
        #        with open('dphi'+str(bra.traj[i].label)+'.dat', 'a') as f:
        #            dphi = integrals.ints.dphi(bra.traj[i])
        #            f.write(str(bra.time)+' '+str(dphi[0])+' '+str(dphi[1])+'\n')
        #
        # **************************************************************

        if integrals.hermitian:
            # compute the S^-1, needed to compute Heff
            timings.start('linalg.pinvh')
            self.matrix['sinv'] = sp_linalg.pinvh(self.matrix['s'])
            #Sinv, cond = linalg.pseudo_inverse2(S)
            timings.stop('linalg.pinvh')
        else:
            # compute the S^-1, needed to compute Heff
            timings.start('hamiltonian.pseudo_inverse')
            self.matrix['sinv'], cond = linalg.pseudo_inverse(self.matrix['s'])
            timings.stop('hamiltonian.pseudo_inverse')

        self.matrix['heff'] = np.dot( self.matrix['sinv'], self.matrix['h'] - 1.j * self.matrix['sdot'] )

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
