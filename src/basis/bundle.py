"""
The Bundle object and its associated functions.
"""
import sys
import copy
import numpy as np
import scipy.linalg as sp_linalg
import src.utils.timings as timings
import src.parse.glbl as glbl
import src.basis.centroid as centroid
import src.dynamics.hamiltonian as fms_ham

class Bundle:
    """Class constructor for the Bundle object."""
    def __init__(self, nstates):
        self.time      = 0.
        self.nalive    = 0
        self.nactive   = 0
        self.ndead     = 0
        self.nstates   = int(nstates)
        self.traj      = []
        self.cent      = []
        self.alive     = []
        self.active    = []
        self.T         = np.zeros((0, 0), dtype=complex)
        self.V         = np.zeros((0, 0), dtype=complex)
        self.S         = np.zeros((0, 0), dtype=complex)
        self.Snuc      = np.zeros((0, 0), dtype=complex)
        self.Sdot      = np.zeros((0, 0), dtype=complex)
        self.Heff      = np.zeros((0, 0), dtype=complex)
        self.traj_ovrlp= np.zeros((0, 0), dtype=complex)
#        try:
#            self.integrals=__import__('src.integrals.'+
#                                       self.integral_type,fromlist=['a'])
#        except ImportError:
#            print('BUNDLE INIT FAIL: src.integrals.'+self.integral_type)

    @timings.timed
    def copy(self):
        """Copys a Bundle object with new references.

        This method is the simplest way i can see to make a copy
        of a bundle with new references. Overriding deepcopy for
        significantly more work.
        """
        new_bundle = Bundle(self.nstates)
        new_bundle.time   = copy.copy(self.time)
        new_bundle.nalive = copy.copy(self.nalive)
        new_bundle.ndead  = copy.copy(self.ndead)
        new_bundle.alive  = copy.deepcopy(self.alive)
        new_bundle.T      = copy.deepcopy(self.T)
        new_bundle.V      = copy.deepcopy(self.V)
        new_bundle.S      = copy.deepcopy(self.S)
        new_bundle.Snuc   = copy.deepcopy(self.Snuc)
        new_bundle.Sdot   = copy.deepcopy(self.Sdot)
        new_bundle.Heff   = copy.deepcopy(self.Heff)
        new_bundle.traj_ovrlp = copy.deepcopy(self.traj_ovrlp)

        # copy the trajectory array
        for i in range(self.n_traj()):
            traj_i = self.traj[i].copy()
            new_bundle.traj.append(traj_i)

        # copy the centroid matrix
#        if new_bundle.integrals.require_centroids:
        if glbl.integrals.require_centroids:
            for i in range(len(self.cent)):
                new_bundle.cent.append([None for j in range(self.n_traj())])
                for j in range(i):
                    new_bundle.cent[i][j] = self.cent[i][j].copy()
                    new_bundle.cent[j][i] = self.cent[j][i].copy()

        return new_bundle

    def n_traj(self):
        """Returns total number of trajectories."""
        return self.nalive + self.ndead

    @timings.timed
    def add_trajectory(self, new_traj):
        """Adds a trajectory to the bundle."""
        self.nalive         += 1
        self.nactive        += 1
        new_traj.alive       = True
        new_traj.active      = True
        new_traj.label       = self.n_traj() - 1
        self.alive.append(new_traj.label)
        self.active.append(new_traj.label)
        self.traj.append(new_traj)
        self.T       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Snuc    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.traj_ovrlp = np.zeros((self.nalive, self.nalive), dtype=complex)
#        if self.integrals.require_centroids:
        if glbl.integrals.require_centroids:
            self.create_centroids()

    def add_trajectories(self, traj_list):
        """Adds a set of trajectories to the bundle."""
        for new_traj in traj_list:
            self.nalive         += 1
            self.nactive        += 1
            new_traj.alive       = True
            new_traj.active      = True
            new_traj.label       = self.n_traj() - 1
            self.alive.append(new_traj.label)
            self.active.append(new_traj.label)
            self.traj.append(new_traj)

        self.T       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Snuc    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.traj_ovrlp = np.zeros((self.nalive, self.nalive), dtype=complex)
#        if self.integrals.require_centroids:
        if glbl.integrals.require_centroids:
            self.create_centroids()

    @timings.timed
    def kill_trajectory(self, label):
        """Moves a live trajectory to the list of dead trajecotries.

        The trajectory will no longer contribute to H, S, etc.
        """
        # Remove the trajectory from the list of living trajectories
        self.alive.remove(label)
        self.traj[label].alive = False
        self.nalive          = self.nalive - 1
        self.ndead           = self.ndead + 1
        # Remove the trajectory from the list of active trajectories
        # iff matching pursuit is not being used
        if not glbl.propagate['matching_pursuit']:
            self.active.remove(tid)
            self.traj[tid].active = False
            self.nactive = self.nactive - 1
        # Reset arrays
        self.T       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Snuc    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.traj_ovrlp = np.zeros((self.nalive, self.nalive), dtype=complex)

    @timings.timed
    def revive_trajectory(self, label):
        """
        Moves a dead trajectory to the list of live trajectories.
        """
        self.traj[label].alive = True

        # Add the trajectory to the list of living trajectories
        self.alive.insert(label,label)

        self.nalive          = self.nalive + 1
        self.ndead           = self.ndead - 1

        self.T       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Snuc    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.traj_ovrlp = np.zeros((self.nalive, self.nalive), dtype=complex)

    @timings.timed
    def update_amplitudes(self, dt, update_ham=True, H=None, Ct=None):
        """Updates the amplitudes of the trajectory in the bundle.
        Solves d/dt C = -i H C via the computation of
        exp(-i H(t) dt) C(t)."""
        if update_ham:
            self.update_matrices()

        # if no Hamiltonian is passed, use the current effective
        # Hamiltonian
        if H is None:
            Hmat = self.Heff
        else:
            Hmat = H

        # if no vector of amplitudes are supplied (to propagate),
        # propagate the current amplitudes
        if Ct is None:
            old_amp = self.amplitudes()
        else:
            old_amp = Ct

        new_amp = np.zeros(self.nalive, dtype=complex)

        B = -1j * Hmat * dt

        if self.nalive < 150:
            # Eigen-decomposition
            umat = sp_linalg.expm2(B)
        else:
            # Pade approximation
            umat = sp_linalg.expm(B)

        new_amp = np.dot(umat, old_amp)

        for i in range(len(self.alive)):
            self.traj[self.alive[i]].update_amplitude(new_amp[i])

    @timings.timed
    def centroid_required(self, traj_i, traj_j):
        """Determine if centroid is required for integral. Data at centroid will
           NOT be computed if:
           a) the nuclear overlap between trajectories is below threshold
        """
#        if not self.integrals.require_centroids:
        if not glbl.integrals.require_centroids:
            return False
        else:
            return True

    @timings.timed
    def renormalize(self):
        """Renormalizes the amplitudes of the trajectories in the bundle."""
        norm_factor = 1. / np.sqrt(self.norm())
        for i in range(self.n_traj()):
            self.traj[i].update_amplitude(self.traj[i].amplitude * norm_factor)

    def prune(self):
        """Kills trajectories that are dead."""
        for i in range(self.nalive):
            continue
        return False

    def amplitudes(self):
        """Returns amplitudes of the trajectories."""
        return np.array([self.traj[self.alive[i]].amplitude
                         for i in range(len(self.alive))], dtype=complex)

    def set_amplitudes(self, amps):
        """Sets the value of the amplitudes."""
        for i in range(self.nalive):
            self.traj[self.alive[i]].amplitude = amps[i]

    def mulliken_pop(self, label):
        """Returns the Mulliken-like population."""
        mulliken = 0.

        if not self.traj[label].alive:
            return mulliken
        i = self.alive.index(label)
        for j in range(len(self.alive)):
            jj = self.alive[j]
            mulliken += abs(self.traj_ovrlp[i,j] *
                            self.traj[label].amplitude.conjugate() *
                            self.traj[jj].amplitude)
        return mulliken

    @timings.timed
    def norm(self):
        """Returns the norm of the wavefunction """
        return np.dot(np.dot(np.conj(self.amplitudes()),
                      self.S),self.amplitudes()).real

    @timings.timed
    def pop(self):
        """Returns the populations on each of the states."""
        pop    = np.zeros(self.nstates, dtype=complex)
        nalive = len(self.alive)

        # live contribution
        for i in range(nalive):
            ii = self.alive[i]
            state = self.traj[ii].state
            for j in range(nalive):
                jj = self.alive[j]
                if self.traj[jj].state != state:
                    continue
                popij = (self.traj_ovrlp[i,j]  *
                         self.traj[jj].amplitude *
                         self.traj[ii].amplitude.conjugate())
                pop[state] += popij

        pop /= sum(pop)

        return pop.real

    @timings.timed
    def pot_classical(self):
        """Returns the classical potential energy of the bundle.

        Currently only includes energy from alive trajectories
        """
        nalive  = len(self.alive)
        pot_vec = np.array([self.traj[self.alive[i]].potential()
                            for i in range(nalive)])
        return sum(pot_vec)/nalive

    @timings.timed
    def pot_quantum(self):
        """Returns the QM (coupled) potential energy of the bundle.
        Currently includes <live|live> (not <dead|dead>,etc,) contributions...
        """
        return np.dot(np.dot(np.conj(self.amplitudes()),
                             self.V), self.amplitudes()).real
        #Sinv = sp_linalg.pinv(self.S)
        #return np.dot(np.dot(np.conj(self.amplitudes()),
        #                     np.dot(Sinv,self.V)),self.amplitudes()).real

    @timings.timed
    def kin_classical(self):
        """Returns the classical kinetic energy of the bundle."""
        nalive   = len(self.alive)
        kecoef   = glbl.interface.kecoeff
        ke_vec   = np.array([np.dot(self.traj[self.alive[i]].p()**2, kecoef)
                           for i in range(nalive)])
        return sum(ke_vec)/nalive

    @timings.timed
    def kin_quantum(self):
        """Returns the QM (coupled) kinetic energy of the bundle."""
        return np.dot(np.dot(np.conj(self.amplitudes()),
                             self.T), self.amplitudes()).real
        #Sinv = sp_linalg.pinv(self.S)
        #return np.dot(np.dot(np.conj(self.amplitudes()),
        #                     np.dot(Sinv,self.T)),self.amplitudes()).real

    def tot_classical(self):
        """Returns the total classical energy of the bundle."""
        return self.pot_classical() + self.kin_classical()

    def tot_quantum(self):
        """Returns the total QM (coupled) energy of the bundle."""
        return self.pot_quantum() + self.kin_quantum()

    def overlap(self, other):
        """Returns the overlap integral of the bundle with another
        bundle."""
        S = 0.
        for i in range(self.nalive):
            for j in range(other.nalive):
                ii = self.alive[i]
                jj = other.alive[j]
                S += (glbl.integrals.traj_overlap(self.traj[ii], other.traj[jj]) *
                      self.traj[ii].amplitude.conjugate() *
                      other.traj[jj].amplitude)
#                S += (self.integrals.traj_overlap(self.traj[ii], other.traj[jj]) *
#                      self.traj[ii].amplitude.conjugate() *
#                      other.traj[jj].amplitude)
        return S

    def overlap_traj(self, traj):
        """Returns the overlap of the bundle with a trajectory (assumes the
        amplitude on the trial trajectory is (1.,0.)"""
        ovrlp = 0j
        for i in range(self.nalive+self.ndead):
            ovrlp += (glbl.integrals.traj_overlap(traj,self.traj[i]) *
                      self.traj[i].amplitude)
#            ovrlp += (self.integrals.traj_overlap(traj,self.traj[i]) *
#                      self.traj[i].amplitude)
        return ovrlp

    #----------------------------------------------------------------------
    #
    # Private methods/functions (called only within the class)
    #
    #----------------------------------------------------------------------
    @timings.timed
    def create_centroids(self):
        """Increases the centroid 'matrix' to account for new basis functions.

        Called by add_trajectory. Make sure centroid array has sufficient
        space to hold required centroids. Note that n_traj includes alive
        AND dead trajectories -- therefore it can only increase. So, only
        need to check n_traj > dim_cent condition.
        """
        dim_cent = len(self.cent)

        # number of centroids already correct
        if self.n_traj() == dim_cent:
            return

        # n_traj includes living and dead -- this condition should never be satisfied
       # if self.n_traj() < dim_cent:
            raise ValueError('n_traj() < dim_cent in bundle. Exiting...')

        # ...else we need to add more centroids
        if self.n_traj() > dim_cent:
            for i in range(dim_cent):
                self.cent[i].extend([None for j in range(self.n_traj() -
                                                         dim_cent)])

            for i in range(self.n_traj() - dim_cent):
                self.cent.append([None for j in range(self.n_traj())])

        for i in range(self.n_traj()):
            if self.traj[i].alive:
                for j in range(i):
                    if self.traj[j].alive:
                        # now check to see if needed index has an existing trajectory
                        # if not, copy trajectory from one of the parents into the
                        # required slots
                        if self.cent[i][j] is None:
                            self.cent[i][j] = centroid.Centroid(traj_i=self.traj[i],
                                                                traj_j=self.traj[j])
                            self.cent[j][i] = self.cent[i][j]

    @timings.timed
    def update_centroids(self):
        """Updates the positions of the centroids."""

        for i in range(self.n_traj()):
            if self.traj[i].alive:
                for j in range(i):
                    if self.traj[j].alive:
                        self.cent[i][j].update_x(self.traj[i],self.traj[j])
                        self.cent[j][i].update_x(self.traj[j],self.traj[i])
                        self.cent[i][j].update_p(self.traj[i],self.traj[j])
                        self.cent[j][i].update_p(self.traj[j],self.traj[i])

    @timings.timed
    def update_matrices(self):
        """Updates T, V, S, Sdot and Heff matrices."""
        # make sure the centroids are up-to-date in order to evaluate
        # self.H -- if we need them

#        if self.integrals.require_centroids:
#            (self.traj_ovrlp, self.T, self.V, self.S, self.Snuc, self.Sdot,
#             self.Heff) = fms_ham.hamiltonian(self.integrals, self.traj, self.alive,
#                                              cent_list=self.cent)
        if glbl.integrals.require_centroids:
            (self.traj_ovrlp, self.T, self.V, self.S, self.Snuc, self.Sdot,
             self.Heff) = fms_ham.hamiltonian(self.traj, self.alive,
                                              cent_list=self.cent)
        else:
#            (self.traj_ovrlp, self.T, self.V, self.S, self.Snuc, self.Sdot,
#             self.Heff) = fms_ham.hamiltonian(self.integrals, self.traj, self.alive)
            (self.traj_ovrlp, self.T, self.V, self.S, self.Snuc, self.Sdot,
             self.Heff) = fms_ham.hamiltonian(self.traj, self.alive)

