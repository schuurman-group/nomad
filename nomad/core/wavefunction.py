"""
The Wavefunction object and its associated functions.
"""
import copy
import sys as sys
import numpy as np
import scipy.linalg as sp_linalg
import nomad.core.timings as timings
import nomad.core.matrices as matrices


class Wavefunction:
    """Class constructor for the Wavefunction object."""
    def __init__(self):
        self.time      = float(0.)
        self.nalive    = 0
        self.nactive   = 0
        self.ndead     = 0
        self.stpop     = None 
        self.wfn_norm  = 0.
        self.traj      = []
        self.alive     = []
        self.active    = []
        self.matrices  = matrices.Matrices() 

    @timings.timed
    def copy(self):
        """Copys a Wavefunction object with new references.

        This method is the simplest way i can see to make a copy
        of a wfn with new references. Overriding deepcopy for
        significantly more work.
        """
        new_wfn          = Wavefunction()
        new_wfn.time     = copy.copy(self.time)
        new_wfn.nalive   = copy.copy(self.nalive)
        new_wfn.ndead    = copy.copy(self.ndead)
        new_wfn.stpop    = copy.deepcopy(self.stpop)
        new_wfn.alive    = copy.deepcopy(self.alive)
        new_wfn.active   = copy.deepcopy(self.active)
        if self.matrices is not None:
            new_wfn.matrices = self.matrices.copy()

        # copy the trajectory array
        for i in range(self.n_traj()):
            new_wfn.traj.append(self.traj[i].copy())

        return new_wfn

    def n_traj(self):
        """Returns total number of trajectories."""
        return self.nalive + self.ndead

    @timings.timed
    def add_trajectory(self, new_traj, relabel=True):
        """Adds a trajectory to the wfn."""
        self.nalive       += 1
        self.nactive      += 1
        new_traj.alive     = True
        new_traj.active    = True
        if relabel:
            new_traj.label = self.n_traj() - 1
        self.alive.append(self.n_traj()-1)
        self.active.append(self.n_traj()-1)
        self.traj.append(new_traj)

    def add_trajectories(self, traj_list, relabel=True):
        """Adds a set of trajectories to the wfn."""
        for new_traj in traj_list:
            self.nalive         += 1
            self.nactive        += 1
            new_traj.alive       = True
            new_traj.active      = True
            if relabel:
                new_traj.label   = self.n_traj() - 1
            self.alive.append(self.n_traj()-1)
            self.active.append(self.n_traj()-1)
            self.traj.append(new_traj)

    @timings.timed
    def kill_trajectory(self, label):
        """Moves a live trajectory to the list of dead trajecotries.

        The trajectory will no longer contribute to H, S, etc.
        """
        # Remove the trajectory from the list of living trajectories
        indx = [self.traj[i].label for i in range(self.n_traj())].index(label)
        try:
            self.alive.remove(indx)
            self.traj[indx].alive = False
            self.nalive           = self.nalive - 1
            self.ndead            = self.ndead + 1
        except ValueError:
            sys.exit('Cannot kill trajectory, label not found')   

    @timings.timed
    def revive_trajectory(self, label):
        """
        Moves a dead trajectory to the list of live trajectories.
        """
        indx = [self.traj[i].label for i in range(self.n_traj())].index(label)

        try:
            self.traj[indx].alive = True
            # Add the trajectory to the list of living trajectories
            self.alive.insert(indx,label)
            self.nalive          = self.nalive + 1
            self.ndead           = self.ndead - 1
        except ValueError:
            sys.exit('Cannot revive trajectory, label not found')

    @timings.timed
    def update_matrices(self, mats):
        """Documentation to come"""
        self.matrices = mats.copy()
 
        # I don't think these should be here (should only
        # call update when amplitudes updated), but will
        # keep here for now to be safe
        self.update_pop()
        self.update_norm()
        return


    @timings.timed
    def update_amplitudes(self, dt, Ct=None):
        """Updates the amplitudes of the trajectory in the wfn.
        Solves d/dt C = -i H C via the computation of
        exp(-i H(t) dt) C(t)."""

        # update the amplitudes
        if Ct is None:
            old_amp = self.amplitudes()
        else:
            old_amp = Ct

        new_amp = np.zeros(self.nalive, dtype=complex)
        B       = -1j * self.matrices.matrix['heff'] * dt
        umat    = sp_linalg.expm(B)
        new_amp = np.dot(umat, old_amp)

        for i in range(len(self.alive)):
            self.traj[self.alive[i]].update_amplitude(new_amp[i])

        # udpate the corresponding state populations
        self.update_pop()
        self.update_norm()
        return

    @timings.timed
    def update_pop(self):
        """Updates the populations"""
        # udpate the corresponding state populations

        if 'popwt' in self.matrices.avail():
            nst    = self.traj[0].nstates
            zpop   = np.zeros(nst, dtype=complex)

            for ib in range(self.nalive):
                tb = self.traj[self.alive[ib]]
                ab = tb.amplitude.conjugate()
                for ik in range(self.nalive):
                    tk = self.traj[self.alive[ik]]
                    ak = tk.amplitude
                    zpop += self.matrices.matrix['popwt'][ib,ik,:]*ab*ak

            #print('popwt='+str(self.matrices.matrix['popwt'][0,0,:]))
            #print('zpop='+str(zpop))
            if np.linalg.norm(zpop.imag) > 1.e-4:
                print('warning: imaginary component of population: '
                        + str(zpop.imag))
            self.stpop = zpop.real / sum(zpop.real)
        return

    @timings.timed
    def update_norm(self):
        """Updates the wavefunction norm using current s_traj matrix"""

        if 's_traj' in self.matrices.avail():
            self.wfn_norm = np.absolute( np.dot(np.dot(
                          self.amplitudes().conjugate(),
                          self.matrices.matrix['s_traj']),
                          self.amplitudes()) )
        return

    @timings.timed
    def renormalize(self):
        """Renormalizes the amplitudes of the trajectories in the wfn."""
        self.update_norm()
        norm_factor = 1. / np.sqrt(self.norm())

        for i in range(self.n_traj()):
            self.traj[i].update_amplitude(self.traj[i].amplitude * norm_factor)
       
        self.update_pop()
        self.update_norm()


    def amplitudes(self):
        """Returns amplitudes of the trajectories."""
        return np.array([self.traj[self.alive[i]].amplitude
                         for i in range(len(self.alive))], dtype=complex)

    def set_amplitudes(self, amps):
        """Sets the value of the amplitudes."""
        for i in range(self.nalive):
            self.traj[self.alive[i]].amplitude = amps[i]

    def pop(self):
        """Returns the total population on each state using the
           contents of matrices['popwt'] to determine contributions
           of each trajectory pair to each electronic state"""

        return self.stpop

    def norm(self):
        """Returns the norm of the wavefunction """
        return self.wfn_norm

    @timings.timed
    def pot_classical(self):
        """Returns the classical potential energy of the wfn.

        Currently only includes energy from alive trajectories
        """
        nalive  = len(self.alive)
        pot_vec = np.array([self.traj[self.alive[i]].potential()
                            for i in range(nalive)])
        return sum(pot_vec)/nalive

    @timings.timed
    def pot_quantum(self):
        """Returns the QM (coupled) potential energy of the wfn.
        Currently includes <live|live> (not <dead|dead>,etc,) contributions...
        """
        if 'v' not in self.matrices.avail():
            return 0.
        
        return np.dot(np.dot(np.conj(self.amplitudes()),
                             self.matrices.matrix['v']), self.amplitudes()).real

    @timings.timed
    def kin_classical(self):
        """Returns the classical kinetic energy of the wfn."""
        nalive   = len(self.alive)
        kecoef   = self.traj[0].kecoef # horrible. messy.
        ke_vec   = np.array([np.dot(self.traj[self.alive[i]].p()**2, kecoef)
                           for i in range(nalive)])
        return sum(ke_vec)/nalive

    @timings.timed
    def kin_quantum(self):
        """Returns the QM (coupled) kinetic energy of the wfn."""

        if 't' not in self.matrices.avail():
            return 0.

        return np.dot(np.dot(np.conj(self.amplitudes()),
                             self.matrices.matrix['t']), self.amplitudes()).real
        #Sinv = sp_linalg.pinv(self.S)
        #return np.dot(np.dot(np.conj(self.amplitudes()),
        #                     np.dot(Sinv,self.T)),self.amplitudes()).reali

    def tot_classical(self):
        """returns the total classical energy"""
        return self.kin_classical() + self.pot_classical()

    def tot_quantum(self):
        """returns the total quantum energy"""
        return self.kin_quantum() + self.pot_quantum()



