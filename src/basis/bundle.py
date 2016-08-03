"""
The Bundle object and its associated functions.
"""
import sys
import copy
import numpy as np
from scipy import linalg
from src.dynamics import timings
from src.fmsio import glbl as glbl
from src.fmsio import fileio as fileio
from src.basis import particle as particle
from src.basis import trajectory as trajectory
from src.basis import build_hamiltonian as mbuild


def copy_bundle(orig_bundle):
    """Copys a Bundle object with new references.

    This method is the simplest way i can see to make a copy
    of a bundle with new references. Overriding deepcopy for
    significantly more work.
    """
    timings.start('bundle.copy_bundle')

    new_bundle = Bundle(orig_bundle.nstates, orig_bundle.integrals)
    new_bundle.time   = copy.copy(orig_bundle.time)
    new_bundle.nalive = copy.copy(orig_bundle.nalive)
    new_bundle.ndead  = copy.copy(orig_bundle.ndead)
    new_bundle.alive  = copy.deepcopy(orig_bundle.alive)
    new_bundle.T      = copy.deepcopy(orig_bundle.T)
    new_bundle.V      = copy.deepcopy(orig_bundle.V)
    new_bundle.S      = copy.deepcopy(orig_bundle.S)
    new_bundle.Sdot   = copy.deepcopy(orig_bundle.Sdot)
    new_bundle.Heff   = copy.deepcopy(orig_bundle.Heff)
    for i in range(new_bundle.n_traj()):
        traj_i = trajectory.copy_traj(orig_bundle.traj[i])
        new_bundle.traj.append(traj_i)
    for i in range(len(orig_bundle.cent)):
        if orig_bundle.cent[i] is None:
            traj_i = None
        else:
            traj_i = trajectory.copy_traj(orig_bundle.cent[i])
        new_bundle.cent.append(traj_i)

    timings.stop('bundle.copy_bundle')

    return new_bundle


def cent_ind(i, j):
    """Returns the index in the cent array of the centroid between
    trajectories i and j."""
    if i == j:
        return -1
    else:
        a = max(i,j)
        b = min(i,j)
        return int(a*(a-1)/2 + b)


class Bundle:
    """Class constructor for the Bundle object."""
    def __init__(self, nstates, int_defs):
        self.integrals = int_defs
        self.time = 0.
        self.nalive = 0
        self.ndead = 0
        self.nstates = int(nstates)
        self.traj  = []
        self.cent  = []
        self.alive = []
        self.T     = np.zeros((0, 0), dtype=complex)
        self.V     = np.zeros((0, 0), dtype=complex)
        self.S     = np.zeros((0, 0), dtype=complex)
        self.Sdot  = np.zeros((0, 0), dtype=complex)
        self.Heff  = np.zeros((0, 0), dtype=complex)
        try:
            self.ints = __import__('src.integrals.' + self.integrals,
                                   fromlist=['a'])
        except ImportError:
            print('BUNDLE INIT FAIL: src.integrals.' + self.integrals)

        self.hambuild = __import__('src.basis.build_hamiltonian_' +
                                   self.ints.hamsym , fromlist=['a'])

    def n_traj(self):
        """Returns total number of trajectories."""
        return self.nalive + self.ndead

    def n_cent(self):
        """Returns length of centroid array."""
        return len(self.cent)

    def cent_len(self, n):
        """Returns length of centroid array for n_traj length
        traj array."""
        n_traj=n+1
        if n_traj == 0:
            return 0
        return int(n_traj * (n_traj - 1) / 2)

    def add_trajectory(self, new_traj):
        """Adds a trajectory to the bundle."""
        timings.start('bundle.add_trajectory')
        self.traj.append(new_traj)
        self.traj[-1].alive = True
        self.nalive        += 1
        self.traj[-1].tid   = self.n_traj() - 1
        self.alive.append(self.traj[-1].tid)
        self.T    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff = np.zeros((self.nalive, self.nalive), dtype=complex)
        timings.stop('bundle.add_trajectory')

    def add_trajectories(self, traj_list):
        """Adds a set of trajectories to the bundle."""
        for traj in traj_list:
            self.traj.append(traj)
            self.nalive        += 1
            self.traj[-1].alive = True
            self.traj[-1].tid   = self.n_traj() - 1
            self.alive.append(self.traj[-1].tid)
        self.T    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff = np.zeros((self.nalive, self.nalive), dtype=complex)

    def kill_trajectory(self, tid):
        """Moves a live trajectory to the list of dead trajecotries.

        The trajectory will no longer contribute to H, S, etc.
        """
        timings.start('bundle.kill_trajectory')
        self.traj[tid].alive = False
        self.nalive          = self.nalive - 1
        self.ndead           = self.ndead + 1
        self.alive.pop(tid)
        self.T    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff = np.zeros((self.nalive, self.nalive), dtype=complex)
        timings.start('bundle.kill_trajectory')

    def update_amplitudes(self, dt, n_max, H=None, Ct=None):
        """Updates the amplitudes of the trajectory in the bundle.
        Solves d/dt C = -i H C via the computation of
        exp(-i H(t) dt) C(t)."""

        timings.start('bundle.update_amplitudes')

        self.update_matrices()

        if H is not None:
            Hmat = H
        else:
            Hmat = self.Heff

        if Ct is not None:
            old_amp = Ct
        else:
            old_amp = self.amplitudes()

        new_amp = np.zeros(self.nalive, dtype=complex)

        B = -complex(0.,1.) * Hmat * dt

        if self.nalive < 150:
            # Eigen-decomposition
            umat = linalg.expm2(B)
        else:
            # Pade approximation
            umat = linalg.expm(B)

        new_amp = np.dot(umat, old_amp)

        for i in range(len(self.alive)):
            self.traj[self.alive[i]].update_amplitude(new_amp[i])

        timings.stop('bundle.update_amplitudes')

    def renormalize(self):
        """Renormalizes the amplitudes of the trajectories in the bundle."""
        timings.start('bundle.renormalize')
        current_pop = self.pop()
        norm = 1. / np.sqrt(sum(current_pop))
        for i in range(self.n_traj()):
            self.traj[i].update_amplitude(self.traj[i].amplitude * norm)
        timings.stop('bundle.renormalize')

    def prune(self):
        """Kills trajectories that are dead."""
        for i in range(self.nalive):
            continue
        return False

    def in_coupled_regime(self):
        """Returns true if we are in a regime of coupled trajectories."""

        # check if trajectories are coupled
        for i in range(self.nalive):
            for j in range(i):
                if abs(self.T[i,j]+self.V[i,j]) > glbl.fms['hij_coup_thresh']:
                    return True

        # THE BUNDLE SHOULDN'T KNOW ABOUT HOW WE SPAWN. THIS CHECK IS HANDLED
        # ELSEWHERE
        # check if any trajectories exceed NAD threshold
        #for i in range(self.nalive):
        #    for j in range(self.nstates):
        #        if abs(self.traj[i].coup_dot_vel(j)) > glbl.fms['spawn_coup_thresh']:
        #            return True

        # else, return false
        return False

    def amplitudes(self):
        """Returns amplitudes of the trajectories."""
        return np.array([self.traj[self.alive[i]].amplitude
                         for i in range(len(self.alive))], dtype=complex)

    def set_amplitudes(self, amps):
        """Sets the value of the amplitudes."""
        for i in range(self.nalive):
            self.traj[self.alive[i]].amplitude = amps[i]

    def mulliken_pop(self, tid):
        """Returns the Mulliken-like population."""
        mulliken = 0.

        if not self.traj[tid].alive:
            return mulliken
        i = self.alive.index(tid)
        for j in range(len(self.alive)):
            jj = self.alive[j]
            mulliken += abs(self.S[i,j] * self.traj[tid].amplitude.conjugate()
                                        * self.traj[jj].amplitude)
        return mulliken

    def pop(self):
        """Returns the populations on each of the states."""
        timings.start('bundle.pop')
        pop = np.zeros(self.nstates)

        # live contribution
        for i in range(len(self.alive)):
            ii = self.alive[i]
            state = self.traj[ii].state
            popii = (self.traj[ii].amplitude *
                     self.traj[ii].amplitude.conjugate())
            pop[state] += popii.real
            for j in range(i):
                jj = self.alive[j]
                if self.traj[ii].state != self.traj[jj].state:
                    continue
                popij = (2. * self.S[i,j] * self.traj[j].amplitude *
                         self.traj[i].amplitude.conjugate())
                pop[state] += popij.real

        # dead contribution?

        timings.stop('bundle.pop')
        return pop

    def pot_classical(self):
        """Returns the classical potential energy of the bundle.

        Currently includes energy from dead trajectories as well...
        """
        timings.start('bundle.pot_classical')
        weight = np.array([self.traj[i].amplitude *
                           self.traj[i].amplitude.conjugate()
                           for i in range(self.n_traj())])
        v_int  = np.array([self.ints.v_integral(self.traj[i])
                           for i in range(self.n_traj())])
        timings.stop('bundle.pot_classical')
        return sum(weight * v_int).real

    def pot_quantum(self):
        """Returns the QM (coupled) potential energy of the bundle.

        Currently includes <live|live> and <dead|dead> contributions...
        """
        timings.start('bundle.pot_quantum')
        energy = 0.
        for i in range(len(self.alive)):
            ii = self.alive[i]
            weight = (self.traj[ii].amplitude *
                      self.traj[ii].amplitude.conjugate())
            v_int  = self.V[i,i]
            energy += (weight * v_int).real
            for j in range(i):
                jj     = self.alive[j]
                weight = (self.traj[jj].amplitude *
                          self.traj[ii].amplitude.conjugate())
                v_int  = self.V[i,j]
                energy += 2. * (weight * v_int).real
        timings.stop('bundle.pot_quantum')
        return energy

    def kin_classical(self):
        """Returns the classical kinetic energy of the bundle."""
        timings.start('bundle.kin_classical')
        weight  = np.array([self.traj[i].amplitude *
                            self.traj[i].amplitude.conjugate()
                            for i in range(self.n_traj())])
        ke_int  = np.array([self.ints.ke_integral(self.traj[i], self.traj[i])
                            for i in range(self.n_traj())])
        timings.stop('bundle.kin_classical')
        return sum(weight * ke_int).real

    def kin_quantum(self):
        """Returns the QM (coupled) kinetic energy of the bundle."""
        timings.start('bundle.kin_quantum')
        energy = 0.
        for i in range(len(self.alive)):
            ii = self.alive[i]
            weight = (self.traj[ii].amplitude *
                      self.traj[ii].amplitude.conjugate())
            ke_int = self.T[i,i]
            energy += (weight * ke_int).real
            for j in range(i):
                jj     = self.alive[j]
                weight = (self.traj[jj].amplitude *
                          self.traj[ii].amplitude.conjugate())
                ke_int = self.T[i,j]
                energy += 2. * (weight * ke_int).real
        timings.stop('bundle.kin_quantum')
        return energy

    def tot_classical(self):
        """Returns the total classical energy of the bundle."""
        return self.pot_classical() + self.kin_classical()

    def tot_quantum(self):
        """Returns the total QM (coupled) energy of the bundle."""
        return self.pot_quantum() + self.kin_quantum()

    def overlap(self, other):
        """Returns the overlap integral of the bundle with another
        bundle."""
        S = complex(0.,0.)
        for i in range(self.nalive):
            for j in range(other.nalive):
                ii = self.alive[i]
                jj = self.alive[j]
                S += (self.traj[ii].overlap(other.traj[jj]) *
                      self.traj[ii].amplitude.conjugate() *
                      other.traj[jj].amplitude)
        return S

    #----------------------------------------------------------------------
    #
    # Private methods/functions (called only within the class)
    #
    #----------------------------------------------------------------------
    def update_centroids(self):
        """Updates the centroids."""
        timings.start('bundle.update_centroids')

        for i in range(self.n_traj()):
            if not self.traj[i].alive:
                continue
            wid_i = self.traj[i].widths()

            for j in range(i):
                if not self.traj[j].alive:
                    continue

                # first check that we have added trajectory i or j (i.e.
                # that the cent array is long enough), if not, append the
                # appropriate number of slots (just do this once for
                if len(self.cent) < self.cent_len(i):
                    n_add = self.cent_len(i) - len(self.cent)
                    for k in range(n_add):
                        self.cent.append(None)
                # now check to see if needed index has an existing trajectory
                # if not, copy trajectory from one of the parents into the
                # required slots
                ij_ind = cent_ind(i,j)
                if self.cent[ij_ind] is None:
                    self.cent[ij_ind] = trajectory.copy_traj(self.traj[i])
                    self.cent[ij_ind].tid = -ij_ind

                    # set cent[ij_ind].c_state (note that cent[ij_ind].state
                    # is set by calling trajectory.copy_traj)
                    self.cent[ij_ind].c_state=self.traj[j].state

                    # now update the position in phase space of the centroid
                    # if wid_i == wid_j, this is clearly just the simply mean
                    # position.
                    wid_j = self.traj[j].widths()
                    new_x = (wid_i * self.traj[i].x() + wid_j *
                             self.traj[j].x()) / (wid_i + wid_j)
                    new_p = (wid_i * self.traj[i].p() + wid_j *
                             self.traj[j].p()) / (wid_i + wid_j)
                    self.cent[ij_ind].update_x(new_x)
                    self.cent[ij_ind].update_p(new_p)

        timings.stop('bundle.update_centroids')
        return
        
    def update_matrices(self):
        """Updates T, V, S, Sdot and Heff matrices."""
        timings.start('bundle.update_matrices')

        # make sure the centroids are up-to-date in order to evaluate
        # self.H -- if we need them
        if self.ints.require_centroids:
            (self.T, self.V, self.S, self.Sdot,
             self.Heff) = self.hambuild.build_hamiltonian(self.integrals, 
                                self.traj,self.alive, cent_list=self.cent)
        else:
            (self.T, self.V, self.S, self.Sdot,
             self.Heff) = self.hambuild.build_hamiltonian(self.integrals,
                                self.traj, self.alive)

        timings.stop('bundle.update_matrices')

    #------------------------------------------------------------------------
    #
    # Functions to read/write bundle to checkpoint files
    #
    #------------------------------------------------------------------------
    def update_logs(self):
        """Updates the log files."""
        timings.start('bundle.update_logs')

        for i in range(self.n_traj()):
            if not self.traj[i].alive:
                continue

            # trajectory files
            if glbl.fms['print_traj']:
                data = [self.time]
                data.extend(self.traj[i].x().tolist())
                data.extend(self.traj[i].p().tolist())
                data.extend([self.traj[i].phase(), self.traj[i].amplitude.real,
                             self.traj[i].amplitude.imag,
                             abs(self.traj[i].amplitude),
                             self.traj[i].state])

                fileio.print_traj_row(self.traj[i].tid, 0, data)

                # potential energy
                data = [self.time]
                data.extend([self.traj[i].energy(j)
                             for j in range(self.nstates)])
                fileio.print_traj_row(self.traj[i].tid, 1, data)

                # coupling
                data = [self.time]
                data.extend([self.traj[i].coupling_norm(j)
                             for j in range(self.nstates)])
                data.extend([self.traj[i].coup_dot_vel(j)
                             for j in range(self.nstates)])
                fileio.print_traj_row(self.traj[i].tid, 2, data)

            # print electronic structure info
            if glbl.fms['print_es']:
                # permanent dipoles
                data = [self.time]
                for j in range(self.nstates):
                    data.extend(self.traj[i].dipole(j).tolist())
                fileio.print_traj_row(self.traj[i].tid, 3, data)

                # transition dipoles
                data = [self.time]
                for j in range(self.nstates):
                    for k in range(j):
                        data.extend(self.traj[i].tdipole(k,j).tolist())
                fileio.print_traj_row(self.traj[i].tid, 4, data)

                # second moments
                data = [self.time]
                for j in range(self.nstates):
                    data.extend(self.traj[i].sec_mom(j).tolist())
                fileio.print_traj_row(self.traj[i].tid, 5, data)

                # atomic populations
                data = [self.time]
                for j in range(self.nstates):
                    data.extend(self.traj[i].atom_pop(j).tolist())
                fileio.print_traj_row(self.traj[i].tid, 6, data)

                # gradients
                data = [self.time]
                data.extend(self.traj[i].derivative(self.traj[i].state).tolist())
                fileio.print_traj_row(self.traj[i].tid, 7, data)

        # now dump bundle information ####################################

        # state populations
        data = [self.time]
        st_pop = self.pop().tolist()
        data.extend(self.pop().tolist())
        data.append(sum(st_pop))
        fileio.print_bund_row(0, data)

        # bundle energy
        data = [self.time,
                self.pot_quantum(), self.kin_quantum(), self.tot_quantum(),
                self.pot_classical(), self.kin_classical(), self.tot_classical()]
        fileio.print_bund_row(1, data)

        # bundle matrices
        if glbl.fms['print_matrices']:
            fileio.print_bund_mat(self.time, 's.dat', self.S)
            fileio.print_bund_mat(self.time, 'h.dat', self.T+self.V)
            fileio.print_bund_mat(self.time, 'heff.dat', self.Heff)
            fileio.print_bund_mat(self.time, 'sdot.dat', self.Sdot)

        # dump full bundle to an checkpoint file
        if glbl.fms['print_chkpt']:
            self.write_bundle(fileio.scr_path + '/last_step.dat','w')

        # wavepacket autocorrelation function
        if glbl.fms['auto'] == 1 and glbl.bundle0 is not None:
            auto = self.overlap(glbl.bundle0)
            data = [self.time, auto.real, auto.imag, abs(auto)]
            fileio.print_bund_row(7, data)

        timings.stop('bundle.update_logs')

    def write_bundle(self, filename, mode):
        """Dumps the bundle to file 'filename'.

        Mode is either 'a'(append) or 'x'(new).
        """
        timings.start('bundle.write_bundle')

        if mode not in ('w','a'):
            raise ValueError('Invalid write mode in bundle.write_bundle')
        npart = self.traj[0].n_particle
        ndim  = self.traj[0].d_particle
        with open(filename, mode) as chkpt:
            # first write out the bundle-level information
            chkpt.write('------------- BEGIN BUNDLE SUMMARY --------------\n')
            chkpt.write('{:10.2f}            current time\n'.format(self.time))
            chkpt.write('{:10d}            live trajectories\n'.format(self.nalive))
            chkpt.write('{:10d}            dead trajectories\n'.format(self.ndead))
            chkpt.write('{:10d}            number of states\n'.format(self.nstates))
            chkpt.write('{:10d}            number of particles\n'.format(npart))
            chkpt.write('{:10d}            dimensions of particles\n'.format(ndim))

            # particle information common to all trajectories
            for i in range(npart):
                chkpt.write('--------- common particle information --------\n')
                self.traj[0].particles[i].write_particle(chkpt)

            # first write out the live trajectories. The function
            # write_trajectory can only write to a pre-existing file stream
            for i in range(len(self.traj)):
                chkpt.write('-------- trajectory {:4d} --------\n'.format(i))
                self.traj[i].write_trajectory(chkpt)
        chkpt.close()

        timings.stop('bundle.write_bundle')

    def read_bundle(self, filename, t_restart):
        """Reads a bundle at time 't_restart' from a chkpt file."""
        try:
            chkpt = open(filename, 'r', encoding='utf-8')
        except:
            raise FileNotFoundError('Could not open: ' + filename)

        # if we're reading from a checkpoint file -- fast-forward
        # to the requested time
        if t_restart != -1:
            t_found = False
            last_pos = chkpt.tell()
            for line in chkpt:
                if 'current time' in line:
                    if float(line[0]) == t_restart:
                        t_found = True
                        break
        # else, we're reading from a last_step file -- and we want to skip
        # over the initial comment line
        else:
            t_found = True
            chkpt.readline()

        if not t_found:
            raise NameError('Could not find time=' + str(t_restart) +
                            ' in ' + str(chkpt.name))

        # read common bundle information
        self.time    = float(chkpt.readline().split()[0])
        self.nalive  = int(chkpt.readline().split()[0])
        self.ndead   = int(chkpt.readline().split()[0])
        self.nstates = int(chkpt.readline().split()[0])
        npart        = int(chkpt.readline().split()[0])
        ndim         = int(chkpt.readline().split()[0])

        # the particle list will be the same for all trajectories
        p_list = []
        for i in range(npart):
            chkpt.readline()
            part = particle.Particle(ndim, 0)
            part.read_particle(chkpt)
            p_list.append(part)

        # read-in trajectories
        for i in range(self.nalive + self.ndead):
            chkpt.readline()
            t_read = trajectory.Trajectory(glbl.fms['interface'],
                                           self.nstates,
                                           particles=p_list, tid=i,
                                           parent=0, n_basis=0)
            t_read.read_trajectory(chkpt)
            self.traj.append(t_read)

        # create the bundle matrices
        self.T    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff = np.zeros((self.nalive, self.nalive), dtype=complex)

        # once bundle is read, close the stream
        chkpt.close()
