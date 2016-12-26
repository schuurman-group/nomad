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
from src.basis import trajectory as trajectory
from src.basis import centroid as centroid
from src.basis import hamiltonian as fms_ham 

@timings.timed
def copy_bundle(orig_bundle):
    """Copys a Bundle object with new references.

    This method is the simplest way i can see to make a copy
    of a bundle with new references. Overriding deepcopy for
    significantly more work.
    """
    new_bundle = Bundle(orig_bundle.nstates, 
                        orig_bundle.integral_type)
    new_bundle.time   = copy.copy(orig_bundle.time)
    new_bundle.nalive = copy.copy(orig_bundle.nalive)
    new_bundle.ndead  = copy.copy(orig_bundle.ndead)
    new_bundle.alive  = copy.deepcopy(orig_bundle.alive)
    new_bundle.T      = copy.deepcopy(orig_bundle.T)
    new_bundle.V      = copy.deepcopy(orig_bundle.V)
    new_bundle.S      = copy.deepcopy(orig_bundle.S)
    new_bundle.Sdot   = copy.deepcopy(orig_bundle.Sdot)
    new_bundle.Heff   = copy.deepcopy(orig_bundle.Heff)
    new_bundle.traj_ovrlp = copy.deepcopy(orig_bundle.traj_ovrlp)

    # copy the trajectory array
    for i in range(orig_bundle.n_traj()):
        traj_i = trajectory.copy_traj(orig_bundle.traj[i])
        new_bundle.traj.append(traj_i)

    # copy the centroid matrix
    if new_bundle.integrals.require_centroids:
        for i in range(len(orig_bundle.cent)):
            new_bundle.cent.append([None for j in range(orig_bundle.n_traj())])
            for j in range(i):
                new_bundle.cent[i][j] = centroid.copy_cent(orig_bundle.cent[i][j])
                new_bundle.cent[j][i] = new_bundle.cent[i][j]

    return new_bundle

def cent_ind(i, j):
    """Returns the index in the cent array of the centroid between
    trajectories i and j."""
    if i == j:
        return -1
    else:
        a = max(i,j)
        b = min(i,j)
        return a*(a-1)//2 + b

class Bundle:
    """Class constructor for the Bundle object."""
    def __init__(self, nstates, integral_type):
        self.integral_type  = integral_type
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
        self.Sdot      = np.zeros((0, 0), dtype=complex)
        self.Heff      = np.zeros((0, 0), dtype=complex)
        self.traj_ovrlp= np.zeros((0, 0), dtype=complex)
        try:
            self.integrals=__import__('src.integrals.'+
                                       self.integral_type,fromlist=['a'])
        except ImportError:
            print('BUNDLE INIT FAIL: src.integrals.'+self.integral_type)

    def n_traj(self):
        """Returns total number of trajectories."""
        return self.nalive + self.ndead

    @timings.timed
    def add_trajectory(self, new_traj):
        """Adds a trajectory to the bundle."""
        self.traj.append(new_traj)
        self.nalive         += 1
        self.nactive        += 1
        self.traj[-1].alive  = True
        self.traj[-1].active = True
        self.traj[-1].tid    = self.n_traj() - 1
        self.alive.append(self.traj[-1].tid)
        self.active.append(self.traj[-1].tid)
        self.T       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.traj_ovrlp = np.zeros((self.nalive, self.nalive), dtype=complex)

    def add_trajectories(self, traj_list):
        """Adds a set of trajectories to the bundle."""
        for traj in traj_list:
            self.traj.append(traj)
            self.nalive         += 1
            self.nactive        += 1
            self.traj[-1].alive  = True
            self.traj[-1].active = True
            self.traj[-1].tid    = self.n_traj() - 1
            self.alive.append(self.traj[-1].tid)
            self.active.append(self.traj[-1].tid)
        self.T       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.traj_ovrlp = np.zeros((self.nalive, self.nalive), dtype=complex)

    @timings.timed
    def kill_trajectory(self, tid):
        """Moves a live trajectory to the list of dead trajecotries.

        The trajectory will no longer contribute to H, S, etc.
        """
        # Remove the trajectory from the list of living trajectories
        self.alive.remove(tid)
        self.traj[tid].alive = False
        self.nalive          = self.nalive - 1
        self.ndead           = self.ndead + 1
        # Remove the trajectory from the list of active trajectories
        # iff matching pursuit is not being used
        if glbl.fms['matching_pursuit'] == 0:
            self.active.remove(tid)
            self.traj[tid].active = False
            self.nactive = self.nactive - 1
        # Reset arrays
        self.T       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.traj_ovrlp = np.zeros((self.nalive, self.nalive), dtype=complex)

    @timings.timed
    def revive_trajectory(self, tid):
        """
        Moves a dead trajectory to the list of live trajectories.
        """
        self.traj[tid].alive = True

        # Add the trajectory to the list of living trajectories
        self.alive.append(tid)

        self.nalive          = self.nalive + 1
        self.ndead           = self.ndead - 1

        self.T       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.traj_ovrlp = np.zeros((self.nalive, self.nalive), dtype=complex)

    @timings.timed
    def update_amplitudes(self, dt, H=None, Ct=None):
        """Updates the amplitudes of the trajectory in the bundle.
        Solves d/dt C = -i H C via the computation of
        exp(-i H(t) dt) C(t)."""
        self.update_matrices()

        # if no Hamiltonian is pased, use the current effective
        # Hamiltonian
        if H is None:
            Hmat = self.Heff
        else:
            Hmat = H 

        # if no vector of amplitdues are supplied (to propagate),
        # propogate the current amplitudes
        if Ct is None:
            old_amp = self.amplitudes()
        else:
            old_amp = Ct 

        new_amp = np.zeros(self.nalive, dtype=complex)

        B = -1j * Hmat * dt

        if self.nalive < 150:
            # Eigen-decomposition
            umat = linalg.expm2(B)
        else:
            # Pade approximation
            umat = linalg.expm(B)

        new_amp = np.dot(umat, old_amp)

        for i in range(len(self.alive)):
            self.traj[self.alive[i]].update_amplitude(new_amp[i])

    @timings.timed
    def renormalize(self):
        """Renormalizes the amplitudes of the trajectories in the bundle."""
        current_pop = self.pop()
        norm = 1. / np.sqrt(sum(current_pop))
        for i in range(self.n_traj()):
            self.traj[i].update_amplitude(self.traj[i].amplitude * norm)

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
            mulliken += abs(self.traj_ovrlp[i,j] * 
                            self.traj[tid].amplitude.conjugate() *
                            self.traj[jj].amplitude)
        return mulliken

    @timings.timed
    def norm(self):
        """Returns the norm of the wavefunction """
        return np.dot(np.dot(
               np.conj(self.amplitudes()),self.traj_ovrlp),self.amplitudes()).real

    @timings.timed
    def pop(self):
        """Returns the populations on each of the states."""
        pop    = np.zeros(self.nstates)
        nalive = len(self.alive) 

        # live contribution
        for i in range(nalive):
            ii = self.alive[i]
            state = self.traj[ii].state
            for j in range(nalive):
                jj = self.alive[j]
                popij = (self.traj_ovrlp[i,j]  * 
                         self.traj[jj].amplitude *
                         self.traj[ii].amplitude.conjugate())
                pop[state] += popij.real

        # dead contribution?
        return pop

    @timings.timed
    def pot_classical(self):
        """Returns the classical potential energy of the bundle.

        Currently only includes energy from alive trajectories
        """
        nalive = len(self.alive)
        v_vec = np.array([self.traj[self.alive[i]].potential()
                          for i in range(nalive)])
        return sum(v_vec).real / nalive
#        weight = np.array([self.traj[self.alive[i]].amplitude *
#                           self.traj[self.alive[i]].amplitude.conjugate() 
#                           for i in range(len(self.alive))])
#        v_int  = np.array([self.V[i,i] 
#                           for i in range(self.nalive)])
#        return np.dot(weight, v_int).real

    @timings.timed
    def pot_quantum(self):
        """Returns the QM (coupled) potential energy of the bundle.
        Currently includes <live|live> (not <dead|dead>,etc,) contributions...
        """
        return np.dot(np.dot(
               np.conj(self.amplitudes()),self.V),self.amplitudes()).real
#        energy = 0.
#        for i in range(len(self.alive)):
#            ii = self.alive[i]
#            weight = (self.traj[ii].amplitude *
#                      self.traj[ii].amplitude.conjugate())
#            v_int  = self.V[i,i]
#            energy += (weight * v_int).real
#            for j in range(len(self.alive)):
#                jj     = self.alive[j]
#                weight = (self.traj[jj].amplitude *
#                          self.traj[ii].amplitude.conjugate())
#                v_int  = self.V[i,j]
#                print("pot quan="+str(weight * v_int))
#                energy += (weight * v_int).real
#        return energy

    @timings.timed
    def kin_classical(self):
        """Returns the classical kinetic energy of the bundle."""
        nalive = len(self.alive)
        ke_vec = np.array([np.dot(self.traj[self.alive[i]].p()*
                                  self.traj[self.alive[i]].p(),
                                  self.traj[self.alive[i]].interface.kecoeff)
                                  for i in range(nalive)])
        return sum(ke_vec).real / nalive
#        weight = np.array([self.traj[self.alive[i]].amplitude *
#                           self.traj[self.alive[i]].amplitude.conjugate() 
#                           for i in range(len(self.alive))])
#        ke_int  = np.array([self.T[i,i] 
#                           for i in range(self.nalive)])
#        return np.dot(weight, ke_int).real

    @timings.timed
    def kin_quantum(self):
        """Returns the QM (coupled) kinetic energy of the bundle."""
        Sinv = np.linalg.pinv(self.traj_ovrlp)
        return np.dot(np.dot(
               np.conj(self.amplitudes()),self.T),self.amplitudes()).real
#        energy = 0.
#        for i in range(len(self.alive)):
#            ii = self.alive[i]
#            weight = (self.traj[ii].amplitude *
#                      self.traj[ii].amplitude.conjugate())
#            ke_int = self.T[i,i]
#            energy += (weight * ke_int).real
#            for j in range(len(self.alive)):
#                jj     = self.alive[j]
#                weight = (self.traj[jj].amplitude *
#                          self.traj[ii].amplitude.conjugate())
#                ke_int = self.T[i,j]
#                print("kin quan="+str(weight * ke_int))
#                energy += (weight * ke_int).real
#        return energy

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
                S += (self.integrals.traj_overlap(self.traj[ii], other.traj[jj]) *
                      self.traj[ii].amplitude.conjugate() *
                      other.traj[jj].amplitude)
        return S

    def overlap_traj(self, traj):
        """Returns the overlap of the bundle with a trajectory (assumes the
        amplitude on the trial trajectory is (1.,0.)"""
        ovrlp = complex(0., 0.)
        for i in range(self.nalive+self.ndead):
            ovrlp += (self.integrals.traj_overlap(traj,self.traj[i]) * 
                                             self.traj[i].amplitude)
        return ovrlp

    #----------------------------------------------------------------------
    #
    # Private methods/functions (called only within the class)
    #
    #----------------------------------------------------------------------
    @timings.timed
    def update_centroids(self):
        """Updates the centroids."""

        # make sure centroid array has sufficient space to hold required 
        # centroids. Note that n_traj includes alive AND dead trajectories --
        # therefore it can only increase. So, only need to check n_traj > dim_cent
        # condition 
        dim_cent = len(self.cent)

        if self.n_traj() < dim_cent:
            sys.exit('n_traj() < dim_cent in bundle. Exiting...')            

        if self.n_traj() > dim_cent:
            for i in range(dim_cent):
                self.cent[i].extend([None for j in range(self.n_traj()-dim_cent)])

            for i in range(self.n_traj() - dim_cent):        
                self.cent.append([])
                self.cent[-1].extend([None for j in range(self.n_traj())])
                
        for i in range(self.n_traj()):
            if not self.traj[i].alive:
                continue

            for j in range(i):
                if not self.traj[j].alive:
                    continue

                # now check to see if needed index has an existing trajectory
                # if not, copy trajectory from one of the parents into the
                # required slots
                if self.cent[i][j] is None:
                    self.cent[i][j] = centroid.Centroid(traj_i=self.traj[i],
                                                        traj_j=self.traj[j])
                    self.cent[j][i] = self.cent[i][j]

    @timings.timed
    def update_matrices(self):
        """Updates T, V, S, Sdot and Heff matrices."""
        # make sure the centroids are up-to-date in order to evaluate
        # self.H -- if we need them
        if self.integrals.require_centroids:
            (self.traj_ovrlp, self.T, self.V, self.S, self.Sdot, self.Heff ) = \
                                      fms_ham.hamiltonian(self.traj, 
                                                          self.alive, 
                                                          cent_list=self.cent)
        else:
            (self.traj_ovrlp, self.T, self.V, self.S, self.Sdot, self.Heff ) = \
                                      fms_ham.hamiltonian(self.traj, 
                                                          self.alive)

    #------------------------------------------------------------------------
    #
    # Functions to read/write bundle to checkpoint files
    #
    #------------------------------------------------------------------------
    @timings.timed
    def update_logs(self):
        """Updates the log files."""

#        dia1      = self.traj[0].pes_data.diabat_pot
#        argt1     = 2. * dia1[0,1] / (dia1[1,1] - dia1[0,0])
#        theta1    = 0.5 * np.arctan(argt1)

#        dia2      = self.traj[1].pes_data.diabat_pot
#        argt2     = 2. * dia2[0,1] / (dia2[1,1] - dia2[0,0])
#        theta2    = 0.5 * np.arctan(argt2)

#        print("theta1, theta2: "+str(theta1)+' '+str(theta2))
#        print("cos,sin1: "+str([np.cos(theta1),np.sin(theta1)]))
#        print("adt_mat:  "+str(self.traj[0].pes_data.adt_mat[:,self.traj[0].state]))
#        print("dat_mat1:  "+str(self.traj[0].pes_data.dat_mat[:,self.traj[0].state]))
#        print("cos,sin2: "+str([np.cos(theta2),np.sin(theta2)]))
#        print("adt_mat:  "+str(self.traj[1].pes_data.adt_mat[:,self.traj[1].state]))
#        print("dat_mat2:  "+str(self.traj[1].pes_data.dat_mat[:,self.traj[1].state]))
        

        for i in range(self.n_traj()):

            if not self.traj[i].active:
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

            # print pes information relevant to the chosen interface
            if glbl.fms['print_es']:

                # print the interface-specific data 
                for key in self.traj[i].pes_data.data_keys:

                    # skip the stuff that is NOT interface speicifc
                    if key in ['geom','poten','deriv']:
                        continue

                    # 

                # permanent dipoles
#                data = [self.time]
#                for j in range(self.nstates):
#                    data.extend(self.traj[i].dipole(j).tolist())
#                fileio.print_traj_row(self.traj[i].tid, 3, data)

                # transition dipoles
#                data = [self.time]
#                for j in range(self.nstates):
#                    for k in range(j):
#                        data.extend(self.traj[i].tdipole(k,j).tolist())
#                fileio.print_traj_row(self.traj[i].tid, 4, data)

                # second moments
#                data = [self.time]
#                for j in range(self.nstates):
#                    data.extend(self.traj[i].sec_mom(j).tolist())
#                fileio.print_traj_row(self.traj[i].tid, 5, data)

                # atomic populations
#                data = [self.time]
#                for j in range(self.nstates):
#                    data.extend(self.traj[i].atom_pop(j).tolist())
#                fileio.print_traj_row(self.traj[i].tid, 6, data)

                # gradients
                data = [self.time]
                data.extend(self.traj[i].derivative(self.traj[i].state).tolist())
                fileio.print_traj_row(self.traj[i].tid, 7, data)

        # now dump bundle information ####################################

#        print("T="+str(self.T))
#        print("V="+str(self.V))

        # state populations
        data = [self.time]
        data.extend(self.pop().tolist())
        data.append(self.norm())
        fileio.print_bund_row(0, data)

        # bundle energy
        data = [self.time,
                self.pot_quantum(), self.kin_quantum(), self.tot_quantum(),
                self.pot_classical(), self.kin_classical(), self.tot_classical()]
        fileio.print_bund_row(1, data)

        # bundle matrices
        if glbl.fms['print_matrices']:
            if self.integrals.basis != 'gaussian':
                fileio.print_bund_mat(self.time, 't_ovrlp.dat', self.traj_ovrlp) 
            fileio.print_bund_mat(self.time, 's.dat', self.S)
            fileio.print_bund_mat(self.time, 'h.dat', self.T+self.V)
 #           fileio.print_bund_mat(self.time, 't.dat', self.T)
 #           fileio.print_bund_mat(self.time, 'v.dat', self.V)
            fileio.print_bund_mat(self.time, 'heff.dat', self.Heff)
            fileio.print_bund_mat(self.time, 'sdot.dat', self.Sdot)

        # dump full bundle to an checkpoint file
        if glbl.fms['print_chkpt']:
            self.write_bundle(fileio.scr_path + '/last_step.dat','w')

        # wavepacket autocorrelation function
        if glbl.fms['auto'] == 1 and glbl.bundle0 is not None:
            auto = self.overlap(glbl.bundle0)
            data = [self.time, auto.real, auto.imag, abs(auto)]
            fileio.print_bund_row(8, data)

    @timings.timed
    def write_bundle(self, filename, mode):
        """Dumps the bundle to file 'filename'.

        Mode is either 'a'(append) or 'x'(new).
        """
        if mode not in ('w','a'):
            raise ValueError('Invalid write mode in bundle.write_bundle')
        crd_dim = self.traj[0].crd_dim
        ndim    = self.traj[0].dim
        with open(filename, mode) as chkpt:
            # first write out the bundle-level information
            chkpt.write('------------- BEGIN BUNDLE SUMMARY --------------\n')
            chkpt.write('{:10.2f}            current time\n'.format(self.time))
            chkpt.write('{:10d}            live trajectories\n'.format(self.nalive))
            chkpt.write('{:10d}            dead trajectories\n'.format(self.ndead))
            chkpt.write('{:10d}            number of states\n'.format(self.nstates))
            chkpt.write('{:10d}            number of coordinates\n'.format(ndim))
            chkpt.write('{:10d}            dimensions of coord system\n'.format(crd_dim))

            # information common to all trajectories
            chkpt.write('--------- common trajectory information --------\n')
            chkpt.write('coordinate widths --\n')
            chkpt.write(str(np.array2string(self.traj[0].widths(), 
                        formatter={'float_kind':lambda x: "%.4f" % x}))+'\n')
            chkpt.write('coordinate masses --\n')
            chkpt.write(str(np.array2string(self.traj[0].masses(), 
                        formatter={'float_kind':lambda x: "%.4f" % x}))+'\n')

            # first write out the live trajectories. The function
            # write_trajectory can only write to a pre-existing file stream
            for i in range(len(self.traj)):
                chkpt.write('-------- trajectory {:4d} --------\n'.format(i))
                self.traj[i].write_trajectory(chkpt)
        chkpt.close()

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
        ndim         = int(chkpt.readline().split()[0])
        crd_dim      = int(chkpt.readline().split()[0])

        # the read common info that will be the same for all trajectories
        chkpt.readline()
        # widths
        chkpt.readline()
        widths = np.fromstring(chkpt.readline(), sep=' ', dtype=float)
        # masses
        chkpt.readline()
        masses = np.fromstring(chkpt.readline(), sep=' ', dtype=float)

        # read-in trajectories
        for i in range(self.nalive + self.ndead):
            chkpt.readline()
            t_read = trajectory.Trajectory(self.nstates,
                                           dim,
                                           width=widths,
                                           mass=masses,
                                           crd_dim=crd_dim,
                                           tid=i,
                                           parent=0,
                                           n_basis=0)
            t_read.read_trajectory(chkpt)
            self.traj.append(t_read)

        # create the bundle matrices
        self.T       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.traj_ovrlp = np.zeros((self.nalive, self.nalive), dtype=complex)
        # once bundle is read, close the stream
        chkpt.close()
