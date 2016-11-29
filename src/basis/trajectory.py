"""
The Trajectory object and its associated functions.
"""
import sys
import copy
import numpy as np
import src.dynamics.timings as timings
import src.fmsio.glbl as glbl

@timings.timed
def copy_traj(orig_traj):
    """Copys a Trajectory object with new references."""
    new_traj = Trajectory(orig_traj.nstates,
                          orig_traj.dim,
                          orig_traj.widths,
                          orig_traj.masses,
                          orig_traj.labels,
                          orig_traj.crd_dim,
                          orig_traj.tid,
                          orig_traj.parent,
                          orig_traj.nbasis)
    new_traj.state      = copy.copy(orig_traj.state)
    new_traj.c_state    = copy.copy(orig_traj.c_state)
    new_traj.alive      = copy.copy(orig_traj.alive)
    new_traj.amplitude  = copy.copy(orig_traj.amplitude)
    new_traj.gamma      = copy.copy(orig_traj.gamma)
    new_traj.deadtime   = copy.copy(orig_traj.deadtime)
    new_traj.nbf        = copy.copy(orig_traj.nbf)
    new_traj.x          = copy.deepcopy(orig_traj.x)
    new_traj.p          = copy.deepcopy(orig_traj.p)
    new_traj.last_spawn = copy.deepcopy(orig_traj.last_spawn)
    new_traj.exit_time  = copy.deepcopy(orig_traj.exit_time)
    new_traj.spawn_coup = copy.deepcopy(orig_traj.spawn_coup)
    new_traj.pes_geom   = copy.deepcopy(orig_traj.pes_geom)
    new_traj.poten      = copy.deepcopy(orig_traj.poten)
    new_traj.deriv      = copy.deepcopy(orig_traj.deriv)
    new_traj.dipoles    = copy.deepcopy(orig_traj.dipoles)
    new_traj.sec_moms   = copy.deepcopy(orig_traj.sec_moms)
    new_traj.atom_pops  = copy.deepcopy(orig_traj.atom_pops)
    new_traj.sct        = copy.deepcopy(orig_traj.sct)
    return new_traj


class Trajectory:
    """Class constructor for the Trajectory object."""
    def __init__(self, 
                 nstates, 
                 dim,
                 widths=None, 
                 masses=None, 
                 labels=None, 
                 crd_dim=3,
                 tid=0, 
                 parent=0, 
                 n_basis=0):

        # total number of states
        self.nstates = nstates
        # dimensionality of the trajectory
        self.dim     = dim
        # widths of gaussians for each dimension
        if widths is None:
            self.widths = np.zeros(dim)
        else:
            self.widths = widths
        # masses associated with each dimension
        if masses is None:
            self.masses = np.zeros(dim)
        else:
            self.masses = masses
        # labels for each dimension (i.e. atom types, etc.)
        if labels is None:
            self.labels = ['c'+str(i) for i in range(dim)]
        else:
            self.labels = labels
        # dimension of the coordinate system 
        #(i.e. ==3 for Cartesian, ==3N-6 for internals)
        self.crd_dim = crd_dim
        # unique identifier for trajectory
        self.tid        = tid
        # trajectory that spawned this one:
        self.parent     = parent
        # number of mos
        self.nbf        = n_basis


        # current position of the trajectory
        self.x          = np.zeros(self.dim)
        # current momentum of the trajecotry
        self.p          = np.zeros(self.dim)
        # state trajectory exists on
        self.state      = 0
        # if a centroid, state for second trajectory
        self.c_state    = -1
        # whether the trajectory is alive (i.e. contributes to the wavefunction)
        self.alive      = True
        # whether the trajectory is active (i.e. is being propagated)
        self.active     = True
        # amplitude of trajectory
        self.amplitude  = complex(0.,0.)
        # phase of the trajectory
        self.gamma      = 0.
        # time from which the death watch begini as
        self.deadtime   = -1.
        # time of last spawn
        self.last_spawn = np.zeros(self.nstates)
        # time trajectory last left coupling region
        self.exit_time  = np.zeros(self.nstates)
        # if not zero, coupled to traj=array value
        self.spawn_coup = np.zeros(self.nstates)
        # geometry of the current potential information
        self.pes_geom   = np.zeros(self.dim)
        # value of the potential energy
        self.poten      = np.zeros(self.nstates)
        # derivatives of the potential -- if off-diagonal, corresponds
        # to Fij (not non-adiabatic coupling vector)
        self.deriv      = np.zeros((self.nstates,self.dim))
        # dipoles and transition dipoles
        self.dipoles    = np.zeros((self.nstates, self.nstates, self.crd_dim))
        # second moment tensor for each state
        self.sec_moms   = np.zeros((self.nstates, self.crd_dim))
        # number of atoms (or internal coordinates, etc.) 
        self.natoms = int(self.dim / self.crd_dim) 
        # electronic populations on the atoms
        self.atom_pops  = np.zeros((self.nstates, self.natoms))
        # Scalar coupling terms involving the state that the
        # trajectory exits on (including DBOCs)        
        self.sct = np.zeros((self.nstates))
        # name of interface to get potential information
        self.interface = __import__('src.interfaces.' +
                               glbl.fms['interface'], fromlist =
                               ['a'])

    #-------------------------------------------------------------------
    #
    # Trajectory status functions
    #
    #--------------------------------------------------------------------
    def dead(self, time, uncoupled_thresh):
        """Returns true if the trajectory is dead."""
        return self.deadtime != -1 and (time - self.deadtime) > uncoupled_thresh

    #----------------------------------------------------------------------
    #
    # Functions for setting basic pes information from trajectory
    #
    #----------------------------------------------------------------------
    def update_x(self, pos):
        """Updates the position of the particles in trajectory.
        """
        self.x = pos

    def update_p(self, mom):
        """Updates the momentum of the particles in the trajectory.
        """
        self.p = mom

    def update_phase(self, phase):
        """Updates the nuclear phase."""
        self.gamma = phase
        if abs(self.gamma) > 2*np.pi:
            self.gamma = self.gamma - int(self.gamma/(2. * np.pi)) * 2. * np.pi

    def update_amplitude(self, amplitude):
        """Updates the amplitude of the trajectory."""
        self.amplitude = amplitude

    def update_pes(self, pes_info):
        """Updates information about the potential energy surface."""
        self.pes_geom  = pes_info[0]
        self.poten     = pes_info[1]
        # centroids do not necessarily require gradient info
        if len(pes_info) >= 3:
            self.deriv     = pes_info[2]
        # if we have electronic structure info
        if len(pes_info) == 6:
            self.dipoles   = pes_info[3]
            self.sec_moms  = pes_info[4]
            self.atom_pops = pes_info[5]
        # SCTs (vibronic interface only, so dipoles, etc. wont be in
        # the pes_geom array)
        if glbl.fms['coupling_order'] > 1:
            self.sct = pes_info[3]

    #-----------------------------------------------------------------------
    #
    # Functions for retrieving basic pes information from trajectory
    #
    #-----------------------------------------------------------------------
    def x(self):
        """Returns the position of the particles in the trajectory as an
        array."""
        return self.x

    def p(self):
        """Returns the momentum of the particles in the trajectory as an
        array."""
        return self.p

    def phase(self):
        """Returns the phase of the trajectory."""
        return self.gamma

    def masses(self):
        """Returns a vector containing masses of particles."""
        return self.masses

    def widths(self):
        """Returns a vector containing the widths of the basis functions
        along each degree of freedom."""
        return self.widths

    #--------------------------------------------------------------------
    #
    # Functions to update information about the potential energy surface
    #
    #--------------------------------------------------------------------
    def energy(self, state):
        """Returns the potential energies.

        Add the energy shift right here. If not current, recompute them.
        """
        if np.linalg.norm(self.pes_geom - self.x()) > glbl.fpzero:
            print('WARNING: trajectory.energy() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.tid))
        return self.poten[state] + glbl.fms['pot_shift']

    def derivative(self, state):
        """Returns the derivative with ket state = rstate.

        Bra state assumed to be the current state.
        """
        if np.linalg.norm(self.pes_geom - self.x()) > glbl.fpzero:
            print('WARNING: trajectory.derivative() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.tid))
        return self.deriv[state,:]

    def dipole(self, state):
        """Returns permanent dipoles."""
        if np.linalg.norm(self.pes_geom - self.x()) > glbl.fpzero:
            print('WARNING: trajectory.dipole() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.tid))
        return self.dipoles[state,state,:]

    def tdipole(self, state_i, state_j):
        """Returns transition dipoles."""
        if np.linalg.norm(self.pes_geom - self.x()) > glbl.fpzero:
            print('WARNING: trajectory.tdipole() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.tid))
        return self.dipoles[state_i,state_j,:]

    def sec_mom(self, state):
        """Returns second moments."""
        if np.linalg.norm(self.pes_geom - self.x()) > glbl.fpzero:
            print('WARNING: trajectory.sec_mom() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.tid))
        return self.sec_moms[state,:]

    def atom_pop(self, state):
        """Returns atomic populations."""
        if np.linalg.norm(self.pes_geom - self.x()) > glbl.fpzero:
            print('WARNING: trajectory.atom_pop() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.tid))
        return self.atom_pops[state,:]

    def scalar_coup(self,state):
        """Returns scalar coupling terms"""
        if np.linalg.norm(self.pes_geom - self.x()) > glbl.fpzero:
            print("WARNING: trajectory.scalar_coup() called, "+
                  "but pes_geom != trajectory.x(). ID="+str(self.tid))
        return self.sct[state]

    #def orbitals(self):
    #    return self.pes.orbitals(self.tid, self.particles, self.state)

    #------------------------------------------------------------------------
    #
    # Computed quantities from the trajectory
    #
    #------------------------------------------------------------------------
    def potential(self):
        """Returns classical potential energy of the trajectory."""
        return self.energy(self.state)

    def kinetic(self):
        """Returns classical kinetic energy of the trajectory."""
        return sum( self.p() * self.p() * self.interface.kecoeff)

    def classical(self):
        """Returns the classical energy of the trajectory."""
        return self.potential() + self.kinetic()

    def velocity(self):
        """Returns the velocity of the trajectory."""        
        return self.p() * 2.0 * self.interface.kecoeff

    def force(self):
        """Returns the gradient of the trajectory state."""
        return -self.derivative(self.state)

    def phase_dot(self):
        """Returns time derivatives of the phase."""
        # d[gamma]/dt = T - V - alpha/(2M)
        if glbl.fms['phase_prop'] == 0:
            return 0.
        else:
            return (self.kinetic() - self.potential() -
                    sum(self.widths() * self.interface.kecoeff))

    def coupling_norm(self, rstate):
        """Returns the norm of the coupling vector."""
        if self.state == rstate:
            return 0.
        return np.linalg.norm(self.derivative(rstate))

    def coup_dot_vel(self, c_state):
        """Returns the coupling dotted with the velocity."""
        if self.state == c_state:
            return 0.
        return np.dot( self.velocity(), self.derivative(c_state) )

    def eff_coup(self,c_state):
        """Returns the effective coupling."""
        if self.state == c_state:
            return 0.
        # F.p/m
        coup = np.dot( self.velocity(), self.derivative(c_state) )
        # G
        if glbl.fms['coupling_order'] > 1:
            coup += self.scalar_coup(c_state)
        return coup

    #-----------------------------------------------------------------------------
    #
    # primitive integral routines
    #
    #-----------------------------------------------------------------------------
    #@timings.timed
#    def nuc_overlap(self, other):
#        """Returns overlap of the nuclear component between two trajectories."""
#        S = np.exp( 1j * (other.gamma - self.gamma) )
#        for i in range(self.n_particle):
#            S = S * self.particles[i].overlap(other.particles[i])
#        return S
#        
#    def overlap_bundle(self, other):
#        """Returns the overlap of a trajectory with a bundle of trajectories"""
#        ovrlp = complex(0., 0.)
#        for i in range(other.nalive+other.ndead):
#            ovrlp += self.ints.stotal_integral(self,other.traj[i]) * other.traj[i].amplitude
#        return ovrlp
#
#    def evaluate_traj(self, x):
#        """Returns the value of the trajectory basis function evaluated at 'x'"""
#        val = np.exp( 1j * self.gamma ) 
#        for i in range(self.n_particle):
#            val = val * self.particles[i].evaluate_particle(
#                                     x[self.d_particle*i:self.d_particle*(i+1)])
#        return val
#
#    #@timings.timed
#    def deldp(self, other, S=None):
#        """Returns the del/dp matrix element between two trajectories --
#           (does not sum over terms). If no value for overlap is given,
#           default is to evaluate the total overlap (i.e. including 
#           electronic component)"""
#        if S is None:
#            S = self.ints.stotal_integral(self, other)
#        if S == 0.:
#            return np.zeros(self.n_particle * self.d_particle, dtype=complex)
#        else:
#            dpval = np.zeros(self.n_particle * self.d_particle, dtype=complex)
#            for i in range(self.n_particle):
#                dpval[self.d_particle*i:self.d_particle*(i+1)] =               \
#                                     self.particles[i].deldp(other.particles[i])
#            return dpval * S
#
#    #@timings.timed
#    def deldx(self, other, S=None):
#        """Returns the del/dx matrix element between two trajectories --
#           (does not sum over terms).If no value for overlap is given,
#           default is to evaluate the total overlap (i.e. including
#           electronic component)"""
#        if S is None:
#            S = self.ints.stotal_integral(self,other)
#        if S == 0.:
#            return np.zeros(self.n_particle * self.d_particle, dtype=complex)
#        else:
#            dxval = np.zeros(self.n_particle * self.d_particle, dtype=complex)
#            for i in range(self.n_particle):
#                dxval[self.d_particle*i:self.d_particle*(i+1)] =               \
#                                    self.particles[i].deldx(other.particles[i])
#            return dxval * S
#
#    #@timings.timed
#    def deld2x(self, other, S=None):
#        """Returns the del2/d2x matrix element between two trajectories --
#           (does not sum over terms).If no value for overlap is given,
#           default is to evaluate the total overlap (i.e. including
#           electronic component)"""
#        if S is None:
#            S = self.ints.stotal_integral(self,other)
#        if S == 0.:
#            return np.zeros(self.n_particle * self.d_particle, dtype=complex)
#        else:
#            d2xval = np.zeros(self.n_particle * self.d_particle, dtype=complex)
#            for i in range(self.n_particle):
#                d2xval[self.d_particle*i:self.d_particle*(i+1)] =              \
#                                    self.particles[i].deld2x(other.particles[i])
#            return d2xval * S
#
#    #@timings.timed
#    def deldx_m(self, other, Snuc=None):
#        """Returns the momentum expectation values multiplied by 2*a_i.
#        
#        Here, the a_i are the coefficients entering into the KE
#        operator 
#        
#        T = sum_i a_i * p_i^2,
#        
#        where p_i is the momentum operator for the ith nuclear dof.
#
#        This appears in the equations of motion on the off diagonal coupling
#        different states together through the NACME.
#        """
#        if Snuc is None:
#            Snuc = self.ints.snuc_integral(self,other)       
#        dxval = np.zeros(self.n_particle * self.d_particle, dtype=np.cfloat)
#        for i in range(self.n_particle):
#            dxval[self.d_particle*i:self.d_particle*(i+1)] = (self.particles[i].deldx(other.particles[i])
#                                                              * 2.0 * self.interface.kecoeff[i*self.d_particle])
#        return dxval * Snuc

    #--------------------------------------------------------------------------
    #
    # routines to write/read trajectory from a file stream
    #
    #--------------------------------------------------------------------------
    def write_trajectory(self, chkpt):
        """Writes trajectory information to a file stream."""
        np.set_printoptions(precision=8, linewidth=80, suppress=False)
        chkpt.write('     {:5s}            alive\n'.format(str(self.alive)))
        chkpt.write('{:10d}            nstates\n'.format(self.nstates))
        chkpt.write('{:10d}            traj ID\n'.format(self.tid))
        chkpt.write('{:10d}            state\n'.format(self.state))
        chkpt.write('{:10d}            parent ID\n'.format(self.parent))
        chkpt.write('{:10d}            n basis function\n'.format(self.nbf))
        chkpt.write('{:10.2f}            dead time\n'.format(self.deadtime))
        chkpt.write('{:16.12f}      phase\n'.format(self.gamma))
        chkpt.write('{:16.12f}         amplitude\n'.format(self.amplitude))
        chkpt.write('# potential energy -- nstates\n')
        self.poten.tofile(chkpt, ' ', '%14.10f')
        chkpt.write('\n')
        chkpt.write('# exit coupling region\n')
        self.exit_time.tofile(chkpt, ' ', '%8.2f')
        chkpt.write('\n')
        chkpt.write('# last spawn\n')
        self.last_spawn.tofile(chkpt, ' ', '%8.2f')
        chkpt.write('\n')
        chkpt.write('# currently coupled\n')
        self.spawn_coup.tofile(chkpt, ' ', '%8d')
        chkpt.write('\n')
        chkpt.write('# position\n')
        self.x().tofile(chkpt, ' ', '%12.8f')
        chkpt.write('\n')
        chkpt.write('# momentum\n')
        self.p().tofile(chkpt, ' ', '%12.8f')
        chkpt.write('\n')

        # Writes out dipole moments in cartesian coordinates
        init_states = [0, self.state]
        for i in init_states:
            for j in range(self.nstates):
                if j == i or j in init_states[0:i]:
                    continue
                chkpt.write('# dipoles state1, state2 = {:4d}, {:4d}'
                            '\n'.format(j,i))
                self.dipoles[j,i,:].tofile(chkpt, ' ', '%10.6f')
                chkpt.write('\n')

        # Writes out gradients
        for i in range(self.nstates):
            chkpt.write('# derivatives state1, state2 = {:4d}, {:4d}'
                        '\n'.format(self.state,i))
            self.deriv[i,:].tofile(chkpt, ' ', '%16.10e')
            chkpt.write('\n')

        # write out second moments
        for i in range(self.nstates):
            chkpt.write('# second moments, state = {:4d}\n'.format(i))
            self.sec_moms[i,:].tofile(chkpt, ' ', '%16.10e')
            chkpt.write('\n')

        # write out atomic populations
        for i in range(self.nstates):
            chkpt.write('# atomic populations, state = {:4d}\n'.format(i))
            self.atom_pops[i,:].tofile(chkpt, ' ', '%16.10e')
            chkpt.write('\n')

    def read_trajectory(self,chkpt):
        """Reads the trajectory information from a file.

        This assumes the trajectory invoking this function has been
        initially correctly and can hold all the information.
        """
        self.alive     = bool(chkpt.readline().split()[0])
        self.nstates   = int(chkpt.readline().split()[0])
        self.tid       = int(chkpt.readline().split()[0])
        self.state     = int(chkpt.readline().split()[0])
        self.parent    = int(chkpt.readline().split()[0])
        self.nbf       = int(chkpt.readline().split()[0])
        self.deadtime  = float(chkpt.readline().split()[0])
        self.gamma     = float(chkpt.readline().split()[0])
        self.amplitude = complex(chkpt.readline().split()[0])

        chkpt.readline()
        # potential energy -- nstates
        self.poten = np.fromstring(chkpt.readline(), sep=' ', dtype=float)
        chkpt.readline()
        # exit coupling region
        self.exit_time = np.fromstring(chkpt.readline(), sep=' ', dtype=float)
        chkpt.readline()
        # last spawn
        self.last_spawn = np.fromstring(chkpt.readline(), sep=' ', dtype=float)
        chkpt.readline()
        # currently coupled
        self.spawn_coup = np.fromstring(chkpt.readline(), sep=' ', dtype=float)
        chkpt.readline()
        # position
        self.update_x(np.fromstring(chkpt.readline(), sep=' ', dtype=float))
        chkpt.readline()
        # momentum
        self.update_p(np.fromstring(chkpt.readline(), sep=' ', dtype=float))

        for i in range(self.nstates):
            for j in range(i + 1):
                chkpt.readline()
                self.dipoles[i,j,:] = np.fromstring(chkpt.readline(),
                                                    sep=' ', dtype=float)

        for i in range(self.nstates):
            chkpt.readline()
            self.deriv[i,:] = np.fromstring(chkpt.readline(),
                                            sep=' ', dtype=float)

        chkpt.readline()
        # orbitals?
