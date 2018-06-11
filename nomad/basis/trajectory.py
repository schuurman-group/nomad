"""
The Trajectory object and its associated functions.
"""
import sys
import copy
import numpy as np
import nomad.utils.timings as timings
import nomad.utils.constants as constants


class Trajectory:
    """Class constructor for the Trajectory object."""
    def __init__(self, nstates, dim, width=None, mass=None,
                 label=0, parent=0, kecoef=None):
        # total number of states
        self.nstates = int(nstates)
        # dimensionality of the trajectory
        self.dim     = int(dim)
        # widths of gaussians for each dimension
        if width is None:
            self.width = np.zeros(dim)
        else:
            self.width = np.asarray(width)
        # masses associated with each dimension
        if mass is None:
            self.mass = np.zeros(dim)
        else:
            self.mass = np.array(mass)
       # the prefactor on the kinetic energy term: default to 1/2M
        if len(np.nonzero(self.mass)) == len(self.mass) and kecoef is None:
            self.kecoef = 0.5 / self.mass   
        else:
            self.kecoef = kecoef

        # unique identifier for trajectory
        self.label        = label
        # trajectory that spawned this one:
        self.parent     = parent

        # current position of the trajectory
        self.pos        = np.zeros(self.dim)
        # current momentum of the trajectory
        self.mom        = np.zeros(self.dim)
        # state trajectory exists on
        self.state      = 0
        # whether the trajectory is alive (i.e. contributes to the wavefunction)
        self.alive      = True
        # whether the trajectory is active (i.e. is being propagated)
        self.active     = True
        # amplitude of trajectory
        self.amplitude  = 0j
        # phase of the trajectory
        self.gamma      = 0.
        # time from which the death watch begini as
        self.deadtime   = -1.
        # time of last spawn
        self.last_spawn = np.zeros(self.nstates)
        # time trajectory last left coupling region
        self.exit_time  = np.zeros(self.nstates)
        # data structure to hold the pes data from the interface
        self.pes        = None

    @timings.timed
    def copy(self):
        """Copys a Trajectory object with new references."""
        new_traj = Trajectory(self.nstates, self.dim, width=self.width, mass=self.mass,
                              label=self.label, parent=self.parent, kecoef=self.kecoef)
        new_traj.state      = copy.copy(self.state)
        new_traj.alive      = copy.copy(self.alive)
        new_traj.amplitude  = copy.copy(self.amplitude)
        new_traj.gamma      = copy.copy(self.gamma)
        new_traj.deadtime   = copy.copy(self.deadtime)
        new_traj.pos        = copy.deepcopy(self.pos)
        new_traj.mom        = copy.deepcopy(self.mom)
        new_traj.last_spawn = copy.deepcopy(self.last_spawn)
        new_traj.exit_time  = copy.deepcopy(self.exit_time)
        if self.pes is not None:
            new_traj.pes = self.pes.copy()
        return new_traj

    #-------------------------------------------------------------------
    #
    # Trajectory status functions
    #
    #--------------------------------------------------------------------
    def dead(self, time, uncoupled_thresh):
        """Returns true if the trajectory is dead."""
        return self.deadtime != -1 and (time - self.deadtime) > uncoupled_thresh

    #--------------------------------------------------------------------
    #
    # reset key parameters, i.e. don't access variables directly 
    #
    #---------------------------------------------------------------------
    def set_mass(self, m_vec):
        """Set the mass vector and update kinetic energy coefficient"""
        self.mass   = m_vec

        return

    def set_width(self, w_vec):
        """Set the width vector"""
        self.width = w_vec

        return

    def set_kecoef(self, ke_vec):
        """Set the definition of the kinetic eneryg operator"""
        self.kecoef = ke_vec

        return

    #----------------------------------------------------------------------
    #
    # Functions for setting basic pes information from trajectory
    #
    #----------------------------------------------------------------------
    def update_x(self, pos):
        """Updates the position of the trajectory.
        """
        self.pos = np.array(pos)

    def update_p(self, mom):
        """Updates the momentum of the trajectory.
        """
        self.mom = np.array(mom)

    def update_phase(self, phase):
        """Updates the nuclear phase."""
        self.gamma = 0.5 * np.dot(self.x(), self.p())
        #self.gamma = phase
        #if abs(self.gamma) > 2*np.pi:
        #    self.gamma = self.gamma % 2*np.pi

    def update_amplitude(self, amplitude):
        """Updates the amplitude of the trajectory."""
        self.amplitude = amplitude

    def update_pes_info(self, new_pes):
        """Updates information about the potential energy surface."""
        self.pes = new_pes.copy()

    #-----------------------------------------------------------------------
    #
    # Functions for retrieving basic pes information from trajectory
    #
    #-----------------------------------------------------------------------
    def x(self):
        """Returns the position of the trajectory as an array."""
        return self.pos

    def p(self):
        """Returns the momentum of the trajectory as an array."""
        return self.mom

    def phase(self):
        """Returns the phase of the trajectory."""
        return self.gamma

    def masses(self):
        """Returns a vector containing masses associated with each dimension"""
        return self.mass

    def widths(self):
        """Returns a vector containing the widths of the basis functions
        along each degree of freedom."""
        return self.width

    #--------------------------------------------------------------------
    #
    # Functions to update information about the potential energy surface
    #
    #--------------------------------------------------------------------
    def energy(self, state):
        """Returns the potential energies.

        Add the energy shift right here. If not current, recompute them.
        """
        if np.linalg.norm(self.pes.get_data('geom') - self.x()) > 10.*constants.fpzero:
            print('WARNING: trajectory.energy() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes.get_data('geom')))
        #return self.pes.get_data('potential')[state] + glbl.propagate['pot_shift']
        return self.pes.get_data('potential')[state]

    def derivative(self, state_i, state_j):
        """Returns the derivative with ket state = rstate.

        Bra state assumed to be the current state.
        """
        if np.linalg.norm(self.pes.get_data('geom') - self.x()) > 10.*constants.fpzero:
            print('WARNING: trajectory.derivative() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes.get_data('geom')))
        return self.pes.get_data('derivative')[:, state_i, state_j]

    def hessian(self, state_i):
        """Returns the hessian of the potential on state state_i
        """
        if np.linalg.norm(self.pes.get_data('geom') - self.x()) > 10.*constants.fpzero:
            print('WARNING: trajectory.hessian() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes.get_data('geom')))
        return self.pes.get_data('hessian')[:, :, state_i]

    def coupling(self, state_i, state_j):
        """Returns the coupling between surfaces state_i and state_j
        """
        if np.linalg.norm(self.pes.get_data('geom') - self.x()) > 10.*constants.fpzero:
            print('WARNING: trajectory.coupling() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes.get_data('geom')))
        return self.pes.get_data('coupling')[state_i, state_j]

    def scalar_coup(self, state_i, state_j):
        """Returns the scalar coupling for Hamiltonian
           block (self.state,c_state)."""
        if 'scalar_coup' not in self.pes.avail_data():
            return 0.
        if np.linalg.norm(self.pes.get_data('geom') - self.x()) > 10.*constants.fpzero:
            print('WARNING: trajectory.scalar_coup() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes.get_data('geom')))
        return self.pes.get_data('scalar_coup')[state_i, state_j]

    def nact(self, state_i, state_j):
        """Returns the derivative coupling between adiabatic states
           block (self.state,c_state)."""
        if 'nac' not in self.pes.avail_data():
            return 0.
        if np.linalg.norm(self.pes.get_data('geom') - self.x()) > 10.*constants.fpzero:
            print('WARNING: trajectory.nact() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes.get_data('geom')))
        return self.pes.get_data('nact')[:,state_i, state_j]


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
        return sum( self.p() * self.p() * self.kecoef )

    def classical(self):
        """Returns the classical energy of the trajectory."""
        return self.potential() + self.kinetic()

    def velocity(self):
        """Returns the velocity of the trajectory."""
        return self.p() * (2. * self.kecoef)

    def force(self):
        """Returns the gradient of the trajectory state."""
        return -self.derivative(self.state, self.state)

    def phase_dot(self):
        """Returns time derivatives of the phase."""
        # d[gamma]/dt = T - V - alpha/(2M)
        return (self.kinetic() - self.potential() -
                np.dot(self.widths(), self.kecoef) )

    def same_state(self, j_state):
        """Determines if a given state is the same as the trajectory state."""
        return self.state == j_state