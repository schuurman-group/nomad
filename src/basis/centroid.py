"""
The Centroid object and its associated functions.
"""
import sys
import copy
import numpy as np
import src.dynamics.timings as timings
import src.fmsio.glbl as glbl


def cent_label(itraj_id, jtraj_id):
    """return the centroid id for centroid between traj_i, traj_j"""
    idi          = max(itraj_id, jtraj_id)
    idj          = min(itraj_id, jtraj_id)
    return -((idi * (idi - 1) // 2) + idj + 1)


class Centroid:
    """Class constructor for the Centroid object."""
    def __init__(self, traj_i=None, traj_j=None, nstates=0, pstates=[-1,-1],
                 dim=0, width=None, label=-1):
        if traj_i is None or traj_j is None:
            # total number of states
            self.nstates = int(nstates)
            # parent states
            self.pstates = pstates
            # dimensionality of the centroid
            self.dim     = int(dim)
            # widths of gaussians for each dimension
            if width is None:
                self.width = np.zeros(dim)
            else:
                self.width = np.asarray(width)
            # unique identifier for centroid
            self.label     = label
            # current position of the centroid
            self.pos     = np.zeros(self.dim)
            # current momentum of the centroid
            self.mom     = np.zeros(self.dim)
            # labels of parent trajectories
            self.parent  = np.zeros(2, dtype=int)
        else:
            idi          = max(traj_i.label, traj_j.label)
            idj          = min(traj_i.label, traj_j.label)
            self.parent  = np.array([idj, idi], dtype=int)
            self.nstates = max(traj_i.nstates,traj_j.nstates)
            self.pstates = [traj_i.state, traj_j.state]
            self.dim     = max(traj_i.dim, traj_j.dim)
            self.label     = -((idi * (idi - 1) // 2) + idj + 1)
            # now update the position in phase space of the centroid
            # if wid_i == wid_j, this is clearly just the simply mean
            # position.
            wid_i = traj_i.widths()
            wid_j = traj_j.widths()
            self.width = 0.5 * (wid_i + wid_j)
            self.pos = (wid_i*traj_i.x() + wid_j*traj_j.x()) / (wid_i+wid_j)
            self.mom = (wid_i*traj_i.p() + wid_j*traj_j.p()) / (wid_i+wid_j)

        # data structure to hold the data from the interface
        self.pes_data  = None

    @timings.timed
    def copy(self):
        """Copys a Centroid object with new references."""
        new_cent = Centroid(nstates=self.nstates, pstates=self.pstates,
                            dim=self.dim, width=self.width,label=self.label)
        new_cent.pos = copy.deepcopy(self.pos)
        new_cent.mom = copy.deepcopy(self.mom)
        if self.pes_data is not None:
            new_cent.pes_data = self.pes_data.copy()
        return new_cent

    #----------------------------------------------------------------------
    #
    # Functions for setting basic pes information from centroid
    #
    #----------------------------------------------------------------------
    def update_x(self, traj_i, traj_j):
        """Updates the position of the centroid."""
        wid_i    = traj_i.widths()
        wid_j    = traj_j.widths()
        self.pos = (wid_i*traj_i.x() + wid_j*traj_j.x()) / (wid_i+wid_j)

    def update_p(self, traj_i, traj_j):
        """Updates the momentum of the centroid."""
        wid_i    = traj_i.widths()
        wid_j    = traj_j.widths()
        self.mom = (wid_i*traj_i.p() + wid_j*traj_j.p()) / (wid_i+wid_j)

    def update_pes_info(self, pes_info):
        """Updates information about the potential energy surface."""
        self.pes_data = pes_info.copy()

    #-----------------------------------------------------------------------
    #
    # Functions for retrieving basic pes information from centroid
    #
    #-----------------------------------------------------------------------
    def x(self):
        """Returns the position of the centroid as an array."""
        return self.pos

    def p(self):
        """Returns the momentum of the centroid as an array."""
        return self.mom

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
        if np.linalg.norm(self.pes_data.geom - self.x()) > glbl.constants['fpzero']:
            print('WARNING: centroid.energy() called, ' +
                  'but pes_geom != centroid.x(). ID=' + str(self.label))
        return self.pes_data.potential[state] + glbl.propagate['pot_shift']

    def derivative(self, state_i, state_j):
        """Returns either a gradient or derivative coupling depending
           on the states in pstates.
        """
        if np.linalg.norm(self.pes_data.geom - self.x()) > glbl.constants['fpzero']:
            print('WARNING: trajectory.derivative() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label))
        return self.pes_data.deriv[:,state_i, state_j]

    def scalar_coup(self, state_i, state_j):
        """Returns the scalar coupling."""
        if np.linalg.norm(self.pes_data.geom - self.x()) > glbl.constants['fpzero']:
            print('WARNING: trajectory.scalar_coup() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label))
        if 'scalar_coup' in self.pes_data.data_keys:
            return self.pes_data.scalar_coup[state_i, state_j]
        return 0.

    #------------------------------------------------------------------------
    #
    # Computed quantities from the
    #
    #------------------------------------------------------------------------
    def potential(self):
        """Returns classical potential energy of the centroid."""
        return 0.5 * (self.energy(self.pstate[0]) + self.energy(self.pstate[1]))

    def kinetic(self):
        """Returns classical kinetic energy of the centroid."""
        return sum( self.p() * self.p() / (2. * self.masses()))

    def classical(self):
        """Returns the classical energy of the centroid."""
        return self.potential() + self.kinetic()

    def velocity(self):
        """Returns the velocity of the centroid."""
        return self.p() / self.masses()

    def force(self):
        """Returns the gradient of the centroid state."""
        if not same_state():
            return np.zeros(self.dim)
        return -self.derivative(self.pstates[0], self.pstates[1])

    def coupling_norm(self):
        """Returns the norm of the coupling vector."""
        if self.same_state():
            return 0.
        return np.linalg.norm(self.derivative(self.pstates[0], self.pstates[1]))

    def coup_dot_vel(self):
        """Returns the coupling dotted with the velocity."""
        if self.same_state():
            return 0.
        return np.dot( self.velocity(),
                       self.derivative(self.pstates[0], self.pstates[1]) )

    def eff_coup(self):
        """Returns the effective coupling."""
        if self.same_state():
            return 0.
        # F.p/m
        coup = self.coup_dot_vel()
        # G
        if glbl.interface['coupling_order'] > 1:
            coup += self.scalar_coup(pstates[0], pstates[1])
        return coup

    def same_state(self):
        """Determines if both trajectories are on the same state."""
        return self.pstate[0] == self.pstate[1]

