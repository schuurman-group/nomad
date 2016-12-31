"""
The Trajectory object and its associated functions.
"""
import sys
import copy
import numpy as np
import src.dynamics.timings as timings
import src.fmsio.glbl as glbl

@timings.timed
def copy_cent(orig_cent):
    """Copys a Trajectory object with new references."""

    # should do more rigorous checking that "orig_cent" is actually 
    # a centroid object
    if orig_cent is None:
        return None

    new_cent = Centroid(nstates=orig_cent.nstates,
                        pstates=orig_cent.pstates,
                        dim    =orig_cent.dim,
                        width  =orig_cent.width,
                        crd_dim=orig_cent.crd_dim,
                        cid    =orig_cent.cid)
    new_cent.pos        = copy.deepcopy(orig_cent.pos)
    new_cent.mom        = copy.deepcopy(orig_cent.mom)
    new_cent.pes_geom   = copy.deepcopy(orig_cent.pes_geom)
    new_cent.poten      = copy.deepcopy(orig_cent.poten)
    new_cent.deriv      = copy.deepcopy(orig_cent.deriv)
    new_cent.pes_data   = orig_cent.interface.copy_surface(orig_cent.pes_data)
    return new_cent

class Centroid:
    """Class constructor for the Trajectory object."""
    def __init__(self, 
                 traj_i=None,
                 traj_j=None,
                 nstates=0,
                 pstates=[-1,-1], 
                 dim=0,
                 width=None, 
                 crd_dim=3,
                 cid=-1):

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
            # dimension of the coordinate system 
            #(i.e. ==3 for Cartesian, == 3N-6 for internals)
            self.crd_dim = crd_dim
            # unique identifier for trajectory
            self.cid        = cid
            # current position of the trajectory
            self.pos        = np.zeros(self.dim)
            # current momentum of the trajecotry
            self.mom        = np.zeros(self.dim)

        else:
            idi          = max(traj_i.tid, traj_j.tid)
            idj          = min(traj_i.tid, traj_j.tid)
            self.nstates = max(traj_i.nstates,traj_j.nstates)
            self.pstates = [traj_i.state, traj_j.state]
            self.dim     = max(traj_i.dim, traj_j.dim)
            self.crd_dim = max(traj_i.crd_dim, traj_j.crd_dim)
            self.cid     = -((idi * (idi - 1) / 2) + idj + 1)
            # now update the position in phase space of the centroid
            # if wid_i == wid_j, this is clearly just the simply mean
            # position.
            wid_i = traj_i.widths()
            wid_j = traj_j.widths()
            self.width = 0.5 * (wid_i + wid_j)
            self.pos = (wid_i*traj_i.x() + wid_j*traj_j.x()) / (wid_i+wid_j)
            self.mom = (wid_i*traj_i.p() + wid_j*traj_j.p()) / (wid_i+wid_j)

        # value of the potential energy
        self.poten      = np.zeros(self.nstates)
        # derivatives of the potential -- if off-diagonal, corresponds
        # to Fij (not non-adiabatic coupling vector)
        self.deriv      = np.zeros(self.dim)
        # geometry of the current potential information
        self.pes_geom   = np.zeros(self.dim)
        # name of interface to get potential information
        self.interface = __import__('src.interfaces.' +
                               glbl.fms['interface'], fromlist = ['a'])

        # data structure to hold the data from the interface
        self.pes_data  = None 

    #----------------------------------------------------------------------
    #
    # Return a new centroid object corresponding to the (j,i) complement to the
    #  current (i,j) object
    #  -- Note: this is likely more appropriate as an 'interface' routine,
    #           as centroid isn't likely to know this info
    #----------------------------------------------------------------------
    def hermitian(self):
        """Return a new centroid object corresponding to the (j,i) complement
           of the current (i,j) object"""
        new_cent = copy_cent(self)

        # if we have no data, just return new centroid object
        if new_cent.pes_data is None:
            return new_cent

        # change sign of derivative coupling
        if new_cent.pstates[0] != new_cent.pstates[1]:
            if 'deriv' in new_cent.pes_data.data_keys:
                new_cent.pes_data.grads[new_cent.pstates[1]] *= -1.
                new_cent.deriv *= -1.

        return new_cent

    #----------------------------------------------------------------------
    #
    # Functions for setting basic pes information from centroid 
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

    def update_pes(self, pes_info):
        """Updates information about the potential energy surface."""
        self.pes_data   = self.interface.copy_surface(pes_info)
        self.pes_geom   = self.pes_data.geom
        self.poten      = self.pes_data.energies
        if 'deriv' in self.pes_data.data_keys:
            self.deriv  = self.pes_data.grads[self.pstates[1]]
        
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
        if np.linalg.norm(self.pes_geom - self.x()) > glbl.fpzero:
            print('WARNING: trajectory.energy() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.tid))
        return self.poten[state] + glbl.fms['pot_shift']

    def derivative(self):
        """Returns either a gradient or derivative coupling depending 
           on the states in pstates.
        """
        if np.linalg.norm(self.pes_geom - self.x()) > glbl.fpzero:
            print('WARNING: trajectory.derivative() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.tid))
        return self.deriv

    #------------------------------------------------------------------------
    #
    # Computed quantities from the 
    #
    #------------------------------------------------------------------------
    def potential(self):
        """Returns classical potential energy of the trajectory."""
        return 0.5 * (self.energy(self.pstate[0]) + self.energy(self.pstate[1]))

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
        if self.pstates[0] != self.pstates[1]:
            return np.zeros(self.dim)
        return -self.derivative()

    def coupling_norm(self):
        """Returns the norm of the coupling vector."""
        if self.pstates[0] == self.pstates[1]:
            return 0.
        return np.linalg.norm(self.derivative())

    def coup_dot_vel(self):
        """Returns the coupling dotted with the velocity."""
        if self.pstates[0] == self.pstates[1]:
            return 0.
        return np.dot( self.velocity(), self.derivative() )

    def eff_coup(self):
        """Returns the effective coupling."""
        if self.pstates[0] == self.pstates[1]:
            return 0.
        # F.p/m
        coup = np.dot( self.velocity(), self.derivative() )
        # G
        if glbl.fms['coupling_order'] > 1:
            coup += self.scalar_coup()

        return coup

    def scalar_coup(self):
        """Returns the scalar coupling."""
        if 'scalar_coup' in self.pes_data.data_keys:
            return self.pes_data.scalar_coup[self.pstates[1]]

        return 0.
    #--------------------------------------------------------------------------
    #
    # routines to write/read trajectory from a file stream
    #
    #--------------------------------------------------------------------------
    def write_centroid(self, chkpt):
        """Writes centroid information to a file stream."""
        np.set_printoptions(precision=8, linewidth=80, suppress=False)
        chkpt.write('{:10d}            nstates\n'.format(self.nstates))
        chkpt.write('{:10d}            cent ID\n'.format(self.cid))
        chkpt.write('{:10d,:10d}            parents \n'.format(self.pstates))
        chkpt.write('# potential energy -- nstates\n')
        self.poten.tofile(chkpt, ' ', '%14.10f')
        chkpt.write('\n')
        chkpt.write('# position\n')
        self.x().tofile(chkpt, ' ', '%12.8f')
        chkpt.write('\n')
        chkpt.write('# momentum\n')
        self.p().tofile(chkpt, ' ', '%12.8f')
        chkpt.write('\n')

        # Writes out gradients
        chkpt.write('# derivative state1, state2 = {:4d}, {:4d}'
                        '\n'.format(self.pstates[0],self.pstates[1]))
        self.derivative().tofile(chkpt, ' ', '%16.10e')
        chkpt.write('\n')

    def read_centroid(self,chkpt):
        """Reads the trajectory information from a file.

        This assumes the trajectory invoking this function has been
        initially correctly and can hold all the information.
        """
        self.nstates   = int(chkpt.readline().split()[0])
        self.cid       = int(chkpt.readline().split()[0])
        self.pstates   = int(chkpt.readline().split()[0])

        chkpt.readline()
        # potential energy -- nstates
        self.poten = np.fromstring(chkpt.readline(), sep=' ', dtype=float)
        chkpt.readline()
        # position
        self.update_x(np.fromstring(chkpt.readline(), sep=' ', dtype=float))
        chkpt.readline()
        # momentum
        self.update_p(np.fromstring(chkpt.readline(), sep=' ', dtype=float))

