import sys
import copy
import cmath
import numpy as np
import src.dynamics.timings as timings
import src.fmsio.glbl as glbl
import src.basis.particle as particle

def copy_traj(orig_traj):
    timings.start('trajectory.copy_traj')
    new_traj = trajectory(orig_traj.nstates)
    p_list = []
    for i in range(orig_traj.n_particle):
        p_list.append(particle.copy_part(orig_traj.particles[i]))
    new_traj.update_particles(p_list)
    new_traj.tid        = copy.copy(orig_traj.tid)
    new_traj.parent     = copy.copy(orig_traj.parent)
    new_traj.state      = copy.copy(orig_traj.state)
    new_traj.c_state    = copy.copy(orig_traj.c_state)
    new_traj.alive      = copy.copy(orig_traj.alive)
    new_traj.amplitude  = copy.copy(orig_traj.amplitude)
    new_traj.phase      = copy.copy(orig_traj.phase)
    new_traj.deadline   = copy.copy(orig_traj.deadtime)
    new_traj.nbf        = copy.copy(orig_traj.nbf)
    new_traj.last_spawn = copy.deepcopy(orig_traj.last_spawn)
    new_traj.exit_time  = copy.deepcopy(orig_traj.exit_time)
    new_traj.spawn_coup = copy.deepcopy(orig_traj.spawn_coup)
    new_traj.pes_geom   = copy.deepcopy(orig_traj.pes_geom)
    new_traj.poten      = copy.deepcopy(orig_traj.poten)
    new_traj.deriv      = copy.deepcopy(orig_traj.deriv)
    new_traj.dipoles    = copy.deepcopy(orig_traj.dipoles)
    new_traj.sec_moms   = copy.deepcopy(orig_traj.sec_moms)
    new_traj.atom_pops  = copy.deepcopy(orig_traj.atom_pops)
    timings.stop('trajectory.copy_traj')
    return new_traj

class trajectory:

    def __init__(self,nstates,particles=None,tid=0,parent=0,n_basis=0):
        # total number of states
        self.nstates    = nstates
        # allow for population of trajectory particles via set_particles
        if not particles:
            self.particles=[]
            self.d_particle = 0
        else:
            self.particles = particles
            self.d_particle = self.particles[0].dim 
        # unique identifier for trajectory
        self.tid        = tid
        # trajectory that spawned this one:
        self.parent     = parent         
        # state trajectory exists on
        self.state      = 0      
        # if a centroid, state for second trajectory
        self.c_state    = -1
        # number of particles comprising the trajectory
        self.n_particle = len(self.particles) 
        # whether trajectory is alive (i.e. propagated)
        self.alive      = True
        # amplitude of trajectory
        self.amplitude  = complex(0.,0.) 
        # phase of the trajectory
        self.phase      = 0.
        # time from which the death watch begini as
        self.deadtime   = -1.            
        # time of last spawn
        self.last_spawn = np.zeros(self.nstates)
        # time trajectory last left coupling region
        self.exit_time  = np.zeros(self.nstates)
        # if not zero, coupled to traj=array value
        self.spawn_coup = np.zeros(self.nstates)
        # number of mos
        self.nbf        = n_basis              
        # geometry of the current potential information
        self.pes_geom   = np.zeros(self.n_particle*self.d_particle, dtype=np.float)
        # value of the potential energy
        self.poten      = np.zeros(self.nstates, dtype=np.float)  
        # derivatives of the potential -- if off-diagonal, corresponds to Fij (not non-adiabatic coupling vector)
        self.deriv      = np.zeros((self.nstates,self.n_particle*self.d_particle),dtype=np.float) 
        # dipoles and transition dipoles
        self.dipoles    = np.zeros((self.nstates,self.nstates,self.d_particle),dtype=np.float)      
        # second moment tensor for each state
        self.sec_moms   = np.zeros((self.nstates,self.d_particle),dtype=np.float)
        # electronic populations on the atoms
        self.atom_pops  = np.zeros((self.nstates,self.n_particle),dtype=np.float)

    #-------------------------------------------------------------------
    #
    # Trajectory status functions
    #
    #--------------------------------------------------------------------
    def dead(time, uncoupled_thresh):
        if self.deadtime != -1:
            if (time - self.deadtime) > uncoupled_thresh:
                return True
        return False   

    #----------------------------------------------------------------------
    #
    # Functions for setting basic pes information from trajectory
    #
    #----------------------------------------------------------------------
    #
    # update particle list
    #
    def update_particles(self,p_list):
        self.particles  = p_list
        self.d_particle = self.particles[0].dim
        self.n_particle = len(self.particles)
        self.deriv      = np.zeros((self.nstates,self.n_particle*self.d_particle))
        self.dipoles    = np.zeros((self.nstates,self.nstates,self.d_particle))
        self.sec_moms   = np.zeros((self.nstates,self.d_particle))
        self.atom_pops  = np.zeros((self.nstates,self.n_particle))
        return

    #
    # Updates the position of the particles in trajectory. Flips up2date switch to False
    #
    def update_x(self,pos):
        for i in range(self.n_particle):
            self.particles[i].x = pos[i*self.d_particle : (i+1)*self.d_particle]
        return

    #
    # Update the momentum of the particles in the trajectory. Flips up2date switch to False
    #
    def update_p(self,mom):
        for i in range(self.n_particle):
            self.particles[i].p = mom[i*self.d_particle : (i+1)*self.d_particle]
        return

    #
    # update the nuclear phase 
    #
    def update_phase(self, phase):
        self.phase = phase
        if abs(self.phase) > 2*np.pi:
            self.phase = self.phase - int(self.phase/(2. * np.pi)) * 2. * np.pi
        return

    #
    # update the amplitude on this trajectory
    #
    def update_amplitude(self, amplitude):
        self.amplitude = amplitude
        return

    #
    # update information about the potential energy surface
    #
    def update_pes(self, pes_info):
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
        return

    #-----------------------------------------------------------------------
    # 
    # Functions for retrieving basic pes information from trajectory
    #
    #-----------------------------------------------------------------------
    #
    # Returns the position of the particles in the trajectory as an array 
    #
    def x(self):
        return np.fromiter((self.particles[i].x[j] 
                           for i in range(self.n_particle) 
                           for j in range(self.d_particle)),dtype=np.float)

    #
    # Returns the momentum of the particles in the trajectory as an array 
    #
    def p(self):
        return np.fromiter((self.particles[i].p[j]
                          for i in range(self.n_particle)
                          for j in range(self.d_particle)),dtype=np.float)

    #
    # return a vector containing masses of particles
    # 
    def masses(self):
        return np.fromiter((self.particles[i].mass 
                          for i in range(self.n_particle) 
                          for j in range(self.d_particle)),dtype=np.float)

    #
    # return a vector containing the widths of the b.f. along each d.o.f
    #
    def widths(self):
        width = np.fromiter((self.particles[i].width
                          for i in range(self.n_particle)
                          for j in range(self.d_particle)),dtype=np.float)
        return width

    #--------------------------------------------------------------------
    #
    # Functions to update information about the potential energy surface
    #
    #--------------------------------------------------------------------
    
    #
    # return the potential energies. Add the energy shift right here. If not current, recompute them
    #
    def energy(self,state):
        if np.linalg.norm(self.pes_geom - self.x()) > glbl.fpzero:
            print("WARNING: trajectory.energy() called, "+
                  "but pes_geom != trajectory.x(). ID="+str(self.tid))
        return self.poten[state] + glbl.fms['pot_shift']

    #
    # return the derivative with ket state = rstate. bra state assumed to be 
    # the current state
    #
    def derivative(self,state):
        if np.linalg.norm(self.pes_geom - self.x()) > glbl.fpzero:
            print("WARNING: trajectory.derivative() called, "+
                  "but pes_geom != trajectory.x(). ID="+str(self.tid))
        return self.deriv[state,:]
        
    #
    # permanent dipoles
    #
    def dipole(self,state):
        if np.linalg.norm(self.pes_geom - self.x()) > glbl.fpzero:
            print("WARNING: trajectory.dipole() called, "+
                  "but pes_geom != trajectory.x(). ID="+str(self.tid))
        return self.dipoles[state,state,:]
        
    #
    # transition dipoles
    #
    def tdipole(self,state_i, state_j):
        if np.linalg.norm(self.pes_geom - self.x()) > glbl.fpzero:
            print("WARNING: trajectory.tdipole() called, "+
                  "but pes_geom != trajectory.x(). ID="+str(self.tid))
        return self.dipoles[state_i,state_j,:]
         
    #
    # second moments
    #
    def sec_mom(self,state):
        if np.linalg.norm(self.pes_geom - self.x()) > glbl.fpzero:
            print("WARNING: trajectory.sec_mom() called, "+
                  "but pes_geom != trajectory.x(). ID="+str(self.tid))
        return self.sec_moms[state,:]

    #
    # atomic populations 
    # 
    def atom_pop(self,state):
        if np.linalg.norm(self.pes_geom - self.x()) > glbl.fpzero:
            print("WARNING: trajectory.atom_pop() called, "+
                  "but pes_geom != trajectory.x(). ID="+str(self.tid))
        return self.atom_pops[state,:]

    #
    #
    #
#    def orbitals(self):
#        return self.pes.orbitals(self.tid, self.particles, self.state)
    
    #-------------------------------------------------------------------------
    #
    # Computed quantities from the trajectory
    #
    #-------------------------------------------------------------------------
    #
    # Classical potential energy
    #
    def potential(self):
        return self.energy(self.state)

    #
    # Classical kinetic energy
    #
    def kinetic(self):
         return 0.5 * sum( self.p() * self.p() / self.masses() ) 

    #
    # Returns the classical energy of the trajectory
    # 
    def classical(self):
         return self.potential() + self.kinetic()

    #
    # return momentum / mass
    #
    def velocity(self):
        return self.p() / self.masses()

    #
    # Return the gradient of the self.state 
    #
    def force(self):
        return -self.derivative(self.state)        

    #
    # Return time derivative of the phase
    #
    def phase_dot(self):
        # d[gamma]/dt = T - V - alpha/(2M)
        return self.kinetic() - self.potential() - 0.5*sum(self.widths()/self.masses())       

    #
    # norm of the coupling vector
    #
    def coupling_norm(self,rstate):
        if self.state == rstate:
            return 0.
        return np.linalg.norm(self.derivative(rstate))

    #
    # Return the coupling.velocity
    #
    def coup_dot_vel(self,c_state):
        if self.state == c_state:
           return 0.
        return np.dot( self.velocity(), self.derivative(c_state) )
        
    #-----------------------------------------------------------------------------
    #
    # primitive integral routines
    #
    #-----------------------------------------------------------------------------
    #
    # overlap of two trajectories
    #
    def overlap(self,other,st_orthog=False):
#        timings.start('trajectory.overlap')
        if st_orthog and self.state != other.state:
            return np.complex(0.,0.)         
        S = cmath.exp( np.complex(0.,1.)*(other.phase - self.phase) )
        for i in range(self.n_particle):
            S = S * self.particles[i].overlap(other.particles[i])
#        timings.stop('trajectory.overlap')
        return S

    #
    # del/dp matrix element between two trajectories 
    #
    def deldp(self, other, S_ij=None):
#        timings.start('trajectory.deldp')
        if not S_ij:
            S_ij = self.overlap(other,st_orthog=True)
        dpval = np.zeros(self.n_particle * self.d_particle,dtype=np.cfloat)
        for i in range(self.n_particle):
            dpval[self.d_particle*i:self.d_particle*(i+1)] = self.particles[i].deldp(other.particles[i])
#        timings.stop('trajectory.deldp')
        return dpval * S_ij

    #
    # del/dx matrix element between two trajectories
    #
    def deldx(self, other, S_ij=None):
#        timings.start('trajectory.deldx')
        if not S_ij:
            S_ij = self.overlap(other,st_orthog=True)
        dxval = np.zeros(self.n_particle * self.d_particle,dtype=np.cfloat)
        for i in range(self.n_particle):
            dxval[self.d_particle*i:self.d_particle*(i+1)] = self.particles[i].deldx(other.particles[i])
#        timings.stop('trajectory.deldx')
        return dxval * S_ij

    #
    # this is the expectation value of the momentum operator over the 2 x mass
    # this appears in the equations of motion on the off diagonal coupling
    # different states together theough the NACME
    #
    def deldx_m(self, other, S_ij=None):
#        timings.start('trajectory.deldx_m')
        if not S_ij:
            S_ij = self.overlap(other,st_orthog=True)
        dxval = np.zeros(self.n_particle * self.d_particle,dtype=np.cfloat)
        for i in range(self.n_particles):
            dxval[self.d_particle*i:self.d_particle*(i+1)] = self.particles[i].deldx(other.particles[i]) / \
                                                             self.particles[i].mass
#        timings.stop('trajectory.deldx_m')
        return dxval * S_ij

   #--------------------------------------------------------------------------
   # 
   # routines to write/read trajectory from a file stream
   #
   #--------------------------------------------------------------------------
    def write_trajectory(self,chkpt):
        np.set_printoptions(precision=8,linewidth=80,suppress=False)
        chkpt.write('     {:5s}            alive\n'.format(str(self.alive)))
        chkpt.write('{:10d}            nstates\n'.format(self.nstates))
        chkpt.write('{:10d}            traj ID\n'.format(self.tid))
        chkpt.write('{:10d}            state\n'.format(self.state))
        chkpt.write('{:10d}            parent ID\n'.format(self.parent))
        chkpt.write('{:10d}            n basis function\n'.format(self.nbf))
        chkpt.write('{:10.2f}            dead time\n'.format(self.deadtime))
        chkpt.write('{:16.12f}      phase\n'.format(self.phase))
        chkpt.write('{:16.12f}         amplitude\n'.format(self.amplitude))
        chkpt.write('# potential energy -- nstates\n')
        self.poten.tofile(chkpt,' ','%14.10f')
        chkpt.write('\n')
        chkpt.write('# exit coupling region\n')
        self.exit_time.tofile(chkpt,' ','%8.2f')
        chkpt.write('\n')
        chkpt.write('# last spawn\n')
        self.last_spawn.tofile(chkpt,' ','%8.2f')
        chkpt.write('\n')
        chkpt.write('# currently coupled\n')
        self.spawn_coup.tofile(chkpt,' ','%8d')
        chkpt.write('\n')
        chkpt.write('# position\n')
        self.x().tofile(chkpt,' ','%12.8f')
        chkpt.write('\n')
        chkpt.write('# momentum\n')
        self.p().tofile(chkpt,' ','%12.8f')
        chkpt.write('\n')

        # Writes out dipole moments in cartesian coordinates
        init_states = [0, self.state]
        for i in init_states:
            for j in range(self.nstates):
                if j==i or j in init_states[0:i]:
                    continue
                chkpt.write('# dipoles state1, state2 = {0:4d}, {1:4d}\n'.format(j,i))
                self.dipoles[j,i,:].tofile(chkpt,' ','%10.6f')
                chkpt.write('\n')

        # Writes out gradients 
        for i in range(self.nstates):
            chkpt.write('# derivatives state1, state2 = {0:4d}, {1:4d}\n'.format(self.state,i))
            self.deriv[i,:].tofile(chkpt,' ','%16.10e')
            chkpt.write('\n')

        # write out second moments 
        for i in range(self.nstates):
            chkpt.write('# second moments, state = {0:4d}\n'.format(i))
            self.sec_moms[i,:].tofile(chkpt,' ','%16.10e')
            chkpt.write('\n')

        # write out atomic populations 
        for i in range(self.nstates):
            chkpt.write('# atomic populations, state = {0:4d}\n'.format(i))
            self.atom_pops[i,:].tofile(chkpt,' ','%16.10e')
            chkpt.write('\n')

    #
    # Read trajectory from file. This assumes the trajectory invoking this
    # function has been initially correctly and can hold all the information
    #
    def read_trajectory(self,chkpt):
        self.alive     = bool(chkpt.readline().split()[0])
        self.nstates   = int(chkpt.readline().split()[0])
        self.tid       = int(chkpt.readline().split()[0])
        self.state     = int(chkpt.readline().split()[0])
        self.parent    = int(chkpt.readline().split()[0])
        self.nbf       = int(chkpt.readline().split()[0])
        self.deadtime  = float(chkpt.readline().split()[0])
        self.phase     = float(chkpt.readline().split()[0])
        self.amplitude = complex(chkpt.readline().split()[0])

        chkpt.readline() # potential energy -- nstates
        self.poten = np.fromstring(chkpt.readline(),sep=' ',dtype=float)
        chkpt.readline() # exit coupling region 
        self.exit_time = np.fromstring(chkpt.readline(),sep=' ',dtype=float)
        chkpt.readline() # last spawn 
        self.spawn_time = np.fromstring(chkpt.readline(),sep=' ',dtype=float)
        chkpt.readline() # currently coupled
        self.spawn_coup = np.fromstring(chkpt.readline(),sep=' ',dtype=float)

        chkpt.readline() # position
        self.update_x(np.fromstring(chkpt.readline(),sep=' ',dtype=float))
        chkpt.readline() # momentum
        self.update_p(np.fromstring(chkpt.readline(),sep=' ',dtype=float))

        for i in range(self.nstates):
            for j in range(i+1):
                chkpt.readline()
                self.dipoles[i,j,:] = np.fromstring(chkpt.readline(),sep=' ',dtype=float)

        for i in range(self.nstates):
            chkpt.readline()
            self.deriv[i,:] = np.fromstring(chkpt.readline(),sep=' ',dtype=float)

        chkpt.readline() # orbitals

