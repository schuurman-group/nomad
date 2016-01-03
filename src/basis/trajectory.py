import cmath
import numpy as np
import src.basis.particle as particle
class trajectory:

    def __init__(self,particle_list,interface,nstates,tid=0,parent=0,n_basis=0):
        try:
            self.pes = __import__('src.interfaces.'+interface,fromlist=['NA'])
        except:
            print("INTERFACE FAIL: "+iterface)
        # unique identifier for trajectory
        self.tid        = tid
        # trajectory that spawned this one:
        self.parent     = parent         
        # total number of states
        self.nstates    = nstates        
        # state trajectory exists on
        self.state      = 0      
        # list of particles in the trajectory
        self.particles  = particle_list
        # number of particles comprising the trajectory
        self.n_particle = len(particle_list) 
        # dimension of the particles comprising the trajectory
        self.d_particle = particle_list[0].dim
        # whether trajectory is alive (i.e. propagated)
        self.alive      = True
        # wheterh trajectory is a centroid
        self.centroid   = False
        # amplitude of trajectory
        self.amplitude  = complex(0.,0.)
        # phase of the trajectory
        self.phase      = 0.
        # time from which the death watch begins
        self.deadtime   = -1.            
        # time of last spawn
        self.last_spawn = np.zeros(self.nstates)
        # time trajectory last left coupling region
        self.exit_time  = np.zeros(self.nstates)
        # if not zero, coupled to traj=array value
        self.spawn_coup = np.zeros(self.nstates)
        # number of mos
        self.nbf        = n_basis              
        # value of the potential energy
        self.poten      = np.zeros(self.nstates)  
        # store the obitals at each step
        self.orbitals   = np.zeros((self.nbf,self.nbf))
        # derivatives of the potential -- if off-diagonal, corresponds to Fij (not non-adiabatic coupling vector)
        self.deriv      = np.zeros((self.nstates,self.n_particle*self.d_particle)) 
        # dipoles and transition dipoles
        self.dipoles    = np.zeros((self.nstates,self.d_particle))      
        # second moment tensor for each state
        self.quadpoles  = np.zeros((self.nstates,6))
        # charges on the atoms
        self.charges    = np.zeros((self.nstates,self.n_particle))
   
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
    # Updates the position of the particles in trajectory. Flips up2date switch to False
    #
    def update_x(self,pos):
        for i in range(self.n_particle):
            self.particles[i].x = pos[i*self.d_particle : (i+1)*self.d_particle-1]

    #
    # Update the momentum of the particles in the trajectory. Flips up2date switch to False
    #
    def update_p(self,mom):
        for i in range(self.n_particle):
            self.particles[i].p = mom[i*self.d_particle : (i+1)*self.d_particle-1]

    #
    # update the nuclear phase 
    #
    def update_phase(self,phase):
        self.phase = phase
        if self.phase > 2*np.pi:
            self.phase = self.phase - int(self.phase/(2 * np.pi)) * 2 * n.pi

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
        width = np.zeros(self.n_particle * self.d_particle)
        icnt = -1
        for i in range(self.n_particle):
            for j in range(self.d_particle):
                icnt += 1
                width[icnt] = self.particles[i].width
        return width

    #
    # return the potential energies. If not current, recompute them
    #
    def energy(self,istate=0):
        self.poten = self.pes.energy(self.tid, self.particles, self.state)
        if istate:
            return self.poten[istate]
        else:
            return self.poten

    #
    # return the derivative with ket state = rstate. bra state assumed to be 
    # the current state
    #
    def derivative(self,rstate):
        self.deriv[rstate,:] = self.pes.derivative(self.tid, self.particles, self.state, rstate)
        return self.deriv[rstate,:]
    
    #
    #
    #
    def dipole(self,rstate):
        self.dipole[rstate,:] = self.pes.dipole(self.tid, self.particles, self.state, rstate)
        return self.dipole[rstate,:]

    #
    #
    #
    def quadpole(self,rstate):
        self.quadpole[rstate,:] = self.pes.quadpole(self.tid, self.particles, self.state, rstate)
        return self.quadpole

    #
    #
    #
    def tdipole(self,rstate):
        self.tdipole[rstate,:] = self.pes.tdipole(self.tid, self.particles, self.state, rstate)

    #
    #
    # 
    def charges(self,rstate):
        self.charges[rstate,:] = self.pes.charges(self.tid, self.particles, self.state, rstate)
        return self.charges[rstate,:]

    #
    #
    #
    def orbitals(self):
        self.orbitals = self.pes.orbitals(self.tid, self.particles, self.state)
        return self.orbitals
    
    #-------------------------------------------------------------------------
    #
    # Computed quantities from the trajectory
    #
    #-------------------------------------------------------------------------
    #
    # Classical potential energy
    #
    def potential(self):
        return self.energy(self,istate=self.state)

    #
    # Classical kinetic energy
    #
    def kinetic(self):
         return 0.5 * sum( self.p() * self.p() / self.masses ) 

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
    # Return the coupling.velocity
    #
    def coup_dot_vel(self,c_state):
        if c_state == self.state:
           return 0.
        return abs(np.vdot( self.velocity(), self.derative[c_state,:] ))
        
    #-----------------------------------------------------------------------------
    #
    # primitive integral routines
    #
    #-----------------------------------------------------------------------------
    #
    # overlap of two trajectories
    #
    def overlap(self,other,st_orthog=False):
        if st_orthog and self.state != other.state:
            return complex(0.,0.)         
        S = cmath.exp( complex(0.,1.)*(self.phase - other.phase) )
        for i in range(self.n_particle):
            S = S * self.particles[i].overlap(other.particles[i])
        return S

    #
    # del/dp matrix element between two trajectories 
    #
    def deldp(self,other):
        dpval = np.zeros(self.n_particle * self.d_particle,dtype=np.cfloat)
        for i in range(self.n_particle):
            dpval[self.d_particle*i:self.d_particle*(i+1)] = self.particles[i].deldp(other.particles[i])
        return dpval * self.overlap(other)

    #
    # del/dx matrix element between two trajectories
    #
    def deldx(self,other):
        dxval = np.zeros(self.n_particle * self.d_particle,dtype=np.cfloat)
        for i in range(self.n_particle):
            dxval[self.d_particle*i:self.d_particle*(i+1)] = self.particles[i].deldx(other.particles[i])
        return dxval * self.overlap(other)

    #
    # this is the expectation value of the momentum operator over the 2 x mass
    # this appears in the equations of motion on the off diagonal coupling
    # different states together theough the NACME
    #
    def deldx_m(self,other):
        dxval = np.zeros(self.n_particle * self.d_particle,dtype=np.cfloat)
        for i in range(self.n_particles):
            dxval[self.d_particle*i:self.d_particle*(i+1)] = self.particles[i].deldx(other.particles[i]) / \
                                                             self.particles[i].mass
        return dxval * self.overlap(other)

   #--------------------------------------------------------------------------
   # 
   # routines to write/read trajectory from a file stream
   #
   #--------------------------------------------------------------------------
    def write_trajectory(self,chkpt):
        chkpt.write('{:5s}             alive'.format(self.alive))
        chkpt.write('{:10d}            nstates'.format(self.nstates))
        chkpt.write('{:10d}            traj ID'.format(self.tid))
        chkpt.write('{:10d}            state  '.format(self.state))
        chkpt.write('{:10d}            parent ID'.format(self.parent))
        chkpt.write('{:10d}            n basis function'.format(self.nbf))
        chkpt.write('{:8.2f}           dead time'.format(self.deadtime))
        chkpt.write('{:16.12f}         phase'.format(self.phase))
        chkpt.write('{:16.12f}         amplitude'.format(self.amplitude))
        chkpt.write('# potential energy -- nstates')
        self.poten.tofile(chkpt,' ',':14.10f')
        chkpt.write('# exit coupling region')
        self.exit_time.tofile(chkpt,' ',':8.2f')
        chkpt.write('# last spawn')
        self.spawn_time.tofile(chkpt,' ',':8.2f')
        chkpt.write('# currently coupled')
        self.spawn_coup.tofile(chkpt,' ',':10d')
        chkpt.write('# position')
        self.position().tofile(chkpt,' ',':12.8f')
        chkpt.write('# momentum')
        self.momentum().tofile(chkpt,' ',':12.8f')
        # Writes out dipole moments in cartesian coordinates
        chkpt.write("# dipoles (n=state is permanent dipole, "  
                                "others are transition dipoles)")
        for i in range(self.nstates):
            chkpt.write('# n = {:4d}    '.format(i))
            self.dipoles[i,:].tofile(chkpt,' ',':10.6f')
        # Writes out dipole moments
        chkpt.write("# derivative matrix (n=state is gradient, "
                                "others are nad coupling)")
        for i in range(self.nstates):
            chkpt.write('# n = {:4d}    '.format(i))
            self.deriv[i,:].tofile(chkpt,' ',':16.10e')
        #
        chkpt.write('# molecular orbitals')
        self.orbitals.tofile(chkpt,' ',':12.8e')

    #
    # Read trajectory from file. This assumes the trajectory invoking this
    # function has been initially correctly and can hold all the information
    #
    def read_trajectory(self,chkpt):
        self.alive     = bool(chkpt.readline()[0])
        self.nstates   = int(chkpt.readline()[0])
        self.tid       = int(chkpt.readline()[0])
        self.state     = int(chkpt.readline()[0])
        self.parent    = int(chkpt.readline()[0])
        self.nbf       = int(chkpt.readline()[0])
        self.deadtime  = float(chkpt.readline()[0])
        self.phase     = float(chkpt.readline()[0])
        self.amplitude = complex(chkpt.readline()[0])

        chkpt.readline() # potential energy -- nstates
        self.poten = np.fromfile(chkpt,float,self.nstates)
        chkpt.readline() # exit coupling region 
        self.exit_time = np.fromstring(chkpt.readline())
        chkpt.readline() # last spawn 
        self.spawn_time = np.fromstring(chkpt.readline())
        chkpt.readline() # currently coupled
        self.spawn_coup = np.fromstring(chkpt.readline())

        chkpt.readline() # position
        pos = np.fromstring(chkpt.readline())
        self.update_position(pos)
        chkpt.readline() # momentum
        mom = np.fromstring(chkpt.readline())
        self.update_momentum(mom)

        chkpt.readline() # dipoles
        for i in range(self.nstates):
            chkpt.readline()
            self.dipoles[i,:] = np.fromstring(chkpt.readline())

        chkpt.readline() # derivatives
        for i in range(self.nstates):
            chkpt.readline()
            self.deriv[i,:] = np.fromstring(chkpt.readline())

        chkpt.readline() # orbitals
        self.orbitals = np.fromfile(chkpt,float,self.nbf**2)

