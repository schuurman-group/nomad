import numpy as np
import particle
class trajectory:

    def _init_(self,interface,parent,nstates,particle_list,n_basis=0):
        try:
            pes = __import__('..interface.'+interface)
        except:
            print("INTERFACE FAIL: "+iterface)
        # trajectory that spawned this one:
        self.parent     = parent         
        # total number of states
        self.nstates    = nstates        
        # state trajectory exists on
        self.state      = 0      
        # number of particles comprising the trajectory
        self.n_particle = len(particle_list) 
        # dimension of the particles comprising the trajectory
        self.d_particle = particle_list[0].dim 
        # whether trajectory is alive (i.e. propagated)
        self.alive      = True
        # time from which the death watch begins
        self.deadtime   = -1.            
        # whether current pes information is accurate
        self.up2date    = dict(
                          orbitals=False,
                          poten=False,
                          derivative=[False]*self.nstates,
                          dipole=[False]*self.nstates,
                          quadpole=[False]*self.nstates,
                          charges=[False]*self.nstates
                          )                               
        # amplitude of trajectory
        self.amplitude  = complex(0.,0.) 
        # phase of the trajectory
        self.phase      = 0.             
        # number of mos
        self.nbf        = n_basis              
        # list of particles in the trajectory
        self.particles  = particle_list      
        # time of last spawn
        self.spawn_time = np.zeros(self.nstates) 
        # time trajectory last left coupling region
        self.exit_time  = np.zeros(self.nstates) 
        # if not zero, coupled to traj=array value
        self.spawn_coup = np.zeros(self.nstates) 
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
        self.charges    = np.zeros(self.nstates,self.n_particle)
  
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
            self.particles[i].x = pos(i*self.d_particle:(i+1)*self.d_particle-1)
        self.up2date = dict.fromkeys(self.up2date,False)

    #
    # Update the momentum of the particles in the trajectory. Flips up2date switch to False
    #
    def update_p(self,mom):
        for i in range(self.n_particle):
            self.particles[i].p = mom(i*self.d_particle:(i+1)*self.d_particle-1)

    #
    # update the nuclear phase 
    #
    def update_phase(self,phase):
        self.phase = phase
        if self.phase > 2*np.pi
            self.phase = self.phase - int(self.phase/(2 * np.pi)) * 2 * n.pi

    # 
    # update the complex amplitude on this trajectory
    #
    def update_amp(self,amp):
        self.amplitude = amp

    #-----------------------------------------------------------------------
    # 
    # Functions for retrieving basic pes information from trajectory
    #
    #-----------------------------------------------------------------------
    #
    # Returns the position of the particles in the trajectory as an array 
    #
    def x(self):
        pos = np.zeros(self.n_particle * self.d_particle)
        for i in range(self.n_particle):
            pos[i*self.d_particle:(i+1)*self.d_particle-1] = self.particles[i].x
        return pos

    #
    # Returns the momentum of the particles in the trajectory as an array 
    #
    def p(self):
        mom = np.zeros(self.n_particle * self.d_particle)
        for i in range(self.n_particle):
            mom[i*self.d_particle:(i+1)*self.d_particle-1] = self.particles[i].p
        return mom

    #
    # return a vector containing masses of particles
    # 
    def masses(self):
        mass = np.zeros(self.n_particle * self.d_particle)
        imass = np.empty(self.d_particle)
        for i in range(self.n_particle):
            mass[i*self.d_particle:(i+1)*self.d_particle-1] = \
                                    imass.fill(self.particle[i].mass) 
        return mass

    #
    # return the potential energies. If not current, recompute them
    #
    def energy(self,istate=0):
        if not self.up2date['energy']:
            self.poten = pes.energy(self.particles,self.nstates)
            self.up2date['energy'] = True
        if istate:
            return self.poten[istate]
        else
            return self.poten

    #
    # return the derivative with ket state = rstate. bra state assumed to be 
    # the current state
    #
    def derivative(self,rstate):
        if not self.up2date['derivative'][rstate]:
           self.deriv[rstate,:] = pes.derivative(self.particles,self.state,rstate)
           self.up2date['derivative'][rstate] = True
        return self.deriv(rstate,:)
    
    #
    #
    #
    def dipole(self,rstate):
        if not self.up2date['dipole'][rstate]:
           self.dipole[rstate,:] = pes.dipole(self.particles,self.state,rstate)
           self.up2date['dipole'][rstate] = True
        return self.dipole(rstate,:)

    #
    #
    #
    def quadpole(self):
        if not self.up2date['quadpole'][rstate]:
           self.quadpole[rstate,:] = pes.quadpole(self.particles,self.state)
           self.up2date['quadpole'][rstate] = True
        return self.quadpole

    #
    #
    # 
    def charges(self,rstate):
        if not self.up2date['charges'][rstate]:
           self.charges[rstate,:] = pes.charges(self.particles,rstate)
           self.up2date['charges'][rstate] = True
        return self.charges[rstate,:]

    #
    #
    #
    def orbitals(self):
        if not self.up2date['orbitals']:
           self.orbitals = pes.orbitals(self.particles)
           self.up2date['orbitals'] = True
        return self.orbitals

    #
    # return momentum / mass
    #
    def velocity(self):
        return self.p / self.masses

    #
    # Return the gradient of the self.state 
    #
    def force(self):
        return -self.derivative(self.state,:)         

    #-----------------------------------------------------------------------------
    #
    # Integral routines
    #
    #-----------------------------------------------------------------------------
    #
    # overlap of two trajectories
    #
    def overlap(self,other):
         S = exp( complex(0.,1.)*(self.phase - other.phase) )
         for i in range(self.nparticles):
             S = S * self.particles[i].overlap(other.particles[i])
         return S

    #
    # del/dp matrix element between two particles
    #
    def deldp(self,other):
         dpval = np.zeros(self.nparticles,'Complex')
         for i in range(self.nparticles):
             dpval[i] = self.particles[i].deldp(other.particles[i])
         return dpval * overlap(self,other)

    #
    # del/dx matrix element between two particles
    #
    def deldx(self,other):
         dxval = np.zeros(self.nparticles,'Complex')
         for i in range(self.nparticles):
             dxval[i] = self.particles[i].deldx(other.particles[i])
         return dxval * overlap(self,other)

    #
    # this is the expectation value of the momentum operator over the 2 x mass
    # this appears in the equations of motion on the off diagonal coupling
    # different states together theough the NACME
    #
    def deldx_m(self,other):
         dxval = np.zeros(self.nparticles,'Complex')
         for i in range(self.nparticles):
             dxval[i] = self.particles[i].deldx(other.particles[i]) /  \
                        self.particles[i].mass
         return dxval * overlap(self,other)

    #
    # potential coupling matrix element between two trajectories
    #
    def v_integral(self,other=None,centroid=None):
         if not other :
             return self.energy(self.state) 
         elif self.state == other.state:
             return self.overlap(other) * centroid.energy(self.state)
         elif not self.state != other.state:
             fij = centroid.deriv(self.state,other.state)
             return np.vdot(fij, deldx_m(self,other)
         else:
             print 'unknown state'
    #
    # kinetic energy integral over trajectories
    #
    def ke_integral(self,other):
         ke = complex(0.,0.)
         if self.state == other.state:
             for i in range(self.nparticles):
                 ke = ke - self.particles[i].deld2x(other.particles[i]) /  \
                           (2.0*self.particles[i].mass)
             return ke * overlap(self,other)
         else:
             return ke

    #
    # return the matrix element over the delS/dt operator
    #
    def sdot_integral(self,other):
         sdot = (-vdot( traj_velocity(other), self.deldx(other) )   \
                 +vdot( other.deriv(other.state)  , self.deldp(other) )   \
                 +complex(0.,1.) * traj_phasedot(other) * overlap(self,other)
         return sdot


   #--------------------------------------------------------------------------
   # 
   # routines to write/read trajectory from a file stream
   #
   #--------------------------------------------------------------------------
    def write_trajectory(self,chkpt):
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
        chkpt.write('# dipoles (n=state is permanent dipole, 
                                others are transition dipoles)')
        for i in range(self.nstates)
            chkpt.write('# n = {:4d}    '.format(i))
            self.dipoles(i,:).tofile(chkpt,' ',':10.6f')
        # Writes out dipole moments
        chkpt.write('# derivative matrix (n=state is gradient, 
                                          others are nad coupling)')
        for i in range(self.nstates)
            chkpt.write('# n = {:4d}    '.format(i))
            self.deriv(i,:).tofile(chkpt,' ',':16.10e')
        #
        chkpt.write('# molecular orbitals')
        self.orbitals.tofile(chkpt,' ',':12.8e')

    #
    # Read trajectory from file. This assumes the trajectory invoking this
    # function has been initially correctly and can hold all the information
    #
    def read_trajectory(self,chkpt):
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
        for i in range(self.nstates)
            chkpt.readline()
            self.dipoles(i,:) = np.fromstring(chkpt.readline())

        chkpt.readline() # derivatives
        for i in range(self.nstates)
            chkpt.readline()
            self.deriv(i,:) = np.fromstring(chkpt.readline())

        chkpt.readline() # orbitals
        self.orbitals = np.fromfile(chkpt,float,self.nbf**2)

