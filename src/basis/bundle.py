import sys
import scipy as sp
import numpy as np
import src.basis.trajectory as trajectory
class bundle:
    def __init__(self,surface_rep):
        try:
            self.ints = __import__('src.basis.int_'+surface_rep,fromlist=['a'])
        except:
            print("BUNDLE INIT FAIL: src.basis.int_"+surface_rep)
        self.nalive = 0
        self.ndead = 0
        self.nstates = 0
        self.traj = []
        self.cent = [] 
        self.H    = np.zeros((0,0),dtype=np.cfloat)
        self.S    = np.zeros((0,0),dtype=np.cfloat)
        self.Sinv = np.zeros((0,0),dtype=np.cfloat)
        self.Sdot = np.zeros((0,0),dtype=np.cfloat)
        self.Heff = np.zeros((0,0),dtype=np.cfloat)

    # total number of trajectories
    def n_total(self):
        return self.nalive + self.ndead

    # add trajectory to the bundle. 
    def add_trajectory(self,new_traj):
        self.traj.append(new_traj)
        self.traj[-1].alive = True
        self.nalive         = self.nalive + 1
        self.bundle_current = False
        new_matrix = np.zeros((self.nalive,self.nalive),dtype=np.cfloat)
        if self.nalive == 1:
            self.H          = new_matrix
            self.S          = new_matrix
            self.Sinv       = new_matrix
            self.Sdot       = new_matrix
            self.Heff       = new_matrix
        else:
            new_matrix[0:,0:]   = self.H
            self.H              = new_matrix
            new_matrix[0:,0:]   = self.S
            self.S              = new_matrix
            new_matrix[0:,0:]   = self.Sinv
            self.Sinv           = new_matrix
            new_matrix[0:,0:]   = self.Sdot
            self.Sdot           = new_matrix
            new_matrix[0:,0:]   = self.Heff
            self.Heff           = new_matrix

    # take a live trajectory and move it to the list of dead trajectories
    # it no longer contributes to H, S, etc.
    def kill_trajectory(self,tid):
        self.traj[tid].alive = False
        self.nalive          = self.nalive - 1
        self.ndead           = self.ndead + 1
        self.bundle_current  = False
        for i in range(2):
            self.H    = np.delete(self.H,index,i)
            self.S    = np.delete(self.S,index,i)
            self.Sinv = np.delete(self.S,index,i)
            self.sdot = np.delete(self.Sdot,index,i)
            self.Heff = np.delete(self.Heff,index,i)

    # update the bundle matrices
    def update_bundle(self):
        self.prune()
        self.update_matrices()

    # update centroids
    def update_centroid(self,traj1,traj2):    
        pass

    # construct the Hamiltonian matrix in basis of trajectories
    def update_matrices(self):
        r = -1
        for i in range(len(self.traj)):
            if self.traj[i].alive:
                r += 1
                self.S[r,r] = self.traj[i].overlap(self.traj[i])
                self.Sdot[r,r] = self.ints.sdot_integral(self.traj[i], 
                                                    self.traj[i])
                self.H[r,r] = self.ints.ke_integral(self.traj[i],
                                               self.traj[i]) +  \
                              self.ints.v_integral(self.traj[i])
                c = -1
                for j in range(i-1):
                    if self.traj[j].alive:
                        c += 1
                        self.S[r,c] = self.traj[i].overlap(self.traj[j])
                        self.S[c,r] = conjugate(self.S(r,c))
                        self.Sdot[r,c] = self.ints.sdot_integral(self.traj[i],
                                                            self.traj[j])
                        self.Sdot[c,r] = conjugate(self.Sdot(r,c))
                        self.H[r,c] = self.ints.ke_integral(self.traj[i],
                                                       self.traj[j]) + \
                                      self.ints.v_integral(self.traj[i],
                                                      self.traj[j],
                                                      self.cent[ij])
                        self.H[c,r] = conjugate(self.H[r,c])
        # compute the S^-1, needed to compute Heff
        self.Sinv = sp.linalg.pinv(S)                 
        self.Heff = np.dot( self.Sinv, self.H - complex(0.,1.)*self.Sdot )

    #
    # update the amplitudes of the trajectories in the bundle
    #
    def propagate_amplitudes(self, dt, n_max):
        # Solve:
        #  d/dt C = -i H C
        #
        # Solution:
        #  C(t+dt) = exp( -i H(t) dt ) C(t)
        #  C_t     = exp( B )          Ct_tdt
        #
        # Basic property of expontial:
        #   exp(B) = exp( B/n ) ** n
        #          = exp( Bn  ) ** n
        # This reduces the size of the exponential we must take.
        #
        # The expontential is written Taylor series expansion to 4th order
        #    exp(Bn) = I + Bn + 1/2 B2**2 + 1/3! Bn**3 + 1/4! Bn**4
        #
        # n is varied until C_tdt is stable
        old_amp   = self.amplitudes()
        new_amp   = np.zeros(self.nalive)
        Id        = np.zeros((self.nalive,self.nalive),dtype=np.cfloat)
        for i in range(nalive):
            Id[i,i] = complex(1.,0.)

        B = -complex(0.,1.) * self.Heff * dt
        amp_prev = complex(0.,0.)

        for n in range(n_max):
            Bn  = B / 2**n
            Bn2 = Bn.Bn
            Bn3 = Bn2.Bn
            Bn4 = Bn2.Bn2

            taylor = Id + Bn + Bn2/2.0 + Bn3/6.0 + Bn4/24.0 
            taylor_prod = taylor
            for i in range(n):
                taylor_prod = taylor_prod.taylor_prod

            new_amp = taylor_prod.old_amp
            error   = cmath.sqrt(np.sum(abs(new_amp-prev_amp)**2))
            if error < 1.e-10:
                break
            else:
                prev_amp = new_amp            

        cnt = -1
        for i in range(self.n_total):
            if traj[i].alive:
                cnt += 1
                traj[i].amplitude = new_amp[cnt]
        return

    #
    # return amplitudes of the trajectories
    #
    def amplitudes(self):
        amps = np.zeros(self.nalive)
        cnt = -1
        for i in range(self.n_total()):
            if traj[i].alive:
                cnt += 1
                amps[cnt] = traj[i].amplitude
        return

    #
    # kills trajectories that are dead
    #
    def prune(self):
        for i in range(self.nalive):
            continue

    #
    # returns true if we are in a regime of coupled trajectories
    #
    def in_coupled_regime(self):
    
        # check if trajectories are coupled
        for i in range(self.nalive):
            for j in range(i-1):
                if abs(self.H[i,j]) > variable['coup_thresh']:
                    return True

        # check if any trajectories exceed NAD threshold
        for i in range(self.nalive):
            for j in range(self.nstates):
                if self.traj[i].coup_dot_vel(j) > variable['nad_thresh']:
                    return True

        # else, return false
        return False

    #
    # return the populations on each of the states
    # 
    def pop(self):
        pop = np.zeros(self.nstates)
        for i in range(self.n_total()):
            state = self.traj[i].state
            for j in range(i-1):
                if self.traj[i].alive != self.traj[j].alive:
                    continue
                olap = self.traj[i].overlap(self.traj[j])
                pop[state] = pop(state) + float( olap * 
                                       self.traj[i].amplitude * 
                                       conjugate(self.traj[j].amplitude) )
            pop[state] = pop(state) + \
                         self.traj[i].amplitude * \
                         conjugate(self.traj[j].amplitude)
        return pop        

    #
    # return the Mulliken-like population 
    #
    def mulliken_pop(self,tid):
        mulliken = 0.

        if not self.traj[tid].alive:
            return mulliken 
        for i in range(len(self.traj)):
            if not self.traj[i].alive:
               continue
            olap = self.traj[tid].overlap(self.traj[i])                      
            mulliken = mulliken + abs(conjugate(self.traj[tid].amplitude)
                                       * olap * self.traj[i].amplitude)
        return mulliken

 #-----------------------------------------------------------------------------
 #
 # functions to read/write bundle to checkpoint files
 #
 #-----------------------------------------------------------------------------
    #
    # dump the bundle to file 'filename'. Mode is either 'a'(append) or 'x'(new)
    #          
    def write_bundle(self,filname,mode):
        if mode not in ('x','a'):
            sys.exit('invalid write mode in bundle.write_bundle')
        npart = self.traj[0].n_particle
        ndim  = self.traj[0].ndim
        with open(filename, mode) as chkpt:
            # 
            # first write out the bundle-level information
            #
            chkpt.write('------------- BEGIN BUNDLE SUMMARY --------------')
            chkpt.write('{:8.2f}            current time'.format(self.time))
            chkpt.write('{:10d}            live trajectories'.format(self.nalive))
            chkpt.write('{:10d}            dead trajectories'.format(self.ndead))
            chkpt.write('{:10d}            number of states'.format(self.nstates))
            chkpt.write('{:10d}            number of particles'.format(npart))
            chkpt.write('{:10d}            dimensions of particles'.format(ndim))
            #
            # Particle information common to all trajectories
            #
            for i in range(npart):
                chkpt.write('--------- common particle information --------')
                self.traj[0].particles[i].write_particle(chkpt)
            #
            # first write out the live trajectories. The function write_trajectory
            # can only write to a pre-existing file stream
            #
            for i in range(len(self.traj)):
                chkpt.write('-------- trajectory {:4d} --------'.format(i))    
                self.traj[i].write_trajectory(chkpt)
        chkpt.close()

    #
    # Reads a bundle at time 't_restart' from a chkpt file
    #
    def read_chkpt(self,filename,t_restart):   
        t_found = False
        with open(filename,'r') as chkpt:
            last_pos = chkpt.tell()
            for line in chkpt:
                if 'current time' in line:
                    if float(line[0]) == t_restart:
                        break
        # populate the bundle with the correct number of trajectories
        traj_template = trajectory.trajectory(0,0,self.nstates,0,npart,ndim)
        for i in range(npart):
            traj_template.add_particle(plist[i])
        for i in range(self.nalive + self.ndead):
            self.add_trajectory(traj_template)
        
        # read-in trajectories
        for i in range(self.alive + self.ndead):
            chkpt.readline() # comment: trajectory X
            self.trajectory[i].read_trajectory(chkpt)

        # once bundle is read, close the stream
        chkpt.close()


