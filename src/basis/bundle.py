import sys
import copy
import cmath
import scipy as sp
import numpy as np
import src.dynamics.timings as timings
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.particle as particle
import src.basis.trajectory as trajectory

#
# this method is the simplest way i can see to make a copy
# of a bundle with new references. Overriding deepcopy for
# significantly more work.
#
def copy_bundle(orig_bundle):

    timings.start('bundle.copy_bundle')

    new_bundle = bundle(orig_bundle.nstates, orig_bundle.integrals)
    new_bundle.time   = copy.copy(orig_bundle.time)
    new_bundle.nalive = copy.copy(orig_bundle.nalive)
    new_bundle.ndead  = copy.copy(orig_bundle.ndead)
    new_bundle.H      = copy.deepcopy(orig_bundle.H)
    new_bundle.S      = copy.deepcopy(orig_bundle.S)
    new_bundle.Sfull  = copy.deepcopy(orig_bundle.Sfull)
    new_bundle.Sinv   = copy.deepcopy(orig_bundle.Sinv)
    new_bundle.Sdot   = copy.deepcopy(orig_bundle.Sdot)
    new_bundle.Heff   = copy.deepcopy(orig_bundle.Heff)
    for i in range(new_bundle.n_total()):
        traj_i = trajectory.copy_traj(orig_bundle.traj[i])
        new_bundle.traj.append(traj_i)
    for i in range(len(orig_bundle.cent)):
        if not orig_bundle.cent[i]:
            traj_i = trajectory.copy_traj(orig_bundle.cent[i])
        else:
            traj_i = None
        new_bundle.cent.append(traj_i)

    timings.stop('bundle.copy_bundle')

    return new_bundle

#
# return the index in the cent array of the centroid between
# trajectories i and j
#
def cent_ind(i, j):
    if i == j:
        return -1
    else:
        a = max(i,j)
        b = min(i,j)
        return int(a*(a-1)/2 + b)

#
# return the length of the centroid array for n_traj length
# traj array
#
def cent_len(n_traj):
    if n_traj == 0:
        return 0
    return int(n_traj * (n_traj - 1) / 2)

#
# Class constructor
#
class bundle:
    def __init__(self, nstates, int_defs):
        self.integrals = int_defs
        self.time = 0.
        self.nalive = 0
        self.ndead = 0
        self.nstates = int(nstates)
        self.traj  = []
        self.cent  = []
        self.H     = np.zeros((0.,0.),dtype=np.complex)
        self.S     = np.zeros((0.,0.),dtype=np.complex)
        self.Sfull = np.zeros((0.,0.),dtype=np.complex)
        self.Sinv  = np.zeros((0.,0.),dtype=np.complex)
        self.Sdot  = np.zeros((0.,0.),dtype=np.complex)
        self.Heff  = np.zeros((0.,0.),dtype=np.complex)
        try:
            self.ints = __import__('src.integrals.'+self.integrals,fromlist=['a'])
        except:
            print("BUNDLE INIT FAIL: src.integrals."+self.integrals)

    # total number of trajectories
    def n_total(self):
        return self.nalive + self.ndead

    # add trajectory to the bundle. 
    def add_trajectory(self,new_traj):
        self.traj.append(new_traj)
        self.traj[-1].alive = True
        self.nalive        += 1
        self.traj[-1].tid   = self.n_total() - 1
        self.H          = np.zeros((self.nalive,self.nalive),dtype=np.complex) 
        self.S          = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.Sfull      = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.Sinv       = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.Sdot       = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.Heff       = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        if self.ints.require_centroids:
            self.update_centroids()

    # add a set of trajectories
    def add_trajectories(self,traj_list):
        for i in range(len(traj_list)):
            self.traj.append(traj_list[i])
            self.nalive        += 1
            self.traj[-1].alive = True
            self.traj[-1].tid   = self.n_total()-1
        self.H          = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.S          = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.Sfull      = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.Sinv       = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.Sdot       = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.Heff       = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        if self.ints.require_centroids:
            self.update_centroids()
            
    # take a live trajectory and move it to the list of dead trajectories
    # it no longer contributes to H, S, etc.
    def kill_trajectory(self,tid):
        self.traj[tid].alive = False
        self.nalive          = self.nalive - 1
        self.ndead           = self.ndead + 1
        self.H          = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.S          = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.Sfull      = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.Sinv       = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.Sdot       = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.Heff       = np.zeros((self.nalive,self.nalive),dtype=np.complex)

    #
    # update the amplitudes of the trajectories in the bundle
    #
    def update_amplitudes(self, dt, n_max):
        # Solve:
        #  d/dt C = -i H C
        #
        # Solution:
        #  C(t+dt) = exp( -i H(t) dt ) C(t)
        #  C(t+dt) = exp( B )          C(t)
        #
        # Basic property of expontial:
        #   exp(B) = exp( B/n ) ** n
        #
        # The expontential is written Taylor series expansion to 4th order
        #    exp(B/n) = I + [B/n] + 1/2![B/n]**2 + 1/3![B/n]**3 + 1/4![B/n]**4
        #
        # n is varied until C_tdt is stable

        #
        # now that centroids reflect the trajectories, update matrices (mainly we just
        # need to determine the effective Hamiltonian (Heff))
        #
        self.update_matrices()

        timings.start('bundle.update_amplitudes')

        old_amp   = self.amplitudes()
        new_amp   = np.zeros(self.nalive,dtype=np.complex)
        Id        = np.identity(self.nalive,dtype=np.complex)

        B = -complex(0.,1.) * self.Heff * dt

        prev_amp = np.zeros(self.nalive,dtype=np.complex) 
        for n in range(1,n_max+1):
            Bn  = B / 2**n
            Bn2 = np.dot(Bn,Bn)
            Bn3 = np.dot(Bn2,Bn)
            Bn4 = np.dot(Bn2,Bn2)

            taylor = Id + Bn + Bn2/2.0 + Bn3/6.0 + Bn4/24.0 
            for i in range(n):
                taylor = np.dot(taylor,taylor)

            new_amp = np.dot(taylor,old_amp)
            error   = cmath.sqrt(np.sum(abs(new_amp-prev_amp)**2))
            if abs(error) < 1.e-10:
                break
            else:
                prev_amp = new_amp            
                if n == n_max:
                    sys.exit('Cannot converge amplitudes...')       
 
        cnt = -1
        for i in range(self.n_total()):
            if self.traj[i].alive:
                cnt += 1
                self.traj[i].amplitude = new_amp[cnt]

        timings.stop('bundle.update_amplitudes')

        return

    # 
    # renormalizes the amplitudes of the trajectories in the bundle
    #
    def renormalize(self):
        current_pop = self.pop() 
        norm = 1./ np.sqrt(sum(current_pop))
        for i in range(self.n_total()):
            self.traj[i].amplitude = self.traj[i].amplitude * norm
       
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
            for j in range(i):
                if abs(self.H[i,j]) > glbl.fms['hij_coup_thresh']:
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

    #
    # return amplitudes of the trajectories
    #
    def amplitudes(self):
        amps = np.zeros(self.nalive,dtype=np.complex)
        cnt = -1
        for i in range(self.n_total()):
            if self.traj[i].alive:
                cnt += 1
                amps[cnt:] = self.traj[i].amplitude
        return amps

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
            mulliken = mulliken + abs( olap * self.traj[tid].amplitude.conjugate()
                                            * self.traj[i].amplitude)
        return mulliken

    #
    # return the populations on each of the states
    # 
    def pop(self):
        pop = np.zeros(self.nstates,dtype=np.float)
        for i in range(self.n_total()):
            state = self.traj[i].state
            popii = self.traj[i].amplitude * self.traj[i].amplitude.conjugate()
            pop[state] += popii.real
            for j in range(i):
                if self.traj[i].alive != self.traj[j].alive: 
                    continue
                S_ij = self.traj[i].overlap(self.traj[j],st_orthog=True)
                popij = 2.0 * S_ij * self.traj[j].amplitude * self.traj[i].amplitude.conjugate()
                pop[state] += popij.real 
        return pop        

    #
    # return the classical potential energy of the bundle
    #  -- currently includes energy from dead trajectories as well...
    # 
    def pot_classical(self):
        weight = np.array([self.traj[i].amplitude * self.traj[i].amplitude.conjugate() for i in range(self.n_total())])
        v_int  = np.array([self.ints.v_integral(self.traj[i],self.traj[i]) for i in range(self.n_total())])
        return sum(weight * v_int).real

    #
    # return the QM (coupled) energy of the bundle,
    #  -- currently includes <live|live> and <dead|dead> contributions...
    #
    def pot_quantum(self):
        energy = 0.
        for i in range(self.n_total()):
            weight = self.traj[i].amplitude * self.traj[i].amplitude.conjugate()
            v_int  = self.ints.v_integral(self.traj[i], self.traj[i])
            energy += (weight * v_int).real
            for j in range(i):
                if self.traj[i].alive != self.traj[j].alive:
                    continue
                weight = self.traj[j].amplitude * \
                         self.traj[i].amplitude.conjugate()
                if self.ints.require_centroids:
                    v_int = self.ints.v_integral(self.traj[i],self.traj[j],
                                                 self.cent[cent_ind(i,j)])
                else:
                    v_int = self.ints.v_integral(self.traj[i],self.traj[j])
                energy += 2.0 * (weight * v_int).real
        return energy

    #
    # return the classical kinetic energy of the bundle
    # 
    def kin_classical(self):
        weight  = np.array([self.traj[i].amplitude * self.traj[i].amplitude.conjugate() for i in range(self.n_total())])
        ke_int  = np.array([self.ints.ke_integral(self.traj[i],self.traj[i]) for i in range(self.n_total())])
        return sum(weight * ke_int).real
 
    #
    # return the QM (coupled) energy of the bundle
    #
    def kin_quantum(self):
        energy = 0.
        for i in range(self.n_total()):
            weight = self.traj[i].amplitude * self.traj[i].amplitude.conjugate()
            ke_int = self.ints.ke_integral(self.traj[i],self.traj[i])
            energy += (weight * ke_int).real
            for j in range(i):
                if self.traj[i].alive !=  self.traj[j].alive:
                    continue
                weight = self.traj[j].amplitude * \
                         self.traj[i].amplitude.conjugate()
                ke_int = self.ints.ke_integral(self.traj[i],self.traj[j])
                energy += 2.0 * (weight * ke_int).real
        return energy

    # 
    # return the total classical energy of the bundle 
    #
    def tot_classical(self):
        return self.pot_classical() + self.kin_classical()

    #
    # return the total QM (coupled) energy of the bundle
    #
    def tot_quantum(self):
        return self.pot_quantum() + self.kin_quantum()

#-----------------------------------------------------------------------
# 
# Private methods/functions (called only within the class)
#
#----------------------------------------------------------------------
    # update centroids
    def update_centroids(self):

        timings.start('bundle.update_centroids')

        for i in range(self.n_total()):  
            if not self.traj[i].alive:
                continue
                wid_i = self.traj[i].widths()
                for j in range(i):
                    if not self.traj[j].alive:
                        continue
                    # first check that we have added trajectory i or j (i.e.
                    # that the cent array is long enough, if not, append the
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
                    # now update the position in phase space of the centroid
                    # if wid_i == wid_j, this is clearly just the simply mean position.
                    wid_j = self.traj[j].width() 
                    new_x = ( wid_i * self.traj[i].x() + wid_j * self.traj[j].x() ) / (wid_i + wid_j)
                    new_p = ( wid_i * self.traj[i].p() + wid_j * self.traj[j].p() ) / (wid_i + wid_j)
                    self.cent[ij_ind].update_x(new_x)
                    self.cent[ij_ind].update_p(new_p)
   
        timings.stop('bundle.update_centroids')

    # construct the Hamiltonian matrix in basis of trajectories
    def update_matrices(self):

        # make sure the centroids are up-to-date in order to evaluate
        # self.H -- if we need them
        if self.ints.require_centroids:
            self.update_centroids()
  
        timings.start('bundle.update_matrices')

        sdot_int      = self.ints.sdot_integral
        v_int         = self.ints.v_integral
        ke_int        = self.ints.ke_integral
        req_centroids = self.ints.require_centroids

        r = -1
        for i in range(self.n_total()):
            if not self.traj[i].alive:
                continue
            r += 1
            c = -1
            for j in range(i+1):
                if not self.traj[j].alive:
                    continue
                c += 1
                self.Sfull[r,c] = self.traj[i].overlap(self.traj[j])
                self.Sfull[c,r] = self.Sfull[r,c].conjugate()

                if self.traj[i].state == self.traj[j].state:
                    self.S[r,c] = self.Sfull[r,c]
                    self.S[c,r] = self.Sfull[c,r]
                else:
                    self.S[r,c] = np.complex(0.,0.)
                    self.S[c,r] = np.complex(0.,0.)

                  
                self.Sdot[r,c]  = sdot_int(self.traj[i], self.traj[j], self.S[r,c])
                self.Sdot[c,r]  = sdot_int(self.traj[j], self.traj[i], self.S[c,r])

                self.H[r,c]     = ke_int(self.traj[i], self.traj[j], self.S[r,c])
                if req_centroids:
                   self.H[r,c] +=  v_int(self.traj[i], self.traj[j],
                                         self.cent[cent_ind(i,j)],
                                         self.Sfull[r,c])
                else:
                   self.H[r,c] +=  v_int(self.traj[i], self.traj[j], self.Sfull[r,c])

                self.H[c,r]     = self.H[r,c].conjugate()
                
        # compute the S^-1, needed to compute Heff
        self.Sinv = np.linalg.pinv(self.S)
        self.Heff = np.dot( self.Sinv, self.H - np.complex(0.,1.)*self.Sdot )

        timings.stop('bundle.update_matrices')

 #-----------------------------------------------------------------------------
 #
 # functions to read/write bundle to checkpoint files
 #
 #-----------------------------------------------------------------------------
    #
    # update the log files
    #
    def update_logs(self):

        timings.start('bundle.update_logs')
 
        for i in range(self.n_total()):
            if not self.traj[i].alive:
                continue

            # trajectory files
            if glbl.fms['print_traj']:
                data = [self.time]
                data.extend(self.traj[i].x().tolist())
                data.extend(self.traj[i].p().tolist())
                data.extend([self.traj[i].phase,self.traj[i].amplitude.real,
                             self.traj[i].amplitude.imag,abs(self.traj[i].amplitude),
                             self.traj[i].state])
                fileio.print_traj_row(self.traj[i].tid,0,data)

                # potential energy
                data = [self.time]
                data.extend([self.traj[i].energy(j) for j in range(self.nstates)])
                fileio.print_traj_row(self.traj[i].tid,1,data)

                # coupling
                data = [self.time]
                data.extend([self.traj[i].coupling_norm(j) for j in range(self.nstates)])
                data.extend([self.traj[i].coup_dot_vel(j) for j in range(self.nstates)])
                fileio.print_traj_row(self.traj[i].tid,2,data)

            # print electronic structure info
            if glbl.fms['print_es']:
                # permanent dipoles
                data = [self.time]
                for j in range(self.nstates):
                    data.extend(self.traj[i].dipole(j).tolist())
                fileio.print_traj_row(self.traj[i].tid,3,data)

                # transition dipoles
                data = [self.time]
                for j in range(self.nstates):
                    for k in range(j):
                        data.extend(self.traj[i].tdipole(k,j).tolist()) 
                fileio.print_traj_row(self.traj[i].tid,4,data)

                # second moments 
                data = [self.time]
                for j in range(self.nstates):
                    data.extend(self.traj[i].sec_mom(j).tolist())
                fileio.print_traj_row(self.traj[i].tid,5,data)
 
                # atomic populations
                data = [self.time]
                for j in range(self.nstates):
                    data.extend(self.traj[i].atom_pop(j).tolist()) 
                fileio.print_traj_row(self.traj[i].tid,6,data)

                # gradients
                data = [self.time]
                data.extend(self.traj[i].derivative(self.traj[i].state).tolist())
                fileio.print_traj_row(self.traj[i].tid,7,data)

        # now dump bundle information #################################################

        # state populations
        data = [self.time]
        st_pop = self.pop().tolist()
        data.extend(self.pop().tolist())
        data.append(sum(st_pop))
        fileio.print_bund_row(0,data)

        # bundle energy
        data = [self.time, self.pot_quantum(),  self.kin_quantum(),   self.tot_quantum(),
                           self.pot_classical(),self.kin_classical(), self.tot_classical()]
        fileio.print_bund_row(1,data)

        # bundle matrices 
        if glbl.fms['print_matrices']:
            self.update_matrices()
            fileio.print_bund_mat(self.time,'s.dat',self.Sfull)
            fileio.print_bund_mat(self.time,'h.dat',self.H)
            fileio.print_bund_mat(self.time,'heff.dat',self.Heff)
            fileio.print_bund_mat(self.time,'sdot.dat',self.Sdot)

        # dump full bundle to an checkpoint file
        if glbl.fms['print_chkpt']:
            write_bundle(fileio.scr_path+'/last_step.dat','w')

        timings.stop('bundle.update_logs')
        return

    #
    # dump the bundle to file 'filename'. Mode is either 'a'(append) or 'x'(new)
    #          
    def write_bundle(self,filename,mode):

        timings.start('bundle.write_bundle')

        if mode not in ('w','a'):
            sys.exit('invalid write mode in bundle.write_bundle')
        npart = self.traj[0].n_particle
        ndim  = self.traj[0].d_particle
        with open(filename, mode) as chkpt:
            # 
            # first write out the bundle-level information
            #
            chkpt.write('------------- BEGIN BUNDLE SUMMARY --------------\n')
            chkpt.write('{:10.2f}            current time\n'.format(self.time))
            chkpt.write('{:10d}            live trajectories\n'.format(self.nalive))
            chkpt.write('{:10d}            dead trajectories\n'.format(self.ndead))
            chkpt.write('{:10d}            number of states\n'.format(self.nstates))
            chkpt.write('{:10d}            number of particles\n'.format(npart))
            chkpt.write('{:10d}            dimensions of particles\n'.format(ndim))
            #
            # Particle information common to all trajectories
            #
            for i in range(npart):
                chkpt.write('--------- common particle information --------\n')
                self.traj[0].particles[i].write_particle(chkpt)
            #
            # first write out the live trajectories. The function write_trajectory
            # can only write to a pre-existing file stream
            #
            for i in range(len(self.traj)):
                chkpt.write('-------- trajectory {:4d} --------\n'.format(i))    
                self.traj[i].write_trajectory(chkpt)
        chkpt.close()

        timings.stop('bundle.write_bundle')
        return

    #
    # Reads a bundle at time 't_restart' from a chkpt file
    #
    def read_bundle(self,filename,t_restart):   
   
        try:
            chkpt = open(filename,'r',encoding='utf-8')
        except:
            sys.exit('could not open: '+filename)
        
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
        # else, we're reading from a last_step file -- and we want to skip over 
        # the initial comment line
        else:
            t_found = True
            chkpt.readline()

        if not t_found:
            sys.exit('could not find time='+str(t_restart)+' in '+str(chkpt.name))

        # read common bundle information
        self.time    = float(chkpt.readline().split()[0])
        self.nalive       = int(chkpt.readline().split()[0])
        self.ndead        = int(chkpt.readline().split()[0])
        self.nstates = int(chkpt.readline().split()[0])
        npart        = int(chkpt.readline().split()[0])
        ndim         = int(chkpt.readline().split()[0])

        # the particle list will be the same for all trajectories
        p_list = []
        for i in range(npart):
            chkpt.readline()
            part = particle.particle(ndim,0)
            part.read_particle(chkpt)
            p_list.append(part)

        # read-in trajectories
        for i in range(self.nalive + self.ndead):
            chkpt.readline()
            t_read = trajectory.trajectory(glbl.fms['interface'],self.nstates,particles=p_list,tid=i,parent=0,n_basis=0)
            t_read.read_trajectory(chkpt)
            self.traj.append(t_read)

        # create the bundle matrices
        self.H          = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.S          = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.Sinv       = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.Sdot       = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        self.Heff       = np.zeros((self.nalive,self.nalive),dtype=np.complex)
        
        # once bundle is read, close the stream
        chkpt.close()


