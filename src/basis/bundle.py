class bundle:
    import sys
    import numpy as np
    import trajectory
    def _init_(self):
        self.nalive = 0
        self.ndead = 0
        self.nstates = 0
        self.trajectories = np.array([])
        self.centroids = np.array([])
        self.H = np.array([[]])
        self.S = np.array([[]])
        self.Sdot = np.array([[]])
        self.Heff = np.array([[]])

    # add trajectory to the bundle. 
    def add_trajectory(traj):
        self.trajectory.append(traj)
        self.bundle_current = False
        new_row = np.zeros(self.nalive)
        new_col = np.zeros(self.nalive+1)
        self.H = np.concatenate(self.H,new_row,axis=0)
        self.H = np.concatenate(self.H,new_col,axis=1)
        self.S = np.concatenate(self.H,new_row,axis=0)
        self.S = np.concatenate(self.H,new_col,axis=1)
        self.sdot = np.concatenate(self.H,new_row,axis=0)
        self.sdot = np.concatenate(self.H,new_col,axis=1)
        self.Heff = np.concatenate(self.H,new_row,axis=0)
        self.Heff = np.concatenate(self.H,new_col,axis=1)
        self.nalive = self.nalive + 1

    # take a live trajectory and move it to the list of dead trajectories
    # it no longer contributes to H, S, etc.
    def kill_trajectory(tid):
        self.trajectories[tid].alive = False
        self.H = np.delete(self.H,index,0)
        self.S = np.delete(self.S,index,0)
        self.sdot = np.delete(self.Sdot,index,0)
        self.Heff = np.delete(self.Heff,index,0)
        self.H = np.delete(self.H,index,1)
        self.S = np.delete(self.S,index,1)
        self.sdot = np.delete(self.Sdot,index,1)
        self.Heff = np.delete(self.Heff,index,1)
        self.ntraj = self.ntraj - 1
        self.bundle_current = False

    # update the bundle matrices
    def update_bundle():
        self.update_matrices()

    # construct the Hamiltonian matrix in basis of trajectories
    def update_matrices():
        r = -1
        c = -1
        for i in range(self.trajectories.size)
            if(self.trajectories[i].alive):
                r += 1
                self.S(r,r) = self.trajectory[i].overlap(self.trajectory[j])
                self.H(r,r) = self.trajectory[i].ke_integral(self.trajectory[j])
                            + self.trajectory[i].v_integral(self.trajectory[j])
                for j in range(i-1):
                    if(self.trajectories[j].alive):
                        c += 1
                        self.S(r,c) = self.trajectory[i].overlap(self.trajectory[j])
                        self.S(c,r) = conjugate(self.S(r,c))
                        self.H(r,c) = self.trajectory[i].ke_integral(self.trajectory[j])
                                    + self.trajectory[i].v_integral(self.trajectory[j])
                        self.H(c,r) = conjugate(self.H(r,c))

 #-----------------------------------------------------------------------------
 #
 # functions to read/write bundle to checkpoint files
 #
 #-----------------------------------------------------------------------------
    #
    # dump the bundle to file 'filename'. Mode is either 'a'(append) or 'x'(new)
    #          
    def write_bundle(filname,mode):
        if mode not in ('x','a'):
            sys.exit('invalid write mode in bundle.write_bundle')
        npart = self.trajectory[0].nparticle
        ndim  = self.trajectory[0].ndim
        with open(filename, mode) as chkpt:
            # 
            # first write out the bundle-level information
            #
            chkpt.write('------------- BEGIN BUNDLE SUMMARY --------------')
            chkpt.write('{:8.2f}            current time'.format(self.time))
            chkpt.write('{:10d}            live trajectories'.format(self.ntraj))
            chkpt.write('{:10d}            dead trajectories'.format(self.ndead))
            chkpt.write('{:10d}            number of states'.format(self.nstates))
            chkpt.write('{:10d}            number of particles'.format(npart))
            chkpt.write('{:10d}            dimensions of particles'.format(ndim))
            #
            # Particle information common to all trajectories
            #
            for i in range(npart):
                chkpt.write('--------- common particle information --------')
                self.trajectory[0].particles[i].write_particle(chkpt)
            #
            # first write out the live trajectories. The function write_trajectory
            # can only write to a pre-existing file stream
            #
            for i in range(self.ntraj):
                chkpt.write('-------- live trajectory {:4d} --------'.format(i))    
                self.trajectory[i].write_trajectory(chkpt)
            #
            # now write the dead trajectories
            #
            for i in range(self.ndead)
                chkpt.write('-------- dead trajectory {:4d} --------'.format(i))
                self.dead_trajectory[i].write_trajectory(chkpt)
        chkpt.close()

    #
    # Reads a bundle at time 't_restart' from a chkpt file
    #
    def read_chkpt(filename,t_restart):   
        t_found = False
        with open(filename,'r') as chkpt
            last_pos = chkpt.tell()
            for line in chkpt:
                if 'current time' in line:
                    if float(line[0]) == t_restart:
                        t_found = True
                        chkpt.seek(last_pos)
                        break
                last_pos = chkpt.tell()
            if(t_found):
                read_bundle(stream=chkpt)
            else:
                chkpt.close()
                sys.exit('time={:8.2f} not found in {s}'.format(t_restart,filename))                

    #
    # read bundle file and load information into bundle. Any existing bundle information
    # will be overwritten. Assumes 
    #
    def read_bundle(fname=None,stream=None):
        if not stream:
            close_on_exit = True
            with open(fname,'r') as chkpt
        else:
            close_on_exit = False
            chkpt = stream

        line = chkpt.readline()  # comment line
        self.time = float(chkpt.readline()[0])
        self.ntraj = int(chkpt.readline()[0])
        self.ndead = int(chkpt.readline()[0])
        self.nstates = int(chkpt.readline()[0])
        npart = int(chkpt.readline()[0])
        ndim = int(chkpt.readline()[0])

        p_list   = ()
        p_global = particle(ndim,0)       
        # read particle lines
        chkpt.readline() # comment: common particle information 
        for i in range(npart)
          p_global.read_particle(chkpt)
          plist.append(pread)

        # populate the bundle with the correct number of trajectories
        traj_template = trajectory(0,0,self.nstates,0,npart,ndim)
        for i in range(npart)
            traj_template.add_particle(plist[i])
        for i in range(self.ntraj)
            self.add_trajectory(traj_template)
        for i in range(self.ndead)
            self.add_trajectory(traj_template)
        
        # read live trajectories
        for i in range(self.ntraj)
            chkpt.readline() # comment: live trajectory X
            self.trajectory[i].read_trajectory(chkpt)

        # read dead trajectories
        for i in range(self.ndead)
            chkpt.readline() # comment: dead trajectory X
            self.trajectory[ntraj+i].read_trajectory(chkpt)
        for i in range(self.ndead)
            self.kill_trajectory(ntraj+i)

        # once bundle is read, close the stream
        chkpt.close()







