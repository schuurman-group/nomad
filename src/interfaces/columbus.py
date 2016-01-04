#
# routines for running a columbus computation
#
import sys
import os
import shutil
import pathlib
import subprocess
import numpy as np
import src.fmsio.glbl as glbl
import src.basis.particle as particle
import src.basis.trajectory as trajectory
import src.basis.bundle as bundle 

# path to the top level of scratch directory
scr_path = ''
# path to columbus input files
input_path = ''
# path to location of 'work'/'restart' directories
work_path = ''
# path to the location of restart files (i.e. mocoef files and civfl)
restart_path = ''

n_atoms      = 0
n_cart       = 0
n_drt        = 1
n_orbs       = 0
n_mcstates   = 0
n_cistates   = 0
mrci_lvl     = 0
mem_str      = ''
current_geom = dict()
energies     = dict()
charges      = dict()
dipoles      = dict()
quadpoles    = dict()
gradients    = dict()

#----------------------------------------------------------------
# 
# Functions called from interface object
#
#----------------------------------------------------------------
#
# read the columbus input and get everything setup to run columbus
# calculations
#
def init_interface():
    global columbus_path, scr_path, input_path, work_path, restart_path, \
           n_atoms, n_cart, n_orbs, n_mcstates, n_cistates, mrci_lvl, \
           mem_str

    # confirm that we can see the COLUMBUS installation (pull the value
    # COLUMBUS environment variable)
    columbus_path = os.environ['COLUMBUS']
    if not os.path.isfile(columbus_path+'/ciudg.x'):
        print("Cannot find COLUMBUS executables in: "+columbus_path)
        sys.exit()
    # ensure scratch directory exists
    scr_path     = os.environ['TMPDIR']
    if os.path.exists(scr_path):
        shutil.rmtree(scr_path)
        os.makedirs(scr_path)
    # ensure COLUMBUS input files are present locally
    if not os.path.exists('input'):
        print("Cannot find COLUMBUS input files in: input")
        sysexit()

    # setup working directories
    input_path    = scr_path+'/input'
    work_path     = scr_path+'/work'
    restart_path  = scr_path+'/restart'

    if os.path.exists(input_path):
        shutil.rmtree(input_path)
    if os.path.exists(work_path):
        shutil.rmtree(work_path)
    if os.path.exists(restart_path):
        shutil.rmtree(restart_path)
    os.makedirs(input_path)
    os.makedirs(work_path)
    os.makedirs(restart_path)

    # copy input directory to scratch and copy file contents to work directory
    for item in os.listdir('input'):
        local_file = os.path.join('input', item)
        input_file = os.path.join(input_path, item)
        work_file  = os.path.join(work_path, item)
        shutil.copy2(local_file, input_file)
        shutil.copy2(local_file, work_file)

    # now -- pull information from columbus input
    n_atoms                = file_len('input/geom')
    n_cart                 = 3 * n_atoms
    n_orbs                 = int(read_pipe_keyword('input/cidrtmsin',
                                                'orbitals per irrep'))
    n_mcstates             = int(read_nlist_keyword('input/mcscfin',
                                                'NAVST'))
    n_cistates              = int(read_nlist_keyword('input/ciudgin.drt1',
                                                'NROOT'))
    mrci_lvl               = int(read_pipe_keyword('input/cidrtmsin',
                                                 'maximum excitation level'))

    # all COLUMBUS modules will be run with the amount of meomry specified by mem_per_core
    mem_str = str(glbl.columbus['mem_per_core'])

    # generate one time input files for columbus calculations
    make_one_time_input()

#
# Initialize the bundle with the initial trajectories at time t=0. Trajectories
#  may contain interface-specific information (i.e. "number of orbitals"). This 
#  approach is taken to maintain modularity. The main driver program should have
#  no idea what an "orbital" is...
#
def populate_bundle(master,geom_list,amp_list):
    for i in range(len(geom_list)):
        print("NSTATES="+str(glbl.fms['n_states']))
        master.add_trajectory(trajectory.trajectory(
                              geom_list[i],
                              glbl.fms['interface'],
                              glbl.fms['n_states'],
                              tid=i,
                              parent=0,
                              n_basis=n_orbs))
        master.traj[i].amplitude = amp_list[i]

#
# returns the energy at the specified geometry. If value on file 
#  not current, or we don't care about saving data -- recompute
#
# geom is a list of particles
def energy(tid, geom, t_state, rstate):
    global energies

    if not in_cache(tid,geom):
        if tid >= 0:
            run_trajectory(tid,geom,t_state)
        else:
            run_centroid(tid,geom,t_state,t_state)
    try:
        return energies[tid][rstate]
    except:
        print("ERROR in fetch_energy")       
        sys.exit("ERROR in columbus module fetching energy")

#
# returns the MOs as an numpy array
#
def orbitals(tid,geom,t_state):
    if not in_cache(tid,geom):
        if tid >= 0:
            run_trajectory(tid,geom,t_state)
        else:
            run_centroid(tid,geom,t_state,t_state)
    try:
        return load_orbitals(tid)
    except:
        print("ERROR in fetch_orbitals")     
        sys.exit("ERROR in columbus module fetching orbitals")

#
# return gradient. If lstate == rstate, gradient on state lstate. Else
#   returns non-adiabatic coupling vector
#
def derivative(tid,geom,t_state,lstate,rstate):
    global gradients 

    if not in_cache(tid,geom):
        if tid >= 0:
            run_trajectory(tid,geom,t_state)
        else:
            run_centroid(tid,geom,lstate,rstate)
    try:
        return gradients[tid][lstate,rstate,:]

    except:
        print("ERROR in fetch_gradients")     
        sys.exit("ERROR in columbus module fetching gradients")

#
# if lstate != rstate, corresponds to transition dipole
#
def dipole(tid,geom,t_state,lstate,rstate):
    global tdipoles

    if not in_cache(tid,geom):
        if tid >=0:
            run_trajectory(tid,geom,t_state)
        else:
            print("invalid id for trajectory: "+str(tid))
            sys.exit("ERROR in columbus module[tdipole] -- invalid id") 
    try:
        return dipoles[tid][lstate,rstate,:]
    except:
        print("ERROR in fetch_dipole")
        sys.exit("ERROR in columbus module fetching dipoles")

#
# return second moment tensor for state=state
#
def quadrupole(tid,geom,t_state,rstate):
    global quadpoles

    if not in_cache(tid,geom):
        if tid >= 0:
            run_trajectory(tid,geom,t_state)
        else:
            print("invalid id for trajectory: "+str(tid))
            sys.exit("ERROR in columbus module[qpole] -- invalid id")
    try:
        return quadpoles[tid][rstate,:]
    except:
        print("ERROR in fetch_quadpole")     
        sys.exit("ERROR in columbus module fetching quadpole")

#
#
#
def atomic_charges(tid,geom,t_state, rstate):
    global charges

    if not in_cache(tid,geom):
        if tid >= 0:
            run_trajectory(tid,geom,t_state)
        else:
            print("invalid id for trajectory: "+str(tid))
            sys.exit("ERROR in columbus module[charges] -- invalid id")
    try:
        return charges[tid][rstate,:]
    except:
        print("ERROR in fetch_charges")     
        sys.exit("ERROR in columbus module fetching charges")

#----------------------------------------------------------------
#
#  "Private" functions
#
#----------------------------------------------------------------
def in_cache(tid,geom):
    global current_geom, n_atoms

    if tid not in current_geom:
        return False
    print("geom1="+str(geom[0].x[0])+" "+str(geom[0].x[1])+" "+str(geom[0].x[2])+" ")
    print("geom2="+str(geom[1].x[0])+" "+str(geom[1].x[1])+" "+str(geom[1].x[2])+" ")
    g = np.fromiter((geom[i].x[j] for i in range(n_atoms) for j in range(3)),np.float)
    if np.linalg.norm(g - current_geom[tid]) <= glbl.fpzero:
        return True
    return False 

#              
# For the columbus module, since gradients are not particularly
# time consuming, it's easier (and probably faster) to compute
# EVERYTHING at once (i.e. all energies, all gradients, all properties)
# Thus, if electronic structure information is not up2date, all methods
# call the same routine: run_single_point.
#
#  This routine will:
#    1. Compute an MCSCF/MRCI energy
#    2. Compute all couplings
#
def run_trajectory(tid,geom,tstate):
    global n_atoms, current_geom

    # write geometry to file
    write_col_geom(geom)

    # generate integrals
    generate_integrals(tid)

    # run mcscf
    run_col_mcscf(tid,tstate)
  
    # run mrci, if necessary
    run_col_mrci(tid,tstate)

    # run properties, dipoles, etc.
    run_col_multipole(tid,tstate)

    # run transition dipoles
    run_col_tdipole(tid,tstate)

    # compute gradient on current state
    run_col_gradient(tid,tstate)

    # run coupling to other states
    run_col_coupling(tid,tstate)

    # save restart files
    make_col_restart(tid)

    # update the geometry in the cache
    g = np.fromiter((geom[i].x[j] for i in range(n_atoms) for j in range(3)),np.float)
    current_geom[tid] = g

#
# For a centroid we really only need an energy (if both trajectories
#  are on the same state), or a coupling (if on different states)
#
def run_centroid(tid,geom,lstate,rstate):
    global n_atoms, current_geom

    # write geometry to file
    write_col_geom(geom)

    # generate integrals
    generate_integrals(tid)

    # run mcscf
    run_col_mcscf(tid,tstate)

    # run mrci, if necessary
    run_col_mrci(tid,tstate)

    if lstate != rstate:
        # run coupling to other states
        run_col_coupling(tid,tstate)

    # save restart files
    make_col_restart(tid)

    # update the geometry in the cache
    g = np.fromiter((geom[i].x[j] for i in range(n_atoms) for j in range(3)),np.float)
    current_geom[tid] = g

#----------------------------------------------------------------
#
# routines for running columbus
#
#---------------------------------------------------------------
#
#
#
def make_one_time_input():
    global work_path

    # all calculations take place in work_dir
    print("work path in dir="+work_path)
    os.chdir(work_path) 

    # rotation matrix
    with open('rotmax', 'w') as rfile:
        rfile.write('  1  0  0\n  0  1  0\n  0  0  1')

    # cidrtfil files
    with open('cidrtmsls','w') as cidrtmsls, open('cidrtmsin','r') as cidrtmsin:
        subprocess.run(['cidrtms.x','-m',mem_str],stdin=cidrtmsin,stdout=cidrtmsls)
    shutil.move('cidrtfl.1','cidrtfl.ci')
    with open('cidrtmsls.cigrd','w') as cidrtmsls_grd, open('cidrtmsin.cigrd','r') as cidrtmsin_grd:
        subprocess.run(['cidrtms.x','-m',mem_str],stdin=cidrtmsin_grd,stdout=cidrtmsls_grd)  
    shutil.move('cidrtfl.1','cidrtfl.cigrd') 

    # check if hermitin exists, if not, copy daltcomm
    if not os.path.exists('hermitin'):
        shutil.copy('daltcomm','hermitin') 

    # make sure ciudgin file exists
    shutil.copy('ciudgin.drt1','ciudgin')

#
# run dalton to generate AO integrals
#
def generate_integrals(tid):
    global work_path    

    os.chdir(work_path)

    # run unik.gets.x script
    with open('unikls', "w") as unikls: 
        subprocess.run(['unik.gets.x'],stdout=unikls,universal_newlines=True,shell=True)

    # run hernew
    subprocess.run(['hernew.x'])
    shutil.move('daltaoin.new','daltaoin')

    # run dalton.x
    shutil.copy('hermitin','daltcomm')
    with open('hermitls', "w") as hermitls:
        subprocess.run(['dalton.x','-m',mem_str],stdout=hermitls,universal_newlines=True,shell=True)

    append_log(tid,'integral')

#
# run mcscf program
#
def run_col_mcscf(tid,t_state):
    global work_path, n_mcstates, mrci_lvl

    os.chdir(work_path)
    
    # get an initial starting set of orbitals
    set_mcscf_restart(tid)

    # allow for multiple DRTs in the mcscf part. For example, We may 
    # want to average over a' and a" in a tri-atomic case
    for i in range(1,n_drt+1):
        shutil.copy('mcdrtin.'+str(i),'mcdrtin')
        with open('mcdrtls','w') as mcdrtls, open('mcdrtin','r') as mcdrtin:
            subprocess.run(['mcdrt.x','-m',mem_str],stdin=mcdrtin,stdout=mcdrtls)
        with open('mcuftls','w') as mcuftls:
            subprocess.run(['mcuft.x'],stdout=mcuftls)

        # save formula tape and log files for each DRT
        shutil.copy('mcdrtfl','mcdrtfl.'+str(i))
        shutil.copy('mcdftfl','mcdftfl.'+str(i))
        shutil.copy('mcuftls','mcuftls.'+str(i))
        shutil.copy('mcoftfl','mcoftfl.'+str(i))

    # if running cas dynamics (i.e. no mrci), make sure we compute the
    # mcscf/cas density (for gradients and couplings)
    if mrci_lvl == 0:
        with open('mcdenin','w',encoding='utf-8') as mcden:
            mcden.write('MCSCF')
            # diagonal densities (for gradients)
            for i in range(n_mcstates):
                mcden.write('1  {:2d}  1  {:2d}').format(i,i)
            # off-diagonal densities (for couplings)
            for i in range(n_mcstates):
                mcden.write('1  {:2d}  1  {:2d}').format(min(i,t_state),max(i,t_state))

    # try running mcscf a couple times this can be tweaked if one
    # develops other strategies to deal with convergence problems
    converged = False
    run_max   = 3
    n_run     = 0
    while not converged and n_run < run_max:
        n_run += 1
        print("nrun="+str(n_run))
        if n_run == 3:
            # disable orbital-state coupling if convergence an issue            
            ncoupl = int(read_nlist_keyword('mcscfin','ncoupl'))
            niter  = int(read_nlist_keyword('mcscfin','niter'))
            set_nlist_keyword('mcscfin','ncoupl',niter+1)         
        subprocess.run(['mcscf.x -m '+mem_str],shell=True)
        # check convergence
        with open('mcscfls','r') as ofile:
            for line in ofile:
                if '*converged*' in line:
                    converged = True
                    break

    # if not converged, we have to die here...
    if not converged:
        sys.exit("MCSCF not converged. Exiting...") 

    # save output
    shutil.copy('mocoef_mc','mocoef')

    # grab mcscfls output
    append_log(tid,'mcscf')

    return

#
# run mrci if running at that level of theory
#
def run_col_mrci(tid,t_state):
    global energies, charges, work_path, n_atoms, n_cistates

    os.chdir(work_path)

    # if restart file exists, create symbolic link to it
    set_mrci_restart(tid)

    # make sure we point to the correct formula tape file
    os.symlink('cidrtfl.ci','cidrtfl')
    os.symlink('cidrtfl.ci','cidrtfl.1')

    # perform the integral transformation
    with open('tranin','w') as ofile:
        ofile.write("&input\nLUMORB=0\n&end")
    subprocess.run(['tran.x','-m',mem_str]) 

    # run mrci
    subprocess.run(['ciudg.x','-m',mem_str])
    ci_ener = []
    ci_res  = []
    ci_tol  = []
    mrci_iter = False
    converged = True
    with open('ciudgsm','r') as ofile:
        for line in ofile: 
            if 'beginning the ci' in line:
                mrci_iter = True
            if 'final mr-sdci  convergence information' in line and mrci_iter:
                for i in range(n_cistates):
                    ci_info = ofile.readline().rstrip().split()
                    ci_ener.append(float(ci_info[4]))
                    ci_res.append(float(ci_info[7]))
                    ci_tol.append(float(ci_info[8]))
                    converged = converged and ci_res[-1] <= ci_tol[-1]
                break

    # determine convergence...
    if not converged:
        print("EXITING -- MRCI did not converge for trajectory "+str(tid))
        sys.exit()

    # if we're good, update energy array
    energies[tid] = np.fromiter(
                    (ci_ener[i] for i in range(n_cistates)),
                     dtype=np.float)

    # now update charges
    ist = -1
    charges[tid] = np.zeros((n_cistates,n_atoms),dtype=float)
    with open('ciudgls','r') as ofile:
        line = ofile.readline()
        if '      gross atomic populations' in line:
            ist += 1
            charg = []
            for i in range(int(math.ceil(n_atoms/6.))):
                for j in range(5):
                    line = ofile.readline()
                ch_val = line.rstrip().split()
                charg.append(float(ch_val[1:]))
                line = ofile.readline()
            charges[tid][ist,:] = np.asarray(charg,dtype=float)
 
    # grab mrci output
    append_log(tid,'mrci')

    # transform integrals using cidrtfl.cigrd
    frzn_core = int(read_nlist_keyword('cigrdin','assume_fc'))
    print("frozen_core="+str(frzn_core))
    if frzn_core == 1:
        os.remove('moints')
        os.remove('cidrtfl')
        os.remove('cidrtfl.1')
        os.symlink('cidrtfl.cigrd','cidrtfl')
        os.symlink('cidrtfl.cigrd','cidrtfl.1')
        shutil.copy(input_path+'/tranin','tranin')
        suprocess.run(['tran.x','-m',mem_str])
 
    return    

#
# run dipoles/quadrupoles
#
def run_col_multipole(tid,t_state):
    global dipoles, quadpoles, work_path, n_cistates, mrci_lvl

    os.chdir(work_path)
    
    nst            = n_cistates
    dipoles[tid]   = np.zeros((n_cistates,n_cistates,3),dtype=np.float)
    quadpoles[tid] = np.zeros((n_cistates,6),dtype=np.float)

    type_str       = 'ci'
    if mrci_lvl == 0:
        type_str   = 'mc'

    for istate in range(nst):
        i1 = istate + 1
        os.symlink('nocoef_'+str(type_str)+'.drt1.state'+str(i1),'mocoef_prop')
        subprocess.run(['exptvl.x','-m',mem_str])
        with open('propls','r') as prop_file:
            line = prop_file.readline()
            if 'Dipole moments' in line:
                for j in range(5):
                    line = prop_file.readline()
                mom_info = line.rstrip().split()
                dip_mom  = np.array([float(mom_info[1]), \
                                     float(mom_info[2]), \
                                     float(mom_info[3])])
                dipoles[tid][istate,istate:] = dip_mom
            if 'Second moments' in line:
                for j in range(5):
                    line = prop_file.readline()   
                mom_info = line.rstrip().split()
                for j in range(5):
                    line = prop_file.readline()
                mom_info = mom_info.append(line.rstrip().split())
                sec_mom  = np.array([float(mom_info[1]), \
                                     float(mom_info[2]), \
                                     float(mom_info[3]), \
                                     float(mom_info[4]), \
                                     float(mom_info[6]), \
                                     float(mom_info[7])])
                quadpoles[tid][istate,:] = sec_mom
        os.remove('mocoef_prop')

    return

#
# Compute transition dipoles between ground and excited state,
# and between trajectory states and other state
#
def run_col_tdipole(tid,t_state):
    global dipoles, n_cistates, work_path, mrci_lvl
 
    os.chdir(work_path)

    init_states = [0, t_state-1]
    # make sure we point to the correct formula tape file
    os.symlink('civfl','civfl.drt1')
    os.symlink('civout','civout.drt1')
    os.symlink('cirefv','cirefv.drt1')

    for istate in init_states:
        i1 = istate + 1
        for jstate in range(n_cistates):
            j1 = jstate + 1
            # only do transition dipoles
            if istate == jstate or (jstate in init_states and jstate < istate):
                continue     

            ii = min(i1,j1)
            jj = max(i1,j1)

            if mrci_lvl == 0:
                with open('transftin','w') as ofile:
                    ofile.write('y\n1\n'+str(ii)+'\n1\n'+str(jj))
                subprocess.run(['transft.x'],stdin='transftin',stdout='transftls')

                with open('transmomin','w') as ofile:
                    ofile.write('MCSCF\n1 '+str(ii)+'\n1\n'+str(jj))
                subprocess.run(['transmom.x','-m',mem_str])

                os.remove('mcoftfl')
                shutil.copy('mcoftfl.1','mcoftfl') 

            else:
                with open('trnciin','w') as ofile:
                    ofile.write('&input\nlvlprt=1,\nnroot1='+str(ii)+',\n'+
                                 'nroot2='+str(jj)+',\ndrt1=1,\ndrt2=1,\n&end')
                subprocess.run(['transci.x','-m',mem_str])
                shutil.move('cid1trfl','cid1trfl.'+str(ii)+'.'+str(jj)) 

            with open('trncils','r') as trncils:
                for line in trncils:
                    if 'total (elec)' in line:
                        line_arr = line.rstrip().split() 
                        for dim in range(3):
                            dipoles[tid][istate,jstate,dim] = float(line_arr[dim+2])
                            dipoles[tid][jstate,istate,dim] = float(line_arr[dim+2])
             

# perform integral transformation and determine gradient on
# trajectory state
#
def run_col_gradient(tid,t_state):
    global gradients, input_path, work_path, mrci_lvl, n_cistates, n_cart

    os.chdir(work_path)
    shutil.copy(input_path+'/cigrdin','cigrdin')
    gradients[tid] = np.zeros((n_cistates,n_cistates,n_cart),dtype=np.float)
    tindex = t_state + 1

    if mrci_lvl > 0:
        os.symlink('cid1fl.drt1.state'+str(tindex),'cid1fl')
        os.symlink('cid2fl.drt1.state'+str(tindex),'cid2fl')
        shutil.copy(input_path+'/trancidenin','tranin')       
    else:
        os.symlink('mcsd1fl.'+str(tindex),'cid1fl')
        os.symlink('mcsd2fl.'+str(tindex),'cid2fl')
        set_nlist_keyword('cigrdin','samcflag',1)
        shutil.copy(input_path+'/tranmcdenin','tranin')

    # run cigrd
    set_nlist_keyword('cigrdin','nadcalc',0)
    subprocess.run(['cigrd.x','-m',mem_str])
    os.remove('cid1fl')
    os.remove('cid2fl')
    shutil.move('effd1fl','modens')
    shutil.move('effd2fl','modens2')

    # run tran
    subprocess.run(['tran.x','-m',mem_str])
    os.remove('modens')
    os.remove('modens2')

    # run dalton
    shutil.copy(input_path+'/abacusin','daltcomm')
    with open('abacusls','w') as abacusls:
        subprocess.run(['dalton.x','-m',mem_str],stdout=abacusls)

    # read in cartesian gradient and save to array
    with open('cartgrd','r') as cartgrd:
        i = 0
        for line in cartgrd:
            l_arr = line.rstrip().split()
            for j in range(3):
                gradients[tid][t_state,t_state,3*i+j] = float(l_arr[j].replace("D","e"))
            i = i + 1

    # grab cigrdls output
    append_log(tid,'cigrd')
       
#
# compute couplings to states within prescribed DE window
#
def run_col_coupling(tid,t_state):
    global couplings, input_path, work_path, n_cistates, mrci_lvl, n_cart

    os.chdir(work_path)
    tindex = t_state + 1

    # copy some clean files to the work directory
    shutil.copy(input_path+'/cigrdin','cigrdin')
    set_nlist_keyword('cigrdin','nadcalc',1)
    if mrci_lvl == 0:
        set_nlist_keyword('cigrdin','samcflag',1)
        shutil.copy(input_path+'/tranmcdenin','tranin')
    else:
        shutil.copy(input_path+'/trancidenin','tranin')

    shutil.copy(input_path+'/abacusin','daltcomm')
    insert_dalton_key('daltcomm','COLBUS','.NONUCG')

    for istate in range(n_cistates):
        i1 = istate+1
        if istate == t_state:
            continue

        s1 = str(min(i1,tindex)).strip()
        s2 = str(max(i1,tindex)).strip()
        if mrci_lvl == 0:
            os.symlink('mcsd1fl.trd'+s1+'to'+s2,'cid1fl.tr')
            os.symlink('mcsd2fl.trd'+s1+'to'+s2,'cid2fl.tr')   
            os.symlink('mcad1fl.'+s1+s2,'cid1trfl')   
        else:
            os.symlink('cid1fl.trd'+s1+'to'+s2,'cid1fl.tr')
            os.symlink('cid2fl.trd'+s1+'to'+s2,'cid2fl.tr')
            os.symlink('cid1trfl.'+s1+'.'+s2,'cid1trfl')

        set_nlist_keyword('cigrdin','drt1',1)
        set_nlist_keyword('cigrdin','drt2',1)
        set_nlist_keyword('cigrdin','root1',s1)
        set_nlist_keyword('cigrdin','root2',s2)

        subprocess.run(['cigrd.x','-m',mem_str])
 
        shutil.move('effd1fl','modens')
        shutil.move('effd2fl','modens2')
 
        subprocess.run(['tran.x','-m',mem_str])
        with open('abacusls','w') as abacusls:
            subprocess.run(['dalton.x','-m',mem_str],stdout=abacusls)

        # read in cartesian gradient and save to array
        with open('cartgrd','r') as cartgrd:
            i = 0
            for line in cartgrd:
                l_arr = line.rstrip().split()
                for j in range(3):
                    gradients[tid][t_state,istate,3*i+j] = float(l_arr[j].replace("D","e"))
                    gradients[tid][istate,t_state,3*i+j] = float(l_arr[j].replace("D","e"))
                i = i + 1

    # grab mcscfls output
    append_log(tid,'nad')

#
# save mocoef and ci files to restart directory
#
def make_col_restart(tid):
    global work_path

    os.chdir(work_path)

    # copy orbitals
    shutil.move('mocoef', restart_path+'/mocoef.'+str(tid))

    # copy all ci vector, ci info files
    shutil.move('civfl' , restart_path+'/civfl.'+str(tid))
    shutil.move('civout' , restart_path+'/civout.'+str(tid))
    shutil.move('cirefv' , restart_path+'/cirefv.'+str(tid))

#-----------------------------------------------------------------
#
# file parsing
#
#-----------------------------------------------------------------

#
# grab key output from columbus listing files (useful for diagnosing
#  electronic structure problems)
#
def append_log(tid, listing_file):

    if listing_file == 'integral':
        with open('hermitls','r') as hermitls:
            for line in hermitls:
                if 'Bond distances' in line:
                    while 'Nuclear repulsion energy' not in line:
                        print(line)
                        line = hermitls.readline()
                    break

    elif listing_file == 'mcscf':
        with open('mcscfls','r') as mcscfls:
            for line in mcscfls:
                if 'final mcscf' in line:
                    while len(line.rstrip()) != 0:
                        print(line)
                        line = mcscfls.readline()
                    break

    elif listing_file == 'mrci':
        with open('ciudgsm','r') as ciudgls:
            for line in ciudgls:
                if 'final mr-sdci  convergence information' in line:
                    while len(line.rstrip()) != 0:
                        print(line)
                        line = ciudgls.readline()
                    break

    elif listing_file == 'cigrd':
        with open('cigrdls','r') as cigrdls:
            for line in cigrdls:
                if 'RESULTS' in line:
                    while 'effective' not in line:
                        print(line)
                        line = cigrdls.readline()
                    break

    elif listing_file == 'nad':
        with open('cigrdls','r') as cigrdls_nad:
            for line in cigrdls_nad:
                if 'RESULTS' in line:
                    while 'effective' not in line:
                        print(line)
                        line = cigrdls_nad.readline()
                    break

    else:
        print("listing file: "+str(listing_file)+" not recognized.")

    return

#
#  copy mocoef file to working directory:
#  1. If first step and parent-less trajectory, take what's in input
#  2. If first step of spawned trajectory, take parents restart info
#  3. If first step of centroid, take one of parent's restart info
#
def set_mcscf_restart(tid):
    global work_path

    os.chdir(work_path)

    # if restart file exists, create symbolic link to it
    mocoef_file = restart_path+'/mocoef.'+str(tid)
    if os.path.exists(mocoef_file):
        shutil.copy(mocoef_file,'mocoef') 
        return True
    else:
        return False   

#
# copy/link ci restart files to working directory:
#  see above for restart logic
#
def set_mrci_restart(tid):
    global work_path

    os.chdir(work_path)

    # if restart file exists, create symbolic link to it
    civfl  = restart_path+'/civfl.'+str(tid)
    civout = restart_path+'/civout.'+str(tid)
    cirefv = restart_path+'/cirefv.'+str(tid)
    if os.path.exists(civfl) and os.path.exists(civout) and os.path.exists(cirefv):
        os.symlink(civfl,'civfl')
        os.symlink(civout,'civout') 
        os.symlink(cirefv,'cirefv')
        return True
    else:
        return False

#
# write a particle array to a COLUMBUS style geom file
#
def write_col_geom(geom):
    global n_atoms, work_path

    os.chdir(work_path)

    f = open('geom','w',encoding='utf-8')
    for i in range(n_atoms):
        f.write(' {:2s}   {:3.1f}  {:12.8f}  {:12.8f}  {:12.8f}  {:12.8f}\n'. \
        format(geom[i].name, geom[i].anum, 
               geom[i].x[0], geom[i].x[1], geom[i].x[2],
               geom[i].mass/glbl.mass2au))
    f.close()

#
# Read from a direct input file via keyword search 
#
def read_pipe_keyword(infile,keyword):
    f = open(infile,'r',encoding='utf-8')
    for line in f:
        if keyword in line:
            f.close()
            return line.split()[0]

#
# Read from a namelist style input
#
def read_nlist_keyword(infile,keyword):
    f = open(infile,'r',encoding='utf-8')
    for line in f:
        if keyword in line:
            f.close()
            line = line.rstrip("\r\n")
            return line.split('=',1)[1].strip(' ,')

#
# Read from a namelist style input
#
def set_nlist_keyword(file_name,keyword,value):
    outfile = str(file_name)+'.tmp'
    key_found = False
    with open(file_name,'r') as ifile, open(outfile,'w') as ofile:
        for line in ifile:
            if keyword in line:
                ofile.write(str(keyword)+' = '+str(value)+'\n')
                key_found = True
            elif '&end' in line and not key_found:
                ofile.write(keyword+' = '+str(value)+',')
                ofile.write(line)
                break
            else:
                ofile.write(line)
    shutil.move(outfile,file_name)

#
# insert a dalton keyword (this is a pretty specialized function given
# the idiosyncracies of dalton input)
# the keyword must already exist in the file
#
def insert_dalton_key(infile,keyword,value):
    with open(infile,'r') as ifile, open("tempfile",'w') as ofile:
        for line in ifile:
            ofile.write(line)
            if keyword in line:
                ofile.write(value)
    shutil.move("tempfile",infile)

#
# return the number of lines in file
#
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

#
# load orbitals into an mocoef file
#
def load_orbitals(tid):
    pass

#
# write orbitals to mocoef file
#
def write_orbitals(fname,orb_array):
    pass

