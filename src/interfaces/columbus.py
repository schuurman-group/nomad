#
# routines for running a columbus computation
#
import sys
import os
import math
import shutil
import pathlib
import subprocess
import numpy as np
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.particle as particle
import src.basis.trajectory as trajectory
import src.basis.bundle as bundle 

# path to columbus input files
input_path = ''
# path to location of 'work'/'restart' directories
work_path = ''
# path to the location of restart files (i.e. mocoef files and civfl)
restart_path = ''

# by default, since this is the columbus module, assume "particles" are
# 3-dimensional (i.e. Cartesian)
p_dim        = 3
# number of atoms
n_atoms      = 0
# total number of cartesian coordinates 
n_cart       = 0
# number of drts (should usually be "1" for C1)
n_drt        = 1
# number of orbitals
n_orbs       = 0
# number of states in sate-averaged MCSCF
n_mcstates   = 0
# number of CI roots 
n_cistates   = 0
# maximum angular momentum in basis set
max_l        = 1 
# excitation level in CI
mrci_lvl     = 0
# amount of memory per process, in MB
mem_str      = ''
current_geom = dict()
energies     = dict()
atom_pops     = dict()
dip_moms     = dict()
sec_moms     = dict()
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
    global columbus_path, input_path, work_path, restart_path, \
           p_dim, n_atoms, n_cart, n_orbs, n_mcstates, n_cistates, \
           max_l,mrci_lvl,mem_str

    print("init interface called...")
    # confirm that we can see the COLUMBUS installation (pull the value
    # COLUMBUS environment variable)
    columbus_path = os.environ['COLUMBUS']
    if not os.path.isfile(columbus_path+'/ciudg.x'):
        print("Cannot find COLUMBUS executables in: "+columbus_path)
        sys.exit()
    # ensure COLUMBUS input files are present locally
    if not os.path.exists('input'):
        print("Cannot find COLUMBUS input files in: input")
        sysexit()

    # setup working directories
    input_path    = fileio.scr_path+'/input'
    work_path     = fileio.scr_path+'/work'
    restart_path  = fileio.scr_path+'/restart'

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
    n_cart                 = p_dim * n_atoms
    n_orbs                 = int(read_pipe_keyword('input/cidrtmsin',
                                                'orbitals per irrep'))
    n_mcstates             = int(read_nlist_keyword('input/mcscfin',
                                                'NAVST'))
    n_cistates              = int(read_nlist_keyword('input/ciudgin.drt1',
                                                'NROOT'))
    mrci_lvl               = int(read_pipe_keyword('input/cidrtmsin',
                                                 'maximum excitation level'))
    max_l                  = ang_mom_dalton('input/daltaoin')


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
        master.add_trajectory(trajectory.trajectory(
                              glbl.fms['interface'],
                              glbl.fms['n_states'],
                              particles=geom_list[i],
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
    global dip_moms 

    if not in_cache(tid,geom):
        if tid >=0:
            run_trajectory(tid,geom,t_state)
        else:
            print("invalid id for trajectory: "+str(tid))
            sys.exit("ERROR in columbus module[dipole] -- invalid id") 
    try:
        return dip_moms[tid][lstate,rstate,:]
    except:
        print("ERROR in fetch_dipole")
        sys.exit("ERROR in columbus module fetching dipoles")

#
# return second moment tensor for state=state
#
def sec_mom(tid,geom,t_state,rstate):
    global sec_moms 

    if not in_cache(tid,geom):
        if tid >= 0:
            run_trajectory(tid,geom,t_state)
        else:
            print("invalid id for trajectory: "+str(tid))
            sys.exit("ERROR in columbus module[qpole] -- invalid id")
    try:
        return sec_moms[tid][rstate,:]
    except:
        print("ERROR in fetch_sec_mom")     
        sys.exit("ERROR in columbus module fetching sec_mom")

#
#
#
def atom_pop(tid,geom,t_state, rstate):
    global atom_pops

    if not in_cache(tid,geom):
        if tid >= 0:
            run_trajectory(tid,geom,t_state)
        else:
            print("invalid id for trajectory: "+str(tid))
            sys.exit("ERROR in columbus module[atom_pops] -- invalid id")
    try:
        return atom_pops[tid][rstate,:]
    except:
        print("ERROR in fetch_atom_pops")     
        sys.exit("ERROR in columbus module fetching atom_pops")

#----------------------------------------------------------------
#
#  "Private" functions
#
#----------------------------------------------------------------
def in_cache(tid,geom):
    global current_geom, n_atoms, p_dim

    if tid not in current_geom:
        return False
    g = np.fromiter((geom[i].x[j] for i in range(n_atoms) for j in range(p_dim)),np.float)
    difg = np.linalg.norm(g - current_geom[tid])
    print("tid, difg="+str(tid)+' '+str(difg))
    if difg <= glbl.fpzero:
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
    global p_dim, n_cistates, n_atoms, current_geom

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
    for i in range(n_cistates):
        for j in range(n_cistates):
            if i != j:
                run_col_tdipole(tid, j, i)

    # compute gradient on current state
    run_col_gradient(tid,tstate)

    # run coupling to other states
    for i in range(n_cistates):
        if i != tstate:
            run_col_coupling(tid, tstate, i)

    # save restart files
    make_col_restart(tid)

    # update the geometry in the cache
    g = np.fromiter((geom[i].x[j] for i in range(n_atoms) for j in range(p_dim)),np.float)
    current_geom[tid] = g

#
# For a centroid we really only need an energy (if both trajectories
#  are on the same state), or a coupling (if on different states)
#
def run_centroid(tid,geom,lstate,rstate):
    global p_dim, n_atoms, current_geom

    # write geometry to file
    write_col_geom(geom)

    # generate integrals
    generate_integrals(tid)

    # run mcscf
    run_col_mcscf(tid, lstate)

    # run mrci, if necessary
    run_col_mrci(tid, lstate)

    if lstate != rstate:
        # run coupling to other states
        run_col_coupling(tid, lstate, rstate)

    # save restart files
    make_col_restart(tid)

    # update the geometry in the cache
    g = np.fromiter((geom[i].x[j] for i in range(n_atoms) for j in range(p_dim)),np.float)
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

    sys.stdout.flush()
    # all calculations take place in work_dir
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

    # and check that there's a transition section
    transition = False
    ciudgin = open('ciudgin','r+')
    for line in ciudgin:
        if 'transition' in line:
            transition = True
            break
    if not transition:
        ciudgin.write('transition\n')
        for i in range(n_cistates):
            for j in range(i):
                ciudgin.write('  1 {0:2d}  1 {1:2d}\n'.format(j+1,i+1))
    ciudgin.close()

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
    global energies, atom_pops, work_path, n_atoms, n_cistates

    os.chdir(work_path)

    # if restart file exists, create symbolic link to it
    set_mrci_restart(tid)

    # make sure we point to the correct formula tape file
    link_force('cidrtfl.ci','cidrtfl')
    link_force('cidrtfl.ci','cidrtfl.1')

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

    # now update atom_pops
    ist = -1
    atom_pops[tid] = np.zeros((n_cistates,n_atoms),dtype=float)
    with open('ciudgls','r') as ciudgls:
        for line in ciudgls: 
            if '   gross atomic populations' in line:
                ist += 1
                pops = []
                for i in range(math.ceil(n_atoms/6.)):
                    for j in range(max_l+3):
                        line = ciudgls.readline()
                    l_arr = line.rstrip().split()
                    pops.extend(l_arr[1:])
                    line = ciudgls.readline()
                atom_pops[tid][ist,:] = np.asarray([float(x) for x in pops],dtype=float)

    # grab mrci output
    append_log(tid,'mrci')

    # transform integrals using cidrtfl.cigrd
    frzn_core = int(read_nlist_keyword('cigrdin','assume_fc'))
    if frzn_core == 1:
        os.remove('moints')
        os.remove('cidrtfl')
        os.remove('cidrtfl.1')
        link_force('cidrtfl.cigrd','cidrtfl')
        link_force('cidrtfl.cigrd','cidrtfl.1')
        shutil.copy(input_path+'/tranin','tranin')
        subprocess.run(['tran.x','-m',mem_str])
 
    return    

#
# run dipoles / second moments
#
def run_col_multipole(tid,t_state):
    global p_dim, dip_moms, sec_moms, work_path, n_cistates, mrci_lvl

    os.chdir(work_path)
    
    nst            = n_cistates
    dip_moms[tid]  = np.zeros((n_cistates,n_cistates,p_dim),dtype=np.float)
    sec_moms[tid]  = np.zeros((n_cistates,p_dim),dtype=np.float)

    type_str       = 'ci'
    if mrci_lvl == 0:
        type_str   = 'mc'

    for istate in range(nst):
        i1 = istate + 1
        link_force('nocoef_'+str(type_str)+'.drt1.state'+str(i1),'mocoef_prop')
        subprocess.run(['exptvl.x','-m',mem_str])
        with open('propls','r') as prop_file:
            for line in prop_file:
                if 'Dipole moments' in line:
                    for j in range(5):
                        line = prop_file.readline()
                    l_arr = line.rstrip().split()
                    dip_moms[tid][istate,istate,:]  = np.array([float(l_arr[1]), 
                                                                float(l_arr[2]), 
                                                                float(l_arr[3])])
                if 'Second moments' in line:
                    for j in range(5):
                        line = prop_file.readline()   
                    l_arr = line.rstrip().split()
                    for j in range(5):
                        line = prop_file.readline()
                    l_arr.extend(line.rstrip().split())
                    # NOTE: we're only taking the diagonal elements
                    sec_moms[tid][istate,:] = np.array([float(l_arr[1]), 
                                                        float(l_arr[4]), 
                                                        float(l_arr[7])])
        os.remove('mocoef_prop')

    return

#
# Compute transition dipoles between ground and excited state,
# and between trajectory states and other state
#
def run_col_tdipole(tid, state_i, state_j):
    global p_dim, dip_moms, n_cistates, work_path, mrci_lvl
 
    os.chdir(work_path)

    # make sure we point to the correct formula tape file
    link_force('civfl','civfl.drt1')
    link_force('civout','civout.drt1')
    link_force('cirefv','cirefv.drt1')

    i1 = min(state_i,state_j) + 1
    j1 = max(state_i,state_j) + 1

    if state_i == state_j:
        return
 
    if mrci_lvl == 0:
        with open('transftin','w') as ofile:
            ofile.write('y\n1\n'+str(j1)+'\n1\n'+str(i1))
            subprocess.run(['transft.x'],stdin='transftin',stdout='transftls')

        with open('transmomin','w') as ofile:
            ofile.write('MCSCF\n1 '+str(j1)+'\n1\n'+str(i1))
            subprocess.run(['transmom.x','-m',mem_str])

        os.remove('mcoftfl')
        shutil.copy('mcoftfl.1','mcoftfl')

    else:
        with open('trnciin','w') as ofile:
            ofile.write(' &input\n lvlprt=1,\n nroot1='+str(i1)+',\n'+
                        ' nroot2='+str(j1)+',\n drt1=1,\n drt2=1,\n &end')
        subprocess.run(['transci.x','-m',mem_str])
        shutil.move('cid1trfl','cid1trfl.'+str(i1)+'.'+str(j1))

    with open('trncils','r') as trncils:
        for line in trncils:
            if 'total (elec)' in line:
                line_arr = line.rstrip().split()
                for dim in range(p_dim):
                    dip_moms[tid][state_i,state_j,dim] = float(line_arr[dim+2])
                    dip_moms[tid][state_j,state_i,dim] = float(line_arr[dim+2])

# perform integral transformation and determine gradient on
# trajectory state
#
def run_col_gradient(tid,t_state):
    global p_dim, gradients, input_path, work_path, mrci_lvl, n_cistates, n_cart

    os.chdir(work_path)
    shutil.copy(input_path+'/cigrdin','cigrdin')
    gradients[tid] = np.zeros((n_cistates,n_cistates,n_cart),dtype=np.float)
    tindex = t_state + 1

    if mrci_lvl > 0:
        link_force('cid1fl.drt1.state'+str(tindex),'cid1fl')
        link_force('cid2fl.drt1.state'+str(tindex),'cid2fl')
        shutil.copy(input_path+'/trancidenin','tranin')       
    else:
        link_force('mcsd1fl.'+str(tindex),'cid1fl')
        link_force('mcsd2fl.'+str(tindex),'cid2fl')
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
    shutil.move('abacusls','abacusls.grad')


    # read in cartesian gradient and save to array
    with open('cartgrd','r') as cartgrd:
        i = 0
        for line in cartgrd:
            l_arr = line.rstrip().split()
            for j in range(p_dim):
                gradients[tid][t_state,t_state,p_dim*i+j] = float(l_arr[j].replace("D","e"))
            i = i + 1

    # grab cigrdls output
    append_log(tid,'cigrd')
       
#
# compute couplings to states within prescribed DE window
#
def run_col_coupling(tid, t_state, c_state):
    global p_dim, couplings, input_path, work_path, n_cistates, mrci_lvl, n_cart

    if t_state == c_state:
        return

    os.chdir(work_path)

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

    s1 = str(min(t_state, c_state) + 1).strip()
    s2 = str(max(t_state, c_state) + 1).strip()

    if mrci_lvl == 0:
        link_force('mcsd1fl.trd'+s1+'to'+s2,'cid1fl.tr')
        link_force('mcsd2fl.trd'+s1+'to'+s2,'cid2fl.tr')   
        link_force('mcad1fl.'+s1+s2,'cid1trfl')   
    else:
        link_force('cid1fl.trd'+s1+'to'+s2,'cid1fl.tr')
        link_force('cid2fl.trd'+s1+'to'+s2,'cid2fl.tr')
        link_force('cid1trfl.'+s1+'.'+s2,'cid1trfl')

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
            for j in range(p_dim):
                gradients[tid][t_state,c_state,p_dim*i+j] = float(l_arr[j].replace("D","e"))
                gradients[tid][c_state,t_state,p_dim*i+j] = float(l_arr[j].replace("D","e"))
            i = i + 1
    delta_e = energies[tid][t_state] - energies[tid][c_state]
    gradients[tid][t_state,c_state,:] /=  delta_e
    gradients[tid][c_state,t_state,:] /= -delta_e
    shutil.move('cartgrd','cartgrd.nad.'+str(s1)+'.'+str(s2))

    # grab mcscfls output
    append_log(tid,'nad')

#
# save mocoef and ci files to restart directory
#
def make_col_restart(tid):
    global work_path

    os.chdir(work_path)

    # move orbitals
    shutil.move('mocoef', restart_path+'/mocoef.'+str(tid))

    # move all ci vector, ci info files
    shutil.move('civfl' , restart_path+'/civfl.'+str(tid))
    shutil.move('civout' , restart_path+'/civout.'+str(tid))
    shutil.move('cirefv' , restart_path+'/cirefv.'+str(tid))

    # do some cleanup
    if os.path.isfile('cirdrtfl'):   os.remove('cidrtfl')

    if os.path.isfile('aoints'):     os.remove('aoints')
    if os.path.isfile('aoints2'):    os.remove('aoints2')
    if os.path.isfile('modens'):     os.remove('modens')
    if os.path.isfile('modens2'):    os.remove('modens2')
    if os.path.isfile('cid1fl.tr'):  os.remove('cid1fl.tr')
    if os.path.isfile('cid2fl.tr'):  os.remove('cid2fl.tr')
    if os.path.isfile('cid1trfl'):   os.remove('cid1trfl')
    if os.path.isfile('civfl.drt1'): os.remove('civfl.drt1')
    if os.path.isfile('civout.drt1'):os.remove('civout.drt1')
    if os.path.isfile('cirefv.drt1'):os.remove('cirefv.drt1')


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
        link_force(civfl,'civfl')
        link_force(civout,'civout') 
        link_force(cirefv,'cirefv')
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
                ofile.write(str(keyword)+' = '+str(value)+',\n')
                ofile.write(line)
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
                ofile.write(value+'\n')
    shutil.move("tempfile",infile)

#
# finds maximum ang. mom in basis set from dalton. Pretty specific...
#
def ang_mom_dalton(infile):

    max_l = 0
    with open(infile,'r') as daltaoin:
        for i in range(4):
            line = daltaoin.readline()
        l_arr = line.rstrip().split()
        n_grps = int(l_arr[1])
        for i in range(n_grps):
            line = daltaoin.readline()
            l_arr = line.rstrip().split()
            n_atm = int(l_arr[1])
            n_ang = int(l_arr[2])-1
            max_l = max(max_l,n_ang) # max_l on first line
            n_con = [int(l_arr[j]) for j in range(3,3+n_ang+1)]
            for j in range(n_atm):
                line = daltaoin.readline()
            for j in range(len(n_con)):
                for k in range(n_con[j]):
                    line = daltaoin.readline()
                    nprim = int(line.rstrip().split()[1])
                    for l in range(nprim):
                        line = daltaoin.readline() 
    return max_l


#
# return the number of lines in file
#
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

#
# create a symbolic link, overwriting existing link if necessary
#
def link_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError:
        os.unlink(link_name)
        os.symlink(target, link_name)

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

