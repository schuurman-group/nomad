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

# path to the columbus executables
columbus_path = ''
# path to the top level of scratch directory
scr_path      = ''
# path to location of 'work'/'restart' directories
work_path     = ''
# path to the location of restart files (i.e. mocoef files and civfl)
restart_path  = ''

n_atoms      = 0
n_cart       = 0
current_geom = dict()
energies     = dict()
gradient     = dict()
couplings    = dict()
tdipoles     = dict()
dipoles      = dict()
quadpoles    = dict()
charges      = dict()

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
    work_path     = scr_path+'/work'
    restart_path  = scr_path+'/restart'
    if os.path.exists(work_path):
        shutil.rmtree(work_path)
    if os.path.exists(restart_path):
        shutil.rmtree(restart_path)
    os.makedirs(work_path)
    os.makedirs(restart_path)

    # copy input directory to scratch and copy file contents to work directory
    os.mkdir(scr_path+'/input')
    for item in os.listdir('input'):
        s = os.path.join('input', item)
        d = os.path.join(work_path, item)
        shutil.copy2(s, d)

    # now -- pull information from columbus input
    n_atoms                = file_len('input/geom')
    n_cart                 = 3 * n_atoms
    glbl.pes['n_orbs']     = int(read_pipe_keyword('input/cidrtmsin',
                                                'orbitals per irrep'))
    glbl.pes['n_mcstates'] = int(read_nlist_keyword('input/mcscfin',
                                                'NAVST'))
    glbl.pes['n_cistates'] = int(read_nlist_keyword('input/ciudgin.drt1',
                                                'NROOT'))

    # generate one time input files for columbus calculations
    make_one_time_input()

#
# returns the energy at the specified geometry. If value on file 
#  not current, or we don't care about saving data -- recompute
#
# geom is a list of particles
def energy(tid,geom,lstate):
    if not in_cache(tid,geom):
        if tid > 0:
            run_trajectory(tid,geom,lstate)
        else:
            run_centroid(tid,geom,lstate,lstate)
    try:
        return energies[tid] 
    except:
        print("ERROR in fetch_energy")       
        sys.exit("ERROR in columbus module fetching energy")

#
# returns the MOs as an numpy array
#
def orbitals(tid,geom,lstate):
    if not in_cache(tid,geom):
        if tid > 0:
            run_trajectory(tid,geom,lstate)
        else:
            run_centroid(tid,geom,lstate,lstate)
    try:
        return load_orbitals(tid)
    except:
        print("ERROR in fetch_orbitals")     
        sys.exit("ERROR in columbus module fetching orbitals")

#
# return gradient. If lstate == rstate, gradient on state lstate. Else
#   returns non-adiabatic coupling vector
#
def derivative(tid,geom,lstate,rstate):
    if not in_cache(tid,geom):
        if tid > 0:
            run_trajectory(tid,geom,lstate)
        else:
            run_centroid(tid,geom,lstate,rstate)
    try:
        return gradients[tid][rstate]
    except:
        print("ERROR in fetch_gradients")     
        sys.exit("ERROR in columbus module fetching gradients")

#
# if lstate != rstate, corresponds to transition dipole
#
def tdipole(tid,geom,lstate,rstate):
    if not in_cache(tid,geom):
        run_trajectory(tid,geom,lstate)
    try:
        return tdipoles[tid][rstate]
    except:
        print("ERROR in fetch_dipole")
        sys.exit("ERROR in columbus module fetching dipoles")

#
# if lstate != rstate, corresponds to transition dipole
#
def dipole(tid,geom,lstate,rstate):
    if not in_cache(tid,geom):
        run_trajectory(tid,geom,lstate)
    try:
        return dipoles[tid][rstate]
    except:
        print("ERROR in fetch_dipole")     
        sys.exit("ERROR in columbus module fetching dipoles")

#
# return second moment tensor for state=state
#
def quadrupole(tid,geom,lstate,rstate):
    if not in_cache(tid,geom):
        run_trajectory(tid,geom,lstate)
    try:
        return quadpoles[tid][rstate]
    except:
        print("ERROR in fetch_quadpole")     
        sys.exit("ERROR in columbus module fetching quadpole")

#
#
#
def charges(tid,geom,lstate,rstate):
    if not in_cache(tid,geom):
        run_trajectory(tid,geom,lstate)
    try:
        return charges[tid][rstate]
    except:
        print("ERROR in fetch_charges")     
        sys.exit("ERROR in columbus module fetching charges")

#----------------------------------------------------------------
#
#  "Private" functions
#
#----------------------------------------------------------------
def in_cache(tid,geom):
    if tid not in current_geom:
        return False
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
    
    # write geometry to file
    write_col_geom(geom)

    # generate integrals
    generate_integrals()

    # run mcscf
    run_col_mcscf(tid,tstate)
  
    # run mrci, if necessary
    run_col_mrci(tid,tstate)

    # run properties, dipoles, etc.
    run_col_prop(tstate)

    # run transition dipoles
    run_col_tdipole(tstate)

    # compute gradient on current state
    run_col_gradient(tstate)

    # run coupling to other states
    run_col_coupling(tstate)

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

    # write geometry to file
    write_col_geom(geom)

    # generate integrals
    generate_integrals()

    # run mcscf
    run_col_mcscf(tstate)

    # run mrci, if necessary
    run_col_mrci(tstate)

    if lstate != rstate:
        # run coupling to other states
        run_col_coupling(tstate)

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
    return
    # rotation matrix
    with open('rotmax', 'w') as rfile:
        rfile.write('  1  0  0\n  0  1  0\n  0  0  1')

    # cidrtfil files
    with open('cidrtmsls','w') as cidrtmsls, open('cidrtmsin','r') as cidrtmsin:
        subprocess.run(['cidrtms.x','-m 2000'],stdin=cidrtmsin,stdout=cidrtmsls)
    shutil.move('cidrtfl.1','cidrtfl.ci')
    with open('cidrtmsls.cigrd','w') as cidrtin.grd, open('cidrtmsin.cigrd','r') as cidrtls.grd:
        subprocess.run(['cidrtms.x','-m 2000'],stdin=cidrtin.grd,stdout=cidrtls.grd)  
    shutil.move('cidrtfl.1','cidrtfl.cigrd') 

#
# run dalton to generate AO integrals
#
def generate_integrals():
    
    # run unik.gets.x script
    with open('unikls', "w") as unikls:    
        subprocess.run(['unik.gets.x'],stdout=unikls)

    # run hernew
    subprocess.run(['hernew.x'])
    shutil.move('daltaoin.new','daltaoin')

    # run dalton.x
    shutil.copy('hermitin','daltcomm')
    with open('hermitls', "w") as hermitls:
        subprocess.run(['dalton.x','-m 2000'],stdout=hermitls)
    append_log('integral')

#
# run mcscf program
#
def run_col_mcscf(tid,t_state):
    
    # get an initial starting set of orbitals
    set_mcscf_restart(tid)

    # allow for multiple DRTs in the mcscf part. For example, We may 
    # want to average over a' and a" in a tri-atomic case
    for i in range(n_drt):
        shutil.copy('mcdirtin.'+str(i),'mcdrtin')
        subprocess.run(['mcdrt.x','-m 2000'],stdin='mcdrtin',stdout='mcdrtls')
        subprocess.run(['mcuft.x'],stdout='mcuftls')

        # save formula tape and log files for each DRT
        suprocess.copy('mcdrtfl','mcdrtfl.'+str(i))
        suprocess.copy('mcdftfl','mcdftfl.'+str(i))
        suprocess.copy('mcuftls','mcuftls.'+str(i))
        suprocess.copy('mcoftfl','mcoftfl.'+str(i))

    # if running cas dynamics (i.e. no mrci), make sure we compute the
    # mcscf/cas density (for gradients and couplings)
    if not col_mrci:
        with open('mcdenin','w',encoding='utf-8') as mcden:
            mcden.write('MCSCF')
            # diagonal densities (for gradients)
            for i in range(glbl.pes['n_mcstates']):
                mcden.write('1  {:2d}  1  {:2d}').format(i,i)
            # off-diagonal densities (for couplings)
            for i in range(glbl.pes['n_mcstates']):
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
        subprocess.run(['mcscf.x','-m 2000'])
        # check convergence
        with open('mcscfls','r') as ofile:
            line = ofile.readline()
            if 'final mcscf' in line:
                if 'converged' in ofile.readline():
                    converged = True
                    break

    # save output
    subprocess.copy('mocoef_mc','mocoef')
    return

#
# run mrci if running at that level of theory
#
def run_col_mrci(tid,t_state):

    # if restart file exists, create symbolic link to it
    set_mrci_restart(tid)

    # perform the integral transformation
    with open('tranin','w') as ofile:
        ofile.write("&input\nLUMORB=0\n&end")
    subprocess.run(['tran.x','-m 2000']) 

    # make sure we compute the necessary transition densities
    
    # run mrci
    subprocess.run(['ciudg.x','-m 2000'])
    ci_ener = []
    ci_res  = []
    ci_tol  = []
    converged = True
    with open('ciudgsm','r') as ofile:
        line = ofile.readline()
        if 'final mr-sdci  convergence information' in line:
            for i in range(glbl.pes['n_cistates']):
                ci_info = ofile.readfile().rstrip().split()
                ci_ener.append(float(ci_info[4]))
                ci_res.append(float(ci_info[7]))
                ci_tol.append(float(ci_info[8]))
                converged = converged and ci_res[-1] <= ci_tol[-1]

    # determine convergence...
    if not converged:
        print("EXITING -- MRCI did not converge for trajectory "+str(tid))
        sys.exit()

    # if we're good, update energy array
    energies[tid] = np.fromiter(
                    (ci_ener[i] for i in range(glbl.pes['n_cistates'])),
                     dtype=np.float)

    # now update charges
    ist = -1
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
            charges[tid][ist*natms:(ist+1)*natms] = charg
 
    return    

#
# run dipoles/quadrupoles
#
def run_col_prop(tid,t_state):
    
    nst            = glbl.pes['n_cistates']
    dipoles[tid]   = zeros(3*nst,dtype=np.float)
    quadpoles[tid] = zeros(6*nst,dtype=np.float)

    type_str       = 'ci'
    if glbl.pes['mrci_lvl'] == 0:
        type_str   = 'mc'

    for i in range(nst):
        nocoef      = 'nocoef_'+str(type_str)+'.drt1.state'+str(i)
        mocoef_prop = pathlib.Path('mocoef_prop')
        mocoef_prop.symlink_to(nocoef)
        subprocess.run(['exptvl.x','-m 2000'])
        with open('propls','r') as prop_file:
            line = prop_file.readline()
            if 'Dipole moments' in line:
                for j in range(5):
                    line = prop_file.readline()
                mom_info = line.rstrip().split()
                dip_mom  = np.array([float(mom_info[1]), \
                                     float(mom_info[2]), \
                                     float(mom_info[3])])
                dipoles[tid][3*i:3*(i+1)] = dip_mom
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
                quadpoles[tid][6*i:6*(i+1)] = sec_mom
        shutil.rmtree('mocoef_prop')
    return

#
# Compute transition dipoles between ground and excited state,
# and between trajectory states and other state
#
def run_col_tdipole(t_state):

    init_states = [1, t_state]

    for i in init_states:
        for j in range(1,glbl.pes['n_cistates']+1):
            # only do transition dipoles
            if i == j:
                continue     

            ii = min(i,j)
            jj = max(i,j)

            if glbl.pes['mrci_lvl'] == 0:
                with open('transftin','w') as ofile:
                    ofile.write('y\n1\n'+str(ii)+'\n1\n'+str(jj))
                subprocess.run(['transft.x'],stdin='transftin',stdout='transftls')

                with open('transmomin','w') as ofile:
                    ofile.write('MCSCF\n1 '+str(ii)+'\n1\n'+str(jj))
                subprocess.run(['transmom.x','-m 2000'])

                shutil.rmtree('mcoftfl')
                subprocess.copy('mcoftfl.1','mcoftfl') 

            else:
                with open('trnciin','w') as ofile:
                    ofile.write('&input\nlvlprt=1,\nnroot1='+str(ii)+',\n'+
                                 'nroot2='+str(jj)+',\ndrt1=1,\ndrt2=1,\n&end')
                subprocess.run(['transci.x','-m 2000'])

# perform integral transformation and determine gradient on
# trajectory state
#
def run_col_gradient(t_state):

    target_cid1 = pathlib.Path('cid1fl')
    target_cid2 = pathlib.Path('cid2fl')    

    if glbl.pes['mrci_lvl'] > 0:
        target_civ  = pathlib.Path('civout')
        target_civ.symlink_to('civout.drt1')
        target_cid1.symlink_to('cid1fl.'+str(t_state))
        target_cid2.symlink_to('cid2fl.'+str(t_state))
        subprocess.copy('trancidenin','trainin')       
    else:
        target_cid1.symlink_to('mcsd1fl.'+str(t_state))
        target_cid2.symlink_to('mcsd2fl.'+str(t_state))
        set_nlist_keyword('cigrdin','samcflag',1)
        subprocess.copy('tranmcdenin','tranin')

    # run cigrd
    subprocess.run(['cigrd.x','-m 2000'])
    shutil.move('effd1fl','modens')
    shutil.move('effd2fl','modens2')

    # run tran
    subprocess.run(['tran.x','-m 2000'])
    shutil.rmtree('modens')
    shutil.rmtree('modens2')

    # run dalton
    subprocess.run(['dalton.x','-m 2000'],stdout='abacusls')

    # read in cartesian gradient and save to array
    gradient[tid] = np.zeros(n_cart,dtype=np.float)
    with open('cartgrdls','r') as cart_grd:
        i = 0
        for line in cart_grd:
            gradient[tid][3*i:3*(i+1)] = float(line.rsplit().split())
       
#
# compute couplings to states within prescribed DE window
#
def run_col_coupling(t_state):
      
      # copy some clean files to the work directory
    shutil.copy(scr_path+'/input/cigrdin','cigrdin')
    set_nlist_keyword('cigrdin','nadcalc',1)
    if glbl.pes['mrci_lvl'] == 0:
        set_nlist_keyword('cigrdin','samcflag',1)
        os.symlink(scr_path+'/input/tranmcdenin','tranin')
    else:
        os.symlink(scr_path+'/input/trancidenin','tranin')

    shutil.copy(scr_path+'/input/abacusin','daltcomm')
    insert_dalton_key('daltcomm','COLBUS','.NONUCG')
    
    os.symlink(scr_path+'/input/cidrtfl.cigrd','cidrtfl')
    os.symlink(scr_path+'/input/cidrtfl.cigrd','cidrtfl.1')
    os.symlink('moints.cigrd','moints')
      

    for i in range(1,glbl.pes['n_cistates']+1):
        if i == t_state:
            continue

        s1 = str(min(i,t_state)).strip()
        s2 = str(max(i,t_state)).strip()
        if glbl.pes['mrci_lvl'] == 0:
            os.symlink('mcsd1fl.'+s1+s2,'cid1fl.tr')
            os.symlink('mcsd2fl.'+s1+s2,'cid2fl.tr')   
            os.symlink('mcad1fl.'+s1+s2,'cid1trfl')   
        else:
            os.symlink('cid1fl.'+s1+s2,'cid1fl.tr')
            os.symlink('cid2fl.'+s1+s2,'cid2fl.tr')
            os.symlink('cid1trl.'+s1+s2,'cid1trfl')

        set_nlist_keyword('cigrdin','drt1',1)
        set_nlist_keyword('cigrdin','drt2',1)
        set_nlist_keyword('cigrdin','root1',s1)
        set_nlist_keyword('cigrdin','root2',s2)

        subprocess.run(['cigrd.x','-m 2000'])
 
        shutil.move('effd1fl','modens')
        shutil.move('effd2fl','modens2')
 
        subprocess.run(['tran.x','-m 2000'])
        subprocess.run(['dalton.x','-m 2000'],stdout='abacusls')

      # read in cartesian gradient and save to array
        gradient[tid] = np.zeros(n_cart,dtype=np.float)
        with open('cartgrdls','r') as cart_grd:
            j = 0
            for line in cart_grd:
                coupling[tid][i,3*j:3*(j+1)] = float(line.rsplit().split())

#
# save mocoef and ci files to restart directory
#
def make_col_restart(tid):

    # copy orbitals
    subprocess.copy('mocoef', restart_path+'/mocoef'+str(tid))

    # copy all ci vector, ci info files
    subprocess.copy('civfl' , restart_path+'/civfl.'+str(tid))
    subprocess.copy('civout' , restart_path+'/civout.'+str(tid))
    subprocess.copy('cirefv' , restart_path+'/cirefv.'+str(tid))

#-----------------------------------------------------------------
#
# file parsing
#
#-----------------------------------------------------------------

#
#  copy mocoef file to working directory:
#  1. If first step and parent-less trajectory, take what's in input
#  2. If first step of spawned trajectory, take parents restart info
#  3. If first step of centroid, take one of parent's restart info
#
def set_mocoef_restart(tid):
    # if restart file exists, create symbolic link to it
    mocoef_file = restart_path+'/mocoef.'+str(tid)
    mocoef = pathlib.Path('mocoef')
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
    # if restart file exists, create symbolic link to it
    civfl  = restart_path+'/civfl.'+str(tid)
    civout = restart_path+'/civout.'+str(tid)
    cirefv = restart_path+'/cirefv.'+str(tid)
    pcivfl = pathlib.Path('civfl')
    pcivout = pathlib.Path('civout')
    pcirefv = pathlib.Path('cirefv')
    if os.path.exists(civfl) and os.path.exists(civout) and os.path.exists(cirefv):
        pcivfl.symlink_to(civfl)
        pcivout.symlink_to(civout)
        pcirefv.symlink_to(cirefv)
        return True
    else:
        return False

#
# write a particle array to a COLUMBUS style geom file
#
def write_col_geom(geom):
    f = open('geom','w',encoding='utf-8')
    for i in range(n_atoms):
        f.write(' {:2s}   {:3.1f}  {:12.8f}  {:12.8f}  {:12.8f}  {:12.8f}'). \
        format(geom[i].name, geom[i].anum, geom[i].x[0], geom[i].x[1], geom[i].x[2])
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
def set_nlist_keyword(infile,keyword,value):
    outfile = infile+'.tmp'
    with open(infile,'r') as ifile, open(outfile,'w') as ofile:
        for line in infile:
            if keyword in line:
                ofile.write(keyword+' = '+str(value))
            elif '&end' in line:
                ofile.write(keyword+' = '+str(value))
                ofile.write(line)
    shutil.move(outfile,infile)

#
# insert a dalton keyword (this is a pretty specialized function given
# the idiosyncracies of dalton input)
# the keyword must already exist in the file
#
def insert_dalton_key(infile,keyword,value):
    with open(infile,'r') as ifile, open(outfile,'w') as ofile:
        for line in infile:
            ofile.write(line)
            if keyword in line:
                ofile.write(value)
    shutil.move(outfile,infile)

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

