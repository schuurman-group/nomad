"""
Routines for running a Columbus computation.
"""
import sys
import os
import shutil
import pathlib
import subprocess
import numpy as np
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.atom_lib as atom_lib

# KE operator coefficients a_i:
# T = sum_i a_i p_i^2,
# where p_i is the momentum operator
kecoeff = None

# set to true if we want to compute electronic structure properties
comp_properties = True
# path to columbus input files
input_path = ''
# path to location of 'work'/'restart' directories
work_path = ''
# path to the location of restart files (i.e. mocoef files and civfl)
restart_path = ''
# atom labels
a_sym        = []
# atomic number
a_num        = []
# atomic masses (amu)
a_mass       = []
# by default, since this is the columbus module, assume atoms are
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

class surface_data:
    def __init__(self, n_states, t_dim, crd_dim):

        # necessary for array allocation
        self.n_states     = n_states
        self.t_dim        = t_dim
        self.crd_dim      = crd_dim

        # these are the standard quantities ALL interface_data objects return
        self.data_keys    = []
        self.geom         = np.zeros(t_dim)
        self.energies     = np.zeros(n_states)
        self.grads        = np.zeros((n_states, t_dim))

        # these are interface-specific quantities

# 
def copy_data(orig_info):

    if orig_info is None:
        return None

    new_info = surface_data(orig_info.n_states,
                            orig_info.t_dim,
                            orig_info.crd_dim)

    new_info.data_keys    = copy.copy(orig_info.data_keys)
    new_info.geom         = copy.deepcopy(orig_info.geom)
    new_info.energies     = copy.deepcopy(orig_info.energies)
    new_info.grads        = copy.deepcopy(orig_info.grads)

    return new_info

#----------------------------------------------------------------
#
# Functions called from interface object
#
#----------------------------------------------------------------
def init_interface():
    """Initializes the Columbus calculation from the Columbus input."""
    global columbus_path, input_path, work_path, restart_path
    global a_sym, a_num, a_mass, n_atoms, n_cart, 
    global n_orbs, n_mcstates, n_cistates, max_l, mrci_lvl, mem_str

    global kecoeff

    # KE operator coefficients: Unscaled Cartesian coordinates,
    # a_i = 1/2m_i
    (natm, crd_dim, amp_data, label_data, geom_data, 
          mom_data, width_data, mass_data) = fileio.read_geometry()

    # set atomic symbol, number, mass, 
    a_sym   = [label_data[i].split()[0] for i in range(0,natm*crd_dim,crd_dim)]
    a_mass  = [mass_data[i]  for i in range(0,natm*crd_dim,crd_dim)]
    for i in range(len(a_sym)):
        if atom_lib.valid_atom(a_sym[i]):
            a_num.append(atom_data(a_sym[i])[2])
        else:
            raise ValueError('Atom: '+str(atom_sym)+' not found in library'))

    # set coefficient for kinetic energy determination            
    kecoeff = 0.5/mass_data[0:natm*crd_dim]

    # confirm that we can see the COLUMBUS installation (pull the value
    # COLUMBUS environment variable)
    columbus_path = os.environ['COLUMBUS']
    if not os.path.isfile(columbus_path + '/ciudg.x'):
        raise FileNotFoundError('Cannot find COLUMBUS executables in: ' +
                                columbus_path)
    # ensure COLUMBUS input files are present locally
    if not os.path.exists('input'):
        raise FileNotFoundError('Cannot find COLUMBUS input files in: input')

    # setup working directories
    input_path    = fileio.scr_path + '/input'
    work_path     = fileio.scr_path + '/work'
    restart_path  = fileio.scr_path + '/restart'

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
    n_atoms    = natm
    n_cart     = natm * crd_dim
    n_orbs     = int(read_pipe_keyword('input/cidrtmsin',
                                       'orbitals per irrep'))
    n_mcstates = int(read_nlist_keyword('input/mcscfin',
                                        'NAVST'))
    n_cistates = int(read_nlist_keyword('input/ciudgin.drt1',
                                        'NROOT'))
    mrci_lvl   = int(read_pipe_keyword('input/cidrtmsin',
                                       'maximum excitation level'))
    max_l      = ang_mom_dalton('input/daltaoin')


    # all COLUMBUS modules will be run with the amount of meomry specified by mem_per_core
    mem_str = str(glbl.columbus['mem_per_core'])

    # generate one time input files for columbus calculations
    make_one_time_input()


def evalutate_trajectory(tid, geom, state):
    """Evaluates all requested electronic structure information for a
    single trajectory."""
    if tid < 0:
        print('evaluate_trajectory called with ' +
              'id associated with centroid, tid=' + str(tid))
    # run_trajectory returns None...
    surf_info = run_trajectory(tid, geom, state)

    return surf_info


def evalutate_centroid(tid, geom, states):
    """Evaluates  all requested electronic structure information at a
    centroid."""
    if tid >= 0:
        print('evaluate_centroid called with ' +
              'id associated with trajectory, tid=' + str(tid))
    # run_centroid returns None...
    state_i = states[0]
    state_j = states[1]
    surf_info = run_centroid(tid, geom, state_i, state_j)

    return surf_info

def evaluate_worker(args, global_vars):
    """Evaluates worker on a slave mode."""
    tid    = args[0]
    geom   = args[1]
    tstate = args[2]
    cstate = args[3]

    set_global_vars(global_vars)

    # run_trajectory and run_centroid both return None...
    if tid >= 0:
        surf_info = run_trajectory(tid, geom, tstate)
    else:
        surf_info = run_centroid(tid, geom, tstate, cstate)

    return surf_info


#----------------------------------------------------------------
#
#  "Private" functions
#
#----------------------------------------------------------------
def in_cache(tid, geom):
    """Determines if electronic structure calculation is in the cache."""
    global current_geom, n_atoms, p_dim

    if tid not in current_geom:
        return False
    difg = np.linalg.norm(geom - current_geom[tid])
    if difg <= glbl.fpzero:
        return True
    return False


def run_trajectory(tid, geom, tstate):
    """Computes MCSCF/MRCI energy and computes all couplings.

    For the columbus module, since gradients are not particularly
    time consuming, it's easier (and probably faster) to compute
    EVERYTHING at once (i.e. all energies, all gradients, all properties)
    Thus, if electronic structure information is not up2date, all methods
    call the same routine: run_single_point.
    """
    global p_dim, n_cistates, n_atoms, current_geom

    # write geometry to file
    write_col_geom(geom)

    # generate integrals
    generate_integrals(tid)

    # run mcscf
    run_col_mcscf(tid, tstate)

    # run mrci, if necessary
    run_col_mrci(tid, tstate)

    # run properties, dipoles, etc.
    run_col_multipole(tid, tstate)

    # run transition dipoles
    init_states = [0, tstate]
    for i in init_states:
        for j in range(n_cistates):
            if i != j or (j in init_states and j < i):
                run_col_tdipole(tid, i, j)

    # compute gradient on current state
    run_col_gradient(tid, tstate)

    # run coupling to other states
    run_col_coupling(tid, tstate)

    # save restart files
    make_col_restart(tid)

    # update the geometry in the cache
    current_geom[tid] = geom

def run_centroid(tid, geom, state_i, state_j):
    """Returns an energy (same state) or coupling (different states)
    of a centroid."""
    global p_dim, n_atoms, current_geom
    s1 = min(state_i, state_j)
    s2 = max(state_i, state_j)

    # write geometry to file
    write_col_geom(geom)

    # generate integrals
    generate_integrals(tid)

    # run mcscf
    run_col_mcscf(tid, state_i)

    # run mrci, if necessary. We also need transition densities if we need gradients
    if state_i != state_j:
        t_den = [[s1, s2]]
    else:
        t_den = []
    run_col_mrci(tid, state_i, density=t_den, int_trans=False, apop=False)

    if state_i != state_j:
        # run coupling to other states
        run_col_coupling(tid, state_i, state_j)

    # save restart files
    make_col_restart(tid)

    # update the geometry in the cache
    current_geom[tid] = geom

#----------------------------------------------------------------
#
# Routines for running columbus
#
#---------------------------------------------------------------
def make_one_time_input():
    """Creates a Columbus input for MRCI calculations."""
    global work_path

    sys.stdout.flush()
    # all calculations take place in work_dir
    os.chdir(work_path)

    # rotation matrix
    with open('rotmax', 'w') as rfile:
        rfile.write('  1  0  0\n  0  1  0\n  0  0  1')

    # cidrtfil files
    with open('cidrtmsls', 'w') as cidrtmsls, open('cidrtmsin', 'r') as cidrtmsin:
        subprocess.run(['cidrtms.x', '-m', mem_str], stdin=cidrtmsin,
                       stdout=cidrtmsls)
    shutil.move('cidrtfl.1', 'cidrtfl.ci')
    with open('cidrtmsls.cigrd', 'w') as cidrtmsls_grd, open('cidrtmsin.cigrd', 'r') as cidrtmsin_grd:
        subprocess.run(['cidrtms.x', '-m', mem_str], stdin=cidrtmsin_grd,
                       stdout=cidrtmsls_grd)
    shutil.move('cidrtfl.1', 'cidrtfl.cigrd')

    # check if hermitin exists, if not, copy daltcomm
    if not os.path.exists('hermitin'):
        shutil.copy('daltcomm', 'hermitin')

    # make sure ciudgin file exists
    shutil.copy('ciudgin.drt1', 'ciudgin')


def generate_integrals(tid):
    """Runs Dalton to generate AO integrals."""
    global work_path

    os.chdir(work_path)

    # run unik.gets.x script
    with open('unikls', 'w') as unikls:
        subprocess.run(['unik.gets.x'], stdout=unikls,
                       universal_newlines=True, shell=True)

    # run hernew
    subprocess.run(['hernew.x'])
    shutil.move('daltaoin.new', 'daltaoin')

    # run dalton.x
    shutil.copy('hermitin', 'daltcomm')
    with open('hermitls', 'w') as hermitls:
        subprocess.run(['dalton.x', '-m', mem_str], stdout=hermitls,
                       universal_newlines=True, shell=True)

    #append_log(tid,'integral')


def run_col_mcscf(tid, t_state):
    """Runs MCSCF program."""
    global work_path, n_mcstates, mrci_lvl

    os.chdir(work_path)

    # get an initial starting set of orbitals
    set_mcscf_restart(tid)

    # allow for multiple DRTs in the mcscf part. For example, We may
    # want to average over a' and a" in a tri-atomic case
    for i in range(1, n_drt+1):
        shutil.copy('mcdrtin.' + str(i), 'mcdrtin')
        with open('mcdrtls', 'w') as mcdrtls, open('mcdrtin', 'r') as mcdrtin:
            subprocess.run(['mcdrt.x', '-m', mem_str], stdin=mcdrtin,
                           stdout=mcdrtls)
        with open('mcuftls', 'w') as mcuftls:
            subprocess.run(['mcuft.x'], stdout=mcuftls)

        # save formula tape and log files for each DRT
        shutil.copy('mcdrtfl', 'mcdrtfl.' + str(i))
        shutil.copy('mcdftfl', 'mcdftfl.' + str(i))
        shutil.copy('mcuftls', 'mcuftls.' + str(i))
        shutil.copy('mcoftfl', 'mcoftfl.' + str(i))

    # if running cas dynamics (i.e. no mrci), make sure we compute the
    # mcscf/cas density (for gradients and couplings)
    if mrci_lvl == 0:
        with open('mcdenin', 'w', encoding='utf-8') as mcden:
            mcden.write('MCSCF')
            # diagonal densities (for gradients)
            for i in range(n_mcstates):
                mcden.write('1  {:2d}  1  {:2d}').format(i, i)
            # off-diagonal densities (for couplings)
            for i in range(n_mcstates):
                mcden.write('1  {:2d}  1  {:2d}').format(min(i, t_state),
                                                         max(i, t_state))

    # try running mcscf a couple times this can be tweaked if one
    # develops other strategies to deal with convergence problems
    converged = False
    run_max   = 3
    n_run     = 0
    while not converged and n_run < run_max:
        n_run += 1
        if n_run == 3:
            # disable orbital-state coupling if convergence an issue
            ncoupl = int(read_nlist_keyword('mcscfin', 'ncoupl'))
            niter  = int(read_nlist_keyword('mcscfin', 'niter'))
            set_nlist_keyword('mcscfin', 'ncoupl', niter+1)
        subprocess.run(['mcscf.x -m ' + mem_str], shell=True)
        # check convergence
        with open('mcscfls', 'r') as ofile:
            for line in ofile:
                if '*converged*' in line:
                    converged = True
                    break

    # if not converged, we have to die here...
    if not converged:
        raise TimeoutError('MCSCF not converged.')

    # save output
    shutil.copy('mocoef_mc', 'mocoef')

    # grab mcscfls output
    #append_log(tid,'mcscf')


def run_col_mrci(tid, t_state, density=None, int_trans=True, apop=True):
    """Runs MRCI if running at that level of theory."""
    global energies, atom_pops, input_path, work_path, n_atoms, n_cistates

    os.chdir(work_path)

    # if restart file exists, create symbolic link to it
    set_mrci_restart(tid)

    # get a fresh ciudgin file
    shutil.copy(input_path + '/ciudgin.drt1', 'ciudgin')

    # update the transition section in ciudgin
    if density:
        tran_den = density

    # otherwise, assume we need transition densities to evaluate
    # transition dipoles and derivative couplings
    else:
        tran_den = []
        init_states = [0, t_state]
        for i in init_states:
            for j in range(n_cistates):
                if i == j or (j in init_states and j < i):
                    continue
                tran_den.append([min(i,j)+1, max(i,j)+1])

    # append entries in tran_den to ciudgin file
    with open('ciudgin', 'r') as ciudgin:
        ci_file = ciudgin.readlines()
    with open('ciudgin', 'w') as ciudgin:
        for line in ci_file:
            ciudgin.write(line)
            if '&end' in line:
                break
        ciudgin.write('transition\n')
        for i in len(tran_den):
            ciudgin.write('  1 {:2d}  1 {:2d}\n'.format(tran_den[i,1],
                                                        tran_den[i,2]))

    # make sure we point to the correct formula tape file
    link_force('cidrtfl.ci', 'cidrtfl')
    link_force('cidrtfl.ci', 'cidrtfl.1')

    # perform the integral transformation
    with open('tranin', 'w') as ofile:
        ofile.write('&input\nLUMORB=0\n&end')
    subprocess.run(['tran.x', '-m', mem_str])

    # run mrci
    subprocess.run(['ciudg.x', '-m', mem_str])
    ci_ener = []
    ci_res  = []
    ci_tol  = []
    mrci_iter = False
    converged = True
    with open('ciudgsm', 'r') as ofile:
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
        raise TimeoutError('MRCI did not converge for trajectory ' + str(tid))

    # if we're good, update energy array
    energies[tid] = np.fromiter((ci_ener[i] for i in range(n_cistates)),
                                dtype=float)

    # now update atom_pops
    if apop:
        ist = -1
        atom_pops[tid] = np.zeros((n_cistates, n_atoms))
        with open('ciudgls', 'r') as ciudgls:
            for line in ciudgls:
                if '   gross atomic populations' in line:
                    ist += 1
                    pops = []
                    for i in range(np.ceil(n_atoms/6.)):
                        for j in range(max_l+3):
                            line = ciudgls.readline()
                        l_arr = line.rstrip().split()
                        pops.extend(l_arr[1:])
                        line = ciudgls.readline()
                    atom_pops[tid][ist,:] = np.array([float(x) for x in pops])

    # grab mrci output
    #append_log(tid,'mrci')

    # transform integrals using cidrtfl.cigrd
    if int_trans:
        frzn_core = int(read_nlist_keyword('cigrdin', 'assume_fc'))
        if frzn_core == 1:
            os.remove('moints')
            os.remove('cidrtfl')
            os.remove('cidrtfl.1')
            link_force('cidrtfl.cigrd', 'cidrtfl')
            link_force('cidrtfl.cigrd', 'cidrtfl.1')
            shutil.copy(input_path + '/tranin', 'tranin')
            subprocess.run(['tran.x', '-m', mem_str])


def run_col_multipole(tid,t_state):
    """Runs dipoles / second moments."""
    global p_dim, dip_moms, sec_moms, work_path, n_cistates, mrci_lvl

    os.chdir(work_path)

    nst            = n_cistates
    dip_moms[tid]  = np.zeros((n_cistates, n_cistates, p_dim))
    sec_moms[tid]  = np.zeros((n_cistates, p_dim))

    if mrci_lvl == 0:
        type_str   = 'mc'
    else:
        type_str   = 'ci'

    for istate in range(nst):
        i1 = istate + 1
        link_force('nocoef_' + str(type_str) + '.drt1.state' + str(i1),
                   'mocoef_prop')
        subprocess.run(['exptvl.x', '-m', mem_str])
        with open('propls', 'r') as prop_file:
            for line in prop_file:
                if 'Dipole moments' in line:
                    for j in range(5):
                        line = prop_file.readline()
                    l_arr = line.rstrip().split()
                    dip_moms[tid][istate,istate,:] = np.array([float(l_arr[1]),
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


def run_col_tdipole(tid, state_i, state_j):
    """Computes transition dipoles between ground and excited state,
    and between trajectory states and other state."""
    global p_dim, dip_moms, n_cistates, work_path, mrci_lvl

    os.chdir(work_path)

    # make sure we point to the correct formula tape file
    link_force('civfl', 'civfl.drt1')
    link_force('civout', 'civout.drt1')
    link_force('cirefv', 'cirefv.drt1')

    i1 = min(state_i, state_j) + 1
    j1 = max(state_i, state_j) + 1

    if state_i == state_j:
        return None

    if mrci_lvl == 0:
        with open('transftin', 'w') as ofile:
            ofile.write('y\n1\n' + str(j1) + '\n1\n' + str(i1))
            subprocess.run(['transft.x'], stdin='transftin',
                           stdout='transftls')

        with open('transmomin', 'w') as ofile:
            ofile.write('MCSCF\n1 ' + str(j1) + '\n1\n' + str(i1))
            subprocess.run(['transmom.x', '-m', mem_str])

        os.remove('mcoftfl')
        shutil.copy('mcoftfl.1', 'mcoftfl')

    else:
        with open('trnciin', 'w') as ofile:
            ofile.write(' &input\n lvlprt=1,\n nroot1=' + str(i1) + ',\n' +
                        ' nroot2=' + str(j1) + ',\n drt1=1,\n drt2=1,\n &end')
        subprocess.run(['transci.x', '-m', mem_str])
        shutil.move('cid1trfl', 'cid1trfl.' + str(i1) + '.' + str(j1))

    with open('trncils', 'r') as trncils:
        for line in trncils:
            if 'total (elec)' in line:
                line_arr = line.rstrip().split()
                for dim in range(p_dim):
                    dip_moms[tid][state_i,state_j,dim] = float(line_arr[dim+2])
                    dip_moms[tid][state_j,state_i,dim] = float(line_arr[dim+2])


def run_col_gradient(tid, t_state):
    """Performs integral transformation and determine gradient on
    trajectory state."""
    global p_dim, gradients, input_path, work_path, mrci_lvl, n_cistates, n_cart

    os.chdir(work_path)
    shutil.copy(input_path + '/cigrdin', 'cigrdin')
    gradients[tid] = np.zeros((n_cistates, n_cart))
    tindex = t_state + 1

    if mrci_lvl > 0:
        link_force('cid1fl.drt1.state' + str(tindex), 'cid1fl')
        link_force('cid2fl.drt1.state' + str(tindex), 'cid2fl')
        shutil.copy(input_path + '/trancidenin', 'tranin')
    else:
        link_force('mcsd1fl.' + str(tindex), 'cid1fl')
        link_force('mcsd2fl.' + str(tindex), 'cid2fl')
        set_nlist_keyword('cigrdin', 'samcflag', 1)
        shutil.copy(input_path + '/tranmcdenin', 'tranin')

    # run cigrd
    set_nlist_keyword('cigrdin', 'nadcalc', 0)
    subprocess.run(['cigrd.x', '-m', mem_str])
    os.remove('cid1fl')
    os.remove('cid2fl')
    shutil.move('effd1fl', 'modens')
    shutil.move('effd2fl', 'modens2')

    # run tran
    subprocess.run(['tran.x', '-m', mem_str])
    os.remove('modens')
    os.remove('modens2')

    # run dalton
    shutil.copy(input_path + '/abacusin', 'daltcomm')
    with open('abacusls', 'w') as abacusls:
        subprocess.run(['dalton.x', '-m', mem_str], stdout=abacusls)
    shutil.move('abacusls', 'abacusls.grad')

    # read in cartesian gradient and save to array
    with open('cartgrd', 'r') as cartgrd:
        i = 0
        for line in cartgrd:
            l_arr = line.rstrip().split()
            for j in range(p_dim):
                gradients[tid][t_state,p_dim*i+j] = float(l_arr[j].replace('D', 'e'))
            i = i + 1

    # grab cigrdls output
    #append_log(tid,'cigrd')


def run_col_coupling(tid, t_state, coup_state=None):
    """Computes couplings to states within prescribed DE window."""
    global p_dim, couplings, input_path, work_path, n_cistates, mrci_lvl, n_cart

    if coup_state is None:
        c_states = range(n_cistates)
    else:
        c_states = [coup_state]

    os.chdir(work_path)

    for c_state in c_states:
        if c_state == t_state:
            continue

        s1 = str(min(t_state, c_state) + 1).strip()
        s2 = str(max(t_state, c_state) + 1).strip()

        # copy some clean files to the work directory
        shutil.copy(input_path + '/cigrdin', 'cigrdin')
        set_nlist_keyword('cigrdin', 'nadcalc', 1)
        if mrci_lvl == 0:
            set_nlist_keyword('cigrdin', 'samcflag', 1)
            shutil.copy(input_path + '/tranmcdenin', 'tranin')
        else:
            shutil.copy(input_path + '/trancidenin', 'tranin')

        shutil.copy(input_path + '/abacusin', 'daltcomm')
        insert_dalton_key('daltcomm', 'COLBUS', '.NONUCG')

        if mrci_lvl == 0:
            link_force('mcsd1fl.trd' + s1 + 'to' + s2, 'cid1fl.tr')
            link_force('mcsd2fl.trd' + s1 + 'to' + s2, 'cid2fl.tr')
            link_force('mcad1fl.' + s1 + s2, 'cid1trfl')
        else:
            link_force('cid1fl.trd' + s1 + 'to' + s2, 'cid1fl.tr')
            link_force('cid2fl.trd' + s1 + 'to' + s2, 'cid2fl.tr')
            link_force('cid1trfl.' + s1 + '.' + s2, 'cid1trfl')

        set_nlist_keyword('cigrdin', 'drt1', 1)
        set_nlist_keyword('cigrdin', 'drt2', 1)
        set_nlist_keyword('cigrdin', 'root1', s1)
        set_nlist_keyword('cigrdin', 'root2', s2)

        subprocess.run(['cigrd.x', '-m', mem_str])

        shutil.move('effd1fl', 'modens')
        shutil.move('effd2fl', 'modens2')

        subprocess.run(['tran.x', '-m', mem_str])
        with open('abacusls', 'w') as abacusls:
            subprocess.run(['dalton.x', '-m', mem_str], stdout=abacusls)

        # read in cartesian gradient and save to array
        with open('cartgrd', 'r') as cartgrd:
            lines = cartgrd.read().splitlines()
        grad = [lines[i].split() for i in range(len(lines))]
        new_coup = np.array([item.replace('D', 'e') for row in grad
                             for item in row], dtype=float)
        print('new_coup=' + str(new_coup))

        delta_e = energies[tid][t_state] - energies[tid][c_state]
        new_coup /= delta_e

        c_phase = get_adiabatic_phase(new_coup,
                                      gradients[tid][t_state,c_state,:])
        gradients[tid][c_state,:] =  c_phase * new_coup
        shutil.move('cartgrd', 'cartgrd.nad.' + str(s1) + '.' + str(s2))

    # grab mcscfls output
    #append_log(tid,'nad')


def make_col_restart(tid):
    """Saves mocoef and ci files to restart directory."""
    global work_path

    os.chdir(work_path)

    # move orbitals
    shutil.move('mocoef', restart_path + '/mocoef.' + str(tid))

    # move all ci vector, ci info files
    shutil.move('civfl', restart_path + '/civfl.' + str(tid))
    shutil.move('civout', restart_path + '/civout.' + str(tid))
    shutil.move('cirefv', restart_path + '/cirefv.' + str(tid))

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


def get_adiabatic_phase(new_coup, old_coup):
    """Determines the phase of the computed coupling that yields smallest
    change from previous coupling."""
    # if the previous coupling is vanishing, phase of new coupling is arbitrary
    if np.linalg.norm(old_coup) <= glbl.fpzero:
        return 1.

    # check the difference between the vectors assuming phases of +1/-1
    norm_pos = np.linalg.norm(new_coup - old_coup)
    norm_neg = np.ligalg.norm(-new_coup - old_coup)

    if norm_pos < norm_neg:
        return 1.
    else:
        return -1.


#----------------------------------------------------------------
#
# Methods for setting and passing global variables (necessary for
# parallel runs
#
#---------------------------------------------------------------
def get_global_vars():
    """Gets the list of global variables."""
    global input_path, work_path, restart_path
    global a_sym, a_num, a_mass, p_dim, n_atoms, n_cart
    global n_drt, n_orbs, n_mcstates, n_cistates, max_l, mrci_lvl, mem_str

    gvars = [input_path, work_path, restart_path, 
             a_sym, a_num, a_mass, p_dim, n_atoms, n_cart,
             n_drt, n_orbs, n_mcstates, n_cistates, max_l, mrci_lvl, mem_str]

    return gvars


def set_global_vars(gvars):
    """Sets the global variables."""
    global input_path, work_path, restart_path
    global a_sym, a_num, a_mass, p_dim, n_atoms, n_cart, 
    global n_drt, n_orbs, n_mcstates, n_cistates, max_l, mrci_lvl, mem_str

    input_path   = gvars[0]
    work_path    = gvars[1]
    restart_path = gvars[2]
    a_sym        = gvals[3]
    a_num        = gvals[4]
    a_mass       = gvals[5]
    p_dim        = gvars[6]
    n_atoms      = gvars[7]
    n_cart       = gvars[8]
    n_drt        = gvars[9]
    n_orbs       = gvars[10]
    n_mcstates   = gvars[11]
    n_cistates   = gvars[12]
    max_l        = gvars[13]
    mrci_lvl     = gvars[14]
    mem_str      = gvars[15]

#-----------------------------------------------------------------
#
# File parsing
#
#-----------------------------------------------------------------
def append_log(tid, listing_file):
    """Grabs key output from columbus listing files.

    Useful for diagnosing electronic structure problems.
    """
    if listing_file == 'integral':
        with open('hermitls', 'r') as hermitls:
            for line in hermitls:
                if 'Bond distances' in line:
                    while 'Nuclear repulsion energy' not in line:
                        print(line)
                        line = hermitls.readline()
                    break
    elif listing_file == 'mcscf':
        with open('mcscfls', 'r') as mcscfls:
            for line in mcscfls:
                if 'final mcscf' in line:
                    while len(line.rstrip()) != 0:
                        print(line)
                        line = mcscfls.readline()
                    break
    elif listing_file == 'mrci':
        with open('ciudgsm', 'r') as ciudgls:
            for line in ciudgls:
                if 'final mr-sdci  convergence information' in line:
                    while len(line.rstrip()) != 0:
                        print(line)
                        line = ciudgls.readline()
                    break
    elif listing_file == 'cigrd':
        with open('cigrdls', 'r') as cigrdls:
            for line in cigrdls:
                if 'RESULTS' in line:
                    while 'effective' not in line:
                        print(line)
                        line = cigrdls.readline()
                    break
    elif listing_file == 'nad':
        with open('cigrdls', 'r') as cigrdls_nad:
            for line in cigrdls_nad:
                if 'RESULTS' in line:
                    while 'effective' not in line:
                        print(line)
                        line = cigrdls_nad.readline()
                    break
    else:
        print('listing file: ' + str(listing_file) + ' not recognized.')


def set_mcscf_restart(tid):
    """Copys mocoef file to working directory.

    1. If first step and parent-less trajectory, take what's in input.
    2. If first step of spawned trajectory, take parents restart info.
    3. If first step of centroid, take one of parent's restart info.
    """
    global work_path

    os.chdir(work_path)

    # if restart file exists, create symbolic link to it
    mocoef_file = restart_path + '/mocoef.' + str(tid)
    if os.path.exists(mocoef_file):
        shutil.copy(mocoef_file, 'mocoef')
        return True
    else:
        return False


def set_mrci_restart(tid):
    """Copys/links CI restart files to working directory.

    Restart logic is the same as set_mcscf_restart.
    """
    global work_path

    os.chdir(work_path)

    # if restart file exists, create symbolic link to it
    civfl  = restart_path + '/civfl.' + str(tid)
    civout = restart_path + '/civout.' + str(tid)
    cirefv = restart_path + '/cirefv.' + str(tid)
    if (os.path.exists(civfl) and os.path.exists(civout)
            and os.path.exists(cirefv)):
        link_force(civfl, 'civfl')
        link_force(civout, 'civout')
        link_force(cirefv, 'cirefv')
        return True
    else:
        return False

def write_col_geom(geom):
    """Writes a array of atoms to a COLUMBUS style geom file."""
    global n_atoms, a_sym, a_num, a_mass, work_path

    os.chdir(work_path)

    f = open('geom', 'w', encoding='utf-8')
    for i in range(n_atoms):
        f.write(' {:2s}   {:3.1f}  {:12.8f}  {:12.8f}  {:12.8f}  {:12.8f}'
                '\n'.format(a_sym[i], a_num[i], 
                            geom[p_dim*i],geom[p_dim*i+1],geom[p_dim*i+2],
                            a_mass[i]/glbl.mass2au))
    f.close()


def read_pipe_keyword(infile, keyword):
    """Reads from a direct input file via keyword search."""
    f = open(infile, 'r', encoding='utf-8')
    for line in f:
        if keyword in line:
            f.close()
            return line.split()[0]


def read_nlist_keyword(infile, keyword):
    """Reads from a namelist style input."""
    f = open(infile, 'r', encoding='utf-8')
    for line in f:
        if keyword in line:
            f.close()
            line = line.rstrip("\r\n")
            return line.split('=', 1)[1].strip(' ,')


def set_nlist_keyword(file_name, keyword, value):
    """Writes a namelist style input."""
    outfile = str(file_name) + '.tmp'
    key_found = False
    with open(file_name, 'r') as ifile, open(outfile, 'w') as ofile:
        for line in ifile:
            if keyword in line:
                ofile.write(str(keyword) + ' = ' + str(value) + '\n')
                key_found = True
            elif '&end' in line and not key_found:
                ofile.write(str(keyword) + ' = ' + str(value) + ',\n')
                ofile.write(line)
            else:
                ofile.write(line)
    shutil.move(outfile, file_name)


def insert_dalton_key(infile, keyword, value):
    """Insert a Dalton keyword.

    This is a pretty specialized function given the idiosyncracies of
    dalton input. The keyword must already exist in the file.
    """
    with open(infile, 'r') as ifile, open('tempfile', 'w') as ofile:
        for line in ifile:
            ofile.write(line)
            if keyword in line:
                ofile.write(value + '\n')
    shutil.move('tempfile', infile)


def ang_mom_dalton(infile):
    """Finds maximum ang. mom. in basis set from dalton."""
    max_l = 0
    with open(infile, 'r') as daltaoin:
        for i in range(4):
            line = daltaoin.readline()
        l_arr = line.rstrip().split()
        n_grps = int(l_arr[1])
        for i in range(n_grps):
            line = daltaoin.readline()
            l_arr = line.rstrip().split()
            n_atm = int(l_arr[1])
            n_ang = int(l_arr[2]) - 1
            # max_l on first line
            max_l = max(max_l, n_ang)
            n_con = [int(l_arr[j]) for j in range(3, 3+n_ang+1)]
            for j in range(n_atm):
                line = daltaoin.readline()
            for j in range(len(n_con)):
                for k in range(n_con[j]):
                    line = daltaoin.readline()
                    nprim = int(line.rstrip().split()[1])
                    for l in range(nprim):
                        line = daltaoin.readline()
    return max_l


def file_len(fname):
    """Returns the number of lines in a file."""
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def link_force(target, link_name):
    """Creates a symbolic link, overwriting existing link if necessary."""
    try:
        os.symlink(target, link_name)
    except FileExistsError:
        os.unlink(link_name)
        os.symlink(target, link_name)


def load_orbitals(tid):
    """Loads orbitals into a mocoef file."""
    pass


def write_orbitals(fname, orb_array):
    """Writes orbitals to mocoef file."""
    pass
