"""
Routines for reading input files and writing log files.
"""
import os
import re
import glob
import shutil
import numpy as np
import src.dynamics.timings as timings
import src.fmsio.glbl as glbl
import src.basis.atom_lib as atom_lib

# Make sure that we print entire arrays
np.set_printoptions(threshold = np.inf)

home_path   = ''
scr_path    = ''
tkeys       = ['traj_dump', 'ener_dump', 'coup_dump', 'dipole_dump',
               'secm_dump', 'tran_dump', 'apop_dump', 'grad_dump']
bkeys       = ['pop_dump', 'bener_dump', 'spawn_dump',
               's_mat', 'sdot_mat', 'h_mat', 'heff_mat','t_ovrlp','auto.dat']
dump_header = dict()
dump_format = dict()
log_format  = dict()
tfile_names = dict()
bfile_names = dict()
print_level = dict()

def read_input_file():
    """Reads the fms.input files.

    This file contains variables related to the running of the
    dynamics simulation.
    """
    global scr_path, home_path

    # save the name of directory where program is called from
    home_path = os.getcwd()

    # set a sensible default for scr_path
    scr_path = os.environ['TMPDIR']
    if os.path.exists(scr_path) and glbl.mpi['rank']==0:
        shutil.rmtree(scr_path)
        os.makedirs(scr_path)

    # Read fms.input. Valid sections are:
    #   initial_conditions
    #   propagation
    #   spawning
    #   interface
    #   geometry
    #   printing

    # Read fms.input. Small enough to gulp the whole thing
    with open('fms.input', 'r') as infile:
        fms_input = infile.readlines()

    # remove comment lines
    fms_input = [item for item in fms_input if 
                 not item.startswith("#") and not item.startswith("!")]
    print("all input="+str(fms_input))

    sec_strings = list(glbl.input_groups)
    print("search string 1=|"+str('begin '+sec_strings[0]+'-section')+"|")
 
    current_line = 0
    # look for begining of input section
    while current_line < len(fms_input):
        print("current_line="+str(fms_input[current_line]))
        sec_start = [re.search(str('begin '+sec_strings[i]+'-section'),fms_input[current_line]) 
                     for i in range(len(sec_strings))]
        print("sec_start="+str(sec_start))
        if all([v is None for v in sec_start]):
            current_line+=1
        else:
            print("about to parse")
            section = next(item for item in sec_start 
                           if item is not None).string
            print("section expand="+section)
            section = section.replace('-section','').replace('begin','').strip()
            print("section: "+section+" found. parsing...")
            current_line = parse_section(fms_input, current_line, section)    
    
    for i in range(len(sec_strings)):
        keys  = list(glbl.input_groups[sec_strings[i]])
        print("keys["+str(i)+"]="+str(keys))
        for j in range(len(keys)):
            print(keys[j]+' = '+glbl.input_groups[sec_strings[i]][keys[j]])

#
# set keywords in the appropriate keyword dictionary by parsing
# input array
#
def parse_section(kword_array, line_start, section):
    """Reads a namelist style input, returns results in dictionary.""" 

    current_line = line_start + 1
    print("current_line="+str(current_line))
    while (current_line < len(kword_array) and 
           re.search('end '+section+'-section',kword_array[current_line]) is None):
        line = kword_array[current_line].rstrip('\r\n')

        # allow for multi-line input
        while ("=" not in kword_array[current_line+1] and 
               'end '+section+'-section' not in kword_array[current_line+1]):
            current_line += 1
            line += kword_array[current_line].rstrip('\r\n').strip()

        print("line="+str(line))
        key,value = line.split('=',1)
        key   = key.strip()
        value = value.strip()
        
        if key not in glbl.input_groups[section].keys():
            if glbl.mpi['rank'] == 0:
                print("Cannot find input parameter: "+key+
                      " in input section: "+section)
        else:
            # put all variable types into a flat list
            # here we explicitly consider dimension 0,1,2 lists: which
            # is pretty messy.
            valid = True
            if glbl.keyword_type[key][1] == 2:
                try:
                    varcast = [glbl.keyword_type[key][0](item) 
                               for item in sublist for sublist in value]
                except ValueError:
                    valid = False
                    print("Can't read variable: "+str(key)+ 
                          " as nested list of "+str(glbl.keyword_type[key][0]))
            elif glbl.keyword_type[key][1] == 1:
                try:
                    varcast = [glbl.keyword_type[key][0](item) for item in value]
                except ValueError:
                    valid = False
                    print("Can't read variable: "+str(key)+ 
                          " as list of "+str(glbl.keyword_type[key][0]))
            else:
                try:
                    varcast = glbl.keyword_type[key][0](value)
                except ValueError:
                    valid = False
                    print("Can't read variable: "+str(key)+
                          " as a "+str(glbl.keyword_type[key][0]))

            if valid:
                glbl.input_groups[section][key] = varcast

        current_line+=1

    return current_line

def init_fms_output():
    """Initialized all the output format descriptors."""
    global home_path, scr_path, log_format, tkeys, bkeys
    global dump_header, dump_format, tfile_names, bfile_names, print_level

    (ncrd, crd_dim, amp_data, label_data,
     geom_data, mom_data, width_data, mass_data, state_data) = read_geometry()

    nst = int(glbl.propagate['n_states'])
    dstr = ('x', 'y', 'z')
    acc1 = 12
    acc2 = 16

    # ----------------- dump formats (trajectory files) -----------------
    # trajectory output
    arr1 = ['{:>12s}'.format('pos' + str(i+1) + '.' + dstr[x])
            for i in range(ncrd) for x in range(crd_dim)]
    arr2 = ['{:>12s}'.format('mom' + str(i+1) + '.' + dstr[x])
            for i in range(ncrd) for x in range(crd_dim)]
    tfile_names[tkeys[0]] = 'trajectory'
    dump_header[tkeys[0]] = ('Time'.rjust(acc1) + ''.join(arr1) +
                             ''.join(arr2) + 'Phase'.rjust(acc1) +
                             'Re[Amp]'.rjust(acc1) + 'Im[Amp]'.rjust(acc1) +
                             'Norm[Amp]'.rjust(acc1) + 'State'.rjust(acc1) +
                             '\n')
    dump_format[tkeys[0]] = ('{:12.4f}'+
                             ''.join('{:12.6f}' for i in range(2*ncrd*crd_dim+5))+
                             '\n')

    # potential energy
    arr1 = ['{:>16s}'.format('potential.' + str(i)) for i in range(nst)]
    tfile_names[tkeys[1]] = 'poten'
    dump_header[tkeys[1]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[1]] = ('{:12.4f}' +
                             ''.join('{:16.10f}' for i in range(nst)) + '\n')

    # gradients
    arr1 = ['        crd' + str(i+1) + '.' + dstr[j]
            for i in range(ncrd) for j in range(crd_dim)]
    tfile_names[tkeys[7]] = 'gradient'
    dump_header[tkeys[7]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[7]] = ('{0:>12.4f}' +
                             ''.join('{' + str(i) + ':14.8f}'
                                     for i in range(1, ncrd*crd_dim+1)) + '\n')

    # coupling
    arr1 = ['{:>12s}'.format('coupling.' + str(i)) for i in range(nst)]
    arr2 = ['{:>12s}'.format('c * v .' + str(i)) for i in range(nst)]
    tfile_names[tkeys[2]] = 'coupling'
    dump_header[tkeys[2]] = ('Time'.rjust(acc1) + ''.join(arr1) +
                             ''.join(arr2) + '\n')
    dump_format[tkeys[2]] = ('{:12.4f}' +
                             ''.join('{:12.5f}' for i in range(2*nst)) + '\n')

    # permanent dipoles
    arr1 = ['{:>12s}'.format('dip_st' + str(i) + '.' + dstr[j])
            for i in range(nst) for j in range(crd_dim)]
    tfile_names[tkeys[3]] = 'dipole'
    dump_header[tkeys[3]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[3]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(nst*crd_dim)) + '\n')

    # transition dipoles
    arr1 = ['  td_s' + str(j) + '.s' + str(i) + '.' + dstr[k]
            for i in range(nst) for j in range(i) for k in range(crd_dim)]
    ncol = int(nst*(nst-1)*crd_dim/2+1)
    tfile_names[tkeys[4]] = 'tr_dipole'
    dump_header[tkeys[4]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[4]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(1, ncol)) + '\n')

    # second moments
    arr1 = ['   sec_s' + str(i) + '.' + dstr[j] + dstr[j]
            for i in range(nst) for j in range(crd_dim)]
    tfile_names[tkeys[5]] = 'sec_mom'
    dump_header[tkeys[5]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[5]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(nst*crd_dim)) + '\n')

    # atomic populations
    arr1 = ['    st' + str(i) + '_a' + str(j+1)
            for i in range(nst) for j in range(ncrd)]
    tfile_names[tkeys[6]] = 'atom_pop'
    dump_header[tkeys[6]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[6]] = ('{:12.4f}' +
                             ''.join('{:10.5f}'
                                     for i in range(nst*ncrd)) + '\n')

    # ----------------- dump formats (bundle files) -----------------

    # adiabatic state populations
    arr1 = ['     state.' + str(i) for i in range(nst)]
    bfile_names[bkeys[0]] = 'n.dat'
    dump_header[bkeys[0]] = ('Time'.rjust(acc1) + ''.join(arr1) +
                             'Norm'.rjust(acc1) + '\n')
    dump_format[bkeys[0]] = ('{:12.4f}' +
                             ''.join('{:12.6f}' for i in range(nst)) +
                             '{:12.6f}\n')

    # the bundle energy
    arr1 = ('   potential(QM)', '     kinetic(QM)', '       total(QM)',
            '  potential(Cl.)', '    kinetic(Cl.)', '      total(Cl.)')
    bfile_names[bkeys[1]] = 'e.dat'
    dump_header[bkeys[1]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[bkeys[1]] = ('{:12.4f}' +
                             ''.join('{:16.10f}' for i in range(6)) + '\n')

    # the spawn log
    lenst = 7
    arr1 = ('time(entry)'.rjust(acc1), 'time(spawn)'.rjust(acc1),
            'time(exit)'.rjust(acc1), 'parent'.rjust(lenst),
            'state'.rjust(lenst), 'child'.rjust(lenst), 'state'.rjust(lenst),
            'ke(parent)'.rjust(acc1), 'ke(child)'.rjust(acc1),
            'pot(parent)'.rjust(acc1), 'pot(child)'.rjust(acc1),
            'total(parent)'.rjust(acc2), 'total(child)'.rjust(acc2))
    bfile_names[bkeys[2]] = 'spawn.dat'
    dump_header[bkeys[2]] = ''.join(arr1) + '\n'
    dump_format[bkeys[2]] = ('{:12.4f}{:12.4f}{:12.4f}{:7d}{:7d}{:7d}{:7d}' +
                             '{:12.8f}{:12.8f}{:12.8f}{:12.8f}' +
                             '{:16.8f}{:16.8f}\n')

    bfile_names[bkeys[3]] = 's.dat'
    bfile_names[bkeys[4]] = 'sdot.dat'
    bfile_names[bkeys[5]] = 'h.dat'
    bfile_names[bkeys[6]] = 'heff.dat'
    bfile_names[bkeys[7]] = 't_ovrlp.dat'

    # autocorrelation function
    arr1 = ('      Re a(t)','         Im a(t)','         abs a(t)')
    bfile_names[bkeys[8]] = 'auto.dat'
    dump_header[bkeys[8]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[bkeys[8]] = ('{:12.4f}' +
                             ''.join('{:16.10f}' for i in range(3)) + '\n')

    # ------------------------- log file formats --------------------------
    with open(home_path+'/fms.log', 'w') as logfile:
        log_str = (' ---------------------------------------------------\n' +
                   ' ab initio multiple spawning dynamics\n' +
                   ' ---------------------------------------------------\n' +
                   '\n' +
                   ' *************\n' +
                   ' input summary\n' +
                   ' *************\n' +
                   '\n' +
                   ' file paths\n' +
                   ' ---------------------------------------\n' +
                   ' home_path   = ' + str(home_path) + '\n' +
                   ' scr_path    = ' + str(scr_path) + '\n')
        logfile.write(log_str)

        log_str = ('\n fms simulation keywords\n' +
                   ' ----------------------------------------\n')
        for group in glbl.input_groups.items():
            logfile.write(" ** "+str(group)+" ** ")
            for k, v in glbl.input_groups[group]:
                log_str += ' {:20s} = {:20s}\n'.format(str(k), str(v))
            logfile.write(log_str+'\n')

        log_str = ('\n ***********\n' +
                   ' propagation\n' +
                   ' ***********\n\n')
        logfile.write(log_str)

    log_format['general']     = '   ** {:60s} **\n'
    log_format['warning']     = ' ** WARNING\: {:100s} **\n'
    log_format['string']      = ' {:160s}\n'
    log_format['t_step']      = ' > time: {:14.4f} step:{:8.4f} [{:4d} trajectories]\n'
    log_format['coupled']     = '  -- in coupling regime -> timestep reduced to {:8.4f}\n'
    log_format['new_step']    = '   -- {:50s} / re-trying with new time step: {:8.4f}\n'
    log_format['spawn_start'] = ('  -- spawning: trajectory {:4d}, ' +
                                 'state {:2d} --> state {:2d}\n' +
                                 'time'.rjust(14) + 'coup'.rjust(10) +
                                 'overlap'.rjust(10) + '   spawn\n')
    log_format['spawn_step']  = '{:14.4f}{:10.4f}{:10.4f}   {:40s}\n'
    log_format['spawn_back']  = '      back propagating:  {:12.2f}\n'
    log_format['spawn_bad_step']= '       --> could not spawn: {:40s}\n'
    log_format['spawn_success'] = ' - spawn successful, new trajectory created at {:14.4f}\n'
    log_format['spawn_failure'] = ' - spawn failed, cannot create new trajectory\n'
    log_format['complete']      = ' ------- simulation completed --------\n'
    #log_format['error']         = '\nError: {}\n ------- simulation terminated  --------\n'
    log_format['error']         = '\n{}\n ------- simulation terminated  --------\n'
    log_format['timings' ]      = '{}'

    print_level['general']        = 5
    pirnt_level['warning']        = 0
    print_level['string']         = 5
    print_level['t_step']         = 0
    print_level['coupled']        = 3
    print_level['new_step']       = 3
    print_level['spawn_start']    = 1
    print_level['spawn_step']     = 1
    print_level['spawn_back']     = 2
    print_level['spawn_bad_step'] = 2
    print_level['spawn_success']  = 1
    print_level['spawn_failure']  = 1
    print_level['complete']       = 0
    print_level['error']          = 0
    print_level['timings']        = 0


def print_traj_row(label, fkey, data):
    """Appends a row of data, formatted by entry 'fkey' in formats to
    file 'filename'."""
    global scr_path, tkeys, tfile_names, dump_header, dump_format
    filename = scr_path + '/' + tfile_names[tkeys[fkey]] + '.' + str(label)

    if not os.path.isfile(filename):
        with open(filename, 'x') as outfile:
            outfile.write(dump_header[tkeys[fkey]])
            outfile.write(dump_format[tkeys[fkey]].format(*data))
    else:
        with open(filename, 'a') as outfile:
            outfile.write(dump_format[tkeys[fkey]].format(*data))

def update_logs(bundle):
    """Determines if it is appropriate to update the traj/bundle logs.

    In general, I assume it's desirable that these time steps are
    relatively constant -- regardless of what the propagator requires
    to do accurate integration.
    """
    dt    = glbl.propagate['default_time_step']
    mod_t = bundle.time % dt

    # this. is. ugly.
    return (mod_t < 0.0001*dt or mod_t > 0.999*dt)


def print_bund_row(fkey, data):
    """Appends a row of data, formatted by entry 'fkey' in formats to
    file 'filename'."""
    global scr_path, bkeys, bfile_names, dump_header, dump_format
    filename = scr_path + '/' + bfile_names[bkeys[fkey]]

    if glbl.mpi['rank'] !=0:
        return

    if not os.path.isfile(filename):
        with open(filename, 'x') as outfile:
            outfile.write(dump_header[bkeys[fkey]])
            outfile.write(dump_format[bkeys[fkey]].format(*data))
    else:
        with open(filename, 'a') as outfile:
            outfile.write(dump_format[bkeys[fkey]].format(*data))


def print_bund_mat(time, fname, mat):
    """Prints a matrix to file with a time label."""
    global scr_path
    filename = scr_path + '/' + fname

    with open(filename, 'a') as outfile:
        outfile.write('{:9.2f}\n'.format(time))
        outfile.write(np.array2string(mat)+'\n')


def print_fms_logfile(otype, data):
    """Prints a string to the log file."""
    global log_format, print_level

    if glbl.mpi['rank'] != 0:
        return

    if otype not in log_format:
        print('CANNOT WRITE otype=' + str(otype) + '\n')
    elif glbl.printing['print_level'] >= print_level[otype]:
        filename = home_path + '/fms.log'
        with open(filename, 'a') as logfile:
            logfile.write(log_format[otype].format(*data))


#----------------------------------------------------------------------------
#
# Read geometry.dat and hessian.dat files
#
#----------------------------------------------------------------------------
def read_geometry(geom_file):
    """Reads position and momenta from an xyz file"""
    geoms = []
    moms  = []

    with open(geom_file, 'r') as gfile:
        gm_file = gfile.readlines()

    # parse file for number of atoms/dof, dimension of coordinates
    # and number of geometries
    ncrd    = int(gm_file[0].strip()[0])
    crd_dim = int(0.5*(len(gm_file[2].strip().split()) - 1))
    ngeoms  = len(gm_file)/(ncrd+2)
    
    # read in the atom/crd labels -- assumes atoms are same for each
    # geometry in the list    
    labels  = [gm_file[j].strip().split()[0] for j in range(2,ncrd)] 

    # loop over geoms, load positions and momenta into arrays
    for i in range(ngeoms):
        geom = []
        mom  = []       

        # delete first and comment lines
        del gm_file[0:1]
        for j in range(ncrd):
            line = gm_file[j].strip().split()
            geom.extend([line[k] for k in range(1,crd_dim+1)])
            mom.extend([line[k] for k in range(crd_dim+1,2*crd_dim+1)])
            del gm_file[0]

        geoms.append(geom)
        moms.append(mom)
        
    return labels,geoms,moms

#
def read_hessian():
    """Reads the non-mass-weighted Hessian matrix from hessian.dat."""
    global home_path

    hessian = np.loadtxt(home_path + '/hessian.dat', dtype=float)
    return hessian


#----------------------------------------------------------------------------
#
# FMS summary output file
#
#----------------------------------------------------------------------------
def cleanup(exception=None):
    """Cleans up the FMS log file."""
    global home_path, scr_path

    if glbl.mpi['rank'] == 0:
        # simulation ended
        if exception is None:
            print_fms_logfile('complete', [])
        else:
            print_fms_logfile('error', [rm_timer(exception)])
            for timer in timings.active_stack[:0:-1]:
                timings.stop(timer.name)

        # print timing information
        timings.stop('global', cumulative=True)
        t_table = timings.print_timings()
        print_fms_logfile('timings', [t_table])

        # move trajectory summary files to an output directory in the home area
        odir = home_path + '/output'
        if os.path.exists(odir):
            shutil.rmtree(odir)
        os.makedirs(odir)

        # move trajectory files
        for key, fname in tfile_names.items():
            for tfile in glob.glob(scr_path + '/' + fname + '.*'):
                if not os.path.isdir(tfile):
                    shutil.move(tfile, odir)

        # move bundle files
        for key, fname in bfile_names.items():
            try:
                shutil.move(scr_path + '/' + fname, odir)
            except IOError:
                pass

        # move chkpt file
        try:
            shutil.move(scr_path + '/last_step.dat', odir)
        except IOError:
            pass


def rm_timer(exc):
    """Removes the timer lines from an Exception traceback."""
    tb = exc.split('\n')
    regex = re.compile('.*timings\.py.*in (hooked|_run_func)')
    i = 0
    while i < len(tb):
        if re.match(regex, tb[i]):
            tb = tb[:i] + tb[i+2:]
        else:
            i += 1
    return '\n'.join(tb)
