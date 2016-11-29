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
import src.dynamics.atom_lib as atom_lib

# Make sure that we print entire arrays
np.set_printoptions(threshold = np.inf)

home_path   = ''
scr_path    = ''
tkeys       = ['traj_dump', 'ener_dump', 'coup_dump', 'dipole_dump',
               'secm_dump', 'tran_dump', 'apop_dump', 'grad_dump']
bkeys       = ['pop_dump', 'bener_dump', 'spawn_dump',
               's_mat', 'sdot_mat', 'h_mat', 'heff_mat','auto.dat']
dump_header = dict()
dump_format = dict()
log_format  = dict()
tfile_names = dict()
bfile_names = dict()
print_level = dict()


def read_input_files():
    """Reads the fms.input files.

    This file contains variables related to the running of the
    dynamics simulation.
    """
    global scr_path, home_path

    # save the name of directory where program is called from
    home_path = os.getcwd()

    # set a sensible default for scr_path
    scr_path = os.environ['TMPDIR']
    if os.path.exists(scr_path):
        shutil.rmtree(scr_path)
        os.makedirs(scr_path)

    # Read fms.input. This contains general simulation variables
    kwords = read_namelist('fms.input')
    for k, v in kwords.items():
        if k in glbl.fms:
            glbl.fms[k] = v
        else:
            print('Variable ' + str(k) +
                  ' in fms.input unrecognized. Ignoring...')

    # Read pes.input. This contains interface-specific user options. Get what
    #  interface we're using via glbl.fms['interface'], and populate the
    #  corresponding dictionary of keywords from glbl module
    # Clumsy. Not even sure this is the best way to do this (need to segregate
    # variables in different dictionaries. fix this later
    kwords = read_namelist('pes.input')
    if glbl.fms['interface'] == 'columbus':
        for k, v in kwords.items():
            if k in glbl.columbus:
                glbl.columbus[k] = v
            else:
                print('Variable '  + str(k) +
                      ' in fms.input unrecognized. Ignoring...')
    elif glbl.fms['interface'] == 'vibronic':
        for k, v in kwords.items():
            if k in glbl.vibronic:
                glbl.vibronic[k] = v
            else:
                print('Variable ' + str(k) +
                      ' in fms.input unrecognized. Ignoring...')
    elif glbl.fms['interface'] == 'boson_model_diabatic':
        for k, v in kwords.items():
            if k in glbl.boson:
                glbl.boson[k] = v
    else:
        print('Interface: ' + str(glbl.fms['interface']) + ' not recognized.')


def read_namelist(filename):
    """Reads a namelist style input, returns results in dictionary."""
    kwords = dict()

    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as infile:
            for line in infile:
                if '=' in line:
                    line = line.rstrip('\r\n')
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        kwords[key] = float(value)
                        if kwords[key].is_integer():
                            kwords[key] = int(value)
                    except ValueError:
                        pass

                    if key not in kwords:
                        kwords[key] = value

    return kwords


def init_fms_output():
    """Initialized all the output format descriptors."""
    global home_path, scr_path, log_format, tkeys, bkeys
    global dump_header, dump_format, tfile_names, bfile_names, print_level

    nump = int(glbl.fms['num_particles'])
    numd = int(glbl.fms['dim_particles'])
    nums = int(glbl.fms['n_states'])
    dstr = ('x', 'y', 'z')
    acc1 = 12
    acc2 = 16

    # ----------------- dump formats (trajectory files) -----------------
    # trajectory output
    arr1 = ['{:>12s}'.format('pos' + str(i+1) + '.' + dstr[x])
            for i in range(nump) for x in range(numd)]
    arr2 = ['{:>12s}'.format('mom' + str(i+1) + '.' + dstr[x])
            for i in range(nump) for x in range(numd)]
    tfile_names[tkeys[0]] = 'trajectory'
    dump_header[tkeys[0]] = ('Time'.rjust(acc1) + ''.join(arr1) +
                             ''.join(arr2) + 'Phase'.rjust(acc1) +
                             'Re[Amp]'.rjust(acc1) + 'Im[Amp]'.rjust(acc1) +
                             'Norm[Amp]'.rjust(acc1) + 'State'.rjust(acc1) +
                             '\n')
    dump_format[tkeys[0]] = ('{:12.4f}'+
                             ''.join('{:12.6f}' for i in range(3*nump*numd)) +
                             '\n')

    # potential energy
    arr1 = ['{:>16s}'.format('potential.' + str(i)) for i in range(nums)]
    tfile_names[tkeys[1]] = 'poten'
    dump_header[tkeys[1]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[1]] = ('{:12.4f}' +
                             ''.join('{:16.10f}' for i in range(nums)) + '\n')

    # coupling
    arr1 = ['{:>12s}'.format('coupling.' + str(i)) for i in range(nums)]
    arr2 = ['{:>12s}'.format('c * v .' + str(i)) for i in range(nums)]
    tfile_names[tkeys[2]] = 'coupling'
    dump_header[tkeys[2]] = ('Time'.rjust(acc1) + ''.join(arr1) +
                             ''.join(arr2) + '\n')
    dump_format[tkeys[2]] = ('{:12.4f}' +
                             ''.join('{:12.5f}' for i in range(2*nums)) + '\n')

    # permanent dipoles
    arr1 = ['{:>12s}'.format('dip_st' + str(i) + '.' + dstr[j])
            for i in range(nums) for j in range(numd)]
    tfile_names[tkeys[3]] = 'dipole'
    dump_header[tkeys[3]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[3]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(nums*numd)) + '\n')

    # transition dipoles
    arr1 = ['  td_s' + str(j) + '.s' + str(i) + '.' + dstr[k]
            for i in range(nums) for j in range(i) for k in range(numd)]
    ncol = int(nums*(nums-1)*numd/2+1)
    tfile_names[tkeys[4]] = 'tr_dipole'
    dump_header[tkeys[4]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[4]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(1, ncol)) + '\n')

    # second moments
    arr1 = ['   sec_s' + str(i) + '.' + dstr[j] + dstr[j]
            for i in range(nums) for j in range(numd)]
    tfile_names[tkeys[5]] = 'sec_mom'
    dump_header[tkeys[5]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[5]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(nums*numd)) + '\n')

    # atomic populations
    arr1 = ['    st' + str(i) + '_p' + str(j+1)
            for i in range(nums) for j in range(nump)]
    tfile_names[tkeys[6]] = 'atom_pop'
    dump_header[tkeys[6]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[6]] = ('{:12.4f}' +
                             ''.join('{:10.5f}'
                                     for i in range(nums*nump)) + '\n')

    # gradients
    arr1 = ['  grad_part' + str(i+1) + '.' + dstr[j]
            for i in range(nump) for j in range(numd)]
    tfile_names[tkeys[7]] = 'gradient'
    dump_header[tkeys[7]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[7]] = ('{0:>12.4f}' +
                             ''.join('{' + str(i) + ':14.8f}'
                                     for i in range(1, nump*numd+1)) + '\n')

    # ----------------- dump formats (bundle files) -----------------

    # adiabatic state populations
    arr1 = ['     state.' + str(i) for i in range(nums)]
    bfile_names[bkeys[0]] = 'n.dat'
    dump_header[bkeys[0]] = ('Time'.rjust(acc1) + ''.join(arr1) +
                             'Norm'.rjust(acc1) + '\n')
    dump_format[bkeys[0]] = ('{:12.4f}' +
                             ''.join('{:12.6f}' for i in range(nums)) +
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

    # autocorrelation function
    arr1 = ('      Re a(t)','         Im a(t)','         abs a(t)')
    bfile_names[bkeys[7]] = 'auto.dat'
    dump_header[bkeys[7]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[bkeys[7]] = ('{:12.4f}' +
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
        for k, v in glbl.fms.items():
            log_str += ' {:20s} = {:20s}\n'.format(str(k), str(v))
        logfile.write(log_str)

        if glbl.fms['interface'] == 'columbus':
            out_key = glbl.columbus
        elif glbl.fms['interface'] == 'vibronic':
            out_key = glbl.vibronic
        elif glbl.fms['interface'] == 'boson_model_diabatic':
            out_key = glbl.boson
        else:
            out_key = dict()

        log_str = '\n ' + str(glbl.fms['interface']) + ' simulation keywords\n'
        log_str += ' ----------------------------------------\n'
        for k, v in out_key.items():
            log_str += ' {:20s} = {:20s}\n'.format(str(k), str(v))
        logfile.write(log_str)

        log_str = ('\n ***********\n' +
                   ' propagation\n' +
                   ' ***********\n\n')
        logfile.write(log_str)

    log_format['general']     = '   ** {:60s} **\n'
    log_format['string']      = ' {:160s}\n'
    log_format['t_step']      = ' > time: {:14.4f} step:{:8.4f} [{:4d} trajectories]\n'
    log_format['coupled']     = '  -- in coupling regime -> timestep reduced to {:8.4f}\n'
    log_format['new_step']    = '   -- error: {:50s} / re-trying with new time step: {:8.4f}\n'
    log_format['spawn_start'] = ('  -- spawing: trajectory {:4d}, ' +
                                 'state {:2d} --> state {:2d}\n' +
                                 'time'.rjust(14) + 'coup'.rjust(10) +
                                 'overlap'.rjust(10) + '   spawn\n')
    log_format['spawn_step']  = '{:14.4f}{:10.4f}{:10.4f}   {:40s}\n'
    log_format['spawn_back']  = '      back propagating:  {:12.2f}\n'
    log_format['spawn_bad_step']= '       --> could not spawn: {:40s}\n'
    log_format['spawn_success'] = ' - spawn successful, new trajectory created at {:14.4f}\n'
    log_format['spawn_failure'] = ' - spawn failed, cannot create new trajectory\n'
    log_format['complete']      = ' ------- simulation completed --------\n'
    log_format['timings' ]      = '{}'

    print_level['general']        = 5
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
    print_level['timings']        = 0


def print_traj_row(tid, fkey, data):
    """Appends a row of data, formatted by entry 'fkey' in formats to
    file 'filename'."""
    global scr_path, tkeys, tfile_names, dump_header, dump_format
    filename = scr_path + '/' + tfile_names[tkeys[fkey]] + '.' + str(tid)

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
    dt    = glbl.fms['default_time_step']
    mod_t = bundle.time % dt

    return mod_t < 0.1*dt or mod_t > 0.9*dt


def print_bund_row(fkey, data):
    """Appends a row of data, formatted by entry 'fkey' in formats to
    file 'filename'."""
    global scr_path, bkeys, bfile_names, dump_header, dump_format
    filename = scr_path + '/' + bfile_names[bkeys[fkey]]

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

    if otype not in log_format:
        print('CANNOT WRITE otype=' + str(otype) + '\n')
    elif glbl.fms['print_level'] >= print_level[otype]:
        filename = home_path + '/fms.log'
        with open(filename, 'a') as logfile:
            logfile.write(log_format[otype].format(*data))


#----------------------------------------------------------------------------
#
# Read geometry.dat and hessian.dat files
#
#----------------------------------------------------------------------------
def read_geometry():
    """Reads position and momenta from geometry.dat."""
    global home_path
    amp_data   = []
    geom_data  = []
    mom_data   = []
    width_data = []
    label_data = []
    mass_data  = []

    with open(home_path + '/geometry.dat', 'r', encoding='utf-8') as gfile:
        gm_file = gfile.readlines()

    not_done = True
    lcnt = -1
    while not_done:
        # comment line -- if keyword "amplitude" is present, set amplitude
        lcnt += 1
        line = [x.strip().lower() for x in re.split('\W+', gm_file[lcnt])]
        if 'amplitude' in line:
            ind = line.index('amplitude')
            amp_data.append(complex(float(line[ind+1]), float(line[ind+2])))
        else:
            amp_data.append(complex(1.,0.))

        # number of atoms/coordinates
        lcnt += 1
        nq = int(gm_file[lcnt])

        # read in geometry
        for i in range(nq):
            lcnt += 1
            geom_data.append(gm_file[lcnt].rstrip().split()[1:])
            crd_dim = len(gm_file[lcnt].rstrip().split()[1:])
            label_data.append([gm_file[lcnt].rstrip().split()[0] 
                                                   for i in range(crd_dim))

        # read in momenta
        for i in range(nq):
            lcnt += 1
            mom_data.append(gm_file[lcnt].rstrip().split())

        # read in widths, if present
        if (lcnt+1) < len(gm_file) and 'alpha' in gm_file[lcnt+1]:
            for i in range(nq):
                lcnt += 1
                width_data.append(float(gm_file[lcnt].rstrip().split()[1:]))
        else:
            labels = label_data[-nq * crd_dim]
            for lbl in labels:
                if atom_lib.valid_atom(lbl):
                    adata = atom_data(lbl)
                    width_data.append(adata[0])
                else:
                    width_data.append(0)

        # read in masses, if present
        if (lcnt+1) < len(gm_file) and 'mass' in gm_file[lcnt+1]:
            for i in range(nq):
                lcnt += 1
                mass_data.append(float(gm_file[lcnt].rstrip().split()[1:]))
        else:
            labels = label_data[-nq * crd_dim]
            for lbl in labels:
                if atom_lib.valid_atom(lbl):
                    adata = atom_data(lbl)
                    mass_data.append(adata[1])
                else:
                    mass_data.append(1.)

        # check if we've reached the end of the file
        if (lcnt+1) == len(gm_file):
            not_done = False

    return (crd_dim, amp_data, label_data, 
            geom_data, mom_data, width_data, mass_data)

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
def cleanup():
    """Cleans up the FMS log file."""
    global home_path, scr_path

    # simulation complete
    print_fms_logfile('complete', [])

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
