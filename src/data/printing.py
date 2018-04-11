"""
Routines for reading input files and writing log files.
"""
import sys
import os
import re
import glob
import ast
import shutil
import traceback
import numpy as np
import src.dynamics.timings as timings
import src.fmsio.glbl as glbl
import src.basis.atom_lib as atom_lib

# Make sure that we print entire arrays
np.set_printoptions(threshold = np.inf)

tkeys       = ['traj_dump', 'ener_dump', 'coup_dump', 'dipole_dump',
               'secm_dump', 'tran_dump', 'apop_dump', 'grad_dump']
bkeys       = ['pop_dump', 'bener_dump', 'spawn_dump',
               'snuc_mat','s_mat', 'sdot_mat', 'h_mat', 'heff_mat','t_ovrlp','auto.dat']
dump_header = dict()
dump_format = dict()
log_format  = dict()
tfile_names = dict()
bfile_names = dict()
print_level = dict()

def init_fms_output():
    """Initialized all the output format descriptors."""
    global log_format, dump_header, dump_format, tfile_names, bfile_names, print_level

    ncart = 3         # assumes expectation values of transition/permanent dipoles in
                      # cartesian coordinates
    ncrd  = len(glbl.nuclear_basis['geometries'][0])
    natm  = max(1,int(ncrd / ncart)) # dirty -- in case we have small number of n.modes
    nst   = glbl.propagate['n_states']
    dstr  = ('x', 'y', 'z')
    acc1  = 12
    acc2  = 16

    # ----------------- dump formats (trajectory files) -----------------
    # trajectory output
    arr1 = ['{:>12s}'.format('    x' + str(i+1)) for i in range(ncrd)]
    arr2 = ['{:>12s}'.format('    p' + str(i+1)) for i in range(ncrd)]
    tfile_names[tkeys[0]] = 'trajectory'
    dump_header[tkeys[0]] = ('Time'.rjust(acc1) + ''.join(arr1) +
                             ''.join(arr2) + 'Phase'.rjust(acc1) +
                             'Re[Amp]'.rjust(acc1) + 'Im[Amp]'.rjust(acc1) +
                             'Norm[Amp]'.rjust(acc1) + 'State'.rjust(acc1) +
                             '\n')
    dump_format[tkeys[0]] = ('{:12.4f}'+
                             ''.join('{:12.6f}' for i in range(2*ncrd+5))+
                             '\n')

    # potential energy
    arr1 = ['{:>16s}'.format('potential.' + str(i)) for i in range(nst)]
    tfile_names[tkeys[1]] = 'poten'
    dump_header[tkeys[1]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[1]] = ('{:12.4f}' +
                             ''.join('{:16.10f}' for i in range(nst)) + '\n')

    # gradients
    arr1 = ['            x' + str(i+1) for i in range(ncrd)]
    tfile_names[tkeys[7]] = 'gradient'
    dump_header[tkeys[7]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[7]] = ('{0:>12.4f}' +
                             ''.join('{' + str(i) + ':14.8f}'
                                     for i in range(1, ncrd+1)) + '\n')

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
            for i in range(nst) for j in range(ncart)]
    tfile_names[tkeys[3]] = 'dipole'
    dump_header[tkeys[3]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[3]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(nst*ncart)) + '\n')
    # transition dipoles
    arr1 = ['  td_s' + str(j) + '.s' + str(i) + '.' + dstr[k]
            for i in range(nst) for j in range(i) for k in range(ncart)]
    ncol = int(nst*(nst-1)*ncart/2+1)
    tfile_names[tkeys[4]] = 'tr_dipole'
    dump_header[tkeys[4]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[4]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(1, ncol)) + '\n')

    # second moments
    arr1 = ['   sec_s' + str(i) + '.' + dstr[j] + dstr[j]
            for i in range(nst) for j in range(ncart)]
    tfile_names[tkeys[5]] = 'sec_mom'
    dump_header[tkeys[5]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[5]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(nst*ncart)) + '\n')

    # atomic populations
    arr1 = ['    st' + str(i) + '_a' + str(j+1)
            for i in range(nst) for j in range(natm)]
    tfile_names[tkeys[6]] = 'atom_pop'
    dump_header[tkeys[6]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[6]] = ('{:12.4f}' +
                             ''.join('{:10.5f}'
                                     for i in range(nst*natm)) + '\n')

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
                   ' scr_path    = ' + os.uname()[1] + ':' + scr_path + '\n')
        logfile.write(log_str)

        logfile.write('\n fms simulation keywords\n' +
                   ' ----------------------------------------\n')

        logfile.write("\n ** constants **\n")
        log_str = ''
        for k,v in glbl.constants.items():
            log_str += ' {:20s} = {:20s}\n'.format(str(k), str(v))
        logfile.write(log_str+'\n')

        logfile.write("\n ** global variables **\n")
        log_str = ''
        for k,v in glbl.variables.items():
            log_str += ' {:20s} = {:20s}\n'.format(str(k), str(v))
        logfile.write(log_str+'\n')

        for group,keywords in glbl.input_groups.items():
            logfile.write('\n ** '+str(group)+' **\n')
            log_str = ''
            for k, v in glbl.input_groups[group].items():
                log_str += ' {:20s} = {:20s}\n'.format(str(k), str(v))
            logfile.write(log_str+'\n')

        logfile.write ('\n ***********\n' +
                         ' propagation\n' +
                         ' ***********\n\n')
    log_format['general']        = '   ** {:60s} **\n'
    log_format['warning']     = ' ** WARNING\: {:100s} **\n'

    log_format['string']         = ' {:160s}\n'
    log_format['t_step']         = ' > time: {:14.4f} step:{:8.4f} [{:4d} trajectories]\n'
    log_format['coupled']        = '  -- in coupling regime -> timestep reduced to {:8.4f}\n'
    log_format['new_step']       = '   -- {:50s} / re-trying with new time step: {:8.4f}\n'
    log_format['spawn_start']    = ('  -- spawning: trajectory {:4d}, ' +
                                    'state {:2d} --> state {:2d}\n' +
                                    'time'.rjust(14) + 'coup'.rjust(10) +
                                    'overlap'.rjust(10) + '   spawn\n')
    log_format['spawn_step']     = '{:14.4f}{:10.4f}{:10.4f}   {:40s}\n'
    log_format['spawn_back']     = '      back propagating:  {:12.2f}\n'
    log_format['spawn_bad_step'] = '       --> could not spawn: {:40s}\n'
    log_format['spawn_success']  = ' - spawn successful, new trajectory created at {:14.4f}\n'
    log_format['spawn_failure']  = ' - spawn failed, cannot create new trajectory\n'
    log_format['complete']       = ' ------- simulation completed --------\n'
    log_format['error']          = '\n{}\n ------- simulation terminated  --------\n'
    log_format['timings' ]       = '{}'

    print_level['general']        = 5
    print_level['warning']        = 0
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
    return mod_t < 0.0001*dt or mod_t > 0.999*dt
def print_bund_row(fkey, data):
    """Appends a row of data, formatted by entry 'fkey' in formats to
    file 'filename'."""
    filename = scr_path + '/' + bfile_names[bkeys[fkey]]

    if glbl.mpi['rank'] == 0:
        if not os.path.isfile(filename):
            with open(filename, 'x') as outfile:
                outfile.write(dump_header[bkeys[fkey]])
                outfile.write(dump_format[bkeys[fkey]].format(*data))
        else:
            with open(filename, 'a') as outfile:
                outfile.write(dump_format[bkeys[fkey]].format(*data))


def print_bund_mat(time, fname, mat):
    """Prints a matrix to file with a time label."""
    filename = scr_path + '/' + fname

    with open(filename, 'a') as outfile:
        outfile.write('{:9.2f}\n'.format(time))
        outfile.write(np.array2string(mat,
                      formatter={'complex_kind':lambda x: '{: 15.8e}'.format(x)})+'\n')


def print_fms_logfile(otype, data):
    """Prints a string to the log file."""
    global log_format, print_level

    if glbl.mpi['rank'] == 0:
        if otype not in log_format:
            print('CANNOT WRITE otype=' + str(otype) + '\n')
        elif glbl.printing['print_level'] >= print_level[otype]:
            filename = home_path + '/fms.log'
            with open(filename, 'a') as logfile:
                logfile.write(log_format[otype].format(*data))


def check_boolean(chk_str):
    """Routine to check if string is boolean.

    Accepts 'true','TRUE','True', etc. and if so, return True or False.
    """

    bool_str = str(chk_str).strip().lower()
    if bool_str == 'true':
        return True
    elif bool_str == 'false':
        return False
    else:
        return chk_str
def copy_output():
    """Copies output files to current working directory."""
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

