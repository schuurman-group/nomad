"""
Routines for reading input files and writing log files.
"""
import os
import shutil
import numpy as np
import src.parse.glbl as glbl

np.set_printoptions(threshold = np.inf)
tkeys       = ['traj_dump', 'ener_dump', 'coup_dump', 'dipole_dump',
               'secm_dump', 'tran_dump', 'apop_dump', 'grad_dump']
bkeys       = ['pop_dump', 'bener_dump', 'spawn_dump',
               'snuc_mat','s_mat', 'sdot_mat', 'h_mat', 'heff_mat','t_ovrlp','auto.dat']
dump_header = dict()
dump_format = dict()
tfile_names = dict()
bfile_names = dict()

#
# This is only called by "extract" -- these files not created/written due during the
# dynamics run
#
def generate_data_formats():
    """Initialized all the output format descriptors."""
    global dump_header, dump_format, tfile_names, bfile_names

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

    # bundle matrices
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

def print_traj_row(label, fkey, data):
    """Appends a row of data, formatted by entry 'fkey' in formats to
    file 'filename'."""
    filename = glbl.scr_path + '/' + tfile_names[tkeys[fkey]] + '.' + str(label)

    if not os.path.isfile(filename):
        with open(filename, 'x') as outfile:
            outfile.write(dump_header[tkeys[fkey]])
            outfile.write(dump_format[tkeys[fkey]].format(*data))
    else:
        with open(filename, 'a') as outfile:
            outfile.write(dump_format[tkeys[fkey]].format(*data))


def print_bund_row(fkey, data):
    """Appends a row of data, formatted by entry 'fkey' in formats to
    file 'filename'."""
    filename = glbl.scr_path + '/' + bfile_names[bkeys[fkey]]

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
    filename = glbl.scr_path + '/' + fname

    with open(filename, 'a') as outfile:
        outfile.write('{:9.2f}\n'.format(time))
        outfile.write(np.array2string(mat,
                      formatter={'complex_kind':lambda x: '{: 15.8e}'.format(x)})+'\n')


#################################################################################
#################################################################################
#
# THIS SHOUDL BE DELETED: nomad should not move files around....
#
def copy_output():
    """Copies output files to current working directory."""
    # move trajectory summary files to an output directory in the home area
    odir = glbl.home_path + '/output'
    if os.path.exists(odir):
        shutil.rmtree(odir)
    os.makedirs(odir)

    # move trajectory files
    for key, fname in tfile_names.items():
        for tfile in glob.glob(glbl.scr_path + '/' + fname + '.*'):
            if not os.path.isdir(tfile):
                shutil.move(tfile, odir)

    # move bundle files
    for key, fname in bfile_names.items():
        try:
            shutil.move(glbl.scr_path + '/' + fname, odir)
        except IOError:
            pass

    # move chkpt file
    try:
        shutil.move(glbl.scr_path + '/ckhpt.hdf5', odir)
    except IOError:
        pass

    return

