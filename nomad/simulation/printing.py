"""
Routines for reading input files and writing log files.
"""
import os
import shutil
import numpy as np
import nomad.simulation.glbl as glbl
np.set_printoptions(threshold = np.inf)


tkeys       = ['traj', 'poten', 'grad', 'coup', 'hessian',
               'dipole', 'tr_dipole', 'secm', 'apop']
bkeys       = ['pop', 'energy', 'auto']
dump_header = dict()
dump_format = dict()
tfile_names = dict()
bfile_names = dict()


def generate_data_formats(ncrd, nst):
    """Initialized all the output format descriptors."""
    global dump_header, dump_format, tfile_names, bfile_names

    ncart = 3         # assumes expectation values of transition/permanent dipoles in
                      # cartesian coordinates
    natm  = max(1,int(ncrd / ncart)) # dirty -- in case we have small number of n.modes
    dstr  = ('x', 'y', 'z')
    acc1  = 12
    acc2  = 16

    # ******************* dump formats *******************************

    # ----------------- trajectory data --------------------------------
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
    tfile_names[tkeys[2]] = 'grad'
    dump_header[tkeys[2]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[2]] = ('{0:>12.4f}' +
                             ''.join('{' + str(i) + ':14.8f}'
                                     for i in range(1, ncrd+1)) + '\n')

    # coupling
    arr1 = ['{:>12s}'.format('coupling.' + str(i)) for i in range(nst)]
    arr2 = ['{:>12s}'.format('c * v .' + str(i)) for i in range(nst)]
    tfile_names[tkeys[3]] = 'coup'
    dump_header[tkeys[3]] = ('Time'.rjust(acc1) + ''.join(arr1) +
                             ''.join(arr2) + '\n')
    dump_format[tkeys[3]] = ('{:12.4f}' +
                             ''.join('{:12.5f}' for i in range(2*nst)) + '\n')


    # ---------------------- interface data --------------------------------
    # permanent dipoles
    arr1 = ['{:>12s}'.format('dip_st' + str(i) + '.' + dstr[j])
            for i in range(nst) for j in range(ncart)]
    tfile_names[tkeys[5]] = 'dipole'
    dump_header[tkeys[5]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[5]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(nst*ncart)) + '\n')
    # transition dipoles
    arr1 = ['  td_s' + str(j) + '.s' + str(i) + '.' + dstr[k]
            for i in range(nst) for j in range(i) for k in range(ncart)]
    ncol = int(nst*(nst-1)*ncart/2+1)
    tfile_names[tkeys[6]] = 'tr_dipole'
    dump_header[tkeys[6]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[6]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(1, ncol)) + '\n')

    # second moments
    arr1 = ['   sec_s' + str(i) + '.' + dstr[j] + dstr[j]
            for i in range(nst) for j in range(ncart)]
    tfile_names[tkeys[7]] = 'sec_mom'
    dump_header[tkeys[7]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[7]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(nst*ncart)) + '\n')

    # atomic populations
    arr1 = ['    st' + str(i) + '_a' + str(j+1)
            for i in range(nst) for j in range(natm)]
    tfile_names[tkeys[8]] = 'atom_pop'
    dump_header[tkeys[8]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[8]] = ('{:12.4f}' +
                             ''.join('{:10.5f}'
                                     for i in range(nst*natm)) + '\n')

    # ----------------- dump formats (wavefunction files) -----------------

    # adiabatic state populations
    arr1 = ['     state.' + str(i) for i in range(nst)]
    bfile_names['pop'] = 'n.dat'
    dump_header['pop'] = ('Time'.rjust(acc1) + ''.join(arr1) +
                             'Norm'.rjust(acc1) + '\n')
    dump_format['pop']  = ('{:12.4f}' +
                             ''.join('{:12.6f}' for i in range(nst)) +
                             '{:12.6f}\n')

    # the bundle energy
    arr1 = ('   potential(QM)', '     kinetic(QM)', '       total(QM)',
            '  potential(Cl.)', '    kinetic(Cl.)', '      total(Cl.)')
    bfile_names['energy'] = 'e.dat'
    dump_header['energy'] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format['energy'] = ('{:12.4f}' +
                             ''.join('{:16.10f}' for i in range(6)) + '\n')

    # autocorrelation function
    arr1 = ('      Re a(t)','         Im a(t)','         abs a(t)')
    bfile_names['auto'] = 'auto.dat'
    dump_header['auto'] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format['auto'] = ('{:12.4f}' +
                             ''.join('{:16.10f}' for i in range(3)) + '\n')

    # trajectory matrices
    tfile_names['hessian']   = 'hessian.dat'

    # bundle matrices
    bfile_names['s']         = 's.dat'
    bfile_names['sdot']      = 'sdot.dat'
    bfile_names['h']         = 'h.dat'
    bfile_names['heff']      = 'heff.dat'
    bfile_names['t_overlap'] = 't_overlap.dat'


def print_traj_row(label, key, data):
    """Appends a row of data, formatted by entry 'fkey' in formats to
    file 'filename'."""
    filename = tfile_names[key] + '.' + str(label)

    if not os.path.isfile(filename):
        with open(filename, 'x') as outfile:
            outfile.write(dump_header[key])
            outfile.write(dump_format[key].format(*data))
    else:
        with open(filename, 'a') as outfile:
            outfile.write(dump_format[key].format(*data))


def print_traj_mat(time, key, mat):
    """Prints a matrix to file with a time label."""
    filename = tfile_names[key]

    with open(filename, 'a') as outfile:
        outfile.write('{:9.2f}\n'.format(time))
        outfile.write(np.array2string(mat,
                      formatter={'complex_kind':lambda x: '{: 15.8e}'.format(x)})+'\n')


def print_wfn_row(key, data):
    """Appends a row of data, formatted by entry 'fkey' in formats to
    file 'filename'."""
    filename = bfile_names[key]

    if not os.path.isfile(filename):
        with open(filename, 'x') as outfile:
            outfile.write(dump_header[key])
            outfile.write(dump_format[key].format(*data))
    else:
        with open(filename, 'a') as outfile:
            outfile.write(dump_format[key].format(*data))


def print_wfn_mat(time, key, mat):
    """Prints a matrix to file with a time label."""
    filename = bfile_names[key]

    with open(filename, 'a') as outfile:
        outfile.write('{:9.2f}\n'.format(time))
        outfile.write(np.array2string(mat,
                      formatter={'complex_kind':lambda x: '{: 15.8e}'.format(x)})+'\n')


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
