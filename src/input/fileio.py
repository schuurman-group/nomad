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

home_path   = ''
scr_path    = ''
dump_header = dict()
dump_format = dict()
log_format  = dict()


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
                 not item.startswith('#') and not item.startswith('!')]

    sec_strings = list(glbl.input_groups)

    current_line = 0
    # look for begining of input section
    while current_line < len(fms_input):
        sec_start = [re.search(str('begin '+sec_strings[i]+'-section'),fms_input[current_line])
                     for i in range(len(sec_strings))]
        if all([v is None for v in sec_start]):
            current_line+=1
        else:
            section = next(item for item in sec_start
                           if item is not None).string
            section = section.replace('-section','').replace('begin','').strip()
            current_line = parse_section(fms_input, current_line, section)

    # ensure that input is internally consistent
    validate_input()


def parse_section(kword_array, line_start, section):
    """Reads a namelist style input, returns results in dictionary.

    Set keywords in the appropriate keyword dictionary by parsing
    input array."""

    current_line = line_start + 1
    while (current_line < len(kword_array) and
           re.search('end '+section+'-section',kword_array[current_line]) is None):
        line = kword_array[current_line].rstrip('\r\n')

        # allow for multi-line input
        while ('=' not in kword_array[current_line+1] and
               'end '+section+'-section' not in kword_array[current_line+1]):
            current_line += 1
            line += kword_array[current_line].rstrip('\r\n').strip()

        key,value = line.split('=',1)
        key   = key.strip()
        value = value.strip().lower() # convert to lowercase

        if key not in glbl.input_groups[section].keys():
            if glbl.mpi['rank'] == 0:
                print('Cannot find input parameter: '+key+
                      ' in input section: '+section)
        else:
            # put all variable types into a flat list
            # here we explicitly consider dimension 0,1,2 lists: which
            # is pretty messy.
            valid = True
            if glbl.keyword_type[key][1] == 2:
                try:
                    value = ast.literal_eval(value)
                except Exception:
                    valid = False
                    print('Cannot interpret input as nested array: '+
                            str(ast.literal_eval('\'%s\'' % value)))
                try:
                    varcast = [[glbl.keyword_type[key][0](check_boolean(item))
                               for item in sublist] for sublist in value]
                except ValueError:
                    valid = False
                    print('Cannot read variable: '+str(key)+
                          ' as nested list of '+str(glbl.keyword_type[key][0]))
            elif glbl.keyword_type[key][1] == 1:
                try:
                    value = ast.literal_eval(value)
                except Exception:
                    valid = False
                    print('Cannot interpret input as list: '+
                            str(ast.literal_eval('\'%s\'' % value)))
                try:
                    varcast = [glbl.keyword_type[key][0](check_boolean(item))
                               for item in value]
                except ValueError:
                    valid = False
                    print('Cannot read variable: '+str(key)+
                          ' as list of '+str(glbl.keyword_type[key][0]))
            else:
                try:
                    varcast = glbl.keyword_type[key][0](check_boolean(value))
                except ValueError:
                    valid = False
                    print('Cannot read variable: '+str(key)+
                          ' as a '+str(glbl.keyword_type[key][0]))

            if valid:
                glbl.input_groups[section][key] = varcast

        current_line+=1

    return current_line


def validate_input():
    """Ensures that input values are internally consistent.

    Currently there are multiple ways to set variables in the nuclear
    basis section. The following lines ensure that subsequent usage of the
    entries in glbl is consistent, regardless of how input specified.
    """
    # set the integral definition
    try:
        glbl.integrals =__import__('src.integrals.'+
                                   glbl.propagate['integrals'],fromlist=['a'])
    except ImportError:
        print('Cannot import integrals: src.integrals.' + 
                               str(glbl.propagate['integrals']))

    try:
        glbl.pes = __import__('src.interfaces.' + 
                               glbl.interface['interface'],fromlist=['NA'])
    except ImportError:
        print('Cannot import pes: src.interfaces.'+
                               str(glbl.interface['interface']))

    try:
        glbl.distrib = __import__('src.sampling.'+glbl.sampling['init_sampling'],
                                 fromlist=['NA'])
    except ImportError:
        print('Cannot import sampling: src.sampling.'+
                               str(glbl.sampling['init_sampling']))

    try:
        glbl.spawn = __import__('src.spawn.'+glbl.spawning['spawning'],
                                   fromlist=['a'])
    except ImportError:
        print('Cannot import spawning: src.spawn.'+
                               str(glbl.spawning['spawning']))

    try:
        glbl.integrator = __import__('src.propagators.'+glbl.propagate['propagator'],
                                     fromlist=['a'])
    except ImportError:
        print('Cannot import propagator: src.propagators.'+
                               str(glbl.propagate['propagator']))


    # if geomfile specified, it's contents overwrite variable settings in fms.input
    if os.path.isfile(glbl.nuclear_basis['geomfile']):
        (labels, geoms, moms)            = read_geometry(glbl.nuclear_basis['geomfile'])
        glbl.nuclear_basis['labels']     = labels
        glbl.nuclear_basis['geometries'] = geoms
        glbl.nuclear_basis['momenta']    = moms

    # if hessfile is specified, its contents overwrite variable settings from fms.input
    if os.path.isfile(glbl.nuclear_basis['hessfile']):
        glbl.nuclear_basis['hessian']    = read_hessian(glbl.nuclear_basis['hessfile'])

    # if use_atom_lib, atom_lib values overwrite variables settings from fms.input
    if glbl.nuclear_basis['use_atom_lib']:
        wlst  = []
        mlst  = []
        for i in range(len(labels)):
            (wid, mass, num) = atom_lib.atom_data(labels[i])
            wlst.append(wid)
            mlst.append(mass)
        glbl.nuclear_basis['widths'] = wlst
        glbl.nuclear_basis['masses'] = mlst

    # set mass array here if using vibronic interface
    if glbl.interface['interface'] == 'vibronic':
        n_usr_freq = len(glbl.nuclear_basis['freqs'])
       
        # automatically set the "mass" of the coordinates to be 1/omega
        # the coefficient on p^2 -> 0.5 omega == 0.5 / m. Any user set masses
        # will override the default
        if len(glbl.nuclear_basis['masses']) >= n_usr_freq:
            pass
        else:
            glbl.nuclear_basis['masses'] = [1. for i in range(n_usr_freq)]
#        else all(freq != 0. for freq in glbl.nuclear_basis['freqs']):
#            glbl.nuclear_basis['masses'] = [1./glbl.nuclear_basis['freqs'][i] for i in 
#                                             range(len(n_usr_freq)]

        # set the widths to automatically be 1/2 (i.e. assumes frequency-weighted coordinates.
        # Any user set widths will override the default
        if len(glbl.nuclear_basis['widths']) >= n_usr_freq:
            pass
        else:
            glbl.nuclear_basis['widths'] = [0.5 for i in range(n_usr_freq)]

    # subsequent code will ONLY use the 'init_states' array. If that array hasn't
    # been set, using the value of 'init_state' to create it
    if glbl.sampling['init_state'] != -1:
        glbl.sampling['init_states'] = [glbl.sampling['init_state'] for
                                        i in range(glbl.sampling['n_init_traj'])]

    elif (any(state == -1 for state in glbl.sampling['init_states']) or
          len(glbl.sampling['init_states']) != glbl.sampling['n_init_traj']):
        raise ValueError('Cannot assign state.')

    # set the surface_rep variable depending on the value of the integrals
    # keyword
    if glbl.propagate['integrals'] == 'vibronic_diabatic':
        glbl.variables['surface_rep'] = 'diabatic'
    else:
        glbl.variables['surface_rep'] = 'adiabatic'
 
    # check array lengths
    #ngeom   = len(glbl.nuclear_basis['geometries'])
    #lenarr  = [len(glbl.nuclear_basis['geometries'][i]) for i in range(ngeom)]

#
#
#
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
    ncrd    = int(gm_file[0].strip().split()[0])
    crd_dim = int(0.5*(len(gm_file[2].strip().split()) - 1))
    ngeoms  = int(len(gm_file)/(ncrd+2))

    # read in the atom/crd labels -- assumes atoms are same for each
    # geometry in the list
    labels = []
    for i in range(2,ncrd+2):
        labels.extend([gm_file[i].strip().split()[0] for j in range(crd_dim)])

    # loop over geoms, load positions and momenta into arrays
    for i in range(ngeoms):
        geom = []
        mom  = []

        # delete first and comment lines
        del gm_file[0:2]
        for j in range(ncrd):
            line = gm_file[0].strip().split()
            geom.extend([float(line[k]) for k in range(1,crd_dim+1)])
            mom.extend([float(line[k]) for k in range(crd_dim+1,2*crd_dim+1)])
            del gm_file[0]

        geoms.append(geom)
        moms.append(mom)

    return labels,geoms,moms


def read_hessian(hess_file):
    """Reads the non-mass-weighted Hessian matrix from hessian.dat."""
    return np.loadtxt(str(hess_file), dtype=float)


#----------------------------------------------------------------------------
#
# FMS summary output file
#
#----------------------------------------------------------------------------
def cleanup_end():
    """Cleans up the FMS log file if calculation completed."""
    # simulation ended
    print_fms_logfile('complete', [])

    # print timing information
    timings.stop('global', cumulative=True)
    t_table = timings.print_timings()
    print_fms_logfile('timings', [t_table])

    # copy output files
    copy_output()


def cleanup_exc(etyp, val, tb):
    """Cleans up the FMS log file if an exception occurs."""
    # print exception
    exception = ''.join(traceback.format_exception(etyp, val, tb))
    print_fms_logfile('error', [rm_timer(exception)])

    # stop remaining timers
    for timer in timings.active_stack[:0:-1]:
        timings.stop(timer.name)

    # print timing information
    timings.stop('global', cumulative=True)
    t_table = timings.print_timings()
    print_fms_logfile('timings', [t_table])

    # copy output files
    copy_output()

    # abort other processes if running in parallel
    if glbl.mpi['parallel']:
        glbl.mpi['comm'].Abort(1)


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
