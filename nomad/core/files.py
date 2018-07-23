"""
Routines for reading input files and writing log files.
"""
import os
import re
import ast
import shutil
import numpy as np
import nomad.core.glbl as glbl
import nomad.core.atom_lib as atom_lib
import nomad.integrals.integral as integral
import nomad.core.matrices as matrices


def read_input(fname):
    """Reads the nomad.input files.

    This file contains variables related to the running of the
    dynamics simulation.

    Valid sections are:
        initial_conditions
        propagation
        spawning
        interface
        geometry
        printing
    """
    # Read input file. Small enough to gulp the whole thing
    with open(glbl.home_path + '/' + fname, 'r') as infile:
        nomad_input = infile.readlines()

    # remove comment lines
    nomad_input = [re.split('#|!', ln)[0] for ln in nomad_input]

    sec_strings = list(glbl.input_groups)

    # look for begining of input section
    current_line = 0
    while current_line < len(nomad_input):
        sec_start = [re.search(str('begin '+sec_strings[i]+'-section'), nomad_input[current_line])
                     for i in range(len(sec_strings))]
        if all([v is None for v in sec_start]):
            current_line += 1
        else:
            section = next(item for item in sec_start
                           if item is not None).string
            section = section.replace('-section','').replace('begin','').strip()
            current_line = parse_section(nomad_input, current_line, section)

    # ensure that input is internally consistent
    validate()


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


#------------------------------------------------------------------
#
# Below functions should not be called outside the module
#
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


def validate():
    """Ensures that input values are internally consistent.

    Currently there are multiple ways to set variables in the nuclear
    basis section. The following lines ensure that subsequent usage of the
    entries in glbl is consistent, regardless of how input specified.
    """
    # set the integral definition
    #try:
    #    glbl.master_int =__import__('nomad.integrals.'+
    #                               glbl.propagate['integrals'],fromlist=['a'])
    #except ImportError:
    #    print('Cannot import integrals: nomad.integrals.' +
    #          str(glbl.propagate['integrals']))

    glbl.master_int = integral.Integral(glbl.propagate['integrals'])
    glbl.master_mat = matrices.Matrices()

    try:
        glbl.interface = __import__('nomad.interfaces.' +
                               glbl.iface_params['interface'],fromlist=['NA'])
    except ImportError:
        print('Cannot import pes: nomad.interfaces.'+
                               str(glbl.iface_params['interface']))

    try:
        glbl.init_conds = __import__('nomad.initconds.'+glbl.sampling['init_sampling'],
                                 fromlist=['NA'])
    except ImportError:
        print('Cannot import initial conditions: nomad.initconds.'+
                               str(glbl.sampling['init_sampling']))

    try:
        glbl.adapt = __import__('nomad.adapt.'+glbl.spawning['spawning'],
                                   fromlist=['a'])
    except ImportError:
        print('Cannot import spawning: nomad.adapt.'+
                               str(glbl.spawning['spawning']))

    try:
        glbl.integrator = __import__('nomad.propagators.'+glbl.propagate['propagator'],
                                     fromlist=['a'])
    except ImportError:
        print('Cannot import propagator: nomad.propagators.'+
                               str(glbl.propagate['propagator']))

    # if geomfile specified, it's contents overwrite variable settings in nomad.input
    if os.path.isfile(glbl.nuclear_basis['geomfile']):
        (labels, geoms, moms)            = read_geometry(glbl.nuclear_basis['geomfile'])
        glbl.nuclear_basis['labels']     = labels
        glbl.nuclear_basis['geometries'] = geoms
        glbl.nuclear_basis['momenta']    = moms

    # if hessfile is specified, its contents overwrite variable settings from nomad.input
    if os.path.isfile(glbl.nuclear_basis['hessfile']):
        glbl.nuclear_basis['hessian']    = read_hessian(glbl.nuclear_basis['hessfile'])

    # if use_atom_lib, atom_lib values overwrite variables settings from nomad.input
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
    if glbl.iface_params['interface'] == 'vibronic':
        n_usr_freq = len(glbl.nuclear_basis['freqs'])

        # automatically set the "mass" of the coordinates to be 1/omega
        # the coefficient on p^2 -> 0.5 omega == 0.5 / m. Any user set masses
        # will override the default
        if len(glbl.nuclear_basis['masses']) >= n_usr_freq:
            pass
        else:
            glbl.nuclear_basis['masses'] = [1. for i in range(n_usr_freq)]
        #else all(freq != 0. for freq in glbl.nuclear_basis['freqs']):
        #    glbl.nuclear_basis['masses'] = [1./glbl.nuclear_basis['freqs'][i] for i in
        #                                     range(len(n_usr_freq)]

        # set the widths to automatically be 1/2 (i.e. assumes frequency-weighted coordinates.
        # Any user set widths will override the default
        if len(glbl.nuclear_basis['widths']) >= n_usr_freq:
            pass
        else:
            glbl.nuclear_basis['widths'] = [0.5 for i in range(n_usr_freq)]

    # set the kinetic energy coefficient
    if glbl.iface_params['interface'] == 'vibronic':
        # KE operator coefficients, mass- and frequency-scaled normal mode
        # coordinates, a_i = 0.5*omega_i
        glbl.kecoef = 0.5 * np.array(glbl.nuclear_basis['freqs'])
    else:
        glbl.kecoef = 0.5 / np.array(glbl.nuclear_basis['masses'])

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
