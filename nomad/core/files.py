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
import nomad.common.constants as constants


def read_input(fname):
    """Reads the nomad input files.

    This file contains variables related to the running of the
    dynamics simulation.

    Valid sections are:
        methods
        properties
        [interface name]
    The methods section must be present to define the simulation
    run type. The properties section is also required with at least
    the 'geometry' keyword.
    """
    # Read input file. Small enough to gulp the whole thing
    #with open(glbl.paths['home_path'] + fname, 'r') as infile:
    with open(fname, 'r') as infile:
        nomad_input = infile.readlines()

    # remove comment lines marked with '#' or '!'
    nomad_input = [re.split('#|!', ln)[0] for ln in nomad_input]

    # parse the methods section
    parse_section('methods', nomad_input)

    # parse the properties section
    parse_section('properties', nomad_input)

    # parse the interface-specific section if present
    try:
        parse_section(glbl.methods['interface'], nomad_input)
    except IOError:
        pass

    # setup calculation consistent with the input
    setup_input()


#------------------------------------------------------------------
#
# Below functions should not be called outside the module
#
def parse_section(section, file_contents):
    """Reads a namelist style input, returns results in dictionary.

    Set keywords in the appropriate keyword dictionary by parsing
    input array.
    """
    nlines = len(file_contents)
    sec_dict = getattr(glbl, section)

    # find the start of the section
    for i in range(nlines):
        if re.search('begin\s+' + section, file_contents[i]):
            break
        elif i == nlines - 1:
            raise IOError(section + ' section not found in input file')

    # read the section contents
    j = i + 1
    while j < nlines:
        if re.search('end\s+' + section, file_contents[j]):
            break
        elif j == nlines - 1:
            raise IOError('no end found for ' + section + ' section')
        else:
            line = file_contents[j]
            lwhite = len(line) - len(line.lstrip())
            var = line.split(maxsplit=1)
            for i in range(1, nlines):
                # keep reading if the next line is indented relative to
                # the variable name
                next_line = file_contents[j+i]
                nlwhite = len(next_line) - len(next_line.lstrip())
                if nlwhite <= lwhite and next_line.lstrip() != '':
                    break
                elif len(var) == 1:
                    # variable contents start on next line
                    var.append(next_line)
                else:
                    var[1] += next_line

            # check and parse variable name and value
            if len(var) == 1:
                # boolean variables don't require an argument
                vname = var[0].replace('no_','')
                if var[0] in sec_dict.keys():
                    sec_dict[var[0]] = True
                elif vname in sec_dict.keys():
                    sec_dict[vname] = False
                else:
                    print('Skipping unrecognized variable ' + vname + ' in ' +
                          section + ' section')
            elif var[0] not in sec_dict.keys():
                print('Skipping unrecognized variable ' + var[0] + ' in ' +
                      section + ' section')
            elif var[0] == 'init_coords':
                sec_dict[var[0]] = parse_coords(var[1])
            elif var[0] == 'hessian':
                sec_dict[var[0]] = parse_hessian(var[1])
            else:
                sec_dict[var[0]] = parse_value(var[1])

            j += i


def parse_coords(valstr):
    """Returns geometry and momenta or a set of geometries and momenta
    from a string or input filename.

    Geometries are in the XYZ file format in atomic units by default.
    Momenta are specified in the XYZ format, i.e. alongside the cartesian
    coordinates of each atom. If not given, momenta are set to zero.
    """
    all_lines = valstr.split('\n')[:-1]
    nlines = len(all_lines)
    if nlines == 1:
        # filename given, read in XYZ file
        with open(all_lines[0], 'r') as f:
            all_lines = f.read().split('\n')[:-1]
            nlines = len(all_lines)

    # read the geometries and momenta if applicable
    natms = []
    comms = []
    labels = []
    coords = []
    i = 0
    while i < nlines:
        natm = int(all_lines[i])
        natms.append(natm)
        comms.append(all_lines[i+1].lower())
        icoord = np.array([line.split() for line in all_lines[i+2:i+natm+2]])
        labels.append(icoord[:,0])
        coords.append(icoord[:,1:])
        i += natm + 2

    nmol = len(natms)
    natms = np.array(natms)
    labels = np.array(labels)
    coords = np.array(coords, dtype=float)

    # make sure all geometries are the same molecule
    if not np.all(natms == natms[0]):
        raise ValueError('All geometries must correspond to the same ' +
                         'molecule.')
    for i in range(1, nmol):
        if not np.all(labels[i] == labels[0]):
            raise ValueError('All geometries must correspond to the same ' +
                             'molecule.')

    # 1D or 3D geometries only
    cshape = coords.shape
    dcoord = cshape[-1]
    if dcoord not in [1, 2, 3, 6]:
        raise ValueError('Coordinates must be 1D or 3D (2D or 6D with momenta')
    if dcoord % 2 == 1:
        # reshape x
        xyz = coords.reshape(cshape[0], cshape[1]*cshape[2])
        # momenta not given, set to zero
        mom = np.zeros_like(xyz)
        coords = np.array([[xi, pi] for xi, pi in zip(xyz, mom)])
        dcoord *= 2
    else:
        # reshape x and p
        xyz = coords[:,:,:dcoord//2].reshape(cshape[0], cshape[1]*cshape[2]//2)
        mom = coords[:,:,dcoord//2:].reshape(cshape[0], cshape[1]*cshape[2]//2)
        coords = np.array([[xi, pi] for xi, pi in zip(xyz, mom)])

    # parse comment line to get units (Bohr is default in XYZ format)
    for i in range(nmol):
        xconv = 1.
        pconv = 1.
        if 'units=' in comms[i] and 'angstrom' in comms[i]:
            #unit = re.sub(r'units\s*=\s*(.*[a-z])\s.*', r'\1', comms[i])
            #if unit == 'bohr':
            xconv = 1. / constants.bohr2ang
            pconv = constants.amu2au / (constants.fs2au * constants.bohr2ang)
        coords[:,0] *= xconv
        coords[:,1] *= pconv

    # set atomic labels
    glbl.properties['crd_labels'] = np.repeat(labels[0], dcoord//2)

    return coords


def parse_hessian(valstr):
    """Reads the non-mass-weighted Hessian matrix from hessian.dat."""
    all_lines = valstr.split('\n')[:-1]
    nlines = len(all_lines)
    if nlines == 1:
        try:
            # 1 x 1 Hessian
            return np.atleast_2d(all_lines[0]).astype(float)
        except ValueError:
            # filename given, read in Hessian file
            return np.loadtxt(all_lines[0], dtype=float)
    else:
        # full Hessian given in input
        return np.array([line.split() for line in all_lines], dtype=float)


def parse_value(valstr):
    """Returns a value converted to the appropriate type and shape.

    By default, spaces and newlines will be treated as delimiters. The
    combination of both will be treated as a 2D array.
    """
    all_lines = valstr.split('\n')[:-1]
    split_lines = [line.split() for line in all_lines]

    if len(all_lines) > 1 and len(split_lines[0]) == 1:
        # newline and space give the same result for a vector
        split_lines = [[line[0] for line in split_lines]]

    if len(split_lines) == 1:
        if len(split_lines[0]) == 1:
            # handle single values
            return convert_value(split_lines[0][0])
        else:
            # handle vectors
            return convert_array(split_lines[0])
    else:
        # handle 2D arrays
        return convert_array(split_lines)


def convert_value(val):
    """Converts a string value to NoneType, bool, int, float or string."""
    if val.lower() == 'none':
        return None
    elif val.lower() == 'true':
        return True
    elif val.lower() == 'false':
        return False

    try:
        return int(val)
    except ValueError:
        pass

    try:
        return float(val)
    except ValueError:
        pass

    return val


def convert_array(val_list):
    """Converts a list of strings to an array of ints, floats or strings."""
    try:
        return np.array(val_list, dtype=int)
    except ValueError:
        pass

    try:
        return np.array(val_list, dtype=float)
    except ValueError:
        pass

    return np.array(val_list, dtype=str)


def setup_input():
    """Sets up global methods and properties based on the input."""

    # set atomic widths and masses unless they are given in the input file
    natm = len(glbl.properties['crd_labels'])
    if glbl.methods['interface'] == 'vibronic' or glbl.methods['interface'] == 'models':
        wlst = np.array([np.sqrt(2) / 2 for i in range(natm)])
        mlst = np.array([1. for i in range(natm)])
    elif glbl.properties['use_atom_lib']:
        labels = glbl.properties['crd_labels']
        wlst = np.empty(len(labels))
        mlst = np.empty(len(labels))
        for i, l in enumerate(labels):
            wlst[i], mlst[i], num = atom_lib.atom_data(l)
    else:
        wlst = np.array([0.0 for i in range(natm)])
        mlst = np.array([1.0 for i in range(natm)])

    if glbl.properties['crd_widths'] is None:
        glbl.properties['crd_widths'] = wlst
    elif isinstance(glbl.properties['crd_widths'], (int, float)):
        glbl.properties['crd_widths'] = np.array([glbl.properties['crd_widths']
                                                  for i in range(natm)], dtype=float)
    if glbl.properties['crd_masses'] is None:
        glbl.properties['crd_masses'] = mlst
    elif isinstance(glbl.properties['crd_masses'], (int, float)):
        glbl.properties['crd_masses'] = np.array([glbl.properties['crd_masses']
                                                  for i in range(natm)], dtype=float)

    # set init_state for all trajectories
    istate = glbl.properties['init_state']
    if isinstance(istate, int):
        glbl.properties['init_state'] = [istate for i in
                                         range(glbl.properties['n_init_traj'])]
