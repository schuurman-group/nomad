"""
Module for methods dealing with text input and output as well as
general file management.
"""
import os
import re
import numpy as np
from glob import glob


def convert_str(string):
    """If possible, converts a string to an int, float, list of ints
    or list of floats."""
    slower = string.lower()
    if slower == 'none':
        return None
    elif slower == 'true':
        return True
    elif slower == 'false':
        return False
    elif ';' in string and ',' in string:
        # 2D list delimited by ';' followed by ','
        slist = [ln.split(',') for ln in string.split(';')]
        try:
            return [[int(el) for el in ln] for ln in slist]
        except ValueError:
            pass
        try:
            return [[float(el) for el in ln] for ln in slist]
        except ValueError:
            return [s.strip() for ln in slist for s in ln]
    elif ';' in string or ',' in string:
        # 1D list delimited by either ',' or ';'
        slist = string.replace(';',',').split(',')
        try:
            return [int(el) for el in slist]
        except ValueError:
            pass
        try:
            return [float(el) for el in slist]
        except ValueError:
            return [s.strip() for s in slist]
    else:
        # not a list, try int or float
        try:
            return int(string)
        except ValueError:
            pass
        try:
            return float(string)
        except ValueError:
            return string.strip()


def read_cfg(fname, reqvars=[]):
    """Reads the configuration file."""
    fvars = dict()
    with open(fname, 'r') as f:
        lines = f.readlines()

    # remove spaces and comments and get inputs
    lines = [ln.partition('#')[0] for ln in lines]
    for ln in lines:
        if ln not in ['\n', '']:
            vv = [string.strip() for string in ln.split('=', 1)]
            if len(vv) < 2:
                raise ValueError('Variables must be set by \'=\'')
            else:
                fvars[vv[0]] = convert_str(vv[1])

    for key in reqvars:
        if key not in fvars:
            raise KeyError('Missing required input variable: {}'.format(key))

    return fvars


def cfg_update(defdict, fname):
    """Updates a dictionary of default variables with values from
    a configuration file."""
    if os.path.exists(fname):
        new_dict = read_cfg(fname)
        for key in new_dict:
            if key not in defdict:
                print('Ignoring unrecognized variable \''+key+'\'.')
        defdict.update(new_dict)


def get_fnames(matchex):
    """Returns a list of filenames based on a matching expression
    that may include wildcards."""
    fnames = []
    if isinstance(matchex, list):
        for reffn in matchex:
            fnames += glob(reffn)
    else:
        fnames += glob(matchex)

    return fnames


def natural_keys(s):
    """Used to sort list of strings by natural numbers."""
    return [int(c) if c.isdigit() else c for c in re.split('(\d+)', s)]


def traj_to_xyz(string, outfile, elem=None, comment=''):
    """Takes a line from a TrajDump file and converts it to an
    XYZ format."""
    line = np.array(string.split(), dtype=float)
    natm = len(line) // 6 - 1
    if elem is None:
        elem = np.array(['X'] * natm)

    xyz = line[1:3*natm+1].reshape(natm, 3) * 0.52917721
    outfile.write(' {:d}\n{:s}\n'.format(natm, comment))
    for atm, xyzi in zip(elem, xyz):
        outfile.write('{:4s}{:12.6f}{:12.6f}{:12.6f}\n'.format(atm, *xyzi))


def read_dat(fname, dtype=float, skiprow=0, skipcol=0, labels=None,
             usecols=None):
    """Reads an array of data from an input file.
    Specifying labels as 'row' or 'col' will return the last skipped
    row/column as well as the data array. The 'usecols' keyword will
    only use the specified columns and takes precedence over 'skipcol'.
    """
    data = np.genfromtxt(fname, dtype=dtype, skip_header=skiprow,
                         usecols=usecols)
    if usecols is None:
        data = data[:,skipcol:]
    if labels == 'row' and skiprow > 0:
        with open(fname, 'r') as f:
            for i in range(skiprow):
                labels = np.array(f.readline().split())
        return data, labels
    elif labels == 'col' and skipcol > 0:
        labels = np.genfromtxt(fname, dtype=str, skip_header=skiprow,
                               usecols=skipcol-1)
        return data, labels
    else:
        return data


def write_dat(fname, data, labels=None, charwid=12, decwid=4, fmttyp='f'):
    """Writes an array of floating point data to an output file."""
    with open(fname, 'w') as f:
        nlin = data.shape[1]
        if labels is not None:
            lblspace = [len(s) + 1 for s in labels]
            charwid = max([charwid] + lblspace)
            f.write((nlin * '{:>{w}s}').format(*labels, w=charwid) + '\n')
        for line in data:
            nlin = len(line)
            f.write((nlin * '{:{w}.{d}{t}}').format(*line, w=charwid, d=decwid,
                                                    t=fmttyp) + '\n')
