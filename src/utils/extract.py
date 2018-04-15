"""
Extract data from checkpoint file into human-readable
files
"""

import sys
import numpy as np
import h5py as h5py

#
#
#
def query_group(chkpt_file, data_group):
"""
Return the size in bytes of data_group in the file
chkpt_file as well as how many members of the group
there are.
"""

return

#
#
#
def extract_group(chkpt_file, data_group, time=None):
"""
Given a checkpoint file and a data group, extract all the
data sets from all times by default. If time is not None,
extract the data for a given time step.
"""

return

#
#
#
def query_set(chkpt_file, data_set):
"""
Return the size in bytes of data_set in the file
chkpt_file as well as the dimensions of data_set and
the data type.
"""

return

#
#
#
def extract_set(chkpt_file, data_set, time=None):
"""
Given a checkpoint file and a data set, extract all the
data sets from all times by default. If time is not None,
extract the data for a given time step.
"""

return

#
#
#
def print_group(group_struct, file_name, directory_name=None):



return

#
#
#
def print_set(set_struct, file_name, directory_name=None):


return

#
#
#
def print_trajectory()
"""Documentation to come"""

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


return







