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

