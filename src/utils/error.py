"""
Linear algebra library routines.
"""

import sys
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio

def abort(msg=""):

    print("ABORTING: "+str(msg))
 
    if glbl.mpi['parallel']:
        # this is too dangerous in the long run. If this 
        # isn't root process, we don't want to be moving files
        # being written by root.
        fileio.copy_output()
        glbl.mpi['comm'].Abort()

    else:
        fileio.cleanup()
        sys.exit(str(msg))

def warning(msg=None):

    print("WARNING: "+str(msg))



