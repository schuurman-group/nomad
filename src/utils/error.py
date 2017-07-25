"""
Linear algebra library routines.
"""

import sys
import src.fmsio.glbl as glbl

def abort(msg=None):

    if glbl.mpi['parallel']:
        glb.mpi['comm'].abort()

    else:
        sys.exit(str(msg))

def warning(msg=None):

    print("WARNING: "+str(msg))



