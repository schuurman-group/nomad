"""
A set of utilities for transferring trajectory data from python
drivers into fortran libraries
"""

import math as math
import numpy as np
import ctypes as ctypes
import nomad.core.glbl as glbl
import nomad.core.surface as surface
import nomad.core.trajectory as trajectory
import nomad.core.matrices as matrices
import nomad.core.checkpoint as checkpoint
import nomad.core.atom_lib as atom_lib
import nomad.common.constants as constants

#
def init_propagate(prop_lib, full_basis, init_method):
    """ initialize the propagation module """

    prop_lib.init_propagate(
         ctypes.byref(convert_ctypes(full_basis,   dtype='logical')),
         ctypes.byref(convert_ctypes(init_method,  dtype='int32')))
  
    return

#
def init_integrals(prop_lib, int_method, alpha, omega, foterms, scal):
    """pass the vibronic hamiltonian parameters to the 
       propagation routines"""

    prop_lib.init_integrals(
         ctypes.byref(convert_ctypes(int_method, dtype='int32')),
         ctypes.byref(convert_ctypes(len(alpha), dtype='int32')),
         ctypes.byref(convert_ctypes(alpha,      dtype='double')),
         ctypes.byref(convert_ctypes(omega,      dtype='double')),
         ctypes.byref(convert_ctypes(foterms,    dtype='double')),
         ctypes.byref(convert_ctypes(scal,       dtype='double')))

    return

#
def propagate(prop_lib, ti, tf):
    """ propgate trajectories in each of the loaded 
        batches from ti to tf"""

    prop_lib.propagate(
               ctypes.byref(convert_ctypes(ti, dtype='double')),
               ctypes.byref(convert_ctypes(tf, dtype='double')))

    return

#
def pop_retrieve_timestep(traj_lib, time, batch, nst):
    """return the state populations at time=time for batch=batch"""

    pops = convert_ctypes(np.zeros(nst), dtype='double')
    norm = convert_ctypes(0.,            dtype='double')

    print('calling pop_retrieve_timestep.')
    traj_lib.pop_retrieve_timestep(
               ctypes.byref(convert_ctypes(time,  dtype='double')),
               ctypes.byref(convert_ctypes(batch, dtype='int32')),
               ctypes.byref(convert_ctypes(nst,   dtype='int32')),
               pops,
               ctypes.byref(norm))

    return np.array(pops),float(norm.value)

#
def retrieve_matrices(amp_lib, time, batch, n, nst):
    """retrieve matrices from amplitude table at time=time"""

    t_current = convert_ctypes(time,  dtype='double')
    i_batch   = convert_ctypes(batch, dtype='int32')
    n_traj    = convert_ctypes(n,     dtype='int32')
    n_state   = convert_ctypes(nst,   dtype='int32')

    # lastly, retrieve the matrices generated using the propagation method of choice
    s_r    = convert_ctypes(np.zeros(n*n, dtype=float), dtype='double')
    s_i    = convert_ctypes(np.zeros(n*n, dtype=float), dtype='double')
    t_r    = convert_ctypes(np.zeros(n*n, dtype=float), dtype='double')
    t_i    = convert_ctypes(np.zeros(n*n, dtype=float), dtype='double')
    v_r    = convert_ctypes(np.zeros(n*n, dtype=float), dtype='double')
    v_i    = convert_ctypes(np.zeros(n*n, dtype=float), dtype='double')
    sdt_r  = convert_ctypes(np.zeros(n*n, dtype=float), dtype='double')
    sdt_i  = convert_ctypes(np.zeros(n*n, dtype=float), dtype='double')
    heff_r = convert_ctypes(np.zeros(n*n, dtype=float), dtype='double')
    heff_i = convert_ctypes(np.zeros(n*n, dtype=float), dtype='double')
    pop_r  = convert_ctypes(np.zeros(n*n*nst, dtype=float), dtype='double')
    pop_i  = convert_ctypes(np.zeros(n*n*nst, dtype=float), dtype='double')

    amp_lib.retrieve_matrices(ctypes.byref(t_current),
                              ctypes.byref(i_batch),
                              ctypes.byref(n_traj),
                              ctypes.byref(n_state),
                              s_r,    s_i,
                              t_r,    t_i,
                              v_r,    v_i,
                              sdt_r,  sdt_i,
                              heff_r, heff_i,
                              pop_r,  pop_i)

    mats     = matrices.Matrices()
    mats.set('s',      np.reshape(np.array(s_r) + 1.j*np.array(s_i), tuple([n,n]),order='F'))
    mats.set('t',      np.reshape(np.array(t_r) + 1.j*np.array(t_i), tuple([n,n]),order='F'))
    mats.set('v',      np.reshape(np.array(v_r) + 1.j*np.array(v_i), tuple([n,n]),order='F'))
    mats.set('sdot',   np.reshape(np.array(sdt_r) + 1.j*np.array(sdt_i), tuple([n,n]),order='F'))
    mats.set('heff',   np.reshape(np.array(heff_r) + 1.j*np.array(heff_i), tuple([n,n]),order='F'))
    mats.set('popwt',  np.reshape(np.array(pop_r) + 1.j*np.array(pop_i), tuple([n,n,nst]), order='F'))
    # assume the trajectory overlap and s matrix are same for now
    mats.set('s_traj', mats.matrix['s'])

    #print('time='+str(time)+' popwt='+str(mats.matrix['popwt']))
 
    return mats

#
def convert_ctypes(py_val, dtype=None):
    """convert a python array into a C-data type"""

    # note: the current approach is used based on:
    # https://bugs.python.org/issue27926
    # Namely, the default Python constructor is _very_ slow.
    # if that changes, then this function can change too...

    # there are fancier ways to do this by querying both the array and
    # machine environment, but this will do for now
    if dtype == 'int32':
        type_sym = 'i';i_size = 4; ctype_sym = ctypes.c_int32
    elif dtype == 'int64':
        type_sym = 'i';i_size = 8; ctype_sym = ctypes.c_int64
    elif dtype == 'double':
        type_sym = 'd';i_size = 8; ctype_sym = ctypes.c_double
    elif dtype == 'logical':
        type_sym = 'i';i_size = 4; ctype_sym = ctypes.c_bool
    else:
        sys.exit('convert_ctypes does not recognize dtype='+str(dtype))

    if isinstance(py_val, (float, int)):
        return ctype_sym(py_val)

    c_arr = (ctype_sym * py_val.size)(*py_val)

    return c_arr


