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
import nomad.math.constants as constants

valid_coord = ['stre', 'bend',  'tors',  '|tors|',
               'oop',  '|oop|', 'plane', '|plane|']
valid_ops   = ['sum', 'max', '|sum|']

#
def init_density(den_lib, chkpt, kwords, max_states, max_prim, max_atoms):
    """ initialize the density table"""
    global valid_coord, valid_ops

    wfn_id, n_traj, n_steps, t_times = checkpoint.retrieve_basis(chkpt)
    n_wfn   = len(wfn_id)
    t_init  = float(kwords['tinit'])
    t_final = float(kwords['tfinal'])
    t_step  = float(kwords['tstep'])

    random_seed= 1
    n_intl     = len(kwords['coords'])
    state_list = np.full( (max_states),-1,  dtype=int)
    coord_list = np.zeros(max_prim*n_intl,  dtype=int)
    cf_list    = np.zeros(max_prim*n_intl,  dtype=float)
    atm_list   = np.zeros(max_atoms*n_intl, dtype=int)
    op_list    = np.zeros(n_intl,   dtype=int)
    bnds       = np.zeros(2*n_intl, dtype=float)
    npts       = np.zeros(n_intl,   dtype=int)

    if kwords['normalize']:
        for i_crd in range(n_intl):
            cf = np.array(kwords['coefs'][i_crd], dtype=float)
            cf *= 1./math.sqrt(np.dot(cf, cf))
            kwords['coefs'][i_crd] = cf.tolist()

    for state in range(len(kwords['states'])):
        state_list[state] = int(kwords['states'][state])
    for crd in range(n_intl):
        npts[crd]     = float(kwords['npts'][crd])
        op_list[crd]  = int(valid_ops.index(kwords['op'][crd])+1)
        bnds[2*crd]   = float(kwords['bounds'][crd][0])
        bnds[2*crd+1] = float(kwords['bounds'][crd][1])
        for prim in range(len(kwords['coords'][crd])):
            indx = max_prim*crd + prim
            coord_list[indx] = int(valid_coord.index(kwords['coords'][crd][prim])+1)
            cf_list[indx]    = float(kwords['coefs'][crd][prim])
        for atm in range(len(kwords['atoms'][crd])):
            indx = max_atoms*crd + atm
            atm_list[indx]   = int(kwords['atoms'][crd][atm])

    den_lib.init_density(
           ctypes.byref(convert_ctypes(random_seed, dtype='int32')),
           ctypes.byref(convert_ctypes(n_wfn,       dtype='int32')),
           ctypes.byref(convert_ctypes(n_intl,      dtype='int32')),
           ctypes.byref(convert_ctypes(state_list,  dtype='int32')),
           ctypes.byref(convert_ctypes(coord_list,  dtype='int32')),
           ctypes.byref(convert_ctypes(cf_list,     dtype='double')),
           ctypes.byref(convert_ctypes(atm_list,    dtype='int32')),
           ctypes.byref(convert_ctypes(op_list,     dtype='int32')),
           ctypes.byref(convert_ctypes(bnds,        dtype='double')),
           ctypes.byref(convert_ctypes(npts,        dtype='int32')))

    return

#
def evaluate_density(den_lib, time, npts):
    """ evaluate the density at t=time"""

    n_grid   = int(np.prod(npts))
    converge = convert_ctypes(0., dtype='double')
    n_sample = convert_ctypes(0,  dtype='int32')
    n_traj   = convert_ctypes(0,  dtype='int32')
    norm_on  = convert_ctypes(0., dtype='double')
    norm_off = convert_ctypes(0., dtype='double')
    int_grid = convert_ctypes(np.zeros(n_grid, dtype=float), dtype='double')

    den_lib.evaluate_density(
            ctypes.byref(convert_ctypes(time,   dtype='double')),             
            ctypes.byref(convert_ctypes(n_grid, dtype='int32')),
            int_grid,
            ctypes.byref(converge),
            ctypes.byref(n_sample),
            ctypes.byref(n_traj),
            ctypes.byref(norm_on),
            ctypes.byref(norm_off))

    out_nd_grid = np.reshape(np.array(int_grid, dtype=float), tuple(npts))
    return out_nd_grid, float(converge.value), int(n_sample.value), \
                        int(n_traj.value),    float(norm_on.value), \
                        float(norm_off.value)

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


