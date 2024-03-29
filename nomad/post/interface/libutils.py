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
def init_table(traj_lib, default_step, step_method):
    """ initialize a trajectory table """

    traj_lib.traj_init_table(
           ctypes.byref(convert_ctypes(default_step, dtype='double')),
           ctypes.byref(convert_ctypes(step_method, dtype='int32')))

    return

#
def init_amplitudes(amp_lib, default_step, step_method):
    """ initialize amplitude table """

    amp_lib.amp_init_table(
         ctypes.byref(convert_ctypes(default_step, dtype='double')),
         ctypes.byref(convert_ctypes(step_method,  dtype='int32')))


    return

#
def create_table(traj_lib, chkpt):
    """create a trajectory table"""

    # return a dictionary of keywords to caller
    kwords = dict()

    # extract information about the basis
    kwords['batches'], kwords['ntraj'], n_steps, t_times = \
                                     checkpoint.retrieve_basis(chkpt)

    kwords['nbatch']  = len(kwords['batches'])
    kwords['ntotal']  = sum(kwords['ntraj'])
    kwords['nstates'] = int(glbl.properties['n_states'])
    kwords['ncrds']   = len(glbl.properties['crd_widths'])
    kwords['widths']  = np.array(glbl.properties['crd_widths'], dtype=float)
    kwords['masses']  = np.array(glbl.properties['crd_masses'], dtype=float)

    x_ref  = np.array(glbl.properties['init_coords'][0,0],dtype=float)
    p_ref  = np.array(glbl.properties['init_coords'][0,1],dtype=float)

    # initialize the trajectory table
    traj_lib.traj_create_table(
        ctypes.byref(convert_ctypes(kwords['nbatch'],  dtype='int32')),
        ctypes.byref(convert_ctypes(kwords['ntotal'],  dtype='int32')),
        ctypes.byref(convert_ctypes(kwords['nstates'], dtype='int32')),
        ctypes.byref(convert_ctypes(kwords['ncrds'],   dtype='int32')),
        ctypes.byref(convert_ctypes(kwords['widths'],  dtype='double')),
        ctypes.byref(convert_ctypes(kwords['masses'],  dtype='double')),
        ctypes.byref(convert_ctypes(x_ref,             dtype='double')),
        ctypes.byref(convert_ctypes(p_ref,             dtype='double')))

    return kwords 

#
def create_amp_table(amp_lib, kwords):
    """ create a table to hold all trajectory amplitudes"""

    # this is already causing problems: amp table needs to be consistent
    # with trajectory batches and there we start indexing at '1' 
    batch_lbl = convert_ctypes(kwords['tbatch']+1, dtype='int32')

    amp_lib.amp_create_table(
             ctypes.byref(convert_ctypes(kwords['nbatch'],  dtype='int32')),
             ctypes.byref(convert_ctypes(kwords['nstates'], dtype='int32')),
             ctypes.byref(convert_ctypes(kwords['ntotal'], dtype='int32')),
                          batch_lbl,
                          convert_ctypes(kwords['nsteps'], dtype='int32'),
                          convert_ctypes(kwords['tbnds'],  dtype='double'))
    return


def load_trajectories(traj_lib, chkpt, kwords, ti, tf):
    """ load all the trajectories from chkpt into traj_lib"""

    # initialize the time and label arrays for the amplitude table
    kwords['tbatch']  = np.zeros(kwords['ntotal'],   dtype=int)
    kwords['nsteps']  = np.zeros(kwords['ntotal'],   dtype=int)    
    kwords['tbnds']   = np.zeros(2*kwords['ntotal'], dtype=float)
    kwords['parent']  = dict()
    kwords['batches'] = []
    kwords['ntraj']   = []
    kwords['traj_id'] = dict()

    batch_labels = np.zeros(kwords['ntotal'],   dtype=int)
    n_steps      = np.zeros(kwords['ntotal'],   dtype=int)
    t_bnds       = np.zeros(2*kwords['ntotal'], dtype=float)
    icnt         = 0

    # start processing trajectory data by looping over wavefunction
    # objects and add each trajectory to the dll
    icnt   = 0
    t_max  = 0.
    widths = kwords['widths']
    masses = kwords['masses']
    dsets  = ['states',       'phase', 'potential','geom',
              'momentum','derivative', 'coupling', 'amp', 'glbl']

    dbl_set = ['time', 'phase', 'potential', 'geom', 'momentum',
               'derivative', 'coupling']
    wfn_keys = [key_val for key_val in chkpt.keys()
                                    if 'wavefunction' in key_val]

    for wfn_grp in wfn_keys:
        # batch starts indexing from 0
        batch = int(wfn_grp[wfn_grp.index('.')+1:])
        kwords['batches'].extend([batch])
        kwords['parent'][batch] = dict()
        kwords['traj_id'][batch] = []
        kwords['ntraj'].extend([0])

        traj_keys = [key_val for key_val in chkpt[wfn_grp].keys()
                                    if checkpoint.isTrajectory(key_val)]
        for traj_grp in traj_keys:

            tid = int(traj_grp)
            kwords['traj_id'][batch].extend([tid])

            data = dict()
            t    = None
            for iset in dsets:
                dset = wfn_grp+"/"+traj_grp+"/"+iset
                if dset in chkpt:
                    t, data[iset] = checkpoint.retrieve_dataset(chkpt, dset, ti, tf)

            # make time a vector (instead of Nx1 matrix)
            t = t.flatten()

            # piece of code for backwards compatability -- can be removed
            # eventually
            if 'glbl' in data.keys():
                data['states'] = np.column_stack((data['glbl'][:,0],
                                                  data['glbl'][:,1]))
                data['phase']  = np.array(data['glbl'][:,2], dtype=float)
                data['amp']    = np.column_stack((data['glbl'][:,3],
                                                  data['glbl'][:,4]))

            # go through and confirm that the leading dimension for all
            # vector arguments is <= nt
            for datum in data.keys():
                if len(data[datum]) != len(t.flatten()):
                    sys.exit("len("+str(datum)+") != "+str(len(t))) 

            kwords['parent'][batch][int(traj_grp)] = int(data['states'][0,0])
            t_max   = max(t_max, np.amax(t))
            state   = int(data['states'][0,1])
            nt      = len(t)
            nd      = len(glbl.properties['crd_widths'])
            data['time'] = t

            # when we pass batch label to fortran, ensure indexing starts at '1' (not '0')
            args = [ctypes.byref(convert_ctypes(batch+1, dtype='int32')),
                    ctypes.byref(convert_ctypes(nt,    dtype='int32')),
                    ctypes.byref(convert_ctypes(tid,   dtype='int32')),
                    ctypes.byref(convert_ctypes(state, dtype='int32')),
                    convert_ctypes(widths,             dtype='double'),
                    convert_ctypes(masses,             dtype='double')]
            # because we're loading by timestep, data in row-major form. 
            # When we flatten, want to do so to transpose. This is done 
            # in one step by flattening to 'row-major' C form (rather 
            # than transposing, then "F" form).
            args  += [convert_ctypes(data[i].flatten('C'),
                             dtype='double') for i in dbl_set]
            # add amps separately
            ampr  = convert_ctypes(data['amp'][:,0], dtype='double')
            ampi  = convert_ctypes(data['amp'][:,1], dtype='double')
            args  += [ampr, ampi]
            traj_lib.add_trajectory(*args)

            # add data for summary output
            kwords['ntraj'][-1]             += 1
            kwords['tbatch'][icnt]           = batch
            kwords['nsteps'][icnt]           = nt
            kwords['tbnds'][2*icnt:2*icnt+2] = np.array([t[0], np.amax(t)])
            icnt += 1

    kwords['tmax'] = t_max
    return kwords 

#
def traj_timestep_info(traj_lib, reset, batch, ntotal):
    """return how many trajectories in batch=batch exist at 
       time specified by the 'current_time' in the trajectory table"""

    time    = convert_ctypes(0., dtype='double')
    n_traj  = convert_ctypes(0,  dtype='int32')
    indices = convert_ctypes(np.zeros(ntotal, dtype=int), dtype='int32')

    traj_lib.traj_timestep_info(
                       ctypes.byref(convert_ctypes(reset, dtype='logical')),
                       ctypes.byref(convert_ctypes(batch, dtype='int32')),
                       ctypes.byref(time),
                       ctypes.byref(n_traj),
                       indices)

    return float(time.value), int(n_traj.value), np.array(indices[:int(n_traj.value)], dtype=int)



#
def amp_timestep_info(amp_lib, reset, batch, ntotal):
    """current information regarding the current time step in the
       amplitude table"""

    time    = convert_ctypes(0., dtype='double')
    n_traj  = convert_ctypes(0,  dtype='int32')
    indices = convert_ctypes(np.zeros(ntotal, dtype=int), dtype='int32')

    amp_lib.amp_timestep_info(ctypes.byref(convert_ctypes(reset, dtype='logical')),
                              ctypes.byref(convert_ctypes(batch, dtype='int32')),
                              ctypes.byref(time),
                              ctypes.byref(n_traj),
                              indices)

    return float(time.value), int(n_traj.value), np.array(indices[:int(n_traj.value)], dtype=int)

#
def traj_retrieve_timestep(traj_lib, kwords, time, batch, ntraj):
    """retrieve a list of trajectories from a trajectory table
       at time = time."""

    nt      = ntraj
    ns      = kwords['nstates']
    nc      = kwords['ncrds']
    widths  = kwords['widths']
    masses  = kwords['masses']   

    t_current = convert_ctypes(time,  dtype='double')
    i_batch   = convert_ctypes(batch, dtype='int32')
    n_traj    = convert_ctypes(ntraj, dtype='int32')
    n_fnd     = convert_ctypes(0,     dtype='int32')

    # retrieve the trajectory information for the time specified from amplitude table
    t_ids = convert_ctypes(np.zeros(nt,          dtype=int),   dtype='int32')
    state = convert_ctypes(np.zeros(nt,          dtype=int),   dtype='int32')
    amp_r = convert_ctypes(np.zeros(nt,          dtype=float), dtype='double')
    amp_i = convert_ctypes(np.zeros(nt,          dtype=float), dtype='double')
    eners = convert_ctypes(np.zeros(ns*nt,       dtype=float), dtype='double')
    x     = convert_ctypes(np.zeros(nc*nt,       dtype=float), dtype='double')
    p     = convert_ctypes(np.zeros(nc*nt,       dtype=float), dtype='double')
    deriv = convert_ctypes(np.zeros(nc*ns*ns*nt, dtype=float), dtype='double')
    coup  = convert_ctypes(np.zeros(ns*ns*nt,    dtype=float), dtype='double')
    traj_lib.traj_retrieve_timestep(ctypes.byref(t_current),
                                    ctypes.byref(i_batch),
                                    ctypes.byref(n_traj),
                                    ctypes.byref(n_fnd),
                                    t_ids,
                                    state,
                                    amp_r,
                                    amp_i,
                                    eners,
                                    x,
                                    p,
                                    deriv,
                                    coup)

    # add trajectories one at a time 
    traj_list = []
    labels    = np.array(t_ids, dtype=int)

    for itraj in range(int(n_fnd.value)):

        parent_traj = -1 
        if batch in kwords['parent'].keys():
            if labels[itraj] in kwords['parent'][batch].keys():
                parent_traj = kwords['parent'][batch][labels[itraj]]
       
        new_surf = surface.Surface()
        new_traj = trajectory.Trajectory(ns, nc,
                                         width=widths,
                                         mass=masses,
                                         label=labels[itraj],
                                         kecoef=0.5/masses,
                                         parent=parent_traj)

        new_traj.state = int(state[itraj])
        new_traj.update_x(np.array(x[nc*itraj:nc*(itraj+1)]))
        new_traj.update_p(np.array(p[nc*itraj:nc*(itraj+1)]))
        new_traj.update_amplitude(complex(float(amp_r[itraj]),float(amp_i[itraj])))

        new_surf.add_data('geom', new_traj.x())
        new_surf.add_data('potential', np.array(eners[ns*itraj:ns*(itraj+1)]))

        dlen       = nc * ns * ns
        derivative = np.reshape(np.array(deriv[dlen*itraj:dlen*(itraj+1)]),
                                        tuple([nc, ns, ns]))
        new_surf.add_data('derivative', derivative)

        clen      = ns * ns
        coupling  = np.reshape(np.array(coup[clen*itraj:clen*(itraj+1)]),
                                    tuple([ns, ns]))
        new_surf.add_data('coupling', coupling)

        # add the potential energy surface object to the trajectory
        new_traj.update_pes_info(new_surf)

        # append trajectory to list
        traj_list.append(new_traj)


    return traj_list

#
def amp_retrieve_timestep(amp_lib, indices):
    """retrieve amplitudes from amplitude table using current_row"""

    # now retrieve the amplitude from the amplitude table
    nt        = len(indices)
    n_traj    = convert_ctypes(nt,           dtype='int32')
    indx_lst  = convert_ctypes(indices,      dtype='int32')
    amp_r     = convert_ctypes(np.zeros(nt), dtype='double')
    amp_i     = convert_ctypes(np.zeros(nt), dtype='double')
    amp_lib.amp_retrieve_timestep(ctypes.byref(n_traj),
                                   indx_lst, 
                                   amp_r,  
                                   amp_i)
    
    amp_list  = np.array([complex(float(amp_r[i]),float(amp_i[i])) 
                                       for i in range(nt)], dtype=complex)

    return amp_list

#
def amp_next_timestep(amp_lib, indices, n):
    """advance the current_row counter in amptable by 1"""

    n_traj   = convert_ctypes(n,       dtype='int32')
    indx_lst = convert_ctypes(indices, dtype='int32')
    max_step = convert_ctypes(False,   dtype='logical')

    # advance to the next time step
    amp_lib.amp_next_timestep(ctypes.byref(n_traj), indx_lst, ctypes.byref(max_step))

    return max_step.value

#
def print_geom_file(traj_list, geom_file, time, prev_step):
    """print a set of geometries to geom_file"""

    for traj in traj_list:
        mass     = traj.masses() / constants.amu2au
        x        = traj.x()
        natm     = int(len(mass)/3.)
        atm_indx = [np.where((mass[3*i] - atom_lib.atom_mass).round() 
                                   == 0 )[0][0] for i in range(natm)]
        atms = [atom_lib.atom_name[atm_indx[i]] for i in range(natm)]

        if traj.label in prev_step.keys():
            prev = prev_step[traj.label]
        elif traj.parent in prev_step.keys():
            prev = prev_step[traj.parent]
        else:
            prev = -1

        out_str = str(natm)+"\n"
        out_str = out_str + " t/fs "   + str(time) + \
                            " id "     + str(traj.label) + \
                            " state "  + str(traj.state) + \
                            " prev  "  + str(prev) + \
                            " energy " + str([traj.energy(st) - 
                                           glbl.properties['pot_shift'] 
                                         for st in range(traj.nstates)]) + \
                            " amp "    + '{:12.8f}'.format(traj.amplitude)+"\n"
        for iatm in range(natm):
            line = atms[iatm] + '{:12.6f} {:12.6f} {:12.6f}'.format(
                                           *x[3*iatm:3*iatm+3])
            out_str = out_str + line + "\n"

        geom_file.write(out_str)

    return

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


