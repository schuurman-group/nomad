"""
Routines for reading input files and writing log files.
"""
import sys
import os
import h5py
import numpy as np 
import src.integrals.integral as integral
import src.basis.matrices as matrices
import src.basis.wavefunction as wavefunction
import src.basis.trajectory as trajectory
import src.basis.centroid as centroid
import src.archive.surface as surface

chkpt_file = ''

#
#
#
def write(data_obj, file_name=None, time=None):
    """Documentation to come"""
    global chkpt_file

    # default is to use file name from previous write
    if file_name is not None:
        chkpt_file = file_name.strip()

    # if this is the first time we're writing to the archive,
    # create the bundle data set and record the time-independent 
    # bundle definitions
    if not os.path.isfile(chkpt_file):
        if isinstance(data_obj, wavefunction.Wavefunction):
            create(chkpt_file, data_obj)
        else:
            sys.exit('chkpt file must be created with wavefunction object. Exiting..')

    # open checkpoint file
    chkpt = h5py.File(chkpt_file, "a", libver='latest')

    # this definition of time over-rules all others as far as writing
    # data is concerned.
    if time is None:
        time = data_obj.time

    #------------------------------------------
    # this data get written for all simulations
    if isinstance(data_obj, wavefunction.Wavefunction):
        write_wavefunction(chkpt, data_obj, time)

    elif isinstance(data_obj, integral.Integral):
        write_integral(chkpt, data_obj, time)

    elif isinstance(data_obj, matrices.Matrices):
        write_matrices(chkpt, data_obj, time)

    else:
        sys.exit('data_obj: '+str(data_obj)+' is not recognized by checkpoint.write')

    chkpt.close()

    return

#
# called when an old checkpoint file is used to populate the
# contents of a bundle, if no time given, read last bundle
#
def read(data_obj, file_name, time=None):
    """Documentation to come"""
    global chkpt_file
    
    # string name of checkpoint file
    chkpt_file = file_name.strip()
  
    # open chkpoint file
    chkpt = h5py.File(chkpt_file, "r", libver='latest')

    if isinstance(data_obj, wavefunction.Wavefunction):

        read_wavefunction(chkpt, data_obj, time)

    # if this is an integral objects, it's going to want to load the centroid data
    # from file
    elif isinstance(data_obj, integral.Integral):

        read_integral(chkpt, data_obj, t_index)

    # load matrices from file
    elif isinstance(data_obj, matrices.Matrices):

        read_matrices(chkpt, data_obj, t_index)

    else:
        sys.exit('data_obj: '+str(data_obj)+' is not recognized by checkpoint.read')

    # close checkpoint file
    chkpt.close()

    return

#
#
#
def time_steps(grp_name, file_name=None):
    """Documentation to come"""
    global chkpt_file

    # default is to use file name from previous write
    if file_name is not None:
        chkpt_file = file_name.strip()

    # open chkpoint file
    chkpt = h5py.File(chkpt_file, "r", libver='latest')

    # if the group name is in the checkpoint file, return
    # the associated time array
    if grp_name in chkpt.keys():
        steps = chkpt[grp_name+'/time'][:]
    #else abort
    else:
        sys.exit('grp_name: '+str(grp_name)+' not present in checkpoint file')

    chkpt.close()

    return steps


#------------------------------------------------------------------------------------
#
# Should not be called outside the module
#

#
# called when a new checkpoint file is created
#
def create(file_name, wfn):
    """Documentation to come"""

    # create chkpoint file
    chkpt = h5py.File(file_name, "w", libver='latest')

    chkpt.create_group('wavefunction')
    ckhpt['wavefunction'].attrs['current_time'] = 0
    chkpt['wavefunction'].attrs['n_rows']       = 0

    chkpt.create_group('integral')
    ckhpt['integral'].attrs['current_time']     = 0
    chkpt['integral'].attrs['n_rows']           = 0

    chkpt.create_group('matrices')
    ckhpt['matrices'].attrs['current_time']     = 0
    chkpt['matrices'].attrs['n_rows']           = 0

    traj0 = wfn.traj[0]
    chkpt.create_group('simulation')
    chkpt['simulation'].attrs['nstates']        = traj0.nstates
    chkpt['simulation'].attrs['dim']            = traj0.dim
    chkpt['simulation'].attrs['widths']         = traj0.widths()
    chkpt['simulation'].attrs['masses']         = traj0.masses()

    # close following initialization
    chkpt.close()

    return

#
#
#
def write_wavefunction(chkpt, wfn, time):
    """Documentation to come"""

    wfn_data    = package_wfn(wfn)
    n_traj      = wfn.n_traj()
    n_rows      = default_blk_size(time)
    resize      = False

    # update the current row index (same for all data sets)
    current_row = chkpt['wavefunction'].attrs['current_time'] + 1

    if current_row > chkpt['wavefunction'].attrs['n_rows']:
        resize   = True
        new_size = chkpt['wavefunction'].attrs['n_rows'] + n_rows
        chkpt['wavefunction'].attrs['n_rows'] = new_size

    # first write items with time-independent dimensions
    for data_label in wfn_data.keys():
        dset = 'wavefunction/'+data_label

        if dset in chkpt:
            if resize:
                d_shape  = (new_size,) + wfn_data[data_label].shape
                chkpt[dset].resize(d_shape)
            chkpt[dset][current_row] = wfn_data[data_label]

        # if this is the first time we're trying to write this bundle,
        # create a new datasets with reasonble default sizes
        else:
            d_shape   = (new_size,) +  wfn_data[data_label].shape
            max_shape = (None,) + wfn_data[data_label].shape
            d_type    = b_data[data_label].dtype
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, dtype=d_type, compression="gzip")
            chkpt[dset][current_row] = wfn_data[data_label]

    # now step through and write trajectories
    for i in range(n_traj):
        write_trajectory(chkpt, wfn.traj[i], time)

    return

#
#
#
def write_integral(chkpt, integral, time):
    """Documentation to come"""

    int_data = package_integral(integral, time)

    # update the current row index (same for all data sets)
    current_row = chkpt['integral'].attrs['current_time'] + 1

    if current_row > chkpt['integral'].attrs['n_rows']:
        resize   = True
        new_size = chkpt['integral'].attrs['n_rows'] + n_rows
        chkpt['integral'].attrs['n_rows'] = new_size

    # first write items with time-independent dimensions
    for data_label in int_data.keys():
        dset = 'integral/'+data_label

        if dset in chkpt:
            if resize:
                d_shape  = (new_size,) + int_data[data_label].shape
                chkpt[dset].resize(d_shape)
            chkpt[dset][current_row] = int_data[data_label]

        # if this is the first time we're trying to write this bundle,
        # create a new datasets with reasonble default sizes
        else:
            d_shape   = (new_size,) +  int_data[data_label].shape
            max_shape = (None,) + int_data[data_label].shape
            d_type    = b_data[data_label].dtype
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, dtype=d_type, compression="gzip")
            chkpt[dset][current_row] = int_data[data_label]

    # now step through centroids, if they're present
    if integral.require_centroids:
        for i in range(len(integral.centroid)):
            for j in range(i):
                 if integral.centroid[i][j] is not None:
                     write_centroid(chkpt, integral.centroid[i][j], time)

    return

#
#
#
def write_matrices(chkpt, matrices, time):
    """Documentation to come"""
    # open the trajectory file
    m_data  = package_marices(matrices, time)
    n_rows  = default_blk_size(time)
    resize  = False

    # if trajectory group already exists, just append current
    # time information to existing datasets
    current_row = chkpt['matrices'].attrs['current_time'] + 1

    if current_row > chkpt['matrices'].attrs['n_rows']:
        resize = True
        new_size = chkpt['matrices'].attrs['n_rows'] + n_rows
        chkpt['matrices'].attrs['n_rows'] = new_size

    for data_label in m_data.keys():
        dset = 'matrices/'+data_label

        if dset in chkpt:
            if resize:
                d_shape  = (new_size,) + m_data[data_label].shape
                chkpt[dset].resize(d_shape)
            chkpt[dset][current_row] = m_data[data_label]

    # if this is the first time we're trying to write this trajectory,
    # create a new data group, and new data sets with reasonble default sizes
        else:
            d_shape   = (new_size,) + m_data[data_label].shape
            max_shape = (None,) + m_data[data_label].shape
            d_type    = m_data[data_label].dtype
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, dtype=d_type, compression="gzip")
            chkpt[dset][current_row] = m_data[data_label]

    return

#
#
#
def write_trajectory(chkpt, traj, time):
    """Documentation to come"""

    # open the trajectory file
    t_data   = package_trajectory(traj, time)
    t_label  = str(traj.label)
    n_rows   = default_blk_size(time)
    resize   = False

    # if trajectory group already exists, just append current
    # time information to existing datasets
    t_grp = 'wavefunction/'+t_label

    if t_grp in chkpt:

        current_row = chkpt[t_grp].attrs['current_time'] + 1

        if current_row > chkpt[t_grp].attrs['n_rows']:
            resize = True
            new_size = chkpt[t_grp].attrs['n_rows'] + n_rows
            chkpt[t_grp].attrs['n_rows'] = new_size

        for data_label in t_data.keys():
            dset = t_grp+'/'+data_label
            if resize:
                d_shape  = (new_size,) + t_data[data_label].shape
                chkpt[dset].resize(d_shape)

            chkpt[dset][current_row] = t_data[data_label]
        
    # if this is the first time we're trying to write this trajectory,
    # create a new data group, and new data sets with reasonble default sizes
    else:

        chkpt.create_group(t_grp)
        current_row                          = 0
        chkpt[t_grp].attrs['current_time']   = current_row
        chkpt[t_grp].attrs['n_rows']         = n_rows
 
        # store surface information from trajectory
        for data_label in t_data.keys():
            dset = t_grp+'/'+data_label
            d_shape   = (n_rows,) + t_data[data_label].shape
            max_shape = (None,) + t_data[data_label].shape
            d_type    = t_data[data_label].dtype
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, dtype=d_type, compression="gzip")
            chkpt[dset][current_row] = t_data[data_label]

    return

#
#
#
def write_centroid(chkpt, cent, time):
    """Documentation to come"""

    # open the trajectory file
    c_data  = package_centroid(traj, time)
    c_label = str(cent.label)
    n_rows  = default_blk_size(time)
    resize  = False

    # if trajectory group already exists, just append current
    # time information to existing datasets
    c_grp = 'integral/'+c_label

    if c_grp in chkpt:

        current_row = chkpt[c_grp].attrs['current_time'] + 1

        if current_row > chkpt[c_grp].attrs['n_rows']:
            resize = True
            new_size = chkpt[c_grp].attrs['n_rows'] + n_rows
            chkpt[c_grp].attrs['n_rows'] = new_size

        for data_label in c_data.keys():
            dset = c_grp+'/'+data_label
            if resize:
                d_shape  = (new_size,) + c_data[data_label].shape
                chkpt[dset].resize(d_shape)

            chkpt[dset][current_row] = c_data[data_label]

    # if this is the first time we're trying to write this trajectory,
    # create a new data group, and new data sets with reasonble default sizes
    else:

        chkpt.create_group(c_grp)
        current_row                          = 0
        chkpt[c_grp].attrs['current_time']   = current_row
        chkpt[c_grp].attrs['n_rows']         = n_rows

        # store surface information from trajectory
        for data_label in c_data.keys():
            dset = c_grp+'/'+data_label
            d_shape   = (n_rows,) + c_data[data_label].shape
            max_shape = (None,) + c_data[data_label].shape
            d_type    = c_data[data_label].dtype
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, dtype=d_type, compression="gzip")
            chkpt[dset][current_row] = c_data[data_label]

    return

#
#
#
def read_wavefunction(chkpt, wfn, time):
    """Documentation to come"""

    nstates = chkpt['simulation'].attrs['nstates']
    dim     = chkpt['simulation'].attrs['dim']
    widths  = chkpt['simulation'].attrs['widths']
    masses  = chkpt['simulation'].attrs['masses']

    # check that we have the desired time:
    read_row = get_time_index(chkpt['wavefunction'], time)
    if read_row is None:
        sys.exit('time='+str(time)+' requested, but not in checkpoint file')
 
    # dimensions of these objects are not time-dependent 
    wfn.time = chkpt['wavefunction/time'][read_row]
    
    for label in chkpt['wavefunction']:

        if label == 'time':
            continue

        t_grp = 'wavefunction/'+label
        t_row = get_time_index(chkpt[t_grp], time)

        if t_row is None:
            continue

        new_traj = trajectory.Trajectory(nstates, dim, width=widths, 
                                         mass=masses, label=label)
        read_trajectory(chkpt, new_traj, t_grp, t_row)
        wfn.add_trajectory(new_traj.copy())

    return

#
#
#
def read_integral(chkpt, integral, time):
    """Documentation to come"""

    nstates = chkpt['simulation'].attrs['nstates']
    dim     = chkpt['simulation'].attrs['dim']
    widths  = chkpt['simulation'].attrs['widths']
    masses  = chkpt['simulation'].attrs['masses']

    # check that we have the desired time:
    read_row = get_time_index(chkpt['integral'], time)
    if read_row is None:
        sys.exit('time='+str(time)+' requested, but not in checkpoint file')

    if integral.require_centroids:
        for label in chkpt['integral']:

            if label == 'time':
                continue

            c_grp = 'integral/'+label
            c_row = get_time_index(chkpt[c_grp, time)
 
            if c_row is None:
                continue

            new_cent = integral.Centroid(nstates=nstates, dim=dim, width=widths)
            read_centroid(chkpt, new_cent, c_grp, c_row_
            integral.add_centroid(new_cent)       

    return

#
#
#
def read_matrices(chkpt, matrices, time):
    """Documentation to come"""

    read_row = get_time_index(chkpt['matrices'], time)
    if read_row is None:
        sys.exit('time='+str(time)+' requested, but not in checkpoint file')

    for mat in chkpt['matrices']:
        dset = 'matrices/'+mat
        matrices.set(mat, chkpt[mat][read_row])

    return True
#
#
#
def read_trajectory(chkpt, new_traj, t_grp, t_row):
    """Documentation to come"""

    # populate the surface object in the trajectory
    pes = surface.Surface()
    for data_label in chkpt[t_grp].keys():  
        if data_label == 'time' and data_label == 'global':
            continue
        dset = chkpt[t_grp+'/'+data_label]
        pes.add_data(data_label, dset[t_row])

    # set information about the trajectory itself
    data_row = chkpt[t_grp+'/global'][t_row]
    [new_traj.parent, new_traj.state, new_traj.gamma, amp_real, amp_imag] = data_row[0:5]
    new_traj.update_amplitude(amp_real+1.j*amp_imag)
    new_traj.last_spawn = data_row[5:]

    new_traj.update_pes_info(pes)
    new_traj.update_x(new_traj.pes.get_data('geom'))
    new_traj.update_p(new_traj.pes.get_data('momentum'))

    return

#
#
#
def read_centroid(chkpt, new_cent, c_grp, c_row):
    """Documentation to come"""

    # populate the surface object in the trajectory
    pes = surface.Surface()
    for data_label in chkpt[c_grp].keys():
        if data_label == 'time' and data_label == 'global':
            continue
        dset = chkpt[c_grp+'/'+data_label]
        pes.add_data(data_label, dset[c_row])

    # set information about the trajectory itself
    [new_cent.parent[0], new_cent.parent[1],
     new_cent.states[0], new_cent.states[1]] = chkpt[c_grp+'/global'][c_row] 

    new_cent.update_pes_info(pes)
    new_cent.update_x(new_cent.pes.get_data('geom'))
    new_cent.update_p(new_cent.pes.get_data('momentum'))

    return

#
#
#
def get_time_index(grp, time):
    """Documentation to come"""

    time_vals = grp['time'][:]
    
    if time is None:
        return grp.attrs['current_time']

    dt       = np.absolute(time_vals - time)
    read_row = np.argmin(dt)

    # this tolerance for matching times is kinda arbitrary
    t_off = np.roll(time_vals,1)
    if dt[read_row] > np.min(np.absolute(time_vals - t_off)):
        read_row = None

    return read_row
     
#
#
#
def package_wfn(wfn): 
    """Documentation to come"""

    wfn_data = dict()

    # dimensions of these objects are not time-dependent 
    wfn_data['time']  = np.array([wfn.time], dtype='float')
    wfn_data['pop']   = wfn.pop()
    wfn_data['energy']= np.array([wfn.pot_quantum(),   wfn.kin_quantum(),
                                  wfn.pot_classical(), wfn.kin_classical()])

    return wfn_data

#
#
#
def package_integral(integral, time):
    """Documentation to come"""

    int_data = dict()

    int_data['time'] = np.array([time],dtype='float')

    return int_data


#
#
#
def package_matrices(matrices, time):
    """Documentation to come"""

    mat_data = dict()

    mat_data['time'] = np.array([time],dtype='float')

    # dimensions of these objects are time-dependent
    for typ,mat in matrices.mat_lst.items():
        mat_data[typ] = mat

    return mat_data

#
#
#
def package_trajectory(traj, time):
    """Documentation to come"""

    traj_data = dict()

    # not an element in a trajectory, but necessary to 
    # uniquely tag everything
    traj_data['time']  = np.array([time],dtype='float')

    # store information about the trajectory itself
    traj_data['global'] = np.concatenate(
                  (np.array([traj.parent,   traj.state,  traj.gamma,
                             traj.amplitude.real, traj.amplitude.imag]),
                   traj.last_spawn))

    # last, store everything about the surface
    for obj in traj.pes.avail_data():
        traj_data[obj] = traj.pes.get_data(obj)

    return traj_data

#
#
#
def package_centroid(cent, time):
    """Documentation to come"""

    cent_data = dict()

    # not an element in a trajectory, but necessary to 
    # uniquely tag everything
    cent_data['time']  = np.array([time],dtype='float')

    cent_data['global'] = np.concatenate((traj.parents, traj.states))

    # last, store everything about the surface
    for obj in cent.pes.avail_data():
        cent_data[obj] = cent.pes.get_data(obj)

    return cent_data

#
#
#
def default_blk_size(time):
    """Documentation to come"""

#    return int(1.1 * (glbl.propagate['simulation_time']-time) / 
#                      glbl.propagate['default_time_step'])

    return 500
