"""
Routines for reading input files and writing log files.
"""
import os
import h5py
import numpy as np
import nomad.parse.glbl as glbl
import nomad.integrals.integral as integral
import nomad.basis.wavefunction as wavefunction
import nomad.basis.trajectory as trajectory
import nomad.archive.surface as surface


chkpt_file = ''


def archive_simulation(wfn, integrals=None, time=None, file_name=None):
    """Documentation to come"""
    write(wfn, file_name=file_name, time=time)
    if integrals is not None:
        write(integrals, file_name=file_name, time=time)


def retrieve_simulation(wfn, integrals=None, time=None, file_name=None):
    """Dochumentation to come"""
    read(wfn, file_name=file_name, time=time)
    if integrals is not None:
        read(integrals, file_name=file_name, time=time)


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
            raise TypeError('chkpt file must be created with wavefunction object.')

    # open checkpoint file
    chkpt = h5py.File(chkpt_file, 'a', libver='latest')

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
    else:
        raise TypeError('data_obj: '+str(data_obj)+' is not recognized by checkpoint.write')

    chkpt.close()


def read(data_obj, file_name, time=None):
    """Reads the checkpoint file.

    Called when an old checkpoint file is used to populate the
    contents of a bundle, if no time given, read last bundle.
    """
    global chkpt_file

    # string name of checkpoint file
    chkpt_file = file_name.strip()

    # open chkpoint file
    chkpt = h5py.File(chkpt_file, 'r', libver='latest')

    if isinstance(data_obj, wavefunction.Wavefunction):

        read_wavefunction(chkpt, data_obj, time)

    # if this is an integral objects, it's going to want to load the centroid data
    # from file
    elif isinstance(data_obj, integral.Integral):
        read_integral(chkpt, data_obj, time)
    else:
        raise TypeError('data_obj: '+str(data_obj)+' is not recognized by checkpoint.read')

    # close checkpoint file
    chkpt.close()


def time_steps(grp_name, file_name=None):
    """Documentation to come"""
    global chkpt_file

    # default is to use file name from previous write
    if file_name is not None:
        chkpt_file = file_name.strip()

    # open chkpoint file
    chkpt = h5py.File(chkpt_file, 'r', libver='latest')

    # if the group name is in the checkpoint file, return
    # the associated time array
    if grp_name in chkpt:
        if 'current_row' in chkpt[grp_name].attrs:
            current_row = chkpt[grp_name].attrs['current_row'] + 1
        else:
            current_row = len(chkpt[grp_name+'/time'][:])
        steps = chkpt[grp_name+'/time'][:current_row, 0]
    #else abort
    else:
        raise ValueError('grp_name: '+str(grp_name)+' not present in checkpoint file')

    chkpt.close()

    return steps


#------------------------------------------------------------------------------------
#
# Should not be called outside the module
#
#------------------------------------------------------------------------------------
def create(file_name, wfn):
    """Creates a new checkpoint file."""
    # create chkpoint file
    chkpt = h5py.File(file_name, 'w', libver='latest')

    chkpt.create_group('wavefunction')
    chkpt['wavefunction'].attrs['current_row'] = -1
    chkpt['wavefunction'].attrs['n_rows']      = -1

    chkpt.create_group('integral')
    chkpt['integral'].attrs['current_row']     = -1
    chkpt['integral'].attrs['n_rows']          = -1

    traj0 = wfn.traj[0]
    chkpt.create_group('simulation')
    chkpt['simulation'].attrs['nstates']  = traj0.nstates
    chkpt['simulation'].attrs['dim']      = traj0.dim
    chkpt['simulation'].attrs['widths']   = traj0.widths()
    chkpt['simulation'].attrs['masses']   = traj0.masses()
    chkpt['simulation'].attrs['kecoef']   = traj0.kecoef

    # close following initialization
    chkpt.close()


def write_wavefunction(chkpt, wfn, time):
    """Documentation to come"""
    wfn_data = package_wfn(wfn)
    n_traj   = wfn.n_traj()
    n_blk    = default_blk_size(time)
    resize   = False

    # update the current row index (same for all data sets)
    current_row = chkpt['wavefunction'].attrs['current_row'] + 1

    if current_row > chkpt['wavefunction'].attrs['n_rows']:
        resize = True
        chkpt['wavefunction'].attrs['n_rows'] += n_blk
    n_rows = chkpt['wavefunction'].attrs['n_rows']

    # first write items with time-independent dimensions
    for data_label in wfn_data.keys():
        dset = 'wavefunction/'+data_label

        if dset in chkpt:
            if resize:
                d_shape  = (n_rows,) + wfn_data[data_label].shape
                chkpt[dset].resize(d_shape)
            chkpt[dset][current_row] = wfn_data[data_label]

        # if this is the first time we're trying to write this bundle,
        # create a new datasets with reasonble default sizes
        else:
            d_shape   = (n_rows,) +  wfn_data[data_label].shape
            max_shape = (None,)   + wfn_data[data_label].shape
            d_type    = wfn_data[data_label].dtype
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, dtype=d_type, compression="gzip")
            chkpt[dset][current_row] = wfn_data[data_label]

    # now step through and write trajectories
    for i in range(n_traj):
        write_trajectory(chkpt, wfn.traj[i], time)

    chkpt['wavefunction'].attrs['current_row'] = current_row


def write_integral(chkpt, integral, time):
    """Documentation to come"""
    int_data = package_integral(integral, time)
    n_blk    = default_blk_size(time)
    resize   = False

    # update the current row index (same for all data sets)
    current_row = chkpt['integral'].attrs['current_row'] + 1

    if current_row > chkpt['integral'].attrs['n_rows']:
        resize   = True
        chkpt['integral'].attrs['n_rows'] += n_blk
    n_rows = chkpt['integral'].attrs['n_rows']

    # first write items with time-independent dimensions
    for data_label in int_data.keys():
        dset = 'integral/'+data_label

        if dset in chkpt:
            if resize:
                d_shape  = (n_rows,) + int_data[data_label].shape
                chkpt[dset].resize(d_shape)
            chkpt[dset][current_row] = int_data[data_label]

        # if this is the first time we're trying to write this bundle,
        # create a new datasets with reasonble default sizes
        else:
            d_shape   = (n_rows,) + int_data[data_label].shape
            max_shape = (None,)   + int_data[data_label].shape
            d_type    = int_data[data_label].dtype
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, dtype=d_type, compression="gzip")
            chkpt[dset][current_row] = int_data[data_label]

    # now step through centroids, if they're present
    if integral.require_centroids:
        for i in range(len(integral.centroid)):
            for j in range(i):
                 if integral.centroid[i][j] is not None:
                     write_centroid(chkpt, integral.centroid[i][j], time)

    chkpt['integral'].attrs['current_row'] = current_row


def write_trajectory(chkpt, traj, time):
    """Documentation to come"""
    # open the trajectory file
    t_data  = package_trajectory(traj, time)
    t_label = str(traj.label)
    n_blk   = default_blk_size(time)
    resize  = False

    # if trajectory group already exists, just append current
    # time information to existing datasets
    t_grp = 'wavefunction/'+t_label

    if t_grp in chkpt:

        chkpt[t_grp].attrs['current_row'] += 1
        current_row = chkpt[t_grp].attrs['current_row']

        if (current_row > chkpt[t_grp].attrs['n_rows']):
            resize = True
            chkpt[t_grp].attrs['n_rows'] += n_blk

        for data_label in t_data.keys():
            dset = t_grp+'/'+data_label
            if resize:
                d_shape  = (n_blk,) + t_data[data_label].shape
                chkpt[dset].resize(d_shape)

            chkpt[dset][current_row] = t_data[data_label]

    # if this is the first time we're trying to write this trajectory,
    # create a new data group, and new data sets with reasonble default sizes
    else:

        chkpt.create_group(t_grp)
        current_row                       = 0
        chkpt[t_grp].attrs['current_row'] = current_row
        chkpt[t_grp].attrs['n_rows']      = n_blk

        # store surface information from trajectory
        for data_label in t_data.keys():
            dset = t_grp+'/'+data_label
            d_shape   = (n_blk,) + t_data[data_label].shape
            max_shape = (None,)   + t_data[data_label].shape
            d_type    = t_data[data_label].dtype
            if d_type.type is np.unicode_:
                d_type = h5py.special_dtype(vlen=str)
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, dtype=d_type, compression="gzip")
            chkpt[dset][current_row] = t_data[data_label]


def write_centroid(chkpt, cent, time):
    """Documentation to come"""
    # open the trajectory file
    c_data  = package_centroid(cent, time)
    c_label = str(cent.label)
    n_blk   = default_blk_size(time)
    resize  = False

    # if trajectory group already exists, just append current
    # time information to existing datasets
    c_grp = 'integral/'+c_label

    if c_grp in chkpt:

        chkpt[c_grp].attrs['current_row'] += 1
        current_row = chkpt[c_grp].attrs['current_row']

        if current_row > chkpt[c_grp].attrs['n_rows']:
            resize = True
            chkpt[c_grp].attrs['n_rows'] += n_blk

        for data_label in c_data.keys():
            dset = c_grp+'/'+data_label
            if resize:
                d_shape  = (n_blk,) + c_data[data_label].shape
                chkpt[dset].resize(d_shape)

            chkpt[dset][current_row] = c_data[data_label]

    # if this is the first time we're trying to write this trajectory,
    # create a new data group, and new data sets with reasonble default sizes
    else:

        chkpt.create_group(c_grp)
        current_row                       = 0
        chkpt[c_grp].attrs['current_row'] = current_row
        chkpt[c_grp].attrs['n_rows']      = n_blk

        # store surface information from trajectory
        for data_label in c_data.keys():
            dset = c_grp+'/'+data_label
            d_shape   = (n_blk,) + c_data[data_label].shape
            max_shape = (None,)   + c_data[data_label].shape
            d_type    = c_data[data_label].dtype
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, dtype=d_type, compression="gzip")
            chkpt[dset][current_row] = c_data[data_label]


def read_wavefunction(chkpt, wfn, time):
    """Documentation to come"""
    nstates = chkpt['simulation'].attrs['nstates']
    dim     = chkpt['simulation'].attrs['dim']
    widths  = chkpt['simulation'].attrs['widths']
    masses  = chkpt['simulation'].attrs['masses']
    kecoef  = chkpt['simulation'].attrs['kecoef']

    # check that we have the desired time:

    read_row = get_time_index(chkpt, 'wavefunction', time)

    if read_row is None:
        raise ValueError('time='+str(time)+' requested, but not in checkpoint file')

    # dimensions of these objects are not time-dependent
    wfn.nstates = nstates
    wfn.time    = chkpt['wavefunction/time'][read_row,0]

    for label in chkpt['wavefunction']:

        if (label=='time' or label=='pop' or label=='energy'):
            continue

        t_grp = 'wavefunction/'+label
        t_row = get_time_index(chkpt, t_grp, time)

        if t_row is None:
            continue

        new_traj = trajectory.Trajectory(nstates, dim,
                                         width=widths,
                                         mass=masses,
                                         label=label,
                                         kecoef=kecoef)
        read_trajectory(chkpt, new_traj, t_grp, t_row)
        wfn.add_trajectory(new_traj.copy())


def read_integral(chkpt, integral, time):
    """Documentation to come"""
    nstates = chkpt['simulation'].attrs['nstates']
    dim     = chkpt['simulation'].attrs['dim']
    widths  = chkpt['simulation'].attrs['widths']
    masses  = chkpt['simulation'].attrs['masses']

    # check that we have the desired time:
    read_row = get_time_index(chkpt, 'integral', time)

    if read_row is None:
        raise ValueError('time='+str(time)+' requested, but not in checkpoint file')

    if integral.require_centroids:
        for label in chkpt['integral']:

            if label == 'time':
                continue

            c_grp = 'integral/'+label
            c_row = get_time_index(chkpt, c_grp, time)

            if c_row is None:
                continue

            new_cent = integral.Centroid(nstates=nstates, dim=dim, width=widths)
            read_centroid(chkpt, new_cent, c_grp, c_row)
            integral.add_centroid(new_cent)


def read_trajectory(chkpt, new_traj, t_grp, t_row):
    """Documentation to come"""
    # populate the surface object in the trajectory

    # set information about the trajectory itself
    data_row = chkpt[t_grp+'/glbl'][t_row]
    [parent, state, new_traj.gamma, amp_real, amp_imag] = data_row[0:5]

    pes = surface.Surface()
    for data_label in chkpt[t_grp].keys():
        if pes.valid_data(data_label):
            dset = chkpt[t_grp+'/'+data_label]
            pes.add_data(data_label, dset[t_row])

    # currently, momentum has to be read in separately
    momt    = chkpt[t_grp+'/momentum'][t_row]

    new_traj.state  = int(state)
    new_traj.parent = int(parent)
    new_traj.update_amplitude(amp_real+1.j*amp_imag)
    new_traj.last_spawn = data_row[5:]

    new_traj.update_pes_info(pes)
    new_traj.update_x(new_traj.pes.get_data('geom'))
    new_traj.update_p(momt)


def read_centroid(chkpt, new_cent, c_grp, c_row):
    """Documentation to come"""

    # set information about the trajectory itself
    parent = [0.,0.]
    states = [0.,0.]
    [parent[0], parent[1], states[0], states[1]] = chkpt[c_grp+'/glbl'][c_row]

    # populate the surface object in the trajectory
    pes = surface.Surface()
    for data_label in chkpt[c_grp].keys():
        if pes.valid_data(data_label):
            dset = chkpt[c_grp+'/'+data_label]
            pes.add_data(data_label, dset[c_row])

    # currently, momentum has to be read in separately
    momt    = chkpt[c_grp+'/momentum'][t_row]

    new_cent.parents = int(parent)
    new_cent.states  = int(states)

    new_cent.update_pes_info(pes)
    new_cent.update_x(new_cent.pes.get_data('geom'))
    new_cent.update_p(momt)


def get_time_index(chkpt, grp_name, time):
    """Documentation to come"""
    time_vals = time_steps(grp_name)

    if time is None:
        return chkpt[grp_name].attrs['current_row']

    dt       = np.absolute(time_vals - time)
    read_row = np.argmin(dt)

    # this tolerance is arbitrary: check if the matched time
    # is further than 0.5 * timestep to the next closest times,
    # else we don't have a match
    match_chk = []
    if read_row > 0:
        match_chk.extend([time_vals[read_row]-time_vals[read_row-1]])
    if read_row < len(time_vals)-1:
        match_chk.extend([time_vals[read_row+1]-time_vals[read_row]])

    if dt[read_row] > 0.5*min(match_chk):
        read_row = None

    return read_row

def package_wfn(wfn):
    """Documentation to come"""
    # dimensions of these objects are not time-dependent
    wfn_data = dict(
        time   = np.array([wfn.time], dtype='float'),
        pop    = np.array(wfn.pop()),
        energy = np.array([wfn.pot_quantum(),   wfn.kin_quantum(),
                           wfn.pot_classical(), wfn.kin_classical()])
                    )

    return wfn_data


def package_integral(integral, time):
    """Documentation to come"""
    int_data = dict(
        time = np.array([time],dtype='float')
                    )
    return int_data


def package_trajectory(traj, time):
    """Documentation to come"""
    # time is not an element in a trajectory, but necessary to
    # uniquely tag everything
    traj_data = dict(
        time     = np.array([time],dtype='float'),
        glbl     = np.concatenate((np.array([traj.parent, traj.state, traj.gamma,
                                 traj.amplitude.real, traj.amplitude.imag]),
                                 traj.last_spawn)),
        momentum = traj.p()
                    )

    # store everything about the surface
    for obj in traj.pes.avail_data():
        traj_data[obj] = traj.pes.get_data(obj)

    return traj_data


def package_centroid(cent, time):
    """Documentation to come"""
    cent_data = dict(
        time     = np.array([time],dtype='float'),
        glbl     = np.concatenate((cent.parents, cent.states)),
        momentum = cent.p()
                     )

    # last, store everything about the surface
    for obj in cent.pes.avail_data():
        cent_data[obj] = cent.pes.get_data(obj)

    return cent_data


def default_blk_size(time):
    """Documentation to come"""
    return int(2.1 * (glbl.propagate['simulation_time']-time) /
                      glbl.propagate['default_time_step'])
