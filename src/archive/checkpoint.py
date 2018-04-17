"""
Routines for reading input files and writing log files.
"""
import sys
import os
import h5py
import numpy as np 
import src.parse.glbl as glbl
import src.basis.trajectory as trajectory
import src.basis.centroid as centroid

chkpt_file = ''

#
# called when an old checkpoint file is used to populate the
# contents of a bundle, if no time given, read last bundle
#
def read(master, file_name, time=None):
    """Documentation to come"""
    global chkpt_file
    
    # string name of checkpoint file
    chkpt_file = file_name.strip()
  
    # open chkpoint file
    chkpt.h5py.File(chkpt_file, "r", libver='latest')

    # create new bundle to store data
    nstates,dim,width,masses = read_bundle_attributes(chkpt)

    # add the trajectories to the bundle
    traj_list = chkpt.keys()
    for grp in traj_list:

        # if this is not a trajectory, move along    
        if grp == 'bundle':
            continue

        # if read_trajectory can't find the trajectory at the requested
        # time, return fasle
        new_traj = read_trajectory(chkpt, grp, time)

        if new_traj is not None:
            if isinstance(new_traj, trajectory.Trajectory):
                master.add_trajectory(new_traj)
            elif isinstance(new_traj, centroid.Centroid):
                ij = [new_traj.parent[0], new_traj.parent[1]]
                master.cent[ij[0]][ij[1]] = new_traj
                master.cent[ij[1]][ij[0]] = new_traj
            else:
                sys.exit("ERROR: object read from chkpt not Trajectory or Centroid")

    # now update the bundle level data
    status = read_bundle_data(chkpt, master, time)
    if not status:
        sys.exit('ERROR: bundle data from time='+str(time)+' could'+
                ' not be located in '+chkpt_file)

    # close checkpoint file
    chkpt.close()

    return

#
#
#
def write(master, file_name=None):
    """Documentation to come"""
    global chkpt_file

    # default is to use file name from previous write
    if file_name is not None:
        chkpt_file = file_name.strip()

    # if this is the first time we're writing to the archive,
    # create the bundle data set and record the time-independent 
    # bundle definitions
    if not os.path.isfile(chkpt_file):
        print("creating checkpoint file="+str(file_name))
        create(chkpt_file, master) 

    # open checkpoint file
    chkpt = h5py.File(chkpt_file, "a", libver='latest')

    # this definition of time over-rules all others as far as writing
    # data is concerned.
    time = master.time

    #------------------------------------------
    # this data get written for all simulations

    # loop over trajectories in bundle
    for i in range(master.nalive):
        write_trajectory(chkpt, time, master.traj[master.alive[i]])

    # ...now write wave function level information
    write_bundle_data(chkpt, time, master)

    #---------------------------------------------
    # what gets written here depends on simulation
    # parameters
    if glbl.integrals.require_centroids:
        for i in range(master.nalive):
            for j in range(i):
                write_trajectory(chkpt, time, master.cent[i][j])

    chkpt.close()

    return

#------------------------------------------------------------------------------------
#
# Should not be called outside the module
#

#
# called when a new checkpoint file is created
#
def create(file_name, master):
    """Documentation to come"""

    # create chkpoint file
    chkpt = h5py.File(file_name, "w", libver='latest')

    # create and store time-indpendent data
    write_bundle_attributes(chkpt, master)

    # close following initialization
    chkpt.close()

    return

#
#
#
def write_trajectory(chkpt, time, traj):
    """Documentation to come"""

    # open the trajectory file
    t_label = str(traj.label)
    n_rows  = default_blk_size(time)
    t_data  = package_trajectory(time, traj)
    resize  = False

    # if trajectory group already exists, just append current
    # time information to existing datasets
    if t_label in chkpt:
        current_row = chkpt[t_label].attrs['current_time'] + 1

        if current_row > chkpt[t_label].attrs['n_rows']:
            resize = True

        for data_label in t_data.keys():
            dset = t_label+'/'+data_label
            if resize:
                new_size = chkpt[t_label].attrs['n_rows'] + n_rows 
                d_shape  = (new_size,) + t_data[data_label].shape
                chkpt[dset].resize(d_shape)
                chkpt[dset].attrs['n_rows'] = new_size

            chkpt[dset][current_row] = t_data[data_label]

    # if this is the first time we're trying to write this trajectory,
    # create a new data group, and new data sets with reasonble default sizes
    else:
        chkpt.create_group(t_label)
        current_row                          = 0
        chkpt[t_label].attrs['current_time'] = current_row
        chkpt[t_label].attrs['n_rows']       = n_rows
        chkpt[t_label].attrs['isTrajectory'] = isinstance(traj, trajectory.Trajectory) 
        chkpt[t_label].attrs['isCentroid']   = isinstance(traj, centroid.Centroid)
 
        # store surface information from trajectory
        for data_label in t_data.keys():
            dset = t_label+'/'+data_label
            d_shape   = (n_rows,) + t_data[data_label].shape
            max_shape = (None,) + t_data[data_label].shape
            d_type    = t_data[data_label].dtype
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, dtype=d_type, compression="gzip")
            chkpt[dset][current_row] = t_data[data_label]

    return

#
#
#
def read_trajectory(chkpt, t_label, time=None):
    """Documentation to come"""

    read_row = get_time_index(chkpt[t_label], time)

    if read_row is None:
        return None 

    # get data necessary to define a trajectory
    nstates,dim,width,mass = read_bundle_attributes(chkpt)
  
    if chkpt[t_label].attrs['isTrajectory']:
        # create new trajectory with the appropriate label
        traj = trajectory.Trajectory(nstates, dim, width=width, 
                                     mass=mass, label=int(label))
    elif chkpt[t_label].attrs['isCentroid']:
        traj = centroid.Centroid(nstates=nstates, dim=dim, width=width,
                                 label=int(label))
    else: 
        sys.exit('unable to read trajectory: neither trajectory nor centroid')


    # populate the surface object in the trajectory
    pes = surface.Surface()
    for data_label in chkpt[t_label].keys():  
        if data_label == 'global' or data_label == 'time':
            continue
        dset = chkpt[label+'/'+data_label]
        pes.add_data(data_label, dset[read_row])

    # set information about the trajectory itself
    if chkpt[t_label].attrs['isTrajectory']:
        data_row = chkpt[t_label+'/global'][read_row]
        [traj.parent, traj.state, traj.gamma, amp_real, amp_imag] = data_row[0:4]
        traj.update_amplitude(amp_real+1.j*amp_imag)
        traj.last_spawn = data_row[5:]
    elif chkpt[t_label].attrs['isCentroid']:
        [traj.parent[0],  traj.parent[1], 
         traj.pstates[0], traj.pstates[1]] = chkpt[t_label+'/global'][read_row]        

    traj.update_pes_info(pes)
    traj.update_x(traj.pes.get_data('geom'))
    traj.update_p(traj.pes.get_data('momentum'))

    return traj 

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
    if dt[read_row] > 0.1*glbl.propagate['default_time_step']:
        read_row = None

    return read_row
     

#
#
#
def write_bundle_data(chkpt, time, master):
    """Documentation to come"""
    dset_static = ['time', 'pop', 'energy']
    dset_matrix = ['overlap', 't_overlap', 'kinetic', 'potential',
                   'sdot',    'heff']

    b_data      = package_bundle(master)
    n_traj      = master.n_traj()
    n_rows      = default_blk_size(time)
    current_row = chkpt['bundle'].attrs['current_time']
    resize_all  = False
    resize_mat  = False
    new_rows    = chkpt['bundle'].attrs['n_rows']

    # update the current row index (same for all data sets)
    current_row = chkpt['bundle'].attrs['current_time'] + 1

    if current_row > chkpt['bundle'].attrs['n_rows']:
        resize_all = True

    if n_traj != chkpt['bundle'].attrs['n_traj']:
        resize_mat = True

    # first write items with time-independent dimensions
    for data_label in dset_static + dset_matrix:

        dset = 'bundle/'+data_label

        if data_label in chkpt['bundle']:
            if resize_all:
                new_rows = chkpt['bundle'].attrs['n_rows'] + n_rows

            if resize_all or (data_label in dset_matrix and resize_mat):
                d_shape  = (new_rows,) + b_data[data_label].shape
                chkpt[dset].resize(d_shape)
            chkpt[dset][current_row] = b_data[data_label]

        # if this is the first time we're trying to write this bundle,
        # create a new datasets with reasonble default sizes
        else:

            d_shape   = (new_rows,) +  b_data[data_label].shape
            max_shape = (None,) + b_data[data_label].shape
            d_type    = b_data[data_label].dtype
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, dtype=d_type, compression="gzip")
#            print("d_shape, max_shape="+str(d_shape)+" "+str(max_shape))
#            print("data="+str(b_data[data_label]))
#            print("label="+str(data_label))
            chkpt[dset][current_row] = b_data[data_label]

    # only update this (potentially) after all data sets have been update
    chkpt[dset].attrs['n_rows'] = new_rows

    return

#
#
#
def read_bundle_data(chkpt, master, time):
    """Documentation to come"""
    read_row = get_time_index(chkpt['bundle'], time)

    if read_row is None:
        return False

    master.time       = chkpt['bundle/time'][read_row]
    master.S          = chkpt['bundle/overlap'][read_row]
    master.traj_ovrlp = chkpt['bundle/t_overlap'][read_row]
    master.T          = chkpt['bundle/kinetic'][read_row]
    master.V          = chkpt['bundle/potential'][read_row]
    master.Sdot       = chkpt['bundle/sdot'][read_row]
    master.Heff       = chkpt['bundle/heff'][read_row]

    return True

#
#
#
def write_bundle_attributes(chkpt, master):
    """Documentation to come"""

    traj0 = master.traj[0]
    dim   = traj0.dim

    if 'bundle' not in chkpt:
        chkpt.create_group('bundle')

    chkpt['bundle'].attrs['nstates'] = master.nstates
    chkpt['bundle'].attrs['dim']     = traj0.dim
    chkpt['bundle'].attrs['widths']  = traj0.widths()
    chkpt['bundle'].attrs['masses']  = traj0.masses()

    chkpt['bundle'].attrs['current_time'] = -1
    chkpt['bundle'].attrs['n_rows']       = default_blk_size(master.time)
    chkpt['bundle'].attrs['n_traj']       = master.n_traj()
 

    return

#
#
#
def read_bundle_attributes(chkpt):
    """Documentation to come"""
    nstates = chkpt['bundle'].attrs['nstates']
    dim     = chkpt['bundle'].attrs['dim']
    width   = chkpt['bundle'].attrs['widths']
    masses  = chkpt['bundle'].attrs['masses']

    return nstates,dim,width,masses

#
#
#
def package_trajectory(time, traj):
    """Documentation to come"""

    traj_data = dict()

    # not an element in a trajectory, but necessary to 
    # uniquely tag everything
    traj_data['time']  = np.array([time],dtype='float')

    # store information about the trajectory itself
    if isinstance(traj, trajectory.Trajectory):
        t_global = np.concatenate(
                   (np.array([traj.parent,   traj.state,  traj.gamma,
                             traj.amplitude.real, traj.amplitude.imag]),
                   traj.last_spawn))
    elif isinstance(traj, centroid.Centroid):
        t_global = np.concatenate((traj.parent, traj.pstates))
    else:
        sys.exit('NOMAD does\'t know how to package trajectory object')
 
    traj_data['global'] = t_global

    # last, store everything about the surface
    for obj in traj.pes.avail_data():
        traj_data[obj] = traj.pes.get_data(obj)

    return traj_data

#
#
#
def package_bundle(master):
    """Documentation to come"""

    bundle_data = dict()

    # dimensions of these objects are not time-dependent 
    bundle_data['time']  = np.array([master.time])
    bundle_data['pop']   = master.pop()
    bundle_data['energy']= np.array([master.pot_quantum(),   master.kin_quantum(), 
                                     master.pot_classical(), master.kin_classical()])
 
    # dimensions of these objects are time-dependent
    bundle_data['overlap']   = master.S
    bundle_data['t_overlap'] = master.traj_ovrlp
    bundle_data['kinetic']   = master.T
    bundle_data['potential'] = master.V
    bundle_data['sdot']      = master.Sdot
    bundle_data['heff']      = master.Heff

    return bundle_data
#
#
#
def default_blk_size(time):
    """Documentation to come"""

    return int(1.1 * (glbl.propagate['simulation_time']-time) / 
                      glbl.propagate['default_time_step'])
