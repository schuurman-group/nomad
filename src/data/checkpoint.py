"""
Routines for reading input files and writing log files.
"""
import h5py
import numpy as np 
import parse.glbl as glbl
import basis.Trajectory as Trajectory
import basis.Centroid as Centroid
import basis.Bundle as Bundle

chkpt_file = ''

#
# called when a new checkpoint file is created
#
def create(file_name, bundle):
"""Documentation to come"""
    global chkpt_file

    # name of checkpoint file
    chkpt_file = file_name.strip()

    # create chkpoint file
    chkpt = h5py.File(chkpt_file, "w", libver='latest')    

    # create and store time-indpendent data
    set_bundle_attributes(chkpt, bundle)

    # close following initialization
    chkpt.close()

    return

#
# called when an old checkpoint file is used to populate the
# contents of a bundle, if no time given, read last bundle
#
def read(file_name, time=None):
"""Documentation to come"""
    global chkpt_file
    
    # string name of checkpoint file
    chkpt_file = file_name.strip()
  
    # open chkpoint file
    chkpt.h5py.File(chkpt_file, "r", libver='latest')

    # create new bundle to store data
    nstates,dim,width,masses = read_bundle_attributes(chkpt)
    bundle  = Bundle(nstates)

    # add the trajectories to the bundle
    traj_list = chkpt.keys()
    for grp in traj_list:

        # if this is not a trajectory, move along    
        if grp == 'bundle':
            continue

        # if read_trajectory can't find the trajectory at the requested
        # time, return fasle
        new_traj = read_trajectory(ckhpt, grp, time)

        if new_traj not None:
            if new_traj is trajectory.Trajectory:
                bundle.add_trajectory(new_traj)
            else:
                ij = [new_traj.parent[0], new_traj.parent[1]]
                bundle.cent[ij[0]][ij[1]] = new_traj
                bundle.cent[ij[1]][ij[0]] = new_traj

    # now update the bundle level data
    status = read_bundle_data(chkpt, bundle, time)
    if not status:
        os.exit('ERROR: bundle data from time='+str(time)+' could
                 not be located in '+chkpt_file)

    # close checkpoint file
    chkpt.close()

    return

#
#
#
def write(bundle):
"""Documentation to come"""
    global chkpt_file

    # open checkpoint file
    chkpt = h5py.File(chkpt_file, "a", libver='latest')

    time = bundle.time

    #------------------------------------------
    # this data get written for all simulations

    # loop over trajectories in bundle
    for i in range(bundle.nalive):
        write_trajectory(chkpt, time, bunde.traj[bundle.alive[i]])

    # ...now write wave function level information
    write_bundle_data(chkpt, time, bundle)

    #---------------------------------------------
    # what gets written here depends on simulation
    # parameters
    if parse.glbl.integrals.require_centroids:
        for i in range(bundle.nalive):
            for j in range(i):
                write_trajectory(chkpt, time, bundle.cent[i][j])

    chkpt.close()

    return

#------------------------------------------------------------------------------------
#
# Should not be called outside the module
#

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

        for data_label in t_data.keys()
            dset = t_label+'/'+data_label
            if resize:
                new_size = chkpt[t_label].attrs['n_rows'] + n_rows 
                d_shape  = (new_size,) + t_data[data_label].shape
                chkpt[dset].resize(d_shape)
                chkpt[dset].attrs['n_rows'] = new_size
            chkpt[dset][current_row,:] = t_data[data_label]

    # if this is the first time we're trying to write this trajectory,
    # create a new data group, and new data sets with reasonble default sizes
    else:
        chkpt.create_group(t_label)
        current_row                          = 0
        chkpt[t_label].attrs['current_time'] = current_row
        chkpt[t_label].attrs['n_rows']       = n_rows
        chkpt[t_label].attrs['isTrajectory'] = traj is trajectory.Trajectory 

        # store surface information from trajectory
        for data_label in t_data.keys():
            dset = traj+'/'+data_label
            d_shape   = (n_rows,) + t_data[data_label].shape
            max_shape = (None,) + t_data[data_label].shape
            chkpt.create_dataset(dset,shape=d_shape, maxshape=max_shape,'f')
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
  
    isTraj = chkpt[t_label].attrs['isTrajectory']
    if isTraj:
        # create new trajectory with the appropriate label
        traj = trajectory.Trajectory(nstates, dim, width=width, 
                                     mass=mass, label=int(label))
    else:
        traj = centroid.Centroid(nstates=nstates, dim=dim, width=width,
                                 label=int(label))

    # populate the surface object in the trajectory
    pes = surface.Surface()
    for data_label in chkpt[t_label].keys()  
        if data_label == 'global' or data_label == 'time':
            continue
        dset = chkpt[label+'/'+data_label]
        pes.add_item(data_label, dset[read_row])

    # set information about the trajectory itself
    if isTraj:
        [traj.parent, traj.state, 
         traj.gamma,  traj.last_spawn,
         amp_real,    amp_imag] = chkpt[t_label+'/global'][read_row,:]
        traj.update_amplitude(amp_real+1.j*amp_imag)
    else:
        [traj.parent[0],  traj.parent[1], 
         traj.pstates[0], traj.pstates[1]] = chkpt[t_label+'/global'][read_row,:]        

    traj.update_pes_info(pes)
    traj.update_x(traj.pes.get_item('geom'))
    traj.update_p(traj.pes.get_item('momentum'))

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
def write_bundle_data(chkpt, time, bundle)
"""Documentation to come"""
    dset_static = ['time', 'pop', 'energy']
    dset_matrix = ['overlap', 't_overlap', 'kinetic', 'potential',
                   'sdot',    'heff']

    # first write
    b_data      = package_bundle(bundle)
    n_traj      = bundle.n_traj()
    n_rows      = default_blk_size(time)
    current_row = chkpt['bundle'].attrs['current_time']
    resize_all  = False
    resize_mat  = False
    new_rows    = chkpt['bundle'].attrs['n_rows']

    if current_row > chkpt['bundle'].attrs['n_rows']:
        resize_all = True

    if n_traj != chkpt['bundle'].attrs['n_traj']:
        resize_mat = True

    # update the current row index (same for all data sets)
    current_row = chkpt[t_label].attrs['current_time'] + 1

    # first write items with time-independent dimensions
    for data_label in dset_static:

        dset = 'bundle/'+data_label

        if data_label in chkpt['bundle']:
            if resize_all:
                new_rows = chkpt['bundle'].attrs['n_rows'] + n_rows

            if resize_all or (data_label in dset_matrix and resize_mat):
                d_shape  = (new_rows,) + b_data[data_label].shape
                chkpt[dset].resize(d_shape)

            chkpt[dset][current_row,:] = b_data[data_label]

        # if this is the first time we're trying to write this bundle,
        # create a new datasets with reasonble default sizes
        else:

            d_shape   = (new_rows,) +  b_data[data_label].shape
            max_shape = (None,) + b_data[data_label].shape
`           chkpt.create_dataset(dset,shape=d_shape,maxshape=max_shape,'f')
            chkpt[dset][current_row] = b_data[data_label]

    # only update this (potentially) after all data sets have been update
    chkpt[dset].attrs['n_rows'] = new_rows

    return

#
#
#
def read_bundle_data(chkpt, bundle, time)
"""Documentation to come"""
    read_row = get_time_index(chkpt['bundle'], time)

    if read_row is None:
        return False

    bundle.time       = chkpt['bundle/time'][read_row]
    bundle.S          = chkpt['bundle/overlap'][read_row,:,:]
    bundle.traj_ovrlp = chkpt['bundle/t_overlap'][read_row,:,:]
    bundle.T          = chkpt['bundle/kinetic'][read_row,:,:]
    bundle.V          = chkpt['bundle/potential'][read_row,:,:]
    bundle.Sdot       = chkpt['bundle/sdot'][read_row,:,:]
    bundle.Heff       = chkpt['bundle/heff'][read_row,:,:]

    return True

#
#
#
def write_bundle_attributes(chkpt, bundle):
"""Documentation to come"""

    traj0 = bundle.traj[0]
    dim   = traj0.dim

    if 'bundle' not in chkpt:
        chkpt.create_group('bundle')

    chkpt['bundle'].attrs['nstates'] = bundle.nstates
    chkpt['bundle'].attrs['dim']     = traj0.dim
    chkpt['bundle'].attrs['widths']  = traj0.widths()
    chkpt['bundle'].attrs['masses']  = traj0.masses()

    chkpt['bundle'].attrs['current_time'] = -1
    chkpt['bundle'].attrs['n_rows']       = default_blk_size(bundle.time)
    ckhpt['bundle'].attrs['n_traj']       = bundle.n_traj()
 

    return

#
#
#
def read_bundle_attributes(chkpt):

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
    traj_data['time']  = time

    # store information about the trajectory itself
    if traj is centroid.Centroid:
        t_global = np.concatenate(traj.parent, traj.pstates)
    else:
        t_global = np.array([traj.parent,         traj.state,  
                             traj.gamma,          traj.amplitude.real, 
                             traj.amplitude.imag, traj.last_spawn])
    traj_data['global'] = t_global_

    # last, store everything about the surface
    for obj in traj.pes.avail_data():
        traj_data[obj] = traj.pes.get_item(obj)

    return traj_data

#
#
#
def package_bundle(bundle):
"""Documentation to come"""

    bundle_data = dict()

    # dimensions of these objects are not time-dependent 
    bundle_data['time']  = bundle.time
    bundle_data['pop']   = bundle.pop
    bundle_data['energy']= [bundle.pot_quantum(),   bundle.kin_quantum(), 
                            bundle.pot_classical(), bundle.kin_classical()]
 
    # dimensions of these objects are time-dependent
    bundle_data['overlap']   = bundle.S
    bundle_data['t_overlap'] = bundle.traj_ovrlp
    bundle_data['kinetic']   = bundle.T
    bundle_data['potential'] = bundle.V
    bundle_data['sdot']      = bundle.Sdot
    bundle_data['heff']      = bundle.Heff

    return bundle_data
#
#
#
def default_blk_size(time):
"""Documentation to come"""

    return int(1.1*(glbl.simulation_time-time)/glbl.default_time_step)
