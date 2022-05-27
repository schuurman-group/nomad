"""
Routines for reading input files and writing log files.
"""
import os
import sys
import math
import h5py
import ast as ast
import shutil as shutil
import numpy as np
import nomad.core.glbl as glbl
import nomad.core.wavefunction as wavefunction
import nomad.core.trajectory as trajectory
import nomad.core.surface as surface
import nomad.core.matrices as matrices
import nomad.integrals.integral as integral
import nomad.integrals.centroid as centroid


np.set_printoptions(threshold = np.inf)
tkeys       = ['traj', 'poten', 'grad', 'coup', 'hessian',
               'dipole', 'tr_dipole', 'sec_mom', 'atom_pop']
bkeys       = ['pop', 'energy', 'auto', 'spawn']
dump_header = dict()
dump_format = dict()
tfile_names = dict()
bfile_names = dict()


def archive_simulation(wfn, integrals, file_name=None, create_new=False):
    """Documentation to come"""
    # default is to use file name from previous write
    if file_name is not None:
        glbl.paths['chkpt_file'] = file_name.strip()

    # if this is the first time we're writing to the archive,
    # create the bundle data set and record the time-independent
    # bundle definitions
    if not os.path.isfile(glbl.paths['chkpt_file']) or create_new:
        create(glbl.paths['chkpt_file'], wfn, integrals)

    # pull the time from the wave function to uniquely timestamp
    # entries
    time = wfn.time

    # open checkpoint file
    chkpt = h5py.File(glbl.paths['chkpt_file'], 'a', libver='latest')

    # write the wave function to file
    write_wavefunction(chkpt, wfn, time)

    # write the integral information to file
    if integrals is not None:
        write_integral(chkpt, integrals, time)

    # close the chkpt file
    chkpt.close()

def retrieve_simulation(time=None, file_name=None, key_words=False,
                        reset_rows=False, save_paths=True):
    """Documentation to come"""
    # default is to use file name from previous write
    if file_name is not None:
        glbl.paths['chkpt_file'] = file_name.strip()

    # open chkpoint file
    chkpt = h5py.File(glbl.paths['chkpt_file'], 'r', libver='latest')

    if key_words:
        read_keywords(chkpt, save_paths=save_paths)

    # when restarting from arbitrary time, we may want to overwrite
    # subsequent time-data, if it exists
    if reset_rows:
        reset_datasets(chkpt, time)

    # read wave function information, including trajectories
    wfn = read_wavefunction(chkpt, time)

    # update the wfn specific data
    ints = read_integral(chkpt, time)

    if ints is not None:
        ints.update(wfn)
        pops = ints.pops(wfn)
        wfn.update_pop(pops)

    # close the checkpoint file
    chkpt.close()

    return wfn, ints

#
def merge_simulations(file_names=None, new_file=None):
    """Documentation to come"""

    if file_names is None:
        sys.exit('No files to merge. Exiting...')

    if new_file is None:
        sys.exit('No target file for merge named. Exiting...')

    for i in range(len(file_names)):
        if not os.path.isfile(file_names[i]):
            sys.exit('Cannot merge '+str(file_names[i])+
                     ': File does not exist. Exiting...')

    # we will copy the first file in file_names to the target,
    # then merge all subsequent files into that on
    shutil.copy(file_names[0], new_file)    
    target = h5py.File(new_file, 'a', libver='latest')   

    # make sure wavefunction groups satisfy naming convention
    # of "wavefunction.x", "integral.x" mostly for backwards compatability
    # we can likely delete this eventually
    lbl = 0
    for grp in target.keys():
        if grp == 'wavefunction': #this wfn does not have a label suffix
            while grp+'.'+str(lbl) in target.keys():
                lbl += 1
            target.move('wavefunction','wavefunction.'+str(lbl))
            if 'integral' in target.keys():
                target.move('integral','integral.'+str(lbl))

    wcnt = sum('wavefunction' in grp for grp in target.keys())
    icnt = sum('integral' in grp for grp in target.keys())
    for i in range(1,len(file_names)):
        chkpt = h5py.File(file_names[i], 'r', libver='latest')    
        for grp in chkpt:
            if 'wavefunction' in grp:
                while 'wavefunction.'+str(wcnt) in target:
                    wcnt += 1
                chkpt.copy(grp, target, name='wavefunction.'+str(wcnt))
            elif 'integral' in grp:
                while 'integral.'+str(icnt) in target:
                    icnt += 1
                chkpt.copy(grp, target, name='integral.'+str(icnt))
        chkpt.close()    

    target.close()

# 
def time_steps(chkpt=None, grp_name=None, file_name=None):
    """Documentation to come"""
    # if file handle is None, get file stream by opening
    # file, file_name
    if chkpt is None and file_name is not None:
        chkpt = h5py.File(file_name.strip(), 'r', libver='latest')

    if grp_name is None:
        grp_name = 'wavefunction.0'

    # if this group doesn't posses a list of times, return
    # 'none'
    if grp_name+'/time' not in chkpt:
        return None

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

    # if opening from file name, close file when done
    if file_name is not None:
        chkpt.close()

    return steps

#
def update_adapt(time, parent, child, file_name=None, name=0):
    """ Update information regarding new basis functions """
    
    adapt_name = 'adapt.'+str(name)
    dset       = adapt_name+'/adapt_log'

    if file_name is not None:
        chkpt = h5py.File(file_name.strip(), 'r', libver='latest')
    else:
        chkpt = h5py.File(glbl.paths['chkpt_file'], 'a', libver='latest')

    data_row = package_adapt(time, parent, child) 
    chkpt[adapt_name].attrs['current_row'] += 1
    chkpt[dset][chkpt[adapt_name].attrs['current_row']] = data_row
    chkpt.close()

    return

#
def retrieve_adapt(file_name=None, name=0):
    """Pulls information about the dynamic changes to trajectory basis"""
    adapt_name = 'adapt.'+str(name)
    dset       = adapt_name+'/adapt_log'

    # default is to use file name from previous write
    if file_name is not None:
        glbl.paths['chkpt_file'] = file_name.strip()

    # open chkpoint file
    chkpt = h5py.File(glbl.paths['chkpt_file'], 'r', libver='latest')

    adapt_data = chkpt[dset][0:chkpt[adapt_name].attrs['current_row']+1]

    chkpt.close()

    return adapt_data

#
def retrieve_basis(chkpt):
    """Returns information about the trajectory basis, including:
       how many trajectories, start/end time of trajectory, how 
       many time steps, etc.""" 

    n_wfn   = sum(isWfn(grp) for grp in chkpt.keys())
    wfn_id  = [-1] * n_wfn
    n_traj  = [0]  * n_wfn
    n_steps = [[] for i in range(n_wfn)]
    t_times = [[] for i in range(n_wfn)]

    w_cnt = -1
    for i_wfn in chkpt.keys():
        if isWfn(i_wfn):
            w_cnt += 1
            wfn_id[w_cnt] = i_wfn
            for i_traj in chkpt[i_wfn].keys():
                if isTrajectory(i_traj):
                    n_traj[w_cnt] += 1 
                    steps = time_steps(chkpt, i_wfn+'/'+i_traj)
                    n_steps[w_cnt].append(len(steps))
                    t_times[w_cnt].append([steps[0],steps[-1]])

    return wfn_id, n_traj, n_steps, t_times

#
def retrieve_dataset(chkpt, dset, ti=None, tf=None):
    """Pulls an entire data set into a numpy array"""

    root     = '/'.join(dset.split('/')[0:-1])
    tset     = '/'.join(dset.split('/')[0:-1])+'/time'
    data_end = chkpt[root].attrs['current_row']  
 
    if dset not in chkpt:
        print("Cannot find "+str(dset)+" in "+str(chkpt.name)+".")
        return False
    if (ti is not None or ti is not None) and tset not in chkpt:
        print("time dataset not found, cannot request specific times")
        return False

    if ti is None:
        ti = chkpt[tset][0]
    start = np.abs(chkpt[tset][:data_end] - ti).argmin()
 
    if tf is None:
        tf = chkpt[tset][data_end]
    end   = np.abs(chkpt[tset][:data_end] - tf).argmin()+1

    narg = len(chkpt[dset].shape)-1
    inds = tuple([slice(start,end+1)]) + tuple(slice(0,None) for i in range(narg))
    return chkpt[tset][start:end+1], chkpt[dset][inds]


#------------------------------------------------------------------------------------
#
# Should not be called outside the module
#
#------------------------------------------------------------------------------------
def create(file_name, wfn, ints):
    """Creates a new checkpoint file."""
    # if a file already exists with this name, remove it
    if os.path.exists(file_name):
        os.remove(file_name)

    # create chkpoint file
    chkpt = h5py.File(file_name, 'w', libver='latest')

    # save the contents of glbl.py
    write_keywords(chkpt)

    # create basis group
    create_basis(chkpt, wfn, name=0)

    # wfn group -- row information
    create_wfn(chkpt, wfn, name=0)

    # integral group -- row information
    if ints is not None:
        create_int(chkpt, ints, name=0)

    # close following initialization
    chkpt.close()

# 
def create_basis(chkpt, wfn, name=0):
    """ Creates a new basis group, with suffix 'name' """
   
    adapt_name = 'adapt.'+str(name)

    if adapt_name in chkpt.keys():
        raise ValueError('adapt='+adapt_name+' already exists.'+
                         'Continuing...') 
    else:
        chkpt.create_group(adapt_name)
        chkpt[adapt_name].attrs['n_traj']      = wfn.n_traj()
        chkpt[adapt_name].attrs['current_row'] = -1
        chkpt[adapt_name].attrs['n_rows']      = 100

        # create the 'spawn.log' table 
        dset     = adapt_name+'/adapt_log'
        dshape   = (chkpt[adapt_name].attrs['n_rows'], 13)
        chkpt.create_dataset(dset, dshape, dtype=float, compression="gzip")

        # add initial trajectories with a 'born time' of 0
        #for i in range(wfn.n_traj()):
        #    data_row  = package_adapt(wfn.time, wfn.traj[i], wfn.traj[i]) 
        #    chkpt[adapt_name].attrs['current_row'] += 1         
        #    chkpt[dset][chkpt[adapt_name].attrs['current_row']] = data_row
        
    return

#
def create_wfn(chkpt, wfn, name=0):
    """Creates a new wavefunction group, with suffix 'name' """

    wfn_name = 'wavefunction.'+str(name)
    
    if wfn_name in chkpt.keys():
        raise ValueError('wavefunction='+wfn_name+' already exists.'+
                         'Continuing...') 
    else:
        chkpt.create_group(wfn_name)
        chkpt[wfn_name].attrs['current_row'] = -1
        chkpt[wfn_name].attrs['n_rows']      = 0

    return

#
def create_int(chkpt, ints, name=0):
    """Creates a new integral group, with suffix 'name' """

    int_name = 'integral.'+str(name)

    if int_name in chkpt.keys():
        raise ValueError('integral='+int_name+' already exists.'+
                         'Continuing...')
    else:
        chkpt.create_group(int_name)
        chkpt[int_name].attrs['current_row']     = -1
        chkpt[int_name].attrs['n_rows']          = 0

        # integral group -- time independent obj properties
        chkpt[int_name].attrs['kecoef']            = ints.kecoef
        chkpt[int_name].attrs['ansatz']            = ints.ansatz
        chkpt[int_name].attrs['numerical']         = ints.numerical
        chkpt[int_name].attrs['hermitian']         = ints.hermitian
        chkpt[int_name].attrs['require_centroids'] = ints.require_centroids

    return

#

def reset_datasets(chkpt, time):
    """Resets 'current_row' attribute on all datasets to correspond
    to 'time', and sets all subsequent data to the equivalent of
    'null'."""
    # if time is null, this corresponds to the current time, so there
    # is nothing to do
    if time is not None:
        # first go through wavefunction datasets
        # we are operating on first 'wavefunction' object we come across,
        # so this is only safe for checkpoints with a single wfn object.
        # We may decide to allow specficiation of a wfn object in
        # a merged chkpt sometime in future
        for grp in chkpt:
            # if this group has a 'time' dataset...
            if time_steps(chkpt, grp) is not None:
                cur_indx = get_time_index(chkpt, grp, time)
                chkpt[grp].attrs['current_row'] = cur_indx
                # if this is wavefunction or integral grp, desend one
                # level to work on trajectory and centroid objects
                if 'wavefunction' in grp or 'integral' in grp:
                    for sub_grp in chkpt[grp]:
                        sub_name = grp+'/'+sub_grp
                        if time_steps(chkpt, sub_name) is not None:
                            cur_indx = get_time_index(chkpt, sub_name, time)
                            chkpt[sub_name].attrs['current_row'] = cur_indx


def write_keywords(chkpt):
    """Writes the contents of glbl to the checkpoint file. This
    is only done once upon the creation of the file"""
    #loop over the dictionaries in glbl
    for keyword_section in glbl.sections.keys():

        #if module/class objects, skip
        if keyword_section == 'modules':
            continue

        grp_name = 'keywords_'+keyword_section
        chkpt.create_group(grp_name)
        for keyword in glbl.sections[keyword_section].keys():
            write_keyword(chkpt, grp_name, keyword,
                          glbl.sections[keyword_section][keyword])


def write_keyword(chkpt, grp, kword, val):
    """Write a keyword to simulation archive"""
    # try writing variable to h5py attribute using native format
    try:
        chkpt[grp].attrs[kword] = val
    except:
        # that fails, write as a string
        try:
            #..and if an array, preserve commas
            if isinstance(val, np.ndarray):
                sval = ','.join([val[i] for i in range(len(val))])
            else:
                sval = str(val)
            d_type = h5py.special_dtype(vlen=str)
            chkpt[grp].attrs.create(kword, sval, dtype=d_type)

        except Exception as e:
            print("Failed to write keyword:"+str(kword)+" = val:"+str(val)+
                  " -- "+str(e)+"\n")


def read_keywords(chkpt=None, save_paths=True):
    """Read keywords from archive file"""
    # open chkpoint file
    close_file = False

    if chkpt is None:
        close_file = True
        chkpt = h5py.File(glbl.paths['chkpt_file'], 'r', libver='latest')

    #loop over the dictionaries in glbl
    for keyword_section in glbl.sections.keys():

        #if module/class objects, skip
        if keyword_section == 'modules':
            continue

        grp_name = 'keywords_'+keyword_section
        for keyword in glbl.sections[keyword_section].keys():
            val = read_keyword(chkpt, grp_name, keyword)
            try:
                glbl.sections[keyword_section][keyword] = val
            except Exception as e:
                print("Failed to set keyword:"+str(keyword)+" -- "+str(e)+"\n")

        # remove global path if save_paths is false
        if keyword_section == 'paths' and not save_paths:
            for kword in glbl.sections['paths']:
                fname = os.path.basename(glbl.sections['paths'][kword])
                glbl.sections['paths'][kword] = fname

    if close_file:
        chkpt.close()

    return


def read_keyword(chkpt, grp, kword):
    """Read a particular keyword attribute"""
    # try writing variable to h5py attribute using native format
    try:
        val = chkpt[grp].attrs[kword]
        return convert_value(kword, val)
    except Exception as e:
        print("Failed to read keyword:"+str(kword)+" -- "+str(e)+"\n")


def convert_value(kword, val):
    """Converts a string value to NoneType, bool, int, float or string."""
    cval = val

    # if we can't interpret this as a list, return the string
    if str(cval).find(',',0) == -1:
        return cval

    # we have some items that are lists of strings which are converted
    # to simple strings. Try to interpret as list, and if that fails, return
    # the value unchanged
    try:
        cval = ast.literal_eval(val)
        if isinstance(cval, list):
            cval = np.ndarray(cval)
            return cval
    except ValueError:
        pass
    try:
        cval = str(cval).split(',')
    except ValueError:
        pass

    # else just a string and return as-is
    return cval

def write_wavefunction(chkpt, wfn, time, name=0):
    """Documentation to come"""
    # this is a little hack to ensure backwards compatibility
    # we should eventually do away with this
    if 'wavefunction.'+str(name) in chkpt:
        wfn_name = 'wavefunction.'+str(name)
        wfn_data,wfn_type = package_wfn(wfn)
    else:
        wfn_name = 'wavefunction'
        wfn_data,wfn_type = package_wfn_old(wfn)

    # if wfn doesn't exist, add it on the fly
    if wfn_name not in chkpt.keys():
        create_wfn(chkpt, wfn, name=name)

    write_package(chkpt, wfn_name, wfn_data, wfn_type)

    # now step through and write trajectories
    for i in range(wfn.n_traj()):
        write_trajectory(chkpt, wfn.traj[i])


def write_integral(chkpt, integral, time, name=0):
    """Documentation to come"""

    # this is a little hack to ensure backwards compatibility
    # we should eventually do away with this
    if 'integral.'+str(name) in chkpt:
        int_name = 'integral.'+str(name)
        int_data, int_type = package_integral(integral, time)
    else:
        int_name = 'integral'
        int_data, int_type = package_integral_old(integral, time)

    # if integral doesn't exist, add it on the fly
    if int_name not in chkpt.keys():
        create_int(chkpt, integral, name=name)

    write_package(chkpt, int_name, int_data, int_type)

    # now step through centroids, if they're present
    if integral.require_centroids:
        for i in range(len(integral.centroids)):
            for j in range(i):
                 if integral.centroids[i][j] is not None:
                     write_centroid(chkpt, integral.centroids[i][j], time)


def write_trajectory(chkpt, traj, name=0):
    """Documentation to come"""

    # this is a little hack to ensure backwards compatibility
    # we should eventually do away with this
    if 'wavefunction.'+str(name) in chkpt:
        grp_name = 'wavefunction.'+str(name)
        old_style = False
        t_data, t_type = package_trajectory(traj)
    else:
        grp_name = 'wavefunction'
        old_style = True
        t_data, t_type = package_trajectory_old(traj, traj.time)

    # write trajectory
    tgrp_name = grp_name+'/'+str(traj.label)
    write_package(chkpt, tgrp_name, t_data, t_type, 
                          grp_attr={'kecoef': traj.kecoef})

    # if MOs exist, write them as an attribute
    if 'mo' in traj.pes.avail_data():
        # until python figures out unicode, we need to explicitly encode strings
        mo_encode = [str(mo_i).encode('utf8') for mo_i in traj.pes.get_data('mo')]
        if 'mo' in chkpt[tgrp_name].attrs.keys():
            chkpt[tgrp_name].attrs.modify('mo', mo_encode)
        else:
            chkpt[tgrp_name].attrs.create('mo', mo_encode)


def write_centroid(chkpt, cent, time, name=0):
    """Documentation to come"""

    # open the trajectory file
    grp_name = 'integral.'+str(name)

    # this is a little hack to ensure backwards compatibility
    # we should eventually do away with this
    if 'integral.'+str(name) in chkpt:
        grp_name = 'integral.'+str(name)
        c_data, c_type  = package_centroid(cent, time)
    else:
        grp_name = 'integral'
        c_data, c_type  = package_centroid_old(cent, time)

    # write centroid
    cgrp_name = grp_name+'/'+str(cent.label)
    write_package(chkpt, cgrp_name, c_data, c_type)

    # if MOs exist, write them as an attribute
    if 'mo' in cent.pes.avail_data():
        # until python figures out unicode, we need to explicitly encode strings
        mo_encode = [str(mo_i).encode('utf8') for mo_i in cent.pes.get_data('mo')]
        if 'mo' in chkpt[cgrp_name].attrs.keys():
            chkpt[cgrp_name].attrs.modify('mo', mo_encode)
        else:
            chkpt[cgrp_name].attrs.create('mo', mo_encode)


def write_package(chkpt, grp, data, types, grp_attr=None):
    """
    A generate function to write a dictionary of 'data'
    to dataset labeled by 'labels'

    Arguments: 
    data:    dictonary of data with keys given by 'labels'

    Returns:
    None
    """
    n_blk   = default_blk_size()

    # if new_grp isn't none, create new group in grp with
    # name new_grp
    if grp not in chkpt.keys(): 
        chkpt.create_group(grp)
        current_row                     = -1
        chkpt[grp].attrs['current_row'] = current_row
        chkpt[grp].attrs['n_rows']      = n_blk
        n_rows                          = chkpt[grp].attrs['n_rows']
        if grp_attr is not None:
            for key,value in grp_attr.items():
                chkpt[grp].attrs[key] = value

    # update the current row index (same for all data sets)
    chkpt[grp].attrs['current_row'] += 1
    current_row = chkpt[grp].attrs['current_row']

    if current_row == chkpt[grp].attrs['n_rows']:
        resize = True
        chkpt[grp].attrs['n_rows'] += n_blk
    else:
        resize = False
    n_rows = chkpt[grp].attrs['n_rows']

    # first write items with time-independent dimensions
    for label in data.keys():
        dset = grp+'/'+label

        if dset in chkpt:
            if resize:
                d_shape  = (n_rows,)
                if h5py.check_dtype(vlen=types[label]) is None:
                    d_shape   =  d_shape   + data[label].shape
                chkpt[dset].resize(d_shape)

        # if this is the first time we're trying to write this bundle,
        # create a new datasets with reasonble default sizes
        else:
            d_shape   = (n_rows,)
            max_shape = (max_dset_size(),)
            if h5py.check_dtype(vlen=types[label]) is None:
                d_shape   =  d_shape   + data[label].shape
                max_shape =  max_shape + data[label].shape
            d_type    = types[label]
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, 
                                 dtype=d_type, compression="gzip")

        chkpt[dset][current_row] = data[label]

    return

def read_wavefunction(chkpt, time, name=0):
    """Documentation to come"""
    nstates  = glbl.properties['n_states']
    widths   = glbl.properties['crd_widths']
    masses   = glbl.properties['crd_masses']
    dim      = len(widths)

    # this is a little hack to ensure backwards compatibility
    # we should eventually do away with this
    if 'wavefunction.'+str(name) in chkpt: 
        wfn_name = 'wavefunction.'+str(name)
        old_style = False
    else:
        wfn_name = 'wavefunction'
        old_style = True

    # check that we have the desired time:
    read_row = get_time_index(chkpt, wfn_name, time)

    if read_row is None:
        ValueError('time='+str(time)+' requested, but not in checkpoint file')
        return None

    # create the wavefunction object to hold the data
    wfn = wavefunction.Wavefunction()
    mat = matrices.Matrices()

    # dimensions of these objects are not time-dependent
    wfn.time    = chkpt[wfn_name+'/time'][read_row,0]

    wfn_grps =  sorted([str(label) for label in chkpt[wfn_name].keys()])
    for label in wfn_grps:

        # these are outputs of functions -- no place to put this data
        if (label=='time' or label=='energy'):
            continue

        # set the population array
        if label=='pop':
            wfn.stpop = chkpt[wfn_name+'/pop'][read_row]

        if label=='norm':
            wfn.wfn_norm = chkpt[wfn_name+'/norm'][read_row,0]

        # if matrices are present, read those in
        if label in mat.mat_list:
            mat_read = chkpt[wfn_name+'/'+label][read_row]
            mat.set(label, mat_read)
            continue

        # if we're here, we're reading a trajectory
        t_grp = wfn_name+'/'+label
        t_row = get_time_index(chkpt, t_grp, time)

        if t_row is None:
            continue

        new_traj = trajectory.Trajectory(nstates, dim,
                                         width=widths,
                                         mass=masses,
                                         label=label)
        if old_style:
            read_trajectory_old(chkpt, new_traj, t_grp, t_row)
        else:
            read_trajectory(chkpt, new_traj, t_grp, t_row)

        # if there was an error reading the trajectory, return None
        if new_traj is None:
            return None

        wfn.add_trajectory(new_traj.copy())

    if len(mat.avail()) > 0:
        wfn.update_matrices(mat)

    return wfn

def read_integral(chkpt, time, name=0):
    """Documentation to come"""

    nstates  = glbl.properties['n_states']
    widths   = glbl.properties['crd_widths']
    dim      = len(widths)
    ansatz   = glbl.methods['ansatz']
    numerics = glbl.methods['integral_eval'] 

    # this is a little hack to ensure backwards compatibility
    # we should eventually do away with this
    if 'integral.'+str(name) in chkpt:
        int_name = 'integral.'+str(name)
    else:
        int_name = 'integral'

    # return None if no integrals section
    if int_name not in chkpt:
        return None

    kecoef   = chkpt[int_name].attrs['kecoef'] 

    # check that we have the desired time:
    read_row = get_time_index(chkpt, int_name, time)

    if read_row is None:
        raise ValueError('time='+str(time)+' requested, but not in checkpoint file')
        return None

    ints = integral.Integral(kecoef, ansatz, numerics)

    if ints.require_centroids:
        for label in chkpt[int_name]:

            if label == 'time':
                continue

            c_grp = int_name+'/'+label
            c_row = get_time_index(chkpt, c_grp, time)

            if c_row is None:
                continue

            new_cent = centroid.Centroid(nstates=nstates, dim=dim, width=widths)
            read_centroid(chkpt, new_cent, c_grp, c_row)

            # if there was an error reading a centroid, return None
            if new_cent is None:
                return None

            ints.add_centroid(new_cent)

    return ints


def read_trajectory(chkpt, new_traj, t_grp, t_row):
    """Documentation to come"""
    # populate the surface object in the trajectory

    # if this time step doesn't exist, return null trajectory
    if t_row > chkpt[t_grp].attrs['current_row']:
        new_traj = None
    else:
        # set information about the trajectory itself
        [time]               = chkpt[t_grp+'/time'][t_row]
        [parent, state]      = chkpt[t_grp+'/states'][t_row]
        [gamma]              = chkpt[t_grp+'/phase'][t_row]
        last_adapt           = chkpt[t_grp+'/adapt'][t_row]
        momt                 = chkpt[t_grp+'/momentum'][t_row]
        try:
            [amp] = chkpt[t_grp+'/amp'][t_row]
        except ValueError:
            [amp_r, amp_i] = chkpt[t_grp+'/amp'][t_row]
            amp            = amp_r + amp_i*1.j

        pes = surface.Surface()
        for data_label in chkpt[t_grp].keys():
            if pes.valid_data(data_label):
                dset = chkpt[t_grp+'/'+data_label]
                pes.add_data(data_label, dset[t_row])

        # if MOs are present as an attribute, read them in
        if 'mo' in chkpt[t_grp].attrs.keys():
            mo_decode = [mo if type(mo) is str else 
                           mo.decode("utf-8","ignore") 
                              for mo in chkpt[t_grp].attrs['mo']]
            pes.add_data('mo', mo_decode)

        # currently, momentum has to be read in separately
        momt    = chkpt[t_grp+'/momentum'][t_row]

        new_traj.time   = time
        new_traj.state  = int(state)
        new_traj.parent = int(parent)
        new_traj.update_amplitude(amp)
        new_traj.last_spawn = last_adapt

        new_traj.update_pes_info(pes)
        new_traj.update_x(new_traj.pes.get_data('geom'))
        new_traj.update_p(momt)


def read_trajectory_old(chkpt, new_traj, t_grp, t_row):
    """Documentation to come"""
    # populate the surface object in the trajectory

    # if this time step doesn't exist, return null trajectory
    if t_row > len(chkpt[t_grp+'/glbl'])-1:
        new_traj = None
    else:
        # set information about the trajectory itself
        data_row = chkpt[t_grp+'/glbl'][t_row]
        [parent, state, new_traj.gamma, amp_real, amp_imag] = data_row[0:5]

        pes = surface.Surface()
        for data_label in chkpt[t_grp].keys():
            if pes.valid_data(data_label):
                dset = chkpt[t_grp+'/'+data_label]
                pes.add_data(data_label, dset[t_row])

        # if MOs are present as an attribute, read them in
        if 'mo' in chkpt[t_grp].attrs.keys():
            mo_decode = [mo_i.decode("utf-8","ignore") for mo_i in chkpt[t_grp].attrs['mo']]
            pes.add_data('mo', mo_decode)

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
    # if this time step doesn't exist, return null centroid
    if c_row > chkpt[c_grp].attrs['current_row']:
        new_cent = None
    else:
        # set information about the trajectory itself
        parent = [0.,0.]
        states = [0.,0.]
        [parent[0], parent[1], states[0], states[1]] = chkpt[c_grp+'/states'][c_row]

        # populate the surface object in the trajectory
        pes = surface.Surface()
        for data_label in chkpt[c_grp].keys():
            if pes.valid_data(data_label):
                dset = chkpt[c_grp+'/'+data_label]
                pes.add_data(data_label, dset[c_row])

        # if MOs are present as an attribute, read them in
        if 'mo' in chkpt[c_grp].attrs.keys():
            mo_decode = [mo if type(mo) is str else 
                           mo.decode("utf-8","ignore") 
                              for mo in chkpt[t_grp].attrs['mo']]

        # currently, momentum has to be read in separately
        momt    = chkpt[c_grp+'/momentum'][c_row]

        new_cent.parents = [int(i) for i in parent]
        new_cent.states  = [int(i) for i in states]

        idi              = max(new_cent.parents)
        idj              = min(new_cent.parents)
        new_cent.label   = -((idi * (idi - 1) // 2) + idj + 1)

        new_cent.update_pes_info(pes)
        new_cent.pos = new_cent.pes.get_data('geom')
        new_cent.mom = momt


def get_time_index(chkpt, grp_name, time):
    """Documentation to come"""
    time_vals = time_steps(chkpt=chkpt, grp_name=grp_name)

    if time_vals is None:
        return None

    if len(time_vals)==0:
        return None

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

    if dt[read_row] != 0.:
        if len(match_chk) == 0:
            read_row = None
        elif dt[read_row] > 0.5*min(match_chk):
            read_row = None

    return read_row

#
def package_adapt(time, parent, child):
    """Record when and from whom new basis functions are spawned"""

    adapt_data = np.zeros(13, dtype=float)
    adapt_data[[0,1,2]]   = [time, 
                            parent.last_spawn[child.state],
                            parent.exit_time[child.state]]
    adapt_data[[3,4,5,6]] = [parent.label,      parent.state,
                            child.label,        child.state]
    adapt_data[[7,8]]     = [parent.kinetic(),   child.kinetic()] 
    adapt_data[[9,10]]    = [parent.potential(), child.potential()]
    adapt_data[[11,12]]   = [parent.classical(), child.classical()]

    return adapt_data

#

#
def package_wfn(wfn):
    """Documentation to come"""
    # dimensions of these objects are not time-dependent
    # determine the populations

    # hack
    if wfn.pop() is None:
        stpop    = np.zeros(glbl.properties['n_states'], dtype=float)
    else:
        stpop    = wfn.pop()

    wfn_data   = dict(
        time   = np.array([wfn.time], dtype='float'),
        pop    = stpop,
        norm   = np.array([wfn.norm()], dtype='float'),
        energy = np.array([wfn.pot_quantum(),   wfn.kin_quantum(),
                           wfn.pot_classical(), wfn.kin_classical()]))

    if glbl.properties['store_matrices']:
        avail_mat = wfn.matrices.avail()
        for label in wfn.matrices.avail():
            wfn_data[label] = wfn.matrices.matrix[label].flatten(order='F')

    wfn_types  = dict(
        time   = np.dtype('float'),
        pop    = np.dtype('float'),
        norm   = np.dtype('float'),
        energy = np.dtype('float'),
        t      = h5py.special_dtype(vlen=np.dtype('complex')),
        v      = h5py.special_dtype(vlen=np.dtype('complex')),
        h      = h5py.special_dtype(vlen=np.dtype('complex')),
        s      = h5py.special_dtype(vlen=np.dtype('complex')),
        s_traj = h5py.special_dtype(vlen=np.dtype('complex')),
        sdot   = h5py.special_dtype(vlen=np.dtype('complex')),
        heff   = h5py.special_dtype(vlen=np.dtype('complex')))

    return wfn_data, wfn_types


def package_integral(integral, time):
    """Documentation to come"""
    int_data = dict(
        time = np.array([time],dtype='float'))

    int_types = dict(
        time = np.dtype('float'))

    return int_data, int_types


def package_trajectory(traj):
    """Documentation to come"""
    traj_data = dict(
        time     = np.array([traj.time],dtype='float'),
        amp      = np.array([traj.amplitude]),
        phase    = np.array([traj.gamma]),
        states   = np.array([traj.parent, traj.state]),
        adapt    = traj.last_spawn,
        momentum = traj.p())

    traj_types = dict(
        time     = np.dtype('float'),
        amp      = h5py.special_dtype(vlen=np.dtype('complex')),
        phase    = np.dtype('float'),
        states   = np.dtype('int'),
        adapt    = np.dtype('float'),
        momentum = np.dtype('float'))

    # store everything about the surface
    for obj in traj.pes.avail_data():

        # don't write MOs as a time-dependent dataset
        if obj == 'mo':
            continue

        traj_data[obj] = traj.pes.get_data(obj)
        # treat all surface properties as floats
        traj_types[obj] = np.dtype('float') 

    return traj_data, traj_types

def package_centroid(cent, time):
    """Documentation to come"""
    cent_data = dict(
        time     = np.array([time],dtype='float'),
        states   = np.concatenate((cent.parents, cent.states)),
        momentum = cent.p())

    cent_types = dict(
        time     = np.dtype('float'),
        states   = np.dtype('int'),
        momentum = np.dtype('float'))

    # last, store everything about the surface
    for obj in cent.pes.avail_data():

        # don't write MOs as a time-dependent dataset
        if obj == 'mo':
            continue

        cent_data[obj] = cent.pes.get_data(obj)
        # treat all surface properties as floats
        cent_types[obj] = np.dtype('float')

    return cent_data, cent_types

def package_wfn_old(wfn):
    """Documentation to come"""
    # dimensions of these objects are not time-dependent
    wfn_data = dict(
        time   = np.array([wfn.time], dtype='float'),
        pop    = np.array(wfn.pop()),
        energy = np.array([wfn.pot_quantum(),   wfn.kin_quantum(),
                           wfn.pot_classical(), wfn.kin_classical()])
                    )

    wfn_types  = dict(
        time   = np.dtype('float'),
        pop    = np.dtype('float'),
        energy = np.dtype('float'))

    return wfn_data, wfn_types


def package_integral_old(integral, time):
    """Documentation to come"""
    int_data = dict(
        time = np.array([time],dtype='float'))

    int_types = dict(
        time = np.dtype('float'))

    return int_data, int_types


def package_trajectory_old(traj, time):
    """Documentation to come"""
    # time is not an element in a trajectory, but necessary to
    # uniquely tag everything
    traj_data = dict(
        time     = np.array([time],dtype='float'),
        glbl     = np.concatenate((np.array([traj.parent, traj.state, traj.gamma,
                                 traj.amplitude.real, traj.amplitude.imag]),
                                 traj.last_spawn)),
        momentum = traj.p())

    traj_types = dict(
        time     = np.dtype('float'),
        glbl     = np.dtype('float'),
        momentum = np.dtype('float'))

    # store everything about the surface
    for obj in traj.pes.avail_data():

        # don't write MOs as a time-dependent dataset
        if obj == 'mo':
            continue

        traj_data[obj] = traj.pes.get_data(obj)
        # treat all surface properties as floats
        traj_types[obj] = np.dtype('float')

    return traj_data, traj_types


def package_centroid_old(cent, time):
    """Documentation to come"""
    cent_data = dict(
        time     = np.array([time],dtype='float'),
        glbl     = np.concatenate((cent.parents, cent.states)),
        momentum = cent.p()
                     )

    cent_types = dict(
        time     = np.dtype('float'),
        glbl     = np.dtype('float'),
        momentum = np.dtype('float'))

    # last, store everything about the surface
    for obj in cent.pes.avail_data():

        # don't write MOs as a time-dependent dataset
        if obj == 'mo':
            continue

        cent_data[obj] = cent.pes.get_data(obj)
        # treat all surface properties as floats
        cent_types[obj] = np.dtype('float')

    return cent_data, cent_types


def max_dset_size():
    """Return the maximum dataset size"""
    return max( default_blk_size(), 
                100*int(glbl.properties['simulation_time'] / 
                      glbl.properties['default_time_step']))

def default_blk_size():
    """Documentation to come"""
    # let's just keep this to small default size: 25
    # need to look into optimizing this more
    return 100 


def isTrajectory(dset_name):
    """ returns true is a dset is a valid name for a trajectory"""
    try:
        int(dset_name)
        return True
    except ValueError:
        return False


def isWfn(dset_name):
    """ returns true is a dset is a valid name for a wavefunction"""
    return 'wavefunction' in dset_name


#-----------------------------------------------------------------------------
#
#  printing routines
#
#-----------------------------------------------------------------------------
def generate_data_formats():
    """Initialized all the output format descriptors."""
    global dump_header, dump_format, tfile_names, bfile_names

    nst   = glbl.properties['n_states']
    ncrd  = len(glbl.properties['crd_widths'])
    ncart = 3         # assumes expectation values of transition/permanent dipoles in
                      # cartesian coordinates
    natm  = max(1,int(ncrd / ncart)) # dirty -- in case we have small number of n.modes
    dstr  = ('x', 'y', 'z')
    acc1  = 12
    acc2  = 16

    # ******************* dump formats *******************************

    # ----------------- trajectory data --------------------------------
    # trajectory output
    arr1 = ['{:>12s}'.format('    x' + str(i+1)) for i in range(ncrd)]
    arr2 = ['{:>12s}'.format('    p' + str(i+1)) for i in range(ncrd)]
    tfile_names[tkeys[0]] = 'trajectory'
    dump_header[tkeys[0]] = ('Time'.rjust(acc1) + ''.join(arr1) +
                             ''.join(arr2) + 'Phase'.rjust(acc1) +
                             'Re[Amp]'.rjust(acc1) + 'Im[Amp]'.rjust(acc1) +
                             'Norm[Amp]'.rjust(acc1) + 'State'.rjust(acc1) +
                             '\n')
    dump_format[tkeys[0]] = ('{:12.4f}'+
                             ''.join('{:12.6f}' for i in range(2*ncrd+5))+
                             '\n')

    # potential energy
    arr1 = ['{:>16s}'.format('potential.' + str(i)) for i in range(nst)]
    tfile_names[tkeys[1]] = 'poten'
    dump_header[tkeys[1]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[1]] = ('{:12.4f}' +
                             ''.join('{:16.10f}' for i in range(nst)) + '\n')

    # gradients
    arr1 = ['            x' + str(i+1) for i in range(ncrd)]
    tfile_names[tkeys[2]] = 'grad'
    dump_header[tkeys[2]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[2]] = ('{0:>12.4f}' +
                             ''.join('{' + str(i) + ':14.8f}'
                                     for i in range(1, ncrd+1)) + '\n')

    # coupling
    arr1 = ['{:>12s}'.format('coupling.' + str(i)) for i in range(nst)]
    arr2 = ['{:>12s}'.format('c * v .' + str(i)) for i in range(nst)]
    tfile_names[tkeys[3]] = 'coup'
    dump_header[tkeys[3]] = ('Time'.rjust(acc1) + ''.join(arr1) +
                             ''.join(arr2) + '\n')
    dump_format[tkeys[3]] = ('{:12.4f}' +
                             ''.join('{:12.5f}' for i in range(2*nst)) + '\n')

    # ---------------------- interface data --------------------------------
    # permanent dipoles
    arr1 = ['{:>12s}'.format('dip_st' + str(i) + '.' + dstr[j])
            for i in range(nst) for j in range(ncart)]
    tfile_names[tkeys[5]] = 'dipole'
    dump_header[tkeys[5]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[5]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(nst*ncart)) + '\n')
    # transition dipoles
    arr1 = ['  td_s' + str(j) + '.s' + str(i) + '.' + dstr[k]
            for i in range(nst) for j in range(i) for k in range(ncart)]
    ncol = int(nst*(nst-1)*ncart/2+1)
    tfile_names[tkeys[6]] = 'tr_dipole'
    dump_header[tkeys[6]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[6]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(1, ncol)) + '\n')

    # second moments
    arr1 = ['   sec_s' + str(i) + '.' + dstr[j] + dstr[j]
            for i in range(nst) for j in range(ncart)]
    tfile_names[tkeys[7]] = 'sec_mom'
    dump_header[tkeys[7]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[7]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(nst*ncart)) + '\n')

    # atomic populations
    arr1 = ['    st' + str(i) + '_a' + str(j+1)
            for i in range(nst) for j in range(natm)]
    tfile_names[tkeys[8]] = 'atom_pop'
    dump_header[tkeys[8]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[8]] = ('{:12.4f}' +
                             ''.join('{:10.5f}'
                                     for i in range(nst*natm)) + '\n')

    # ----------------- dump formats (wavefunction files) -----------------

    # adiabatic state populations
    arr1 = ['     state.' + str(i) for i in range(nst)]
    bfile_names['pop'] = 'n.dat'
    dump_header['pop'] = ('Time'.rjust(acc1) + ''.join(arr1) +
                             'Norm'.rjust(acc1) + '\n')
    dump_format['pop']  = ('{:12.4f}' +
                             ''.join('{:12.6f}' for i in range(nst)) +
                             '{:12.6f}\n')

    # the bundle energy
    arr1 = ('   potential(QM)', '     kinetic(QM)', '       total(QM)',
            '  potential(Cl.)', '    kinetic(Cl.)', '      total(Cl.)')
    bfile_names['energy'] = 'e.dat'
    dump_header['energy'] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format['energy'] = ('{:12.4f}' +
                             ''.join('{:16.10f}' for i in range(6)) + '\n')

    # autocorrelation function
    arr1 = ('      Re a(t)','         Im a(t)','         abs a(t)')
    bfile_names['auto'] = 'auto.dat'
    dump_header['auto'] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format['auto'] = ('{:12.4f}' +
                             ''.join('{:16.10f}' for i in range(3)) + '\n')

    # spawn table
    lenst        = 7
    arr1  = ('time(entry)','time(spawn)', 'time(exit)')
    arr2  = ('parent','state','child','state')
    arr3  = ('ke(parent)','ke(child)',
             'pot(parent)','pot(child)',
             'total(parent)','total(child)')
    bfile_names['spawn'] = 'spawn.dat'
    dump_header['spawn'] = (''.join([arr1[i].rjust(acc1) for i in range(len(arr1))]) +
                            ''.join([arr2[i].rjust(7) for i in range(len(arr2))]) + 
                            ''.join([arr3[i].rjust(acc2) for i in range(len(arr3))])+'\n')
    dump_format['spawn'] = ('{:12.4f}{:12.4f}{:12.4f}{:7.1f}{:7.1f}{:7.1f}{:7.1f}' +
                            '{:16.8f}{:16.8f}{:16.8f}{:16.8f}' +
                            '{:16.8f}{:16.8f}\n')

    # trajectory matrices
    tfile_names['hessian']   = 'hessian.dat'

    # bundle matrices
    bfile_names['t']         = 't.dat'
    bfile_names['v']         = 'v.dat'
    bfile_names['s']         = 's.dat'
    bfile_names['sinv']      = 'sinv.dat'
    bfile_names['s_nuc']     = 's_nuc.dat'
    bfile_names['s_elec']    = 's_elec.dat'
    bfile_names['sdot']      = 'sdot.dat'
    bfile_names['h']         = 'h.dat'
    bfile_names['heff']      = 'heff.dat'
    bfile_names['t_overlap'] = 't_overlap.dat'


def print_traj_row(label, key, data):
    """Appends a row of data, formatted by entry 'fkey' in formats to
    file 'filename'."""
    filename = tfile_names[key] + '.' + str(label)

    if not os.path.isfile(filename):
        with open(filename, 'x') as outfile:
            outfile.write(dump_header[key])
            outfile.write(dump_format[key].format(*data))
    else:
        with open(filename, 'a') as outfile:
            outfile.write(dump_format[key].format(*data))


def print_traj_mat(time, key, mat):
    """Prints a matrix to file with a time label."""
    filename = tfile_names[key]

    with open(filename, 'a') as outfile:
        outfile.write('{:9.2f}\n'.format(time))
        outfile.write(np.array2string(mat,
                      formatter={'complex_kind':lambda x: '{: 15.8e}'.format(x)})+'\n')


def print_wfn_row(key, data):
    """Appends a row of data, formatted by entry 'fkey' in formats to
    file 'filename'."""
    filename = bfile_names[key]

    if not os.path.isfile(filename):
        with open(filename, 'x') as outfile:
            outfile.write(dump_header[key])
            outfile.write(dump_format[key].format(*data))
    else:
        with open(filename, 'a') as outfile:
            outfile.write(dump_format[key].format(*data))


def print_wfn_mat(time, key, mat):
    """Prints a matrix to file with a time label."""
    filename = bfile_names[key]

    with open(filename, 'a') as outfile:
        outfile.write('{:9.2f}\n'.format(time))
        outfile.write(np.array2string(mat,
                      formatter={'complex_kind':lambda x: '{: 15.8e}'.format(x)})+'\n')
