import os
import shutil
import numpy as np
import src.fmsio.glbl as glbl

home_path   = ''
scr_path    = ''
tkeys       = ['traj_dump', 'ener_dump',   'coup_dump', 'dipole_dump', 'secm_dump',
               'tran_dump', 'apop_dump', 'grad_dump']
bkeys       = ['pop_dump', 'bener_dump', 'hmat_dump', 'smat_dump', 'spawn_dump']
dump_header = dict()
dump_format = dict()
log_format  = dict()
tfile_names = dict()
bfile_names = dict()

# 
# Read the fms.input file. This contains variables related
# to the running of the dynamics simulation.
#
def read_input_files():
    global scr_path, home_path

    # save the name of directory where program is called from
    home_path = os.getcwd()

    # set a sensible default for scr_path
    scr_path = os.environ['TMPDIR']
    if os.path.exists(scr_path):
        shutil.rmtree(scr_path)
        os.makedirs(scr_path)

    #
    # Read fms.input. This contains general simulation variables
    # 
    kwords = read_namelist('fms.input')
    for k,v in kwords.items():
        if k in glbl.fms:
            glbl.fms[k] = v
        else:
            print("Variable "+str(k)+" in fms.input unrecognized. Ignoring...")

    #
    # Read pes.input. This contains interface-specific user options. Get what
    #  interface we're using via glbl.fms['interface'], and populate the corresponding
    #  dictionary of keywords from glbl module
    #
    # Clumsy. Not even sure this is the best way to do this (need to segregate
    # variables in different dictionaries. fix this later
    kwords = read_namelist('pes.input')
    if glbl.fms['interface'] == 'columbus':
        for k,v in kwords.items():
            if k in glbl.columbus:
                glbl.columbus[k] = v
            else:
                print("Variable "+str(k)+" in fms.input unrecognized. Ignoring...")
    elif glbl.fms['interface'] == 'vibronic':
        for k,v in kwords.items():
            if k in glbl.vibronic:
                glbl.vibronic[k] = v
            else:
                print("Variable "+str(k)+" in fms.input unrecognized. Ignoring...")
    elif glbl.fms['interface'] == 'boson_model_diabatic':
        for k,v in kwords.items():
            if k in glbl.vibronic:
                glbl.boson[k] = v
    else:
        print("Interface: "+str(glbl.fms['interface'])+" not recognized.")

    return

#
# Reads a namelist style input, returns results in dictionary
#
def read_namelist(filename):
    kwords = dict()
    
    if os.path.exists(filename):
        with open(filename,'r',encoding="utf-8") as infile: 
            for line in infile:    
                if "=" in line:
                    line = line.rstrip("\r\n")
                    (key,value) = line.split('=',1)
                    key = str(key.strip())
                    value = str(value.strip())
                    try:
                        kwords[key] = float(value)
                        if kwords[key].is_integer():
                            kwords[key] = int(value)
                    except ValueError:
                        pass
                    if key not in kwords:
                        kwords[key] = value

    return kwords

#
# initialize all the ouptut format descriptors
#
def init_fms_output():
    global home_path, scr_path, \
           log_format, tkeys, bkeys, dump_header, dump_format, \
           tfile_names, bfile_names

    np = int(glbl.fms['num_particles'])
    nd = int(glbl.fms['dim_particles'])
    ns = int(glbl.fms['n_states'])
    dstr = ("x","y","z")

    # ----------------- dump formats (trajectory files) -----------------
    # trajectory output
    arr1 = ['      pos' + str(i+1) + '.' + dstr[x] for i in range(np) for x in range(nd)]
    arr2 = ['      mom' + str(i+1) + '.' + dstr[x] for i in range(np) for x in range(nd)]
    tfile_names[tkeys[0]] = 'trajectory'
    dump_header[tkeys[0]]     = '     Time' + ''.join(arr1) + ''.join(arr2) + '     Phase' + \
                            '   Re[Amp]' + '   Im[Amp]' + ' Norm[Amp]'+'     State\n'
    dump_format[tkeys[0]]     = '{0:9.2f}'+ \
                             ''.join('{'+str(i)+':12.6f}' for i in range(1,np*nd+1))         + \
                             ''.join('{'+str(i)+':12.6f}' for i in range(np*nd+1,2*np*nd+1)) + \
                             ''.join('{'+str(i)+':10.6f}'  for i in range(2*np*nd+1,2*np*nd+6))+'\n'

    # potential energy
    arr1 = ['     potential.'+str(i) for i in range(ns)]
    tfile_names[tkeys[1]] = 'poten'
    dump_header[tkeys[1]]     = '     Time' + ''.join(arr1)+'\n'
    dump_format[tkeys[1]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':16.10f}' for i in range(1,ns+1)) + '\n' 

    # coupling
    arr1 = ['  coupling.'+str(i) for i in range(ns)]
    arr2 = ['    c * v .'+str(i) for i in range(ns)]
    tfile_names[tkeys[2]] = 'coupling'
    dump_header[tkeys[2]]     = '     Time' + ''.join(arr1) + ''.join(arr2) + '\n'
    dump_format[tkeys[2]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':12.5f}' for i in range(1,2*ns+1)) + '\n'

    # permanent dipoles
    arr1 = ['   dip_st'+str(i)+'.'+dstr[j] for i in range(ns) for j in range(nd)]
    tfile_names[tkeys[3]] = 'dipole'
    dump_header[tkeys[3]]     = '     Time' + ''.join(arr1) + '\n'
    dump_format[tkeys[3]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':12.5f}' for i in range(1,ns*nd+1)) + '\n'

    # transition dipoles
    arr1 = ['  td_s'+str(j)+'.s'+str(i)+'.'+dstr[k] for i in range(ns) for j in range(i) for k in range(nd)]
    ncol = int(ns*(ns-1)*nd/2+1)
    tfile_names[tkeys[4]] = 'tr_dipole'
    dump_header[tkeys[4]]     = '     Time' + ''.join(arr1) + '\n'
    dump_format[tkeys[4]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':12.5f}' for i in range(1,ncol)) + '\n'

    # second moments 
    arr1 = ['   sec_s'+str(i)+'.'+dstr[j]+dstr[j] for i in range(ns) for j in range(nd)]
    tfile_names[tkeys[5]] = 'sec_mom'
    dump_header[tkeys[5]]     = '     Time' + ''.join(arr1) + '\n'
    dump_format[tkeys[5]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':12.5f}' for i in range(1,ns*nd+1)) + '\n'

    # atomic populations
    arr1 = ['    st'+str(i)+'_p'+str(j+1) for i in range(ns) for j in range(np)]
    tfile_names[tkeys[6]] = 'atom_pop'
    dump_header[tkeys[6]]     = '     Time' + ''.join(arr1) + '\n'
    dump_format[tkeys[6]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':10.5f}' for i in range(1,ns*np+1)) + '\n'

    # gradients
    arr1 = ['  grad_part'+str(i+1)+'.'+dstr[j] for i in range(np) for j in range(nd)]
    tfile_names[tkeys[7]] = 'gradient'
    dump_header[tkeys[7]]     = '     Time' + ''.join(arr1) + '\n'
    dump_format[tkeys[7]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':14.8f}' for i in range(1,np*nd+1)) + '\n'

    # ----------------- dump formats (bundle files) -----------------

    # adiabatic state populations
    arr1 = ['     state.'+str(i) for i in range(ns)]
    bfile_names[bkeys[0]] = 'n.dat'
    dump_header[bkeys[0]]     = '     Time' + ''.join(arr1) + '      Norm' + '\n'
    dump_format[bkeys[0]]     = '{0:9.2f}' + \
                            ''.join('{'+str(i)+':12.6f}' for i in range(1,ns+1)) + \
                            '{'+str(ns+1)+':10.6f}\n'

    # the bundle energy
    arr1 = ('   potential(QM)','  potential(Cl.)','     kinetic(QM)',
            '    kinetic(Cl.)','       total(QM)','      total(Cl.)')
    bfile_names[bkeys[1]] = 'e.dat'
    dump_header[bkeys[1]]     = '     Time' + ''.join(arr1) + '\n'
    dump_format[bkeys[1]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':16.10f}' for i in range(1,6+1)) + '\n'

    # the hamiltonian matrix
    bfile_names[bkeys[2]] = 'h.dat'
    dump_format[bkeys[2]]     = ''

    # the trajectory overlap matrix
    bfile_names[bkeys[3]] = 's.dat'
    dump_format[bkeys[3]]     = ''    

    # the spawn log
    arr1 = ('time (entry)','time(spawn)','time(exit)',
            'parent','state','child','state',
            'ke(parent)','pot(parent)','ke(child)','pot(child)')
    bfile_names[bkeys[4]] = 'spawn.dat'
    dump_header[bkeys[4]]     = ' '.join(arr1)
    dump_format[bkeys[4]]     = '{0:8.2f} {1:8.2f} {2:8.2f} '  + \
                            '{3:4d} {4:4d} {5:4d} {6:4d} ' + \
                            '{7:12.8f} {8:12.8f} {9:12.8f} {10:12.8f}'

    # ------------------------- log file formats --------------------------
    with open(home_path+"/fms.log","w") as logfile:
        log_str = ' ---------------------------------------------------\n' + \
                  ' ab initio multiple spawning dynamics\n' + \
                  ' ---------------------------------------------------\n' + \
                  '\n' + \
                  ' *************\n' + \
                  ' input summary\n' + \
                  ' *************\n' + \
                  '\n' + \
                  ' file paths\n' + \
                  ' ---------------------------------------\n' + \
                  ' home_path   = '+str(home_path) + '\n' + \
                  ' scr_path    = '+str(scr_path) + '\n'
        logfile.write(log_str)

        log_str = '\n fms simulation keywords\n' + \
                    ' ----------------------------------------\n'
        for k,v in glbl.fms.items():
            log_str += ' {0:20s} = {1:20s}\n'.format(str(k),str(v))
        log_str += ' {0:20s} = {1:20s}\n'.format('coup_thresh',str(glbl.coup_thresh))
        log_str += ' {0:20s} = {1:20s}\n'.format('sij_thresh',str(glbl.sij_thresh))
        logfile.write(log_str)    

        if glbl.fms['interface'] == 'columbus':
            out_key = glbl.columbus
        elif glbl.fms['interface'] == 'vibronic':
            out_key = glbl.vibronic
        elif glbl.fms['interface'] == 'boson_model_diabatic':
            out_key = glbl.boson
        else:
            out_key = dict()  

        log_str = '\n '+str(glbl.fms['interface'])+' simulation keywords\n'
        log_str += ' ----------------------------------------\n' 
        for k,v in out_key.items():
            log_str += ' {0:20s} = {1:20s}\n'.format(str(k),str(v))
        logfile.write(log_str)

        log_str = '\n ***********\n' + \
                    ' propagation\n' + \
                    ' ***********\n\n'
        logfile.write(log_str)

    log_format['general']     = '   ** {:60s} **\n'
    log_format['t_step']      = ' > time: {0:12.2f} step:{1:8.2f} [{2:4d} trajectories]\n\n'
    log_format['coupled']     = '  -- in coupling regime -> timestep reduced to {:8.2f}\n'
    log_format['new_step']    = '   -- step failed, re-trying with new time step: {:8.2f}\n'
    log_format['spawn_start'] = '  -- spawing: trajectory {0:4d}, state {1:2d} --> state {2:2d}\n' +\
                                '              time      coup   overlap     spawn\n'
    log_format['spawn_step']  = '      {0:12.2f} {1:9.4f} {2:9.4f} {3:>9s}\n'
    log_format['spawn_back']  = '      back propagating:  {0:12.2f}\n'
    log_format['spawn_bad_step']= '       --> could not spawn: {:40s}\n'
    log_format['spawn_success'] = ' - spawn successful, new trajectory created at {0:12.2f}\n'
    log_format['spawn_failure'] = ' - spawn failed, cannot create new trajectory\n'
    log_format['complete']      = ' ------- simulation completed --------\n'
    log_format['timings' ]      = '\n timing information\n' + \
                                    ' ------------------\n' + \
                                    ' propagate (non-spawning): {0:12.6f}\n' + \
                                    ' spawning:                 {1:12.6f}\n' + \
                                    ' hamiltonian construction: {2:12.6f}\n' + \
                                    ' centroid evaluation:      {3:12.6f}\n' + \
                                    ' total runtime:            {4:12.6f}\n'

#
# Appends a row of data, formatted by entry 'fkey' in formats to file
# 'filename'
#
def print_traj_row(tid,fkey,data):
    global scr_path, tkeys, tfile_names, dump_header, dump_format
    filename = scr_path+'/'+tfile_names[tkeys[fkey]]+'.'+str(tid)
   
    if not os.path.isfile(filename):
        with open(filename, "x") as outfile:
            outfile.write(dump_header[tkeys[fkey]])
            outfile.write(dump_format[tkeys[fkey]].format(*data))
    else:
        with open(filename, "a") as outfile:
            outfile.write(dump_format[tkeys[fkey]].format(*data))
    return

#
# Appends a row of data, formatted by entry 'fkey' in formats to file
# 'filename'
#
def print_bund_row(fkey,data):
    global scr_path, bkeys, bfile_names, dump_header, dump_format
    filename = scr_path+'/'+bfile_names[bkeys[fkey]]

    if not os.path.isfile(filename):
        with open(filename, "x") as outfile:
            outfile.write(dump_header[bkeys[fkey]])
            outfile.write(dump_format[bkeys[fkey]].format(*data))
    else:
        with open(filename, "a") as outfile:
            outfile.write(dump_format[bkeys[fkey]].format(*data))
    return

#
# prints a matrix to file with a time label
#
def print_bund_mat(time,fname,mat):
     global scr_path    
     filename = scr_path+'/'+fname

     with open(filename,"a") as outfile:
         outfile.write('{:9.2f}\n'.format(time))
         outfile.write(np.array2string(mat)+'\n')
     return

#
# print a string to the log file
#
def print_fms_logfile(otype,data):
     global log_format

     if otype not in log_format:
         print("CANNOT WRITE otype="+str(otype)+'\n')
         return

     filename = home_path+'/fms.log'      
     with open(filename,'a') as logfile:
         logfile.write(log_format[otype].format(*data))
     return


#----------------------------------------------------------------------------
#
# Read geometry.dat and hessian.dat files
#
#----------------------------------------------------------------------------
#
# read in geometry.dat: position and momenta
#
def read_geometry():
    global home_path 
    geom_data = []
    mom_data  = [] 
    width_data = []

    gm_file = open(home_path+'/geometry.dat','r',encoding='utf-8')
    # comment line
    gm_file.readline()
    # number of atoms
    natm = int(gm_file.readline()[0]) 

    # read in geometry
    for i in range(natm):
        geom_data.append(gm_file.readline().rstrip().split())

    # read in momenta
    for i in range(natm):
        mom_data.append(gm_file.readline().rstrip().split())

    # read in widths, if present
    for i in range(natm):
        ln = gm_file.readline()
        if ln is None:
            break
        width_data.append(float(ln.rstrip()))

    gm_file.close()

    return geom_data,mom_data,width_data

#
# Read a hessian matrix (not mass weighted)
#
def read_hessian():
    global home_path

    hessian = np.loadtxt(home_path+'/hessian.dat',dtype=np.float) 
    return hessian

#----------------------------------------------------------------------------
#
# FMS summary output file
#
#----------------------------------------------------------------------------
def cleanup():

    # print timing information
    print_fms_logfile('complete',[])

    t_order = ['propagate','spawning','hamiltonian','centroid']
    data = [glbl.timings[t_type] for t_type in t_order]
    print_fms_logfile('timings',data)

    # move trajectory summary files to an output directory in the home area
    odir = home_path+'/output'
    if os.path.exists(odir):
        shutil.rmtree(odir)
        os.makedirs(odir)
        
    # move trajectory files
    for key,fname in tfile_names.items():
        shutil.move(scr_path+'/'+fname+'.*',odir)

    # move trajectory files
    for key,fname in bfile_names.items():
        shutil.move(scr_path+'/'+fname,odir)

    return
