import os
import numpy as np
import src.fmsio.glbl as glbl

tkeys       = ['traj_dump', 'ener_dump',   'coup_dump', 'dipole_dump', 'quad_dump',
               'tran_dump', 'charge_dump', 'grad_dump']
bkeys       = ['pop_dump', 'bener_dump', 'hmat_dump', 'smat_dump', 'spawn_dump']
formats     = dict()
headers     = dict()
tfile_names = dict()
bfile_names = dict()

# 
# Read the fms.input file. This contains variables related
# to the running of the dynamics simulation.
#
def read_input_files():

    #
    # Read fms.input. This contains general simulation variables
    # 
    kwords = read_namelist('fms.input')
    for k,v in kwords.items():
        if k in glbl.fms:
            glbl.fms[k] = v
        else:
            print("Variable "+str(k)+" in fms.input unrecognized. Ignoring...")
    glbl.working_dir = os.getcwd()

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
    else:
        print("Interface: "+str(glbl.fms['interface'])+" not recognized.")

    return

#
# Reads a namelist style input, returns results in dictionary
#
def read_namelist(filename):
    kwords = dict()
    f = open(filename,'r',encoding="utf-8")
    for line in f:
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
    f.close()
    return kwords

#
# initialize all the ouptut format descriptors
#
def init_fms_output():
    global tkeys, bkeys, headers, formats, tfile_names, bfile_names

    np = glbl.num_particles
    nd = glbl.dim_particles
    ns = glbl.n_states
    space = ' '

    # trajectory output
    dstr = ("x","y","z")
    arr1 = [' pos' + str(i+1) + '.' + dstr[x] for i in range(np) for x in range(nd)]
    arr2 = [' mom' + str(i+1) + '.' + dstr[x] for i in range(np) for x in range(nd)]
    tfile_names[tkeys[0]] = 'trajectory'
    headers[tkeys[0]]     = 'Time  ' + ' '.join(arr1) + ' '.join(arr2) + ' Phase' + ' Amp ' + ' State'
    formats[tkeys[0]]     = '{0:8.2f} '+ ' '+\
                             space.join('{'+str(i)+':10.6f}' for i in range(1,np*nd+1))         + \
                             space.join('{'+str(i)+':10.6f}' for i in range(np*nd+1,2*np*nd+1)) + \
                             space.join('{'+str(i)+':8.4f}'  for i in range(2*np*nd+1,2*np*nd+4))

    # potential energy
    arr1 = ['potential.'+str(i+1) for i in range(ns)]
    tfile_names[tkeys[1]] = 'poten'
    headers[tkeys[1]]     = ' Time ' + ' '.join(arr1)
    formats[tkeys[1]]     = '{0:8.2f} ' + ' ' + \
                             space.join('{'+str(i)+':16.10f}' for i in range(1,ns+1)) 

    # coupling
    arr1 = ['coupling.'+str(i+1) for i in range(ns)]
    arr2 = ['  c * v .'+str(i+1) for i in range(ns)]
    tfile_names[tkeys[2]] = 'coupling'
    headers[tkeys[2]]     = '     Time' + ' '.join(arr1) + ' '.join(arr2)
    formats[tkeys[2]]     = '{0:8.2f} ' + ' ' + \
                             space.join('{'+str(i)+':10.5f}' for i in range(1,2*ns+1))

    # permanent dipoles
    arr1 = ['dip_st'+str(i+1)+'.'+dstr(j+1) for i in range(ns) for j in range(nd)]
    tfile_names[tkeys[3]] = 'dipole'
    headers[tkeys[3]]     = ' Time ' + ' '.join(arr1)
    formats[tkeys[3]]     = '{0:8.2f} ' + ' ' + \
                             space.join('{'+str(i)+':8.5f}' for i in range(1,ns*nd+1))

    # transition dipoles
    arr1 = ['td_st'+str(i+1)+str(j+1)+'.'+dstr(k+1) for i in range(ns) for j in range(i) for k in range(nd)]
    ncol = int(ns*(ns-1)*nd/2+1)
    tfile_names[tkeys[4]] = 'tr_dipole'
    headers[tkeys[4]]     = ' Time ' + ' '.join(arr1)
    formats[tkeys[4]]     = '{0:8.2f} '+ ' ' + \
                             space.join('{'+str(i)+':8.5f}' for i in range(1,ncol))

    # quadrupoles
    arr1 = ['quad_st'+str(i+1)+'.'+dstr(j)+dstr(j) for i in range(ns) for j in range(nd)]
    tfile_names[tkeys[5]] = 'quadrupole'
    headers[tkeys[5]]     = ' Time ' + ' '.join(arr1)
    formats[tkeys[5]]     = '{0:8.2f} ' + ' ' + \
                             space.join('{'+str(i)+':8.5f}' for i in range(1,ns*nd+1))

    #charges
    arr1 = ['part.'+str(i+1) for i in range(np)]
    tfile_names[tkeys[6]] = 'charge'
    headers[tkeys[6]]     = ' Time ' + ' '.join(arr1)
    formats[tkeys[6]]     = '{0:8.2f} ' + ' ' + \
                             space.join('{'+str(i)+':8.5f}' for i in range(1,np+1))

    # gradients
    arr1 = ['grad_part'+str(i+1)+'.'+dstr(j) for i in range(np) for j in range(nd)]
    tfile_names[tkeys[7]] = 'gradient'
    headers[tkeys[7]]     = ' Time ' + ' '.join(arr1)
    formats[tkeys[7]]     = '{0:8.2f} ' + ' ' + \
                             space.join('{'+str(i)+':8.5f}' for i in range(1,np*nd+1))

    # bundle output
    # adiabatic state populations
    arr1 = ['state '+str(i+1) for i in range(ns)]
    bfile_names[bkeys[0]] = 'n.dat'
    headers[bkeys[0]]     = ' Time ' + ' '.join(arr1) + ' Norm'
    formats[bkeys[0]]     = '{0:8.2f} '+  ' ' + \
                            space.join('{'+str(i)+':8.5f}' for i in range(1,ns+1)) + \
                            ' {'+str(ns+1)+':8.5f}'

    # the bundle energy
    arr1 = ('potential(QM)','potential(Cl.)','kinetic(QM)',
            'kinetic(Cl.)','total(QM)','total(Cl.)')
    bfile_names[bkeys[1]] = 'e.dat'
    headers[bkeys[1]]     = ' Time ' + ' '.join(arr1)
    formats[bkeys[1]]     = '{0:8.2f} ' + ' ' + \
                             space.join('{'+str(i)+':14.10f}' for i in range(1,6+1))

    # the hamiltonian matrix
    bfile_names[bkeys[2]] = 'h.dat'
    formats[bkeys[2]]     = ''

    # the trajectory overlap matrix
    bfile_names[bkeys[3]] = 's.dat'
    formats[bkeys[3]]     = ''    

    # the spawn log
    arr1 = ('time (entry)','time(spawn)','time(exit)',
            'parent','state','child','state',
            'ke(parent)','pot(parent)','ke(child)','pot(child)')
    bfile_names[bkeys[4]] = 'spawn.dat'
    headers[bkeys[4]]     = ' '.join(arr1)
    formats[bkeys[4]]     = '{0:8.2f} {1:8.2f} {2:8.2f} '  + \
                            '{3:4d} {3:4d} {3:4d} {3:4d} ' + \
                            '{12.8f} {12.8f} {12.8f} {12.8f}'

#
# Appends a row of data, formatted by entry 'fkey' in formats to file
# 'filename'
#
def print_traj_row(tid,fkey,data):
    global tkeys,tfile_names,headers,formats
    filename = tfile_names[tkeys[fkey]]+'.'+str(tid)

    if not os.path.isfile(filename):
        with open(filename, "x") as outfile:
            outfile.write(headers[tkeys[fkey]])
            outfile.write(formats[tkeys[fkey]].format(data))
    else:
        with open(filename, "a") as outfile:
            outfile.write(formats[tkeys[fkey]].format(data))

#
# Appends a row of data, formatted by entry 'fkey' in formats to file
# 'filename'
#
def print_bund_row(fkey,data):
    global bkeys,bfile_names,headers,formats
    filename = bfile_names[bkeys[fkey]]

    if not os.path.isfile(filename):
        with open(filename, "x") as outfile:
            outfile.write(headers[bkeys[fkey]])
            outfile.write(formats[bkeys[fkey]].format(data))
    else:
        with open(filename, "a") as outfile:
            outfile.write(formats[bkeys[fkey]].format(data))

#----------------------------------------------------------------------------
#
# Read geometry.dat and hessian.dat files
#
#----------------------------------------------------------------------------
#
# read in geometry.dat: position and momenta
#
def read_geometry():
    geom_data = [] 

    gm_file = open(glbl.working_dir+'/geometry.dat','r',encoding='utf-8')
    # comment line
    gm_file.readline()
    # number of atoms
    natm = int(gm_file.readline()[0]) 

    # read in geometry
    for i in range(natm):
        geom_data.append(gm_file.readline().rstrip().split())

    # read in momenta
    for i in range(natm):
        geom_data[i].extend(gm_file.readline().rstrip().split())

    gm_file.close()

    return geom_data

#
# Read a hessian matrix (not mass weighted)
#
def read_hessian():
    hessian = np.loadtxt(glbl.working_dir+'/hessian.dat',dtype='float') 
    return hessian

#----------------------------------------------------------------------------
#
# FMS summary output file
#
#----------------------------------------------------------------------------
def cleanup():
    print("timings info not implemented")

