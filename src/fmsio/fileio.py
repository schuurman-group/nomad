import os
import numpy as np
import src.fmsio.glbl as glbl

output_path = ''
tkeys       = ['traj_dump', 'ener_dump',   'coup_dump', 'dipole_dump', 'secm_dump',
               'tran_dump', 'apop_dump', 'grad_dump']
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
    global output_path

    # set a sensible default for output_path
    output_path = os.environ['TMPDIR']

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

    np = int(glbl.fms['num_particles'])
    nd = int(glbl.fms['dim_particles'])
    ns = int(glbl.fms['n_states'])
    dstr = ("x","y","z")

    # trajectory output
    arr1 = ['      pos' + str(i+1) + '.' + dstr[x] for i in range(np) for x in range(nd)]
    arr2 = ['      mom' + str(i+1) + '.' + dstr[x] for i in range(np) for x in range(nd)]
    tfile_names[tkeys[0]] = 'trajectory'
    headers[tkeys[0]]     = '     Time' + ''.join(arr1) + ''.join(arr2) + '     Phase' + \
                            '   Re[Amp]' + '   Im[Amp]' + ' Norm[Amp]'+'     State\n'
    formats[tkeys[0]]     = '{0:9.2f}'+ \
                             ''.join('{'+str(i)+':12.6f}' for i in range(1,np*nd+1))         + \
                             ''.join('{'+str(i)+':12.6f}' for i in range(np*nd+1,2*np*nd+1)) + \
                             ''.join('{'+str(i)+':10.6f}'  for i in range(2*np*nd+1,2*np*nd+6))+'\n'

    # potential energy
    arr1 = ['     potential.'+str(i) for i in range(ns)]
    tfile_names[tkeys[1]] = 'poten'
    headers[tkeys[1]]     = '     Time' + ''.join(arr1)+'\n'
    formats[tkeys[1]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':16.10f}' for i in range(1,ns+1)) + '\n' 

    # coupling
    arr1 = ['  coupling.'+str(i) for i in range(ns)]
    arr2 = ['    c * v .'+str(i) for i in range(ns)]
    tfile_names[tkeys[2]] = 'coupling'
    headers[tkeys[2]]     = '     Time' + ''.join(arr1) + ''.join(arr2) + '\n'
    formats[tkeys[2]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':12.5f}' for i in range(1,2*ns+1)) + '\n'

    # permanent dipoles
    arr1 = ['   dip_st'+str(i)+'.'+dstr[j] for i in range(ns) for j in range(nd)]
    tfile_names[tkeys[3]] = 'dipole'
    headers[tkeys[3]]     = '     Time' + ''.join(arr1) + '\n'
    formats[tkeys[3]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':12.5f}' for i in range(1,ns*nd+1)) + '\n'

    # transition dipoles
    arr1 = ['  td_s'+str(j)+'.s'+str(i)+'.'+dstr[k] for i in range(ns) for j in range(i) for k in range(nd)]
    ncol = int(ns*(ns-1)*nd/2+1)
    tfile_names[tkeys[4]] = 'tr_dipole'
    headers[tkeys[4]]     = '     Time' + ''.join(arr1) + '\n'
    formats[tkeys[4]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':12.5f}' for i in range(1,ncol)) + '\n'

    # second moments 
    arr1 = ['   sec_s'+str(i)+'.'+dstr[j]+dstr[j] for i in range(ns) for j in range(nd)]
    tfile_names[tkeys[5]] = 'sec_mom'
    headers[tkeys[5]]     = '     Time' + ''.join(arr1) + '\n'
    formats[tkeys[5]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':12.5f}' for i in range(1,ns*nd+1)) + '\n'

    # atomic populations
    arr1 = ['    st'+str(i)+'_p'+str(j+1) for i in range(ns) for j in range(np)]
    tfile_names[tkeys[6]] = 'atom_pop'
    headers[tkeys[6]]     = '     Time' + ''.join(arr1) + '\n'
    formats[tkeys[6]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':10.5f}' for i in range(1,ns*np+1)) + '\n'

    # gradients
    arr1 = ['  grad_part'+str(i+1)+'.'+dstr[j] for i in range(np) for j in range(nd)]
    tfile_names[tkeys[7]] = 'gradient'
    headers[tkeys[7]]     = '     Time' + ''.join(arr1) + '\n'
    formats[tkeys[7]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':14.8f}' for i in range(1,np*nd+1)) + '\n'

    # bundle output
    # adiabatic state populations
    arr1 = ['     state.'+str(i) for i in range(ns)]
    bfile_names[bkeys[0]] = 'n.dat'
    headers[bkeys[0]]     = '     Time' + ''.join(arr1) + '      Norm' + '\n'
    formats[bkeys[0]]     = '{0:9.2f}' + \
                            ''.join('{'+str(i)+':12.6f}' for i in range(1,ns+1)) + \
                            '{'+str(ns+1)+':10.6f}\n'

    # the bundle energy
    arr1 = ('   potential(QM)','  potential(Cl.)','     kinetic(QM)',
            '    kinetic(Cl.)','       total(QM)','      total(Cl.)')
    bfile_names[bkeys[1]] = 'e.dat'
    headers[bkeys[1]]     = '     Time' + ''.join(arr1) + '\n'
    formats[bkeys[1]]     = '{0:9.2f}' + \
                             ''.join('{'+str(i)+':16.10f}' for i in range(1,6+1)) + '\n'

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
    global output_path, tkeys, tfile_names, headers, formats
    filename = output_path+'/'+tfile_names[tkeys[fkey]]+'.'+str(tid)
   
    if not os.path.isfile(filename):
        with open(filename, "x") as outfile:
            outfile.write(headers[tkeys[fkey]])
            outfile.write(formats[tkeys[fkey]].format(*data))
    else:
        with open(filename, "a") as outfile:
            outfile.write(formats[tkeys[fkey]].format(*data))

#
# Appends a row of data, formatted by entry 'fkey' in formats to file
# 'filename'
#
def print_bund_row(fkey,data):
    global output_path, bkeys, bfile_names, headers, formats
    filename = output_path+'/'+bfile_names[bkeys[fkey]]

    if not os.path.isfile(filename):
        with open(filename, "x") as outfile:
            outfile.write(headers[bkeys[fkey]])
            outfile.write(formats[bkeys[fkey]].format(*data))
    else:
        with open(filename, "a") as outfile:
            outfile.write(formats[bkeys[fkey]].format(*data))

#
# prints a matrix to file with a time label
#
def print_bund_mat(time,fname,mat):
     global output_path    
     filename = output_path+'/'+fname

     with open(filename,"a") as outfile:
         outfile.write('{:9.2f}\n'.format(time))
         outfile.write(np.array2string(mat)+'\n')

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

