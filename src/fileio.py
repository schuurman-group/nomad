import variable

formats = dict()
headers = dict()
traj_keys = ('trajectory','potential','coupling','dipoles','quadrupoles','charges')
bund_keys = ('pop','energy')

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
            key = key.strip()
            value = value.strip()
            try:
                kwords[key] = float(value)
                if kwords[key].is_integer():
                    kwords[key] = int(value)
            except ValueError:
                pass
            if key not in kwords:
                print("setting "+str(key)+" to a string")
                kwords[key] = value
    f.close()
    return kwords

# 
# Read the fms.input file. This contains variables related
# to the running of the dynamics simulation.
#
def read_fms_input():
    # 
    kwords = read_namelist('fms.input')
    for k,v in kwords.items():
        if k in variable.fms:
            variable.fms[k] = v
        else:
            print("Variable "+str(k)+" in fms.input unrecognized. Ignoring...")

#
# initialize all the ouptut format descriptors
#
def init_fms_output():
    np = variable.num_particles
    nd = variable.dim_particles
    ns = variable.n_states
    # trajectory
    dstr = ("x","y","z")
    arr1 = [' pos' + str(x+1) + '.' + dstr[y] for x in range(np) for y in range(nd)]
    arr2 = [' mom' + str(x+1) + '.' + dstr[y] for x in range(np) for y in range(nd)]
    headers[traj_keys[0]] = 'Time  ' + ' '.join(arr1) + ' '.join(arr2) + ' Phase' + ' Amp ' + ' State'
    formats[traj_keys[0]] = ''
    # potential energy
    arr1 = ['potential.'+str(i+1) for i in range(ns)]
    headers[traj_keys[1]] = ' Time ' + ' '.join(arr1)
    formats[traj_keys[1]] = ''
    # coupling
    arr1 = ['coupling.'+str(i+1) for i in range(ns)]
    arr2 = ['  c * v .'+str(i+1) for i in range(ns)]
    headers[traj_keys[2]] = ' Time ' + ' '.join(arr1) + ' '.join(arr2)
    formats[traj_keys[2]] = '' 
    # dipoles
    arr1 = ['d'+str(i+1)+str(j+1)+'.'+dstr(k+1) for i in range(ns) for j in range(ns) for k in range(nd)]
    headers[traj_keys[3]] = ' Time ' + ' '.join(arr1)
    formats[traj_keys[3]] = ''
    # quadrupoles
    arr1 = ['q'+str(i+1)+'.'+dstr(j)+dstr(j) for i in range(ns) for j in range(nd)]
    headers[traj_keys[4]] = ' Time ' + ' '.join(arr1)
    formats[traj_keys[4]] = ''
    #charges
    arr1 = ['part.'+str(i+1) for i in range(np)]
    headers[traj_keys[5]] = ' Time ' + ' '.join(arr1)
    formats[traj_keys[5]] = ''
    # bundle output
    arr1 = ['state '+str(i+1) for i in range(ns)]
    headers['pop']     = ' Time ' + ' '.join(arr1) + ' Norm'
    formats['pop']     = ''
    arr1 = ('potential(QM)','potential(Cl.)','kinetic(QM)',
            'kinetic(Cl.)','total(QM)','total(Cl.)')
    headers['energy']  = ' Time ' + ' '.join(arr1)
    formats['energy']  = ''
    arr1 = ('time (entry)','time(spawn)','time(exit)','parent','state',
            'child','state','ke(parent)','pot(parent)','ke(child)','pot(child)')
    headers['spawn']  = ' '.join(arr1)
    formats['spawn']  = ''

#
# Appends a row of data, formatted by entry 'fkey' in formats to file
# 'filename'
#
def print_row(filename,fkey,data):
    if not os.path.isfile(filename):
        with open(filename, "x") as outfile:
            outfile.write(headers[fkey])
            outfile.write(formats[fkey].format(data))
    else:
        with open(filename, "a") as outfile:
            outfile.write(formats[fkey].format(data))
    outfile.close()

#
# Update the trajectory and bundle logs at each time step
#
#def update_logs():
#    # loop over trajectory information
#    for i in range(master.ntraj)
#        tid = '.'+str(master.trajectory[i].tid)
#        for j in range(len(traj_keys))
#            traj_data = pull_traj_data(i,traj_keys[j])
#            print_row(traj_keys[j]+tid,traj_keys[j],traj_data)
#
#    # update bundle level information
#    for i in range(len(bund_keys))
#        bun_data = pull_bundle_data(bund_keys[i])
#        print_row(bund_keys[i]+'.log',bund_keys[i],bun_data)

#
# Update the spawn log after a spawn
#    
def update_spawn_log(spawn_data):
    print_row('spawn.log','spawn',spawn_data)

#
#
#
def update_output():
    print("current_time="+str(variable.fms['current_time']))

#------------------------------------------------------------------------------
#
# FMS summary output file
#
#------------------------------------------------------------------------------
def cleanup():
    print("timings info not implemented")

