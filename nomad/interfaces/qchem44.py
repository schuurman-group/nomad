"""
Routines for running a q-chem computation.
"""
import sys
import os
import shutil
import subprocess
import numpy as np
import nomad.core.log as log
import nomad.core.glbl as glbl
import nomad.core.surface as surface
import nomad.core.atom_lib as atom_lib
import nomad.math.constants as constants
import nomad.core.trajectory as trajectory
import nomad.integrals.centroid as centroid

#path to qchem Input file:
input_file = glbl.qchem44['input_file']
#number of coordinites:
n_dim  = 3
#number of atoms:
natm = 0
#Multiplicity for Q-Chem calculations:
multiplicity =  glbl.qchem44['multiplicity']
try:
    s_squareds   = [((m-1) / 2)**2 + ((m-1)/2)  for m in multiplicity]
except TypeError:
    s_squareds = [((multiplicity-1)/2)**2 + ((multiplicity-1)/2)]

state_cache  = {}
n_states = glbl.properties['n_states']
#The number of roots to calculate in the q-chem calculation - this may need to change:
n_roots = 0

#----------------------------------------------------------------
#
# Functions called from interface object
#
#----------------------------------------------------------------
def init_interface():
    """Initializes the qchem calculation from the qchem input."""
    global input_path, work_path, a_sym, natm, n_cart, qchem_states 
    # setup working directories
    # input and restart are shared
    input_path    = glbl.paths['cwd']+'/input'
    restart_path  = glbl.paths['cwd']+'/restart'
    # ...but each process has it's own work directory
    work_path     = glbl.paths['cwd']+'/work'

    #If the work directoy doesn't exist, create it:
    if not os.path.exists(work_path):
        os.mkdir(work_path)


    # set atomic symbol, number, mass,
    natm    = len(glbl.properties['crd_labels']) // n_dim
    a_sym   = glbl.properties['crd_labels'][::n_dim]
    n_cart  = natm * n_dim
    a_data  = []



    # we need to go through this to pull out the atomic numbers for
    # correct writing of input
    for i in range(natm):
        if atom_lib.valid_atom(a_sym[i]):
            a_data.append(atom_lib.atom_data(a_sym[i]))
        else:
            raise ValueError('Atom: '+str(a_sym[i])+' not found in library')

    #The initial spin-flip states to use:
    if glbl.qchem44['sf_states'] is not None:
        qchem_states = glbl.qchem44['sf_states']
    else:
        qchem_states = [st for st in range(n_states)]


    #A Cache to store which states we are using:
    state_cache['states']    = qchem_states
    state_cache['strengths'] = np.zeros(n_states)
    state_cache['ssquareds'] = np.zeros(n_states)

    #Set up qchem scratch directory:
    os.environ['QCSCRATCH'] = work_path

    if os.path.exists(work_path):
        shutil.rmtree(work_path)
    os.makedirs(work_path)

    if glbl.mpi['rank'] == 0:
        if os.path.exists(restart_path):
            shutil.rmtree(restart_path)
        os.makedirs(restart_path)

    # make sure process 0 is finished populating the input directory
    if glbl.mpi['parallel']:
        glbl.mpi['comm'].barrier()




def evaluate_trajectory(traj, time=None):
    global qchem_states, state_strengths, state_ssquareds, n_roots

    label           = traj.label
    if label < 0:
        print('evaluate_trajectory called with ' +
              'id associated with centroid, label=' + str(label))

    # create surface object to hold potential information
    qchem_surf = surface.Surface()
    qchem_surf.add_data('geom', traj.x())

    #If we are running this calculation again for the same time step, use the 'cached cache', otherwise, just pull them from the regular cache:

    if 'time' in state_cache and time == state_cache['time']:
        qchem_states    = state_cache['prev_states']
        state_ssquareds = state_cache['prev_ssquareds']
        state_strengths = state_cache['prev_strengths']
    else:    
        #cache the cache, and get the current statesi, and their strengths and multiplicities:
        state_cache['prev_states']    = state_cache['states']
        state_cache['prev_ssquareds'] = state_cache['ssquareds']
        state_cache['prev_strengths'] = state_cache['strengths']
        qchem_states    = state_cache['states']
        state_ssquareds = state_cache['ssquareds']
        state_strengths = state_cache['strengths']
    


    #Run a qchem calculation:
    #Read through the output, if thee are errors, we have to run qcham again:
    try:
        run_qchem(traj, time, qchem_surf)
    except NotEnoughStatesError:
   #    #Increase the number of roots we're calculating and try again:
        if n_roots < 30:
            n_roots +=5
            log.print_message('warning',['Could not find all the states, increasing the number of roots calculated to ' + str(n_roots)]) 
            evaluate_trajectory(traj,time)
        else:
            sys.exit('could not find the right states')
    except StatesDontMatchError:
        #The states should be updated, so we can just run it again:
        run_qchem(traj,time)


    #re-cache the updated states, strengths, multiplicities:
    state_cache['states']    = qchem_states
    state_cache['ssquareds'] = state_ssquareds
    state_cache['strengths'] = state_strengths
    #also cache the time, so we know if we've been here before:
    state_cache['time'] = time

    return qchem_surf


def evaluate_centroid(cent, t=None):
    """Evaluates all requested electronic structure information at a
    centroid."""
    global n_cart

    label   = cent.label
    nstates = cent.nstates

    if label >= 0:
        print('evaluate_centroid called with ' +
              'id associated with trajectory, label=' + str(label))
    return evaluate_trajectory(cent, t=None)


def evaluate_coupling(traj):
    """evaluate coupling between electronic states"""

    nstates = traj.nstates
    state   = traj.state

    # Effective Coupling for aidbatic repreentation: nac projected onto velocity
    if glbl.methods['surface'] == 'adiabatic':
        coup = np.zeros((nstates, nstates))
        vel  = traj.velocity()
        for i in range(nstates):
            if i != state:
                coup[state, i] = np.dot(vel, traj.derivative(state,i))
                coup[i, state] = -coup[state, i]
        traj.pes.add_data('coupling', coup)
    #Diabatic representation: 
    


def append_log(label, listing_file, time):
    """Grabs key output from columbus listing files.
    Useful for diagnosing electronic structure problems.
    """
    # check to see if time is given, if not -- this is a spawning
    # situation
    if time is None:
        tstr = 'spawning'
    else:
        tstr = str(time)

    
    # open the running log for this process
    #log_file = open(glbl.home_path+'/columbus.log.'+str(glbl.mpi['rank']), 'a')
    #log_file = open('qchem.log.'+str(glbl.mpi['rank']), 'a')
    #log_file.close()


def run_qchem(traj, time, qchem_surf, jobs_failed = 0):
    """Runs the qchem calculation:"""
    
    #don not run the calculation if too many jobs have failed in a row:
    if jobs_failed > 3:
        sys.exit('Q-chem has failed too many times. Giving up.' )

    state        = traj.state
    n_states     = traj.nstates
    current_outs = os.listdir('input') 
    output_file  = input_file.replace( '.inp',  str(len(current_outs))+'_' + str(time) + '.out') 

    #Run The Calculation:
    subprocess.run(['qchem',  write_qchem_input(traj, time, jobs_failed), output_file, 'save'])  
    
    
    #Check to make sure the job didn't fail:
    f = open(output_file, 'r')
    output = f.read()
    f.close()

    if 'fatal error' in output or 'failure' in output:
        log.print_message('warning', ['Q-chem job failed, changing the input file and trying again.'])
        #Re-run the calculation
        run_qchem(traj, time, qchem_surf,  jobs_failed+1)

    #If we get this far, read the output file - this is where the job could potentially fail:
    read_qchem_output(traj, output_file, qchem_surf)
    
    





def write_qchem_input(traj, time, jobs_failed):
    """Writes a qchem input file"""
    global n_roots

    label    = traj.label
    state    = traj.state
    n_states = traj.nstates
    
    #read the input:
    f = open(input_file, 'r')
    qchem_input = f.read()
    f.close()
    
    #make the whole thing lowercase, since qchem is not case sensitive:
    qchem_input = qchem_input.casefold()

    open_input = open(input_file, 'w')
    
    #First, split the input into two (or more) jobs:
    joblist = qchem_input.split('@@@')

    #there should be at least two jobs (force and sp energy) in order to get all the information we need. Check this:
    if len(joblist)<2:
        #Either ask the user to write a second job, or just copy over the input from the first job and change what we need to to get the data we need:
        #exit for now, but write this later:
        sys.exit('need two jobs')

    for job in joblist:
        #variables relevant to both jobs:
        #These are the variables to change if qchem fails:
        max_scf_cycles   = 50
        scf_algorithm    = 'diis' 
        
        if job is not joblist[0]:
            open_input.write('@@@')

        #Split each job into sections:
        job_sections = job.split('$')
        
        for section in job_sections:
            section_lines = section.split('\n')
            if 'molecule' in section:
                open_input.write('\n$molecule\n')
                
                #The first line of the input is the multiplicity and charge - copy from previous, for now, but this might have to change:
                open_input.write(section_lines[1] + '\n')

                #Write all the geometry - overwrite whatever is there already:
                fmt = '{:3s}{:14.8f}{:14.8f}{:14.8f}\n'
                for i in range(natm):
                    open_input.write(fmt.format(a_sym[i], traj.x()[n_dim*i], traj.x()[(n_dim*i)+1], traj.x()[(n_dim*i)+2]))

                #END OF MOLECULE SECTION

            if 'rem' in section:
                jobtype = ''
                #start writing this section:
                open_input.write('$')    
                for line in section_lines:
                    #Check jobtype - should be either single-point energy of force: 
                    if 'jobtype' in line:
                        jobtype = line.split()[-1]
                        open_input.write(line + '\n')
                        continue

                    if 'max_scf_cycles' in line:
                        scf_max_cycles = str(line.split()[-1])
                        open_input.write(line + '\n')
                        continue

                    if 'scf_algorithm' in line:
                        scf_algorithm = line.split()[-1]
                        open_input.write(line + '\n')
                        continue

                    if 'cis_state_deriv' in line:
                        if state > 0:
                            open_input.write('cis_state_deriv    ' + str(qchem_states[state]) + '\n')
                        continue
                        
                    if 'cis_n_roots' in line:
                        input_n_roots = int(line.split()[-1])
                        if input_n_roots > n_roots:
                            n_roots = input_n_roots
                        open_input.write('cis_n_roots      ' + str(n_roots)+ '\n')
                        continue
                    #Print all the remaining lines to the input file:
                    else:
                        if line is not '':
                            open_input.write(line + '\n')

                if 'cis_state_deriv' not in section and state > 0:
                    #Write it to the input:
                    open_input.write('cis_state_deriv    ' + str(qchem_states[state]) + '\n')
                if 'input_bohr' not in section:
                    open_input.write('input_bohr      true\n')
                
                #Everything required in the sp calculation (coupling, potential):
                if jobtype == 'sp' or 'jobtype' not in section:

                    #add the derivative coupling section:
                    if 'cis_der_couple' not in section:
                        #Add this section:
                        open_input.write('cis_deriv_couple    true\n')   
                    if 'cis_der_numstate' not in section:
                    #add this section:
                        open_input.write('cis_der_numstate    ' + str(n_states) + '\n')
                    
                    #Now, if the previous job failed, change some things before running again - maximum of 3 times:
                if jobs_failed > 0:
                    #try increasing the number of scf cycles, to a maximum:
                    if  max_scf_cycles < 200:
                        max_scf_cycles += 50

                            
                    #if that doesn't work, try another scf algorithm:
                    elif scf_algorithm is not 'rca_diis':
                        scf_algorithm = 'rca_diis'
                        
                        #Write these things to the input file:
                        open_input.write('max_scf_cycles    ' + str(max_scf_cycles) + '\n')
                        open_input.write('scf_algorithm     ' + scf_algorithm + '\n')

                    #END OF REM SECTION


            if 'derivative_coupling' in section:

                #start writing this section:
                open_input.write('$derivative_coupling\n')
                #The comment line:
                open_input.write(section.split('\n')[1] + '\n')
                coup_states = [str(st) for st in qchem_states]
                open_input.write(' '.join(coup_states) + '\n')
                #END OF DERIVATIVE COUPLING SECTION

            #To close all the sections:
            if 'end' in section.casefold():
                    open_input.write('$end\n')
    #done writing input
    open_input.close()
    
    return input_file


def read_qchem_output( traj, output_file, qchem_surf):
    """Reads the Q-Chem output file and determines if the calculation needs to be run again in order to get more information."""
    global qchem_states, state_ssquareds, state_strengths

    state         = traj.state
    n_states      = traj.nstates


    #to read the output file:
    f = open(output_file)
    output = f.readlines()
    f.close()
    
    potential  = np.zeros(n_dim) 
    derivative = np.zeros((n_cart, traj.nstates, traj.nstates))
    

   

    #go through the file and pull out data when it's found:
    #split the output up into lines:
    for line in output:
        #Make two arrays: One of multiplicities, one of state energies, and another of their corresponding s^2 values:
                          
    #First, Look through the energy calculation to make sure we have the right states (and enough of them) - this is only required for sf-dft:
        if 'SF-DFT Excitation Energies' in line:
            st = 0
            #Go through the following lines:
            e_line = output.index(line)
            while 'Calculating Relaxed Density' not in output[e_line]:
                
                if 'Total energy for state' in output[e_line] and st < n_states:
                    state_number    =  output[e_line].split()[4].strip(':')
                    state_ssquared  = float(output[e_line + 1].split()[-1])
                    state_strength  = float(output[e_line + 3].split()[-1])
                    #print(state_number, state_ssquared) 
                    #Check if the multiplicity matches those specified in the input:
                    for s2 in s_squareds:
                        if s2 - glbl.qchem44['ssquared_tol'] < state_ssquared < s2 + glbl.qchem44['ssquared_tol']:
                            #Now, in case that another state seems to 'show up', the state must have either an s2 or strength in a specific window - lets say +/- 0.05, for now:
                            tol = 0.05
                            if (state_ssquareds[st] - tol < state_ssquared < state_ssquareds[st] + tol or 
                                state_strengths[st] - tol < state_strength < state_strengths[st] + tol or
                                    state_strengths[st] == 0): 
                                    
                                qchem_states[st]    = state_number
                                state_ssquareds[st] = state_ssquared 
                                state_strengths[st] = state_strength
                                st +=1
                e_line += 1

            #Check that we have all the states. If not, raise an error to run again:
            if st < n_states:
                raise NotEnoughStatesError('Not Enough States!')
        #Check if the state numbering has changed. If it has, we have to run the calculation again to get the right gradient and derivative couplings:
        if qchem_states.all() != state_cache['states'].all():

            raise StatesDontMatchError('States have changed')
        #potential:
        for a in range(n_states):
            st = qchem_states[a]
            if st == 0:
                if 'Total energy in the final basis set' in line:
                    potential[a] = float(line.split()[-1])
            if 'Total energy for state' in line and ' ' + str(st) + ':' in line:
                potential[a] = float(line.split()[-1])
        
        
        #Derivative - diagonal is the derivatives, off-diagonal is the derivative couplings
        if 'Gradient of' in line:
            #q-chem prints the gradient in 6 atoms per line - figure out how many lines there are:
            n_lines = int(np.ceil(natm/6))
            deriv = np.zeros((natm, n_dim))
            for grad_line in range(n_lines):
                for dim in range(n_dim): 
                    deriv[:,dim] = output[output.index(line) + 1 + ((grad_line + 1) * (dim + 1))].split()[1:]
            derivative[:, state, state] = deriv.flatten() 

        #Coupling:
        for a in range(n_states): 
        #range(n_states):
            for b in range(n_states): 
                st_1 = qchem_states[a]
                st_2 = qchem_states[b]
                #range(n_states):  
                if st_1 != st_2:
                    if 'between states ' + str(st_1) + ' and ' + str(st_2) in line:
                        deriv_coup =    np.zeros((natm, n_dim)) 
                        for atm in range(natm):
                            atm_coup = output[output.index(line) + atm + 11].split()[1:]
                            deriv_coup[atm] = atm_coup
                            
                        state_i = min(a, b)
                        state_j = max(a, b)
                        derivative[:, state_i, state_j] =   deriv_coup.flatten()      
                        derivative[:, state_j, state_i] = -1 * deriv_coup.flatten() 
                        norm_0 = np.linalg.norm(derivative[:, state, 0])
                        norm_1 = np.linalg.norm(derivative[:, state, 1])
                        norm_2 = np.linalg.norm(derivative[:, state, 2])


    #Add all the data:
    qchem_surf.add_data('potential', potential + glbl.properties['pot_shift'])
    qchem_surf.add_data('derivative', derivative)
    #qchem_surf.add_data('dipole', dipole)
    #atomic charges:





#For errors in reading the output file:

class Error(Exception):
    pass

class NotEnoughStatesError(Error):
    pass

class StatesDontMatchError(Error):
    pass






