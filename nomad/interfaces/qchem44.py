"""
Routines for running a q-chem computation.
"""
import sys
import os
import shutil
import subprocess
import numpy as np
import nomad.math.constants as constants
import nomad.core.glbl as glbl
import nomad.core.atom_lib as atom_lib
import nomad.core.trajectory as trajectory
import nomad.core.surface as surface
import nomad.integrals.centroid as centroid

#path to qchem Input file:
input_file = glbl.qchem44['input_file']
#number of coordinites:
n_dim  = 3
#number of atoms:
natm = 0


#----------------------------------------------------------------
#
# Functions called from interface object
#
#----------------------------------------------------------------
def init_interface():
    """Initializes the qchem calculation from the qchem input."""
    global a_sym, natm, n_cart, n_dim

    # setup working directories
    # input and restart are shared
    input_path    = glbl.paths['cwd']+'/input'
    restart_path  = glbl.paths['cwd']+'/restart'
    # ...but each process has it's own work directory
    work_path     = glbl.paths['cwd']+'/work'

    #If the work directoy doesn't exist, create it:
    if not os.path.exists(work_path):
        os.mkdir(work_path)

    #Load the qchem module
#    module('load', 'qchem/4.4')


    # set atomic symbol, number, mass,
    natm    = len(glbl.properties['crd_labels']) // n_dim
    a_sym   = glbl.properties['crd_labels'][::n_dim]
    

    #qk find a place to put this:
    n_cart = natm * n_dim

    a_data  = []
    # we need to go through this to pull out the atomic numbers for
    # correct writing of input
    for i in range(natm):
        if atom_lib.valid_atom(a_sym[i]):
            a_data.append(atom_lib.atom_data(a_sym[i]))
        else:
            raise ValueError('Atom: '+str(a_sym[i])+' not found in library')




    #Path to qchem: 
    print(os.environ['QC'])

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







def evaluate_trajectory(traj, t=None):

    label   = traj.label
    state   = traj.state
    n_states = traj.nstates
    

    if label < 0:
        print('evaluate_trajectory called with ' +
              'id associated with centroid, label=' + str(label))
    
    #Run a qchem calculation:
    run_qchem(traj)

    #Now - Read the Qchem input:
    # create surface object to hold potential information
    qchem_surf = surface.Surface()
    qchem_surf.add_data('geom', traj.x())

    #to read the output file:
    output_file = input_file.replace('.inp', '.out')
    f = open(output_file)
    output = f.readlines()
    f.close()
    print(qchem_surf.avail_data())
    
    
    if 'Excitation Energies' not in output:
        print('Could not find energies in output file. Check input file.')

    if 'Derivative Couplings' not in output:
        print('Could not find derivative couplings in output file. Check Input.')

    if 'Analytical Derivative' not in output:
        print('Could not find gradient in output file. Check Input.')
    
    if 'Hessian' not in output:
        print('Could not find hessian in output file. Check input.')

    potential = np.zeros(n_dim) 
    derivative = np.zeros((n_cart, traj.nstates, traj.nstates))
    #go through the file and pull out data when it's found:
    for line in output:
        #potential:
        for st in range(glbl.properties['n_states']):
            if st == 0:
                if 'Total energy in the final basis set' in line:
                    potential[st] = float(line.split()[-1]) 
            if 'Total energy for state   ' + str(st) in line:
                potential[st] = float(line.split()[-1])    
    
        #Derivative - diagonal is the derivatives, off-diagonal is the derivative couplings
        if 'Calculating analytic gradient of' in line:
            #q-chem prints the gradient in 6 atoms per line - figure out how many lines there are:
            n_lines = int(np.ceil(natm/6))
            deriv = []
            for atm in range(natm):
                atm_grad = []
                for grad_line in range(n_lines):             
                    if atm <(grad_line + 1) * 6:
                        for dim in range(n_dim):
                            atm_grad[dim] = output[output.index(line)+3 + dim + (grad_line * 4)].split()[atm - (grad_line * 6)]
                deriv += atm_grad
            derivative[:, state, state] = deriv

        #Coupling:
        for st_1 in range(n_states):
            for st_2 in range(n_states):  
                if st_1 != st_2:
                    if 'between states ' + str(st_1) + 'and ' + str(st_2) in line:
                        deriv_coup = []
                        for atm in range(natm):
                            atm_coup = output[line + atm + 11].split()
                            atm_coup.remove(atm_coup[0])
                            deriv_coup += atm_coup
                        derivative[:, st_1, st_2] = deriv_coup
    


    #Add all the data:
    qchem_surf.add_data('potential', potential)
    qchem_surf.add_data('derivative', derivative)
    #qchem_surf.add_data('hessian', hessian)
    #qchem_surf.add_data('dipole', dipole)
    #atomic charges:


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

    state_i = min(cent.states)
    state_j = max(cent.states)

    # create surface object to hold potential information
    qchem_surf = surface.Surface()
    qchem_surf.add_data('geom', cent.x())
    print( 'cemtroid', qchem_surf.avail_data())
    return qchem_surf


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
                coup[state,i] = np.dot(vel, traj.derivative(state,i))
                coup[i,state] = -coup[state,i]
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


def run_qchem(traj):
    """Writes the qchem input then runs calculation:"""
    label    = traj.label
    state    = traj.state
    n_states = traj.nstates

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

    #now, split up the jobs:
    for job in joblist:
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
                open_input.write('$rem\n')
                    
                for line in section_lines:

                    #Check jobtype - should be either single-point energy of force: 
                    if 'jobtype' in line:
                        jobtype = line.split()[-1]
                                                        
                    #Now update the input lines that need to change through the calculation:
                    if 'cis_state_deriv' in line and state > 0:
                        open_input.write('cis_state_deriv    ' + str(state) + '\n')
                        continue
                        
                    #Print all the remaining lines to the input file:
                    open_input.write(line + '\n')

                if 'cis_state_deriv' not in section and state > 0:
                    #Write it to the input:
                    open_input.write('cis_state_deriv    ' + str(state) + '\n')
                
                #Everything required in the sp calculation (coupling, potential):
                if jobtype == 'sp' or 'jobtype' not in section:
                #add the derivative coupling section:
                    if 'cis_der_couple' not in section:
                        #Add this section:
                        open_input.write('cis_deriv_couple    true\n')   
                    if 'cis_der_numstate' not in section:
                    #add this section:
                        open_input.write('cis_der_numstate    ' + str(n_states) + '\n')
                    
                    #END OF REM SECTION


            if 'derivative_coupling' in section.casefold():
                #start writing this section:
                open_input.write('\n$derivative_coupling\n')
                #The comment line:
                open_input.write(section.split('\n')[1] + '\n')

                coup_states = [str(st) for st in range(n_states)]
                open_input.write(' '.join(coup_states) + '\n')
                #END OF DERIVATIVE COUPLING SECTION

            #To close all the sections:
            if 'end' in section.casefold():
                    open_input.write('$end\n')
    open_input.close()
                    
    #Run The Calculation:
    output_file = input_file.replace('.inp', '.out')
    subprocess.run(['qchem', '-nt', '1', input_file, output_file, 'save'])    






def write_qchem_geom(geom):
    """Writes a array of atoms to a COLUMBUS style geom file."""



