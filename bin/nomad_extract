#!/usr/bin/env python
"""
Main module used to initiate and run FMSpy.
"""
import os
import sys
import random
import numpy as np
import src.archive.checkpoint as checkpoint
import src.archive.printing as printing
import src.basis.wavefunction as wavefunction
import src.basis.matrices as matrices
import src.integrals.integral as integral

wfn0 = None
ints = None

#
#
#
def main():
    """Runs the main FMSpy routine."""
    global wfn0, ints

    chkpt_file, time, data = process_arguments(sys.argv)

    # set the read time: here, there are 3 options:
    # time = None, -1, t, where these return: data
    # at all times, data at last time, data  at time 
    # t. In checkpoint, time is either None (current
    # time) or t (specific time)
    read_time = None
    if time is not None and time != -1:
        read_time = time

    # create an integral object in case we need to evaluate
    # integrals (i.e. autocorrelation function)
    ints = integral.Integral(data['-integrals'])

    # read t=0 wfn to determine autocorrelation function
    wfn0 = wavefunction.Wavefunction() 
    checkpoint.read(wfn0, chkpt_file, 0.)

    # get printing machine ready
    wfn     = wavefunction.Wavefunction()
    mat     = matrices.Matrices()
    ncrd    = wfn0.traj[0].dim
    nstates = wfn0.traj[0].nstates
    printing.generate_data_formats(ncrd, nstates)

    # if we want all times, run through all the data
    if time is None:
        tvals = checkpoint.time_steps('wavefunction',chkpt_file)
        for i in range(len(tvals)):
            wfn = wavefunction.Wavefunction()
            checkpoint.retrieve_simulation(wfn, integrals=ints, 
                                          time=tvals[i], file_name=chkpt_file)
            mat.build(wfn, ints)
            wfn.update_matrices(mat)
            extract_data(wfn, mat, data)

    # else, print data from current bundle
    else:
        # read bundle for the case of a single time
        checkpoint.retrieve_simulation(wfn, integrals=ints,
                                       time=read_time, file_name=chkpt_file)
        mat.build(wfn, ints)
        wfn.update_matrices(mat)
        extract_data(wfn, mat, data)

    return

#
#
#
def process_arguments(args):
    """Process the command line arguments 
    determine what data to extract from the checkpoint file, and where
    to put it.
    """

    # a list of valid arguments and the default values
    data_lst = {'-all':False,      '-energy':False,       '-pop':False, '-auto':False,          
                '-matrices':False, '-trajectories':False, '-interface_data':False, 
                '-integrals':'bra_ket_averaged'}

    # this defaults to extracting data at all times
    time = None

    if len(args) == 0:
        exit_error('input file not specified')

    chkpt_file = args[1]

    if not os.path.isfile(chkpt_file):
        exit_error('input file: '+str(ckhpt_file)+' not found')

    iarg = 2
    while iarg < len(args):

        arg_val = args[iarg].strip()

        # just skip unrecognized commands
        if arg_val not in data_lst:
            print('argument: '+str(arg_val)+' not recognized. Skipping...')
            continue

        elif arg_val == '-time':
            time = float(args[iarg+1])
            iarg += 1

        else:
            data_lst[arg_val] = True

        iarg += 1

    return chkpt_file, time, data_lst

#
#
#
def extract_data(wfn, mat, dlst):
    """Documentation to come"""

    if '-all' in dlst or '-energy' in dlst:
        extract_energy(wfn)

    if '-all' in dlst or '-pop' in dlst:
        extract_pop(wfn)

    if '-all' in dlst or '-auto' in dlst:
        extract_auto(wfn)

    if '-all' in dlst or '-matrices' in dlst:
        extract_matrices(wfn, mat)
    
    if '-all' in dlst or '-trajectories' in dlst:
        extract_trajectories(wfn)

    if '-all' in dlst or '-interface_data' in dlst:
        extract_interface_data(wfn)

    return

#
#
#
def extract_energy(wfn):
    """Documentation to come"""
   
    prnt_data = [wfn.time]

    prnt_data.extend([wfn.pot_quantum(),   wfn.kin_quantum(),   wfn.tot_quantum(),
                      wfn.pot_classical(), wfn.kin_classical(), wfn.tot_classical()])

    printing.print_wfn_row('energy', prnt_data)

    return

#
#
#
def extract_pop(wfn):
    """Documentation to come"""

    prnt_data = [wfn.time]
    prnt_data.extend(wfn.pop())
    prnt_data.extend([wfn.norm()])

    printing.print_wfn_row('pop', prnt_data)

    return

#
#
#
def extract_auto(wfn):
    """Documentation to come"""
    global wfn0, ints

    wfn_overlap = ints.wfn_overlap(wfn0, wfn)
    prnt_data = [wfn.time, wfn_overlap.real, wfn_overlap.imag, np.absolute(wfn_overlap)]
 
    printing.print_wfn_row('auto', prnt_data)
 
    return

#
#
#
def extract_matrices(wfn, mat):
    """Documentation to come"""

    printing.print_wfn_mat(wfn.time, 's', mat.mat_dict['s'])
    printing.print_wfn_mat(wfn.time, 'sdot', mat.mat_dict['sdot'])
    printing.print_wfn_mat(wfn.time, 'h', mat.mat_dict['h'])
    printing.print_wfn_mat(wfn.time, 'heff', mat.mat_dict['heff'])
    printing.print_wfn_mat(wfn.time, 't_overlap', mat.mat_dict['s_traj'])

    return

#
#
#
def extract_trajectories(wfn):
    """Documentation to come"""

    for i in range(wfn.nalive):
        traj  = wfn.traj[wfn.alive[i]]
        label = traj.label
        state = traj.state 

        # print trajectory file        
        prnt_data = [wfn.time]
        prnt_data.extend(traj.x())
        prnt_data.extend(traj.p())
        prnt_data.extend([traj.phase(), traj.amplitude.real, traj.amplitude.imag, 
                          np.absolute(traj.amplitude), traj.state])
        printing.print_traj_row(label, 'traj', prnt_data)

        # print potential energies
        prnt_data = [wfn.time]
        prnt_data.extend([traj.energy(i) for i in range(traj.nstates)])
        printing.print_traj_row(label, 'poten', prnt_data)

        # print gradients
        prnt_data = [wfn.time]
        prnt_data.extend(traj.derivative(traj.state,traj.state))
        printing.print_traj_row(label, 'grad', prnt_data)

        # print coupling
        prnt_data = [wfn.time]
        prnt_data.extend([np.linalg.norm(traj.derivative(state,i)) for i in range(traj.nstates)])
        prnt_data.extend([traj.coupling(state,i) for i in range(traj.nstates)])
        printing.print_traj_row(label,'coup', prnt_data)

        # print hessian
        printing.print_traj_mat(wfn.time, 'hessian', traj.hessian(traj.state))

    return


#
#
#
def extract_interface_data(wfn):
    """Documentation to come"""

   
    return


if __name__ == '__main__':

    # run the main routine
    main()
