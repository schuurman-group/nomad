#!/usr/bin/env python
"""
Main module used to initiate and run FMSpy.
"""
import os
import sys
import random
import numpy as np
import src.archive.checkpoint as chkpt
import src.archive.printing as printing
import src.basis.bundle as bundle

bundle0 = None

#
#
#
def main():
    """Runs the main FMSpy routine."""

    chkpt_file, time, data = process_arguments(sys.argv)

    # create a bundle object to hold data
    master = bundle.Bundle()

    # set the read time: here, there are 3 options:
    # time = None, -1, t, where these return: data
    # at all times, data at last time, data  at time 
    # t. In checkpoint, time is either None (current
    # time) or t (specific time)
    read_time = None
    if time is not None and time != -1:
        read_time = time

    # read bundle for the case of a single time
    chkpt.read(master, chkpt_file, read_time)

    # we need to the t=0 bundle if auto-correlation 
    # function requested
    if data['-auto']:
        bundle0 = bundle.Bundle()
        chkpt.read(bundle0, chkpt_file, 0.)

    # get printing machine ready
    ncrd    = master.traj[0].dim
    nstates = master.traj[0].nstates
    printing.generate_data_formats(ncrd, nst)

    # if we want all times, run through all the data
    if time is None:
        tvals = chkpt.time_steps(chkpt_file)
        for i in range(len(tvals)):
            chkpt.read(master,chkpt_file,tvals[i])
            extract_data(master, data)

    # else, print data from current bundle
    else:
        extract_data(master, data)

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
                '-matrices':False, '-trajectories':False, '-interface_data':False}

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
def extract_data(bund, dlst):
    """Documentation to come"""

    if '-all' in dlst or '-energy' in dlst:
        extract_energy(bund)

    if '-all' in dlst or '-pop' in dlst:
        extract_pop(bund)

    if '-all' in dlst or '-auto' in dlst:
        extract_auto(bund)

    if '-all' in dlst or '-matrices' in dlst:
        extract_matrices(bund)
    
    if '-all' in dlst or '-trajectories' in dlst:
        extract_trajectories(bund)

    if '-all' in dlst or '-interface_data' in dlst:
        extract_interface_data(bund)

    return

#
#
#
def extract_energy(bund):
    """Documentation to come"""
    
    prnt_data = [bund.time]
    prnt_data.append(bund.pot_quantum(),   bund.kin_quantum(),   bund.tot_quantum(),
                     bund.pot_classical(), bund.kin_classical(), bund.tot_classical())

    printing.print_bund_row('energy', prnt_data)

    return

#
#
#
def extract_pop(bund):
    """Documentation to come"""

    prnt_data = [bund.time]
    prnt_data.extend(bund.pop())
    prnt_data.append(bund.norm())

    printing.print_bund_row('pop', prnt_data)

    return

#
#
#
def extract_auto(bund):
    """Documentation to come"""
    global bundle0

    b_overlap = bundle0.overlap(bund)
    prnt_data = [bund.time, b_overlap.imag, b_overlap.real, np.absolute(b_overlap)]
 
    printing.print_bund_row('auto', prnt_data)
 
    return

#
#
#
def extract_matrices(bund):
    """Documentation to come"""

    printing.print_bund_mat(bund.time, 's', bund.S)
    printing.print_bund_mat(bund.time, 'sdot', bund.Sdot)
    printing.print_bund_mat(bund.time, 'h', bund.H)
    printing.print_bund_mat(bund.time, 'heff', bund.Heff)
    printing.print_bund_mat(bund.time, 't_overlap', bund.traj_ovrlp)

    return

#
#
#
def extract_trajectories(bund):
    """Documentation to come"""

    for i in range(bund.nalive):
        traj = bund.traj[alive[i]]
        label = traj.label

        # print trajectory file        
        prnt_data = [bund.time]
        prnt_data.extend(traj.x())
        prnt_data.extend(traj.p())
        prnt_data.extend([traj.phase(), traj.amplitude.real, traj.amplitude.imag, 
                          np.absolute(traj.amplitude, traj.state)])
        printing.print_traj_row(label, 'traj', prnt_data)

        # print potential energies
        prnt_data = [bund.time]
        prnt_data.extend([traj.energy(i) for i in range(traj.nstates)])
        printing.print_traj_row(label, 'traj', prnt_data)

        # print gradients
        prnt_data = [bund.time]
        prnt_data.extend(traj.derivative(traj.state,traj.state))
        printing.print_traj_row(label, 'grad', prnt_data)

        # print coupling
        prnt_data = [bund.time]
        prnt_data.extend([traj.coupling_norm(i) for i in range(traj.nstates)])
        prnt_data.extend([traj.coup_dot_vel(i) for i in range(traj.nstates)])
        printing.print_traj_row(label,'coup', prnt_data)

        # print hessian
        printing_print_traj_mat(bund.time, 'hessian', traj.hessian(traj.state))

    return


#
#
#
def extract_interface_data(bund):
    """Documentation to come"""

   
    return


if __name__ == '__main__':

    # run the main routine
    main()
