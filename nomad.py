#!/usr/bin/env python
"""
Main module used to initiate and run FMSpy.
"""
import os
import sys
import random
import numpy as np
import mpi4py.MPI as MPI
import src.parse.glbl as glbl
import src.parse.files as files
import src.parse.log as log
import src.archive.checkpoint as checkpoint
import src.basis.wavefunction as wavefunction
import src.basis.initialize as initialize
import src.dynamics.step as step
import src.utils.timings as timings
import src.utils.cleanup as cleanup

def init():
    """Initializes the FMSpy inputs.

    This must be separate from main so that an error which occurs
    before the input file is created will be written to stdout.
    """
    # initialize MPI communicator
    if glbl.mpi['parallel']:
        glbl.mpi['comm']  = MPI.COMM_WORLD
        glbl.mpi['rank']  = glbl.mpi['comm'].Get_rank()
        glbl.mpi['nproc'] = glbl.mpi['comm'].Get_size()
    else:
        glbl.mpi['rank']  = 0
        glbl.mpi['nproc'] = 1

    # start the global timer
    timings.start('global')

    # read in options/variables pertaining to the running
    # of the dynamics, pass the starting time of the simluation
    # a nd the end time
    files.read_input()

    # initialize random number generator
    random.seed(glbl.sampling['seed'])

    # initialize the running calculation log
    log.init_logfile('nomad.log')

def main():
    """Runs the main FMSpy routine."""
    # Create the collection of trajectories
    master = wavefunction.Wavefunction()

    # set the initial conditions for trajectories
    initialize.init_wavefunction(master)

    while master.time < glbl.propagate['simulation_time']:
        # set the time step --> top level time step should always
        # be default time step. fms_step_wfn will decide if/how
        # dt should be shortened for numerics
        time_step = step.time_step(master)

        # take an fms dynamics step
        master = step.step_wavefunction(master, time_step)

        # if no more live trajectories, simulation is complete
        if master.nalive == 0:
            break

        # determine whether it is necessary to update the output logs
        if glbl.mpi['rank'] == 0:
            # update the fms output files, as well as checkpoint, if necessary
            checkpoint.write(master, time=master.time)
            checkpoint.write(glbl.master_mat, time=master.time)
            checkpoint.write(glbl.master_int, time=master.time)

    # clean up, stop the global timer and write logs
    cleanup.cleanup_end()


if __name__ == '__main__':
    # parse command line arguments
    if '-mpi' in sys.argv:
        glbl.mpi['parallel'] = True

    # initialize
    init()
    # if an error occurs, cleanup and report the error
    sys.excepthook = cleanup.cleanup_exc
    # run the main routine
    main()
