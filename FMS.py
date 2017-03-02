#!/usr/bin/env python
"""
Main module used to initiate FMSpy.
"""
import os
import sys
import traceback
import random
import numpy as np
import mpi4py.MPI as MPI
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.bundle as bundle
import src.dynamics.timings as timings
import src.dynamics.initial as initial
import src.dynamics.step as step


def main():
    # initialize MPI communicator
    if glbl.mpi_parallel:
        glbl.mpi_comm  = MPI.COMM_WORLD
        glbl.mpi_rank  = glbl.mpi_comm.Get_rank()
        glbl.mpi_nproc = glbl.mpi_comm.Get_size()
    else:
        glbl.mpi_rank  = 0
        glbl.mpi_nproc = 1

    # start the global timer
    timings.start('global')

    # read in options/variables pertaining to the running
    # of the dynamics, pass the starting time of the simluation
    # a nd the end time
    fileio.read_input_files()

    # initialize random number generator
    random.seed(glbl.fms['seed'])

    # Create the collection of trajectories
    master = bundle.Bundle(glbl.fms['n_states'], glbl.fms['integrals'])

    # set the initial conditions for trajectories
    initial.init_bundle(master)

    # propagate the trajectories
    while master.time < glbl.fms['simulation_time']:

        # set the time step --> top level time step should always
        # be default time step. fms_step_bundle will decide if/how
        # dt should be shortened for numerics
        time_step = step.fms_time_step(master)

        # take an fms dynamics step
        master = step.fms_step_bundle(master, time_step)

        # if no more live trajectories, simulation is complete
        if master.nalive == 0:
            break

        # determine whether it is necessary to update the output logs
        if fileio.update_logs(master) and glbl.mpi_rank==0:
            # update the fms output files, as well as checkpoint, if necessary
            master.update_logs()

    # clean up, stop the global timer and write logs
    fileio.cleanup()

if __name__ == '__main__':
    #pypath     = os.environ['PYTHONPATH']
    #fmspy_path = os.environ['FMSPY_PATH']
    #os.environ['PYTHONPATH'] = pypath+':'+fmspy_path

    # parse command line arguments
    if '-mpi' in sys.argv:
        glbl.mpi_parallel = True

    try:
        main()
    except:
        fileio.cleanup(traceback.format_exc())
    #except Exception as exc:
    #    fileio.cleanup(exc) # only gives error message
