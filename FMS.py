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
import src.dynamics.initialize as initialize
import src.dynamics.step as step


def main():
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
    fileio.read_input_file()

    print("all done read_input_file..")

    # initialize random number generator
    random.seed(glbl.sampling['seed'])

    print("random seed done")

    # Create the collection of trajectories
    master = bundle.Bundle(glbl.propagate['n_states'],
                           glbl.interface['integrals'])

    print("master bundle done")

    # set the initial conditions for trajectories
    initialize.init_bundle(master)

    print("initialize done")

    # propagate the trajectories
    while master.time < glbl.propagate['simulation_time']:

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
        if fileio.update_logs(master) and glbl.mpi['rank']==0:
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
        glbl.mpi['parallel'] = True

    try:
        main()
    except:
        fileio.cleanup(traceback.format_exc())
    #except Exception as exc:
    #    fileio.cleanup(exc) # only gives error message
