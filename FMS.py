import os
#from pyspark import SparkContext

def main(sc):
    import sys
    import random
    import numpy as np
    import src.fmsio.glbl as glbl
    import src.fmsio.fileio as fileio
    import src.basis.bundle as bundle
    import src.dynamics.timings as timings
    import src.dynamics.initial as initial
    import src.dynamics.step as step

    #
    # start the master timer
    #
    timings.start('global')

    #
    # set the sparkContext, if running on cluster
    #
    glbl.sc = sc

    #
    # read in options/variables pertaining to the running
    # of the dynamics, pass the starting time of the simluation
    # and the end time
    #
    fileio.read_input_files()
    #
    # initialize random number generator
    #
    random.seed(glbl.fms['seed'])
    #
    # Create the collection of trajectories
    #
    master = bundle.bundle(glbl.fms['n_states'],glbl.fms['integrals'])
    #
    # set the initial conditions for trajectories
    #
    initial.init_bundle(master)
    #
    # propagate the trajectories
    #
    while master.time < glbl.fms['simulation_time']:
        # set the time step
        time_step    = step.time_step(master)

        # take an fms dynamics step
        master = step.fms_step_bundle(master,time_step)

        # if no more live trajectories, simulation is complete
        if master.nalive == 0:
            break

        # determine whether it is necessary to update the output logs
        if fileio.update_logs(master):
            # update the fms output files, as well as checkpoint, if necessary
            master.update_logs()

    fileio.cleanup()

if __name__ == "__main__":
    if False:
        pypath     = os.environ['PYTHONPATH']
        fmspy_path = os.environ['FMSPY_PATH']
        os.environ['PYTHONPATH'] = pypath+":"+fmspy_path
        sc     = SparkContext("local[4]", "FMS job queue")
    else:
        sc     = None

    main(sc)
