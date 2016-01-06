def main(): 
    import random
    import src.fmsio.glbl as glbl
    import src.fmsio.fileio as fileio
    import src.dynamics.initial as initial
    import src.dynamics.step as step
    import src.basis.bundle as bundle
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
    master = bundle.bundle(glbl.fms['n_states'],glbl.fms['surface_type'])
    #
    # set the initial conditions for trajectories
    #
    initial.init_bundle(master)
    #
    # propagate the trajectories 
    #
    while master.time < glbl.fms['simulation_time']:
        # set the time step 
        current_time = master.time
        time_step    = step.time_step(master)         

        # take an fms dynamics step
        step.fms_step_bundle(master,current_time,time_step)   

        # if no more live trajectories, simulation is complete
        if(master.nalive == 0):
            break

        # update the fms output files, as well as checkpoint, if necessary
        master.time += time_step
        master.update_logs() 

    fileio.cleanup()

if __name__ == "__main__":
    main()
