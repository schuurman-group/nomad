def main(): 
    import random
    import variable 
    import fileio
    import interface
    import initial    
    import dynamics
    import basis.bundle
    #
    # read in options/variables pertaining to the running 
    # of the dynamics, pass the starting time of the simluation
    # and the end time
    # 
    fileio.read_fms_input()
    #
    # initialize random number generator
    #
    random.seed(variable.fms['seed'])
    #
    # initialize data/variables related to the representation of
    # the potential. This will depend on what interface we're
    # using.
    #
    interface.load_interface()

    #
    # Create the collection of trajectories
    #
    master = basis.bundle()
    #
    # set the initial conditions for trajectories
    #
    if(variable.fms['restart']):
        initial.init_restart(master)
    else:
        initial.init_conditions(master)
    #
    # propagate the trajectories 
    #
    
    while variable.fms['current_time'] < variable.fms['simulation_time']:
        print("current time="+str(variable.fms['current_time']))
        # set the time step 
        dt = dynamics.time_step(master)         
        #
        final_time = variable.fms['current_time'] + dt
        # take an fms dynamics step, return the number at conclusion of step
        dynamics.fms_step(master,variable.fms['current_time'],final_time,dt)   
        # update the fms output files, as well as checkpoint, if necessary
        fileio.update_output(master)
        # if no more live trajectories, simulation is complete
        if(master.nalive == 0): break

    fileio.cleanup()

if __name__ == "__main__":
    main()
