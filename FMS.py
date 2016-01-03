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
    print("surface_type="+glbl.fms['surface_type'])
    master = bundle.bundle(glbl.fms['surface_type'])
    #
    # set the initial conditions for trajectories
    #
    print("calling init_trajectories")
    initial.init_trajectories(master)
    print('initial geom='+str(repr(master.traj[0].x())))
    #
    # propagate the trajectories 
    #
    while glbl.fms['current_time'] < glbl.fms['simulation_time']:
        # set the time step 
        current_time = glbl.fms['current_time']
        default_dt   = step.time_step(master)         
        final_time   = current_time + default_dt
        #
        # take an fms dynamics step, return the number at conclusion of step
        step.fms_step_bundle(master,current_time,final_time,default_dt)   
        # update the fms output files, as well as checkpoint, if necessary
        fileio.update_output(master)
        # if no more live trajectories, simulation is complete
        if(master.nalive == 0): break

    fileio.cleanup()

if __name__ == "__main__":
    main()
