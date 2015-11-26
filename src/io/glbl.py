#
# CONVERSTION FACTORS AND CONSTANTS 
#
fs2au    = 41.34137221718
bohr2ang = 0.529177249
mass2au  = 1822.887
au2cm    = 219474.63
fpzero   = 1.e-10
#
# Simulation parameters read form the fms.input file
#
fms = dict( 
       current_time = 0.,
       simulation_time = 0.,
       default_time_step = 1., 
       coupled_time_step = 0.25,
       interface = 'columbus',
       init_sampling = 'gs_wigner',
       n_init_traj = 1,
       seed = 0, 
       restart = False, 
       num_particles = 1,
       dim_particles = 3, 
       n_states = 1,
       init_state = 1,
       init_brightest = False,                
       restart_time = 0.,
          )
#
# Electronic structure information read from interface-specific
# routines 
#
es  = dict(
       n_mcstates = 1, n_cistates=1
          )

#
# Vibronic multistate representation, loaded by operator parsing
# function
#

