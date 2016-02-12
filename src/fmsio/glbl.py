#
# CONVERSTION FACTORS AND CONSTANTS 
#
# convert fs to au
fs2au       = 41.34137221718
# convert bohr to angstrom
bohr2ang    = 0.529177249
# convert mass in amu to au
mass2au     = 1822.887
# convert hartree to cm-1
au2cm       = 219474.63
# floating point zero
fpzero      = 1.e-10
 
#
# Simulation parameters read form the fms.input file
#
fms = dict( 
       spawning               = 'forward_backward',
       simulation_time        = 0.,
       default_time_step      = 10., 
       coupled_time_step      = 5.,
       interface              = 'boson_model_diabatic',
       init_sampling          = 'gs_wigner',
       integrals              = 'saddle_point',
       n_init_traj            = 1,
       seed                   = 0, 
       restart                = False, 
       num_particles          = 1,
       dim_particles          = 3, 
       n_states               = 1,
       init_state             = 1,
       init_brightest         = False,                
       restart_time           = 0.,
       propagator             = 'velocity_verlet',
       spawn_pop_thresh       = 0.025,
       spawn_coup_thresh      = 0.02,
       spawn_olap_thresh      = 0.7,
       energy_jump_toler      = 0.0001,
       pop_jump_toler         = 0.0001,
       pot_shift              = 0.,
       init_mode_min_olap     = 0.,
       continuous_min_overlap = 0.5,
       sij_thresh             = 1.e-5,
       hij_coup_thresh        = 0.001,
       print_traj             = 1,
       print_es               = 1,
       print_matrices         = 1,
       mirror_basis           = 0
          )

#
# Electronic structure information read from interface-specific
# routines 
#
# COLUMBUS input variables
columbus  = dict(
       # memory per core in MB   
       mem_per_core      = 100,
          )

#
# Vibronic multistate representation, loaded by operator parsing
# function
#
vibronic = dict (
       # highest polynomial order in vibronic expansion
       ordr_max          = 1,
          )

boson    = dict (
       coupling          = 0.009
          )

