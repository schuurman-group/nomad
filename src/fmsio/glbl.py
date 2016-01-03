#
# CONVERSTION FACTORS AND CONSTANTS 
#
fs2au    = 41.34137221718
bohr2ang = 0.529177249
mass2au  = 1822.887
au2cm    = 219474.63
fpzero   = 1.e-10

# current working directory
working_dir = ''

#
# Simulation parameters read form the fms.input file
#
fms = dict( 
       current_time      = 0.,
       simulation_time   = 0.,
       default_time_step = 1., 
       coupled_time_step = 0.25,
       interface         = 'columbus',
       init_sampling     = 'gs_wigner',
       surface_type      = 'adiabatic',
       n_init_traj       = 1,
       seed              = 0, 
       restart           = False, 
       num_particles     = 1,
       dim_particles     = 3, 
       n_states          = 1,
       init_state        = 1,
       init_brightest    = False,                
       restart_time      = 0.,
       propagator        ='velocity_verlet'
          )
#
# Electronic structure information read from interface-specific
# routines 
#
# COLUMBUS input variables
columbus  = dict(
       # level of mrci: 0-mcscf, 1-foci, 2-soci, etc.
       mrci_lvl          = 1,
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

