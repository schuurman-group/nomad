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
# if off-diagonal element of H matrix is greater than coup_thresh,
# trajectories are 'coupled' and small time step required
coup_thresh = 0.001,
# if new_traj overlap with any traj in bundle is > sij_thresh, don't spawn
sij_thresh  = 1.e-5,
# current working directory
working_dir = ''

#
# Simulation parameters read form the fms.input file
#
fms = dict( 
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
       propagator        ='velocity_verlet',
       spawn_pop_thresh  = 0.025,
       spawn_coup_thresh = 0.02,
       pot_shift         = 0.,
       output_path       = ''
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

