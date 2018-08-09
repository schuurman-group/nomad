"""
Global variables used throughout nomad
"""

interface   = None
init_conds  = None
adapt       = None
integrator  = None
surface_rep = 'adiabatic'

bundle0     = None
master_mat  = None
master_int  = None

home_path   = ''
log_file    = ''
chkpt_file  = ''

# MPI variables (set by nomad_driver)
mpi = dict(
    parallel = False,
    comm     = None,
    rank     = 0,
    nproc    = 1
           )

# Method variables (no default values)
methods = dict(
    adapt_basis   = None,
    ansatz        = None,
    init_conds    = None,
    integral_eval = None,
    interface     = None,
    propagator    = None
               )

# Interface-specific variables
columbus = dict(
    hessian        = None, # required if init_conds == wigner, can be filename or direct input
    mem_per_core   = 1000.,
    coup_de_thresh = 100.
                )

vibronic = dict(
    opfile         = None, # required filename, should contain frequencies
    mem_per_core   = 1000.,
    coupling_order = 1,
    ordr_max       = 1
                )

interfaces = dict(
    columbus = columbus,
    vibronic = vibronic
                  )

# Remaining properties (mostly optional)
properties = dict(
    freqs = [0.003351536, 0.004001999, 0.006515652, 0.00947397, 0.013645016], # this is a hack to test the rest of input
    init_coords         = None, # required, can be filename or XYZ format
    atm_labels          = None,
    atm_widths          = None,
    atm_masses          = None,
    n_states            = 2,
    n_init_traj         = 1,
    init_state          = -1,
    init_brightest      = False,
    init_mode_min_olap  = 0.,
    virtual_basis       = False,
    distrib_compression = 1.,
    seed                = 0,
    restart             = False,
    restart_time        = -1,
    simulation_time     = 1000.,
    default_time_step   = 10.,
    coupled_time_step   = 5.,
    integral_order      = 1,
    energy_jump_toler   = 1e-4,
    pop_jump_toler      = 1e-4,
    pot_shift           = 0.,
    renorm              = False,
    sinv_thrsh          = -1.,
    norm_thresh         = 1.1,
    auto                = False,
    phase_prop          = False,
    sij_thresh          = 0.7,
    hij_coup_thresh     = 1e-3,
    matching_pursuit    = False,
    spawn_pop_thresh    = 0.025,
    spawn_coup_thresh   = 0.02,
    spawn_olap_thresh   = 0.7,
    continuous_min_overlap = 0.5,
    init_amps           = [1+0j],
    init_amp_overlap    = True,
    print_level         = 1
                  )

# lists keywords, the datatype of the keyword and the dimension
# 0=scalar, 1=list, 2=nested list.
# note that multi-dimensional keywords are python lists
keyword_type = dict(
    bundle0                = [None,0],
    surf_rep               = [str,0],
    parallel               = [bool,0],
    comm                   = [None,0],
    rank                   = [int,0],
    nproc                  = [int,0],
    restart                = [bool,0],
    init_sampling          = [str,0],
    n_init_traj            = [int,0],
    init_state             = [int,0],
    init_states            = [int,1],
    init_brightest         = [bool,0],
    restart_time           = [float,0],
    init_mode_min_olap     = [float,0],
    seed                   = [int,0],
    virtual_basis          = [bool,0],
    distrib_compression    = [float,0],
    n_states               = [int,0],
    matching_pursuit       = [bool,0],
    simulation_time        = [float,0],
    default_time_step      = [float,0],
    coupled_time_step      = [float,0],
    integrals              = [str,0],
    integral_order         = [int,0],
    propagator             = [str,0],
    energy_jump_toler      = [float,0],
    pop_jump_toler         = [float,0],
    pot_shift              = [float,0],
    renorm                 = [bool,0],
    sinv_thrsh             = [float,0],
    norm_thresh            = [float,0],
    auto                   = [bool,0],
    phase_prop             = [bool,0],
    sij_thresh             = [float,0],
    hij_coup_thresh        = [float,0],
    spawning               = [str,0],
    spawn_pop_thresh       = [float,0],
    spawn_coup_thresh      = [float,0],
    spawn_olap_thresh      = [float,0],
    continuous_min_overlap = [float,0],
    interface              = [str,0],
    coupling_order         = [int,0],
    mem_per_core           = [float,0],
    coup_de_thresh         = [float,0],
    opfile                 = [str,0],
    ordr_max               = [int,0],
    use_atom_lib           = [bool,0],
    init_amp_overlap       = [bool,0],
    geometries             = [float,2],
    momenta                = [float,2],
    hessian                = [float,2],
    freqs                  = [float,1],
    labels                 = [str,1],
    amplitudes             = [complex,1],
    widths                 = [float,1],
    masses                 = [float,1],
    print_level            = [int,0],
                    )
