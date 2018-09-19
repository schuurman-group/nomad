"""
Global variables used throughout nomad
"""

modules = dict(
    interface  = None,
    init_conds = None,
    adapt      = None,
    integrator = None,
    matrices   = None,
    integrals  = None,
    wfn0       = None
            )

paths = dict(
    log_file   = '',
    chkpt_file = ''
#    home_path
            )

# MPI variables (set by nomad_driver)
mpi = dict(
    parallel = False,
    comm     = None,
    rank     = 0,
    nproc    = 1
           )

# Method variables (no default values)
methods = dict(
    ansatz        = None,
    adapt_basis   = None,
    init_conds    = None,
    integral_eval = None,
    interface     = None,
    propagator    = None,
    surface       = None
               )

# Interface-specific variables
columbus = dict(
    hessian        = None, # required if init_conds == wigner, can be filename or direct input
    mem_per_core   = 1000.,
    coup_de_thresh = 100.
                )

vibronic = dict(
    opfile         = '', # required filename, should contain frequencies
    mem_per_core   = 1000.,
    coupling_order = 1,
    ordr_max       = 1
                )

# Remaining properties (mostly optional)
properties = dict(
    init_coords         = None, # required, can be filename or XYZ format
    crd_widths          = None,
    crd_masses          = None,
    crd_labels          = None,
    n_states            = 2,
    n_init_traj         = 1,
    init_state          = -1,
    init_brightest      = False,
    init_mode_min_olap  = 0.,
    virtual_basis       = False,
    use_atom_lib        = True,
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

sections =  {
    'modules' : modules,
    'paths'   : paths,
    'mpi'     : mpi,
    'methods' : methods,
    'columbus' : columbus,
    'vibronic' : vibronic,
    'interfaces' : columbus,
    'vibronic'   : vibronic,
    'properties' : properties
             }

