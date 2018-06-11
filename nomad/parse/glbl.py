""":
Conversion factors and constants for FMSpy.
"""

interface = None
distrib = None
spawn = None
integrator = None

master_mat = None
master_int = None

scr_path = ''
home_path = ''

# contains global variables that are not user-specified
variables = dict(
  # t=0 bundle
  bundle0  = None,
  # surface representation: either adiabatic or diabatic
  surface_rep = 'adiabatic'
)

# MPI variables
mpi = dict(
    parallel               = False,
    comm                   = None,
    rank                   = 0,
    nproc                  = 1
           )

# input related to initial conditions
sampling = dict(
    restart                = False,
    init_sampling          = 'wigner',
    n_init_traj            = 1,
    init_state             = -1,
    init_states            = [0],
    init_brightest         = False,
    restart_time           = None,
    init_mode_min_olap     = 0.,
    seed                   = 0,
    virtual_basis          = False,
    distrib_compression    = 1.0
                )

propagate = dict(
    n_states               = 1,
    simulation_time        = 0.,
    default_time_step      = 10.,
    coupled_time_step      = 5.,
    integrals              = 'saddle_point',
    integral_order          = 1,
    propagator             = 'velocity_verlet',
    energy_jump_toler      = 0.0001,
    pop_jump_toler         = 0.0001,
    pot_shift              = 0.,
    renorm                 = False,
    sinv_thrsh             = -1.0,
    norm_thresh            = 1.,
    auto                   = False,
    phase_prop             = True,
    sij_thresh             = 0.7,
    hij_coup_thresh        = 0.001,
    matching_pursuit       = False
                 )

# Electronic structure information read from interface-specific
# routines

spawning = dict(
    spawning               = 'optimal',
    spawn_pop_thresh       = 0.025,
    spawn_coup_thresh      = 0.02,
    spawn_olap_thresh      = 0.7,
    continuous_min_overlap = 0.5
                )

iface_params = dict(
    # pertain to all interfaces
    interface              = 'vibronic',
    coupling_order         = 1,

    # parameters that apply to the COLUMBUS interface
    mem_per_core           = 100,
    coup_de_thresh         = 100.,

    # parameters that apply to vibronic interface
    opfile                 = 'vibronic.op',
    # highest polynomial order in vibronic expansion
    ordr_max               = 1
                 )

nuclear_basis = dict(
    use_atom_lib           = True,
    init_amp_overlap       = True,
    geometries             = [[0]],
    momenta                = [[0]],
    geomfile               = "",
    hessian                = [[0]],
    hessfile               = "",
    freqs                  = [0],
    labels                 = [""],
    amplitudes             = [1.+0.j],
    widths                 = [0],
    masses                 = [0]
                     )

printing = dict(
    print_level            = 1,
    print_traj             = True,
    print_es               = True,
    print_matrices         = True,
    print_chkpt            = True
                )

# this is a list of valid dictionary names. groups of input need to be added to
# this last (obvs)
input_groups = dict(
    mpi                    = mpi,
    sampling               = sampling,
    propagate              = propagate,
    spawning               = spawning,
    nuclear_basis          = nuclear_basis,
    iface_params           = iface_params,
    printing               = printing
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
    geomfile               = [str,0],
    hessian                = [float,2],
    hessfile               = [str,0],
    freqs                  = [float,1],
    labels                 = [str,1],
    amplitudes             = [complex,1],
    widths                 = [float,1],
    masses                 = [float,1],
    print_level            = [int,0],
    print_traj             = [bool,0],
    print_es               = [bool,0],
    print_matrices         = [bool,0],
    print_chkpt            = [bool,0]
                    )
