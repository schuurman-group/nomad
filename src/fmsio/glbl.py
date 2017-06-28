"""
Conversion factors and constants for FMSpy.
"""
# convert fs to au
fs2au    = 41.34137221718
# convert bohr to angstrom
bohr2ang = 0.529177249
# convert mass in amu to au
mass2au  = 1822.887
# convert hartree to eV
au2ev    = 27.21138505
# convert hartree to cm-1
au2cm    = 219474.63
# floating point zero
fpzero   = 1.e-10

# t=0 bundle
bundle0  = None

# MPI variables
mpi_parallel = False
mpi_comm     = None
mpi_rank     = None
mpi_nproc    = 1

# input related to initial conditions
sampling = dict(
    restart                = False,
    init_sampling          = 'gs_wigner',
    n_init_traj            = 1,
    init_state             = [1],
    init_brightest         = False,
    restart_time           = 0.,
    init_mode_min_olap     = 0.,
    seed                   = 0,
    virtual_basis          = 0,
    sampling_compression   = 1.0,
)

propagate = dict(
    n_states               = 1,
    matching_pursuit       = 0,
    simulation_time        = 0.,
    default_time_step      = 10.,
    coupled_time_step      = 5.,
    integrals              = 'saddle_point',
    propagator             = 'velocity_verlet',
    energy_jump_toler      = 0.0001,
    pop_jump_toler         = 0.0001,
    pot_shift              = 0.,
    renorm                 = 0,
    sinv_thrsh             = -1.0,
    norm_thresh            = 10.,
    auto                   = 0,
    phase_prop             = 1,
    sij_thresh             = 1.e-5,
    hij_coup_thresh        = 0.001,
)

spawning = dict(
    spawning               = 'forward_backward',
    spawn_pop_thresh       = 0.025,
    spawn_coup_thresh      = 0.02,
    spawn_olap_thresh      = 0.7,
    continuous_min_overlap = 0.5,
)

interface = dict(
    # pertain to all interfaces
    interface              = 'boson_model_diabatic',
    coupling_order         = 1,

    # parameters that apply to the COLUMBUS interface
    mem_per_core           = 100,
    coup_de_thresh         = 100.,

    # parameters that apply to vibronic interface
    opfile                 = 'fms.op',
    # highest polynomial order in vibronic expansion
    ordr_max               = 1
)

nuclear_basis = dict(
    use_atom_lib     = True,
    init_amp_overlap = True,
    geometry = [0],
    momenta  = [0],
    geomfile = '',
    hessian  = [0],
    hessfile = '',
    amplitudes = [0],
    widths   = [0],
    states   = [0],
    masses   = [0],
)

printing = dict(
    print_level            = 1,
    print_traj             = 1,
    print_es               = 1,
    print_matrices         = 1,
    print_chkpt            = 1,
)

# this is a list of valid dictionary names. groups of input need to be added to 
# this last (obvs)
input_groups   = {'sampling':sampling,
                  'propagate':propagate,
                  'spawning':spawning,
                  'nuclear_basis':nuclear_basis,
                  'interface':interface,
                  'printing':printing}

