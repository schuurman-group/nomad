"""
The default dictionary values for different routines.
When a console routine is called, the default dictionary is called from this
file. Alternatively, the command getdefault [routine_name] will create
an input file with the default set of inputs. Similarly,
geninput [routine_name] will query for input for each dictionary variable.
"""
import os


inpname = dict(
    contourplot = 'contour.inp',
    denplot = 'den.inp',
    getgeoms = 'geoms.inp',
    histplot = 'hist.inp',
    scatterplot = 'scatter.inp',
    trpesplot = 'trpes.inp',
    pesplot = 'pes.inp',
    popassign = 'assign.inp',
    popplot = 'pop.inp'
               )

contourplot = dict(
    nstates = 1,
    istate = 0,
    infname = 'energies.out',
    eshift = 0.,
    econv = 27.21138505,
    xpts = 30,
    ypts = 30,
    xshift = 0.,
    yshift = 0.,
    xfac = 1.,
    yfac = 1.,
    xlabel = 'x',
    ylabel = 'y',
    xmeci = None,
    ymeci = None,
    plot_name = 'contour.pdf'
                   )

denplot = dict(
    file_stub = '../density/fmsDen_t',
    tinc = 50,
    tmin = 0.,
    tmax = 12000.,
    tunits = 'fs',
    ncoord = 361,
    coordmin = 0.,
    coordmax = 3.14159265359,
    coordunits = 'deg',
    coordlbl = r'Angle / degrees',
    data_name = 'den.dat',
    plot_name = 'den.pdf'
               )

getgeoms = dict(
    seed_stub = '../seed.',
    traj_stub = 'TrajDump.',
    spawn_log = 'Spawn.log',
    tinc = 100,
    tmax = 48000,
    elem = None,
    xyz_stub = 'geoms.'
                )

histplot = dict(
                )

scatterplot = dict(
    states = [1, 0],
    traj_files = '../seed.*/TrajDump.[1-9]*',
    xcoord = 'stre',
    ycoord = 'stre',
    xinds = [0, 1],
    yinds = [0, 1],
    xunits = 'auto',
    yunits = 'auto',
    xlabel = 'x',
    ylabel = 'y',
    data_name = 'scatter.dat',
    plot_name = 'scatter.pdf',
    pop_weight = False
                   )

trpesplot = dict(
    seed_files = 'output/seed.*',
    tinc = 100,
    nebins = 150,
    eprobe = 6.199,
    eunits = 'ev',
    tunits = 'fs',
    emin = 0,
    emax = 5,
    tmin = -200,
    tmax = 800,
    esig = 0.100,
    tsig = 42.4,
    dyson_norms = True,
    calc_err = False,
    n_bootstrap = 10000,
    boot_thrsh = 1e-3,
    data_name = 'trpes.dat',
    err_name = 'trpes.err',
    plot_name = 'trpes.pdf',
    err_plot_name = 'trpes_err.pdf'
                 )

pesplot = dict(
    states = None,
    infname = 'nrg.out',
    plotorder = None,
    gwid = 1,
    gsep = 1,
    lblrot = 90,
    maxener = None,
    minener = None,
    econv = 27.21138505,
    data_name = 'pes.dat',
    plot_name = 'pes.pdf',
    show_grid = True
               )

popassign = dict(
    traj_files = '../seed.*/TrajDump.*',
    spawn_fname = 'spawn.xyz',
    states = None,
    ref_geoms = None,
    permute = None,
    symm_equiv = None,
    mass_wgt = False,
    dir_name = 'assign',
    pop_fname = 'branch.dat'
                 )

popplot = dict(
    states = [0, 1, 2],
    fms_time_increment = 100,
    time_conv = 0.02418884326505,
    tmin = 0.0,
    tmax = 1000.0,
    ndat_files = '../seed.*/N.dat',
    calc_err = False,
    n_bootstrap = 100000,
    boot_thrsh = 1e-3,
    amplitude_data_name = 'pop.dat',
    amplitude_err_name = 'pop.err',
    plot_name = 'pop.pdf',
    fit_function = None,
    p0 = [10, 50, 0.1],
    err_thrsh = 1e-5,
    fit_data_name = 'pop.fit',
    plot_fit = True
               )


def convert2str(val):
    """Converts a value to a basic string."""
    if isinstance(val, list):
        if isinstance(val[0], list):
            # 2D list, delimit with ',' and ';'
            return '; '.join([', '.join([str(i) for i in j]) for j in val])
        else:
            # 1D list, delimit with ','
            return ', '.join([str(i) for i in val])
    else:
        # str() will handle ints, floats, bools and NoneType
        return str(val)


def write_default(routine):
    """Writes a default input file based on the routine name."""
    routine_dict = globals()[routine]
    fname = inpname[routine]

    # get existing input if possible
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            orig_inp = f.readlines()
    else:
        orig_inp = []

    with open(fname, 'w') as f:
        # keep old input but comment it out
        for line in orig_inp:
            if '# default' not in line:
                f.write('#' + line)

        for key in routine_dict:
            sval = convert2str(routine_dict[key])
            f.write('{:s} = {:s} # default\n'.format(key, sval))


def generate_input(routine):
    """Generates an input file from the command line based on user input."""
    routine_dict = globals()[routine]
    fname = inpname[routine]
    #printdocs = input('Print documentation? [n] ').strip().lower()
    #if printdocs in ['y', 'yes']:

    with open(fname, 'w') as f:
        print('# Input generated with the geninput routine')
        for key in routine_dict:
            sval = convert2str(routine_dict[key])
            inp = input(key+' ['+sval+'] = ').strip()
            if inp != '' and inp != sval:
                f.write('{:s} = {:s}\n'.format(key, inp))
