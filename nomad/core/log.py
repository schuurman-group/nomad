"""
Routines for reading input files and writing log files.
"""
import os
import re
import traceback
import nomad.core.timings as timings
import nomad.core.glbl as glbl


log_format  = dict()
print_level = dict()


def init_logfile():
    """Documentation to come"""
    # generate the allowed log file formats
    generate_formats()

    # print the log file header, including values of all variables
    if glbl.mpi['rank'] == 0:
        print_header()


def print_message(otype, data):
    """Prints a string to the log file."""
    global log_format, print_level

    if glbl.mpi['rank'] == 0:
        if otype not in log_format:
            print('CANNOT WRITE otype=' + str(otype) + '\n')

        elif glbl.properties['print_level'] >= print_level[otype]:
            with open(glbl.paths['log_file'], 'a') as logfile:
                logfile.write(log_format[otype].format(*data))


def print_spawn_log(data):
    """Print the spawn log.

    This is apparently a 'temporary hack'.
    """
    file_name    = 'spawn.log'
    fwid1        = 12
    fwid2        = 16
    lenst        = 7
    spawn_format = ('{:12.4f}{:12.4f}{:12.4f}{:7d}{:7d}{:7d}{:7d}' +
                    '{:12.8f}{:12.8f}{:12.8f}{:12.8f}' +
                    '{:16.8f}{:16.8f}\n')
    spawn_header = ('time(entry)'.rjust(fwid1), 'time(spawn)'.rjust(fwid1),
                    'time(exit)'.rjust(fwid1), 'parent'.rjust(lenst),
                    'state'.rjust(lenst), 'child'.rjust(lenst), 'state'.rjust(lenst),
                    'ke(parent)'.rjust(fwid1), 'ke(child)'.rjust(fwid1),
                    'pot(parent)'.rjust(fwid1), 'pot(child)'.rjust(fwid1),
                    'total(parent)'.rjust(fwid2), 'total(child)'.rjust(fwid2))

    if glbl.mpi['rank'] == 0:

        if os.path.isfile(file_name):
            with open(file_name, 'a') as outfile:
                outfile.write(spawn_format.format(*data))

        else:
            with open(file_name, 'x') as outfile:
                outfile.write(''.join(spawn_header)+'\n')
                outfile.write(spawn_format.format(*data))


def cleanup_end():
    """Cleans up the nomad log file if calculation completed."""
    # simulation ended
    print_message('complete', [])

    # print timing information
    timings.stop('global', cumulative=True)
    t_table = timings.print_timings()
    print_message('timings', [t_table])


def cleanup_exc(etyp, val, tb):
    """Cleans up the nomad log file if an exception occurs."""
    # print exception
    exception = ''.join(traceback.format_exception(etyp, val, tb))
    print_message('error', [rm_timer(exception)])

    # stop remaining timers
    for timer in timings.active_stack[:0:-1]:
        timings.stop(timer.name)

    # print timing information
    timings.stop('global', cumulative=True)
    t_table = timings.print_timings()
    print_message('timings', [t_table])

    # abort other processes if running in parallel
    if glbl.mpi['parallel']:
        glbl.mpi['comm'].Abort(1)


#-----------------------------------------------------------------------------------
#
# Private Functions
#
#-----------------------------------------------------------------------------------
def print_header():
    """Documentation to come"""
    # ------------------------- log file formats --------------------------
    with open(glbl.paths['log_file'], 'w') as logfile:
        log_str = (' ---------------------------------------------------\n' +
                   ' NOMAD: Nonadiabatic Multistate Adaptive Dynamics    \n' +
                   ' ---------------------------------------------------\n' +
                   '\n' +
                   ' *************\n' +
                   ' input summary\n' +
                   ' *************\n' +
                   '\n' +
                   ' file paths\n' +
                   ' ---------------------------------------\n' +
                   ' cwd         = ' + os.uname()[1] + ':' + str(glbl.paths['cwd']) + '\n' +
                   ' log_file    = ' + os.uname()[1] + ':' + glbl.paths['log_file'] + '\n' +
                   ' chkpt_file  = ' + os.uname()[1] + ':' + glbl.paths['chkpt_file'] + '\n')
        logfile.write(log_str)

        logfile.write('\n nomad simulation keywords\n' +
                   ' ----------------------------------------\n')

        logfile.write('\n ** method variables **\n')
        log_str = ''
        for k,v in glbl.methods.items():
            log_str += ' {:20s} = {:20s}\n'.format(str(k), str(v))
        logfile.write(log_str+'\n')

        logfile.write('\n ** mpi variables **\n')
        log_str = ''
        for k,v in glbl.methods.items():
            log_str += ' {:20s} = {:20s}\n'.format(str(k), str(v))
        logfile.write(log_str+'\n')

        logfile.write('\n ** property variables **\n')
        log_str = ''
        for k,v in glbl.properties.items():
            log_str += ' {:20s} = {:20s}\n'.format(str(k), str(v))
        logfile.write(log_str+'\n')

        logfile.write('\n ** '+glbl.methods['interface']+' variables **\n')
        log_str = ''
        for k,v in glbl.sections[glbl.methods['interface']].items():
            log_str += ' {:20s} = {:20s}\n'.format(str(k), str(v))
        logfile.write(log_str+'\n')
        return

def generate_formats():
    """Documentation to come"""
    global log_format, print_level

    log_format['general']        = '   ** {:60s} **\n'
    log_format['warning']     = ' ** WARNING\: {:100s} **\n'

    log_format['string']         = ' {:160s}\n'
    log_format['t_step']         = ' > time: {:14.4f} step:{:8.4f} [{:4d} trajectories]\n'
    log_format['coupled']        = '  -- in coupling regime -> timestep reduced to {:8.4f}\n'
    log_format['new_step']       = '   -- {:50s} / re-trying with new time step: {:8.4f}\n'
    log_format['spawn_start']    = ('  -- spawning: trajectory {:4d}, ' +
                                    'state {:2d} --> state {:2d}\n' +
                                    'time'.rjust(14) + 'coup'.rjust(10) +
                                    'overlap'.rjust(10) + '   spawn\n')
    log_format['spawn_step']     = '{:14.4f}{:10.4f}{:10.4f}   {:40s}\n'
    log_format['spawn_back']     = '      back propagating:  {:12.2f}\n'
    log_format['spawn_bad_step'] = '       --> could not spawn: {:40s}\n'
    log_format['spawn_success']  = ' - spawn successful, new trajectory created at {:14.4f}\n'
    log_format['spawn_failure']  = ' - spawn failed, cannot create new trajectory\n'
    log_format['complete']       = ' ------- simulation completed --------\n'
    log_format['error']          = '\n{}\n ------- simulation terminated  --------\n'
    log_format['timings' ]       = '{}'

    print_level['general']        = 2
    print_level['warning']        = 0
    print_level['string']         = 2 
    print_level['t_step']         = 0
    print_level['coupled']        = 2
    print_level['new_step']       = 2
    print_level['spawn_start']    = 1
    print_level['spawn_step']     = 1
    print_level['spawn_back']     = 2
    print_level['spawn_bad_step'] = 2
    print_level['spawn_success']  = 1
    print_level['spawn_failure']  = 1
    print_level['complete']       = 0
    print_level['error']          = 0
    print_level['timings']        = 0


def rm_timer(exc):
    """Removes the timer lines from an Exception traceback."""
    tb = exc.split('\n')
    regex = re.compile('.*timings\.py.*in (hooked|_run_func)')
    i = 0
    while i < len(tb):
        if re.match(regex, tb[i]):
            tb = tb[:i] + tb[i+2:]
        else:
            i += 1
    return '\n'.join(tb)
