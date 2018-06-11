import os
import re
import shutil
import traceback
import src.utils.timings as timings
import src.parse.glbl as glbl
import src.parse.log as log
import src.archive.printing as printing

#----------------------------------------------------------------------------
#
# FMS summary output file
#
#----------------------------------------------------------------------------
def cleanup_end():
    """Cleans up the FMS log file if calculation completed."""
    # simulation ended
    log.print_message('complete', [])

    # print timing information
    timings.stop('global', cumulative=True)
    t_table = timings.print_timings()
    log.print_message('timings', [t_table])

    # copy output files
    printing.copy_output()


def cleanup_exc(etyp, val, tb):
    """Cleans up the FMS log file if an exception occurs."""
    # print exception
    exception = ''.join(traceback.format_exception(etyp, val, tb))
    log.print_message('error', [rm_timer(exception)])

    # stop remaining timers
    for timer in timings.active_stack[:0:-1]:
        timings.stop(timer.name)

    # print timing information
    timings.stop('global', cumulative=True)
    t_table = timings.print_timings()
    log.print_message('timings', [t_table])

    # copy output files
    printing.copy_output()

    # abort other processes if running in parallel
    if glbl.mpi['parallel']:
        glbl.mpi['comm'].Abort(1)

#
#
#
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
