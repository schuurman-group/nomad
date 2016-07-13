"""
Timers and timing information for subroutines.

Allows for nested timers. Constructed to ensure no overlap between timers,
so thereotically sum(timers) = total time
"""
import sys
import time


timer_list = dict()
active_stack = []


class Timer:
    """Timer for a process 'name'."""
    def __init__(self, name):
        self.name        = name
        self.calls       = 0
        self.cpu_time    = 0.
        self.wall_time   = 0.
        self.cpu_start   = 0.
        self.wall_start  = 0.
        self.cpu_kids    = 0.
        self.wall_kids   = 0.
        self.running     = False


def start(name):
    """Starts the timer 'name'.

    If 'name' doesn't exist, it is created and added to the list of timers.
    """
    global timer_list, active_stack

    # if timer already in list, activate it
    if name not in timer_list:
        timer_list[name] = Timer(name)

    # add the timer to the active stack
    active_stack.append(timer_list[name])

    # if already running, let me know...
    #print("\nstart items on stack -- ")
    #for i in range(len(active_stack)):
    #    print(str(i)+': '+str(active_stack[i].name))

    if active_stack[-1].running:
        sys.exit('START timer: ' + str(name) + ' called while running.\n'
                 ' => could be recursion issue')

    active_stack[-1].calls     += 1
    active_stack[-1].running    = True
    active_stack[-1].wall_start = time.time()
    active_stack[-1].cpu_start  = time.clock()
    active_stack[-1].wall_kids  = 0.
    active_stack[-1].cpu_kids   = 0.


def stop(name, cumulative=False):
    """Stops the timer 'name'.

    Trying to stop an unknown timer throws an error.
    """
    global active_stack

    # for time being, assume, last function added to stack, is the first to
    # be stopped. If more flexibility is required, we'll address it at that time
    if active_stack[-1].name != name:
        raise NameError('STOP timer: ' + str(name) +
                        ' called, but timer not at top of active stack.\n')

    d_cpu  = time.clock() - active_stack[-1].cpu_start
    d_wall = time.time()  - active_stack[-1].wall_start

    # If cumulative, don't subtract off time spent in nested timers.
    if cumulative:
        kid_wall = 0.
        kid_cpu  = 0.
    else:
        kid_wall = active_stack[-1].wall_kids
        kid_cpu  = active_stack[-1].cpu_kids

    #print("\nstop items on stack -- ")
    #for i in range(len(active_stack)):
    #    if i < len(active_stack)-1:
    #        print(str(i)+': '+str(active_stack[i].name))
    #    else:
    #        print('pop '+str(i)+': '+str(active_stack[i].name))

    # pop the child off the stack, then go through the active_stack and update
    # the child times for each of the parent timers
    active_stack[-1].running = False
    active_stack[-1].wall_time += (d_wall - kid_wall)
    active_stack[-1].cpu_time  += (d_cpu  - kid_cpu)
    active_stack.pop()

    # add the time for the child to the parent
    if len(active_stack) > 0:
        active_stack[-1].wall_kids += d_wall
        active_stack[-1].cpu_kids  += d_cpu


def print_timings():
    """Prints out a timing report, sorted from highest wall time to
    lowest."""
    global timer_list

    # ensure that the global timer has finished and get the total execution time
    if timer_list['global'].running:
        stop('global', cumulative=True)
    tot_cpu  = timer_list['global'].cpu_time
    tot_wall = timer_list['global'].wall_time

    sort_list = sorted(timer_list.items(),
                       key=lambda unsort: unsort[1].wall_time, reverse=True)

    # pass timing information as a string
    ostr =  '\n' + '-'*39 + ' timings summary ' + '-'*39 + ' \n'
    ostr += ('routine'.ljust(35) + 'calls'.rjust(12) +
             'wall time'.rjust(16)+'frac.'.rjust(8) +
             'cpu time'.rjust(16) +'frac.'.rjust(8) + '\n')
    ofrm = '{0:<35s}{1:>12d}{2:>16.4f}{3:>8.2f}{4:>16.4f}{5:>8.2f}\n'
    frac_wall = 0.
    frac_cpu  = 0.
    for i in range(len(sort_list)):
        rout = str(sort_list[i][0])
        if rout == 'global':
            continue
        ncall = sort_list[i][1].calls
        wtim  = sort_list[i][1].wall_time
        ctim  = sort_list[i][1].cpu_time
        ostr += ofrm.format(rout, ncall, wtim, wtim/tot_wall, ctim,
                            ctim/tot_cpu)
        frac_wall += wtim/tot_wall
        frac_cpu  += ctim/tot_cpu

    ostr += '-'*95 + '\n'
    ostr += ('**total**'.ljust(47) + '{0:>16.4f}{1:>8.2f}{2:>16.4f}{3:>8.2f}' +
             '\n\n'.format(tot_wall,frac_wall,tot_cpu,frac_cpu))
    return ostr
