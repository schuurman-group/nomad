import time
import operator
#
# timers and timing information for subroutines
#
#
timer_list = dict()

class timer:
  
    #
    # timer for process "name"
    #
    def __init__(self, name):
        self.name        = name
        self.calls       = 0
        self.cpu_time    = 0
        self.wall_time   = 0 
        self.cpu_start   = 0
        self.wall_start  = 0
        self.running     = False

#
# start the timer given by "name". If it doesn't exist, create it and 
#  add it to the list of timers
#
def start(name):
    global timer_list

    # 
    # if timer already in list, activate it
    #
    if name not in timer_list:
        timer_list[name] = timer(name)

    #
    # if already running, let me know...
    #
    if timer_list[name].running:
        print("timer: "+str(name)+" already active")
    
    timer_list[name].calls     += 1
    timer_list[name].running    = True
    timer_list[name].cpu_start  = time.clock()
    timer_list[name].wall_start = time.time()

    return

#
# stop the timer given by "name". Trying to stop an unknown timer throws
#  an error
#
def stop(name):
    global timer_list

    #
    # if timer not in list, print an error and return
    #
    if name not in timer_list:
       print("timer: "+str(name)+" does not exist, cannot stop")

    timer_list[name].cpu_time  += (time.clock() - timer_list[name].cpu_start)
    timer_list[name].wall_time += (time.time() - timer_list[name].wall_start)
    timer_list[name].running = False

#
# print out a timing report, sorted from highest wall time to lowest
#
def print_timings():
    global timer_list

    # ensure that the global timer has finished and get the total execution time
    if timer_list['global'].running:
        stop('global') 
    tot_cpu  = timer_list['global'].cpu_time
    tot_wall = timer_list['global'].wall_time

    sort_list = sorted(timer_list.items(), key=lambda unsort: unsort[1].wall_time,reverse=True)


    # pass timing information as a string
    
    ostr =  '\n'+'-'*32+' timings summary '+'-'*32+' \n'
    ostr +=  '-routine-'.ljust(30) \
            +'-wall time-'.rjust(12)+'-frac.-'.rjust(12) \
            +'-cpu time-'.rjust(12) +'-frac.-'.rjust(12)+'\n'
    ofrm = "{0:<30s}{1:>12.4f}{3:>12.2f}{2:>12.4f}{3:>12.2f}\n"
    timed_total = 0.
    for i in range(len(sort_list)):
        rout = str(sort_list[i][0])
        if rout == 'global':
            continue
        wtim = sort_list[i][1].wall_time
        ctim = sort_list[i][1].cpu_time
        ostr += ofrm.format(rout,wtim,wtim/tot_wall,ctim,ctim/tot_cpu)
        timed_total += wtim

    ostr += '-'*81+'\n'
    ostr += '**total**'.ljust(30)+'{0:>12.4f}{1:>24.4f}\n\n'.format(tot_wall,tot_cpu) 
    ostr += ' timed segments account for {0:6.3f} of the total time.'.format(timed_total/tot_wall)

    return ostr       




    


