import time

timer_list = dict()
active_stack = []

class timer:
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


def time_func(f):
    global timer_list, active_stack

    def tt(*args, **kwargs):
        name = f.__name__
        if name not in timer_list:
            timer_list[name] = timer[name]

        twall = time.time()
        tcpu  = time.clock()
        timer_list[name].calls     += 1
        timer_list[name].wall_start = twall
        timer_list[name].cpu_start  = tcpu
        active_stack.append(timer_list[name])

        res = f(*args, **kwargs)

        f_wall = time.time() - 
        f_cpu  = time.clock()
        

        print('{tabs}Function <{name}> execution time: {time:.3f} seconds'.format(
            tabs=tabs, name=name, time=time.time() - t0))
        time_it.active -= 1
        return res
    return tt
