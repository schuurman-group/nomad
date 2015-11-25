import globalvars
import columbus
import vibronic
#
# update the potential energy surface
#
def update_pes(pes_info):
    # update the potential energy surface information encoded by 
    # pes_info (i.e. energy and/or gradient and/or couplings, etc.)
    # by calling the interface defined by the global variable 'interface'
    if(interface == 'columbus'):
        run_columbus()
    elif(interface == 'vibronic'):
        run_vibronic()
    else:
        print "interface not recognized"
        os._exit(0)    


