#
# Depending the the user-requested interface, initialze 
# the user-requested interface based on value of interface
# variable
#
import variable
import vibronic
import columbus
#
#
#

def load_interface():
    iface = variable.fms['interface']
    if iface == 'vibronic':
        vibronic.load_operator()
    elif iface == 'columbus':
        pass
    elif iface == 'gamess':
        print("GAMESS interface not yet available")
    elif iface == 'molpro':
        print("GAMESS interface not yet available")
    else:
        print("ERROR: interface "+str(ifac)+" not recognized.")


