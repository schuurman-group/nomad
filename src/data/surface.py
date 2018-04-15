"""
A class object to contain information about a potential energy surface
"""
import os

class Surface:
    """Object containing potential energy surface data."""
    standard_objs = ['geom','potential','derivative','hessians','coupling']
    optional_objs = ['mos','dipoles','atom_pop','sec_mom',
                     'diabat_pot','diabat_deriv','diabat_hessian',
                     'adiabat_pot','adiabat_deriv','adiabat_hessian',
                     'adt_mat','nac','scalar_coup']

    def __init__(self):
        # these are the standard quantities ALL interface_data objects return
        self.data     = dict()

    #
    #
    #
    def rm_item(key):
        """Adds new item to dictionary"""
        del self.data[key]

        return

    #
    #
    #
    def add_item(key, value):
        """Adds new item to dictionary"""
        if key in standard_objs+optional_objs:
            self.data[key] = value
        else:
            os.exit('Cannot add key='+str(key)+" to Surface instance: invalid key")

        return

    #
    #
    #
    def get_data(key):
        """Adds new item to dictionary"""
        if key in self.data:
            return self.data[key]
        else:
            os.exit('trying to get_item '+str(key)+' from Surface: item no present')
        return

    #
    #
    #
    def avail_data():
        """Adds new item to dictionary"""
        return self.data.keys()


    #
    #
    #
    def copy(self):
        """Creates a copy of a Surface object."""
        new_surface = Surface()

        # required potential data
        for key,value in self.data:
            new_surface.data[key] = copy.deepcopy(value)

        return new_surface

