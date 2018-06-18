"""
A class object to contain information about a potential energy surface
"""
import copy


class Surface:
    """Object containing potential energy surface data."""
    def __init__(self):
        self.standard_objs = ['geom','momentum','potential','derivative','hessian','coupling']
        self.optional_objs = ['mo','dipole','atom_pop','sec_mom',
                         'diabat_pot','diabat_deriv','diabat_hessian',
                         'adt_mat','dat_mat','nac','scalar_coup']

        # these are the standard quantities ALL interface_data objects return
        self.data = dict()

    def rm_data(self, key):
        """Adds new item to dictionary"""
        del self.data[key]

    def add_data(self, key, value):
        """Adds new item to dictionary"""
        if key in self.standard_objs + self.optional_objs:
            self.data[key] = value
        else:
            raise KeyError('Cannot add key='+str(key)+' to Surface instance: invalid key')

    def get_data(self, key):
        """Adds new item to dictionary"""
        if key in self.data:
            return self.data[key]
        else:
            raise ValueError('(get_data('+str(key)+') from Surface: datum not present')

    def avail_data(self):
        """Adds new item to dictionary"""
        return self.data.keys()

    def copy(self):
        """Creates a copy of a Surface object."""
        new_surface = Surface()

        # required potential data
        for key,value in self.data.items():
            new_surface.data[key] = copy.deepcopy(value)

        return new_surface
