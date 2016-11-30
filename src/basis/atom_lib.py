"""
A library containing atom widths, atomic numbers, etc.
"""

atom_name  = ['H', 'D', 'T', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
                  'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']
atom_width = [4.5, 4.5, 4.5, 0.0, 0.0, 0.0, 0.0, 22.5, 19.5, 13.0, 8.5,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.5, 0.0, 0.0]
atom_mass  = [1.0, 2.0, 3.0, 4.0, 7.0, 9.0, 11.0, 12.0, 14.0, 16.0, 19.0,
                  20.0, 23.0, 24.0, 27.0, 28.0, 31.0, 32.0, 35.45, 40.0]
atom_anum  = [1., 1., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                  10., 11., 12., 13., 14., 15., 16., 17., 18.]


def valid_atom(atom_sym):
    """Returns True if the atomic symbol is in the library."""
    return atom_sym in atom_name

def atom_data(atom_sym):
    """Returns the gaussian width, mass, and atomic number from the
       atomic symbol"""
    
    if atom_sym in atom_name:
        index = atom_name.index(atom_sym)
        return atom_width[index],atom_mass[index],atom_anum[index]

    else:
        raise ValueError('Atom: '+str(atom_sym)+' not found in library') 
        return
