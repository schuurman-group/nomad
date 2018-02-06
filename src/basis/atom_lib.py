"""
A library containing atom widths, atomic numbers, etc.
"""
import src.fmsio.glbl as glbl


atom_name  = ['H', 'D', 'T', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
              'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']
atom_width = [4.5, 4.5, 4.5, 0.0, 0.0, 0.0, 0.0, 22.5, 19.5, 13.0, 8.5,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.5, 0.0, 0.0]
atom_mass  = [1.007825037, 2.015650074, 3.023475111, 4.00260325, 7.0160045,
              9.0121825, 11.0093053, 12.0, 14.003074008, 15.99491464, 18.99840325,
              19.9924391, 22.9897697, 23.9850450, 26.9815413, 27.9769284,
              30.9737634, 31.9720718, 34.968852729, 39.9623831]
atom_anum  = [1., 1., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
              10., 11., 12., 13., 14., 15., 16., 17., 18.]


def valid_atom(atom_sym):
    """Returns True if the atomic symbol is in the library."""
    return atom_sym in atom_name


def atom_data(atom_sym):
    """Returns the gaussian width, mass, and atomic number from the
       atomic symbol"""

    if valid_atom(atom_sym):
        index = atom_name.index(atom_sym)
        return atom_width[index],atom_mass[index]*glbl.constants['amu2au'],atom_anum[index]

    else:
        raise KeyError('Atom: '+str(atom_sym)+' not found in library')
