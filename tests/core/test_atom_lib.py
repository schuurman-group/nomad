"""
Tests for the core.atom_lib module.
"""
import pytest
import numpy as np
import nomad.core.atom_lib as atom_lib


def test_valid_atom():
    assert atom_lib.valid_atom('H')
    assert atom_lib.valid_atom('C')
    assert not atom_lib.valid_atom('CH4')
    assert not atom_lib.valid_atom('ijk')


def test_atom_data_X():
    dat = atom_lib.atom_data('X')
    assert np.allclose(dat, np.zeros(3))


def test_atom_data_H():
    dat = atom_lib.atom_data('H')
    assert np.isclose(dat[0], 4.5)
    assert np.isclose(dat[1], 1.007825037*1822.887)
    assert np.isclose(dat[2], 1)


def test_atom_data_fails():
    with pytest.raises(KeyError, match=r'Atom: .* not found in library'):
        atom_lib.atom_data('ijk')
