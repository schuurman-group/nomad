"""
Module for interpreting potential energies for plotting.
"""
import os
import numpy as np
from fmsinterpreter import fileio


def conv_nrg(lbl, e, order=None, conv=1., base=None, states=None):
    """Reorders labels and energies according to pltorder and converts
    energies to desired units."""
    if order is None:
        order = range(len(lbl))
    if base is None:
        base = np.amin(e)
    if states is None:
        states = range(len(e[0]))

    new_lbl = [lbl[i] for i in order]
    new_e = conv * (e[order][:,states] - base)

    return new_lbl, new_e
