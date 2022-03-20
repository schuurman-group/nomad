"""
Library of functions for fitting populations curves.
All fits assume the final 'state' is unpopulated in the beginning,
and the sum of populations of other 'states' is unity. The functions
return an array of size nstates x ntimes (always 2D, even if nstates == 1).
              k1   k1
Sequential: A -> B -> ...
               k1    k2
Equilibria: A <=> B <=> ...
              k-1   k-2
Deriving these equations involves solving systems of ODEs. For a concise
review, see https://pubs.acs.org/doi/abs/10.1021/ed076p1578
"""
import numpy as np


varlbls = dict(
    single_exp = ['tau1'],
    single_expc = ['tau1', 'c'],
    single_biexp = ['amp1', 'tau1', 'amp2', 'tau2'],
    single_triexp = ['amp1', 'tau1', 'amp2', 'tau2', 'amp3', 'tau3'],
    double_seq = ['tau1'],
    double_eq = ['tau1', 'tau-1'],
    triple_seq = ['A0', 'tau1', 'tau2'],
    triple_1steq = ['A0', 'tau1', 'tau-1', 'tau2'],
    triple_2ndeq = ['A0', 'tau1', 'tau2', 'tau-2'],
    triple_alleq = ['A0', 'tau1', 'tau-1', 'tau2', 'tau-2']
               )


def get_labels(name):
    """Returns a list of variable labels based on function name."""
    if 'delay_' in name:
        basename = name.replace('delay_', '')
        return ['t0'] + varlbls[basename]
    else:
        return varlbls[name]


def single_exp(x, tau1):
    """Returns an exponential function."""
    return np.atleast_2d(1 - np.exp(-x / tau1))


def single_expc(x, tau1, c):
    """Returns an exponential function plus a constant."""
    return np.atleast_2d(1 - np.exp(-x / tau1) - c)


def single_biexp(x, a1, tau1, a2, tau2):
    """Returns a biexponential function."""
    return np.atleast_2d(1 - a1*np.exp(-x / tau1) - a2*np.exp(-x / tau2))


def single_triexp(x, a1, tau1, a2, tau2, a3, tau3):
    """Returns a triexponential function."""
    return np.atleast_2d(1 - a1*np.exp(-x / tau1) - a2*np.exp(-x / tau2) -
                         a3*np.exp(-x / tau3))


def double_seq(x, tau1):
    """Returns reactant-product sequential functions."""
    ab = _empty_a(2, x)
    ab[1] = np.exp(-x / tau1)
    ab[0] = 1 - ab[1]
    return ab


def double_eq(x, tau1, taum1):
    """Returns reactant-product equilibrium functions."""
    ab = _empty_a(2, x)
    ttau = tau1 + taum1
    ab[1] = (1/ttau)*(tau1 + taum1*np.exp(-ttau*x / (tau1*taum1)))
    ab[0] = 1 - ab[1]
    return ab


def triple_seq(x, a0, tau1, tau2):
    """Returns reactant-intermediate-product sequential functions."""
    abc = _empty_a(3, x)
    abc[2] = a0*np.exp(-x/tau1)
    abc[1] = a0*tau2/(tau1 - tau2) * (np.exp(-x/tau1) -
                                      np.exp(-x/tau2)) + (1-a0)*np.exp(-x/tau2)
    abc[0] = 1 - abc[2] - abc[1]
    return abc


def triple_1steq(x, a0, tau1, taum1, tau2):
    """Returns reactant-intermediate-product with R-I equilibrium."""
    abc = _empty_a(3, x)
    k1, km1, k2 = 1/tau1, 1/taum1, 1/tau2
    s = k1 + km1 + k2
    c = k1*k2
    g1 = 0.5*(s + np.sqrt(s**2 - 4*c))
    g2 = 0.5*(s - np.sqrt(s**2 - 4*c))
    abc[2] = ((a0*(g1 - km1 - k2) - (1-a0)*km1)/(g1 - g2) * np.exp(-g1*x) -
              (a0*(g2 - km1 - k2) - (1-a0)*km1)/(g1 - g2) * np.exp(-g2*x))
    abc[1] = ((-a0*k1 + (1-a0)*(g1 - k1))/(g1 - g2) * np.exp(-g1*x) -
              (-a0*k1 + (1-a0)*(g2 - k1))/(g1 - g2) * np.exp(-g2*x))
    abc[0] = 1 - abc[2] - abc[1]
    return abc


def triple_2ndeq(x, a0, tau1, tau2, taum2):
    """Returns reactant-intermediate-product with I-P equilibrium."""
    abc = _empty_a(3, x)
    k1, k2, km2 = 1/tau1, 1/tau2, 1/taum2
    s = k1 + k2 + km2
    c = k1*(k2 + km2)
    g1 = 0.5*(s + np.sqrt(s**2 - 4*c))
    g2 = 0.5*(s - np.sqrt(s**2 - 4*c))
    abc[2] = a0*np.exp(-x/tau1)
    abc[1] = ((a0*k1*(km2 - g1) +
               (1-a0)*(g1 - k1)*(g1 - km2))/(g1*(g1 - g2)) * np.exp(-g1*x) -
              (a0*k1*(km2 - g2) +
               (1-a0)*(g2 - k1)*(g2 - km2))/(g2*(g1 - g2)) * np.exp(-g2*x) +
              k1*km2/(g1*g2))
    abc[0] = 1 - abc[2] - abc[1]
    return abc


def triple_alleq(x, a0, tau1, taum1, tau2, taum2):
    """Returns reactant-intermediate-product with R-I and I-P equilibria."""
    abc = _empty_a(3, x)
    k1, km1, k2, km2 = 1/tau1, 1/taum1, 1/tau2, 1/taum2
    s = k1 + km1 + k2 + km2
    c = km1*km2 + k1*(k2 + km2)
    g1 = 0.5*(s + np.sqrt(s**2 - 4*c))
    g2 = 0.5*(s - np.sqrt(s**2 - 4*c))
    abc[2] = ((a0*k1*(g1 - k2 - km2) +
               (1-a0)*km1*(km2 - g1))/(g1*(g1 - g2)) * np.exp(-g1*x) -
              (a0*k1*(g2 - k2 - km2) +
               (1-a0)*km1*(km2 - g2))/(g2*(g1 - g2)) * np.exp(-g2*x) +
              km1*km2/(g1*g2))
    abc[1] = ((a0*k1*(km2 - g1) +
               (1-a0)*(g1 - k1)*(g1 - km2))/(g1*(g1 - g2)) * np.exp(-g1*x) -
              (a0*k1*(km2 - g2) +
               (1-a0)*(g2 - k1)*(g2 - km2))/(g2*(g1 - g2)) * np.exp(-g2*x) +
              k1*km2/(g1*g2))
    abc[0] = 1 - abc[2] - abc[1]
    return abc


def add_delay(func):
    """Returns a decay function with a time delay."""
    def delayed(x, x0, *args):
        xp = x - x0
        if np.all(xp < 0):
            abc = func(0, *args)
        else:
            pos = func(xp[xp>=0], *args)
            abc = _empty_a(len(pos), x)
            abc[:,xp>=0] = pos
            abc[:,xp<0] = func(0, *args)

        return abc

    return delayed


def ravelf(func):
    """Returns a function which returns the raveled output for tiled x.
    This is necessary when using scipy.optimize.curve_fit for multiple
    curves with a single independent variable.
    """
    def raveled(x, *args):
        nr = len(func(0, *args))
        return func(x[:len(x)//nr], *args).ravel()

    return raveled


def _empty_a(n, x):
    """Sets up the empty array based on the type and size of x."""
    try:
        return np.empty((n, len(x)))
    except TypeError:
        return np.empty((n, 1))
