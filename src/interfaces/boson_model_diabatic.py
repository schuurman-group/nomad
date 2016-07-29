"""
Routines for running spin-boson model calculations.
"""
import numpy as np
from src.fmsio import glbl


comp_properties = False
ncrd = 4
delta = 1.
omega_c = 2.5 * delta
omega = np.zeros(ncrd)
C = np.zeros(ncrd)


def init_interface():
    """Initializes global variables."""
    global C, omega, omega_c, delta

    if ncrd == 1:
        omega = np.array(1.34)
        delta   = 0.
        omega_c = 2.5
        C = np.zeros(ncrd)
    elif ncrd == 4:
        omega   = np.array([0.01, 1.34, 2.67, 4.00])
        delta   = 1.
        omega_c = 2.5 * delta
        d_omega = 1.33
        alpha   = glbl.boson['coupling']
        C = np.sqrt(d_omega * alpha * omega * np.exp(-omega/omega_c))


def energy(geom):
    """Evaluates energy in the spin-boson model.

    This does not include the kinetic component of h0, given
    by sum(0.5 * omega * momentum**2).
    """
    sgn = np.array([-1., 1.])
    h0 = sum(0.5 * omega * geom**2)
    hk0 = sum(C * geom)
    return h0 + sgn * hk0


def derivative(geom, t_state):
    """Returns the energy gradient in the spin-boson model."""
    grads = np.zeros((2, ncrd))

    sgn = -1. + 2.*t_state
    grads[t_state] = omega*geom + sgn*C

    coup = delta
    grads[1-t_state] = np.array([coup for i in range(ncrd)])
    return grads


def evaluate_trajectory(tid, geom, t_state):
    """Evaluates trajectory energy and gradients."""
    gm = np.array([geom[i].x[j] for i in range(ncrd)
                   for j in range(geom[i].dim)], dtype=float)
    eners = energy(gm)
    grads = derivative(gm, t_state)
    return gm, eners, grads


def evaluate_worker(packet, global_var):
    """Evaluates worker for parallel job.

    Global variables passed as parameters.
    """
    global ncrd, omega, C, delta

    tid = packet[0]
    geom = packet[1]
    t_state = packet[2]

    set_global_vars(global_var)

    xval = [geom[i].x[0] for i in range(len(geom))]
    dims = [geom[i].dim for i in range(len(geom))]

    gm = np.array([geom[i].x[j] for i in range(ncrd)
                   for j in range(geom[i].dim)], dtype=float)

    eners = energy(gm)
    grads = derivative(gm, t_state)

    return gm, eners, grads


def set_global_vars(gvars):
    """Sets the value of global variables."""
    global ncrd, omega, C, delta

    ncrd  = gvars[0]
    omega = gvars[1]
    delta = gvars[2]
    C     = gvars[3]


def get_global_vars():
    """Returns the global variables."""
    return ncrd, omega, delta, C
