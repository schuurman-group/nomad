"""
Compute saddle-point integrals over trajectories traveling on adiabataic
potentials

This currently uses first-order saddle point.
"""
import sys
import math
import numpy as np
import src.fmsio.glbl as glbl
import src.integrals.nuclear_gaussian as nuclear
interface = __import__('src.interfaces.' + glbl.interface['interface'],
                       fromlist = ['a'])

# Let FMS know if overlap matrix elements require PES info
overlap_requires_pes = False

# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the Hamiltonian symmetry
hermitian = False 

# Returns functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'

def nuc_overlap(t1, t2):
    """ Returns < Chi | Chi' >, the nuclear overlap integral of two trajectories"""
    return nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                           t2.phase(),t2.widths(),t2.x(),t2.p())

# the bra and ket functions for the s_integral may be different
# (i.e. pseudospectral/collocation methods).
def traj_overlap(t1, t2, nuc_only=False, Snuc=None):
    """ Returns < Psi | Psi' >, the overlap integral of two trajectories"""
    return s_integral(t1, t2, nuc_only=nuc_only, Snuc=Snuc)

# returns total overlap of trajectory basis function
def s_integral(t1, t2, nuc_only=False, Snuc=None):
    """ Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""
    if t1.state != t2.state and not nuc_only:
        return 0j
    else:
        if Snuc is None:
            return nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())
        else:
            return Snuc

def v_integral(t1, t2, centroid=None, Snuc=None):
    """Returns potential coupling matrix element between two trajectories."""
    # if we are passed a single trajectory, this is a diagonal
    # matrix element -- simply return potential energy of trajectory

    if Snuc is None:
        Snuc = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                               t2.phase(),t2.widths(),t2.x(),t2.p())

    if t1.state == t2.state:
        state = t1.state
        # Adiabatic energy
        vi = t1.energy(state) * Snuc
        vj = t2.energy(state) * Snuc

        if glbl.propagate['integral_order'] > 0:
            prim1 = nuclear.ordr1_vec(t1.widths(),t1.x(),t1.p(),
                                      t2.widths(),t2.x(),t2.p()) 
            vi += np.dot(prim1 - t1.x()*Snuc, t1.derivative(state,state))
            vj += np.dot(prim1 - t2.x()*Snuc, t2.derivative(state,state))

        if glbl.propagate['integral_order'] > 1:
            xcen  = (t1.widths()*t1.x() + t2.widths()*t2.x()) / (t1.widths()+t2.widths())
            prim2 = nuclear.ordr2_vec(t1.widths(),t1.x(),t1.p(),
                                      t2.widths(),t2.x(),t2.p())
            for k in range(t1.dim):
                vi += 0.5*prim2[k]*t1.hessian(state,state)[k,k]
                vj += 0.5*prim2[k]*t2.hessian(state,state)[k,k]
                for l in range(k):
                    vi += 0.5 * (2.*prim1[k]*prim1[l] 
                                 - xcen[k]*prim1[l] - xcen[l]*prim1[k] 
                                 - prim1[k]*t1.x()[l] - prim1[l]*t1.x()[k] 
                                 + (t1.x()[k]*xcen[l] + t1.x()[l]*xcen[k])*Snuc) 
                                 * t1.hessian(state,state)[k,l] 
                    vj += 0.5 * (2.*prim1[k]*prim1[l]
                                 - xcen[k]*prim1[l] - xcen[l]*prim1[k] 
                                 - prim1[k]*t2.x()[l] - prim1[l]*t2.x()[k] 
                                 + (t2.x()[k]*xcen[l] + t2.x()[l]*xcen[k])*Snuc) 
                                 * t2.hessian(state,state)[k,l]

        if glbl.propagate['integral_order'] > 2:
            sys.exit('integral_order > 2 not implemented for bra_ket_averaged')
          
    # [necessarily] off-diagonal matrix element between trajectories
    # on different electronic states
    else:
        # Derivative coupling
        fij = t1.derivative(t1.state, t2.state)

        vi = 2.*np.vdot(t1.derivative(t1.state,t2.state), interface.kecoeff *
                        nuclear.deldx(Snuc,t1.phase(),t1.widths(),t1.x(),t1.p(),
                                           t2.phase(),t2.widths(),t2.x(),t2.p()))
        vj = 2.*np.vdot(t2.derivative(t1.state,t2.state), interface.kecoeff *
                        nuclear.deldx(Snuc,t2.phase(),t2.widths(),t2.x(),t2.p(),
                                           t1.phase(),t1.widths(),t1.x(),t1.p()))
 
        if glbl.propagate['integral_order'] > 0:

        if glbl.propagate['integral_order'] > 1:

        if glbl.propagate['integral_order'] > 2:
            sys.exit('integral_order > 2 not implemented for bra_ket_averaged')
      
    return 0.5*(vi+vj) 


# kinetic energy integral
def ke_integral(t1, t2, Snuc=None):
    """Returns kinetic energy integral over trajectories."""
    if t1.state != t2.state:
        return 0j

    else:
        if Snuc is None:
            Snuc = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())

        ke = nuclear.deld2x(Snuc,t1.phase(),t1.widths(),t1.x(),t1.p(),
                                 t2.phase(),t2.widths(),t2.x(),t2.p())

        return -np.dot(ke, interface.kecoeff)

# time derivative of the overlap
def sdot_integral(t1, t2, Snuc=None, e_only=False, nuc_only=False):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if t1.state != t2.state:
        return 0j

    else:
        if Snuc is None:
            Snuc = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())

        deldx = nuclear.deldx(Snuc,t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())
        deldp = nuclear.deldp(Snuc,t1.phase(),t1.widths(),t1.x(),t1.p(),
                                   t2.phase(),t2.widths(),t2.x(),t2.p())

        sdot = (np.dot(deldx,t2.velocity()) + np.dot(deldp,t2.force())
                +1j * t2.phase_dot() * Snuc)

        return sdot
