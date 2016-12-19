"""
Compute saddle-point integrals over trajectories traveling on adiabataic
potentials

This currently uses first-order saddle point.
"""
import numpy as np
import src.integrals.nuclear_gaussian as nuclear
import src.interfaces.vibronic as vibronic 
import src.interfaces.vcham.hampar as ham

# Let propagator know if we need data at centroids to propagate
require_centroids = False 

# Determines the Hamiltonian symmetry
hermitian = False 

# returns basis in which matrix elements are evaluated
basis = 'gaussian'

def elec_overlap(traj1, traj2, centroid=None):
    """ Returns < Psi | Psi' >, the overlap integral of two trajectories"""

    # determine overlap of adiabatic wave functions in diabatic basis
    return complex( np.dot(traj1.pes_data.adt_mat[:,traj1.state],
                           traj2.pes_data.adt_mat[:,traj2.state]), 0.)

# returns the overlap between two trajectories (differs from s_integral in that
# the bra and ket functions for the s_integral may be different
# (i.e. pseudospectral/collocation methods). 
def traj_overlap(traj1, traj2, centroid=None, nuc_only=False):
    """ Returns < Psi | Psi' >, the overlap integral of two trajectories"""
    nuc_ovrlp = nuclear.overlap(traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                                traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    if nuc_only:
        return nuc_ovrlp
    else:
        return nuc_ovrlp * elec_overlap(traj1, traj2, centroid=centroid)

# total overlap of trajectory basis function
def s_integral(traj1, traj2, centroid=None, nuc_only=False, Snuc=None):
    """ Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""
    if Snuc is None:
        Snuc = traj_overlap(traj1, traj2, centroid=centroid, nuc_only=True)

    if nuc_only:
        return Snuc
    else:
        return Snuc * elec_overlap(traj1, traj2, centroid=centroid) 

def prim_v_integral(N, a1, x1, p1, a2, x2, p2, gauss_overlap=None):
    """Returns the matrix element <cmplx_gaus(q,p)| q^N |cmplx_gaus(q,p)>
     -- up to an overlap integral -- 
    """
    # since range(N) runs up to N-1, add "1" to result of floor
    n_2 = int(np.floor(0.5 * N) + 1)
    a   = a1 + a2
    b   = complex(2.*(a1*x1 + a2*x2),-(p1-p2))

    # generally these should be 1D harmonic oscillators. If
    # multi-dimensional, the final result is a direct product of
    # each dimension
    v_int = complex(0.,0.)
    for i in range(n_2):
        v_int += (a**(i-N) * b**(N-2*i) /
                 (np.math.factorial(i) * np.math.factorial(N-2*i)))

    # refer to appendix for derivation of these relations
    return v_int * np.math.factorial(N) / 2.**N

#
def v_integral(traj1, traj2, centroid=None, Snuc=None):
    """Returns potential coupling matrix element between two trajectories."""
    # evaluate just the nuclear component (for re-use)
    if Snuc is None:
        Snuc = nuclear.overlap(traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                               traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    # get the linear combinations corresponding to the adiabatic states
    nst = traj1.nstates
    st1 = traj1.pes_data.adt_mat[:,traj1.state]
    st2 = traj2.pes_data.adt_mat[:,traj2.state]

    # roll through terms in the hamiltonian
    v_total = complex(0.,0.)
    for i in range(ham.nterms):
        s1    = ham.stalbl[i,0] - 1
        s2    = ham.stalbl[i,1] - 1
      
        # adiabatic states in diabatic basis -- cross terms between orthogonal
        # diabatic states are zero
        if s1 == s2:
            cf    = ham.coe[i]
            v_term = complex(1.,0.)
            for q in range(ham.nmode_active):
                if ham.order[i,q] > 0:
                    v_term *= prim_v_integral(ham.order[i,q],
                               traj1.widths()[q],traj1.x()[q],traj1.p()[q],
                               traj2.widths()[q],traj2.x()[q],traj2.p()[q])            
            v_term *= cf * Snuc
        
            # now determine electronic factor
            v_term *= st1[s1] * st2[s2]     

            # add this term to the total integral
            v_total += v_term


    # return potential matrix element, multplied by nuclear overlap
    return v_total 

# kinetic energy integral
def ke_integral(traj1, traj2, centroid=None, Snuc=None):
    """Returns kinetic energy integral over trajectories."""
    # evaluate just the nuclear component (for re-use)
    if Snuc is None:
        Snuc = nuclear.overlap(traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                               traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    # overlap of electronic functions
    Selec = elec_overlap(traj1, traj2, centroid=centroid)

    # < chi | del^2 / dx^2 | chi'> 
    ke = nuclear.deld2x(Snuc,traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                             traj2.phase(),traj2.widths(),traj2.x(),traj2.p())
   
    return -sum( ke * vibronic.kecoeff) * Selec
#    return -sum(ke * np.array([0.5 for i in range(traj1.dim)]) ) * Selec

    
# time derivative of the overlap
def sdot_integral(traj1, traj2, centroid=None, Snuc=None):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if Snuc is None:
        Snuc = nuclear.overlap(traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                               traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    # overlap of electronic functions
    Selec = elec_overlap(traj1, traj2, centroid=centroid)

    # < chi | d / dx | chi'>
    deldx = nuclear.deldx(Snuc,traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                               traj2.phase(),traj2.widths(),traj2.x(),traj2.p())
    # < chi | d / dp | chi'>
    deldp = nuclear.deldp(Snuc,traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                               traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    # the nuclear contribution to the sdot matrix
    sdot = ( -np.dot(traj2.velocity(), deldx) + np.dot(traj2.force(), deldp)
            + 1j * traj2.phase_dot() * Snuc) * Selec

    # the derivative coupling
    deriv_coup = traj2.pes_data.grads[traj1.state,:]

    dia      = traj2.pes_data.diabat_pot
    diaderiv = traj2.pes_data.diabat_deriv
    v12      = dia[0,1]
    de       = dia[1,1] - dia[0,0]
    argt     = 2. * v12 / de
    theta    = 0.5 * np.arctan(argt)
    dtheta = np.array([(diaderiv[q,0,1]/de - v12*(diaderiv[q,1,1]- diaderiv[q,0,0])/de**2)/(1+argt**2) for q in range(traj2.dim)])

    st1  = traj1.pes_data.dat_mat[:,traj1.state]
    dst12  = np.array([[-np.sin(theta),np.cos(theta)],[-np.cos(theta),-np.sin(theta)]])
    deriv_coup2 = np.array([np.dot(dst12[traj2.state,:],np.array([dtheta[q],dtheta[q]])) for q in range(traj2.dim)])
#    print("d1 = "+str(deriv_coup))
#    print("d2 = "+str(deriv_coup2))
#    print("")

    # time-derivative of the electronic component
    sdot += np.dot(deriv_coup2, traj2.velocity()) * Snuc

    return sdot
 
