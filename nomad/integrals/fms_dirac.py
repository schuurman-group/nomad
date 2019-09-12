"""
Computes matrix elements over a dirac delta function at the centre of
trajectory 1 and trajectory 2.
"""
import numpy as np
import nomad.core.glbl as glbl
import nomad.compiled.nuclear_dirac as dirac
import nomad.compiled.nuclear_gaussian as gauss


# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the Hamiltonian symmetry
hermitian = False

# functional form of bra function ('dirac_delta', 'gaussian')
basis = 'dirac_delta'


def elec_overlap(t1, t2):
    """ Returns < Psi | Psi' >, the nuclear overlap integral of two trajectories"""
    return float(t1.state == t2.state)


def nuc_overlap(t1, t2):
    """ Returns < Chi | Psi >, the nuclear overlap integral under the pseudospectral projection"""
    return dirac.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                         t2.phase(),t2.widths(),t2.x(),t2.p())


def traj_overlap(t1, t2):
    """ Returns < Chi | Chi' >, the overlap integral of two trajectories"""
    return elec_overlap(t1,t2) * gauss.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                               t2.phase(),t2.widths(),t2.x(),t2.p())


def s_integral(traj1, traj2, nuc_ovrlp, elec_ovrlp):
    """ Returns < chi | Psi' >, the overlap integral under the pseudospectral
    projection."""
    return nuc_ovrlp * elec_ovrlp


def v_integral(traj1, traj2, kecoef, nuc_ovrlp, elec_ovrlp):
    """ Returns < delta(R-R1) | V | g2 > if state1 = state2, else
    returns < delta(R-R1) | F . d/dR | g2 > """

    # Off-diagonal element between trajectories on the same adiabatic
    # state
    if  traj1.state == traj2.state:
        # Adiabatic energy
        return traj1.energy(traj1.state) * nuc_ovrlp

    # off-diagonal matrix elements between trajectories on different
    # elecronic states
    elif traj1.state != traj2.state:
        # Derivative coupling
        fij = traj1.derivative(traj2.state)
        v = np.dot(fij, 2. *kecoef*
                          dirac.deldx(nuc_ovrlp,traj1.x(),
                                      traj2.widths(),traj2.x(),traj2.p()))
        return v * nuc_ovrlp
    else:
        print('ERROR in v_integral -- argument disagreement')
        return 0j


def t_integral(traj1, traj2, kecoef, nuc_ovrlp, elec_ovrlp):
    """ Returns < delta(R-R1) | T | g2 > """
    if elec_ovrlp == 0.:
        return 0j

    else:
        ke = dirac.deld2x(nuc_ovrlp,traj1.x(),
                                 traj2.widths(),traj2.x(),traj2.p())

        return -sum(ke * kecoef)


def sdot_integral(traj1, traj2, nuc_ovrlp=None):
    """ Returns < delta(R-R1) | d/dt g2 >

    Note that for now we have to pass nuc_ovrlp=1 to deldx and deldp so
    that these functions do not multiply the result by the overlap
    of traj1 and traj2. This isn't ideal, but will do for now.
    """
    if traj1.state != traj2.state:
        return 0j
    else:
        if nuc_ovrlp is None:
            nuc_ovrlp = dirac.overlap(traj1.x(),
                                 traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

        sdot = (np.dot(traj2.velocity(),
                       dirac.deldx(nuc_ovrlp,traj1.x(),
                          traj2.widths(),traj2.x(),traj2.p())) +
                np.dot(traj2.force(),
                       dirac.deldp(nuc_ovrlp,traj1.x(),traj2.x())) +
                1j * traj2.phase_dot() * nuc_ovrlp)

        return sdot
