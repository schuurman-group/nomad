"""
Compute Bra-ket averaged Taylor expansion integrals over trajectories
traveling on adiabataic potentials
"""
import numpy as np
import nomad.compiled.nuclear_gaussian as nuclear


# Determines the Hamiltonian symmetry
hermitian = True

# Returns functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'

def initialize():
    """ initialize integrals"""

    return


def state_vector(t1):
    """Return the component of each electronic state in trajectory
       electronic function"""
    return np.asarray([1. if state==t1.state else 0. for state 
                                  in range(t1.nstates)], dtype=float)

def elec_overlap(t1, t2):
    """ Returns < Psi | Psi' >, the electronic overlap integral of two trajectories"""
    return float(t1.state == t2.state)


def nuc_overlap(t1, t2):
    """ Returns < Chi | Chi' >, the nuclear overlap integral of two trajectories"""
    return nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                           t2.phase(),t2.widths(),t2.x(),t2.p())

def traj_overlap(t1, t2):
    """ Returns < Chi | Chi' >, the total overlap integral of two trajectories"""
    return elec_overlap(t1,t2) * nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                                 t2.phase(),t2.widths(),t2.x(),t2.p())


def s_integral(t1, t2, nuc_ovrlp, elec_ovrlp):
    """ Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""

    return nuc_ovrlp * elec_ovrlp


def t_integral(t1, t2, nuc_ovrlp, elec_ovrlp):
    """Returns kinetic energy integral over trajectories."""
    if elec_ovrlp == 0.:
        return 0.j

    ke = nuclear.deld2x(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                  t2.widths(),t2.x(),t2.p())

    return -np.dot(ke, t1.kecoef)


def sdot_integral(t1, t2, nuc_ovrlp, elec_ovrlp):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if elec_ovrlp == 0.:
        return elec_ovrlp

    deldx = nuclear.deldx(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                    t2.widths(),t2.x(),t2.p())
    deldp = nuclear.deldp(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                    t2.widths(),t2.x(),t2.p())

    sdot = (np.dot(deldx,t2.velocity()*complex(1.,0.)) + 
            np.dot(deldp,t2.force()*complex(1.,0.)) +
            1j * t2.phase_dot() * nuc_ovrlp)

    return sdot

def popwt(t1, t2, nuc_ovrlp=None):
    """returns the population weight in adiabatic basis"""

    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    pop = np.zeros(t1.nstates, dtype=complex)

    if t1.state == t2.state:
      pop[t1.state] = nuc_ovrlp

    return pop

