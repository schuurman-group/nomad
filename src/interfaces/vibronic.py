"""
Routines for running a vibronic coupling calculation.

Much of this could benefit from changing for loops to numpy array operations.
(But this is so computationally cheap that it really doesn't matter...)
"""
import sys
import copy
import numpy as np
import scipy.linalg as sp_linalg
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio


kecoeff = None
ham = None
nsta = glbl.fms['n_states']
data_cache = dict()


class Surface:
    """Object containing potential energy surface data."""
    def __init__(self, n_states, t_dim, crd_dim):
        # necessary for array allocation
        self.n_states = n_states
        self.t_dim    = t_dim
        self.crd_dim  = crd_dim

        # these are the standard quantities ALL interface_data objects return
        self.data_keys = []
        self.geom      = np.zeros(t_dim)
        self.potential = np.zeros(n_states)
        self.deriv     = np.zeros((t_dim, n_states, n_states))

        # these are interface-specific quantities
        self.scalar_coup   = np.zeros((n_states, n_states))
        self.adt_mat       = np.zeros((n_states, n_states))
        self.dat_mat       = np.zeros((n_states, n_states))
        self.ddat_mat      = np.zeros((t_dim,n_states, n_states))
        self.diabat_pot    = np.zeros((n_states, n_states))
        self.diabat_deriv  = np.zeros((t_dim, n_states, n_states))
        self.adiabat_pot   = np.zeros((n_states, n_states))
        self.adiabat_deriv = np.zeros((t_dim, n_states, n_states))


def copy_surface(orig_info):
    """Creates a copy of a Surface object."""
    # Perhaps should have more robust checking that "orig_info" is in fact
    # a 'Surface' object
    if orig_info is None:
        return None

    new_info = Surface(orig_info.n_states,
                       orig_info.t_dim,
                       orig_info.crd_dim)

    new_info.data_keys     = copy.copy(orig_info.data_keys)
    new_info.geom          = copy.deepcopy(orig_info.geom)
    new_info.potential     = copy.deepcopy(orig_info.potential)
    new_info.deriv         = copy.deepcopy(orig_info.deriv)

    new_info.scalar_coup   = copy.deepcopy(orig_info.scalar_coup)
    new_info.adt_mat       = copy.deepcopy(orig_info.adt_mat)
    new_info.dat_mat       = copy.deepcopy(orig_info.dat_mat)
    new_info.ddat_mat      = copy.deepcopy(orig_info.ddat_mat)
    new_info.diabat_pot    = copy.deepcopy(orig_info.diabat_pot)
    new_info.diabat_deriv  = copy.deepcopy(orig_info.diabat_deriv)
    new_info.adiabat_pot   = copy.deepcopy(orig_info.adiabat_pot)
    new_info.adiabat_deriv = copy.deepcopy(orig_info.adiabat_deriv)
    return new_info


class VibHam:
    """Object containing the vibronic Hamiltonian parameters."""
    def __init__(self):
        # Paramters
        self.npar = 0
        self.apar = None
        self.par = None

        # Hamiltonian terms
        self.nterms = 0
        self.coe = None
        self.stalbl = None
        self.order = None
        #self.mode = None
        self.freq = None
        self.mlbl_total = None
        self.mlbl_active = None
        self.nmode_total = 0
        self.nmode_active = 0
        self.freqmap = dict()
        self.parmap = dict()

    def rdfreqfile(self, fname):
        """Reads and interprets a freq.dat file."""
        with open(fileio.home_path + '/' + fname, 'r') as infile:
            keywords = get_kwds(infile)

        self.nmode_active = len(keywords)
        self.mlbl_active = ['' for i in range(self.nmode_active)]
        self.freq = np.zeros(self.nmode_active)
        for i, kwd in enumerate(keywords):
            self.mlbl_active[i] = kwd[0]
            self.freq[i] = conv(kwd[1:])
            self.freqmap[kwd[0]] = self.freq[i]

    def rdoperfile(self, fname):
        """Reads and interprets an operator file."""
        with open(fileio.home_path + '/' + fname, 'r') as infile:
            keywords = get_kwds(infile)

        parstart = keywords.index(['parameter-section'])
        parend   = keywords.index(['end-parameter-section'])
        self.npar = parend - parstart - 1
        self.apar = ['' for i in range(self.npar)]
        self.par  = np.zeros(self.npar)
        for i in range(self.npar):
            kwd = keywords[parstart + i + 1]
            self.apar[i] = kwd[0]
            if kwd[1] != '=':
                raise NameError('No argument has been given with the keyword: ' +
                                kwd[0])
            self.par[i] = conv(kwd[2:])
            self.parmap[kwd[0]] = self.par[i]

        hamstart = keywords.index(['hamiltonian-section'])
        hamend   = keywords.index(['end-hamiltonian-section'])
        i = hamstart + 1
        while i < hamend:
            kwd = keywords[i]
            if kwd[0] != 'modes' and i == hamstart + 1:
                raise ValueError('The Hamiltonian section must start with the '
                                 'mode specification!')
            elif kwd[0] != 'modes':
                break
            while '|' in kwd:
                kwd.remove('|')
            self.nmode_total += len(kwd) - 1
            self.mlbl_total = kwd[1:]
            i += 1

        nterms = hamend - i
        coe = np.zeros(nterms)
        order = np.zeros((nterms, self.nmode_total), dtype=int)
        # Should just have a list of modes and a list of orders. There is no need
        # to loop through lists of mostly unset variables.
        #mode = ['' for i in range(self.nterms)]
        #order = np.zeros(self.nterms, dtype=int)
        stalbl = np.zeros((nterms, 2), dtype=int)
        active = np.ones(nterms, dtype=bool)
        for i in range(nterms):
            kwd = keywords[hamend - nterms + i]
            if '^' in kwd:
                coeend = len(kwd) - 4
                modei = int(kwd[coeend]) - 1
                order[i,modei] = int(kwd[coeend+2])
                #self.mode[i] = int(kwd[coeend])
                #self.order[i] = int(kwd[coeend+2])
                active[i] = self.mlbl_total[modei] in self.mlbl_active
            else:
                coeend = len(kwd) - 1
            coe[i] = getcoe(kwd[:coeend])
            states = sorted(kwd[-1][1:].split('&'))
            stalbl[i] = [int(s) for s in states]

        self.nterms = sum(active)
        self.coe = coe[active]
        self.stalbl = stalbl[active]
        self.order = order[active]


def init_interface():
    """Reads the freq.dat file and the operator file.

    Note that, at least for now, the no. active modes and their
    labels are determined from the freq.dat file.
    As such, we must read the freq.dat file BEFORE reading the
    operator file.
    """
    global kecoeff, ham

    # Read in frequency and operator files
    ham = VibHam()
    ham.rdfreqfile('freq.dat')
    ham.rdoperfile(glbl.fms['opfile'])

    # KE operator coefficients, mass- and frequency-scaled normal mode
    # coordinates, a_i = 0.5*omega_i
    kecoeff = 0.5*ham.freq

    # Ouput some information about the Hamiltonian
    fileio.print_fms_logfile('string', ['*'*72])
    fileio.print_fms_logfile('string',
                             ['* Vibronic Coupling Hamiltonian Information'])
    fileio.print_fms_logfile('string', ['*'*72])
    fileio.print_fms_logfile('string',
                             ['Operator file: ' + glbl.fms['opfile']])
    fileio.print_fms_logfile('string',
                             ['Number of Hamiltonian terms: ' + str(ham.nterms)])
    string = 'Total no. modes: ' + str(ham.nmode_total)
    fileio.print_fms_logfile('string', [string])

    string = 'No. active modes: ' + str(ham.nmode_active)
    fileio.print_fms_logfile('string', [string])

    fileio.print_fms_logfile('string', ['Active mode labels:'])
    for i in range(ham.nmode_active):
        string = str(i+1) + ' ' + ham.mlbl_active[i]
        fileio.print_fms_logfile('string', [string])

    fileio.print_fms_logfile('string', ['Active mode frequencies (a.u.):'])
    for i in range(ham.nmode_active):
        string = str(i+1) + ' ' + str(ham.freq[i])
        fileio.print_fms_logfile('string', [string])


def evaluate_trajectory(tid, geom, stateindx):
    """Evaluates the trajectory."""
    global data_cache

    # Set the current normal mode coordinates
    qcoo = geom

    # Calculation of the diabatic potential matrix
    diabpot = calc_diabpot(qcoo)

    # Calculation of the adiabatic potential vector and ADT matrix
    adiabpot, adtmat = calc_adt(tid, diabpot)

    # Calculation of the nuclear derivatives of the diabatic potential
    diabderiv1 = calc_diabderiv1(qcoo)

    # Calculation of the NACT matrix
    nactmat = calc_nacts(adiabpot, adtmat, diabderiv1)

    # Calculation of the gradients of the adiabatic potential
    adiabderiv1 = calc_adiabderiv1(adtmat, diabderiv1)

    # Calculation of the Laplacian of the diabatic potential wrt the
    # nuclear DOFs
    diablap = calc_diablap(qcoo)

    # Calculation of the scalar couplings terms (SCTs)
    # Note that in order to calculate the SCTs, we get the gradients of the
    # diagonal Born-Oppenheimer corrections (DBOCs). Consequently, we
    # save these as a matter of course.
    sctmat, dbocderiv1 = calc_scts(adiabpot, adtmat, diabderiv1,
                                   nactmat, adiabderiv1, diablap)

    #print("adt="+str(adtmat))
    #print("dat="+str(np.linalg.inv(adtmat)))
    #de    = diabpot[1,1]-diabpot[0,0]
    #v12   = diabpot[0,1]
    #argt  = 2.*v12/de
    #theta = 0.5*np.arctan(argt)
    #print("dat2="+str([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]))
    #dderiv = np.array([(diabderiv1[q,0,1]/de - v12*(diabderiv1[q,1,1]- diabderiv1[q,0,0])/de**2)/(1+argt**2) for q in range(len(qcoo))])
    #ddat2  = np.array([[[-np.sin(theta)*dderiv[i],np.cos(theta)*dderiv[i]],[-np.cos(theta)*dderiv[i],-np.sin(theta)*dderiv[i]]] for i in range(len(qcoo))])
    #print("ddat2="+str(ddat2))

    t_data = Surface(nsta, ham.nmode_active, 1)
    t_data.geom      = qcoo
    t_data.potential = adiabpot
    t_data.deriv = [np.diag(adiabderiv1[m]) for m in
                    range(ham.nmode_active)] + nactmat

    t_data.scalar_coup   = 0.5*sctmat #account for the 1/2 prefactor in the EOMs
    t_data.adt_mat       = adtmat
    t_data.dat_mat       = sp_linalg.inv(adtmat)
    #t_data.ddat_mat      = ddat2
    t_data.diabat_pot    = diabpot
    t_data.diabat_deriv  = diabderiv1
    t_data.adiabat_pot   = adiabpot
    t_data.adiabat_deriv = adiabderiv1
    t_data.data_keys     = ['geom','poten','deriv',
                            'scalar_coup','adt_mat','dat_mat','ddat_mat',
                            'diabat_pot','diabat_deriv',
                            'adiabat_pot','adiabat_deriv']

    data_cache[tid] = t_data
    return t_data


def evaluate_centroid(tid, geom, stateindices):
    """Evaluates the centroid.

    At the moment, this function is just evaluate_trajectory.
    """
    return evaluate_trajectory(tid, geom, stateindices[0])


#----------------------------------------------------------------------
#
# Private functions (called only within the module)
#
#----------------------------------------------------------------------
def conv(val_list):
    """Takes a list and converts it into atomic units.

    The input list can either be a single float or a float followed by a
    comma and the units to convert from."""
    if len(val_list) == 1:
        return float(val_list[0])
    elif len(val_list) > 2 and val_list[1] == ',':
        if val_list[2] == 'au':
            return float(val_list[0])
        elif val_list[2] == 'ev':
            return float(val_list[0]) / glbl.au2ev
        elif val_list[2] == 'cm':
            return float(val_list[0]) / glbl.au2cm
    else:
        raise ValueError('Unknown parameter format:', val_list)


def get_kwds(infile):
    """Reads a file and returns keywords.

    By default, everything is converted to lowercase."""
    delim = ['=', ',', '(', ')', '[', ']', '{', '}', '|', '*', '/', '^']
    rawtxt = infile.readlines()
    kwds = [[] for i in range(len(rawtxt))]

    for i, line in enumerate(rawtxt):
        if '#' in line:
            line = line[:line.find('#')]
        for d in delim:
            line = line.replace(d, ' ' + d + ' ')
        kwds[i] = line.lower().split()

    while [] in kwds:
        kwds.remove([])
    return kwds


def getcoe(val_list):
    """Gets a coefficient from a list of factors"""
    if len(val_list) % 2 != 1:
        raise ValueError('Coefficient specification must have odd number of '
                         'terms including arithmetic operators.')
    val_list.insert(0, '*')
    coeff = 1
    for j in range(len(val_list) // 2):
        try:
            fac = float(val_list[2*j+1])
        except ValueError:
            fac = ham.parmap[val_list[2*j+1]]
        if val_list[2*j] == '*':
            coeff *= fac
        elif val_list[2*j] == '/':
            coeff /= fac
        else:
            raise ValueError('Aritmetic operator must be \'*\' or \'/\'')
    return coeff


def calc_diabpot(q):
    """Constructs the diabatic potential matrix for a given nuclear
    geometry q.

    N.B. the terms held in the coe, ord and stalbl contribute only
    to the lower-triangle of the diabatic potential matrix.
    """
    diabpot = np.zeros((nsta, nsta))

    # Fill in the lower-triangle
    for i in range(ham.nterms):
        s1, s2 = ham.stalbl[i] - 1
        diabpot[s1,s2] += ham.coe[i] * np.prod(q**ham.order[i])

    # Fill in the upper-triangle
    diabpot += diabpot.T - np.diag(diabpot.diagonal())
    return diabpot


def calc_adt(tid, diabpot):
    """Diagonalises the diabatic potential matrix to yield the adiabatic
    potentials and the adiabatic-to-diabatic transformation matrix."""
    adiabpot, adtmat = sp_linalg.eigh(diabpot)

    if tid in data_cache:
        # Ensure phase continuity from geometry to another
        adtmat *= np.sign(np.dot(adtmat.T, data_cache[tid].adt_mat).diagonal())
    else:
        # Set phase convention that the greatest abs element in adt column
        # vector is positive
        adtmat *= np.sign(adtmat[range(len(adiabpot)),
                                 np.argmax(np.abs(adtmat), axis=0)])
    return adiabpot, adtmat


def calc_diabderiv1(q):
    """Calculates the 1st derivatives of the elements of the diabatic
    potential matrix wrt the nuclear DOFs."""
    diabderiv1 = np.zeros((ham.nmode_active, nsta, nsta))

    # This is actually significantly slower than the version with loops!
    # It may only be efficient for large numbers of active modes.
    # Fill in the lower-triangle
    #if not np.any(abs(q) < glbl.fpzero):
    #    n = ham.nmode_active
    #    qcol = np.repeat(q, n).reshape(n, n)
    #    for i in range(ham.nterms):
    #        s1, s2 = ham.stalbl[i] - 1
    #        o = ham.order[i]
    #        fac = np.prod((qcol - np.diag(q) + np.diag(o)) *
    #                      (q ** (o - 1))[:,np.newaxis], axis=0)
    #        diabderiv1[:,s1,s2] += ham.coe[i] * fac

    # Fill in the lower-triangle
    for i in range(ham.nterms):
        s1, s2 = ham.stalbl[i] - 1
        for m in range(ham.nmode_active):
            fac = 0.
            if ham.order[i,m] != 0:
                fac = ham.coe[i]
                for n in range(ham.nmode_active):
                    p = ham.order[i,n]
                    if n == m:
                        fac *= p * q[n]**(p-1)
                    else:
                        fac *= q[n]**p
            diabderiv1[m,s1,s2] += fac

    # Fill in the upper-triangle
    diabderiv1 += (np.transpose(diabderiv1, axes=(0, 2, 1)) -
                   [np.diag(diabderiv1[m].diagonal()) for m in
                    range(ham.nmode_active)])
    return diabderiv1


def calc_nacts(adiabpot, adtmat, diabderiv1):
    """Calculates the matrix of non-adiabatic coupling terms from the
    adiabatic potentials, the ADT matrix and the nuclear derivatives of
    the diabatic potentials.

    Equation used: F_aij = [S (d/dX_a W) S^T]_ij / (V_j - V_i)
    F_aij = < psi_i | d/dX_a psi_j >: non-adiabatic coupling terms
    W: diabatic potential matrix
    S^T: matrix of eigenvectors of W
    V: vector of adiabatic energies
    """
    # Fill in the matrix
    nactmat = np.zeros((ham.nmode_active, nsta, nsta))
    fac = -np.subtract.outer(adiabpot, adiabpot) + np.eye(nsta)
    for m in range(ham.nmode_active):
        nactmat[m] = np.dot(np.dot(adtmat.T, diabderiv1[m]), adtmat) / fac

    # Subtract the diagonal to make sure it is zero
    nactmat -= [np.diag(nactmat[m].diagonal()) for m in range(ham.nmode_active)]
    return nactmat


def calc_adiabderiv1(adtmat, diabderiv1):
    """Calculates the gradients of the adiabatic potentials.

    Equation used: d/dX V_ii = (S{d/dX W}S^T)_ii"""
    # Get the diagonal elements of the matrix
    adiabderiv1 = np.zeros((ham.nmode_active, nsta))
    for m in range(ham.nmode_active):
        adiabderiv1[m] = np.dot(np.dot(adtmat.T, diabderiv1[m]),
                                adtmat).diagonal()
    return adiabderiv1


def calc_diablap(q):
    """Calculates the Laplacian of the diabatic potential matrix wrt
    the nuclear DOFs at the point q."""
    diablap = np.zeros((nsta, nsta))

    # This is actually significantly slower than the version with loops!
    # It may only be efficient for large numbers of active modes.
    # Fill in the lower-triangle
    #if not np.any(abs(q) < glbl.fpzero):
    #    n = ham.nmode_active
    #    qcol = np.repeat(q, n).reshape(n, n)
    #    for i in range(ham.nterms):
    #        s1, s2 = ham.stalbl[i] - 1
    #        o = ham.order[i]
    #        fac = np.prod((qcol**2 - np.diag(q**2) + np.diag(o * (o - 1))) *
    #                      (q ** (o - 2))[:,np.newaxis], axis=0)
    #        diablap[s1,s2] += np.sum(ham.coe[i] * fac)

    # Fill in the lower-triangle
    for i in range(ham.nterms):
        s1, s2 = ham.stalbl[i] - 1
        for m in range(ham.nmode_active):
            fac = 0.
            if ham.order[i,m] > 1:
                fac = ham.coe[i]
                for n in range(ham.nmode_active):
                    p = ham.order[i,n]
                    if n == m:
                        fac *= p * (p-1) * q[n]**(p-2)
                    else:
                        fac *= q[n]**p
                diablap[s1,s2] += fac

    # Fill in the upper-triangle
    diablap += diablap.T - np.diag(diablap.diagonal())
    return diablap


def calc_scts(adiabpot, adtmat, diabderiv1, nactmat, adiabderiv1, diablap):
    """Calculates the scalar coupling terms.

    Uses the equation:
    G = (d/d_X F) - F.F

    d/d_X F_ij = d/dX (xi_ij*T_ij), where, xi_ij = (V_j-V_i)^-1
                                           T = S {d/dX W} S^T

    Additionally, the gradients of the on-diagonal SCTs (DBOCs)
    are calculated

    In the following:

    ximat      <-> xi
    tmat       <-> T
    deltmat    <-> d/dX T
    delximat   <-> d/dX xi
    delnactmat <-> d/dX F
    fdotf      <-> F.F
    """

    #-------------------------------------------------------------------
    # (1) Construct d/dX F (delnactmat)
    #-------------------------------------------------------------------

    # (a) deltmat = S {Del^2 W} S^T + -FS{d/dX W}S^T + S{d/dX W}S^TF

    # tmp1 <-> S {Del^2 W} S^T
    tmp1 = np.dot(np.dot(adtmat.T, diablap), adtmat)

    # tmp2 <-> -F S{d/dX W}S^T
    tmp2 = np.zeros((nsta, nsta))
    for i in range(nsta):
        for j in range(nsta):
            for k in range(nsta):
                for l in range(nsta):
                    for m in range(nsta):
                        dp = np.sum(ham.freq * nactmat[:,i,k] *
                                    diabderiv1[:,l,m], axis=0)
                        tmp2[i,j] -= adtmat[l,k] * adtmat[m,j] * dp

    # tmp3 <-> S{d/dX W}S^T F
    tmp3 = np.zeros((nsta, nsta))
    for i in range(nsta):
        for j in range(nsta):
            for k in range(nsta):
                for l in range(nsta):
                    for m in range(nsta):
                        dp = np.sum(ham.freq * nactmat[:,m,j] *
                                    diabderiv1[:,k,l], axis=0)
                        tmp3[i,j] += adtmat[k,i] * adtmat[l,m] * dp

    # deltmat
    deltmat = tmp1 + tmp2 + tmp3

    # (b) delximat
    delximat = np.zeros((ham.nmode_active, nsta, nsta))
    for m in range(ham.nmode_active):
        delximat[m] = -(np.subtract.outer(adiabderiv1[m], adiabderiv1[m]) /
                        (np.subtract.outer(adiabpot, adiabpot) ** 2 -
                         np.eye(nsta)))

    # (c) tmat
    tmat = np.zeros((ham.nmode_active, nsta, nsta))
    for m in range(ham.nmode_active):
        tmat[m] = np.dot(np.dot(adtmat.T, diabderiv1[m]), adtmat)

    # (d) ximat
    ximat = 1. / (-np.subtract.outer(adiabpot, adiabpot) +
                  np.eye(nsta)) - np.eye(nsta)

    # (f) delnactmat_ij = delximat_ij*tmat_ij + ximat_ij*deltmat_ij (i.ne.j)
    delnactmat = np.sum(delximat[:,i,j] * tmat[:,i,j], axis=0) + ximat * deltmat

    #-------------------------------------------------------------------
    # (2) Construct F.F (fdotf)
    #-------------------------------------------------------------------
    fdotf = np.zeros((nsta, nsta))
    for i in range(nsta):
        for j in range(nsta):
            for k in range(nsta):
                for m in range(ham.nmode_active):
                    fdotf[i,j]+=ham.freq[m]*nactmat[m,i,k]*nactmat[m,k,j]

    #-------------------------------------------------------------------
    # (3) Calculate the scalar coupling terms G = (d/dX F) - F.F
    #-------------------------------------------------------------------
    sctmat = delnactmat - fdotf

    #-------------------------------------------------------------------
    # (4) Calculation of the 1st derivatives of the DBOCs
    #
    # CHECK THIS CAREFULLY!
    #-------------------------------------------------------------------
    dbocderiv1 = np.zeros((ham.nmode_active, nsta))
    for i in range(nsta):
        for m in range(ham.nmode_active):
            for k in range(nsta):
                dbocderiv1[m,i] -= 2. * delnactmat[i,k] * nactmat[m,i,k]

    return sctmat, dbocderiv1
