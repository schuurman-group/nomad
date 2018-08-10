"""
Routines for running a vibronic coupling calculation.
"""
import numpy as np
import scipy.linalg as sp_linalg
import nomad.math.constants as constants
import nomad.core.glbl as glbl
import nomad.core.log as log
import nomad.core.surface as surface


ham = None
nsta = glbl.properties['n_states']
data_cache = dict()


class VibHam:
    """Object containing the vibronic Hamiltonian parameters."""
    def __init__(self):
        # Parameters
        self.npar   = 0
        self.apar   = None
        self.par    = None
        self.parmap = dict()

        # Hamiltonian terms
        self.nterms       = 0
        self.coe          = None
        self.stalbl       = None
        self.order        = None
        self.mode         = None
        self.freq         = None
        self.mlbl_total   = []
        self.mlbl_active  = []
        self.nmode_total  = 0
        self.nmode_active = 0
        self.mrange       = None

    def rdoperfile(self, fname):
        """Reads and interprets an operator file."""
        with open(fname, 'r') as infile:
            keywords = get_kwds(infile)

        # read the parameter section
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

        # read the Hamiltonian section
        hamstart = keywords.index(['hamiltonian-section'])
        hamend   = keywords.index(['end-hamiltonian-section'])
        i = hamstart + 1
        # read Hamiltonian mode labels
        while i < hamend:
            kwd = keywords[i]
            if kwd[0] != 'modes' and i == hamstart + 1:
                raise ValueError('The Hamiltonian section must start with the '
                                 'mode specification')
            elif kwd[0] != 'modes':
                break
            while '|' in kwd:
                kwd.remove('|')
            self.mlbl_total += kwd[1:]
            self.nmode_total += len(kwd[1:])
            i += 1

        # find mapping from total to active modes
        actmap = [self.mlbl_total.index(j) for j in self.mlbl_active]

        # read Hamiltonian KE values (frequencies)
        self.freq = np.zeros(self.nmode_active)
        istart = i
        while i < hamend:
            kwd = keywords[i]
            if kwd[-1] != 'ke' and i == istart:
                print(kwd)
                raise ValueError('The Hamiltonian section must have KE terms '
                                 'following mode specification')
            elif kwd[-1] != 'ke':
                break
            div = kwd.index('|')
            modei = int(kwd[div+1]) - 1
            if modei in actmap:
                self.freq[actmap.index(modei)] += get_coe(kwd[:div])
            i += 1

        # read Hamiltonian terms
        nterms = hamend - i
        coe    = np.zeros(nterms)
        mode   = [[] for j in range(nterms)]
        order  = [[] for j in range(nterms)]
        stalbl = np.zeros((nterms, 2), dtype=int)
        active = np.zeros(nterms, dtype=bool)
        for i in range(nterms):
            kwd = keywords[hamend - nterms + i]
            if '^' in kwd:
                powind = [j for j in range(len(kwd)) if kwd[j] == '^']
                for j in powind:
                    modei = int(kwd[j-1]) - 1
                    if modei in actmap:
                        mode[i].append(actmap.index(modei))
                        order[i].append(int(kwd[j+1]))
                        active[i] = True
            else:
                powind = [len(kwd)]
                active[i] = True
            coe[i] = get_coe(kwd[:powind[0]-1])
            if '&' not in kwd[-1]:
                raise ValueError('The Hamiltonian term must end with state '
                                 'specification')
            states = sorted(kwd[-1][1:].split('&'))
            stalbl[i] = [int(s) for s in states]

        self.nterms = sum(active)
        self.coe    = coe[active]
        self.stalbl = stalbl[active]
        self.mode   = np.array(mode)[active]
        self.order  = np.array(order)[active]
        self.mrange = [self.mlbl_total.index(lbl) for lbl in self.mlbl_active]


def init_interface():
    """Reads the operator file.

    Note that the order of active modes is determined by the init_coords
    read from the input file.
    """
    global ham

    # Read in operator file
    ham = VibHam()

    ham.mlbl_active  = [lbl.lower() for lbl in glbl.crd_labels]
    ham.nmode_active = len(glbl.crd_labels)

    # operator file will always be a separate file
    ham.rdoperfile(glbl.home_path + '/' + glbl.vibronic['opfile'])

    # (re)set the kinetic energy coefficient
    glbl.kecoef = 0.5 * ham.freq

    # Ouput some information about the Hamiltonian
    log.print_message('string', ['*'*72])
    log.print_message('string',
                             ['* Vibronic Coupling Hamiltonian Information'])
    log.print_message('string', ['*'*72])
    log.print_message('string',
                             ['Operator file: ' + glbl.vibronic['opfile']])
    log.print_message('string',
                             ['Number of Hamiltonian terms: ' + str(ham.nterms)])
    string = 'Total no. modes: ' + str(ham.nmode_total)
    log.print_message('string', [string])

    string = 'No. active modes: ' + str(ham.nmode_active)
    log.print_message('string', [string])

    log.print_message('string', ['Active mode labels:'])
    for i in range(ham.nmode_active):
        string = str(i+1) + ' ' + ham.mlbl_active[i]
        log.print_message('string', [string])

    log.print_message('string', ['Active mode frequencies (a.u.):'])
    for i in range(ham.nmode_active):
        string = str(i+1) + ' ' + str(ham.freq[i])
        log.print_message('string', [string])


def evaluate_trajectory(traj, t=None):
    """Evaluates the trajectory."""
    global nsta, data_cache

    label = traj.label
    geom  = traj.x()

    # Calculation of the diabatic potential matrix
    diabpot = calc_diabpot(geom)

    # Calculation of the nuclear derivatives of the diabatic potential
    diabderiv1 = calc_diabderiv1(geom)

    # Calculatoin of the hessian matrix for each sub-block of the diabatic potential
    # matrix
    diabderiv2 = calc_diabderiv2(geom)

    # Calculation of the Laplacian of the diabatic potential wrt the
    # nuclear DOFs
    diablap = calc_diablap(geom)

    # load the data into the pes object to pass to trajectory
    t_data = surface.Surface()
    t_data.add_data('geom',geom)

    t_data.add_data('diabat_pot',diabpot)
    t_data.add_data('diabat_deriv',diabderiv1)
    t_data.add_data('diabat_hessian',diabderiv2)

    if glbl.surface_rep == 'adiabatic':
        # Calculation of the adiabatic potential vector and ADT matrix
        adiabpot, datmat = calc_dat(label, diabpot)

        # Calculation of the NACT matrix
        nactmat = calc_nacts(adiabpot, datmat, diabderiv1)

        # Calculation of the gradients (for diagonal elements) and derivative
        # couplings (off-diagonal elements)
        adiabderiv1 = calc_adiabderiv1(datmat, diabderiv1)

        # Calculation of the hessians (for diagonal elements) and derivative of
        # derivative couplings (off-diagonal elements)
        adiabderiv2 = calc_adiabderiv2(datmat, diabderiv2)

        # Calculation of the scalar couplings terms (SCTs)
        # Note that in order to calculate the SCTs, we get the gradients of the
        # diagonal Born-Oppenheimer corrections (DBOCs). Consequently, we
        # save these as a matter of course.
        sctmat, dbocderiv1 = calc_scts(adiabpot, datmat, diabderiv1,
                                   nactmat, adiabderiv1, diablap)

        t_data.add_data('potential',adiabpot)
        t_data.add_data('derivative', np.array([np.diag(adiabderiv1[m]) for m in
                                      range(ham.nmode_total)] + nactmat))
        t_data.add_data('hessian',adiabderiv2)

        # non-standard items
        t_data.add_data('nac',nactmat)
        t_data.add_data('scalar_coup',0.5*sctmat) #account for the 1/2 prefactor in the EOMs
        t_data.add_data('dat_mat',datmat)
        t_data.add_data('adt_mat',datmat.T)

    else:
        # determine the effective diabatic coupling: Hij / (Ei - Ej)
        diab_effcoup     = calc_diabeffcoup(diabpot)

        t_data.add_data('potential',np.array([diabpot[i,i] for i in range(nsta)]))
        t_data.add_data('derivative',diabderiv1)
        t_data.add_data('hessian',diabderiv2)

    data_cache[label] = t_data
    return t_data


def evaluate_centroid(traj, t=None):
    """Evaluates the centroid.

    At the moment, this function is just evaluate_trajectory.
    """
    return evaluate_trajectory(traj, t)


def evaluate_coupling(traj):
    """Updates the coupling to the other states"""
    if glbl.surface_rep == 'adiabatic':
        vel   = traj.velocity()
        deriv = traj.pes.get_data('derivative')

        coup = np.array([[np.dot(vel, deriv[:,i,j]) for i in range(nsta)]
                                                    for j in range(nsta)])
        coup -= np.diag(coup.diagonal())
        traj.pes.add_data('coupling',coup)

    else:
        diabpot          = traj.pes.get_data('diabat_pot')
        diab_effcoup     = calc_diabeffcoup(diabpot)
        traj.pes.add_data('coupling',diab_effcoup)


#----------------------------------------------------------------------
#
# Private functions (called only within the module)
#
def conv(val_list):
    """Takes a list and converts it into atomic units.

    The input list can either be a single float or a float followed by a
    comma and the units to convert from. Currently supported units are
    Hartrees (au), electron volts (ev) and wavenumbers (cm).
    """
    if len(val_list) == 1:
        return float(val_list[0])
    elif len(val_list) > 2 and val_list[1] == ',':
        if val_list[2] == 'au':
            return float(val_list[0])
        elif val_list[2] == 'ev':
            return float(val_list[0]) / constants.au2ev
        elif val_list[2] == 'cm':
            return float(val_list[0]) / constants.au2cm
        else:
            raise ValueError('Unknown units:', val_list[2])
    else:
        raise ValueError('Unknown parameter format:', val_list)


def get_kwds(infile):
    """Reads a file and returns keywords.

    By default, everything is converted to lowercase.
    """
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


def get_coe(val_list):
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
        if ham.mode[i] == []:
            diabpot[s1,s2] += ham.coe[i]
        else:
            diabpot[s1,s2] += ham.coe[i] * np.prod(q[ham.mode[i]]**ham.order[i])

    # Fill in the upper-triangle
    diabpot += diabpot.T - np.diag(diabpot.diagonal())
    return diabpot


def calc_dat(label, diabpot):
    """Diagonalises the diabatic potential matrix to yield the adiabatic
    potentials and the adiabatic-to-diabatic transformation matrix."""
    adiabpot, datmat = sp_linalg.eigh(diabpot)

    if label in data_cache:
        # Ensure phase continuity from geometry to another
        datmat *= np.sign(np.dot(datmat.T, data_cache[label].get_data('dat_mat').diagonal()))
    else:
        # Set phase convention that the greatest abs element in dat column
        # vector is positive
        datmat *= np.sign(datmat[range(len(adiabpot)),
                                 np.argmax(np.abs(datmat), axis=0)])
    return adiabpot, datmat


def calc_ddat(label, q, diabpot, dat_mat):
    """Returns the derviative of the diabatic to adiabatic transformation
       matrix via numerical differentiation"""
    ddat_mat = np.zeros((ham.nmode_total, nsta, nsta))
    dx = 0.001

    return ddat_mat


def calc_diabderiv1(q):
    """Calculates the 1st derivatives of the elements of the diabatic
    potential matrix wrt the nuclear DOFs."""
    diabderiv1 = np.zeros((ham.nmode_total, nsta, nsta))

    # Fill in the lower-triangle
    for i in range(ham.nterms):
        nmodes = len(ham.mode[i])
        s1, s2 = ham.stalbl[i] - 1
        if nmodes == 1:
            m = ham.mode[i][0]
            o = ham.order[i][0]
            diabderiv1[m,s1,s2] += ham.coe[i] * o * q[m]**(o - 1)
        elif nmodes > 1:
            m = ham.mode[i]
            o = np.array(ham.order[i])
            qcol = np.repeat(q[m], nmodes).reshape(nmodes, nmodes)
            fac = np.prod((qcol - np.diag(q[m]) + np.diag(o)) *
                          (q[m]**(o - 1))[:,np.newaxis], axis=0)
            diabderiv1[m,s1,s2] += ham.coe[i] * fac

    # Fill in the upper-triangle
    diabderiv1[ham.mrange] += (np.transpose(diabderiv1[ham.mrange], axes=(0,2,1)) -
                               [np.diag(diabderiv1[m].diagonal()) for m in
                                ham.mrange])
    return diabderiv1


def calc_diabderiv2(q):
    """Calculates the 2nd derivatives of the elements of the diabatic
    potential matrix wrt the nuclear DOFs."""
    diabderiv2 = np.zeros((ham.nmode_total, ham.nmode_total, nsta, nsta))

    # Fill in the lower-triangle
    for i in range(ham.nterms):
        nmodes = len(ham.mode[i])
        s1, s2 = ham.stalbl[i] - 1
        m = ham.mode[i]
        o = np.array(ham.order[i])

        # first do diagonal d^2/dx^2 terms
        elem = [ind for ind,val in enumerate(o) if val > 1]
        nelem = len(elem)
        if nelem > 0:
            prim = np.repeat(q[m]**o, nelem).reshape(len(m), nelem)
            derv = o[elem]*(o[elem]-1)*(q[elem]**(o[elem]-2))
            for j in range(nelem):
                prim[j,elem[j]] = derv[j]
                trm = ham.coe[i] * np.prod(prim[j])
                diabderiv2[m[elem[j]],m[elem[j]],s1,s2] += trm
                if s1 != s2:
                    diabderiv2[m[elem[j]],m[elem[j]],s2,s1] += trm

        #now do bilinear terms
        elem = [ind for ind,val in enumerate(o) if val > 0]
        nelem = len(elem)
        if nelem > 1:
            nterm = nelem * (nelem-1) / 2
            prim = np.repeat(q[m]**o, nelem).reshape(len(m), nterm)
            derv = o[elem]*q[elem]**(o[elem]-1)
            icnt = 0
            for j in range(nelem):
                for k in range(i):
                    prim[icnt,elem[j]] = derv[j]
                    prim[icnt,elem[k]] = derv[k]
                    trm = ham.coe[i] * np.prod(prim[icnt])
                    diabderiv2[m[elem[j]],m[elem[k]],s1,s2] += trm
                    if elem[j] != elem[k]:
                        diabderiv2[m[elem[k]],m[elem[j]],s1,s2] += trm
                    if s1 != s2:
                        diabderiv2[m[elem[j]],m[elem[k]],s2,s1] += trm
                        if elem[j] != elem[k]:
                            diabderiv2[m[elem[k]],m[elem[j]],s2,s1] += trm
                    icnt+=1

    return diabderiv2


def calc_diabeffcoup(diabpot):
    """Calculates the effective diabatic coupling between diabatic states i, j via
       eff_coup = Hij / (H[i,i] - H[j,j])
    """
    demat = np.array([[max([constants.fpzero,diabpot[i,i]-diabpot[j,j]],key=abs)
                       for i in range(nsta)] for j in range(nsta)])
    eff_coup = np.divide(diabpot - np.diag(diabpot.diagonal()), demat)

    return eff_coup


def calc_nacts(adiabpot, datmat, diabderiv1):
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
    nactmat = np.zeros((ham.nmode_total, nsta, nsta))
    fac = -np.subtract.outer(adiabpot, adiabpot) + np.eye(nsta)
    for m in ham.mrange:
        nactmat[m] = np.dot(np.dot(datmat.T, diabderiv1[m]), datmat) / fac

    # Subtract the diagonal to make sure it is zero
    nactmat -= [np.diag(nactmat[m].diagonal()) for m in range(ham.nmode_total)]
    return nactmat


def calc_adiabderiv1(datmat, diabderiv1):
    """Calculates the gradients of the adiabatic potentials.

    Equation used: d/dX V_ii = (S{d/dX W}S^T)_ii
    """
    # Get the diagonal elements of the matrix
    adiabderiv1 = np.zeros((ham.nmode_total, nsta))
    for m in ham.mrange:
        adiabderiv1[m] = np.dot(np.dot(datmat.T, diabderiv1[m]),
                                datmat).diagonal()
    return adiabderiv1


def calc_adiabderiv2(datmat, diabderiv2):
    """Calculates the hessians of the adiabatic potentials.

    Equation used: d^2/dXidXj V_ii = (S{d^2/dXidXj W}S^T)_ii
    """
    # Get the diagonal elements of the matrix
    adiabderiv2 = np.zeros((ham.nmode_total, ham.nmode_total, nsta))

    for m in ham.mrange:
        for n in range(m+1):
            adiabderiv2[m,n] = np.dot(np.dot(datmat.T, diabderiv2[m,n]),
                                             datmat).diagonal()
            if m != n:
                adiabderiv2[n,m] = adiabderiv2[m,n]

    return adiabderiv2


def calc_diablap(q):
    """Calculates the Laplacian of the diabatic potential matrix wrt
    the nuclear DOFs at the point q."""
    diablap = np.zeros((nsta, nsta))

    # Fill in the lower-triangle
    for i in range(ham.nterms):
        nmodes = len(ham.mode[i])
        s1, s2 = ham.stalbl[i] - 1
        if nmodes == 1:
            m = ham.mode[i][0]
            o = ham.order[i][0]
            if o > 1:
                diablap[s1,s2] += ham.coe[i] * o * (o - 1) * q[m]**(o - 2)
        elif nmodes > 1:
            m = ham.mode[i]
            o = np.array(ham.order[i])
            q2col = np.repeat(q[m]**2, nmodes).reshape(nmodes, nmodes)
            fac = np.prod((q2col - np.diag(q[m]**2) + np.diag(o * (o - 1))) *
                          (q[m]**(o - 2))[:,np.newaxis], axis=0)
            diablap[s1,s2] += ham.coe[i] * np.sum(fac)

    # Fill in the upper-triangle
    diablap += diablap.T - np.diag(diablap.diagonal())
    return diablap


def calc_scts(adiabpot, datmat, diabderiv1, nactmat, adiabderiv1, diablap):
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
    tmp1 = np.dot(np.dot(datmat.T, diablap), datmat)

    # tmp2 <-> -F S{d/dX W}S^T
    mat2 = [np.dot(np.dot(np.dot(nactmat[m], datmat.T), diabderiv1[m]), datmat)
            for m in ham.mrange]
    tmp2 = -np.sum(ham.freq[:,np.newaxis,np.newaxis] * mat2, axis=0)

    # tmp3 <-> S{d/dX W}S^T F
    mat3 = [np.dot(np.dot(np.dot(datmat.T, diabderiv1[m]), datmat), nactmat[m])
            for m in ham.mrange]
    tmp3 = np.sum(ham.freq[:,np.newaxis,np.newaxis] * mat3, axis=0)

    # deltmat
    deltmat = tmp1 + tmp2 + tmp3

    # (b) delximat
    ## This is slightly slower for nsta == 2
    #delximat = np.zeros((ham.nmode_total, nsta, nsta))
    #for m in ham.mrange:
    #    delximat[m] = -(np.subtract.outer(adiabderiv1[m], adiabderiv1[m]) /
    #                    (np.subtract.outer(adiabpot, adiabpot) ** 2 +
    #                     np.eye(nsta)))
    delximat = np.zeros((ham.nmode_total, nsta, nsta))
    for m in ham.mrange:
        for i in range(nsta):
            for j in range(nsta):
                if i != j:
                    delximat[m,i,j] = ((adiabderiv1[m,i] - adiabderiv1[m,j]) /
                                       (adiabpot[j] - adiabpot[i])**2)

    # (c) tmat
    tmat = np.zeros((ham.nmode_total, nsta, nsta))
    for m in ham.mrange:
        tmat[m] = np.dot(np.dot(datmat.T, diabderiv1[m]), datmat)

    # (d) ximat
    ## This is slightly slower for nsta == 2
    #ximat = 1. / (-np.subtract.outer(adiabpot, adiabpot) +
    #              np.eye(nsta)) - np.eye(nsta)
    ximat = np.zeros((nsta, nsta))
    for i in range(nsta):
        for j in range(nsta):
            if i != j:
                ximat[i,j] = 1. / (adiabpot[j] - adiabpot[i])

    # (f) delnactmat_ij = delximat_ij*tmat_ij + ximat_ij*deltmat_ij (i.ne.j)
    delnactmat = np.sum(delximat * tmat, axis=0) + ximat * deltmat

    #-------------------------------------------------------------------
    # (2) Construct F.F (fdotf)
    #-------------------------------------------------------------------
    matf = [np.dot(nactmat[m], nactmat[m]) for m in ham.mrange]
    fdotf = np.sum(ham.freq[:,np.newaxis,np.newaxis] * matf, axis=0)

    #-------------------------------------------------------------------
    # (3) Calculate the scalar coupling terms G = (d/dX F) - F.F
    #-------------------------------------------------------------------
    sctmat = delnactmat - fdotf

    #-------------------------------------------------------------------
    # (4) Calculation of the 1st derivatives of the DBOCs
    #-------------------------------------------------------------------
    dbocderiv1 = np.zeros((ham.nmode_total, nsta))
    for m in ham.mrange:
        dbocderiv1[m] = -2.*np.sum(delnactmat * nactmat[m], axis=0)

    return sctmat, dbocderiv1
