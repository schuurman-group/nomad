"""
Script for testing and evaluating exact integration of Gaussian basis
functions on LVC model two-state surfaces.

LVC Hamiltonian:

    H = 0.5*[ (Sigma - D'*m*D)*I - Delta*sigma_z + Gamma*sigma_x ]

    D = (d/dq_1, d/dq_2, ...)
    Sigma = q' * ss * q + q' * s + es
    Delta = q' * d + ed
    Gamma = q' * g + eg

Gaussian basis functions:

    g_k = exp(-q' * f * q + q' * b_k + c_k)

Common Parameters
-----------------
m : (N, N) array_like
    The scaling matrix of kinetic energy, related to the frequencies.
ss : (N, N) array_like
    The scaling factors for I elements of H that depend on q**2.
s : (N,) array_like
    The scaling factors for I elements of H that depend on q.
es : float
    The scalar portion for I elements of H.
d : (N,) array_like
    The scaling factors for sigma_z elements of H that depend on q.
ed : float
    The scalar portion for sigma_z elements of H.
g : (N,) array_like
    The scaling factors for sigma_x elements of H that depend on q.
eg : float
    The scalar portion for sigma_x elements of H.
f : (N, N) array_like
    The scaling matrix of q**2 elements of Gaussians, related to the
    widths (assumed independent of basis function).
bk : (N,) array_like
    The scaling factors of elements of Gaussian basis function k
    proportional to q.
bl : (N,) array_like
    The scaling factors of elements of Gaussian basis function l
    proportional to q.
ck : (N,) array_like
    The scalar portion of Gaussian basis function k.
cl : (N,) array_like
    The scalar portion of Gaussian basis function l.
aa : (N, N) array_like
    Polynomial expansion term proportional to q**2.
a : (N,) array_like
    Polynomial expansion term proportional to q.
ea : float
    Scalar polynomial expansion term.
"""
import numpy as np
import scipy.linalg as spl

# Determines the Hamiltonian symmetry
hermitian = True

# Returns functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'


def elec_overlap(t1, t2):
    """ Returns < Psi | Psi' >, the electronic overlap integral of two trajectories"""
    return float(t1.state == t2.state)

def nuc_overlap(t1, t2):
    """Calculates the Gaussian overlap.

    Returns
    -------
    float
        The Gaussian basis function overlap.
    """
    bkl = bk.conjugate() + bl
    ckl = ck.conjugate() + cl
    oarg = np.dot(bkl, spl.solve(f, bkl))/8 + ckl
    return np.exp(oarg) / np.sqrt(spl.det(2*f/np.pi))


def traj_overlap(t1, t2):
    """ Returns < Chi | Chi' >, the total overlap integral of two trajectories"""
    return elec_overlap(t1,t2) * nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                                 t2.phase(),t2.widths(),t2.x(),t2.p())

def s_integral(t1, t2, nuc_ovrlp, elec_ovrlp):
    """ Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""

    return nuc_ovrlp * elec_ovrlp


def ci_coord_shift(ss, s, d, g, ed, eg):
    """Find the coordinate shift to move the CI to the origin.

    Returns
    -------
    (N,) ndarray
        The position of the CI.
    """
    n = len(ss)
    T_ci = np.zeros((n + 2, n + 2))
    T_ci[:n,:n] = 2*ss
    T_ci[:n,-2] = d
    T_ci[:n,-1] = g
    T_ci[-2,:n] = d.conjugate()
    T_ci[-1,:n] = g.conjugate()
    return -spl.solve(T_ci, np.hstack((s, ed, eg)))[:n]


def shift_poly(q_shift, mat, vec, sca):
    """Shifts position-dependent parameters to new values.

    Parameters
    ----------
    q_shift : (N,) array_like
        The shift in position.
    mat : (N, N) array_like
        The matrix of 2nd order expansion coefficients.
    vec : (N,) array_like
        The vector of 1st order expansion coefficients.
    sca : float
        The scalar expansion coefficient.

    Returns
    -------
    (N,) ndarray
        The shifted vector values, vec + 2*q_shift*mat.
    float
        The shifted scalar value, sca + q_shift*mat*q_shift + q_shift*vec.
    """
    new_vec = vec + 2*np.dot(mat, q_shift)
    new_sca = sca + np.dot(q_shift, np.dot(mat, q_shift)) + np.dot(q_shift, vec)
    return new_vec, new_sca


def shift_lvc_params(q_ci, ss, s, es, aa, a, ea):
    """Shifts a set of LVC parameters such that the CI is at the origin.

    These values are constant and only need to be determined once.

    Parameters
    ----------
    q_ci : (N,) array_like
        The position of the CI to be shifted to the origin.

    Returns
    -------
    new_s : (N,) ndarray
        The shifted value of s.
    new_es : float
        The shifted value of es.
    new_a : (N,) ndarray
        The shifted value of a.
    new_ea : float
        The shifted value of ea.
    """
    new_s, new_es = shift_poly(q_ci, ss, s, es)
    new_a, new_ea = shift_poly(q_ci, aa, a, ea)
    return new_s, new_es, new_a, new_ea


def shift_gauss_params(q_ci, f, bk, bl, ck, cl):
    """Shifts a set of Gaussian parameters such that the CI is at the origin.

    These values need to be updated throughout a simulation.

    Parameters
    ----------
    q_ci : (N,) array_like
        The position of the CI to be shifted to the origin.

    Returns
    -------
    new_bk : (N,) ndarray
        The shifted value of bk.
    new_bl : (N,) ndarray
        The shifted value of bl.
    new_ck : float
        The shifted value of ck.
    new_cl : float
        The shifted value of cl.
    """
    new_bk, new_ck = shift_poly(q_ci, -f, bk, ck)
    new_bl, new_cl = shift_poly(q_ci, -f, bl, cl)
    return new_bk, new_bl, new_ck, new_cl


def bspace_transform(d, g, f):
    """Calculates the branching space magnitudes as well as scaling
    and rotation matrices.

    Returns
    -------
    (N,) ndarray
        The branching space magnitudes.
    (N, N) ndarray
        The transformation matrix for scaling and rotation.
    """
    finv2sq = spl.sqrtm(spl.inv(f)) / np.sqrt(2)
    gg = np.outer(d, d.conjugate()) + np.outer(g, g.conjugate())
    gp, u = spl.eigh(np.dot(finv2sq, np.dot(gg, finv2sq)))
    du = spl.det(u)
    return gp[[-1,-2]], du*np.dot(u[:,::-1].T.conjugate(), finv2sq)


def exact_poly(f, bk, bl, aa=None, a=None, ea=0):
    """Evaluates analytic integrals of 2nd order polynomials.

    Returns
    -------
    float
        The integration result divided by the overlap.
    """
    bkl = bk.conjugate() + bl
    bf = spl.solve(f, bkl)
    poly = ea
    if aa is not None:
        poly += np.trace(spl.solve(f, aa)) / 4
        poly += np.dot(bf, np.dot(aa, bf)) / 16
    if a is not None:
        poly += np.dot(a, bf)

    return poly


def exact_nac(dp, gp, bp, ngrdi=10000, max_iter=12, thresh=1e-3):
    """Evaluates integrals for the nonadiabatic coupling matrix elements.

    Inputs are shifted to the CI and tranformed into branching space
    coordinates.

    Parameters
    ----------
    dp : (2,) array_like
        The gradient terms of the integral transformed into the branching
        space.
    gp : (2,) array_like
        Branching space magnitudes.
    bp : (2,) array_like
        Linear terms of the Gaussian basis functions transformed into the
        branching space.
    ngrdi : int, optional
        The initial number of grid points, 1e5 by default.
    max_iter : int, optional
        The maximum number of iterations for the integration, 12 by default.
    thresh : float, optional
        The error threshold for numerical integration, 1e-6 by default.

    Returns
    -------
    float
        The integration result divided by the overlap.

    Raises
    ------
    ValueError
        When the integral isn't converged after the maximum iterations.
    """
    dgp = gp[0] - gp[1]
    dugrd = 1 / (gp[0]*(ngrdi + 1))
    n = np.copy(ngrdi)
    integ = 0
    for i in range(max_iter):
        prev = np.copy(integ)
        ugrd = np.linspace(dugrd, 1/gp[0] - dugrd, n)
        if i == 0:
            # add first and last grid points
            fac = np.dot(dp, bp) - dp[0]*bp[0]*dgp/gp[0]
            fac *= np.sqrt(gp[0] / gp[1])**3
            earg = -(np.dot(bp, gp*bp) - bp[0]**2*dgp)/(4*gp[1])
            integ += (np.dot(dp, bp) + fac*np.exp(earg))/2

        ivdgp = 1 / (1 - ugrd*dgp)
        fac = (dp[0]*bp[0] + dp[1]*bp[1]*ivdgp) * np.sqrt(ivdgp)
        earg = -(gp[0]*bp[0]**2 + gp[1]*bp[1]**2*ivdgp)*ugrd/4
        integ += np.sum(fac*np.exp(earg))
        err = np.abs((integ - 2*prev) / (2*integ + 4*prev))
        if err < thresh:
            break
        elif i == max_iter - 1:
            print('err =', err)
            raise ValueError('Integral not converged in '+str(max_iter)+
                             ' iterations')
        else:
            # add more grid points
            dugrd /= 2
            n = 2**i * (ngrdi + 1)

    return integ*dugrd/2


def exact_coul(gp, bp, aap=None, ap=None, ea=0, ngrdi=10000, max_iter=12,
                     thresh=1e-3):
    """Evaulates integrals for the electronic potential matrix elements.

    Inputs are shifted to the CI and tranformed into branching space
    coordinates.

    Parameters
    ----------
    gp : (2,) array_like
        Branching space magnitudes.
    bp : (N,) array_like
        Linear terms of the Gaussian basis functions transformed into the
        branching space.
    aap : (N, N) array_like, optional
        Quadratic terms of the Coulomb integral transformed into the
        branching space.
    ap : (N,) array_like, optional
        Linear terms of the Coulomb integral transformed into the
        branching space.
    ea : float, optional
        Constant term of the Coulomb integral. Zero by default.
    ngrdi : int, optional
        The initial number of grid points, 1e5 by default.
    max_iter : int, optional
        The maximum number of iterations for the integration, 12 by default.
    thresh : float, optional
        The error threshold for numerical integration, 1e-6 by default.

    Returns
    -------
    float
        The integration result divided by the overlap.

    Raises
    ------
    ValueError
        When the integral isn't converged after the maximum iterations.
    """
    dgp = gp[0] - gp[1]
    bp2 = bp[2:]

    const = np.pi*np.exp(-np.dot(bp[:2], bp[:2])/4)/(2*np.sqrt(gp[1]))
    fac1 = ea
    if aap is not None and len(aap) > 2:
        aap02 = aap[0,2:]
        aap12 = aap[1,2:]
        aap22 = aap[2:,2:]
        fac1 += np.trace(aap22)/2 + np.dot(bp2, np.dot(aap22, bp2))/4
    if ap is not None and len(ap) > 2:
        ap2 = ap[2:]
        fac1 += np.dot(bp2, ap2)/2

    const *= fac1
    bpexp = np.sqrt(gp[0]/gp[1])*np.exp(-np.dot(bp[:2], bp[:2])/4)
    dugrd = 1 / (np.sqrt(gp[0])*(ngrdi + 1))
    n = np.copy(ngrdi)
    integ = 0
    for i in range(max_iter):
        prev = np.copy(integ)
        ugrd = np.linspace(dugrd, 1/np.sqrt(gp[0]) - dugrd, n)
        if i == 0:
            # add first and last grid points
            integ += ea/2
            if aap is not None:
                if len(aap) > 2:
                    integ += np.trace(aap)/4 + np.dot(bp, np.dot(aap, bp))/8
                else:
                    integ += np.trace(aap)/4 + np.dot(bp[:2],
                                                      np.dot(aap, bp[:2]))/8

            if ap is not None:
                integ += np.dot(bp, ap)/4

            integ -= fac1*bpexp/2

        vgrd = ugrd**2
        ivdgp = 1 / (1 - vgrd*dgp)
        vgp0 = 1 - vgrd*gp[0]
        vexp = np.exp(-vgrd*(gp[0]*bp[0]**2 + gp[1]*bp[1]**2*ivdgp)/4)

        fac2 = 0
        if aap is not None:
            fac2 += ((aap[0,0] - aap[0,1]**2/aap[1,1])*bp[0]**2 +
                     aap[1,1]*(ivdgp*bp[1] + aap[0,1]*bp[0]/aap[1,1])**2)*vgp0/2
            fac2 += aap[0,0] + aap[1,1]*ivdgp
            if len(aap) > 2:
                fac2 += (np.dot(bp2, aap02)*bp[0] +
                         np.dot(bp2, aap12)*bp[1]*ivdgp)
        if ap is not None:
            fac2 += bp[0]*ap[0] + bp[1]*ap[1]*ivdgp

        integ += np.sum(fac2*vexp*np.sqrt(vgp0)*np.sqrt(ivdgp)/2 +
                        fac1*(vexp*np.sqrt(ivdgp) - bpexp) / np.sqrt(vgp0))

        err = np.abs(0.5*(integ - 2*prev) / (integ + 2*(prev + const/dugrd)))
        if err < thresh:
            break
        elif i == max_iter - 1:
            print('err =', err)
            raise ValueError('Integral not converged in '+str(max_iter)+
                             ' iterations')
        else:
            # add more grid points
            dugrd /= 2
            n = 2**i * (ngrdi + 1)

    return 2*(integ*dugrd + const)/np.sqrt(np.pi)


def rand_array(*dims, rng=[-1, 1], cplx=False, pdef=False, herm=False):
    """Generates a random array with desired properties.

    Parameters
    ----------
    *dims : list
        The length of each dimension of the array.
    rng : (2,) array_like, optional
        The bounds of the random distribution, [-1, 1) by default.
    cplx : bool, optional
        Whether the array is complex, False by default.
    pdef : bool, optional
        Whether the array is positive definite, False by default. Takes
        precedence over the Hermitian keyword.
    herm : bool, optional
        Whether the array is Hermitian, False by default.

    Returns
    -------
    (*dims) ndarray
        The random array with desired properties.
    """
    sc = rng[1] - rng[0]
    if cplx:
        rand = sc*(np.random.rand(*dims) +
                   1j*np.random.rand(*dims)) - rng[0]*(1 + 1j)
    else:
        rand = sc*np.random.rand(*dims) + rng[0]

    if pdef:
        rand = np.dot(rand, rand.T.conjugate())
    elif herm:
        rand += rand.T.conjugate()

    return rand


def main():
    """The main routine."""
    # number of dimensions
    nd = 3

    if False:
        # generate random inputs
        m = rand_array(nd, nd, pdef=True)
        ss = rand_array(nd, nd, pdef=True)
        s = rand_array(nd)
        es = 2*np.random.random() - 1
        d = rand_array(nd)
        ed = 2*np.random.random() - 1
        g = rand_array(nd)
        eg = 2*np.random.random() - 1
        f = rand_array(nd, nd, pdef=True)
        bk = rand_array(nd, cplx=True)
        bl = rand_array(nd, cplx=True)
        ck = 2*(np.random.random() + 1j*np.random.random()) - 1 - 1j
        cl = 2*(np.random.random() + 1j*np.random.random()) - 1 - 1j
        aa = rand_array(nd, nd, herm=True)
        a = rand_array(nd)
        ea = 2*np.random.random() - 1
    else:
        # Loic's inputs
        m = np.array([[0.313227079248483,0.626228060226705,0.521675237195224],
                      [0.626228060226705,1.637057508181126,1.294722464693297],
                      [0.521675237195224,1.294722464693297,1.076771372836708]])
        ss = np.array([[0.379593161345493,0.724047887537639,0.584660119064714],
                       [0.724047887537639,1.466884617859896,1.240108828116058],
                       [0.584660119064714,1.240108828116058,1.117411708340015]])
        s = np.array([0.847254177007983,-0.306165416522831,-0.109695218533074])
        es = 0.734758184267841
        d = np.array([-0.766035392685062,-0.199614774561284,0.336253585453249])
        ed = 0.801174946284663
        g = np.array([-0.687054786342362,-0.680418567475937,0.528877980528916])
        eg = -0.925779535137176
        f = np.array([[0.642275153855834,0.232598304238554,0.217770570690963],
                      [0.232598304238554,0.481266145485479,0.455077148051131],
                      [0.217770570690963,0.455077148051131,1.373562126936702]])
        bk = np.array([0.531967567093217 - 0.545798143588230j,
                       0.499702297633184 + 0.705241568680322j,
                       -0.275710378099966 + 0.475337046662535j])
        bl = np.array([0.202355905183683 - 0.828209434059533j,
                       0.200800179987297 - 0.157479010834386j,
                       -0.170525854364227 + 0.959370436525927j])
        ck = 0.274054829145811 - 0.839293537640341j
        cl = -0.331328257378850 + 0.307495580598330j
        aa = np.array([[0.00391261154785014,0.62438393496042599,1.18928860852260376],
                       [0.62438393496042599,0.25200219478273189,1.31467970828275882],
                       [1.18928860852260376,1.31467970828275882,1.82133772177546982]])
        a = np.array([-0.572606755793359 + 0.536595887334750j,
                      -0.192331551051216 - 0.337309947273823j,
                      -0.259820546544332 - 0.555054117030144j])
        ea = 0.497105279496541 - 0.968380489744195j


    # shift CI to the origin (time-independent)
    q_ci = ci_coord_shift(ss, s, d, g, ed, eg)
    s, es, a, ea = shift_lvc_params(q_ci, ss, s, es, aa, a, ea)
    bk, bl, ck, cl = shift_gauss_params(q_ci, f, bk, bl, ck, cl)
    gp, trans = bspace_transform(d, g, f)
    bkc = bk.conjugate()

    # get time-dependent values
    olap = overlap(f, bk, bl, ck, cl)
    dd = np.dot(np.outer(d, g.conjugate()) - np.outer(g, d.conjugate()),
                np.dot(m, bl - bk.conjugate()))
    dp = np.dot(trans[:2], dd)
    bp = np.dot(trans, bkc + bl)
    aap = np.dot(trans, np.dot(aa, trans.T.conjugate()))
    ap = np.dot(trans, a)
    print('olap =', olap)
    print('dp =', dp)
    print('gp =', gp)
    print('bp =', bp)
    print('aap =', aap)
    print('ap =', ap)
    print('ea =', ea)
    print('')

    # evaluate integrals and print results
    tnme = olap*(2*np.trace(np.dot(ss, f)) + np.dot(bkc, np.dot(ss, bkc)) +
                 np.dot(bl, np.dot(ss, bl))) # nuclear kinetic energy
    print('tnme =', tnme)
    sgme = olap*exact_poly(f, bk, bl, aa=ss, a=s, ea=es) # average potential energy
    print('sgme =', sgme)
    nacme = olap*exact_nac(dp, gp, bp[:2]) # nonadiabatic coupling
    print('nacme =', nacme)
    aame = olap*exact_coul(gp, bp, aap=aap, ap=ap, ea=ea) # arbitrary Coulomb term
    print('aame =', aame)
    ggme = olap*exact_coul(gp, bp, aap=np.diag(gp)) # adiabatic energy difference
    print('ggme =', ggme)
    s0me = olap*exact_coul(gp, bp, ap=np.dot(trans, d)) # 1st term of diabatic pop
    print('s0me =', s0me)
    s1me = olap*exact_coul(gp, bp, ap=np.dot(trans, g)) # 2nd term of diabatic pop
    print('s1me =', s1me)

    #TODO: add SP and BAT integration


if __name__ == '__main__':
    main()
