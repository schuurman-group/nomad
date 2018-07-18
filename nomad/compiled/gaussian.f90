!
!
! Evaluates integrals of operators over gaussian functions
!
!
module integral_gaus

 integer, parameter      :: dpk = selected_real_kind(14,17)
 complex(dpk), parameter :: I   = (0., 1.) 

  !
  !
  !
  function overlap(g1, a1, x1, p1, g2, a2, x2, p2) result(S)
    implicit none
    real(dpk),intent(in) :: g1, a1, x1, p1, g2, a2, x2, p2 

    real(dpk)    :: dx = x1 - x2
    real(dpk)    :: dp = p1 - p2
    real(dpk)    :: prefactor = sqrt(2. * sqrt(a1*a2) / (a1+a2))
    real(dpk)    :: x_center  = (a1 * x1 + a2 * x2) / (a1 + a2)
    real(dpk)    :: real_part = (a1*a2*dx**2 + 0.25*dp**2) / (a1 + a2)
    real(dpk)    :: imag_part = (p1 * x1 - p2 * x2) - x_center * dp
    complex(dpk) :: S = exp(I * (g2 - g1)) * product(prefactor) * exp(sum(-real_part + I*imag_part))

    return
  end function overlap

  !
  !
  !
  function deldp(S, a1, x1, p1, a2, x2, p2) result(int_delp)
    implicit none
    complex(dpk), intent(in) :: S
    real(dpk), intent(in)    :: a1,x1,p1,a2,x2,p2

    real(dpk) :: dx = x1 - x2
    real(dpk) :: dp = p1 - p2
    complex(dpk) int_delp = S * (dp + 2.*I*a1*dx) / (2.*(a1+a2)) 

    return
  end function deldep




#@timings.timed
def deldx(S, a1, x1, p1, a2, x2, p2):
    """Returns the del/dx matrix element between the nuclear component
    of two trajectories for each componet of 'x' (does not sum over terms)"""

    dx    = x1 - x2
    psum  = a1*p2 + a2*p1
    dxval = (2. * a1 * a2 * dx - 1j * psum) / (a1 + a2)
    return dxval * S


#@timings.timed
def deld2x(S, a1, x1, p1, a2, x2, p2):
    """Returns the del^2/d^2x matrix element between the nuclear component
    of two trajectories for each componet of 'x' (does not sum over terms)"""

    dx     = x1 - x2
    psum   = a1*p2 + a2*p1
    d2xval = -(1j * 4. * a1 * a2 * dx * psum + 2. * a1 * a2 * (a1 + a2) -
               4. * dx**2 * a1**2 * a2**2 + psum**2) / (a1 + a2)**2
    return d2xval * S


def prim_v_integral(N, a1, x1, p1, a2, x2, p2):
    """Returns the matrix element <cmplx_gaus(q,p)| q^N |cmplx_gaus(q,p)>
     -- up to an overlap integral --
    """
    # since range(N) runs up to N-1, add "1" to result of floor
    n_2 = int(np.floor(0.5 * N) + 1)
    a   = a1 + a2
    b   = complex(2.*(a1*x1 + a2*x2),-(p1-p2))

    if np.absolute(b) < constants.fpzero:
        if N % 2 != 0:
            return 0.
        else:
            n_2 -= 1 # get rid of extra "1" for range value
            v_int = (a**(-n_2))/np.math.factorial(n_2)
            return v_int * np.math.factorial(N) / 2.**N

    # generally these should be 1D harmonic oscillators. If
    # multi-dimensional, the final result is a direct product of
    # each dimension
    v_int = complex(0.,0.)
    for i in range(n_2):
        v_int += (a**(i-N) * b**(N-2*i) /
                 (np.math.factorial(i) * np.math.factorial(N-2*i)))

    # refer to appendix for derivation of these relations
    return v_int * np.math.factorial(N) / 2.**N


def ordr1_vec(a1, x1, p1, a2, x2, p2):
    """Docstring missing."""
    a   = np.array(a1 + a2, dtype=float)
    b   = np.array(2.*(a1*x1 + a2*x2) - 1.j*(p1-p2),dtype=complex)
    v_int = b / (2 * a)

    return v_int


def ordr2_vec(a1, x1, p1, a2, x2, p2):
    """Docstring missing."""
    a   = np.array(a1 + a2, dtype=float)
    b   = np.array(2.*(a1*x1 + a2*x2) - 1.j*(p1-p2),dtype=complex)
    v_int = 0.5 * (b**2 / (2 * a**2) + (1/a))

    return v_int


