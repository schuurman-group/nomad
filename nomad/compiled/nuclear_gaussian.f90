!
!
! Evaluates integrals over operators of Gaussian basis functions
!
!

  !
  ! Returns the overlap of two Gaussian basis functions
  !
  subroutine overlap(g1, a1, x1, p1, g2, a2, x2, p2, S)
    implicit none
    double precision, intent(in) :: g1, g2
    double precision, intent(in) :: a1(:),x1(:),p1(:),a2(:),x2(:),p2(:)
    double complex, intent(out)  :: S

    double precision :: dx(size(a1)), dp(size(a1)), prefactor(size(a1))
    double precision :: x_center(size(a1)), real_part(size(a1)), imag_part(size(a1))
    double complex,parameter  :: I = (0., 1.)

    dx = x1 - x2
    dp = p1 - p2
    prefactor = sqrt(2. * sqrt(a1*a2) / (a1+a2))
    x_center  = (a1 * x1 + a2 * x2) / (a1 + a2)
    real_part = (a1*a2*dx**2 + 0.25*dp**2) / (a1 + a2)
    imag_part = (p1 * x1 - p2 * x2) - x_center * dp
    S = exp(I * (g2 - g1)) * product(prefactor) * exp(sum(-real_part + I*imag_part))

    return
  end subroutine overlap

  !
  ! Returns the del/dp integral between two Gaussian basis functions
  !
  subroutine deldp(S, a1, x1, p1, a2, x2, p2, delp)
    implicit none
    double complex, intent(in)   :: S
    double precision, intent(in) :: a1(:),x1(:),p1(:),a2(:),x2(:),p2(:)
    double complex, intent(out)  :: delp(size(a1))

    double precision :: dx(size(a1)), dp(size(a1))
    double complex,parameter  :: I = (0., 1.)

    dx   = x1 - x2
    dp   = p1 - p2
    delp = S * (dp + 2.*I*a1*dx) / (2.*(a1+a2)) 

    return
  end subroutine deldp

  !
  ! Returns the del/dx integral between two Gaussian basis functions
  !
  subroutine deldx(S, a1, x1, p1, a2, x2, p2, delx)
    implicit none
    double complex, intent(in)   :: S
    double precision, intent(in) :: a1(:),x1(:),p1(:),a2(:),x2(:),p2(:)
    double complex, intent(out):: delx(size(a1))

    double precision :: dx(size(a1)), psum(size(a1))
    double complex,parameter  :: I = (0., 1.)

    dx   = x1 - x2
    psum = a1*p2 + a2*p1
    delx = S * (2. * a1 * a2 * dx - I * psum) / (a1 + a2)

    return
  end subroutine deldx

  !
  ! Returns the del^2/d^2x integral between two Gaussian basis functions
  !
  subroutine deld2x(S, a1, x1, p1, a2, x2, p2, del2x)
    implicit none
    double complex, intent(in)   :: S
    double precision, intent(in) :: a1(:),x1(:),p1(:),a2(:),x2(:),p2(:)
    double complex, intent(out)  :: del2x(size(a1))

    double precision :: dx(size(a1)), psum(size(a1))
    double complex,parameter  :: I = (0., 1.)

    dx    = x1 - x2
    psum  = a1*p2 + a2*p1
    del2x = S * (-4.*a1*a2*dx*psum*I - 2.*a1*a2*(a1+a2) + 4.*dx**2*a1**2*a2**2 - psum**2) / (a1+a2)**2

    return
  end subroutine deld2x

  !
  ! Returns the matrix element <cmplx_gaus(q,p)| q^N |cmplx_gaus(q,p)>
  !   -- up to an overlap integral --
  !
  subroutine qn_integral(n, a1, x1, p1, a2, x2, p2, int_qn)
    implicit none
    integer, intent(in)          :: n
    double precision, intent(in) :: a1,x1,p1,a2,x2,p2
    double complex, intent(out)  :: int_qn

    integer          :: i, n_2
    double precision :: a
    double complex   :: b
    double complex, parameter    :: Im = (0., 1.)
    double precision, parameter  :: zero = 1.e-16

    n_2    = int(floor(0.5*n))
    a      = a1 + a2
    b      = 2.*(a1*x1 + a2*x2) - Im*(p1-p2)

    int_qn = (0., 0.)

    if (abs(b) < zero) then
      if (mod(n,2) == 0) then
        int_qn = a**(-n_2) * factorial(n) / (factorial(n_2) * 2.**n)
      endif
      return
    endif

    do i = 0,n_2
      int_qn = int_qn + a**(i-n) * b**(n-2*i) / (factorial(i) * factorial(n-2*i))
    enddo
    int_qn = int_qn * factorial(n) / 2.**n

    return

  contains
    recursive function factorial(n) result (fac)
      integer, intent(in)  :: n
      integer              :: fac

      if (n==0) then
         fac = 1
      else
         fac = n * factorial(n-1)
      endif
    end function factorial

  end subroutine qn_integral

  !
  ! Returns the matrix elements <cmplx_gaus(q1i,p1i)| q^N |cmplx_gaus(q2i,p2i)>
  !
  subroutine qn_vector(n, S, a1, x1, p1, a2, x2, p2, ints_qn)
    implicit none
    integer, intent(in)          :: n
    double complex, intent(in)   :: S
    double precision, intent(in) :: a1(:),x1(:),p1(:),a2(:),x2(:),p2(:)
    double complex, intent(out)  :: ints_qn(size(a1))

    integer        :: i
    double complex :: int_qn

    ints_qn = (0., 0.)

    do i = 1, size(ints_qn)
       call qn_integral(n,a1(i),x1(i),p1(i),a2(i),x2(i),p2(i),int_qn)
       ints_qn(i) = S * int_qn
    enddo

    return

  end subroutine qn_vector

