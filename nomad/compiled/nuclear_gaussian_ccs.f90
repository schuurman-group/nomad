
  !
  !
  !
  subroutine overlap(g1, a1, x1, p1, g2, a2, x2, p2, S)
    implicit none
    double precision, intent(in) :: g1, g2
    double precision, intent(in) :: a1(:),x1(:),p1(:),a2(:),x2(:),p2(:)
    double complex, intent(out)   :: S

    double precision :: dx(size(a1)), dp(size(a1)), prefactor(size(a1))
    double precision :: x_center(size(a1)), real_part(size(a1)), imag_part(size(a1))
    double complex,parameter  :: I = (0., 1.)

    dx = x1 - x2
    dp = p1 - p2
    prefactor = sqrt(2. * sqrt(a1*a2) / (a1+a2))
    x_center  = (a1 * x1 + a2 * x2) / (a1 + a2)
    real_part = (a1*a2*dx**2 + 0.25*dp**2) / (a1 + a2)
    imag_part = 0.5 * (x_center * dp + (a1 * p2 * x2 - a2 * p1 * x1)) 
    S = exp(I * (g2 - g1)) * product(prefactor) * exp(sum(-real_part - I*imag_part))

    return
  end subroutine overlap

  !
  !
  !
  subroutine deldp(S, a1, x1, p1, a2, x2, p2, int_delp)
    implicit none
    double complex, intent(in)   :: S
    double precision, intent(in) :: a1(:),x1(:),p1(:),a2(:),x2(:),p2(:)
    double complex, intent(out)  :: int_delp(size(a1))

    double precision :: dx(size(a1)), dp(size(a1))
    double complex,parameter  :: I = (0., 1.)   

    dx       = x1 - x2
    dp       = p1 - p2
    int_delp = S * (dp + 2.*I*a1*dx) / (2.*(a1+a2))

    return
  end subroutine deldp

  ! 
  !
  ! 
  subroutine deldx(S, a1, x1, p1, a2, x2, p2, int_delx)
    implicit none 
    double complex, intent(in)   :: S
    double precision, intent(in) :: a1(:),x1(:),p1(:),a2(:),x2(:),p2(:)
    double complex, intent(out)  :: int_delx(size(a1))

    double precision :: dx(size(a1)), psum(size(a1))
    double complex,parameter  :: I = (0., 1.)

    dx       = x1 - x2
    psum     = a1*p2 + a2*p1
    int_delx = S * (2. * a1 * a2 * dx - I * psum) / (a1 + a2)

    return
  end subroutine deldx

  !
  ! Returns the del^2/d^2x matrix element between the nuclear component
  ! of two trajectories for each componet of 'x' (does not sum over terms)
  !
  subroutine deld2x(S, a1, x1, p1, a2, x2, p2, int_del2x)
    implicit none
    double complex, intent(in)   :: S
    double precision, intent(in) :: a1(:),x1(:),p1(:),a2(:),x2(:),p2(:)
    double complex,intent(out)   :: int_del2x(size(a1))

    double precision :: dx(size(a1)), psum(size(a1))
    double precision :: real_part(size(a1)), imag_part(size(a1))
    double complex,parameter  :: I = (0., 1.)

    dx        = x1 - x2
    psum      = a1*p2 + a2*p1
    real_part = 4.*dx**2*a1**2*a2**2 - psum**2 - 2.*a1*a2*(a1+a2)
    imag_part = -4.*a1*a2*dx*psum
    int_del2x = S * (real_part + I * imag_part) / (a1+a2)**2

    return
  end subroutine deld2x
