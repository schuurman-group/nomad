module quadrature
  use accuracy
  use math
  use quadpack
  use constants
  use, intrinsic :: ieee_arithmetic
  implicit none

  public quadpack_int
  public gauss_legendre
  public :: QUADPACK_ERROR

  interface
    function quadpack_integrand(x) result(f)
      real(kind(1.d0)), intent(in) :: x
      real(kind(1.d0))             :: f
    end function
  end interface

  character(*), parameter :: QUADPACK_ERROR(6) = [ &
    "maximum number of subdivisions achieved",     &
    "roundoff error detected                ",     &
    "extremely bad integrand behaviour      ",     &
    "algorithm does not converge            ",     &
    "integral is probably divergent         ",     &
    "input is invalid                       "      &
  ]

  !external :: dqags, dqagi

  contains
  
   subroutine gauss_legendre(x0, x1, n, w, x)
     real(drk), intent(in)    :: x0, x1
     integer(ik), intent(in)  :: n
     real(drk), intent(out)   :: w(n), x(n)

     integer(ik)              :: i, j, m
     real(drk)                :: p1, p2, p3, pp
     real(drk)                :: xl, xm, z, z1
     real(drk)                :: eps = 3.d-14

     m = (n+1)/2
     xm = 0.5d0 * (x1 + x0)
     xl = 0.5d0 * (x1 - x0)
     z1 = 2.0d0

     do i = 1,m
       z  = cos( pi * (i - 0.25d0) / (n + 0.5d0) )

       do while( abs(z-z1) .gt. eps )
         p1 = 1.d0
         p2 = 0.d0

         do j = 1,n
           p3 = p2
           p2 = p1
           p1 = ( (2.d0*j - 1.d0) *z * p2 - ( j-1.d0 )*p3) / j
         enddo

         pp = n * (z * p1 - p2) / (z * z -1.d0)
         z1 = z
         z  = z1 - p1/pp

       enddo 

       x(i)      = xm - xl * z
       x(n+1-i)  = xm + xl * z
       w(i)      = 2.d0 * xl / ( (1.d0 -z*z)*pp*pp )
       w(n+1-i)  = w(i)

     enddo

     return
   end subroutine gauss_legendre

  subroutine quadpack_int(f, a, b, atol, rtol, result, aerr, neval, ier, limit, last)
    procedure(quadpack_integrand)       :: f      !! function to integrate
    real(drk),              intent(in)  :: a      !! lower integration limit
    real(drk),              intent(in)  :: b      !! upper integration limit
    real(drk),              intent(in)  :: atol   !! absolute tolerance
    real(drk),              intent(in)  :: rtol   !! relative tolerance
    real(drk),              intent(out) :: result !! output result of integration
    real(drk),    optional, intent(out) :: aerr   !! optional: output estimate of the modulus of the absolute error, which should equal or exceed abs(i-result)
    integer(ik), optional, intent(out)  :: neval  !! optional: output number of integrand evaluations
    integer(ik), optional, intent(out) :: ier     !! optional: output error code
    integer(ik), optional, intent(in)  :: limit   !! optional: maximum number of subintervals (default: 4096)
    integer(ik), optional, intent(out) :: last    !! optional, output number of subintervals used

    ! local variables
    integer(ik)                    :: neval_, ier_, limit_, lenw, last_, inf
    real(drk)                      :: aerr_, bnd
    real(drk), allocatable, save   :: awork(:), bwork(:), rwork(:), ework(:)
    integer(ik), allocatable, save :: iwork(:)
    logical                        :: a_inf, b_inf

    limit_ = 4096
    if (present(limit)) limit_ = limit
    lenw = limit_ * 4

    if(.not.allocated(awork)) then
      allocate (awork(limit_), bwork(limit_), rwork(limit_), ework(limit_))
      allocate (iwork(limit_))
    elseif(size(awork) /= limit_) then
      deallocate(awork, bwork, rwork, ework, iwork)
      allocate (awork(limit_), bwork(limit_), rwork(limit_), ework(limit_))
      allocate (iwork(limit_))
    endif

    a_inf = (ieee_class(a) == IEEE_NEGATIVE_INF)
    b_inf = (ieee_class(b) == IEEE_POSITIVE_INF)

    bnd = 0
    if (a_inf .and. b_inf) then
      inf = 2
    elseif (a_inf) then
      bnd = b
      inf = -1
    elseif (b_inf) then
      bnd = a
      inf = +1
    else
      inf = 0
    end if

    if (inf == 0) then
      call dqagse(f, a, b,     atol, rtol, limit_, result, aerr_, neval_, ier_, awork, bwork, rwork, ework, iwork, last_)
    else
      call dqagie(f, bnd, inf, atol, rtol, limit_, result, aerr_, neval_, ier_, awork, bwork, rwork, ework, iwork, last_)
    end if

    if (present(aerr )) aerr  = aerr_
    if (present(neval)) neval = neval_
    if (present(ier  )) ier   = ier_
    if (present(last )) last  = last_
  end subroutine

end module quadrature



