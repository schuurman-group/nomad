!
! This library was created to peform the matrix exponential, but 
! may end up being a repository for a nubmer of other routines
!
! M. S. Schuurman, Jan 6., 2020
!
!
module math
  use accuracy
  implicit none

  public factorial
  public expm_complex 
  public poly_fit_array

  interface norm1
    module procedure norm1_real
    module procedure norm1_complex
  end interface

  interface solve_getrs
    module procedure solve_getrs_real
    module procedure solve_getrs_complex
  end interface

  interface inverse_gelss
    module procedure inverse_gelss_real
    module procedure inverse_gelss_complex
  end interface

  interface poly_fit
    module procedure poly_fit_real
    module procedure poly_fit_complex
  end interface

  !
  integer(8), parameter :: pade_b(0:13,1:5) = reshape( (/ 120, 30240, 17297280, 17643225600, 64764752532480000, & 
                                                           60, 15120, 8648640,  8821612800,  32382376266240000, &
                                                           12, 3360,  1995840,  2075673600,  7771770303897600 , &
                                                            1, 420,   277200,   302702400,   1187353796428800 , &
                                                            0, 30,    25200,    30270240,    129060195264000  , &
                                                            0, 1,     1512,     2162160,     10559470521600   , &
                                                            0, 0,     56,       110880,      670442572800     , &
                                                            0, 0,     1,        3960,        33522128640      , &
                                                            0, 0,     0,        90,          1323241920       , &
                                                            0, 0,     0,        1,           40840800         , &
                                                            0, 0,     0,        0,           960960           , &
                                                            0, 0,     0,        0,           16380            , &
                                                            0, 0,     0,        0,           182              , &
                                                            0, 0,     0,        0,           1                /), shape(pade_b), order=(/2, 1/))
 
  real(drk), parameter      :: pade_theta(1:5) = (/ 0.01495585217958292,0.2539398330063230, &
                                                 0.9504178996162932, 2.097847961257068, &
                                                 5.371920351148152 /)
  integer(ik), parameter    :: pade_ordr(1:5)  = (/ 3,5,7,9,13 /)

  real(drk), parameter      :: pi   = 3.141592653589793
  real(drk), parameter      :: log2 = 0.693147180559945309417232 

 contains

  !
  !
  recursive function factorial(n) result (fac)
    integer(ik), intent(in)   :: n
    integer(hik)              :: fac

    if (n==0) then
       fac = 1
    else
       fac = n * factorial(n-one_ik)
    endif
  end function factorial

  !
  !
  !
  function norm1_real(matrix) result(norm)
    real(drk), intent(in)            :: matrix(:,:)

    real(drk)                        :: norm
    integer(ik)                      :: i, n1, n2

    n1   = size(matrix, 1)
    n2   = size(matrix, 2)
    norm = 0.
    do i = 1,n2
      norm = max(norm, sum(abs(matrix(:n1,n2))))
    enddo

    return
  end function norm1_real

  !
  !
  !
  function norm1_complex(matrix) result(norm)
    complex(drk), intent(in)         :: matrix(:,:)

    real(drk)                        :: norm
    integer(ik)                      :: i, n1, n2

    n1   = size(matrix, 1)
    n2   = size(matrix, 2)
    norm = 0.
    do i = 1,n2
      norm = max(norm, sum(abs(matrix(:n1,n2))))
    enddo

    return
  end function norm1_complex

  !
  !
  !
  function poly_fit_array(ordr, x0, x_data, y_data) result(y_fit)
    integer(ik), intent(in)         :: ordr
    real(drk), intent(in)           :: x0
    real(drk), intent(in)           :: x_data(:)
    real(drk), intent(in)           :: y_data(:,:) !y_data is nfit x npt array

    real(drk), allocatable          :: y_fit(:)
    real(drk), allocatable          :: beta(:,:)
    real(drk), allocatable          :: X(:,:)
    real(drk)                       :: XtX(0:ordr, 0:ordr)
    real(drk)                       :: XtXinv(0:ordr, 0:ordr)
    integer(ik)                     :: npt
    integer(ik)                     :: nfit
    integer(ik)                     :: i

    npt  = size(x_data)
    nfit = size(y_data, dim=1)

    allocate(X(npt, 0:ordr))
    allocate(beta(0:ordr, nfit))
    allocate(y_fit(nfit))

    do i = 0,ordr
      X(:,i) = (x_data-x0)**i
    enddo

    XtX      = matmul(transpose(X), X)
    XtXinv   = inverse_gelss_real(ordr+1, XtX)
    beta     = matmul(XtXinv, matmul(transpose(X), transpose(y_data)))
    y_fit    = beta(0,:)

    deallocate(X, beta)
    return
  end function poly_fit_array

  !
  !
  !
  function poly_fit_real(ordr, x0, x_data, y_data) result(y_fit)
    integer(ik), intent(in)         :: ordr
    real(drk), intent(in)           :: x0
    real(drk), intent(in)           :: x_data(:)
    real(drk), intent(in)           :: y_data(:)

    real(drk)                       :: y_fit
    real(drk), allocatable          :: y_mat(:,:)
    real(drk), allocatable          :: beta(:,:)
    real(drk), allocatable          :: X(:,:)
    real(drk)                       :: XtX(0:ordr, 0:ordr)
    real(drk)                       :: XtXinv(0:ordr, 0:ordr)
    integer(ik)                     :: npt
    integer(ik)                     :: i

    npt  = size(x_data)

    allocate(X(npt, 0:ordr))
    allocate(beta(0:ordr, 1))
    allocate(y_mat(1, npt))

    y_mat(1, :npt) = y_data(:npt)
    do i = 0,ordr
      X(:,i) = (x_data-x0)**i
    enddo
    XtX      = matmul(transpose(X), X)
    XtXinv   = inverse_gelss_real(size(XtX, dim=1), XtX)
    beta     = matmul(XtXinv, matmul(transpose(X), transpose(y_mat)))
    y_fit    = beta(0,1)

    deallocate(X, y_mat, beta)
    return
  end function poly_fit_real

  !
  !
  !
  function poly_fit_complex(ordr, x0, x_data, y_data) result(y_fit)
    integer(ik), intent(in)         :: ordr
    real(drk), intent(in)           :: x0
    real(drk), intent(in)           :: x_data(:)
    complex(drk), intent(in)        :: y_data(:)

    complex(drk)                    :: y_fit
    complex(drk), allocatable       :: y_mat(:,:)
    complex(drk), allocatable       :: beta(:,:)
    complex(drk), allocatable       :: X(:,:)
    complex(drk)                    :: XtX(0:ordr, 0:ordr)
    complex(drk)                    :: XtXinv(0:ordr, 0:ordr)
    integer(ik)                     :: npt
    integer(ik)                     :: i

    npt = size(x_data)

    allocate(X(npt, 0:ordr))
    allocate(beta(0:ordr, 1))
    allocate(y_mat(1, npt))

    y_mat(1, :npt) = y_data(:npt)
    do i = 0,ordr
        X(:,i) = one_c*(x_data-x0)**i
    enddo
    XtX      = matmul(transpose(X), X)
    XtXinv   = inverse_gelss_complex(size(XtX,dim=1), XtX)
    beta     = matmul(XtXinv, matmul(transpose(X), transpose(y_mat)))
    y_fit    = beta(0,1)

    deallocate(X, beta, y_mat)
    return
  end function poly_fit_complex

  !
  ! This uses the 
  !
  subroutine expm_complex(n, A, e_A)
    integer(ik), intent(in)          :: n
    complex(drk), intent(in)         :: A(n,n)
    complex(drk), intent(inout)      :: e_A(n,n)

    integer(ik)                      :: s
    integer(ik)                      :: i,j
    complex(drk),allocatable         :: U(:,:), V(:,:)
    complex(drk),allocatable         :: An(:,:,:)
    complex(drk),allocatable         :: Id(:,:)
    complex(drk)                     :: traceA
    real(drk)                        :: znormA, residual

    allocate(U(n,n), V(n,n))
    allocate(Id(n,n))
    allocate(An(n,n,4)) !corresponds to A, A^2, A^4, A^6

    ! this is not really necessary, but makes things clearer, at
    ! least to start
    Id = zero_c    
    do i = 1,n
      Id(i,i) = one_c
    enddo 
 
    ! pre-process to reduce the norm 
    traceA    = sum( (/ (A(i,i), i=1,n) /) )
    An(:,:,1) = A - (traceA/n)*Id
    znormA    = norm1_complex(A)
    s         = 0

    ! check if lower-order pade approximants have
    ! sufficient accuracy
    do i = 1,size(pade_ordr)
 
      ! I'm sure there's a more elegant way to do this, and
      ! I'll probably implement it some day...

      ! form requisite U/V matrices in case we need to form
      ! corresponding Pade approximant 
      select case(pade_ordr(i))
        case(3)
          An(:,:,2) = matmul(An(:,:,1),An(:,:,1)) ! A^2
          U = pade_b(3,1)*matmul(An(:,:,1),An(:,:,2)) + pade_b(1,1)*An(:,:,1)
          V = pade_b(2,1)*An(:,:,2)                   + pade_b(0,1)*Id
        case(5)
          An(:,:,3) = matmul(An(:,:,2),An(:,:,2)) ! A^4
          U = pade_b(5,2)*matmul(An(:,:,1),An(:,:,3)) + pade_b(3,2)*matmul(An(:,:,1),An(:,:,2)) + pade_b(1,2)*An(:,:,1)
          V = pade_b(4,2)*An(:,:,3)                   + pade_b(2,2)*An(:,:,2)                   + pade_b(0,2)*Id 
        case(7)
          An(:,:,4) = matmul(An(:,:,2),An(:,:,3)) ! A^6
          U = pade_b(7,3)*matmul(An(:,:,1),An(:,:,4)) + pade_b(5,3)*matmul(An(:,:,1),An(:,:,3)) + pade_b(3,3)*matmul(An(:,:,1),An(:,:,2)) + pade_b(1,3)*An(:,:,1) 
          V = pade_b(6,3)*An(:,:,4)                  + pade_b(4,3)*An(:,:,3)                    + pade_b(2,3)*An(:,:,2)                   + pade_b(0,3)*Id
        case(9)
          e_A = matmul(An(:,:,3), An(:,:,3)) ! store A^8 in e_A
          U = pade_b(9,4)*matmul(An(:,:,1),e_A)+ pade_b(7,4)*matmul(An(:,:,1),An(:,:,4)) + pade_b(5,4)*matmul(An(:,:,1),An(:,:,3)) + pade_b(3,4)*matmul(An(:,:,1),An(:,:,2)) + pade_b(1,4)*An(:,:,1)
          V = pade_b(8,4)*e_A                  + pade_b(6,4)*An(:,:,4)                   + pade_b(4,4)*An(:,:,3)                   + pade_b(2,4)*An(:,:,2)                   + pade_b(0,4)*Id
        case(13)
          s = max(0,ceiling( log(znormA/pade_theta(5)) / log(2.) ))
          An(:,:,1) = An(:,:,1) / (2.**s)
          U = pade_b(13,5)*An(:,:,4) + pade_b(11,5)*An(:,:,3) + pade_b(9,5)*An(:,:,2)
          U = matmul(An(:,:,4),U)
          U = U + pade_b(7,5)*An(:,:,4) + pade_b(5,5)*An(:,:,3) + pade_b(3,5)*An(:,:,2) + pade_b(1,5)*Id
          U = matmul(An(:,:,1),U)

          V = pade_b(12,5)*An(:,:,4) + pade_b(10,5)*An(:,:,3) + pade_b(8,5)*An(:,:,2)
          V = matmul(An(:,:,4),V)
          V = V + pade_b(6,5)*An(:,:,4) + pade_b(4,5)*An(:,:,3) + pade_b(2,5)*An(:,:,2) + pade_b(0,5)*Id
      end select

      if(znormA <= pade_theta(i) .or. pade_ordr(i) == 13) then
        e_A = solve_getrs_complex(n, -U+V, U+V)
        residual = sum(abs(matmul(-U+V, e_A)-U-V))
        if(abs(residual) > 1d-10) then
          e_A = inverse_gelss_complex(n, -U+V)
          e_A = matmul(e_A,  U+V)
          residual = sum(abs(matmul(-U+V, e_A)-U-V))
        endif
        do j = 1,s
          e_A = matmul(e_A, e_A)
        enddo
        e_A = exp(traceA/n) * e_A
        deallocate(U, V, An, Id)
        return
      endif

    enddo

  end subroutine expm_complex

  !
  !
  !
  function solve_getrs_real(n, a, b) result(c)
    integer(ik), intent(in)        :: n
    real(drk), intent(in)          :: a(:,:)
    real(drk), intent(in)          :: b(:,:)

    external zgetrf
    external zgetrs
    real(drk)                      :: c(n,n)
    real(drk)                      :: lu(n,n)
    integer(ik)                    :: na
    integer(ik)                    :: ipiv(n)
    integer(ik)                    :: info

    na   = n
    lu   = a(:na, :na)
    c    = b(:na, :na)

    call dgetrf(na, na, lu, na, ipiv, info)
    if(info /=0) stop 'error in mkl_dgetrf'

    call dgetrs('N', na, na, lu, na, ipiv, c, na, info)
    if(info /=0) stop 'error in mkl_dgetrs'

    return
  end function solve_getrs_real

  !
  !
  !
  function solve_getrs_complex(n, a, b) result(c)
    integer(ik), intent(in)        :: n
    complex(drk), intent(in)       :: a(:,:)
    complex(drk), intent(in)       :: b(:,:)
 
    external zgetrf
    external zgetrs 
    complex(drk)                   :: c(n,n)
    complex(drk)                   :: lu(n,n)
    integer(ik)                    :: na
    integer(ik)                    :: ipiv(n)
    integer(ik)                    :: info
   
    na   = n
    lu   = a(:na, :na)
    c    = b(:na, :na)
 
    call zgetrf(na, na, lu, na, ipiv, info)
    if(info /=0) stop 'error in mkl_zgetrf'

    call zgetrs('N', na, na, lu, na, ipiv, c, na, info)
    if(info /=0) stop 'error in mkl_zgetrs'

    return
  end function solve_getrs_complex

  !
  !
  !
  function inverse_gelss_real(n, a) result(b)
    integer(ik), intent(in)         :: n
    real(drk), intent(in)           :: a(:,:)

    external gelss
    real(drk)                       :: b(n,n)
    real(drk)                       :: svd(n,n)
    real(drk), allocatable          :: work(:)
    real(drk)                       :: s(n)
    integer(ik), save               :: lwork
    integer(ik)                     :: i, na, info,rank
    
    na   = n
    svd  = a(:na,:na)
    b    = zero_drk

    do i = 1,na
      b(i,i) = one_drk
    enddo
    if(lwork < 5*na)lwork = 5*na
    allocate(work(lwork))

    call dgelss(na, na, na, svd, na, b, na, s, 100.0d0*spacing(1.0d0), &
                             rank, work, lwork, info)

    if(info == 0) then
      lwork = int(abs(work(1)))
    else
      stop 'error in lapack_dgelss'
    endif

    deallocate(work)
    return
  end function inverse_gelss_real

  !
  !
  !
  function inverse_gelss_complex(n, a) result(b)
    integer(ik),  intent(in)        :: n
    complex(drk), intent(in)        :: a(:,:)

    external zgelss
    complex(drk)                    :: b(n,n)
    complex(drk)                    :: svd(n,n)
    complex(drk), allocatable       :: work(:)
    real(drk)                       :: rwork(5*n)
    real(drk)                       :: s(n)
    integer(ik), save               :: lwork
    integer(ik)                     :: i, na, info, rank

    na   = n
    svd  = a(:na,:na)
    b    = zero_c

    do i = 1,na
      b(i,i) = one_c
    enddo
    if(lwork < 5*na)lwork = 5*na
    allocate(work(lwork))

    call zgelss(na, na, na, svd, na, b, na, s, 100.0d0*spacing(1.0d0), &
                             rank, work, size(work), rwork, info)
    if(info == 0) then
      lwork = int(abs(work(1)))
    else
      print *,'info=',info
      stop 'error in lapack_zgelss'
    endif

    deallocate(work)
    return
  end function inverse_gelss_complex

end module math 

