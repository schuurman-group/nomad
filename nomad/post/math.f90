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

  private
  public factorial
  public zexpm 
  public zinverse
  public zsolve

  !
  integer(hik), parameter    :: pade_b(0:13,1:5) = (/ (/120, 30240, 17297280, 17643225600, 64764752532480000/), & 
                                                      (/ 60, 15120, 8648640,  8821612800,  32382376266240000/), &
                                                      (/ 12, 3360,  1995840,  2075673600,  7771770303897600 /), &
                                                      (/  1, 420,   277200,   302702400,   1187353796428800 /), &
                                                      (/  0, 30,    25200,    30270240,    129060195264000  /), &
                                                      (/  0, 1,     1512,     2162160,     10559470521600   /), &
                                                      (/  0, 0,     56,       110880,      670442572800     /), &
                                                      (/  0, 0,     1,        3960,        33522128640      /), &
                                                      (/  0, 0,     0,        90,          1323241920       /), &
                                                      (/  0, 0,     0,        1,           40840800         /), &
                                                      (/  0, 0,     0,        0,           960960           /), &
                                                      (/  0, 0,     0,        0,           16380            /), &
                                                      (/  0, 0,     0,        0,           182              /), &
                                                      (/  0, 0,     0,        0,           1                /)/)
 
  real(drk), parameter      :: pade_theta(1:5) = (/ 0.01495585217958292,0.2539398330063230, &
                                                 0.9504178996162932, 2.097847961257068, &
                                                 5.371920351148152 /)
  integer(ik), parameter    :: pade_ordr(1:5)  = (/ 3,5,7,9,13 /)

 contains

  !
  !
  !
  recursive function factorial(n) result (fac)
    integer(ik), intent(in)  :: n
    integer(ik)              :: fac

    if (n==0) then
       fac = 1
    else
       fac = n * factorial(n-one_ik)
    endif
  end function factorial
  
  !
  ! This uses the 
  !
  subroutine zexpm(n, A, e_A)
    integer(ik), intent(in)          :: n
    complex(drk), intent(inout)      :: A(n,n)
    complex(drk), intent(inout)      :: e_A(n,n)

    integer(ik)                      :: s
    integer(ik)                      :: i
    complex(drk),allocatable         :: U(:,:), V(:,:)
    complex(drk),allocatable         :: An(:,:,:)
    complex(drk),allocatable         :: Id(:,:)
    complex(drk)                     :: traceA
    real(drk)                        :: znormA

    allocate(U(n,n), V(n,n))
    allocate(Id(n,n))
    allocate(An(n,n,3)) !corresponds to A^2, A^4, A^6

    ! this is not really necessary, but makes things clearer, at
    ! least to start
    Id = zero_c    
    do i = 1,n
      Id(i,i) = one_c
    enddo 

    ! pre-process to reduce the norm 
    traceA      = sum( (/ (A(i,i), i=1,n) /) )
    do i = 1,n
      A(i,i) = A(i,i) - traceA/n
    enddo
    znormA = znorm1(A)

    ! check if lower-order pade approximants have
    ! sufficient accuracy
    do i = 1,size(pade_ordr)
 
      ! I'm sure there's a more elegant way to do this, and
      ! I'll probably implement it some day...

      ! form requisite U/V matrices in case we need to form
      ! corresponding Pade approximant 
      select case(pade_ordr(i))
        case(3)
          An(:,:,1) = matmul(A,A)
          U = pade_b(3,1)*matmul(A,An(:,:,1)) + pade_b(1,1)*A
          V = pade_b(2,1)*An(:,:,1)           + pade_b(0,1)*Id
        case(5)
          An(:,:,2) = matmul(An(:,:,1),An(:,:,1))
          U = pade_b(5,2)*matmul(A,An(:,:,2)) + pade_b(3,2)*matmul(A,An(:,:,1)) + pade_b(1,2)*A
          V = pade_b(4,2)*An(:,:,2)           + pade_b(2,2)*An(:,:,1)           + pade_b(0,2)*Id 
        case(7)
          An(:,:,3) = matmul(An(:,:,1),An(:,:,2))
          U = pade_b(7,3)*matmul(A,An(:,:,3)) + pade_b(5,3)*matmul(A,An(:,:,2)) + pade_b(3,3)*matmul(A,An(:,:,1)) + pade_b(1,3)*A 
          V = pade_b(6,3)*An(:,:,3)           + pade_b(4,3)*An(:,:,2)           + pade_b(2,3)*An(:,:,1)           + pade_b(0,3)*Id
        case(9)
          e_A = matmul(An(:,:,2), An(:,:,2)) ! store A^8 in e_A
          U = pade_b(9,4)*matmul(A,e_A)       + pade_b(7,4)*matmul(A,An(:,:,3)) + pade_b(5,4)*matmul(A,An(:,:,2)) + pade_b(3,4)*matmul(A,An(:,:,1)) + pade_b(1,4)*A
          V = pade_b(8,4)*e_A                 + pade_b(6,4)*An(:,:,3)           + pade_b(4,4)*An(:,:,2)           + pade_b(2,4)*An(:,:,1)           + pade_b(0,4)*Id
      end select

      if(znormA <= pade_theta(i)) then
        e_A = zsolve(n, -U+V, U+V)
        e_A = exp(traceA/n) * e_A
        deallocate(U, V, An, Id)
        return
      endif

    enddo

    ! if we're still here, need to go order 13
    s = ceiling( log(znormA/pade_theta(5)) / log(2.) )
    A = A / (2.**s)
 
    U = pade_b(13,5)*An(:,:,3) + pade_b(11,5)*An(:,:,2) + pade_b(9,5)*An(:,:,1)
    U = matmul(An(:,:,3),U)
    U = U + pade_b(7,5)*An(:,:,3) + pade_b(5,5)*An(:,:,2) + pade_b(3,5)*An(:,:,1) + pade_b(1,5)*Id
    U = matmul(A,U)

    V = pade_b(12,5)*An(:,:,3) + pade_b(10,5)*An(:,:,2) + pade_b(8,5)*An(:,:,1)
    V = matmul(An(:,:,3),V)
    V = V + pade_b(6,5)*An(:,:,3) + pade_b(4,5)*An(:,:,2) + pade_b(2,5)*An(:,:,1) + pade_b(0,5)*Id 

    e_A = zsolve(n, -U+V, U+V)
    do i = 1,s
      e_A = matmul(e_A, e_A)
    enddo
    e_A = exp(traceA/n) * e_A

    deallocate(U, V, An, Id)
    return
  end subroutine

  !
  !
  !
  function znorm1(matrix) result(norm)
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
  end function znorm1

  !
  !
  !
  function zsolve(n, a, b) result(C)
    integer(ik), intent(in)        :: n
    complex(drk), intent(in)       :: a(:,:)
    complex(drk), intent(in)       :: b(:,:)
 
    external zgetrf
    external zgetrs 
    complex(drk)                   :: c(n,n)
    complex(drk)                   :: lu(n,n)
    integer                        :: na
    integer                        :: ipiv(n)
    integer                        :: info
   
    na   = n
    lu   = a(:na, :na)
    c    = b(:na, :na)
    info = 0 
    call zgetrf(na, na, lu, na, ipiv, info)
    call zgetrs('N', na, na, lu, na, ipiv, c, na, info)

    if(info /=0 )stop 'error in lapack_zgetrs'

    return
  end function zsolve


  !
  !
  !
  function zinverse(n, a) result(b)
    integer(ik),  intent(in)        :: n
    complex(drk), intent(in)        :: a(:,:)

    external zgelss
    double complex                  :: b(n,n)
    double complex                  :: svd(n,n)
    double complex, allocatable     :: work(:)
    double precision                :: rwork(5*n)
    double precision                :: s(n)
    integer, save                   :: lwork=0
    integer                         :: i, na, info, rank

    na   = n
    svd  = a(:na,:na)
    b    = zero_c
    info = 0
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
      stop 'error in lapack_zgelss'
    endif

    deallocate(work)
    return
  end function zinverse

end module math 

