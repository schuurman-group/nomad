!
! This library was created to peform the matrix exponential, but 
! may end up being a repository for a nubmer of other routines
!
! M. S. Schuurman, Jan 6., 2020
!
!
module constants 
  use accuracy
  implicit none

  real(drk), parameter      :: pi   = 3.141592653589793
  real(drk), parameter      :: log2 = 0.693147180559945309417232 
  ! Euler-Mascheroni constant
  real(drk), parameter      :: euler = 0.577215664901532860606512090082

end module constants 
