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
  public  expm 

  !
 
 contains

  
  !
  ! This uses the 
  !
  subroutine expm(matrix, exp_matrix)
    complex(drk), intent(in)         :: matrix(:,:)
    complex(drk), intent(inout)      :: exp_matrix(:,:)
    integer(ik)                      :: n

    n = size(matrix, dim=1)
    exp_matrix = zero_c

    return
  end subroutine
  

end module math 

