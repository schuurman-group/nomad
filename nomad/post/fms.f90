!
! Builds the Hamiltonian and other requisite quantities
! necessary to propagate the TDSE under the canonical FMS wavefunction
!
! M. S. Schuurman, Oct. 11, 2018
!
!
module fms
  use libprop
  public init_propagate
  public propagate

  private overlap
  private ke

  real, allocatable           :: H(:,:)
  real, allocatable           :: S(:,:)
  real, allocatable           :: Sdot(:,:)
  real, allocatable           :: Heff(:,:)



  contains

    !
    !
    !
    subroutine init_propagate() bind(c, name='init_propagate')
      implicit none


    end subroutine init_propagate

    !
    !
    !
    subroutine propagate() bind(c, name='propagate')
      implicit none


    end subroutine propagate
    
  
    !*********************************************
    !
    !
    !
    subroutine overlap(bra_t, ket_t)
      implicit none
      type(trajectory), intent(in)  :: bra_t, ket_t

    end subroutine overlap

    !
    !
    ! 
    subroutine ke(bra_t, ket_t)
      implicit none
      type(trajectory), intent(in)  :: bra_t, ket_t

    end subroutine ke

end module fms 

