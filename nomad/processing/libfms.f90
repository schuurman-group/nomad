!
! Builds the Hamiltonian and other requisite quantities
! necessary to propagate the TDSE under the canonical FMS wavefunction
!
! M. S. Schuurman, Oct. 11, 2018
!
!
module lib_fms


  private
  public build_matrices

  real, allocatable           :: H(:,:)
  real, allocatable           :: S(:,:)
  real, allocatable           :: Sdot(:,:)
  real, allocatable           :: Heff(:,:)



  contains


  subroutine build_matrices(t_basis)



  end subroutine build_matrices

