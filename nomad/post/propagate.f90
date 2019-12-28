!
! lib_nomad -- a module that defines a data structure to hold
!              simulation parameters for a nomad simulation
!
! M. S. Schuurman -- Oct. 11, 2018
!
module propagate 
  implicit none
  public 
    integer, parameter :: sik         = selected_int_kind(4)       ! Small integers
    integer, parameter :: ik          = selected_int_kind(8)       ! 64-bit integers
    integer, parameter :: hik         = selected_int_kind(15)      ! "Pointer" integers - sufficient to mem address
    integer, parameter :: drk         = kind(1.d0)                 ! Double-precision real kind
    integer, parameter :: rk          = selected_real_kind(14,17)  ! Standard reals and complex
    integer, parameter :: ark         = selected_real_kind(14,17)  ! Possible increased precision reals and complex

    !
    !
    type trajectory
      integer(sik)                :: state
      real(drk),allocatable       :: energy(:)
      real(drk),allocatable       :: x(:)
      real(drk),allocatable       :: p(:)
      real(drk),allocatable       :: deriv(:,:)
      real(drk),allocatable       :: hessian(:,:)
    end type trajectory

    !
    !
    type traj_table
      integer(sik)                :: state
      real(drk),allocatable       :: time(:)
      real(drk),allocatable       :: energy(:,:)
      real(drk),allocatable       :: x(:,:)
      real(drk),allocatable       :: p(:,:)
      real(drk),allocatable       :: deriv(:,:,:)
      real(drk),allocatable       :: hessian(:,:,:)
    end type traj_table 

 contains

  subroutine add_dataset(



end module propagate

