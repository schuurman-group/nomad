!
! libprop -- a module that defines the data structures to hold trajectory
!            data, as well as general data management
!
! M. S. Schuurman -- Dec. 27, 2019
!
module libprop 
  implicit none
  public init_trajectories
  public add_trajectory
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
      integer(ik)                 :: batch
      integer(ik)                 :: label
      integer(ik)                 :: state
      integer(ik)                 :: current_row
      real(drk)                   :: current_time
      real(drk),allocatable       :: time(:)
      real(drk),allocatable       :: energy(:,:)
      real(drk),allocatable       :: x(:,:)
      real(drk),allocatable       :: p(:,:)
      real(drk),allocatable       :: deriv(:,:,:,:)
      real(drk),allocatable       :: hessian(:,:,:)
    end type trajectory

    type trajectory_params
      real(drk),allocatable       :: widths
      real(drk),allocatable       :: masses
    end type trajectory_params
    
    type trajectory_table
      integer(ik)                   :: current_traj
      type(trajectory), allocatable :: trajectories(:)
    end type trajectory_table

    type(trajectory_table)        :: traj_table
    type(trajectory_params)       :: traj_params

 contains

  !
  !
  !
  subroutine init_trajectories(n, widths, masses)
    implicit none
    integer(ik), intent(in)        :: n
    real(drk), intent(in)          :: widths
    real(drk), intent(in)          :: masses  

    allocate(traj_table.trajectories(n))
    allocate(traj_params.widths(size(widths)))
    allocate(traj_params.masses(size(masses)))

    traj_table.current_traj = 0
    traj_params.widths      = widths
    traj_params.masses      = masses 

    return 
  end subroutine init_trajectories

  !
  !
  !
  subroutine add_trajectory(batch, state, times, energy, x, p, deriv, hessian)
    implicit none
    integer(ik), intent(in)        :: batch
    integer(ik), intent(in)        :: state
    real(ik), intent(in)           :: times(:)
    real(ik), intent(in)           :: energy(:,:),x(:,:), p(:,:)
    real(ik), intent(in)           :: deriv(:,:,:)
    real(ik), intent(in), optional :: hessian(:,:,:)
    
    type(trajectory)               :: new_traj

    traj_table.current_traj = traj_table.current_traj + 1
    new_traj.label = traj_table.current_traj
    new_traj.batch = batch
    new_traj.state = state
    
    np = size(times)
    ns = size(poten, dim=2)
    nd = size(x, dim=2)

    allocate(new_traj.time(np))
    allocate(new_traj.energy(ns, np))
    allocate(new_traj.x(nd, np))
    allocate(new_traj.p(nd, np))
    allocate(new_traj.deriv(nd, ns, ns, np))
    if(present(hessian))allocate(new_traj.hessian(nd, nd, np)

    new_traj.time   = times
    new_traj.energy = energy
    new_traj.x      = x
    new_traj.p      = p
    new_traj.deriv  = deriv
    if(present(hessian))new_traj.hessian = hessian

    traj_table(traj_table.current_traj) = new_traj

  end subroutine add_trajectory

end module libprop 

