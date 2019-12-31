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
      integer(ik)                 :: nstates
      integer(ik)                 :: current_row
      real(drk)                   :: current_time
      real(drk),allocatable       :: time(:)
      real(drk),allocatable       :: widths(:)
      real(drk),allocatable       :: masses(:)
      real(drk),allocatable       :: energy(:,:)
      real(drk),allocatable       :: x(:,:)
      real(drk),allocatable       :: p(:,:)
      real(drk),allocatable       :: deriv(:,:,:,:)
    end type trajectory

    type trajectory_table
      integer(ik)                   :: current_traj
      type(trajectory), allocatable :: trajectories(:)
    end type trajectory_table

    type(trajectory_table)        :: traj_table

 contains

  !
  !
  !
  subroutine init_trajectories(n) bind(c, name='init_trajectories')
    implicit none
    integer(ik), intent(in)        :: n ! number of trajectories

    allocate(traj_table.trajectories(n))
    traj_table.current_traj = 0

    return 
  end subroutine init_trajectories

  !
  !
  !
  subroutine add_trajectory(batch, np, ns, nd, state, widths, masses, time, energy, x, p, deriv) bind(c, name='add_trajectory')
    implicit none
    integer(ik), intent(in)         :: batch
    integer(ik), intent(in)         :: np, ns, nd !number of steps, states, and trajectory dimensions
    integer(ik), intent(in)         :: state
    real(drk), intent(in)           :: widths(nd)
    real(drk), intent(in)           :: masses(nd)
    real(drk), intent(in)           :: time(np)
    real(drk), intent(in)           :: energy(ns*np), x(nd*np), p(nd*np)
    real(drk), intent(in)           :: deriv(nd*ns*ns*np)
    type(trajectory)                :: new_traj

    traj_table.current_traj = traj_table.current_traj + 1
    new_traj.batch          = batch
    new_traj.label          = traj_table.current_traj
    new_traj.state          = state
    new_traj.nstates        = ns
    new_traj.current_row    = 1
    new_traj.current_time   = time(1)
   
    print *,'np, ns, nd',np,ns,nd
    call flush()

    allocate(new_traj.widths(nd))
    allocate(new_traj.masses(nd))   
    allocate(new_traj.time(np))
    allocate(new_traj.energy(ns, np))
    allocate(new_traj.x(nd, np))
    allocate(new_traj.p(nd, np))
    allocate(new_traj.deriv(nd, ns, ns, np))

    new_traj.widths = widths
    new_traj.masses = masses
    new_traj.time   = time
    new_traj.energy = reshape(energy, (/ns, np/))
    new_traj.x      = reshape(x,      (/nd, np/))
    new_traj.p      = reshape(p,      (/nd, np/))
    new_traj.deriv  = reshape(deriv,  (/nd, ns, ns, np/))

    print *,'time(1)=',new_traj.time(1)
    print *,'energy(1)=',new_traj.energy(:,1)
    print *,'x=',new_traj.x(:,1)

    traj_table.trajectories(traj_table.current_traj) = new_traj

  end subroutine add_trajectory

end module libprop 

