!
! libprop -- a module that defines the data structures to hold trajectory
!            data, as well as general data management
!
! M. S. Schuurman -- Dec. 27, 2019
!
module libprop 
  use accuracy
  use math
  implicit none
  public
   public init_trajectories
   public set_parameters
   public add_trajectory
   
   public collect_trajectories
   public locate_trajectory
   public extract_trajectory
   public interpolate_trajectory
   public locate_amplitude
   public interpolate_amplitude
   public update_amplitude
   public propagate_amplitude
   public normalize_amplitude

   public get_time_index
   public tr_potential
   public tr_kinetic
   public tr_velocity
   public tr_force
   public nuc_overlap
   public qn_integral
   public qn_vector
   public traj_size
   public amp_size

    ! total number of trajectories
    integer(ik)                   :: n_total
    ! number of 'batches', or, initial conditions
    integer(ik)                   :: n_batch
    ! number of nuclear degrees of freedom
    integer(ik)                   :: n_crd
    ! total number of electronic states
    integer(ik)                   :: n_state
    ! the trajectory count for the _current timestep_
    integer(ik)                   :: tstep_cnt
    ! default time step used in simulation
    real(drk)                     :: t_step
    ! tolerance for identifying discrete time step with a given time
    real(drk)                     :: dt_toler

    ! if .true., construct Hamiltonian in full trajectory basis, else,
    ! incoherently average over initial conditions (i.e. indep. first. gen.)
    logical                       :: full_basis
    ! Is Hamiltonian hermitian?
    logical                       :: hermitian
    ! allowed amplitude initialization schemes
    character(len=8),parameter    :: initial_conds(1:3) = (/ 'overlap ','uniform ','explicit' /)
    character(len=8)              :: init_amps
    ! implemented integration schemes
    character(len=6),parameter    :: integral_methods(1:3) = (/'saddle',  'taylor', 'dirac '/)
    character(len=6)              :: integrals
    integer(ik)                   :: integral_ordr

    ! widths are genereally _not_ time- or trajectory-dependent. Let's put this here for now
    real(drk), allocatable        :: widths(:)
    ! masses are generally _not_ time- or trajectory-dependent. Let's put this here for now
    real(drk), allocatable        :: masses(:)

    !
    ! this data type holds trajectory data
    ! at a single time-step
    ! 
    type trajectory
      integer(ik)                 :: batch
      integer(ik)                 :: label
      integer(ik)                 :: state
      real(drk)                   :: time
      real(drk)                   :: phase
      real(drk), allocatable      :: energy(:)
      real(drk), allocatable      :: x(:)
      real(drk), allocatable      :: p(:)
      real(drk), allocatable      :: deriv(:,:,:)
    end type trajectory
    !
    ! this data type holds all the trajectory data
    ! for a series of times
    !
    type trajectory_table
      integer(ik)                 :: batch
      integer(ik)                 :: label
      integer(ik)                 :: state
      integer(ik)                 :: nstates
      integer(ik)                 :: nsteps
      integer(ik)                 :: current_row
      real(drk),allocatable       :: time(:)
      real(drk),allocatable       :: phase(:)
      real(drk),allocatable       :: widths(:)
      real(drk),allocatable       :: masses(:)
      real(drk),allocatable       :: energy(:,:)
      real(drk),allocatable       :: x(:,:)
      real(drk),allocatable       :: p(:,:)
      real(drk),allocatable       :: deriv(:,:,:,:)
    end type trajectory_table
    !
    !
    !
    type amplitude_table
      integer(ik)                 :: batch
      integer(ik)                 :: label   
      integer(ik)                 :: current_row
      real(drk),allocatable       :: time(:)
      complex(drk),allocatable    :: amplitude(:)
    end type amplitude_table   

    ! this array of trajectory_tables represents _all_ the 
    ! trajectory basis information for a series of times. This
    ! is the 'master' list that is queried as we propagate
    type(trajectory_table), allocatable :: basis_table(:)
    type(amplitude_table), allocatable  :: amp_table(:)
    type(trajectory), allocatable       :: traj_list(:)
    type(trajectory)                    :: ref_traj

 contains

  !********************************************************************************
  !********************************************************************************
  ! Externally invoked subroutines 

  !
  !
  !
  subroutine init_trajectories(n_grp, n_tot, n_states, n_dim, ref_width, ref_mass, x, p) bind(c, name='init_trajectories')
    integer(ik), intent(in)        :: n_grp ! number of trajectory "batches" -- generally derived
                                            ! from the same initial condition
    integer(ik), intent(in)        :: n_tot ! total number of trajectories
    integer(ik), intent(in)        :: n_states ! number of electronic states
    integer(ik), intent(in)        :: n_dim ! dimension of the trajectories
    real(drk), intent(in)          :: ref_width(n_dim) ! widths of each degree of freedom
    real(drk), intent(in)          :: ref_mass(n_dim)  ! masses of each degree of freedom
    real(drk), intent(in)          :: x(n_dim)  ! positions of the reference structure
    real(drk), intent(in)          :: p(n_dim)  ! momentum of the reference structure   

    n_batch = n_grp
    n_total = n_tot 
    n_crd   = n_dim
    n_state = n_states

    ! allocate the basis set table
    allocate(basis_table(n_total))

    ! allocate the amplitude table
    allocate(amp_table(n_total))

    ! allocate trajectory-specific quantities
    allocate(widths(n_crd))
    allocate(masses(n_crd))
    allocate(ref_traj%x(n_crd))
    allocate(ref_traj%p(n_crd))

    widths     = ref_width
    masses     = ref_mass
    ref_traj%x = x
    ref_traj%p = p

    return 
  end subroutine init_trajectories

  !
  !
  !
  subroutine set_parameters(time_step, coherent, init_method, int_method, int_order) bind(c, name='set_parameters')
    real(drk), intent(in)           :: time_step
    logical, intent(in)             :: coherent
    integer(ik), intent(in)         :: init_method
    integer(ik), intent(in)         :: int_method
    integer(ik), intent(in)         :: int_order

    t_step     = time_step
    dt_toler   = 0.01 * t_step
    full_basis = coherent

    if(init_method <= size(initial_conds)) then
      init_amps = initial_conds(init_method)
    else
      stop 'amplitude initialization not recognized'
    endif
 
    if(int_method <= size(integral_methods)) then
      integrals = integral_methods(int_method)
    else
      stop 'integral method not recognized'
    endif

    integral_ordr = int_order
    hermitian     = .true.

    return
  end subroutine set_parameters

  !
  !
  !
  subroutine add_trajectory(batch, np, state, widths, masses, time, phase, energy, x, p, deriv) bind(c, name='add_trajectory')
    integer(ik), intent(in)         :: batch
    integer(ik), intent(in)         :: np !number of time steps
    integer(ik), intent(in)         :: state
    real(drk), intent(in)           :: widths(n_crd)
    real(drk), intent(in)           :: masses(n_crd)
    real(drk), intent(in)           :: time(np)
    real(drk), intent(in)           :: phase(np)
    real(drk), intent(in)           :: energy(np*n_state), x(np*n_crd), p(np*n_crd)
    real(drk), intent(in)           :: deriv(np*n_crd*n_state*n_state)

    integer(ik), save               :: traj_cnt = 0

    ! Set and allocate the trajectory table
    traj_cnt                          = traj_cnt + 1
    basis_table(traj_cnt)%batch       = batch
    basis_table(traj_cnt)%label       = traj_cnt
    basis_table(traj_cnt)%state       = state+1   ! fortran states go 1:n_state, nomad goes 0:n_state-1
    basis_table(traj_cnt)%nsteps      = np  
    basis_table(traj_cnt)%current_row = 1

    allocate(basis_table(traj_cnt)%widths(n_crd))
    allocate(basis_table(traj_cnt)%masses(n_crd))   
    allocate(basis_table(traj_cnt)%time(np))
    allocate(basis_table(traj_cnt)%phase(np))
    allocate(basis_table(traj_cnt)%energy(n_state, np))
    allocate(basis_table(traj_cnt)%x(n_crd, np))
    allocate(basis_table(traj_cnt)%p(n_crd, np))
    allocate(basis_table(traj_cnt)%deriv(n_state, n_state, n_crd, np))

    basis_table(traj_cnt)%widths    = widths
    basis_table(traj_cnt)%masses    = masses
    basis_table(traj_cnt)%time      = time
    basis_table(traj_cnt)%phase     = phase
    basis_table(traj_cnt)%energy    = reshape(energy, (/n_state, np/))
    basis_table(traj_cnt)%x         = reshape(x,      (/n_crd, np/))
    basis_table(traj_cnt)%p         = reshape(p,      (/n_crd, np/))
    basis_table(traj_cnt)%deriv     = reshape(deriv,  (/n_crd, n_state, n_state, np/), (/zero_drk, zero_drk, zero_drk, zero_drk/), (/2, 3, 1, 4/))

    ! allocate the correpsonding amplitude table
    allocate(amp_table(traj_cnt)%time(np))
    allocate(amp_table(traj_cnt)%amplitude(np))
    amp_table(traj_cnt)%time        = -1.
    amp_table(traj_cnt)%batch       = batch
    amp_table(traj_cnt)%label       = traj_cnt
    amp_table(traj_cnt)%current_row = 1
    amp_table(traj_cnt)%amplitude   = zero_c

  end subroutine add_trajectory

  !****************************************************************************************
  !****************************************************************************************
  ! Trajectory/data storage, manipulation and retrieval

  !
  !
  !
  subroutine collect_trajectories(t, batch, labels)
    real(drk), intent(in)                   :: t
    integer(ik), intent(in)                 :: batch
    integer(ik), allocatable, intent(inout) :: labels(:)

    integer(ik)                             :: i

    ! grab all the trajectories that exist at time t
    call locate_trajectory(t, batch)

    ! if we found too many, remake the traj_list to accommodate and call 
    ! again.
    if(tstep_cnt > traj_size(traj_list)) then
      call destroy_traj_list()
      call create_traj_list(tstep_cnt)
      call locate_trajectory(t, batch)
    endif

    if(.not.allocated(labels)) then
      allocate(labels(tstep_cnt))
    elseif(size(labels) /= tstep_cnt) then
      deallocate(labels)
      allocate(labels(tstep_cnt))
    endif
 
    do i = 1,tstep_cnt
      labels(i) = traj_list(i)%label
    enddo

    return
  end subroutine collect_trajectories

  !
  ! Locate all the trajectories that exist at a specific time
  ! and return their indices
  !
  subroutine locate_trajectory(time, batch)
    real(drk), intent(in)             :: time 
    integer(ik), intent(in)           :: batch

    integer(ik)                       :: i   ! counter variable
    integer(ik)                       :: t_row(2) ! row in traj table correpsonding to 'time'
    integer(ik)                       :: max_traj

    max_traj  = traj_size(traj_list)
    tstep_cnt = 0 

    do i = 1,n_total
  
      ! if we're averaging over batch, we'll select on what
      ! initial condition we're propagating   
      if(batch > 0 .and. basis_table(i)%batch /= batch) continue  

      t_row = get_time_index('traj', basis_table(i)%label, time)

      if(t_row(1) /= 0) then
        basis_table(i)%current_row = t_row(1)
        tstep_cnt                  = tstep_cnt + 1
        if(t_row(2) == 0) then
          if(tstep_cnt <= max_traj)call extract_trajectory(i, t_row(1), tstep_cnt)
        else
          if(tstep_cnt <= max_traj)call interpolate_trajectory(i, time, t_row, tstep_cnt)
        endif
      endif

    enddo

    return
  end subroutine locate_trajectory

  !
  !
  !
  subroutine extract_trajectory(table_indx, t_row, indx)
    integer(ik), intent(in)           :: table_indx
    integer(ik), intent(in)           :: t_row
    integer(ik), intent(in)           :: indx

    traj_list(indx)%batch  = basis_table(table_indx)%batch
    traj_list(indx)%label  = basis_table(table_indx)%label
    traj_list(indx)%state  = basis_table(table_indx)%state

    traj_list(indx)%time   = basis_table(table_indx)%time(t_row)
    traj_list(indx)%phase  = basis_table(table_indx)%phase(t_row)
    traj_list(indx)%energy = basis_table(table_indx)%energy(:,t_row)
    traj_list(indx)%x      = basis_table(table_indx)%x(:,t_row)
    traj_list(indx)%p      = basis_table(table_indx)%p(:,t_row)
    traj_list(indx)%deriv  = basis_table(table_indx)%deriv(:,:,:,t_row)
    
    return
  end subroutine extract_trajectory

  !
  !
  !
  subroutine interpolate_trajectory(table_indx, time, table_row, list_indx)
    integer(ik), intent(in)           :: table_indx    ! index of the trajectory
    real(drk), intent(in)             :: time          ! the requested time
    integer(ik), intent(in)           :: table_row(2)  ! closest time in table
    integer(ik), intent(in)           :: list_indx     ! interpolated trajectory

    real(drk)                         :: fac
    integer(ik)                       :: t0,t1 
    real(drk)                         :: intrp_ener(2), intrp_g(2)
    real(drk)                         :: intrp_x(2), intrp_p(2)
    real(drk)                         :: intrp_deriv(2)

    ! confirm that the requested can be interpolated from the 
    ! existing data in trajctory table
    t0  = basis_table(table_indx)%time(table_row(1))
    t1  = basis_table(table_indx)%time(table_row(2))
    fac = (time - t0) / (t1 - t0)

    intrp_ener   = (/ basis_table(table_indx)%energy(:,table_row(1)), &
                      basis_table(table_indx)%energy(:,table_row(2)) -&
                      basis_table(table_indx)%energy(:,table_row(1)) /)
    intrp_g      = (/ basis_table(table_indx)%phase(table_row(1)), &
                      basis_table(table_indx)%phase(table_row(2)) -&
                      basis_table(table_indx)%phase(table_row(1)) /)
    intrp_x      = (/ basis_table(table_indx)%x(:,table_row(1)), &
                      basis_table(table_indx)%x(:,table_row(2)) -&
                      basis_table(table_indx)%x(:,table_row(1)) /)
    intrp_p      = (/ basis_table(table_indx)%p(:,table_row(1)), &
                      basis_table(table_indx)%p(:,table_row(2)) -&
                      basis_table(table_indx)%p(:,table_row(1)) /)
    intrp_deriv  = (/ basis_table(table_indx)%deriv(:,:,:,table_row(1)), &
                      basis_table(table_indx)%deriv(:,:,:,table_row(2)) -&
                      basis_table(table_indx)%deriv(:,:,:,table_row(1)) /)

    traj_list(list_indx)%batch  = basis_table(table_indx)%batch
    traj_list(list_indx)%label  = basis_table(table_indx)%label
    traj_list(list_indx)%state  = basis_table(table_indx)%state
    traj_list(list_indx)%time   = time
    traj_list(list_indx)%energy = intrp_ener(1)  + fac * intrp_ener(2)
    traj_list(list_indx)%phase  = intrp_g(1)     + fac * intrp_g(2)
    traj_list(list_indx)%x      = intrp_x(1)     + fac * intrp_x(2)
    traj_list(list_indx)%p      = intrp_p(1)     + fac * intrp_p(2)
    traj_list(list_indx)%deriv  = intrp_deriv(1) + fac * intrp_deriv(2)

    return
  end subroutine interpolate_trajectory

  !
  ! Allocate the trajectory list -- the amplitude list is 
  !  hitchhiking along for now
  !
  subroutine create_traj_list(n_traj)
    integer(ik), intent(in)          :: n_traj
    integer(ik)                      :: i_traj

    if(allocated(traj_list))call destroy_traj_list()

    allocate(traj_list(n_traj))
    do i_traj = 1,n_traj
      allocate(traj_list(i_traj)%energy(n_state))
      allocate(traj_list(i_traj)%x(n_crd))
      allocate(traj_list(i_traj)%p(n_crd))
      allocate(traj_list(i_traj)%deriv(n_crd, n_state, n_state))
    enddo

    return
  end subroutine create_traj_list

  !
  ! Deallocate the trajectory list -- the amplitude list is
  !  hitchhiking along for now
  !
  subroutine destroy_traj_list()
    integer(ik)                     :: i_traj

    do i_traj = 1,traj_size(traj_list)
      deallocate(traj_list(i_traj)%energy)
      deallocate(traj_list(i_traj)%x)
      deallocate(traj_list(i_traj)%p)
      deallocate(traj_list(i_traj)%deriv)
    enddo

    if(allocated(traj_list))deallocate(traj_list)

    return
  end subroutine destroy_traj_list

  !*******************************************************************************
  !*******************************************************************************
  ! Amplitude initialization, manipulation/storage and propagation

  !
  !
  !
  function init_amplitude(time) result(amps)
    real(drk)                       :: time
    complex(drk), allocatable       :: amps(:)
    integer(ik)                     :: i

    allocate(amps(size(traj_list)))

    select case(init_amps)
      case('overlap ')
        do i = 1,size(traj_list)
          amps(i)   = nuc_overlap(ref_traj, traj_list(i))
        enddo

      case('uniform ')
        amps = one_c

      case('explicit')
        amps = one_c

    end select

    amps = normalize_amplitude(amps)
    print *,'time=',time,' amps=',amps
    call update_amplitude(time, amps)

    return
  end function init_amplitude
 
  !
  !
  ! 
  function update_basis(new_label, old_label, c) result(c_new)
    integer(ik),intent(in)                  :: new_label(:)
    integer(ik), allocatable, intent(inout) :: old_label(:)
    complex(drk), intent(in)                :: c(:)

    complex(drk), allocatable       :: c_new(:)
    integer(ik)                     :: n_new, n_old
    integer(ik)                     :: i, new_i

    n_new = size(new_label)
    n_old = size(old_label)
    allocate(c_new(n_new))

    c_new = zero_c
    do i = 1,n_old
      new_i = findloc(new_label, old_label(i), dim=1)
      if(new_i /= 0) c_new(new_i) = c(i)     
    enddo

    deallocate(old_label)
    allocate(old_label(n_new))
    old_label = new_label

    return
  end function update_basis

  !
  !
  !
  function propagate_amplitude(c0, H, t_start, t_end) result(new_amp)
    complex(drk), intent(in)        :: c0(:)
    complex(drk), intent(in)        :: H(:,:)
    real(drk), intent(in)           :: t_start
    real(drk), intent(in)           :: t_end

    complex(drk),allocatable        :: new_amp(:)
    complex(drk),allocatable        :: amps(:,:)
    complex(drk),allocatable        :: B(:,:)
    complex(drk),allocatable        :: U(:,:)
    integer(ik)                     :: n

    n = size(c0)
    if(n /= size(H, dim=1))stop 'ERROR: size c0 and H do not match'
    allocate(new_amp(n), amps(n,1), B(n,n), U(n,n))

    amps(:n,1) = c0
    print *,'old_amp=',amps
    B = -I * H * (t_end - t_start)
  
    print *,'B=',B
    call zexpm(n, B, U)

    print *,'U=',U
    amps = matmul(U, amps)
    new_amp = amps(:n,1)
    print *,'new_amp=',new_amp
    print *,'new amp norm=',sqrt(dot_product(new_amp, new_amp))

    return
  end function propagate_amplitude

  !
  !
  !
  function locate_amplitude(time) result(amps)
    real(drk), intent(in)             :: time

    complex(drk)                      :: amps(tstep_cnt)
    integer(ik)                       :: i 
    integer(ik)                       :: t_row(2)  


    ! for now, we need the dimension of the amplitude vector to
    ! be the same as the dimension of the trajectory list. So, we
    ! assume that list is accurate and pull trajectory labels from there.
    do i = 1,size(traj_list)
      t_row    = get_time_index('amps', traj_list(i).label, time)

      ! if time step doesn't exist, set amplitude to zero
      if(t_row(1) == 0 .or. t_row(2) == -1) then
        amps(i) = zero_c

      ! if we have exactly this time step, use it
      elseif(t_row(2) == 0) then
        amps(i) = amp_table(traj_list(i).label)%amplitude(t_row(1))
  
      !...otherwise we have to interpolate
      elseif(t_row(2) /= -1) then
        amps(i) = interpolate_amplitude(traj_list(i).label, time, t_row)

      ! shouldn't get here.. 
      else
        print *,'undefined behavior in locate_amplitude'
        amps(i) = zero_c
      endif
    enddo

    print *,'amps=',amps
    return
  end function locate_amplitude

  !
  !
  !
  function interpolate_amplitude(amp_indx, time, t_row) result(intrp_amp)
    integer(ik), intent(in)           :: amp_indx
    real(drk), intent(in)             :: time
    integer(ik), intent(in)           :: t_row(2)
  
    complex(drk)                      :: intrp_amp
    complex(drk)                      :: c0,c1
    real(drk)                         :: t0,t1
    real(drk)                         :: fac

    t0  = amp_table(amp_indx)%time(t_row(1))
    t1  = amp_table(amp_indx)%time(t_row(2))
    c0  = amp_table(amp_indx)%amplitude(t_row(1))
    c1  = amp_table(amp_indx)%amplitude(t_row(2))
    fac = (time - t0) / (t1 - t0)

    intrp_amp = c0 + fac * (c1 - c0)

    return
  end function interpolate_amplitude

  !
  !
  !
  subroutine update_amplitude(time, amplitudes)
    real(drk), intent(in)            :: time
    complex(drk),intent(in)          :: amplitudes(:)

    integer(ik)                      :: i, table_indx
    integer(ik)                      :: row(2)   

    do i=1,tstep_cnt    
      table_indx = traj_list(i)%label
      row = get_time_index('amps', table_indx, time)
      if(row(2) > 0) stop 'trying update amplitude at unkonwn time'
      amp_table(table_indx)%time(row(1))      = time
      amp_table(table_indx)%amplitude(row(1)) = amplitudes(i)
    enddo

    return
  end subroutine update_amplitude

  !
  !
  !
  function get_time_index(list_type, indx, time) result(rows)
    character(len=4), intent(in)     :: list_type
    integer(ik), intent(in)          :: indx
    real(drk), intent(in)            :: time

    real(drk)                        :: dt
    integer(ik)                      :: rows(2)

    rows = 0

    select case(list_type)

      case('traj')
        ! to save searching through the table over and over again, 
        ! we'll use the previously accessed row as a starting guess
        if(basis_table(indx)%time(basis_table(indx)%current_row) <= time)then
          rows(1) = basis_table(indx)%current_row
        else
          rows(1) = 1
        endif

        ! search the table for the requested time
        dt = abs(basis_table(indx)%time(rows(1))-time) 
        do 
          if(rows(1) == basis_table(indx)%nsteps)exit
          if(dt < dt_toler .or. basis_table(indx)%time(rows(1))>time) exit
          rows(1) = rows(1) + 1
          dt      = abs(basis_table(indx)%time(rows(1))-time)
        enddo 

        ! either the time exists explicitly in the table..
        if(dt < dt_toler) then
          continue
        ! ...the requested time is obtained via interpolation
        elseif(basis_table(indx)%time(rows(1)) > time) then 
          if(rows(1) > 1) then
            rows(2) = rows(1)
            rows(1) = rows(2)-1
          else
            rows(1) = 0
          endif
        ! ...else the trajectory doesn't exist at the this time
        else
          rows(1) = 0
        endif

      case('amps')

        ! to save searching through the table over and over again, 
        ! we'll use the previously accessed row as a starting guess
        if(amp_table(indx)%time(amp_table(indx)%current_row) <= time) then
          rows(1) = amp_table(indx)%current_row
        else
          rows(1) = 1
        endif

        ! search the table for the requested time
        dt = abs(basis_table(indx)%time(rows(1))-time)
        do 
          if(rows(1) == rows(1) == basis_table(indx)%nsteps) exit
          if(dt < dt_toler .or. &
             amp_table(indx)%time(rows(1)) == -1. .or. &
             amp_table(indx)%time(rows(1)) > time) exit
          rows(1) = rows(1) + 1
          dt      = abs(amp_table(indx)%time(rows(1))-time)
        enddo
        
        ! found the time, return the row
        if(dt < dt_toler) then
          continue
        ! this is a new time, set second index to -1 (super hacky -- will fix later)
        elseif(amp_table(indx)%time(rows(1)) == -1.) then
          rows(2) = -1
        ! see if we can interpolate
        elseif(amp_table(indx)%time(rows(1)) > time) then
          if(rows(1) > 1) then
            rows(2) = rows(1)
            rows(1) = rows(2)-1
          else
            rows(1) = 0
          endif
        ! else I don't know what to with this...
        else
          rows(1) = 0
        endif

    end select

    return
  end function get_time_index

  !**********************************************************************************************
  !**********************************************************************************************
  ! Numerical routines for computing integrals, trajectory properties
  !

  !
  !
  !
  function normalize_amplitude(c) result(c_norm)
    complex(drk), intent(in)           :: c(:)

    complex(drk),allocatable           :: c_norm(:)
    complex(drk)                       :: norm_amp

    allocate(c_norm(size(c)))

    norm_amp = sqrt(dot_product(c, c))
    if(abs(real(aimag(norm_amp),kind=drk)) > mp_drk) stop 'norm error'
    c_norm = c / real(norm_amp,kind=drk)

    return
  end function normalize_amplitude

  !
  !
  ! 
  function tr_potential(traj) result(poten)
    type(trajectory), intent(in)       :: traj

    real(drk)                          :: poten

    poten = traj%energy(traj%state)
    return
  end function tr_potential

  !
  !
  !
  function tr_kinetic(traj) result(kinetic)
    type(trajectory), intent(in)       :: traj

    real(drk)                          :: kinetic

    kinetic = sum( 0.5 * traj%p * traj%p / masses)

    return
  end function tr_kinetic

  !
  !
  !
  function tr_velocity(traj) result(velocity)
    type(trajectory), intent(in)      :: traj

    real(drk)                         :: velocity(n_crd)

    velocity = traj%p / masses

    return
  end function tr_velocity

  !
  !
  !
  function tr_force(traj) result(force)
    type(trajectory)                  :: traj
  
    real(drk)                         :: force(n_crd)

    force = -traj%deriv(:, traj%state, traj%state)

    return
  end function tr_force

  !
  ! Computes _just_ the nuclear part of the overlap integral
  !
  function nuc_overlap(bra_t, ket_t) result(S)
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t

    complex(drk)                       :: S
    real(drk)                          :: w1_w2(n_crd), w1w2(n_crd)
    real(drk)                          :: dx(n_crd), dp(n_crd), prefactor(n_crd)
    real(drk)                          :: x_center(n_crd), real_part(n_crd), imag_part(n_crd)

    dx        = bra_t%x - ket_t%x
    dp        = bra_t%p - ket_t%p
    w1_w2     = widths + widths
    w1w2      = widths * widths
    prefactor = sqrt(2. * sqrt(w1w2) / (w1_w2))
    x_center  = (widths * bra_t%x + widths * ket_t%x) / (w1_w2)
    real_part = (w1w2*dx**2 + 0.25*dp**2) / (w1_w2)
    imag_part = (bra_t%x * bra_t%p - ket_t%x * ket_t%p) - x_center * dp
    S         = exp(I * (ket_t%phase - bra_t%phase)) * product(prefactor) * &
                exp(sum(-real_part + I*imag_part))
    
    return
  end function nuc_overlap

  !
  ! Returns the matrix element <cmplx_gaus(q,p)| q^N |cmplx_gaus(q,p)>
  !   -- up to an overlap integral --
  !
  function qn_integral(n, a1, x1, p1, a2, x2, p2) result(qn_int)
    integer(ik), intent(in)      :: n
    real(drk), intent(in)        :: a1, x1, p1
    real(drk), intent(in)        :: a2, x2, p2

    complex(drk)                 :: qn_int
    integer(ik)                  :: i, n_2
    real(drk)                    :: a
    complex(drk)                 :: b   
    integer(ik)                  :: two_ik=2

    n_2    = int(floor(0.5*n))
    a      = a1+a2 
    b      = 2.*(a1*x1 + a2*x2) - I*(p1 - p2)

    qn_int = zero_c

    if (abs(b) < mp_drk) then
      if (mod(n,2) == 0) then
        qn_int = a**(-n_2) * factorial(n) / (factorial(n_2) * 2.**n)
      endif
      return
    endif

    do i = 0,n_2
      qn_int = qn_int + a**(i-n) * b**(n-two_ik*i) / (factorial(i) * factorial(n-two_ik*i))
    enddo
    qn_int = qn_int * factorial(n) / 2.**n

    return
  end function qn_integral

  !
  ! Returns the matrix elements <cmplx_gaus(q1i,p1i)| q^N |cmplx_gaus(q2i,p2i)>
  !
  function qn_vector(n, bra_t, ket_t, S) result(qn_vec)
    integer(ik), intent(in)      :: n
    type(trajectory), intent(in) :: bra_t, ket_t
    complex(drk), intent(in)        :: S

    complex(drk)                 :: qn_vec(n_crd)
    integer(ik)                  :: i

    qn_vec = zero_c

    do i = 1,n_crd
      qn_vec(i) = S * qn_integral(n, widths(i), bra_t%x(i), bra_t%p(i), &
                                     widths(i), ket_t%x(i), ket_t%p(i))
    enddo

    return
  end function qn_vector

  !
  ! computes the size of an array of trajectory objects, returns
  ! zero if array is unallocated
  !
  function traj_size(vector) result(tsize)
    type(trajectory), allocatable, intent(in) :: vector(:)
    integer(ik)                               :: tsize

    if(.not.allocated(vector)) then
      tsize = 0
    else
      tsize  = size(vector)
    endif
     
    return
  end function traj_size

  !
  ! computes the size of an array of complex numbers, returns
  ! zero if the array is unallocat
  ! 
  function amp_size(vector) result(asize)
    complex(drk), allocatable, intent(in)     :: vector(:)
    integer(ik)                               :: asize

    if(.not.allocated(vector)) then
      asize = 0
    else
      asize = size(vector)
    endif
    
    return
  end function amp_size

end module libprop 

