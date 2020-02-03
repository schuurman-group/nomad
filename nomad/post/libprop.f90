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
   public timestep_info
   public retrieve_timestep
   public next_timestep

   public collect_trajectories
   public locate_trajectory
   public extract_trajectory
   public interpolate_trajectory
   public locate_amplitude
   public update_amplitude
   public propagate_amplitude
   public normalize_amplitude

   public get_time_index
   public get_current_time
   public step_current_time

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
    ! polynomial regression order, currently hardwired to quadratic fit
    integer(ik),parameter         :: fit_order=2

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
      real(drk), allocatable      :: coup(:,:)
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
      real(drk),allocatable       :: coup(:,:,:)
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
  subroutine add_trajectory(batch, np, state, widths, masses, time, phase, energy, x, p, deriv, coup) bind(c, name='add_trajectory')
    integer(ik), intent(in)         :: batch
    integer(ik), intent(in)         :: np !number of time steps
    integer(ik), intent(in)         :: state
    real(drk), intent(in)           :: widths(n_crd)
    real(drk), intent(in)           :: masses(n_crd)
    real(drk), intent(in)           :: time(np)
    real(drk), intent(in)           :: phase(np)
    real(drk), intent(in)           :: energy(np*n_state), x(np*n_crd), p(np*n_crd)
    real(drk), intent(in)           :: deriv(np*n_crd*n_state*n_state)
    real(drk), intent(in)           :: coup(np*n_state*n_state)

    integer(ik), save               :: traj_cnt = 0
    integer(ik)                     :: n_amp

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
    allocate(basis_table(traj_cnt)%coup(n_state, n_state, np))

    basis_table(traj_cnt)%widths    = widths
    basis_table(traj_cnt)%masses    = masses
    basis_table(traj_cnt)%time      = time
    basis_table(traj_cnt)%phase     = phase
    basis_table(traj_cnt)%energy    = reshape(energy, (/n_state, np/))
    basis_table(traj_cnt)%x         = reshape(x,      (/n_crd, np/))
    basis_table(traj_cnt)%p         = reshape(p,      (/n_crd, np/))
    basis_table(traj_cnt)%deriv     = reshape(deriv,  (/n_crd, n_state, n_state, np/), (/zero_drk, zero_drk, zero_drk, zero_drk/), (/2, 3, 1, 4/))
    basis_table(traj_cnt)%coup      = reshape(coup,   (/n_state, n_state, np/), (/zero_drk, zero_drk, zero_drk/), (/2, 1, 3/))

    ! allocate the correpsonding amplitude table
    ! for now, calculate how many slots given constant time step. This is not optimal, will fix later
    n_amp = ceiling( (time(size(time))-time(1)) / t_step)+10
    allocate(amp_table(traj_cnt)%time(n_amp))
    allocate(amp_table(traj_cnt)%amplitude(n_amp))
    amp_table(traj_cnt)%time        = -1.
    amp_table(traj_cnt)%batch       = batch
    amp_table(traj_cnt)%label       = traj_cnt
    amp_table(traj_cnt)%current_row = 1
    amp_table(traj_cnt)%amplitude   = zero_c

  end subroutine add_trajectory

  !
  !
  !
  subroutine timestep_info(first_time, batch, time, n_traj, indices) bind(c, name='timestep_info')
    logical, intent(in)                  :: first_time
    integer(ik), intent(in)              :: batch
    real(drk), intent(out)               :: time
    integer(ik), intent(out)             :: n_traj
    integer(ik), intent(out)             :: indices(n_total)

    call get_current_time(first_time, 'amps', batch, time, n_traj, indices)

    return
  end subroutine timestep_info

  !
  !
  ! 
  subroutine retrieve_timestep(batch, n, indices, states, amp_r, amp_i, eners, x, p, deriv, coup) bind(c, name='retrieve_timestep')
    integer(ik), intent(in)                 :: batch
    integer(ik), intent(in)                 :: n
    integer(ik), intent(in)                 :: indices(n)
    integer(ik), intent(out)                :: states(n)
    real(drk), intent(out)                  :: amp_r(n)
    real(drk), intent(out)                  :: amp_i(n)
    real(drk), intent(out)                  :: eners(n*n_state)
    real(drk), intent(out)                  :: x(n*n_crd)
    real(drk), intent(out)                  :: p(n*n_crd)
    real(drk), intent(out)                  :: deriv(n*n_crd*n_state*n_state)
    real(drk), intent(out)                  :: coup(n*n_state*n_state)

    integer(ik)                             :: i, row, dlen, clen
    real(drk)                               :: time
    integer(ik), allocatable                :: labels(:)

    do i = 1,n
      row      = amp_table(indices(i))%current_row
      time     = amp_table(indices(i))%time(row)
      amp_r(i) = real(amp_table(indices(i))%amplitude(row))
      amp_i(i) = aimag(amp_table(indices(i))%amplitude(row))
    enddo

    call collect_trajectories(time, batch, labels)

    if(tstep_cnt /= n) stop 'disagreement in retrieve_timestep n /= tstep_cnt'

    dlen = n_crd*n_state*n_state
    clen = n_state*n_state
    do i = 1,tstep_cnt
      if(indices(i) /= traj_list(i)%label) stop 'indices(i) /= label: unpredictable results possible'
      states(i)                        = traj_list(i)%state-1 ! convert back to python numbering
      eners(n_state*(i-1)+1:n_state*i) = traj_list(i)%energy
      x(n_crd*(i-1)+1:n_crd*i)         = traj_list(i)%x
      p(n_crd*(i-1)+1:n_crd*i)         = traj_list(i)%p
      deriv(dlen*(i-1)+1:dlen*i)       = reshape(traj_list(i)%deriv, shape=(/ dlen /))
      coup(clen*(i-1)+1:clen*i)        = reshape(traj_list(i)%coup, shape=(/ clen /))
    enddo

    return
  end subroutine retrieve_timestep

  !
  !
  !
  subroutine next_timestep(n, indices, done) bind(c, name='next_timestep')
    integer(ik), intent(in)                 :: n
    integer(ik), intent(in)                 :: indices(n)
    logical, intent(out)                    :: done

    done = step_current_time('amps', n, indices)

    return
  end subroutine next_timestep


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
    integer(ik)                       :: max_traj
    integer(ik)                       :: time_indx
    real(drk)                         :: time_val

    max_traj  = traj_size(traj_list)
    tstep_cnt = 0 

    do i = 1,n_total
  
      ! if we're averaging over batch, we'll select on what
      ! initial condition we're propagating   
      if(batch > 0 .and. basis_table(i)%batch /= batch) continue  

      call get_time_index('traj', i, time, time_indx, time_val)

      if(time_indx /= -1) then
        basis_table(i)%current_row = time_indx
        tstep_cnt                  = tstep_cnt + 1
        if(abs(time_val-time) <= dt_toler) then
          if(tstep_cnt <= max_traj)call extract_trajectory(i, time_indx, tstep_cnt)
        else
          if(tstep_cnt <= max_traj)call interpolate_trajectory(i, time, time_indx, tstep_cnt)
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
    traj_list(indx)%coup   = basis_table(table_indx)%coup(:,:,t_row)    

    return
  end subroutine extract_trajectory

  !
  ! does polynomial interpolation
  !
  subroutine interpolate_trajectory(table_indx, time, time_indx, list_indx)
    integer(ik), intent(in)           :: table_indx    ! index of the trajectory
    real(drk), intent(in)             :: time          ! the requested time
    integer(ik), intent(in)           :: time_indx     ! closest time in table
    integer(ik), intent(in)           :: list_indx     ! interpolated trajectory

    integer(ik)                       :: sgn, dx1, dx2, tbnd1, tbnd2 
    integer(ik)                       :: i,j

    ! Polynomial regression using fit_order+1 points centered about the
    ! nearest time point 
    sgn   = int(sign(1.,time - basis_table(table_indx)%time(time_indx)))
    ! if fit_order is odd, take extra point in direction of requested time, i.e. sgn(time-t0)
    dx1   =  sgn*ceiling(0.5*fit_order)
    dx2   = -sgn*floor(0.5*fit_order)
    if(time_indx+dx1 > basis_table(table_indx)%nsteps .or. &
       time_indx+dx2 > basis_table(table_indx)%nsteps) then
       dx1 = dx1 + min(0, basis_table(table_indx)%nsteps - (time_indx+dx2))
       dx2 = dx2 + min(0, basis_table(table_indx)%nsteps - (time_indx+dx1))
    endif
    if(time_indx+dx1 < 1 .or. time_indx+dx2 < 1) then
       dx1 = dx1 - min(0, 1 - (time_indx+dx2))
       dx2 = dx2 - min(0, 1 - (time_indx+dx1))
    endif
    tbnd1 = max(min(time_indx+dx1, time_indx+dx2),1)
    tbnd2 = min(max(time_indx+dx1, time_indx+dx2), basis_table(table_indx)%nsteps)

    if(tbnd2-tbnd1 /= fit_order) then
      stop 'insufficient data to interpolate trajectory'
    endif

    traj_list(list_indx)%batch  = basis_table(table_indx)%batch
    traj_list(list_indx)%label  = basis_table(table_indx)%label
    traj_list(list_indx)%state  = basis_table(table_indx)%state
    traj_list(list_indx)%time   = time

    traj_list(list_indx)%phase  = poly_fit(      fit_order, time, basis_table(table_indx)%time(tbnd1:tbnd2), basis_table(table_indx)%phase(tbnd1:tbnd2))
    traj_list(list_indx)%energy = poly_fit_array(fit_order, time, basis_table(table_indx)%time(tbnd1:tbnd2), basis_table(table_indx)%energy(:,tbnd1:tbnd2))
    traj_list(list_indx)%x      = poly_fit_array(fit_order, time, basis_table(table_indx)%time(tbnd1:tbnd2), basis_table(table_indx)%x(:,tbnd1:tbnd2))
    traj_list(list_indx)%p      = poly_fit_array(fit_order, time, basis_table(table_indx)%time(tbnd1:tbnd2), basis_table(table_indx)%p(:,tbnd1:tbnd2))
    do i = 1,n_state
      do j = 1,i
        traj_list(list_indx)%deriv(:,i,j) = poly_fit_array(fit_order, time, basis_table(table_indx)%time(tbnd1:tbnd2), basis_table(table_indx)%deriv(:,i,j,tbnd1:tbnd2))
        traj_list(list_indx)%coup(i,j)    = poly_fit(      fit_order, time, basis_table(table_indx)%time(tbnd1:tbnd2), basis_table(table_indx)%coup(i,j,tbnd1:tbnd2))
        if(i /= j) then
          traj_list(list_indx)%deriv(:,j,i) = -traj_list(list_indx)%deriv(:,i,j)
          traj_list(list_indx)%coup(j,i)    = -traj_list(list_indx)%coup(i,j)
        endif
      enddo
    enddo

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
      allocate(traj_list(i_traj)%coup(n_state, n_state))
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
      deallocate(traj_list(i_traj)%coup)
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
          amps(i)   = nuc_overlap(traj_list(i), ref_traj)
        enddo

      case('uniform ')
        amps = one_c

      case('explicit')
        amps = one_c

    end select

    amps = normalize_amplitude(amps)
    call update_amplitude(time, amps)

    return
  end function init_amplitude
 
  !
  !
  ! 
  function update_basis(new_label, old_label, c) result(c_new)
    integer(ik),intent(in)                  :: new_label(:)
    integer(ik), allocatable, intent(in)    :: old_label(:)
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
    B = -I_drk * H * (t_end - t_start)
    call expm_complex(n, B, U)
    amps = matmul(U, amps)
    new_amp = amps(:n,1)

    return
  end function propagate_amplitude

  !
  !
  !
  function locate_amplitude(time) result(amps)
    real(drk), intent(in)             :: time

    complex(drk)                      :: amps(tstep_cnt)
    integer(ik)                       :: i 
    integer(ik)                       :: table_indx
    integer(ik)                       :: time_indx
    real(drk)                         :: time_val


    ! for now, we need the dimension of the amplitude vector to
    ! be the same as the dimension of the trajectory list. So, we
    ! assume that list is accurate and pull trajectory labels from there.
    do i = 1,size(traj_list)

      table_indx = traj_list(i).label
      call get_time_index('amps', table_indx, time, time_indx, time_val)

      ! if time step doesn't exist, set amplitude to zero
      if(time_indx == -1) then
        amps(i) = zero_c

      ! if we have exactly this time step, use it
      elseif(abs(time_val - time) <= dt_toler) then
        amps(i) = amp_table(traj_list(i).label)%amplitude(time_indx)
  
      ! else, obtain amplitude via interpolation
      else
        amps(i) = interpolate_amplitude(traj_list(i).label, time, time_indx)

      endif
    enddo

    return
  end function locate_amplitude

  !
  !
  !
  function interpolate_amplitude(amp_indx, time, time_indx) result(new_amp)
    integer(ik), intent(in)          :: amp_indx
    real(drk), intent(in)            :: time
    integer(ik), intent(in)          :: time_indx

    complex(drk)                     :: new_amp
    integer(ik)                      :: sgn, dx, tbnd1, tbnd2

    ! Polynomial regression using fit_order+1 points centered about the
    ! nearest time point 
    sgn   = int(sign(1.,time - amp_table(amp_indx)%time(time_indx)))
    ! if fit_order is odd, take extra point in direction of requested time, i.e. sgn(time-t0)
    dx    =  sgn*ceiling(0.5*fit_order)
    tbnd1 = min(max(1, time_indx+dx), size(amp_table(amp_indx)%time))
    dx    = -sgn*abs(fit_order - abs(time_indx-tbnd1))
    tbnd2 = min(max(1, time_indx+dx), size(amp_table(amp_indx)%time))

    if(abs(tbnd1 - tbnd2) /= fit_order)stop 'insufficient data to interpolate amplitude'

    new_amp = poly_fit(fit_order, time, amp_table(amp_indx)%time(tbnd1:tbnd2), amp_table(amp_indx)%amplitude(tbnd1:tbnd2))

    return
  end function interpolate_amplitude

  !
  !
  !
  subroutine update_amplitude(time, amplitudes)
    real(drk), intent(in)            :: time
    complex(drk),intent(in)          :: amplitudes(:)

    integer(ik)                      :: i, table_indx
    integer(ik)                      :: time_indx
    real(drk)                        :: time_val

    do i=1,tstep_cnt    
      table_indx = traj_list(i)%label
      call get_time_index('amps', table_indx, time, time_indx, time_val)
      if(time_indx == -1) stop 'Cannot update amplitude for requested time'
      amp_table(table_indx)%current_row          = time_indx
      amp_table(table_indx)%time(time_indx)      = time
      amp_table(table_indx)%amplitude(time_indx) = amplitudes(i)
    enddo

    return
  end subroutine update_amplitude

  !
  !
  !
  subroutine get_current_time(first_time, list_type, batch, time, n_fnd, indices)
    logical, intent(in)              :: first_time
    character(len=4), intent(in)     :: list_type
    integer(ik), intent(in)          :: batch
    real(drk), intent(out)           :: time
    integer(ik), intent(out)         :: n_fnd
    integer(ik), intent(out)         :: indices(:)

    integer(ik)                      :: i
    integer(ik)                      :: n_test
    integer(ik)                      :: i_max
    integer(ik)                      :: index_test(n_total)

    i_max  = size(indices)
    time   = -1.
    n_fnd  = 0
    n_test = 0
    select case(list_type)

      case('traj')
        do i = 1,n_total

          ! if this is not selected batch, move on to next one
          if(batch > 0 .and. basis_table(i)%batch /= batch) continue
          
          ! if setting counters to first index
          if(first_time) basis_table(i)%current_row = 1

          ! add this indx to indices to check after time established
          n_test             = n_test + 1
          index_test(n_test) = i 

          ! determine what the time is for the next step by taking the mininum of the
          ! current times of from the eligible trajectories
          if(time < 0 .and. basis_table(i)%current_row <= basis_table(i)%nsteps) then
            time = basis_table(i)%time(basis_table(i)%current_row)
          else
            time = min(time, basis_table(i)%time(basis_table(i)%current_row))
          endif
        enddo

        do i = 1,n_test
          if( abs(basis_table(index_test(i))%time(basis_table(index_test(i))%current_row)-time) < mp_drk) then
            n_fnd          = n_fnd + 1
            if(n_fnd <= i_max)indices(n_fnd) = index_test(i)
          endif
        enddo 

      case('amps')
        do i = 1,n_total

          ! if this is not selected batch, move on to next one
          if(batch > 0 .and. amp_table(i)%batch /= batch) continue

          ! if setting counters to first index
          if(first_time) amp_table(i)%current_row = 1

          ! add this indx to idices to check after time established
          n_test             = n_test + 1
          index_test(n_test) = i

          ! determine what the time is for the next step by taking the mininum of the
          ! current times of from the eligible trajectories
          if(time < 0. .and. amp_table(i)%current_row <= size(amp_table(i)%time)) then
            time = amp_table(i)%time(amp_table(i)%current_row)
          elseif(amp_table(i)%time(amp_table(i)%current_row) >= 0.)then
            time = min(time, amp_table(i)%time(amp_table(i)%current_row))
          endif
        enddo

        if(time >= 0.)then
          do i = 1,n_test
            if( abs(amp_table(index_test(i))%time(amp_table(index_test(i))%current_row)-time) < mp_drk) then
              n_fnd          = n_fnd + 1
              if(n_fnd <= i_max)indices(n_fnd) = index_test(i)
            endif
          enddo
        endif 

      end select

      return
  end subroutine get_current_time

  !
  ! Iterate all current_row pointers which are <= value of time by "1"
  !
  function step_current_time(list_type, n_step, indices) result(done)
    character(len=4), intent(in)     :: list_type
    integer(ik), intent(in)          :: n_step
    integer(ik), intent(in)          :: indices(:)
 
    integer(ik)                      :: i
    logical                          :: done
 
    if(size(indices) < n_step) stop 'ERROR in step_current_time: n_step > size(indices)'
    done = .true.

    select case(list_type)
      case('traj')
        do i = 1,n_step
          if(basis_table(indices(i))%current_row < basis_table(indices(i))%nsteps) then
            basis_table(indices(i))%current_row = basis_table(indices(i))%current_row + 1
            done = .false.
          endif
        enddo

      case('amps')
        do i = 1,n_step
          if(amp_table(indices(i))%current_row < size(amp_table(indices(i))%time)) then
            amp_table(indices(i))%current_row = amp_table(indices(i))%current_row + 1
            done = .false.
          endif
        enddo
    end select
    
    return
  end function step_current_time


  !
  !
  !
  subroutine get_time_index(list_type, indx, time, fnd_indx, fnd_time)
    character(len=4), intent(in)     :: list_type
    integer(ik), intent(in)          :: indx
    real(drk), intent(in)            :: time
    integer(ik), intent(out)         :: fnd_indx
    real(drk), intent(out)           :: fnd_time

    real(drk), allocatable           :: time_buf(:)
    complex(drk), allocatable        :: amp_buf(:)
    real(drk)                        :: current_time, dt, dt_new

    select case(list_type)

      case('traj')
        ! to save searching through the table over and over again, 
        ! we'll use the previously accessed row as a starting guess
        current_time = basis_table(indx)%time(basis_table(indx)%current_row)
        if(current_time <= time)then
          fnd_indx = basis_table(indx)%current_row
        else
          fnd_indx = 1
        endif

        ! search the table for the requested time
        ! this returns the time closest to the request time       
        fnd_time = basis_table(indx)%time(fnd_indx)
        dt       = abs(fnd_time-time) 
        do
          if(fnd_indx == basis_table(indx)%nsteps .or. dt <= dt_toler)exit
          dt_new   = abs(basis_table(indx)%time(fnd_indx+1)-time)
          if(dt_new > dt) exit
          fnd_indx = fnd_indx + 1
          fnd_time = basis_table(indx)%time(fnd_indx)
          dt       = dt_new 
        enddo 

        ! if the request time sits outside the range of available times
        ! return a fnd_indx of -1
        if(fnd_indx == 1 .and. fnd_time > time) fnd_indx = -1
        if(fnd_indx == basis_table(indx)%nsteps .and. fnd_time < time)fnd_indx = -1

      case('amps')

        ! to save searching through the table over and over again, 
        ! we'll use the previously accessed row as a starting guess
        current_time = amp_table(indx)%time(amp_table(indx)%current_row)
        if(current_time <= time .and. current_time /= -1.) then
          fnd_indx = amp_table(indx)%current_row
        else
          fnd_indx = 1
        endif

        ! search the table for the requested time
        fnd_time = amp_table(indx)%time(fnd_indx)
        dt = abs(fnd_time-time)
        do 
          if(fnd_indx == size(amp_table(indx)%time) .or. dt <= dt_toler .or. fnd_time==-1.) exit
          dt_new   = abs(amp_table(indx)%time(fnd_indx+1)-time)
          if(dt_new > dt .and. amp_table(indx)%time(fnd_indx+1) /= -1.) exit
          fnd_indx = fnd_indx + 1
          fnd_time = amp_table(indx)%time(fnd_indx)
          dt       = dt_new
        enddo

        ! an unset time of "-1" is always allowed. If this is what is found, exit
        if(fnd_time /= -1.) then

          ! if fnd_indx is bigger than the amplitude table, expand the table
          if(fnd_indx == size(amp_table(indx)%time) .and. fnd_time < time) then
            allocate(time_buf(size(amp_table(indx)%time)))
            allocate(amp_buf(size(amp_table(indx)%amplitude)))
            time_buf = amp_table(indx)%time
            amp_buf  = amp_table(indx)%amplitude
            deallocate(amp_table(indx)%time)
            deallocate(amp_table(indx)%amplitude)
            allocate(amp_table(indx)%time(floor(1.5*size(time_buf))))
            allocate(amp_table(indx)%amplitude(floor(1.5*size(amp_buf))))
            amp_table(indx)%time                       = -1.
            amp_table(indx)%amplitude                  = zero_c
            amp_table(indx)%time(1:size(time_buf))     = time_buf
            amp_table(indx)%amplitude(1:size(amp_buf)) = amp_buf
            fnd_indx = fnd_indx + 1
            fnd_time = amp_table(indx)%time(fnd_indx)
            deallocate(time_buf, amp_buf)
          endif

          ! if the requested time is before the initial time, return fnd_indx. If time is
          ! larger than largest existing time, return index of next open time slot.
          if(fnd_indx == 1 .and. fnd_time > time) fnd_indx = -1
        endif        

    end select

    return
  end subroutine get_time_index

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
    S         = exp(I_drk * (ket_t%phase - bra_t%phase)) * product(prefactor) * &
                exp(sum(-real_part + I_drk*imag_part))
    
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
    b      = 2.*(a1*x1 + a2*x2) - I_drk*(p1 - p2)

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

