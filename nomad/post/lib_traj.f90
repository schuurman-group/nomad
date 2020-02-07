!
! lib_traj -- a module that defines the data structures to hold trajectory
!             data, as well as trajectory retrieval and interpolation
!
! M. S. Schuurman -- Dec. 27, 2019
!
module lib_traj
  use accuracy
  use math
  implicit none
  private

   public traj_init_table
   public traj_create_table
   public add_trajectory
   public locate_trajectories
   public traj_table_exists
   public n_trajectories
   public n_batches

   public traj_timestep_info
   public traj_retrieve_timestep
   public traj_retrieve_next_timestep
   public traj_next_timestep
   public traj_time_step

   public tr_potential
   public tr_kinetic
   public tr_velocity
   public tr_force
   public ref_overlap
   public nuc_overlap
   public nuc_density
   public qn_integral
   public qn_vector
   public traj_size

   private extract_trajectory
   private interpolate_trajectory
   private get_time_index
   private get_current_time
   private step_current_time

    ! total number of trajectories
    integer(ik)                   :: n_total=0
    ! number of 'batches', or, initial conditions
    integer(ik)                   :: n_batch=0
    ! number of nuclear degrees of freedom
    integer(ik)                   :: n_crd=0
    ! total number of electronic states
    integer(ik)                   :: n_state=0
    ! tolerance for identifying discrete time step with a given time
    real(drk)                     :: dt_toler
    ! polynomial regression order, currently hardwired to quadratic fit
    integer(ik),parameter         :: fit_order=2

    ! allowed time step determination methods
    character(len=7),parameter    :: tstep_methods(1:2) = (/'nascent', 'static '/)
    character(len=7)              :: tstep_method
    real(drk)                     :: default_tstep

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
      complex(drk)                :: amplitude
      real(drk), allocatable      :: mass(:)
      real(drk), allocatable      :: width(:)
      real(drk), allocatable      :: energy(:)
      real(drk), allocatable      :: x(:)
      real(drk), allocatable      :: p(:)
      real(drk), allocatable      :: deriv(:,:,:)
      real(drk), allocatable      :: coup(:,:)
    end type trajectory
    public trajectory

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
      real(drk),allocatable       :: mass(:)
      real(drk),allocatable       :: width(:)
      complex(drk),allocatable    :: amplitude(:)
      real(drk),allocatable       :: time(:)
      real(drk),allocatable       :: phase(:)
      real(drk),allocatable       :: energy(:,:)
      real(drk),allocatable       :: x(:,:)
      real(drk),allocatable       :: p(:,:)
      real(drk),allocatable       :: deriv(:,:,:,:)
      real(drk),allocatable       :: coup(:,:,:)
    end type trajectory_table
    private trajectory_table

    ! this array of trajectory_tables represents _all_ the 
    ! trajectory basis information for a series of times. This
    ! is the 'master' list that is queried as we propagate
    type(trajectory_table), allocatable :: basis_table(:)
    type(trajectory)                    :: ref_traj

 contains

  !********************************************************************************
  !********************************************************************************
  ! Externally invoked subroutines 

  !
  !
  !
  subroutine traj_init_table(default_timestep, t_method) bind(c, name='traj_init_table')
    real(drk), intent(in)           :: default_timestep
    integer(ik), intent(in)         :: t_method

    default_tstep = default_timestep
    dt_toler      = 0.01 * default_tstep
    tstep_method  = tstep_methods(1)

    if(t_method <= size(tstep_methods)) then
      tstep_method = tstep_methods(t_method)
    else
      stop 'time step method not recognized'
    endif

    return
  end subroutine traj_init_table

  !
  !
  !
  subroutine traj_create_table(n_grp, n_tot, n_states, n_dim, ref_width, ref_mass, x, p) bind(c, name='traj_create_table')
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

    ! allocate trajectory-specific quantities
    allocate(ref_traj%mass(n_crd))
    allocate(ref_traj%width(n_crd))
    allocate(ref_traj%x(n_crd))
    allocate(ref_traj%p(n_crd))
 
    ref_traj%mass  = ref_mass
    ref_traj%width = ref_width
    ref_traj%x     = x
    ref_traj%p     = p

    return 
  end subroutine traj_create_table

  !
  ! return 'true' if the trajectory table exists
  !
  function traj_table_exists() result(exists)
    logical                         :: exists

    exists = allocated(basis_table) .and. (n_total > 0)
    exists = exists .and. (n_crd > 0) .and. (n_batch) > 0

    return
  end function traj_table_exists

  ! 
  ! return the total number of trajectories in the table
  ! 
  function n_trajectories() result(n_traj)
    integer(ik)                     :: n_traj

    n_traj = n_total
    return
  end function n_trajectories

  !
  ! return the number of batches
  !
  function n_batches() result(n_bat)
    integer(ik)                     :: n_bat

    n_bat = n_batch
    return
  end function n_batches

  !
  !
  !
  subroutine add_trajectory(batch, np, state, widths, masses, time, phase, energy, x, p, deriv, coup, ampr, ampi) bind(c, name='add_trajectory')
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
    real(drk), intent(in)           :: ampr(np)
    real(drk), intent(in)           :: ampi(np)

    integer(ik), save               :: traj_cnt = 0

    ! Set and allocate the trajectory table
    traj_cnt                          = traj_cnt + 1
    basis_table(traj_cnt)%batch       = batch
    basis_table(traj_cnt)%label       = traj_cnt
    basis_table(traj_cnt)%state       = state+1   ! fortran states go 1:n_state, nomad goes 0:n_state-1
    basis_table(traj_cnt)%nsteps      = np  
    basis_table(traj_cnt)%current_row = 1

    allocate(basis_table(traj_cnt)%width(n_crd))
    allocate(basis_table(traj_cnt)%mass(n_crd))   
    allocate(basis_table(traj_cnt)%time(np))
    allocate(basis_table(traj_cnt)%phase(np))
    allocate(basis_table(traj_cnt)%energy(n_state, np))
    allocate(basis_table(traj_cnt)%x(n_crd, np))
    allocate(basis_table(traj_cnt)%p(n_crd, np))
    allocate(basis_table(traj_cnt)%deriv(n_state, n_state, n_crd, np))
    allocate(basis_table(traj_cnt)%coup(n_state, n_state, np))
    allocate(basis_table(traj_cnt)%amplitude(np))

    basis_table(traj_cnt)%width     = widths
    basis_table(traj_cnt)%mass      = masses
    basis_table(traj_cnt)%time      = time
    basis_table(traj_cnt)%phase     = phase
    basis_table(traj_cnt)%energy    = reshape(energy, (/n_state, np/))
    basis_table(traj_cnt)%x         = reshape(x,      (/n_crd, np/))
    basis_table(traj_cnt)%p         = reshape(p,      (/n_crd, np/))
    basis_table(traj_cnt)%deriv     = reshape(deriv,  (/n_crd, n_state, n_state, np/), (/zero_drk, zero_drk, zero_drk, zero_drk/), (/2, 3, 1, 4/))
    basis_table(traj_cnt)%coup      = reshape(coup,   (/n_state, n_state, np/), (/zero_drk, zero_drk, zero_drk/), (/2, 1, 3/))
    basis_table(traj_cnt)%amplitude = ampr + I_drk*ampi

  end subroutine add_trajectory

  !
  !
  !
  subroutine traj_timestep_info(first_time, batch, time, n_traj, indices) bind(c, name='traj_timestep_info')
    logical, intent(in)                  :: first_time
    integer(ik), intent(in)              :: batch
    real(drk), intent(out)               :: time
    integer(ik), intent(out)             :: n_traj
    integer(ik), intent(out)             :: indices(n_total)

    call get_current_time(first_time, batch, time, n_traj, indices)

    return
  end subroutine traj_timestep_info

  !
  !
  !
  subroutine traj_retrieve_timestep(time, batch, n, states, amp_r, amp_i, eners, x, p, deriv, coup) bind(c, name='traj_retrieve_timestep')
    real(drk), intent(in)                   :: time
    integer(ik), intent(in)                 :: batch
    integer(ik), intent(in)                 :: n
    integer(ik), intent(out)                :: states(n)
    real(drk), intent(out)                  :: amp_r(n)
    real(drk), intent(out)                  :: amp_i(n)
    real(drk), intent(out)                  :: eners(n*n_state)
    real(drk), intent(out)                  :: x(n*n_crd)
    real(drk), intent(out)                  :: p(n*n_crd)
    real(drk), intent(out)                  :: deriv(n*n_crd*n_state*n_state)
    real(drk), intent(out)                  :: coup(n*n_state*n_state)

    integer(ik)                             :: i, dlen, clen
    integer(ik)                             :: traj_labels(n)
    type(trajectory), allocatable,save      :: traj_list(:)

    call locate_trajectories(time, batch, traj_labels, traj_list)

    if(size(traj_list) > n)stop 'cannot retrieve all data in traj_retrieve_timestep, size(traj_list) > n'

    dlen = n_crd*n_state*n_state
    clen = n_state*n_state
    do i = 1,n
      states(i)                        = traj_list(i)%state-1 ! convert back to python numbering
      amp_r(i)                         = real(traj_list(i)%amplitude)
      amp_i(i)                         = real(aimag(traj_list(i)%amplitude))
      eners(n_state*(i-1)+1:n_state*i) = traj_list(i)%energy
      x(n_crd*(i-1)+1:n_crd*i)         = traj_list(i)%x
      p(n_crd*(i-1)+1:n_crd*i)         = traj_list(i)%p
      deriv(dlen*(i-1)+1:dlen*i)       = reshape(traj_list(i)%deriv, shape=(/ dlen /))
      coup(clen*(i-1)+1:clen*i)        = reshape(traj_list(i)%coup, shape=(/ clen /))
    enddo

    return
  end subroutine traj_retrieve_timestep

  !
  !
  ! 
  subroutine traj_retrieve_next_timestep(n, indices, states, amp_r, amp_i, eners, x, p, deriv, coup) bind(c, name='traj_retrieve_next_timestep')
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

    integer(ik)                             :: i, t_indx, dlen, clen 

    dlen = n_crd*n_state*n_state
    clen = n_state*n_state
    do i = 1,n
      t_indx                           = basis_table(indices(i))%current_row
      states(i)                        = basis_table(indices(i))%state-1 ! convert back to python numbering
      eners(n_state*(i-1)+1:n_state*i) = basis_table(indices(i))%energy(:,t_indx)
      amp_r(i)                         = real(basis_table(indices(i))%amplitude(t_indx))
      amp_i(i)                         = real(aimag(basis_table(indices(i))%amplitude(t_indx)))
      x(n_crd*(i-1)+1:n_crd*i)         = basis_table(indices(i))%x(:,t_indx)
      p(n_crd*(i-1)+1:n_crd*i)         = basis_table(indices(i))%p(:,t_indx)
      deriv(dlen*(i-1)+1:dlen*i)       = reshape(basis_table(indices(i))%deriv(:,:,:,t_indx), shape=(/ dlen /))
      coup(clen*(i-1)+1:clen*i)        = reshape(basis_table(indices(i))%coup(:,:,t_indx), shape=(/ clen /))
    enddo

    return
  end subroutine traj_retrieve_next_timestep

  !
  !
  !
  subroutine traj_next_timestep(n, indices, done) bind(c, name='traj_next_timestep')
    integer(ik), intent(in)                 :: n
    integer(ik), intent(in)                 :: indices(n)
    logical, intent(out)                    :: done

    done = step_current_time(n, indices)

    return
  end subroutine traj_next_timestep

  !
  !
  !
  function traj_time_step(batch, current_time) result (new_time)
    integer(ik), intent(in)                 :: batch
    real(drk), intent(in)                   :: current_time

    real(drk)                               :: new_time
    logical                                 :: done
    integer(ik)                             :: n_traj
    integer(ik)                             :: indices(n_total)

    select case(tstep_method)
      case('nascent')
        call get_current_time(.false., batch, new_time, n_traj, indices)
        if(new_time > current_time)call get_current_time(.true., batch, new_time, n_traj, indices)
        done = .false.
        ! iterate until new_time is larger than current_time
        do while(new_time-current_time <= mp_drk .and. .not.done)
          done = step_current_time(n_traj, indices)
          call get_current_time(.false., batch, new_time, n_traj, indices)
        enddo
        ! if we've reached the end of the time array, or, we've hit a padded zero, return 
        ! the default_tstep to ensure proper ending behavior
        if(done) new_time = current_time + default_tstep

      case('static ')
        new_time = current_time + default_tstep

      end select

      return
  end function traj_time_step

  !
  !
  !
  subroutine locate_trajectories(time, batch, labels, traj_list)
    real(drk), intent(in)                      :: time
    integer(ik), intent(in)                    :: batch
    integer(ik), intent(out)                   :: labels(:)
    type(trajectory), intent(out), allocatable :: traj_list(:)

    integer(ik)                                :: i
    integer(ik)                                :: n_fnd
    integer(ik)                                :: time_indx
    real(drk)                                  :: time_val

    ! go through table once to figure out how many trajectories
    ! exist at the requested time
    n_fnd = 0
    do i = 1,size(basis_table)
      ! skip if trajectory(i) is in wrong batch
      if(batch > 0 .and. basis_table(i)%batch /= batch) continue
      call get_time_index(i, time, time_indx, time_val)
      if(time_indx /= -1) then
        n_fnd = n_fnd + 1
        if(size(labels)>=n_fnd) labels(n_fnd) = i
      endif
    enddo

    ! for now, fail catastropically if we can't store all the traj labels
    if(size(labels) < n_fnd) stop 'size(labels) < n_fnd in locate_trajectories'

    ! now set traj_list to the right size and populate with the 
    ! trajectories
    if(allocated(traj_list)) then
      if(size(traj_list) /= n_fnd) then
        do i = 1,n_fnd
          deallocate(traj_list(i)%x,traj_list(i)%p)
          deallocate(traj_list(i)%mass, traj_list(i)%width)
          deallocate(traj_list(i)%energy,traj_list(i)%deriv,traj_list(i)%coup)
        enddo
        deallocate(traj_list)
      endif
    endif

    ! create the trajectory list
    if(.not.allocated(traj_list)) then
      allocate(traj_list(n_fnd))
      do i = 1,n_fnd 
        allocate(traj_list(i)%mass(n_crd))
        allocate(traj_list(i)%width(n_crd))
        allocate(traj_list(i)%x(n_crd))
        allocate(traj_list(i)%p(n_crd))
        allocate(traj_list(i)%energy(n_state))
        allocate(traj_list(i)%deriv(n_crd,n_state,n_state))
        allocate(traj_list(i)%coup(n_state,n_state))
      enddo
    endif

    ! now go through and add entries to the trajectory list
    do i = 1,n_fnd

      call get_time_index(labels(i), time, time_indx, time_val)

      if(time_indx /= -1) then
        basis_table(labels(i))%current_row = time_indx
        if(abs(time_val-time) <= dt_toler) then
          call extract_trajectory(labels(i), time_indx, traj_list(i))
        else
          call interpolate_trajectory(labels(i), time, time_indx, traj_list(i))
        endif
      endif

    enddo

    return
  end subroutine locate_trajectories

  !*******************************************************************************
  !*******************************************************************************
  ! Private routines and functions
  !

  !
  !
  !
  subroutine extract_trajectory(table_indx, time_indx, traj)
    integer(ik), intent(in)           :: table_indx
    integer(ik), intent(in)           :: time_indx
    type(trajectory), intent(out)     :: traj

    traj%batch  = basis_table(table_indx)%batch
    traj%label  = basis_table(table_indx)%label
    traj%state  = basis_table(table_indx)%state

    traj%mass   = basis_table(table_indx)%mass
    traj%width  = basis_table(table_indx)%width
    traj%time   = basis_table(table_indx)%time(time_indx)
    traj%phase  = basis_table(table_indx)%phase(time_indx)
    traj%energy = basis_table(table_indx)%energy(:,time_indx)
    traj%x      = basis_table(table_indx)%x(:,time_indx)
    traj%p      = basis_table(table_indx)%p(:,time_indx)
    traj%deriv  = basis_table(table_indx)%deriv(:,:,:,time_indx)
    traj%coup   = basis_table(table_indx)%coup(:,:,time_indx)    

    return
  end subroutine extract_trajectory

  !
  ! does polynomial interpolation
  !
  subroutine interpolate_trajectory(table_indx, time, time_indx, traj)
    integer(ik), intent(in)           :: table_indx    ! index of the trajectory
    real(drk), intent(in)             :: time          ! the requested time
    integer(ik), intent(in)           :: time_indx     ! closest time in table
    type(trajectory), intent(out)     :: traj

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

    traj%batch  = basis_table(table_indx)%batch
    traj%label  = basis_table(table_indx)%label
    traj%state  = basis_table(table_indx)%state
    traj%mass   = basis_table(table_indx)%mass
    traj%width  = basis_table(table_indx)%width
    traj%time   = time

    traj%phase  = poly_fit(      fit_order, time, basis_table(table_indx)%time(tbnd1:tbnd2), basis_table(table_indx)%phase(tbnd1:tbnd2))
    traj%energy = poly_fit_array(fit_order, time, basis_table(table_indx)%time(tbnd1:tbnd2), basis_table(table_indx)%energy(:,tbnd1:tbnd2))
    traj%x      = poly_fit_array(fit_order, time, basis_table(table_indx)%time(tbnd1:tbnd2), basis_table(table_indx)%x(:,tbnd1:tbnd2))
    traj%p      = poly_fit_array(fit_order, time, basis_table(table_indx)%time(tbnd1:tbnd2), basis_table(table_indx)%p(:,tbnd1:tbnd2))
    do i = 1,n_state
      do j = 1,i
        traj%deriv(:,i,j) = poly_fit_array(fit_order, time, basis_table(table_indx)%time(tbnd1:tbnd2), basis_table(table_indx)%deriv(:,i,j,tbnd1:tbnd2))
        traj%coup(i,j)    = poly_fit(      fit_order, time, basis_table(table_indx)%time(tbnd1:tbnd2), basis_table(table_indx)%coup(i,j,tbnd1:tbnd2))
        if(i /= j) then
          traj%deriv(:,j,i) = -traj%deriv(:,i,j)
          traj%coup(j,i)    = -traj%coup(i,j)
        endif
      enddo
    enddo

    return
  end subroutine interpolate_trajectory

  !
  !
  !
  subroutine get_current_time(first_time, batch, time, n_fnd, indices)
    logical, intent(in)              :: first_time
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
    do i = 1,size(basis_table)

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

    return
  end subroutine get_current_time

  !
  ! Iterate all current_row pointers which are <= value of time by "1"
  !
  function step_current_time(n_step, indices) result(done)
    integer(ik), intent(in)          :: n_step
    integer(ik), intent(in)          :: indices(:)
 
    integer(ik)                      :: i
    logical                          :: done
 
    if(size(indices) < n_step) stop 'ERROR in step_current_time: n_step > size(indices)'
    done = .true.

    do i = 1,n_step
      if(basis_table(indices(i))%current_row < basis_table(indices(i))%nsteps) then
        basis_table(indices(i))%current_row = basis_table(indices(i))%current_row + 1
        done = .false.
      endif
    enddo

    return
  end function step_current_time


  !
  !
  !
  subroutine get_time_index(indx, time, fnd_indx, fnd_time)
    integer(ik), intent(in)          :: indx
    real(drk), intent(in)            :: time
    integer(ik), intent(out)         :: fnd_indx
    real(drk), intent(out)           :: fnd_time

    real(drk)                        :: current_time, dt, dt_new

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

    return
  end subroutine get_time_index

  !**********************************************************************************************
  !**********************************************************************************************
  ! Numerical routines for computing integrals, trajectory properties
  !

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

    kinetic = sum( 0.5 * traj%p * traj%p / traj%mass)

    return
  end function tr_kinetic

  !
  !
  !
  function tr_velocity(traj) result(velocity)
    type(trajectory), intent(in)      :: traj

    real(drk)                         :: velocity(n_crd)

    velocity = traj%p / traj%mass 

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
  ! compute the overlap with the reference trajectory
  !
  function ref_overlap(bra_t) result(S)
    type(trajectory), intent(in)       :: bra_t
    
    complex(drk)                       :: S

    S = nuc_overlap(bra_t, ref_traj)
    return
  end function ref_overlap

  !
  ! Computes _just_ the nuclear part of the overlap integral
  !
  function nuc_overlap(bra_t, ket_t) result(S)
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t

    complex(drk)                       :: S
    real(drk)                          :: a1(n_crd)
    real(drk)                          :: a2(n_crd)
    real(drk)                          :: dx(n_crd)
    real(drk)                          :: dp(n_crd)
    real(drk)                          :: prefactor(n_crd)
    real(drk)                          :: x_center(n_crd)
    real(drk)                          :: real_part(n_crd)
    real(drk)                          :: imag_part(n_crd)

    a1        = bra_t%width
    a2        = ket_t%width
    dx        = bra_t%x - ket_t%x
    dp        = bra_t%p - ket_t%p
    prefactor = sqrt(2. * sqrt(a1*a2) / (a1+a2))
    x_center  = (a1 * bra_t%x + a2 * ket_t%x) / (a1+a2)
    real_part = (a1*a2*dx**2 + 0.25*dp**2) / (a1+a2)
    imag_part = (bra_t%x * bra_t%p - ket_t%x * ket_t%p) - x_center * dp
    S         = exp(I_drk * (ket_t%phase - bra_t%phase)) * product(prefactor) * &
                exp(sum(-real_part + I_drk*imag_part))
    
    return
  end function nuc_overlap

  !
  !
  !
  function nuc_density(bra_t, ket_t, pt) result(den)
    type(trajectory), intent(in)        :: bra_t
    type(trajectory), intent(in)        :: ket_t
    real(drk)                           :: pt(n_crd)
   
    complex(drk)                        :: den
    complex(drk)                        :: den_vec(n_crd)
    real(drk)                           :: argr(n_crd)
    real(drk)                           :: argi(n_crd)
    real(drk)                           :: a1(n_crd)
    real(drk)                           :: a2(n_crd)
 
    a1      = bra_t%width
    a2      = ket_t%width
    argr    = a1*(pt - bra_t%x)**2   + a2*(pt - ket_t%x)**2
    argi    = ket_t%p*(pt - ket_t%x) - bra_t%p*(pt - bra_t%x)
    den_vec = sqrt(2.*a1/pi) * exp(-argr + I_drk*argi)
    den     = product(den_vec)

    return
  end function nuc_density


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
      qn_vec(i) = S * qn_integral(n, bra_t%width(i), bra_t%x(i), bra_t%p(i), &
                                     ket_t%width(i), ket_t%x(i), ket_t%p(i))
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

end module lib_traj

