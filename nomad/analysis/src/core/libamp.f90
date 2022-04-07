!
! lib_amp-- a module that defines the data structures to hold amplitude
!            data, as well as amplitude retrieval and interpolation
!
! M. S. Schuurman -- Dec. 27, 2019
!
module libamp 
  use accuracy
  use math
  use timer
  implicit none
  private

   public amp_init_table
   public amp_create_table
   public amp_timestep_info
   public amp_retrieve_timestep
   public amp_next_timestep
   public amp_table_exists

   public init_method_code
   public init_amplitude
   public update_basis
   public update_amplitude
   public locate_amplitude
   public propagate_amplitude
   public normalize_amplitude

   private interpolate_amplitude
   private get_time_index
   private get_current_time
   private step_current_time
   private amp_size

    ! total number of basis functions at all times
    integer(ik)                   :: amp_total = 0
    ! tolerance for identifying discrete time step with a given time
    real(drk)                     :: dt_toler
    ! polynomial regression order, currently hardwired to quadratic fit
    integer(ik),parameter         :: fit_order=2

    ! allowed amplitude initialization schemes
    character(len=8),parameter    :: init_methods(1:3) = (/ 'overlap ','uniform ','explicit' /)

    ! allowed time step determination methods
    character(len=7),parameter    :: tstep_methods(1:2) = (/'nascent', 'static '/)
    character(len=7)              :: tstep_method
    real(drk)                     :: default_tstep

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

    ! this array of amplitude_tables represents _all_ the 
    ! trajectory basis information for a series of times. This
    ! is the 'master' list that is queried as we propagate
    type(amplitude_table), allocatable  :: amp_table(:)

 contains

  !********************************************************************************
  !********************************************************************************
  ! Externally invoked subroutines 
  ! 

  !
  !
  !
  subroutine amp_init_table(default_timestep, t_method) bind(c,name='amp_init_table')
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
  end subroutine amp_init_table

  !
  !
  !
  subroutine amp_create_table(n_grp, n_st, n_total, batch_labels, n_steps, t_minmax) bind(c,name='amp_create_table')
    integer(ik), intent(in)                 :: n_grp
    integer(ik), intent(in)                 :: n_st
    integer(ik), intent(in)                 :: n_total
    integer(ik), intent(in)                 :: batch_labels(n_total)
    integer(ik), intent(in)                 :: n_steps(n_total)
    real(drk), intent(in)                   :: t_minmax(2*n_total)

    integer(ik)                             :: n_amp
    integer(ik)                             :: i

    ! allocate the amplitude table
    amp_total = n_total
    allocate(amp_table(amp_total))

    ! allocate the correpsonding amplitude table
    ! for now, calculate how many slots given constant time step. This is not optimal, will fix later
    do i = 1,amp_total

      if(tstep_method == 'nascent') then
        n_amp = n_steps(i) + 1
      elseif(tstep_method == 'static ') then
        n_amp = ceiling( (t_minmax(2*i) - t_minmax(2*i-1)) / default_tstep) + 1
      endif
      allocate(amp_table(i)%time(n_amp))
      allocate(amp_table(i)%amplitude(n_amp))
      amp_table(i)%time        = -1.
      amp_table(i)%batch       = batch_labels(i)  
      amp_table(i)%label       = i
      amp_table(i)%current_row = 1
      amp_table(i)%amplitude   = zero_c
    enddo

    return
  end subroutine amp_create_table

  !
  !
  !
  function amp_table_exists() result(exists)
    logical                                 :: exists

    exists = allocated(amp_table) .and. (amp_total > 0)

    return
  end function amp_table_exists

  !
  !
  !
  subroutine amp_timestep_info(first_time, batch, time, n_traj, indices) bind(c, name='amp_timestep_info')
    logical, intent(in)                  :: first_time
    integer(ik), intent(in)              :: batch
    real(drk), intent(out)               :: time
    integer(ik), intent(out)             :: n_traj
    integer(ik), intent(out)             :: indices(amp_total)

    call get_current_time(first_time, batch, time, n_traj, indices)

    return
  end subroutine amp_timestep_info

  !
  !
  !
  subroutine init_method_code(init_code, init_str)
    integer(ik)                          :: init_code
    character(len=8)                     :: init_str

    if(init_code <= size(init_methods)) then
      init_str = init_methods(init_code)
    else
      init_str = ''
    endif  

    return
  end subroutine init_method_code

  !
  !
  ! 
  subroutine amp_retrieve_timestep(n, indices, amp_r, amp_i) bind(c, name='amp_retrieve_timestep')
    integer(ik), intent(in)                 :: n
    integer(ik), intent(in)                 :: indices(n)
    real(drk), intent(out)                  :: amp_r(n)
    real(drk), intent(out)                  :: amp_i(n)

    integer(ik)                             :: i, row 

    do i = 1,n
      row      = amp_table(indices(i))%current_row
      amp_r(i) = real(amp_table(indices(i))%amplitude(row))
      amp_i(i) = real(aimag(amp_table(indices(i))%amplitude(row)))
    enddo

    return
  end subroutine amp_retrieve_timestep

  !
  !
  !
  subroutine amp_next_timestep(n, indices, done) bind(c, name='amp_next_timestep')
    integer(ik), intent(in)                 :: n
    integer(ik), intent(in)                 :: indices(n)
    logical, intent(out)                    :: done

    done = step_current_time(n, indices)

    return
  end subroutine amp_next_timestep

 !
  !
  !
  function init_amplitude(time, method, indices, smat, origin_svec) result(amps)
    real(drk), intent(in)              :: time
    character(len=8)                   :: method
    integer(ik), intent(in)            :: indices(:) ! indices of the trajectories/amplitudes
    complex(drk), intent(in)           :: smat(:,:)  ! trajectory overlap matrix
    complex(drk), intent(in), optional :: origin_svec(:)

    complex(drk), allocatable       :: amps(:)
    integer(ik)                     :: i

    allocate(amps(size(indices)))

    if(.not.any(init_methods == method)) &
      stop 'initialization method not recognized'

    if(trim(method) == 'overlap' .and. .not.present(origin_svec)) &
      stop 'cannot initialize amplitudes without overlap values'

    select case(trim(method))
      case('overlap')
        do i = 1,size(indices)
          amps(i)   = origin_svec(i)
        enddo

      case('uniform')
        amps = one_c

      case('explicit')
        amps = one_c
   
      case default
        stop 'initializatoin method not recognized in init_amplitude'

    end select

    amps = normalize_amplitude(amps, smat)
    call update_amplitude(time, indices, amps)

    return
  end function init_amplitude

  !
  !
  ! 
  function update_basis(new_label, old_label, c) result(c_new)
    integer(ik), intent(in)                 :: new_label(:)
    integer(ik), intent(in)                 :: old_label(:)
    complex(drk), intent(in)                :: c(:)

    complex(drk), allocatable               :: c_new(:)
    integer(ik)                             :: n_new, n_old
    integer(ik)                             :: i, new_i

    call TimerStart('update_basis')

    n_new = size(new_label)
    n_old = size(old_label)
    allocate(c_new(n_new))

    c_new = zero_c
    do i = 1,n_old
      new_i = findloc(new_label, old_label(i), dim=1)
      if(new_i /= 0) c_new(new_i) = c(i)
    enddo

    call TimerStop('update_basis')
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

    call TimerStart('propagate_amplitude')

    n = size(c0)
    if(n /= size(H, dim=1))stop 'ERROR: size c0 and H do not match'
    allocate(new_amp(n), amps(n,1), B(n,n), U(n,n))

    amps(:n,1) = c0
    B = -I_drk * H * (t_end - t_start)
    call expm_complex(n, B, U)
    amps = matmul(U, amps)
    new_amp = amps(:n,1)

    call TimerStop('propagate_amplitude')
    return
  end function propagate_amplitude

 !
  !
  !
  function locate_amplitude(time, indices) result(amps)
    real(drk), intent(in)             :: time
    integer(ik), intent(in)           :: indices(:)

    complex(drk),allocatable          :: amps(:)
    integer(ik)                       :: i 
    integer(ik)                       :: table_indx
    integer(ik)                       :: time_indx
    real(drk)                         :: time_val

    allocate(amps(size(indices)))

    ! for now, we need the dimension of the amplitude vector to
    ! be the same as the dimension of the trajectory list. So, we
    ! assume that list is accurate and pull trajectory labels from there.
    do i = 1,size(indices)

      table_indx = indices(i)
      call get_time_index(table_indx, time, time_indx, time_val)

      ! if time step doesn't exist, set amplitude to zero
      if(time_indx == -1) then
        amps(i) = zero_c

      ! if we have exactly this time step, use it
      elseif(abs(time_val - time) <= dt_toler) then
        amps(i) = amp_table(indices(i))%amplitude(time_indx)
  
      ! else, obtain amplitude via interpolation
      else
        amps(i) = interpolate_amplitude(indices(i), time, time_indx)

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

    call TimerStart('interpolate_amplitude')

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

    call TimerStop('interpolate_amplitude')

    return
  end function interpolate_amplitude

  !
  !
  !
  subroutine update_amplitude(t, table_indices, amplitudes)
    real(drk), intent(in)            :: t
    integer(ik), intent(in)          :: table_indices(:)
    complex(drk),intent(in)          :: amplitudes(:)

    integer(ik)                      :: i, table_indx
    integer(ik)                      :: time_indx
    real(drk)                        :: time_val
  
    call TimerStart('update_amplitude')

    do i=1,size(table_indices)
      table_indx = table_indices(i)
      call get_time_index(table_indx, t, time_indx, time_val)
      if(time_indx == -1) stop 'Cannot update amplitude for requested time'
      amp_table(table_indx)%current_row          = time_indx
      amp_table(table_indx)%time(time_indx)      = t
      amp_table(table_indx)%amplitude(time_indx) = amplitudes(i)
    enddo

    call TimerStop('update_amplitude')
    return
  end subroutine update_amplitude

  !############################################################################
  ! Private routines not accessible outside module
  !
  !

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
    integer(ik)                      :: index_test(amp_total)

    i_max  = size(indices)
    time   = -1.
    n_fnd  = 0
    n_test = 0

    !print *,'batch=',batch,' time=',time

    do i = 1,amp_total

      !print *,'amp_table(i)%batch=',amp_table(i)%batch

      ! if this is not selected batch, move on to next one
      if(batch >= 0 .and. amp_table(i)%batch /= batch) continue

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
      if(amp_table(indices(i))%current_row < size(amp_table(indices(i))%time)) then
        amp_table(indices(i))%current_row = amp_table(indices(i))%current_row + 1
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

    real(drk), allocatable           :: time_buf(:)
    complex(drk), allocatable        :: amp_buf(:)
    real(drk)                        :: current_time, dt, dt_new

    call TimerStart('get_time_index')

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

    call TimerStop('get_time_index')
    return
  end subroutine get_time_index

  !
  !
  !
  function normalize_amplitude(c, smat) result(c_norm)
    complex(drk), intent(in)           :: c(:)
    complex(drk), intent(in)           :: smat(:,:)

    complex(drk),allocatable           :: c_norm(:)
    complex(drk)                       :: norm

    allocate(c_norm(size(c)))

    norm      = sqrt(dot_product(c, matvec_prod(smat, c)))
    if(abs(real(aimag(norm))) > 1000.d0*mp_drk) stop 'norm error'
    c_norm = c / norm

    return
  end function normalize_amplitude


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

end module libamp 

