!
! Builds the Hamiltonian and other requisite quantities
! necessary to propagate the TDSE under the canonical FMS wavefunction
!
! M. S. Schuurman, Oct. 11, 2018
!
!
module dynamics
  use accuracy
  use timer
  use math
  use libtraj
  use libamp
  use gauss_ints
  use saddlemod
  use taylormod
  use diracmod 
  use lvc_exactmod
  use lvc_mcamod
  implicit none

  public  init_propagate
  public  init_integrals
  public  propagate
  public  retrieve_matrices
  public  populations

  ! whether to propagate in full trajectory basis
  logical                       :: full_basis = .false.
  ! whether hamiltonian is hermitian
  logical                       :: hermitian = .true.
  ! implemented integration schemes
  character(len=8),parameter    :: integral_methods(1:5) = (/'saddle  ', 'taylor  ', 'dirac   ', 'lvc_adia', 'lvc_mca '/)
  character(len=8)              :: integral_method
  ! method for how to initialize amplitudes
  character(len=8)              :: init_method = ''

  ! integral types are determined at run-time
  class(integral), allocatable  :: nomad_ints

  type population_table
    integer(ik)                 :: resize
    integer(ik)                 :: nstates
    integer(ik)                 :: label
    real(drk), allocatable      :: time(:)
    real(drk), allocatable      :: pop(:,:)
  end type population_table

  type(population_table), allocatable        :: pop_table(:)

 contains

  !
  !
  !
  subroutine init_propagate(coherent, initial) bind(c, name='init_propagate')
    logical, intent(in)                :: coherent
    integer(ik), intent(in)            :: initial

    integer(ik)                        :: i

    full_basis = coherent
    call init_method_code(initial, init_method)

    return
  end subroutine init_propagate

  !
  !
  !
  subroutine init_integrals(int_method, nc, alpha, omega, focons, scalars) bind(c, name='init_integrals')
    integer(sik), intent(in)              :: int_method
    integer(sik), intent(in)              :: nc
    real(drk), optional, intent(in)       :: alpha(nc)
    real(drk), optional, intent(in)       :: omega(nc)
    real(drk), optional, intent(in)       :: focons(3*nc)
    real(drk), optional, intent(in)       :: scalars(3)

    !integer(ik)                           :: nc
    integer(sik)                          :: nstblk=3
    real(drk),allocatable                 :: fo_terms(:,:)    
    logical                               :: init_lvc


    if(int_method <= size(integral_methods)) then
      integral_method = integral_methods(int_method)
    else
      stop 'integral method not recognized'
    endif

    init_lvc = .false.
    if(present(focons)) then
      !nc = int(size(focons)/3.)
      allocate(fo_terms(nc, nstblk))
      fo_terms  = reshape( real(focons(:nstblk*nc)), shape=(/ nc, nstblk /) )
      if(present(alpha).and.present(omega).and.present(scalars)) init_lvc = .true.
    endif

    select case(int_method)
      case(1)
        allocate(saddle::nomad_ints)
      case(2)
        allocate(taylor::nomad_ints)
      case(3)
        allocate(dirac::nomad_ints)
      case(4)
        allocate(lvc_exact::nomad_ints)
        call nomad_ints%init(alpha, omega, fo_terms, scalars)
      case(5)
        allocate(lvc_mca::nomad_ints)
        call nomad_ints%init(alpha, omega, fo_terms, scalars)
      case default
        allocate(taylor::nomad_ints)
    end select
     
  end subroutine init_integrals

  !
  !
  !
  subroutine propagate(ti , tf) bind(c, name='propagate')
    real(drk), intent(in)              :: ti, tf !initial and final propagation times

    real(drk)                          :: time, t_step
    integer(ik)                        :: i,j
    integer(ik)                        :: n_runs
    integer(ik)                        :: i_bat
    integer(ik)                        :: batch_label
    complex(drk), allocatable          :: c(:)
    type(trajectory), allocatable      :: traj(:)
    complex(drk), allocatable          :: ovrlp(:)
    complex(drk)                       :: nuc_ovrlp

    integer(ik)                        :: n_current, n_old
    integer(ik),allocatable            :: current_traj(:)
    integer(ik),allocatable            :: old_traj(:)

    complex(drk), allocatable          :: s(:,:), t(:,:), v(:,:), h(:,:)
    complex(drk), allocatable          :: sdt(:,:), sinv(:,:), heff(:,:)
    complex(drk), allocatable          :: popwt(:,:,:)

    logical                            :: adapt_basis

    call TimerStart('propagate')

    if(.not.traj_table_exists() .or. .not.amp_table_exists()) &
       stop 'lib_traj needs to be initialized before fms'
    current_traj  = 0
    old_traj      = 0

    if(full_basis) then
      n_runs      =  1
      batch_label = -1 
    else
      n_runs      = n_batches()
    endif

    ! Initialize the population table
    allocate(pop_table(n_runs))
    do i = 1,n_runs
      pop_table(i)%resize  = 1000 ! some default value
      pop_table(i)%nstates = n_states()
      pop_table(i)%label   = i
      allocate(pop_table(i)%time(pop_table(i)%resize))
      allocate(pop_table(i)%pop(pop_table(i)%nstates+1, pop_table(i)%resize))
      pop_table(i)%time    = -1.d0
    enddo

    do i_bat = 1,n_runs

      ! if we're doing batch by batch processing, batch_label = i_bat
      if(.not.full_basis) batch_label = i_bat

      ! locate the t=ti trajectories for the current batch
      call locate_trajectories(ti, batch_label, old_traj, traj)
      n_old = traj_size(traj)
      allocate(c(n_old))
      allocate(s(n_old, n_old), t(n_old, n_old), v(n_old, n_old))
      allocate(h(n_old, n_old), sdt(n_old, n_old), sinv(n_old, n_old))
      allocate(heff(n_old,n_old))      
      allocate(popwt(n_old, n_old, n_states()))

      ! if we're using overlaps with a reference trajectory to initializae
      ! the amplitudes, compute those overlaps
      allocate(ovrlp(n_old))
      do i = 1,n_old
        ovrlp(i) = nuc_overlap_origin(traj(i))
        do j = 1, i
          nuc_ovrlp    = nuc_overlap(traj(i), traj(j))
          s(i,j)       = nomad_ints%overlap(traj(i), traj(j), nuc_ovrlp)
          popwt(i,j,:) = nomad_ints%pop(n_states(), traj(i), traj(j), nuc_ovrlp)
          if(i /= j) then
            s(j,i)       = conjg(s(i,j))
            popwt(j,i,:) = conjg(popwt(i,j,:))
          endif
        enddo
      enddo

      ! initialize the amplitudes using method=init_method
      c      = init_amplitude(ti, init_method, old_traj, s, ovrlp)
      time   = ti
      ! set the initial time populations
      call update_populations(time, i_bat, n_states(), c, s, popwt)
      t_step = traj_time_step(batch_label, time) - time

      !
      ! Main time propagation loop
      !
      do while(time <= tf)
        call locate_trajectories(time, batch_label, tindx0, traj0)
        n0 = traj_size(traj0)        

        ! if no more trajectories, exit the loop
        if(n_current == 0) exit

        ! for the matrices, we only care if the number of trajectories for this
        ! timestep is larger than the number for the previous
        if(n_current > n_old) then
          deallocate(s, t, v, h, sdt, sinv, heff, popwt)
          allocate(s(n0, n0), t(n0, n0), v(n0, n0))
          allocate(h(n0, n0), sdt(n0, n0), sinv(n0, n0))
          allocate(heff(n0, n0))
          allocate(popwt(n0, n0, n_states()))
        endif

        ! if the basis has changed in any way, make sure the basis label order
        ! in c matches the new trajectory list. Currently, 'update_basis' will
        ! change prev_traj to active_traj once update is complete. 
        adapt_basis = .false.
        if(n_old /= n_current) then
          adapt_basis = .true.
        elseif(.not.all(old_traj == current_traj)) then
          adapt_basis = .true.
        endif
        if(adapt_basis) then
          c = update_basis(current_traj, old_traj, c)
          old_traj = current_traj
          n_old = n_current
        endif

        call build_hamiltonian(traj0, traj0, s, t, v, h, sdt, sinv, heff, popwt)

        if(prop_expm) then

        if(time > ti) then
          c = propagate_amplitude(c, heff, time-0.5*t_step, time)
          call update_amplitude(time, current_traj, c)
          call update_populations(time, i_bat, n_states(), c, s, popwt)
        endif
 
        c      = propagate_amplitude(c, Heff, time, time+0.5*t_step)
        t_step = traj_time_step(batch_label, time) - time
        time   = time + t_step

      enddo ! do while(time<=tf)

      write(*, 1000)i_bat
    enddo ! do i_bat = 1,n_bat

    write(*, 1001)
    deallocate(old_traj, current_traj, c) 

    call TimerStop('propagate')
    call TimerReport
    return

1000 format(' Propagation of simulation ',i3,' complete.')
1001 format(' Propagation of all simulations complete using FMS')
  end subroutine propagate

  !
  !
  ! 
  subroutine retrieve_matrices(time, batch, n, nst, s_r, s_i, t_r, t_i, v_r, v_i, sdt_r, sdt_i, heff_r, heff_i, pop_r, pop_i) bind(c, name='retrieve_matrices')
    real(drk), intent(in)                   :: time           ! requested time
    integer(ik), intent(in)                 :: batch          ! batch number
    integer(ik), intent(in)                 :: n              ! expected dimension of the (square) matrices
    integer(ik), intent(in)                 :: nst            ! number of electronic states
    real(drk), intent(out)                  :: s_r(n*n),    s_i(n*n)
    real(drk), intent(out)                  :: t_r(n*n),    t_i(n*n)
    real(drk), intent(out)                  :: v_r(n*n),    v_i(n*n)
    real(drk), intent(out)                  :: sdt_r(n*n),  sdt_i(n*n)
    real(drk), intent(out)                  :: heff_r(n*n), heff_i(n*n)
    real(drk), intent(out)                  :: pop_r(n*n*nst), pop_i(n*n*nst)

    integer(ik),allocatable                 :: indices(:)
    type(trajectory), allocatable           :: traj(:)

    ! matrices
    complex(drk), allocatable,save          :: s(:,:)
    complex(drk), allocatable,save          :: t(:,:)
    complex(drk), allocatable,save          :: v(:,:)
    complex(drk), allocatable,save          :: h(:,:)
    complex(drk), allocatable,save          :: sdt(:,:)
    complex(drk), allocatable,save          :: sinv(:,:)
    complex(drk), allocatable,save          :: heff(:,:)
    complex(drk), allocatable,save          :: popwt(:,:,:)    

    ! well, for now, we'll simply recompute the matrices. Not ideal
    call locate_trajectories(time, batch, indices, traj)
    !print *,'time=',time,' batch=',batch,' n=',n,' traj_size(traj) = ',traj_size(traj)
    if(n /= traj_size(traj)) stop 'n != size(traj) in retrieve_matrices'
    if(nst /= n_states()) stop 'nst /= n_states() in retrieve_matrices'

    if(allocated(s)) then
      if(size(s, dim=1) /= traj_size(traj)) then
        deallocate(s, t, v, h, sdt, sinv, heff, popwt)
      endif
    endif
    if(.not.allocated(s)) then
      allocate(s(n,n),t(n,n),v(n,n),h(n,n))
      allocate(sdt(n,n),sinv(n,n),heff(n,n), popwt(n,n,n_states()))
    endif

    call build_hamiltonian(traj, s, t, v, h, sdt, sinv, heff, popwt)

    s_r    = reshape( real(s(:n, :n)), shape=(/ n*n /) )
    s_i    = reshape( real(aimag(s(:n, :n))), shape=(/ n*n /) )
    t_r    = reshape( real(t(:n, :n)), shape=(/ n*n /) )
    t_i    = reshape( real(aimag(t(:n, :n))), shape=(/ n*n /) )
    v_r    = reshape( real(v(:n, :n)), shape=(/ n*n /) )
    v_i    = reshape( real(aimag(v(:n, :n))), shape=(/ n*n /) )
    sdt_r  = reshape( real(sdt(:n, :n)), shape=(/ n*n /) )
    sdt_i  = reshape( real(aimag(sdt(:n, :n))), shape=(/ n*n /) )
    heff_r = reshape( real(heff(:n, :n)), shape=(/ n*n /) )
    heff_i = reshape( real(aimag(heff(:n, :n))), shape=(/ n*n /) )
    pop_r  = reshape( real(popwt(:n, :n, :n_states())), shape=(/ n*n*n_states() /) )
    pop_i  = reshape( real(aimag(popwt(:n, :n, :n_states()))), shape=(/ n*n*n_states() /) )

    !print *,'time,pop_i=',time,pop_i

    return
  end subroutine retrieve_matrices

  !
  !
  !
  subroutine pop_retrieve_timestep(time, batch, nst, pop, norm) bind(c, name='pop_retrieve_timestep')
    real(drk), intent(in)           :: time
    integer(ik), intent(in)         :: batch
    integer(ik), intent(in)         :: nst
    real(drk), intent(out)          :: pop(nst)
    real(drk), intent(out)          :: norm

    integer(ik), save               :: indx = 1
    real(drk)                       :: t_toler


    if(batch > size(pop_table)) stop 'batch > size(pop_table)'

    t_toler = 0.1*(traj_time_step(batch, time) - time)
    print *,'t_toler=',t_toler
    print *,'poptimes=',pop_table(batch)%time

    if(indx > size(pop_table(batch)%time)) indx = 1 ! if indx out of bounds, reset to '1'
    if(time < pop_table(batch)%time(indx)) indx = 1 ! if indx already > time, reset to '1'
    print *,'starting indx=',indx,'max=',size(pop_table(batch)%time)

    do while(abs(time-pop_table(batch)%time(indx)) > t_toler)
      if (pop_table(batch)%time(indx) > time) stop 'time not foudn in pop_retrieve_timestep, time(indx)>time'
      indx = indx + 1
      if (indx > size(pop_table(batch)%time)) stop 'time not found in pop_retrieve_timestep, indx > size(time)'
    enddo
    print *,'time=',time,' indx=',indx

    pop  = pop_table(batch)%pop(:nst, indx)
    norm = pop_table(batch)%pop(nst+1, indx)

  end subroutine pop_retrieve_timestep

  !
  !
  !
  subroutine update_populations(t, batch, nst, c, s, popwt)
    real(drk), intent(in)           :: t
    integer(ik), intent(in)         :: batch
    complex(drk), intent(in)        :: c(:)
    complex(drk), intent(in)        :: s(:,:)
    complex(drk), intent(in)        :: popwt(:,:,:)

    complex(drk)                    :: aic, aj, znorm
    complex(drk)                    :: zpop(nst)
    real(drk)                       :: pop(nst)
    real(drk)                       :: norm
    integer(ik)                     :: i,j
    integer(ik)                     :: n
    integer(ik)                     :: nst
    integer(ik),save                :: indx = 1

    pop    = zero_drk
    norm   = zero_drk

    zpop   = zero_c
    znorm  = zero_c

    n = size(c)

    if(batch > size(pop_table)) stop 'batch > size(pop_table)'
    
    do while(t < pop_table(batch)%time(indx) .and. pop_table(batch)%time(indx) /= -1.d0)
      indx = indx + 1
      if (indx > size(pop_table(batch)%time)) then
        call resize_pop_table(batch)
        exit
      endif
    enddo

    ! update the time list
    pop_table(batch)%time(indx) = t

    do i = 1, n
      aic = conjg(c(i)) ! c.c
      do j = 1, n
        aj    = c(j)
        zpop  = zpop  + aic * aj * popwt(i,j,:)
        znorm = znorm + aic * aj * s(i,j)
      enddo
    enddo

    pop = real(zpop)
    pop = pop / sum(pop)
    pop_table(batch)%pop(:nst, indx) = pop
    if(sum(abs(aimag(zpop))) > 1.e-10) then
      print *,'sum(aimag(zpop)) = ',sum(abs(aimag(zpop)))
      stop 'error in populations -- imaginary component in populations'
    endif

    norm = real(znorm)
    pop_table(batch)%pop(nst+1, indx) = norm
    if(abs(aimag(znorm)) > 1.e-10) then
      print *,'aimag(znorm) = ',abs(aimag(znorm))
      stop 'error in populations -- imaginary component in wfn norm'
    endif

  end subroutine update_populations

  !
  !
  !
  subroutine resize_pop_table(batch)
    integer(ik), intent(in)           :: batch

    integer(ik), allocatable          :: time_tmp(:)
    real(drk), allocatable            :: pop_tmp(:,:)

    integer(ik)                       :: current
    integer(ik)                       :: resized

    current = size(pop_table(batch)%time)
    resized = current + pop_table(batch)%resize

    allocate(time_tmp(current))
    allocate(pop_tmp(n_states()+1, current))

    time_tmp = pop_table(batch)%time
    pop_tmp  = pop_table(batch)%pop

    deallocate(pop_table(batch)%time, pop_table(batch)%pop)
   
    allocate(pop_table(batch)%time(resized))
    allocate(pop_table(batch)%pop(n_states()+1, resized))

    pop_table(batch)%time            = -1.d0 
    pop_table(batch)%time(:current)  = time_tmp
    pop_table(batch)%pop(:,:current) = pop_tmp

    deallocate(time_tmp, pop_tmp)

  end subroutine resize_pop_table

  !
  !
  !
  subroutine populations(time, n_amp, amp_r, amp_i, nst, batch, pop, norm) bind(c, name='populations')
    real(drk), intent(in)          :: time
    integer(ik), intent(in)        :: n_amp
    real(drk), intent(in)          :: amp_r(n_amp)
    real(drk), intent(in)          :: amp_i(n_amp)
    integer(ik), intent(in)        :: nst
    integer(ik), intent(in)        :: batch
    real(drk), intent(out)         :: pop(nst)
    real(drk), intent(out)         :: norm

    type(trajectory), allocatable  :: traj_list(:)
    integer(ik), allocatable       :: indices(:)
    integer(ik)                    :: n_traj
    integer(ik)                    :: i,j
    complex(drk)                   :: zpop(size(pop))
    complex(drk)                   :: aic, aj
    complex(drk)                   :: Snuc
    complex(drk)                   :: znorm

    if(nst /= n_states()) stop 'nst /= n_states() in populations'

    call locate_trajectories(time, batch, indices, traj_list)
    pop    = zero_drk
    norm   = zero_drk

    zpop   = zero_c
    znorm  = zero_c
    n_traj = traj_size(traj_list)

    if(n_traj /= n_amp) stop 'n_traj /= n_amp in populations'

    do i = 1,n_traj
      aic = cmplx(amp_r(i), -amp_i(i)) ! c.c
      do j = 1,n_traj
        aj    = cmplx(amp_r(j), amp_i(j))
        Snuc  = nuc_overlap(traj_list(i), traj_list(j))
        zpop  = zpop  + aic * aj * nomad_ints%pop(n_states(), traj_list(i), traj_list(j), Snuc)
        znorm = znorm + aic * aj * nomad_ints%overlap(traj_list(i), traj_list(j), Snuc)
      enddo
    enddo

    pop = real(zpop)
    pop = pop / sum(pop)
    if(sum(abs(aimag(zpop))) > 1.e-10) then
      print *,'sum(aimag(zpop)) = ',sum(abs(aimag(zpop)))
      stop 'error in populations -- imaginary component in populations'
    endif

    norm = real(znorm)
    if(abs(aimag(znorm)) > 1.e-10) then
      print *,'aimag(znorm) = ',abs(aimag(znorm))
      stop 'error in populations -- imaginary component in wfn norm'
    endif

    return
  end subroutine populations

  !**************************************************************************************
  !
  ! Routines to build Hamiltonian and propagate amplitudes
  !
  !
  !
  subroutine build_hamiltonian(traj, s, t, v, h, sdt, sinv, heff, popwt)
    type(trajectory), intent(in)             :: traj(:)
    complex(drk), intent(out)                :: s(:,:)
    complex(drk), intent(out)                :: t(:,:)
    complex(drk), intent(out)                :: v(:,:)
    complex(drk), intent(out)                :: h(:,:)
    complex(drk), intent(out)                :: sdt(:,:)
    complex(drk), intent(out)                :: sinv(:,:)
    complex(drk), intent(out)                :: heff(:,:)
    complex(drk), intent(out)                :: popwt(:,:,:)

    integer(ik)                              :: n
    integer(ik)                              :: bra, ket
    complex(drk)                             :: nuc_ovrlp

    call TimerStart('build_hamiltonian')

    n = size(traj)
    if(size(s, dim=1) < n) stop 'matrices wrong dimension in build_hamiltonian'
    
    s    = zero_c
    t    = zero_c
    v    = zero_c
    h    = zero_c
    sdt  = zero_c
    sinv = zero_c
    heff = zero_c
    popwt = zero_c

    do bra = 1, n
      do ket = 1, bra

        nuc_ovrlp    = nuc_overlap(traj(bra), traj(ket))

        s(bra, ket)   = nomad_ints%overlap(    traj(bra), traj(ket), nuc_ovrlp)
        t(bra, ket)   = nomad_ints%kinetic(    traj(bra), traj(ket), nuc_ovrlp)
        v(bra, ket)   = nomad_ints%potential(  traj(bra), traj(ket), nuc_ovrlp)
        sdt(bra, ket) = nomad_ints%sdot(       traj(bra), traj(ket), nuc_ovrlp)
        h(bra, ket)   = t(bra, ket) + v(bra, ket)
        popwt(bra, ket, :) = nomad_ints%pop(n_states(), traj(bra), traj(ket), nuc_ovrlp)

        ! assume hermitian for now
        if(ket /= bra) then
          sdt(ket,bra)       = nomad_ints%sdot( traj(ket), traj(bra), conjg(nuc_ovrlp))
          s(ket,bra)         = conjg(s(bra, ket))
          t(ket,bra)         = conjg(t(bra, ket))
          v(ket,bra)         = conjg(v(bra, ket))
          h(ket,bra)         = conjg(h(bra, ket))
          popwt(ket, bra, :) = conjg(popwt(bra, ket, :))
        endif


      enddo
    enddo

    sinv = inverse_gelss(n, s(:n, :n))
    heff = matmul(sinv(:n, :n), h(:n, :n) - I_drk*sdt(:n, :n))

    !print *,'t=',t
    !print *,'v=',v
    !print *,'sdot=',sdt
    !print *,'heff=',heff

    call TimerStop('build_hamiltonian')

    return
  end subroutine build_hamiltonian

end module dynamics



