!
! Builds the Hamiltonian and other requisite quantities
! necessary to propagate the TDSE under the canonical FMS wavefunction
!
! M. S. Schuurman, Oct. 11, 2018
!
!
module fms
  use accuracy
  use math
  use lib_traj
  use lib_amp
  implicit none
  private
  public  init_propagate
  public  propagate
  public  retrieve_matrices

    ! whether to propagate in full trajectory basis
    logical                       :: full_basis = .false.
    ! whether hamiltonian is hermitian
    logical                       :: hermitian = .true.
    ! implemented integration schemes
    character(len=6),parameter    :: integral_methods(1:3) = (/'saddle',  'taylor', 'dirac '/)
    character(len=6)              :: integral_method
    integer(ik)                   :: integral_ordr=1
    ! method for how to initialize amplitudes
    character(len=8)              :: init_method = ''

    interface potential  
      module procedure potential_taylor
      module procedure potential_saddle
    end interface

 contains

  !
  !
  !
  subroutine init_propagate(coherent, initial, int_method, int_order) bind(c, name='init_propagate')
    logical, intent(in)                :: coherent
    integer(ik), intent(in)            :: initial
    integer(ik), intent(in)            :: int_method
    integer(ik), intent(in)            :: int_order 

    full_basis = coherent

    call init_method_code(initial, init_method)

    if(int_method <= size(integral_methods)) then
      integral_method = integral_methods(int_method)
    else
      stop 'time step method not recognized'
    endif

    if(int_order <= 2) then
      integral_ordr = int_order
    endif

    return
  end subroutine init_propagate

  !
  !
  !
  subroutine propagate(ti , tf) bind(c, name='propagate')
    real(drk), intent(in)              :: ti, tf !initial and final propagation times
    real(drk)                          :: time, t_step
    integer(ik)                        :: i
    integer(ik)                        :: n_runs
    integer(ik)                        :: i_bat
    integer(ik)                        :: batch_label
    complex(drk), allocatable          :: c(:)
    type(trajectory), allocatable      :: traj(:)
    complex(drk), allocatable          :: ovrlp(:)

    integer(ik)                        :: n_current, n_old
    integer(ik),allocatable            :: current_traj(:)
    integer(ik),allocatable            :: old_traj(:)

    complex(drk), allocatable          :: s(:,:), t(:,:), v(:,:), h(:,:)
    complex(drk), allocatable          :: sdt(:,:), sinv(:,:), heff(:,:)

    logical                            :: adapt_basis

    if(.not.traj_table_exists() .or. .not.amp_table_exists()) &
       stop 'lib_traj needs to be initialized before fms'
    current_traj  = 0
    old_traj      = 0

    if(full_basis) then
      n_runs      = 1
      batch_label = 0
    else
      n_runs      = n_batches()
    endif

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

      ! if we're using overlaps with a reference trajectory to initializae
      ! the amplitudes, compute those overlaps
      if(trim(init_method) == 'overlap') then
        allocate(ovrlp(n_old))
        do i = 1,n_old
          ovrlp(i) = ref_overlap(traj(i))
        enddo
      endif

      ! initialize the amplitudes using method=init_method
      c      = init_amplitude(ti, init_method, old_traj, ovrlp)
      time   = ti
      t_step = traj_time_step(batch_label, time) - time

      do while(time <= tf)
        call locate_trajectories(time, batch_label, current_traj, traj)
        n_current = traj_size(traj)        

        ! if no more trajectories, exit the loop
        if(n_current == 0) exit

        ! for the matrices, we only care if the number of trajectories for this
        ! timestep is larger than the number for the previous
        if(n_current > n_old) then
          deallocate(s, t, v, h, sdt, sinv, heff)
          allocate(s(n_current, n_current), t(n_current, n_current), v(n_current, n_current))
          allocate(h(n_current, n_current), sdt(n_current, n_current), sinv(n_current, n_current))
          allocate(heff(n_current,n_current))
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

        call build_hamiltonian(traj, s, t, v, h, sdt, sinv, heff)
        if(time > ti) then
          c = propagate_amplitude(c, Heff, time-0.5*t_step, time)
          call update_amplitude(time, current_traj, c)
        endif

        c      = propagate_amplitude(c, Heff, time, time+0.5*t_step)
        t_step = traj_time_step(batch_label, time) - time
        time   = time + t_step
      enddo

      write(*, 1000)i_bat
    enddo

    write(*, 1001)
    deallocate(old_traj, current_traj, c) 
    return

1000 format(' Propagation of simulation ',i3,' complete.')
1001 format(' Propagation of all simulations complete using FMS')
  end subroutine propagate

  !
  !
  ! 
  subroutine retrieve_matrices(time, batch, n, s_r, s_i, t_r, t_i, v_r, v_i, sdt_r, sdt_i, heff_r, heff_i) bind(c, name='retrieve_matrices')
    real(drk), intent(in)                   :: time           ! requested time
    integer(ik), intent(in)                 :: batch          ! batch number
    integer(ik), intent(in)                 :: n              ! expected dimension of the (square) matrices
    real(drk), intent(out)                  :: s_r(n*n),    s_i(n*n)
    real(drk), intent(out)                  :: t_r(n*n),    t_i(n*n)
    real(drk), intent(out)                  :: v_r(n*n),    v_i(n*n)
    real(drk), intent(out)                  :: sdt_r(n*n),  sdt_i(n*n)
    real(drk), intent(out)                  :: heff_r(n*n), heff_i(n*n)

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
    
    ! well, for now, we'll simply recompute the matrices. Not ideal
    call locate_trajectories(time, batch, indices, traj)
    if(n /= traj_size(traj)) stop 'n != size(traj) in retrieve_matrices'

    if(allocated(s)) then
      if(size(s, dim=1) /= traj_size(traj)) then
        deallocate(s, t, v, h, sdt, sinv, heff)
      endif
    endif
    if(.not.allocated(s)) then
      allocate(s(n,n),t(n,n),v(n,n),h(n,n))
      allocate(sdt(n,n),sinv(n,n),heff(n,n))
    endif

    call build_hamiltonian(traj, s, t, v, h, sdt, sinv, heff)

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

    return
  end subroutine retrieve_matrices


  !**************************************************************************************

  ! Routines to build Hamiltonian and propagate amplitudes
  !
  !
  !
  subroutine build_hamiltonian(traj, s, t, v, h, sdt, sinv, heff)
    type(trajectory), intent(in)             :: traj(:)
    complex(drk), intent(out)                :: s(:,:)
    complex(drk), intent(out)                :: t(:,:)
    complex(drk), intent(out)                :: v(:,:)
    complex(drk), intent(out)                :: h(:,:)
    complex(drk), intent(out)                :: sdt(:,:)
    complex(drk), intent(out)                :: sinv(:,:)
    complex(drk), intent(out)                :: heff(:,:)

    integer(ik)                              :: n
    integer(ik)                              :: bra, ket
    complex(drk)                             :: nuc_ovrlp

    n = size(traj)
    if(size(s, dim=1) < n) stop 'matrices wrong dimension in build_hamiltonian'
    
    s    = zero_c
    t    = zero_c
    v    = zero_c
    h    = zero_c
    sdt  = zero_c
    sinv = zero_c
    heff = zero_c

    do bra = 1, n
      do ket = 1, n

        if(hermitian .and. ket < bra) then
          s(bra,ket)   = conjg(s(ket, bra))
          t(bra,ket)   = conjg(t(ket, bra))
          v(bra,ket)   = conjg(v(ket, bra))
          h(bra,ket)   = conjg(h(ket, bra))
          sdt(bra,ket) = sdot(traj(ket), traj(bra), S(ket, bra))
          cycle
        endif

        nuc_ovrlp     = nuc_overlap(traj(bra), traj(ket))
        s(bra, ket)   = overlap(    traj(bra), traj(ket))
        t(bra, ket)   = ke(         traj(bra), traj(ket), nuc_ovrlp)
        v(bra, ket)   = potential(  traj(bra), traj(ket), nuc_ovrlp)
        h(bra, ket)   = t(bra, ket) + v(bra, ket)
        sdt(bra, ket) = sdot(       traj(bra), traj(ket), nuc_ovrlp)

      enddo
    enddo

    sinv = inverse_gelss(n, s(:n, :n))
    heff = matmul(sinv(:n, :n), h(:n, :n) - I_drk*sdt(:n, :n))

    return
  end subroutine build_hamiltonian

  !***********************************************************************
  ! Numerical routines for evaluating matrix elements
  !

  !
  !
  !
  function overlap(bra_t, ket_t) result(Sij)
    type(trajectory), intent(in)  :: bra_t, ket_t
    complex(drk)                  :: Sij

    if(bra_t%state /= ket_t%state) then
      Sij = zero_c
    else
      Sij = nuc_overlap(bra_t, ket_t)
    endif

    return
  end function overlap

  !
  !
  !
  function overlap_mat(t_list) result(S)
    type(trajectory), intent(in)      :: t_list(:)

    complex(drk), allocatable         :: S(:,:)
    integer(ik)                       :: i,j

    allocate(S(size(t_list), size(t_list)))

    S = zero_c
    do i = 1,size(t_list)
      do j = 1,i
        S(i,j) = overlap(t_list(i), t_list(j))
        S(j,i) = conjg(S(i,j))
      enddo
    enddo
 
    return
  end function

  !
  !
  ! 
  function ke(bra_t, ket_t, Sij) result(ke_int)
    type(trajectory), intent(in)  :: bra_t, ket_t
    complex(drk), intent(in)      :: Sij

    complex(drk)                  :: ke_int
    complex(drk)                  :: ke_vec(size(bra_t%x))
    complex(drk)                  :: dx(size(bra_t%x)), psum(size(bra_t%x))
    real(drk)                     :: a1(size(bra_t%x)), a2(size(bra_t%x))

    if(bra_t%state /= ket_t%state) then
      ke_int = zero_c
      return
    endif

    a1     = bra_t%width
    a2     = ket_t%width
    dx     = bra_t%x - ket_t%x
    psum   = a1*ket_t%p + a2*bra_t%p
    ke_vec = Sij * (-4.*a1*a2*dx*psum*I_drk - 2.*a1*a2*(a1+a2) + &
                     4.*dx**2 * a1**2 * a2**2 + psum**2) / (a1+a2)**2

    ke_int = -dot_product(0.5/bra_t%mass, ke_vec)
    return
  end function ke

  !
  !
  !
  function potential_taylor(bra_t, ket_t, Sij) result(pot_int)
    type(trajectory), intent(in)  :: bra_t, ket_t
    complex(drk), intent(in)      :: Sij

    complex(drk)                  :: pot_int 
    complex(drk)                  :: vij, vji, Sji
    complex(drk)                  :: o1_ij(size(bra_t%x)), o1_ji(size(bra_t%x))
    real(drk)                     :: f_ij(size(bra_t%x)), f_ji(size(bra_t%x))
    integer(ik)                   :: state, bra, ket 

    Sji = conjg(Sij)

    if(bra_t%state == ket_t%state) then
      state = bra_t%state
      vij   = bra_t%energy(state) * Sij
      vji   = ket_t%energy(state) * Sji

      if(integral_ordr > 0) then
        o1_ij = qn_vector(1, bra_t, ket_t, Sij)
        o1_ji = qn_vector(1, ket_t, bra_t, Sji) 
        vij   = vij + dot_product(o1_ij - bra_t%x*Sij, bra_t%deriv(:, state, state))
        vji   = vji + dot_product(o1_ji - ket_t%x*Sji, ket_t%deriv(:, state, state))
      endif

      if(integral_ordr > 1) then
        print *,'second ordr not yet implemented'
      endif

    else
      
      bra    = bra_t%state
      ket    = ket_t%state
      f_ij   = bra_t.deriv(:, bra, ket)
      f_ji   = ket_t.deriv(:, ket, bra)
      o1_ij  = 0.5 * delx(bra_t, ket_t, Sij) / bra_t%mass
      o1_ji  = 0.5 * delx(ket_t, bra_t, Sji) / ket_t%mass

      vij   = 2.*dot_product(f_ij, o1_ij)
      vji   = 2.*dot_product(f_ji, o1_ji)       

    endif

    pot_int = 0.5 * (vij + conjg(vji))

    return
  end function potential_taylor

  !
  !
  !
  function potential_saddle(bra_t, ket_t, centroid, Sij) result(pot_int)
    type(trajectory), intent(in)  :: bra_t, ket_t, centroid 
    complex(drk), intent(in)      :: Sij

    complex(drk)                  :: pot_int

    pot_int = zero_c 
 
    return
  end function potential_saddle

  !
  !
  !
  function sdot(bra_t, ket_t, Sij) result(sdot_int)
    type(trajectory), intent(in)  :: bra_t, ket_t
    complex(drk), intent(in)      :: Sij

    complex(drk)                  :: sdot_int
    complex(drk)                  :: deldx(size(bra_t%x)), deldp(size(bra_t%x))

    if(bra_t.state /= ket_t.state) then
      sdot_int = zero_c
      return
    endif

    deldx     = conjg(delx(bra_t, ket_t, Sij))
    deldp     = conjg(delp(bra_t, ket_t, Sij))

    sdot_int = dot_product(deldx, tr_velocity(ket_t)) +&
               dot_product(deldp, tr_force(ket_t)) +&
               I_drk * phase_dot(ket_t) * Sij

    return
  end function sdot

  !
  !
  !
  function delx(bra_t, ket_t, Sij) result(delx_int)
    type(trajectory), intent(in)  :: bra_t, ket_t
    complex(drk), intent(in)      :: Sij
    
    complex(drk)                  :: delx_int(size(bra_t%x))
    real(drk)                     :: dx(size(bra_t%x)), psum(size(bra_t%x))
    real(drk)                     :: a1(size(bra_t%x)), a2(size(bra_t%x))

    a1    = bra_t%width
    a2    = ket_t%width
    dx    = bra_t%x - ket_t%x
    psum  = a1*ket_t%p + a2*bra_t%p
    delx_int = Sij * (2. * a1 * a2 * dx - I_drk * psum) / (a1+a2)

    return
  end function delx

  !
  !
  !
  function delp(bra_t, ket_t, Sij) result(delp_int)
    type(trajectory), intent(in)  :: bra_t, ket_t
    complex(drk), intent(in)      :: Sij
 
    complex(drk)                  :: delp_int(size(bra_t%x))
    real(drk)                     :: dx(size(bra_t%x)), dp(size(bra_t%x))
    real(drk)                     :: a1(size(bra_t%x)), a2(size(bra_t%x))

    a1    = bra_t%width
    a2    = ket_t%width
    dx    = bra_t%x - ket_t%x
    dp    = bra_t%p - ket_t%p
    delp_int = Sij * (dp + 2. * I_drk * a1 * dx) / (2.*(a1+a2))
 
    return
  end function delp

  !
  !
  !
  function phase_dot(traj) result(phase_deriv)
    type(trajectory), intent(in)  :: traj
 
    real(drk)                     :: phase_deriv

    phase_deriv = tr_kinetic(traj) - tr_potential(traj) - &
                  dot_product(traj%width, 0.5/traj%mass)

    return
  end function phase_dot

  !
  !
  !
  function populations(n_states, n_traj, traj_list) result(pops)
    integer(ik), intent(in)        :: n_states
    integer(ik), intent(in)        :: n_traj
    type(trajectory), intent(in)   :: traj_list(n_traj)
 
    real(drk)                      :: pops(n_states)
    complex(drk)                   :: zpops(n_states)
    complex(drk)                   :: zpop
    integer(ik)                    :: i,j
    integer(ik)                    :: state 

    pops  = zero_drk
    zpops = zero_c

    do i = 1,n_traj
      state = traj_list(i)%state
      if(state > n_states)stop 'error in populations, state>n_states'
      do j = 1,i
        zpop = conjg(traj_list(i)%amplitude) * traj_list(j)%amplitude * &
               overlap(traj_list(i), traj_list(j)) 
        zpops(state) = zpops(state) + zpop 
        if(i/=j) zpops(state) = zpops(state) + conjg(zpop)
      enddo
    enddo

    pops = real(zpops)
    if(sum(abs(aimag(zpops))) > mp_drk) then
      stop 'error in populations -- imaginary component in populations'
    endif

  end function populations
 
end module fms 



