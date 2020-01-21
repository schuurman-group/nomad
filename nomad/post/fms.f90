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
  use libprop
  implicit none

  private
  public  overlap 
  public  ke
  public  potential
  public  sdot

  interface potential  
    module procedure potential_taylor
    module procedure potential_saddle
  end interface

  !
 
 contains

  !
  !
  !
  subroutine propagate(ti , tf) bind(c, name='propagate')
    real(drk), intent(in)         :: ti, tf !initial and final propagation times
    real(drk)                     :: t
    integer(ik)                   :: n_runs, i_bat, batch_label
    real(drk)                     :: pops(n_state)
    complex(drk), allocatable     :: Heff(:,:)
    complex(drk), allocatable     :: c(:)
    integer(ik), allocatable      :: prev_traj(:), active_traj(:)
    logical                       :: update_traj
!    integer(ik)                   :: i

    if(full_basis) then
      n_runs      = 1
      batch_label = 0
    else
      n_runs = n_batch
    endif

    update_traj = .false.

    do i_bat = 1,n_runs

      if(.not.full_basis) batch_label = i_bat

      t = ti
      call collect_trajectories(ti, batch_label, prev_traj)
      allocate(Heff(size(prev_traj), size(prev_traj)))
      allocate(c(size(prev_traj)))
      c = init_amplitude(ti)

      do while(t < tf)
        call collect_trajectories(t, batch_label, active_traj)

        ! if the basis has changed in any way, make sure the basis label order
        ! in c matches the new trajectory list. Currently, 'update_basis' will
        ! change prev_traj to active_traj once update is complete. 
        if(size(active_traj) /= size(prev_traj)) then
          update_traj = .true.
        elseif(.not.all(active_traj == prev_traj)) then
          update_traj = .true.
        else
          update_traj = .false.
        endif

       if(update_traj) then
          c = update_basis(active_traj, prev_traj, c)
          deallocate(prev_traj)
          allocate(prev_traj(size(active_traj)))
          prev_traj = active_traj
        endif

        call build_hamiltonian(Heff)
        if(t > ti) then
          c = propagate_amplitude(c, Heff, t-0.5*t_step, t)
          call update_amplitude(t, c)
          pops = determine_populations(c)
        endif

        c = propagate_amplitude(c, Heff, t, t+0.5*t_step)
        t = t + t_step
      enddo

    enddo

    return
  end subroutine propagate


  ! Routines to build Hamiltonian and propagate amplitudes
  !
  !
  !
  subroutine build_hamiltonian(Heff)
    complex(drk), allocatable, intent(inout) :: Heff(:,:)
    integer(ik)                              :: n
    integer(ik)                              :: bra, ket
    complex(drk),allocatable                 :: H(:,:), T(:,:), V(:,:)
    complex(drk),allocatable                 :: S(:,:), Sinv(:,:), Sdt(:,:)
    complex(drk)                             :: nuc_ovrlp

    n = size(traj_list)
    if(n > size(Heff, dim=1)) then
      deallocate(Heff)
      allocate(Heff(n,n))
    endif
    allocate(H(n,n))
    allocate(S(n,n))
    allocate(T(n,n))
    allocate(V(n,n))
    allocate(Sinv(n,n))
    allocate(Sdt(n,n))

    H    = zero_c
    S    = zero_c
    T    = zero_c
    V    = zero_c
    Sinv = zero_c
    Sdt  = zero_c

    do bra = 1, n
      do ket = 1, n

        if(hermitian .and. ket < bra) then
          S(bra,ket)   = conjg(S(ket, bra))
          T(bra,ket)   = conjg(T(ket, bra))
          V(bra,ket)   = conjg(V(ket, bra))
          H(bra,ket)   = conjg(H(ket, bra))
          Sdt(bra,ket) = sdot(traj_list(ket), traj_list(bra), S(ket, bra))
          cycle
        endif

        nuc_ovrlp     = nuc_overlap(traj_list(bra), traj_list(ket))
        S(bra, ket)   = overlap(  traj_list(bra), traj_list(ket))
        Sdt(bra, ket) = sdot(     traj_list(bra), traj_list(ket), nuc_ovrlp)
        T(bra, ket)   = ke(       traj_list(bra), traj_list(ket), nuc_ovrlp)
        V(bra, ket)   = potential(traj_list(bra), traj_list(ket), nuc_ovrlp)
        H(bra, ket)   = T(bra,ket) + V(bra,ket)

      enddo
    enddo

    Sinv = zinverse(n, S)
    Heff = matmul(Sinv, H - I_drk*Sdt)
    deallocate(H, S, T, V, Sinv, Sdt)

    return
  end subroutine build_hamiltonian

  !
  !
  !
  function determine_populations(amp) result(pops)
    complex(drk), intent(in)         :: amp(:)

    complex(drk)                     :: pops(n_state)
    complex(drk),allocatable         :: Sij(:,:)
    integer(ik)                      :: st, n_traj
    integer(ik)                      :: i, j

    n_traj = size(traj_list) 
    pops   = zero_drk
    allocate(Sij(n_traj, n_traj))
    Sij = overlap_mat(traj_list)

    do i = 1,n_traj
      do j = 1,n_traj
        st       = traj_list(i)%state
        pops(st) = pops(st) + conjg(amp(i))*amp(j)*Sij(i,j)
      enddo
    enddo

    pops = pops / sum(pops)

    return
  end function determine_populations

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
    complex(drk)                  :: ke_vec(n_crd)
    complex(drk)                  :: dx(n_crd), psum(n_crd)
    real(drk)                     :: w1w2(n_crd), w1_w2(n_crd)

    if(bra_t%state /= ket_t%state) then
      ke_int = zero_c
      return
    endif

    dx     = bra_t%x - ket_t%x
    psum   = widths*ket_t%p + widths*bra_t%p
    w1w2   = widths*widths
    w1_w2  = widths+widths
    ke_vec = Sij * (-4.*w1w2*dx*psum*I_drk - 2.*w1w2*w1_w2 + &
                     4.*dx**2 * widths**2 * widths**2 + psum**2) / w1_w2**2

    ke_int = -dot_product(0.5/masses, ke_vec)
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
    complex(drk)                  :: o1_ij(n_crd), o1_ji(n_crd)
    real(drk)                     :: f_ij(n_crd), f_ji(n_crd)
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
      o1_ij  = 0.5 * delx(bra_t, ket_t, Sij) / masses
      o1_ji  = 0.5 * delx(ket_t, bra_t, Sji) / masses

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
    complex(drk)                  :: deldx(n_crd), deldp(n_crd)

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
    
    complex(drk)                  :: delx_int(n_crd)
    real(drk)                     :: dx(n_crd), psum(n_crd)
    real(drk)                     :: w1w2(n_crd), w1_w2(n_crd)

    w1w2  = widths * widths
    w1_w2 = widths + widths
    dx    = bra_t%x - ket_t%x
    psum  = widths*ket_t%p + widths*bra_t%p
    delx_int = Sij * (2. * w1w2 * dx - I_drk * psum) / w1_w2

    return
  end function delx

  !
  !
  !
  function delp(bra_t, ket_t, Sij) result(delp_int)
    type(trajectory), intent(in)  :: bra_t, ket_t
    complex(drk), intent(in)      :: Sij
 
    complex(drk)                  :: delp_int(n_crd)
    real(drk)                     :: dx(n_crd), dp(n_crd)
    real(drk)                     :: w1_w2(n_crd)

    dx    = bra_t%x - ket_t%x
    dp    = bra_t%p - ket_t%p
    w1_w2 = widths + widths
    delp_int = Sij * (dp + 2. * I_drk * widths * dx) / (2.*w1_w2)
 
    return
  end function delp

  !
  !
  !
  function phase_dot(traj) result(phase_deriv)
    type(trajectory), intent(in)  :: traj
 
    real(drk)                     :: phase_deriv

    phase_deriv = tr_kinetic(traj) - tr_potential(traj) - dot_product(widths, 0.5/masses)

    return
  end function phase_dot

 
end module fms 



