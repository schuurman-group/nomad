!
! Builds the Hamiltonian and other requisite quantities
! necessary to propagate the TDSE under the canonical FMS wavefunction
!
! M. S. Schuurman, Oct. 11, 2018
!
!
module fms
  use accuracy
  use libprop
  implicit none

  private
  public  overlap 
  public  ke
  public  potential
  public  sdot

  complex(drk), allocatable          :: Heff(:,:)

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

    if(full_basis) then
      n_runs      = 1
      batch_label = 0
    else
      n_runs = n_batch
    endif

    do i_bat = 1,n_runs

      if(.not.full_basis) batch_label = i_bat

      t = ti
      call collect_trajectories(ti, batch_label)
      allocate(Heff(tstep_cnt, tstep_cnt))
      call init_amplitude(ti)

      do while(t < tf)
        call collect_trajectories(t, batch_label)
        call build_hamiltonian()
        call propagate_amplitude(Heff, t, t+t_step)
        t = t + t_step
      enddo

    enddo

    return
  end subroutine propagate


  ! Routines to build Hamiltonian and propagate amplitudes
  !
  !
  !
  subroutine build_hamiltonian()
    integer(ik)                 :: n
    integer(ik)                 :: bra, ket
    complex(drk),allocatable    :: H(:,:), T(:,:), V(:,:)
    complex(drk),allocatable    :: S(:,:), Sinv(:,:), Sdot(:,:)

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
    allocate(Sdot(n,n))

    H    = zero_c
    S    = zero_c
    T    = zero_c
    V    = zero_c
    Sinv = zero_c
    Sdot = zero_c

    do bra = 1, n
      do ket = 1, n

        if(hermitian .and. ket > bra) then
          S(ket, bra) = conjg(S(bra, ket))
          H(ket, bra) = conjg(H(bra, ket))
          cycle
        endif

        S(bra, ket)    = overlap(  traj_list(bra), traj_list(ket))
        T(bra, ket)    = ke(       traj_list(bra), traj_list(ket), S(bra, ket))
        V(bra, ket)    = potential(traj_list(bra), traj_list(ket), S(bra, ket))
        H(bra, ket)    = T(bra,ket) + V(bra,ket)

      enddo
    enddo

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

    if(bra_t.state /= ket_t.state) then
      Sij = zero_c
    else
      Sij = nuc_overlap(bra_t, ket_t)
    endif

    return
  end function overlap

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

    dx     = bra_t.x - ket_t.x
    psum   = widths*ket_t.p + widths*bra_t.p
    w1w2   = widths*widths
    w1_w2  = widths+widths
    ke_vec = Sij * (-4.*w1w2*dx*psum*I - 2.*w1w2*w1_w2 + &
                     4.*dx**2 * widths**2 * widths**2 + psum**2) / w1_w2**2

    ke_int = dot_product(0.5*masses, ke_vec)
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
    integer(ik)                   :: state, bra, ket 

    Sji = conjg(Sij)

    if(bra_t.state == ket_t.state) then
      state = bra_t.state
      vij   = bra_t.energy(state) * Sij
      vji   = ket_t.energy(state) * Sji

      if(integral_ordr > 0) then
        o1_ij = qn_vector(1, bra_t, ket_t, Sij)
        o1_ji = qn_vector(1, ket_t, bra_t, Sji) 
        vij   = vij + dot_product(o1_ij - bra_t.x*Sij, bra_t.deriv(:, state, state))
        vji   = vji + dot_product(o1_ji - ket_t.x*Sji, ket_t.deriv(:, state, state))
      endif

      if(integral_ordr > 1) then
        print *,'second ordr not yet implemented'
      endif

    else
      
      bra    = bra_t.state
      ket    = ket_t.state
      o1_ij  = 0.5 * delx(bra_t, ket_t, Sij) / masses
      o1_ji  = 0.5 * delx(ket_t, bra_t, Sji) / masses

      vij   = 2.*dot_product(bra_t.deriv(:, bra, ket), o1_ij)
      vji   = 2.*dot_product(ket_t.deriv(:, ket, bra), o1_ji)       

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

    deldx     = delx(bra_t, ket_t, Sij)
    deldp     = delp(bra_t, ket_t, Sij)

    sdot_int = dot_product(deldx, tr_velocity(ket_t)) +&
               dot_product(deldp, tr_force(ket_t)) +&
               I * phase_dot(ket_t) * Sij

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

    dx   = bra_t.x - ket_t.x
    psum = widths*ket_t.p + widths*bra_t.p
    delx_int = Sij * (2. * widths * widths * dx - I * psum) / (widths + widths)

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

    dx = bra_t.x - ket_t.x
    dp = bra_t.p - ket_t.p
    delp_int = Sij * (dp + 2.*I*widths*dx) / (2*(widths+widths))
 
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

