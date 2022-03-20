module taylormod
 use intmod 
 use lvc_ints
 
   type, extends(integral) :: taylor
     integer(sik) :: integral_ordr  = 0
     contains
       procedure, public :: overlap   => overlap_taylor 
       procedure, public :: kinetic   => kinetic_taylor
       procedure, public :: potential => potential_taylor
       procedure, public :: sdot      => sdot_taylor 
       procedure, public :: pop       => pop_taylor
   end type taylor

 contains

  function overlap_taylor(self, bra_t, ket_t, Snuc) result(S)
    class(taylor)                      :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: S

    if(bra_t%state == ket_t%state) then
      S = Snuc
    else
      S = cmplx(0.,0.)
    endif

    return
  end function overlap_taylor


  function kinetic_taylor(self, bra_t, ket_t, Snuc) result(T)
    class(taylor)                      :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: T

    if(bra_t%state == ket_t%state) then
        T = nuc_kinetic(bra_t, ket_t, -0.5/bra_t%mass, Snuc)
    else
        T = cmplx(0.,0.)
    endif

    return
  end function kinetic_taylor

  !
  !
  !
  function sdot_taylor(self, bra_t, ket_t, Snuc) result(delS)
    class(taylor)                      :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: delS
    real(drk)                          :: vel(self%nc), force(self%nc)

    delS = zero_c
    if(bra_t%state == ket_t%state) then
      vel   = tr_velocity(ket_t)
      force = tr_force(ket_t)
      delS  = nuc_sdot(bra_t, ket_t, Snuc, vel, force)
    endif

    return
  end function sdot_taylor

  !
  !
  !
  function pop_taylor(self, nst, bra_t, ket_t, Snuc) result(spop)
    class(taylor)                  :: self
    integer(ik), intent(in)        :: nst
    type(trajectory), intent(in)   :: bra_t
    type(trajectory), intent(in)   :: ket_t
    complex(drk), intent(in)       :: Snuc

    complex(drk)                   :: spop(nst)

    spop = zero_drk
    if(bra_t.state == ket_t.state) spop(bra_t.state) = Snuc 

    return
  end function pop_taylor

  !
  !
  !
  function potential_taylor(self, bra_t, ket_t, Snuc) result(V)
    class(taylor)                 :: self
    type(trajectory), intent(in)  :: bra_t
    type(trajectory), intent(in)  :: ket_t
    complex(drk), intent(in)      :: Snuc

    complex(drk)                  :: V
    complex(drk)                  :: bk(size(bra_t%x)),bl(size(bra_t%x))
    complex(drk)                  :: ck, cl
    complex(drk)                  :: vij, vji, Sji
    complex(drk)                  :: o1_ij(size(bra_t%x)), o1_ji(size(bra_t%x))
    real(drk)                     :: f_ij(size(bra_t%x)), f_ji(size(bra_t%x))
    integer(ik)                   :: state, bra, ket
    real(drk)                     :: alpha(size(bra_t%x), size(bra_t%x))
    real(drk)                     :: q(size(bra_t%x))
    integer(sik)                  :: i

    Sji = conjg(Snuc)

    if(bra_t%state == ket_t%state) then
      state = bra_t%state
      vij   = bra_t%energy(state) * Snuc
      vji   = ket_t%energy(state) * Sji

      if(self%integral_ordr > 0) then
        alpha = zero_drk
        q     = one_drk
        do i = 1,size(bra_t%x)
          alpha(i,i) = bra_t%width(i)
        enddo
        call extract_gauss_params(bra_t, bk, ck)
        call extract_gauss_params(ket_t, bl, cl)
        o1_ij = Snuc * exact_poly(alpha, bk, bl, 0.d0, q)
        o1_ji = Sji  * exact_poly(alpha, bl, bk, 0.d0, q)
        vij   = vij + dot_product(o1_ij - bra_t%x*Snuc, bra_t%deriv(:, state, state))
        vji   = vji + dot_product(o1_ji - ket_t%x*Sji, ket_t%deriv(:, state, state))
      endif

      if(self%integral_ordr > 1) then
        print *,'second ordr not yet implemented'
      endif

    else

      bra    = bra_t%state
      ket    = ket_t%state
      f_ij   = bra_t.deriv(:, bra, ket)
      f_ji   = ket_t.deriv(:, ket, bra)
      o1_ij  = 0.5 * nuc_delx(bra_t, ket_t, Snuc) / bra_t%mass
      o1_ji  = 0.5 * nuc_delx(ket_t, bra_t, Sji) / ket_t%mass

      vij   = 2.*dot_product(f_ij, o1_ij)
      vji   = 2.*dot_product(f_ji, o1_ji)

    endif

    V = 0.5 * (vij + conjg(vji))

    return
  end function potential_taylor

end module taylormod

