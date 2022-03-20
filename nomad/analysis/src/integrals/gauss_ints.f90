!
! gauss_ints -- a module that returns integrals over gaussian function
!
! M. S. Schuurman -- Dec. 27, 2019
!
module gauss_ints
  use accuracy
  use math
  use libtraj

  public nuc_overlap_origin
  public nuc_overlap
  public nuc_kinetic
  public nuc_delx
  public nuc_delp
  public nuc_sdot
  public nuc_density

 contains

  !
  ! Computes _just_ the nuclear part of the overlap integral
  !
  function nuc_overlap(bra_t, ket_t) result(S)
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t

    complex(drk)                       :: S
    real(drk)                          :: dx(size(bra_t%x)),dp(size(bra_t%x))
    real(drk)                          :: a1(size(bra_t%x)),a2(size(bra_t%x))
    real(drk)                          :: prefactor(size(bra_t%x))
    real(drk)                          :: x_center(size(bra_t%x))
    real(drk)                          :: real_part(size(bra_t%x))
    real(drk)                          :: imag_part(size(bra_t%x))

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
  ! Computes _just_ the nuclear part of the overlap integral
  !
  function nuc_overlap_origin(bra_t) result(S)
    type(trajectory), intent(in)       :: bra_t

    complex(drk)                       :: S
    real(drk)                          :: dx(size(bra_t%x)),dp(size(bra_t%x))
    real(drk)                          :: a1(size(bra_t%x)),a2(size(bra_t%x))
    real(drk)                          :: prefactor(size(bra_t%x))
    real(drk)                          :: x_center(size(bra_t%x))
    real(drk)                          :: real_part(size(bra_t%x))
    real(drk)                          :: imag_part(size(bra_t%x))
    
    a1        = bra_t%width
    a2        = bra_t%width
    dx        = bra_t%x
    dp        = bra_t%p
    prefactor = sqrt(2. * sqrt(a1*a2) / (a1+a2))
    x_center  = (a1 * bra_t%x) / (a1+a2)
    real_part = (a1*a2*dx**2 + 0.25*dp**2) / (a1+a2)
    imag_part = (bra_t%x * bra_t%p) - x_center * dp

    S         = product(prefactor) * exp(sum(-real_part + I_drk*imag_part))

    return
  end function nuc_overlap_origin

  !
  !
  ! 
  function nuc_kinetic(bra_t, ket_t, kecoef, Sij) result(ke_int)
    type(trajectory), intent(in)  :: bra_t, ket_t
    real(drk), intent(in)         :: kecoef(:)
    complex(drk), intent(in)      :: Sij

    complex(drk)                  :: ke_int
    complex(drk)                  :: ke_vec(size(bra_t%x))
    real(drk   )                  :: dx(size(bra_t%x)), psum(size(bra_t%x))
    real(drk)                     :: a1(size(bra_t%x)), a2(size(bra_t%x))

    if(bra_t%state /= ket_t%state) then
      ke_int = zero_c
      return
    endif

    a1     = bra_t%width
    a2     = ket_t%width
    dx     = bra_t%x - ket_t%x
    psum   = a1*ket_t%p + a2*bra_t%p
    ke_vec = Sij * (-4.d0*a1*a2*dx*psum*I_drk - 2.d0*a1*a2*(a1+a2) + &
                     4.d0*dx**2 * a1**2 * a2**2 - psum**2) / (a1+a2)**2

    ke_int = dot_product(conjg(cmplx(kecoef(:size(bra_t%x)))), ke_vec)
    return
  end function nuc_kinetic

  !
  !
  !
  function nuc_delx(bra_t, ket_t, Sij) result(delx_int)
    type(trajectory), intent(in)  :: bra_t, ket_t
    complex(drk), intent(in)      :: Sij

    complex(drk)                  :: delx_int(size(bra_t%x))
    real(drk)                     :: dx(size(bra_t%x)), psum(size(bra_t%x))
    real(drk)                     :: a1(size(bra_t%x)), a2(size(bra_t%x))

    a1    = bra_t%width
    a2    = ket_t%width
    dx    = bra_t%x - ket_t%x
    psum  = a1*ket_t%p + a2*bra_t%p
    delx_int = Sij * (2.d0 * a1 * a2 * dx - I_drk * psum) / (a1+a2)

    return
  end function nuc_delx

  !
  !
  !
  function nuc_delp(bra_t, ket_t, Sij) result(delp_int)
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
  end function nuc_delp

  !
  !
  !
  function nuc_sdot(bra_t, ket_t, Sij, vel, force) result(sdot_int)
    type(trajectory), intent(in)  :: bra_t, ket_t
    complex(drk), intent(in)      :: Sij
    real(drk), intent(in)         :: vel(:), force(:)

    complex(drk)                  :: sdot_int
    complex(drk)                  :: deldx(size(bra_t%x)), deldp(size(bra_t%x))
    integer(sik)                  :: i, nc

    nc     = size(bra_t%x)
 
    deldx  = nuc_delx(bra_t, ket_t, Sij)
    deldp  = nuc_delp(bra_t, ket_t, Sij)

    ! dot_product evaluates conjg(x).y, need to undo that
    sdot_int = dot_product(conjg(deldx), cmplx(vel(:nc))) + dot_product(conjg(deldp), cmplx(force(:nc)))

    return
  end function nuc_sdot

  !
  !
  !
  function nuc_density(bra_t, ket_t, pt) result(den)
    type(trajectory), intent(in)        :: bra_t
    type(trajectory), intent(in)        :: ket_t
    real(drk), intent(in)               :: pt(size(bra_t%x))
   
    complex(drk)                        :: den
    complex(drk)                        :: den_vec(size(bra_t%x))
    real(drk)                           :: argr(size(bra_t%x))
    real(drk)                           :: argi(size(bra_t%x))
    real(drk)                           :: a1(size(bra_t%x))
    real(drk)                           :: a2(size(bra_t%x))
 
    a1      = bra_t%width
    a2      = ket_t%width
    argr    = -a1*(pt - bra_t%x)**2   - a2*(pt - ket_t%x)**2
    argi    = -bra_t%p*(pt - bra_t%x) + ket_t%p*(pt - ket_t%x)
    den_vec = ((4*a1*a2/(pi**2))**0.25) * exp(argr + I_drk*argi)
    den     = product(den_vec)

    return
  end function nuc_density


end module gauss_ints

