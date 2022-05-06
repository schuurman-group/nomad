!
! gauss_ints -- a module that returns integrals over gaussian function
!
! M. S. Schuurman -- Dec. 27, 2019
!
module lvc_exactmod 
  use intmod
  use lvcmod
  use lvc_ints 
  implicit none

   type, extends(integral) :: lvc_exact
     ! whether or not to include dboc term
     logical                    :: include_dboc
     type(lvc)                  :: model

     contains
       procedure, public  :: init      => init_exact
       procedure, public  :: overlap   => overlap_exact
       procedure, public  :: kinetic   => kinetic_exact
       procedure, public  :: potential => potential_exact
       procedure, public  :: sdot      => sdot_exact
       procedure, public  :: pop       => pop_exact
   end type lvc_exact

 contains

  subroutine init_exact(self, alpha, omega, focons, scalars)
    class(lvc_exact)                   :: self
    real(drk), intent(in)              :: alpha(:)
    real(drk), intent(in)              :: omega(:)
    real(drk), intent(in)              :: focons(:,:)
    real(drk), intent(in)              :: scalars(:)

    self%nc = size(alpha)
    self%include_dboc = .true.
    call self%model%init(alpha, omega, focons, scalars)

    return
  end subroutine init_exact


  function overlap_exact(self, bra_t, ket_t, Snuc) result(S)
    class(lvc_exact)                   :: self
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
  end function overlap_exact

  !
  !
  function kinetic_exact(self, bra_t, ket_t, Snuc) result(T)
    class(lvc_exact)                   :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: T
    complex(drk)                       :: bk(self%nc), bl(self%nc), btmp(self%nc)
    complex(drk)                       :: bkc(self%nc), db(self%nc)
    complex(drk)                       :: ck, cl, ctmp
    complex(drk)                       :: beta(2)   
    real(drk)                          :: J(self%nc, self%nc)
    real(drk)                          :: K(self%nc, self%nc), k_alpha(2,2)
    real(drk)                          :: sgn
    complex(drk)                       :: p_alpha(2)
 
    complex(drk)                       :: nacme, dbocme


    call TimerStart('kinetic_exact')

    call extract_gauss_params(bra_t, btmp, ctmp)   
    call shift_gauss_params(self%model%qci, self%model%alpha, btmp, ctmp, bk, ck) 

    call extract_gauss_params(ket_t, btmp, ctmp)
    call shift_gauss_params(self%model%qci, self%model%alpha, btmp, ctmp, bl, cl)

    bkc   = conjg(bk)
    db    = bkc - bl
    beta  = matvec_prod(self%model%At, bkc + bl)

    ! nonadiabatic coupling matrix element
    ! ------------------------------------
    J       = 0.5*(outer_prod( self%model%z_vec, self%model%x_vec) - &
                   outer_prod( self%model%x_vec, self%model%z_vec))
    p_alpha = matvec_prod(matmul( self%model%At, transpose(J)), matvec_prod(self%model%w_mat, db))

    nacme   = Snuc * exact_nac(beta, self%model%d_alpha, p_alpha)

    ! DBOC matrix element
    ! -------------------
    if (self%include_dboc) then
        K       = matmul(matmul(J, self%model%w_mat), transpose(J))
        k_alpha = matmul(matmul(self%model%At, K), transpose(self%model%At))
        dbocme  = Snuc * exact_dboc(beta, self%model%d_alpha, k_alpha)
    else
        dbocme = zero_c
    endif

    !print *,'bra_t,ket_t, nuc, nac, dboc=',nuc_kinetic(bra_t, ket_t, -0.5*self%omega, Snuc),nacme,dbocme

    if(bra_t%state == ket_t%state) then
        T = nuc_kinetic(bra_t, ket_t, -0.5d0*self%model%omega, Snuc) + 0.5d0 * nacme * I_drk + dbocme
    else
        sgn = real(bra_t%state - ket_t%state)
        T = sgn * (-0.5d0 * nacme + dbocme * I_drk)
    endif

    call TimerStop('kinetic_exact')
    return
  end function kinetic_exact

  !
  !
  function potential_exact(self, bra_t, ket_t, Snuc) result(V)
    class(lvc_exact)                   :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: V
    complex(drk)                       :: bk(self%nc), bl(self%nc), btmp(self%nc)
    complex(drk)                       :: bkc(self%nc), db(self%nc)
    complex(drk)                       :: ck, cl, ctmp
    complex(drk)                       :: beta(2)
    integer(ik)                        :: sgn   
 
    complex(drk)                       :: w_me, delta_me

    call TimerStart('potential_exact')

    if(bra_t%state /= ket_t%state) then
      V = zero_c

    else

     call extract_gauss_params(bra_t, btmp, ctmp)
     call shift_gauss_params(self%model%qci, self%model%alpha, btmp, ctmp, bk, ck)
 
     call extract_gauss_params(ket_t, btmp, ctmp)
     call shift_gauss_params(self%model%qci, self%model%alpha, btmp, ctmp, bl, cl)

     bkc            = conjg(bk)
     beta           = matvec_prod(self%model%At, bkc + bl)

     ! average potential energy
     w_me  = Snuc * exact_poly(self%model%alpha_mat, bk, bl, self%model%ew, self%model%w_vec, aa=0.5*self%model%w_mat)

     ! delta energy contribution
     delta_me  = Snuc * exact_delta(beta, self%model%d_alpha)

     ! the average energy contribution is state independent
     V = w_me

     ! the energy shift contribution is -sigma_z
     sgn = -3 + 2 * bra_t%state
     V = V + sgn * delta_me

    endif

    call TimerStop('potential_exact')

    return  
  end function potential_exact

  !
  !
  function sdot_exact(self, bra_t, ket_t, Snuc) result(delS)
    class(lvc_exact)                   :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: delS
    real(drk)                          :: vel(self%nc), force(self%nc)

    call TimerStart('sdot_exact')

    if(bra_t.state == ket_t.state) then
      vel   = self%model%omega * ket_t%p
      force = self%model%force(ket_t%x, ket_t%state)
      delS  = nuc_sdot(bra_t, ket_t, Snuc, vel, force)
    else
      delS = zero_c
    endif

    call TimerStop('sdot_exact')

    return
  end function sdot_exact

  !
  !
  function pop_exact(self, nst, bra_t, ket_t, Snuc) result(spop)
    class(lvc_exact)                   :: self
    integer(ik), intent(in)            :: nst
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: spop(nst)

    spop = zero_drk
    if(bra_t.state == ket_t.state) spop(bra_t.state) = Snuc

    return
  end function pop_exact


end module lvc_exactmod

