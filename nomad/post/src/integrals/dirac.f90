module diracmod 
  use intmod
  
   type, extends(integral) :: dirac 

     contains
       procedure, public  :: overlap   => overlap_dirac
       procedure, public  :: kinetic   => kinetic_dirac
       procedure, public  :: potential => potential_dirac
       procedure, public  :: sdot      => sdot_dirac
       procedure, public  :: pop       => pop_dirac
   end type dirac

 contains

  function overlap_dirac(self, bra_t, ket_t, Snuc) result(S)
    class(dirac)                       :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: S

    S = zero_c
    if(bra_t%state == ket_t%state) S = Snuc

    return
  end function overlap_dirac


  function kinetic_dirac(self, bra_t, ket_t, Snuc) result(T)
    class(dirac)                       :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: T

    T = zero_C    
    if(bra_t%state == ket_t%state) T = nuc_kinetic(bra_t, ket_t, -0.5/bra_t%mass, Snuc)

    return
  end function kinetic_dirac

  !
  !
  !
  function sdot_dirac(self, bra_t, ket_t, Snuc) result(delS)
    class(dirac)                       :: self
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
  end function sdot_dirac

  !
  !
  !
  function pop_dirac(self, nst, bra_t, ket_t, Snuc) result(spop)
    class(dirac)                   :: self
    integer(ik), intent(in)        :: nst
    type(trajectory), intent(in)   :: bra_t
    type(trajectory), intent(in)   :: ket_t
    complex(drk), intent(in)       :: Snuc

    complex(drk)                      :: spop(nst)

    spop = zero_drk
    if(bra_t.state == ket_t.state) spop(bra_t.state) = 1.d0 

    return
  end function pop_dirac

  !
  !
  !
  function potential_dirac(self, bra_t, ket_t, Snuc) result(V)
    class(dirac)                  :: self
    type(trajectory), intent(in)  :: bra_t
    type(trajectory), intent(in)  :: ket_t
    complex(drk), intent(in)      :: Snuc

    complex(drk)                  :: V
    complex(drk)                  :: vij, vji, Sji
    complex(drk)                  :: o1_ij(size(bra_t%x)), o1_ji(size(bra_t%x))
    real(drk)                     :: f_ij(size(bra_t%x)), f_ji(size(bra_t%x))
    integer(ik)                   :: state, bra, ket

    V = zero_c

    return
  end function potential_dirac

end module diracmod 

