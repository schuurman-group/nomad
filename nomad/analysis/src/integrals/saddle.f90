module saddlemod
  use intmod  

   type, extends(integral) :: saddle

     contains
       procedure, public :: overlap   => overlap_saddle
       procedure, public :: kinetic   => kinetic_saddle
       procedure, public :: potential => potential_saddle
       procedure, public :: sdot      => sdot_saddle
       procedure, public :: pop       => pop_saddle
   end type saddle 

 contains

  function overlap_saddle(self, bra_t, ket_t, Snuc) result(S)
    class(saddle)                      :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: S

    S = zero_c
    if(bra_t%state == ket_t%state) S = Snuc

    return
  end function overlap_saddle


  function kinetic_saddle(self, bra_t, ket_t, Snuc) result(T)
    class(saddle)                      :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: T

    T = zero_c
    if(bra_t%state == ket_t%state) T = nuc_kinetic(bra_t, ket_t, -0.5/bra_t%mass, Snuc)

    return
  end function kinetic_saddle

  !
  !
  !
  function sdot_saddle(self, bra_t, ket_t, Snuc) result(delS)
    class(saddle)                      :: self
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
  end function sdot_saddle

  !
  !
  !
  function pop_saddle(self, nst, bra_t, ket_t, Snuc) result(spop)
    class(saddle)                  :: self
    integer(ik), intent(in)        :: nst
    type(trajectory), intent(in)   :: bra_t
    type(trajectory), intent(in)   :: ket_t
    complex(drk), intent(in)       :: Snuc

    complex(drk)                      :: spop(nst)

    spop = zero_drk
    if(bra_t.state == ket_t.state) spop(bra_t.state) = Snuc

    return
  end function pop_saddle

  !
  !
  !
  function potential_saddle(self, bra_t, ket_t, Snuc) result(V)
    class(saddle)                 :: self
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
  end function potential_saddle

end module saddlemod

