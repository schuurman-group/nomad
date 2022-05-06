module intmod
  use accuracy
  use math
  use gauss_ints
  implicit none
  
   type, public  :: integral
     integer(ik)         :: nc

     contains
       procedure, public  :: init      => init_dummy
       procedure, public  :: overlap   => overlap
       procedure, public  :: kinetic   => kinetic
       procedure, public  :: potential => potential
       procedure, public  :: sdot      => sdot
       procedure, public  :: pop       => pop

   end type integral

 contains

  subroutine init_dummy(self, alpha, omega, focons, scalars)
    class(integral)                    :: self
    real(drk), intent(in)              :: alpha(:)
    real(drk), intent(in)              :: omega(:)
    real(drk), intent(in)              :: focons(:,:)
    real(drk), intent(in)              :: scalars(:)
    

  end subroutine init_dummy

  function overlap(self, bra_t, ket_t, Snuc) result(S)
    class(integral)                    :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: S

    return
  end function overlap


  function kinetic(self, bra_t, ket_t, Snuc) result(T)
    class(integral)                    :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: T

    return
  end function kinetic

  !
  !
  !
  function sdot(self, bra_t, ket_t, Snuc) result(delS)
    class(integral)                    :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: delS

    return
  end function sdot

  !
  !
  !
  function pop(self, nst, bra_t, ket_t, Snuc) result(spop)
    class(integral)                    :: self
    integer(ik), intent(in)            :: nst
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: spop(nst)

    return
  end function pop

  !
  !
  !
  function potential(self, bra_t, ket_t, Snuc) result(V)
    class(integral)               :: self
    type(trajectory), intent(in)  :: bra_t
    type(trajectory), intent(in)  :: ket_t
    complex(drk), intent(in)      :: Snuc

    complex(drk)                  :: V

    return
  end function potential

end module intmod

