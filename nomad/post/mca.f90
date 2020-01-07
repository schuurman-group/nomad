!
! Builds the Hamiltonian and other requisite quantities
! necessary to propagate the TDSE using the moving crude adiabatic
! approach
!
! M. S. Schuurman, Jan. 4, 2020
!
!
module fms
  use libprop
  implicit none

  private
  public  overlap 
  public  ke
  public  potential
  public  sdot
  public  delx
  public  delp

  interface potential  
    module procedure potential_taylor
  end interface

  !
 
 contains

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


    return
  end function ke

  !
  !
  !
  function potential_taylor(bra_t, ket_t, Sij) result(pot_int)
    type(trajectory), intent(in)  :: bra_t, ket_t
    complex(drk), intent(in)      :: Sij

    complex(drk)                  :: pot_int 
 

    return
  end function potential_taylor

  !
  !
  !
  function sdot(bra_t, ket_t, Sij) result(sdot_int)
    type(trajectory), intent(in)  :: bra_t, ket_t
    complex(drk), intent(in)      :: Sij

    complex(drk)                  :: sdot_int


    return
  end function sdot

  !
  !
  !
  function delx(bra_t, ket_t, Sij) result(delx_int)
    type(trajectory), intent(in)  :: bra_t, ket_t
    complex(drk), intent(in)      :: Sij
    
    complex(drk)                  :: delx_int(n_crd)


    return
  end function delx

  !
  !
  !
  function delp(bra_t, ket_t, Sij) result(delp_int)
    type(trajectory), intent(in)  :: bra_t, ket_t
    complex(drk), intent(in)      :: Sij
 
    complex(drk)                  :: delp_int(n_crd)

    return
  end function delp



end module fms 

