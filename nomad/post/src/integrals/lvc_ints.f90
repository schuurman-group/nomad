module lvc_ints
  use quadrature
  use lvcmod
  use libtraj
  use timer
  use math
  implicit none

  public exact_poly
  public exact_delta
  public exact_dboc
  public exact_nac
  public exact_pop
  public extract_gauss_params

  contains

   !
   !
   function exact_poly(alpha, bk, bl, ea, a, aa) result(poly)
     real(drk), intent(in)             :: alpha(:,:)
     complex(drk), intent(in)          :: bk(:)
     complex(drk), intent(in)          :: bl(:)
     real(drk), intent(in)             :: ea
     real(drk), intent(in)             :: a(:)
     real(drk), optional, intent(in)   :: aa(:,:)

     complex(drk)                      :: poly
     complex(drk)                      :: bkl(size(bk))
     complex(drk)                      :: bf(size(bk))
     integer(ik)                       :: n,i
     real(drk)                         :: ainv(size(alpha,dim=2),size(alpha,dim=1))
     real(drk)                         :: atmp(size(alpha,dim=2),size(alpha,dim=1))

     call TimerStart('lvc_ints:exact_poly')

     n = size(alpha, dim=1)

     bkl  = conjg(bk) + bl
     ainv = inverse_gelss(n, alpha)
     bf   = matvec_prod(ainv, bkl)

     poly = ea + dot_product(conjg(cmplx(a)), bf) / 2.
      
     if(present(aa)) then
       atmp = matmul(ainv, aa)
       do i = 1,n
         poly = poly + atmp(i,i) / 2.
       enddo
       poly = poly + dot_product(conjg(bf), matvec_prod(aa, bf)) / 4.
     endif

     call TimerStop('lvc_ints:exact_poly')
     return
   end function exact_poly

   !
   !
   function exact_delta(beta, d_alpha) result(delta)
     complex(drk), intent(in)          :: beta(2)
     real(drk), intent(in)             :: d_alpha(2) 

     complex(drk)                      :: delta
     real(drk)                         :: delta_int(2)
     real(drk)                         :: delta_a
     real(drk)                         :: ul
     real(drk)                         :: error
     real(drk)                         :: toler = 1.d-8
     real(drk)                         :: err_r, err_i
     integer(ik)                       :: ier
     integer(ik)                       :: neval_r, neval_i

     call TimerStart('lvc_ints:exact_delta')

     delta   = zero_c

     ! if d_alpha[x] == 0, integral is 0
     if (d_alpha(1) == 0.d0) return
     delta_a = d_alpha(2) - d_alpha(1)
     ul      = 1.d0 / sqrt(d_alpha(1))

     ! integrate the real and imaginary parts separately
     call quadpack_int(integrand_real, 0.d0, ul, 0.1*toler, 0.d0, delta_int(1), err_r, neval=neval_r, ier=ier, limit=4096)
     if(ier /= 0) then
       print *,QUADPACK_ERROR(ier)
       stop
     endif

     call quadpack_int(integrand_imag, 0.d0, ul, 0.1*toler, 0.d0, delta_int(2), err_i, neval=neval_i, ier=ier, limit=4096)
     if(ier /= 0) then
       print *,QUADPACK_ERROR(ier)
       stop
     endif

     error = 0.5*(abs(err_r) + abs(err_i))
     delta = cmplx(delta_int(1), delta_int(2))

     if(error.gt.toler) then
        print *,' delta integral not converged: ',error,' > ',toler
        stop 'delta integral did not converge'
     endif

     call TimerStop('lvc_ints:exact_delta')
     return

     contains

      function integrand_real(x) result(f)
        real(drk), intent(in)        :: x 

        real(drk)                    :: f
        complex(drk)                 :: zf
        real(drk)                    :: d1, d2

        d1 = 1.d0 + delta_a * x**2
        d2 = 1.d0 - d_alpha(1) * x**2

        zf = d_alpha(1) + d_alpha(2)/d1 + 0.5d0 * d2 * (d_alpha(1)*beta(1)**2 + d_alpha(2)*beta(2)**2 / d1**2)
        zf = zf * eff_overlap(x, beta, d_alpha) / sqrt(d2)
        f  = real(zf)
     end function integrand_real

     function integrand_imag(x) result(f)
        real(drk), intent(in)        :: x

        real(drk)                    :: f
        complex(drk)                 :: zf
        real(drk)                    :: d1, d2

        d1 = 1.d0 + delta_a * x**2
        d2 = 1.d0 - d_alpha(1) * x**2

        zf = d_alpha(1) + d_alpha(2)/d1 + 0.5d0 * d2 * (d_alpha(1)*beta(1)**2 + d_alpha(2)*beta(2)**2 / d1**2)
        zf = zf * eff_overlap(x, beta, d_alpha) / sqrt(d2)
        f  = aimag(zf)
     end function integrand_imag

   end function exact_delta

   !
   !
   function exact_nac(beta, d_alpha, pxy) result(nac)
     complex(drk), intent(in)         :: beta(2)
     real(drk),    intent(in)         :: d_alpha(2)
     complex(drk), intent(in)         :: pxy(2)

     complex(drk)                      :: nac
     real(drk)                         :: nac_int(2)
     real(drk)                         :: delta_a
     real(drk)                         :: ul
     real(drk)                         :: error
     real(drk)                         :: toler = 1.d-8
     real(drk)                         :: err_r, err_i
     integer(ik)                       :: ier
     integer(ik)                       :: neval_r, neval_i

     call TimerStart('lvc_ints:exact_nac')

     nac = zero_c

     ! if d_alpha[x] == 0, integral is 0
     if (d_alpha(1) == 0.d0) return
     delta_a = d_alpha(2) - d_alpha(1)
     ul      = 1.d0 / sqrt(d_alpha(1))

     ! integrate the real and imaginary parts separately
     call quadpack_int(integrand_real, 0.d0, ul, 0.1*toler, 0.d0, nac_int(1), err_r, neval=neval_r, ier=ier, limit=4096)
     if(ier /= 0) then
       print *,QUADPACK_ERROR(ier)
       stop
     endif

     call quadpack_int(integrand_imag, 0.d0, ul, 0.1*toler, 0.d0, nac_int(2), err_i, neval=neval_i, ier=ier, limit=4096)
     if(ier /= 0) then
       print *,QUADPACK_ERROR(ier)
       stop
     endif

     error = 0.5*(abs(err_r) + abs(err_i))
     nac = cmplx(nac_int(1), nac_int(2))

     if(error.gt.toler) then
        print *,' NAC integral not converged: ',error,' > ',toler
        stop 'NAC integral did not converge'
     endif

     call TimerStop('lvc_ints:exact_nac')
     return

     contains

      function integrand_real(x) result(f)
        real(drk), intent(in)        :: x

        real(drk)                    :: f
        complex(drk)                 :: zf
        real(drk)                    :: d1, d2

        d1 = 1.d0 + delta_a * x**2
        d2 = 1.d0 - d_alpha(1) * x**2

        zf = x * eff_overlap(x, beta, d_alpha) * ( pxy(1)*beta(1) + pxy(2)*beta(2)/d1 ) / d2
        f  = real(zf)
     end function integrand_real

     function integrand_imag(x) result(f)
        real(drk), intent(in)        :: x

        real(drk)                    :: f
        complex(drk)                 :: zf
        real(drk)                    :: d1, d2

        d1 = 1.d0 + delta_a * x**2
        d2 = 1.d0 - d_alpha(1) * x**2

        zf = x * eff_overlap(x, beta, d_alpha) * ( pxy(1)*beta(1) + pxy(2)*beta(2)/d1 ) / d2
        f  = aimag(zf)
     end function integrand_imag

   end function exact_nac

   !
   !
   function exact_dboc(beta, d_alpha, kxy) result(dboc)
     complex(drk), intent(in)         :: beta(2)
     real(drk),    intent(in)         :: d_alpha(2)
     real(drk),    intent(in)         :: kxy(2,2)    

     complex(drk)                     :: dboc
     complex(drk)                     :: dboc_div
     real(drk)                        :: dboc_int(2)
     complex(drk)                     :: Plim
     real(drk)                        :: delta_a
     real(drk)                        :: delta
     real(drk)                        :: dscale
     real(drk)                        :: ul
     real(drk)                        :: error
     real(drk)                        :: err_r, err_i
     real(drk)                        :: toler = 1.d-7
     integer(ik)                      :: ier
     integer(ik)                      :: neval_r, neval_i
 

     call TimerStart('lvc_ints:exact_dboc')

     dboc = zero_c

     dscale  = 1.d3
     ! if d_alpha[x] == 0, integral is 0
     if (d_alpha(1) == 0.d0) return

     delta_a = d_alpha(2) - d_alpha(1)
     ul      = 1.d0 / sqrt(d_alpha(1))

     ! determine asymptotic component
     Plim = (kxy(1,1) / (d_alpha(1)*sqrt(d_alpha(2))) + kxy(2,2) / d_alpha(2)**1.5d0) * exp(-dot_product(conjg(beta),beta)/4.d0)

     ! integrate the real and imaginary parts separately
     call quadpack_int(integrand_real, 0.d0, ul, 0.1*toler, 0.d0, dboc_int(1), err_r, neval=neval_r, ier=ier, limit=4096)
     if(ier /= 0) then
       print *,QUADPACK_ERROR(ier)
       stop
     endif

     call quadpack_int(integrand_imag, 0.d0, ul, 0.1*toler, 0.d0, dboc_int(2), err_i, neval=neval_i, ier=ier, limit=4096)
     if(ier /= 0) then
       print *,QUADPACK_ERROR(ier)
       stop
     endif

     error = 0.5*(abs(err_r) + abs(err_i))
     dboc  = cmplx(dboc_int(1), dboc_int(2))

     !print *,'error-total=',error
     !print *,'aerr_r, aerr_i=',err_r, err_i
     !print *,'neval_r, neval_i=',neval_r,neval_i

     if(error.gt.toler) then
        print *,' dboc integral not converged: ',error,' > ',toler
        stop 'dboc integral did not converge'
     endif

     ! approximate the divergent part -- will be a function of delta
     delta    = d_alpha(1) / dscale 
     dboc_div = Plim / (2*sqrt(d_alpha(1))) * (log(4*d_alpha(1)) - 2*log(delta) - euler)

     ! total integral is convergent part + approximated divergent component
     !print *,'dboc, dboc_div=',dboc,dboc_div
     dboc = dboc + dboc_div

     call TimerStop('lvc_ints:exact_dboc')
     return

     contains

      function integrand_real(x) result(f)
        real(drk), intent(in)       :: x
        real(drk)                   :: f
 
        complex(drk)                :: zf
        real(drk)                   :: d1, d2
        complex(drk)                :: earg
 
        d1   = 1.d0 + delta_a * x**2
        d2   = 1.d0 - d_alpha(1) * x**2
        earg = -0.25d0 * x**2 * (d_alpha(1)*beta(1)**2 + (d_alpha(2)*beta(2)**2) / d1)

        ! divergent integrals 
        zf = (kxy(1,1) + kxy(2,2)/d1) / d2

        ! convergent integrals
        zf = zf - 0.5d0*( kxy(1,1)*beta(1)**2 + (kxy(2,2)*beta(2)**2)/d1**2 + 2.d0*kxy(1,2)*beta(1)*beta(2)/d1 )

        ! common factors
        zf = x**3 * exp(earg) * zf / sqrt(d1)

        ! subtract asymptotic part 
        zf = Plim/d2 - zf 

        ! convergent integrals
        f = real(zf)
      end function integrand_real
 
      function integrand_imag(x) result(f)
        real(drk), intent(in)       :: x
        real(drk)                   :: f

        complex(drk)                :: zf
        real(drk)                   :: d1, d2
        complex(drk)                :: earg

        d1   = 1.d0 + delta_a * x**2
        d2   = 1.d0 - d_alpha(1) * x**2
        earg = -0.25d0 * x**2 * (d_alpha(1)*beta(1)**2 + (d_alpha(2)*beta(2)**2) / d1)

        ! divergent integrals 
        zf = ( kxy(1,1) + kxy(2,2)/d1 ) / d2

        ! convergent integrals
        zf = zf - 0.5d0*( kxy(1,1)*beta(1)**2 + (kxy(2,2)*beta(2)**2)/d1**2 + 2.d0*kxy(1,2)*beta(1)*beta(2)/d1 )

        ! common factors
        zf = x**3 * exp(earg) * zf / sqrt(d1)

        ! subtract asymptotic part 
        zf = Plim/d2 - zf

        ! convergent integrals
        f = aimag(zf)
      end function integrand_imag

   end function exact_dboc

  !
  !
  subroutine exact_pop(beta, d_alpha, b_alpha, zme, xme)
     complex(drk), intent(in)         :: beta(2)
     real(drk),    intent(in)         :: d_alpha(2)
     real(drk),    intent(in)         :: b_alpha(2,2)   
     complex(drk), intent(out)        :: zme
     complex(drk), intent(out)        :: xme

     complex(drk)                      :: pop
     real(drk)                         :: pop_int(2)
     complex(drk)                      :: beta_zx(2)
     real(drk)                         :: delta_a
     real(drk)                         :: ul
     real(drk)                         :: error
     real(drk)                         :: toler = 1.d-8
     integer(sik)                      :: zx
     real(drk)                         :: err_r, err_i
     integer(ik)                       :: ier
     integer(ik)                       :: neval_r, neval_i

     call TimerStart('lvc_ints:exact_pop')

     zme = zero_c
     xme = zero_c

     !print *,'beta=',beta
     !print *,'d_alpha=',d_alpha
     !print *,'b_alpha(:,1)=',b_alpha(:,1)
     !print *,'b_alpha(:,2)=',b_alpha(:,2)

     ! if d_alpha[x] == 0, integral is 0
     if (d_alpha(1) == 0.d0) return
     delta_a = d_alpha(2) - d_alpha(1)
     ul      = 1.d0 / sqrt(d_alpha(1))

     ! integrate integrand_delta from 0 to ul
     do zx = 1,2

       ! set the Z or X contributions
       beta_zx = cmplx(b_alpha(:,zx))
   
       ! integrate the real and imaginary parts separately
       call quadpack_int(integrand_real, 0.d0, ul, 0.1*toler, 0.d0, pop_int(1), err_r, neval=neval_r, ier=ier, limit=4096)
       if(ier /= 0) then
         print *,QUADPACK_ERROR(ier)
         stop
       endif

       call quadpack_int(integrand_imag, 0.d0, ul, 0.1*toler, 0.d0, pop_int(2), err_i, neval=neval_i, ier=ier, limit=4096)
       if(ier /= 0) then
         print *,QUADPACK_ERROR(ier)
         stop
       endif

       pop   = cmplx(pop_int(1), pop_int(2)) / sqrt(pi)
       error = 0.5*(abs(err_r) + abs(err_i))

       if(error.gt.toler) then
         print *,' pop integral not converged: ',error,' > ',toler
         stop 'pop integral did not converge'
       endif

       if(zx == 1) then
         zme = pop
       else
         xme = pop
       endif

     enddo

     call TimerStop('lvc_ints:exact_pop')
     return

     contains

       function integrand_real(x) result (f)
         real(drk), intent(in)        :: x
         real(drk)                    :: f

         complex(drk)                 :: zf
         real(drk)                    :: d1, d2

         d1 = 1.d0 + delta_a * x**2
         d2 = 1.d0 - d_alpha(1)  * x**2
         zf = eff_overlap(x, beta, d_alpha) * (beta(1)*beta_zx(1) + beta(2)*beta_zx(2)/d1) / sqrt(d2)
         f  = real(zf)
       end function integrand_real

       function integrand_imag(x) result (f)
         real(drk), intent(in)        :: x
         real(drk)                    :: f

         complex(drk)                 :: zf
         real(drk)                    :: d1, d2

         d1 = 1.d0 + delta_a * x**2
         d2 = 1.d0 - d_alpha(1)  * x**2
         zf = eff_overlap(x, beta, d_alpha) * (beta(1)*beta_zx(1) + beta(2)*beta_zx(2)/d1) / sqrt(d2)
         f  = aimag(zf)
       end function integrand_imag

  end subroutine exact_pop

!-----------------------------------------------------------------
!
!  private routines
!
  function eff_overlap(u, beta, d_alpha) result(Sd)
    real(drk), intent(in)     :: u
    complex(drk), intent(in)  :: beta(2)
    real(drk), intent(in)     :: d_alpha(2)

    complex(drk)              :: Sd
    real(drk)                 :: delta_a
    real(drk)                 :: denom
    real(drk)                 :: fac
    complex(drk)              :: earg

    delta_a = d_alpha(2) - d_alpha(1)
    denom   = 1.d0 + delta_a * u**2
    fac     = (1.d0 - d_alpha(1) * u**2) / sqrt(denom)
    earg    = -0.25d0 * u**2 * (d_alpha(1)*beta(1)**2 + d_alpha(2)*beta(2)**2 / denom)
    Sd      = fac * exp(earg)

    return
  end function eff_overlap

  ! 
  !
  ! 
  subroutine extract_gauss_params(traj, b, c)
    type(trajectory), intent(in) :: traj

    complex(drk), intent(out)    :: b(size(traj%x))
    complex(drk), intent(out)    :: c

    b = 2.d0 * traj%width * traj%x + I_drk * traj%p
    c = -dot_product(traj%x, traj%width*traj%x) - I_drk*dot_product(traj%x, traj%p)

    return
  end subroutine extract_gauss_params

  !
  !
  subroutine shift_gauss_params(x_shft, alpha, bold, cold, bnew, cnew)
    real(drk), intent(in)        :: x_shft(:)
    real(drk), intent(in)        :: alpha(:)
    complex(drk), intent(in)     :: bold(size(x_shft))
    complex(drk), intent(in)     :: cold
    complex(drk), intent(out)    :: bnew(size(x_shft))
    complex(drk), intent(out)    :: cnew
    
    real(drk)                    :: shft_vec(size(x_shft))

    shft_vec = -alpha * x_shft

    bnew = bold + 2.d0 * shft_vec
    cnew = cold + dot_product(x_shft, bold) + dot_product(x_shft, shft_vec)

    return
  end subroutine shift_gauss_params

end module lvc_ints



