!
! gauss_ints -- a module that returns integrals over gaussian function
!
! M. S. Schuurman -- Dec. 27, 2019
!
module lvc_mcamod
  use intmod 
  use lvcmod
  use lvc_ints
  use constants
  implicit none

   type                       :: cache
     integer(ik)              :: cache_size = 0
     integer(ik)              :: max_size = 0
     integer(ik)              :: batch_size = 500
     real(drk),allocatable    :: thetas(:)
     integer(ik), allocatable :: labels(:)

     contains
       procedure, public      :: init_cache
       procedure, public      :: resize_cache
       procedure, public      :: retrieve
       procedure, public      :: set
       procedure, public      :: cached
   end type cache

   type, extends(integral) :: lvc_mca
     type(lvc)             :: model

     contains
       procedure, public  :: init      => init_mca
       procedure, public  :: overlap   => overlap_mca
       procedure, public  :: kinetic   => kinetic_mca
       procedure, public  :: potential => potential_mca
       procedure, public  :: sdot      => sdot_mca
       procedure, public  :: pop       => pop_mca
   end type lvc_mca

   type(cache)                :: theta_cache

 contains

  subroutine init_mca(self, alpha, omega, focons, scalars)
    class(lvc_mca)                     :: self
    real(drk), intent(in)              :: alpha(:)
    real(drk), intent(in)              :: omega(:)
    real(drk), intent(in)              :: focons(:,:)
    real(drk), intent(in)              :: scalars(:)

    self%nc          = size(alpha)
    call self%model%init(alpha, omega, focons, scalars)

    ! initialize the theta cache
    call theta_cache%init_cache()

    return
  end subroutine init_mca

  !
  !
  !
  function overlap_mca(self, bra_t, ket_t, Snuc) result(S)
    class(lvc_mca)                     :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc
 
    complex(drk)                       :: S

    S = Snuc * elec_overlap(self, bra_t, ket_t)

    return
  end function overlap_mca

  !
  !
  function kinetic_mca(self, bra_t, ket_t, Snuc) result(T)
    class(lvc_mca)                     :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: T
    complex(drk)                       :: t_int

    t_int = nuc_kinetic(bra_t, ket_t, -0.5d0*self%model%omega, Snuc)
    T     = dot_product( conjg(phi(self, bra_t)*t_int), phi(self, ket_t))
    T     = T * elec_overlap(self, bra_t, ket_t)

    return
  end function kinetic_mca

  !
  !
  function potential_mca(self, bra_t, ket_t, Snuc) result(V)
    class(lvc_mca)                     :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: V
    complex(drk)                       :: bk(self%nc), bl(self%nc), btmp(self%nc)
    complex(drk)                       :: bkc(self%nc), db(self%nc)
    complex(drk)                       :: ck, cl, ctmp
    complex(drk)                       :: beta(2)
    complex(drk)                       :: v_mat(2,2), wme, xme, zme
    integer(ik)                        :: sgn

    call extract_gauss_params(bra_t, btmp, ctmp)
    call shift_gauss_params(self%model%qci, self%model%alpha, btmp, ctmp, bk, ck)

    call extract_gauss_params(ket_t, btmp, ctmp)
    call shift_gauss_params(self%model%qci, self%model%alpha, btmp, ctmp, bl, cl)

    bkc  = conjg(bk)
    beta = matvec_prod(self%model%At, bkc + bl)
    
    v_mat = zero_c
    wme   = exact_poly(self%model%alpha_mat, bk, bl, self%model%ew, self%model%w_vec, aa=0.5*self%model%w_mat)
    zme   = exact_poly(self%model%alpha_mat, bk, bl, self%model%ez, self%model%z_vec)
    xme   = exact_poly(self%model%alpha_mat, bk, bl, self%model%ex, self%model%x_vec)

    v_mat(1,1) = wme + zme
    v_mat(1,2) = xme
    v_mat(2,1) = xme
    v_mat(2,2) = wme - zme

    V = dot_product(phi(self, bra_t), matvec_prod(v_mat, phi(self, ket_t)))
    return
  end function potential_mca

  !
  !
  function sdot_mca(self, bra_t, ket_t, Snuc) result(delS)
    class(lvc_mca)                     :: self
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: delS
    real(drk)                          :: vel(self%nc), force(self%nc), t_deriv(self%nc)
    complex(drk)                       :: nuc_sdt, elec_sdt

    ! nuclear contribution
    vel     = self%model%omega * ket_t%p
    force   = self%model%force(ket_t%x, ket_t%state)
    nuc_sdt = nuc_sdot(bra_t, ket_t, Snuc, vel, force)
    
    ! electronic contribution
    t_deriv  = matvec_prod(dphi(self, ket_t), phi(self, bra_t))
    elec_sdt = dot_product(t_deriv, vel) * Snuc

    ! total
    delS = nuc_sdt + elec_sdt

    return
  end function sdot_mca

  !
  !
  function pop_mca(self, nst, bra_t, ket_t, Snuc) result(spop)
    class(lvc_mca)                     :: self
    integer(ik), intent(in)            :: nst
    type(trajectory), intent(in)       :: bra_t
    type(trajectory), intent(in)       :: ket_t
    complex(drk), intent(in)           :: Snuc

    complex(drk)                       :: spop(nst)
    complex(drk)                       :: zme, xme
    complex(drk)                       :: bk(self%nc), bl(self%nc), btmp(self%nc)
    complex(drk)                       :: bkc(self%nc), db(self%nc)
    complex(drk)                       :: beta(2)
    complex(drk)                       :: ck, cl, ctmp
    complex(drk)                       :: sigmaz(2,2)
    complex(drk)                       :: za

    call extract_gauss_params(bra_t, btmp, ctmp)
    call shift_gauss_params(self%model%qci, self%model%alpha, btmp, ctmp, bk, ck)

    call extract_gauss_params(ket_t, btmp, ctmp)
    call shift_gauss_params(self%model%qci, self%model%alpha, btmp, ctmp, bl, cl)

    bkc  = conjg(bk)
    !print *,'bk,bl=',bk,bl
    beta = matvec_prod(self%model%At, bkc + bl)

    call exact_pop(beta, self%model%d_alpha, self%model%b_alpha, zme, xme)

    ! add nuclear overlap contribution
    zme = Snuc * zme
    xme = Snuc * xme

    ! for sigmaz matrix
    sigmaz(1,:) = (/  zme, -xme /) 
    sigmaz(2,:) = (/ -xme, -zme /) 

    !print *,'phi(bra)=',phi(self,bra_t)
    !print *,'phi(ket)=',phi(self,ket_t)    
    !rint *,'theta(bra)=',theta(self, bra_t%label, bra_t%x)
    !print *,'theta(ket)=',theta(self, ket_t%label, ket_t%x)
    za   = dot_product(phi(self, bra_t), matvec_prod(sigmaz, phi(self, ket_t)))
    spop = (/ 0.5*(1-za), 0.5*(1+za) /)

    !print *,'sigmaz=',sigmaz
    !print *,'phi1=',phi(self,bra_t)
    !print *,'phi2=',phi(self,ket_t)
    !print *,'za, spop=',za,spop
    return
  end function pop_mca


!------------------------------------------------------------------
!
! Private functions

  function elec_overlap(self, bra, ket) result(Selec)
    class(lvc_mca)               :: self
    type(trajectory), intent(in) :: bra, ket

    complex(drk)                    :: Selec

    Selec = cmplx(dot_product(phi(self, bra), phi(self, ket)), 0.d0)
    return
  end function elec_overlap

  !
  ! Adiabatic electronic wfn in basis of diabatic states
  !
  function phi(self, traj) result(phi_vec)
    class(lvc_mca)               :: self
    type(trajectory), intent(in) :: traj

    real(drk)                    :: phi_vec(2)
    real(drk)                    :: angle
    real(drk)                    :: rotmat(2,2)

    angle   = theta(self, traj%label, traj%x)
    rotmat  = rot(self, angle)
    phi_vec = rotmat(:, traj%state)
  end function phi

  !
  ! ADT rotation angle theta 
  !
  function theta(self, label, q) result(angle)
    class(lvc_mca)            :: self
    integer(ik), intent(in)   :: label
    real(drk), intent(in)     :: q(self%nc)

    real(drk)                 :: angle
    real(drk)                 :: qshft(self%nc)
    real(drk)                 :: X, Z
    real(drk)                 :: pi_mult(3)
    real(drk)                 :: dif_vec(3)
    real(drk)                 :: angle_cache
    logical                   :: incache

    qshft = q - self%model%qci
    X     = dot_product(self%model%x_vec, qshft)
    Z     = dot_product(self%model%z_vec, qshft)

    angle = 0.5d0 * atan(X/Z)

    ! check against the cached value
    if(theta_cache%cached(label)) then
      pi_mult     = (/ 0.d0, -pi, pi /)
      angle_cache = theta_cache%retrieve(label)
      dif_vec     = abs(angle - angle_cache + pi_mult)
      if(minloc(dif_vec, dim=1) /= 1) angle = angle + pi_mult(minloc(dif_vec, dim=1))
    endif
     
    call theta_cache%set(label, angle)
 
     return
  end function theta

  !
  ! diabatic state rotation matrix
  ! 
  function rot(self, theta) result(mat)
    class(lvc_mca)            :: self
    real(drk), intent(in)     :: theta

    real(drk)                 :: mat(2,2)

    mat(1,:) = (/ cos(theta), -sin(theta) /)
    mat(2,:) = (/ sin(theta),  cos(theta) /)
  end function rot

  !
  !
  function dphi(self, traj) result(phi_grad)
    class(lvc_mca)            :: self
    type(trajectory)          :: traj

    real(drk)                 :: phi_grad(self%nc, 2)
    real(drk)                 :: angle
    real(drk)                 :: drot_mat(2,2)
    real(drk)                 :: dangle(self%nc)
    
    angle    = theta(self, traj%label, traj%x)
    drot_mat = drot(self, angle)
    dangle = dtheta(self, traj%x)

    phi_grad(:,1) = dangle * drot_mat(1, traj%state)
    phi_grad(:,2) = dangle * drot_mat(2, traj%state) 
  end function dphi

  !
  ! derivative of theta wrt nuclear coordinates
  ! 
  function dtheta(self, q) result(theta_grad)
    class(lvc_mca)            :: self
    real(drk), intent(in)     :: q(self%nc)

    real(drk)                 :: theta_grad(self%nc)
    real(drk)                 :: qshft(self%nc)
    real(drk)                 :: X, Z, arg  
  
    qshft = q - self%model%qci

    X = dot_product(self%model%x_vec, qshft)
    Z = dot_product(self%model%z_vec, qshft)
    if(abs(Z) < mp_drk)Z = mp_drk

    arg    = X / Z
    theta_grad = 0.5d0 * (self%model%x_vec/Z - X*self%model%z_vec/Z**2) / (1.d0 + arg**2)    
  end function dtheta

  !
  !
  function drot(self, theta) result(dmat)
    class(lvc_mca)            :: self
    real(drk), intent(in)     :: theta

    real(drk)                 :: dmat(2,2)

    dmat(1,:) = (/ -sin(theta), -cos(theta) /)
    dmat(2,:) = (/  cos(theta), -sin(theta) /)

  end function drot

!------------------------------------------------------------------------
! 
! cache routines

  ! 
  !
  subroutine init_cache(self)
    class(cache)             :: self

    if(.not.allocated(self%labels)) then
      allocate(self%labels(self%batch_size))
      allocate(self%thetas(self%batch_size))
      self%max_size = self%batch_size 
    endif

    self%cache_size = 0

  end subroutine init_cache

  !
  !
  subroutine resize_cache(self)
    class(cache)             :: self

    integer(ik), allocatable :: tmp_labels(:)
    real(drk), allocatable   :: tmp_angs(:)

    allocate(tmp_labels(size(self%labels)))
    allocate(tmp_angs(size(self%thetas)))
    tmp_labels = self%labels
    tmp_angs   = self%thetas

    deallocate(self%labels, self%thetas)
    allocate(self%labels(size(tmp_labels) + self%batch_size))
    allocate(self%thetas(size(tmp_angs)   + self%batch_size))
    deallocate(tmp_labels, tmp_angs)

    self%labels(:size(tmp_labels)) = tmp_labels
    self%thetas(:size(tmp_angs))   = tmp_angs
    self%max_size = self%max_size + self%batch_size

  end subroutine resize_cache

  !
  !
  subroutine set(self, label, ang)
    class(cache)             :: self
    integer(ik)              :: label
    real(drk)                :: ang

    integer(ik)              :: i

    ! run through cache to find label
    i = 1
    do while(i <= self%cache_size)
      if(self%labels(i) == label) exit
      i = i + 1
    enddo

    ! if label in cache, then set to new angle
    if(i <= self%cache_size) then
      self%thetas(i) = ang
    ! else create a new entry
    else
      if(i > self%max_size) call self%resize_cache()
      self%labels(i) = label
      self%thetas(i) = ang
      self%cache_size = self%cache_size + 1
    endif

    return

  end subroutine

  ! 
  !
  function retrieve(self, label) result(ang)
    class(cache)            :: self
    integer(ik), intent(in) :: label

    real(drk)               :: ang
    integer(ik)             :: i

    ang = -1000.
    i   = 1
    do while(i <= self%cache_size)
      if(self%labels(i) == label) exit
      i = i + 1
    enddo

    if(i <= self%cache_size)ang = self%thetas(i)

  end function retrieve

  !
  ! 
  function cached(self, label) result(found)
    class(cache)            :: self
    integer(ik), intent(in) :: label

    logical                 :: found
    integer(ik)             :: i

    found = .false.
    i = 1
    do while(i <= self%cache_size)
      if(self%labels(i) == label) then
        found = .true.
        exit
      endif
      i = i + 1
    enddo
   
    return
  end function cached

end module lvc_mcamod

