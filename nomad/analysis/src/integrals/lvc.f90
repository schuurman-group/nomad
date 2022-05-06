module lvcmod
  use accuracy
  use math
  implicit none

  type, public             :: lvc
    ! number of nuclear coordinates
    integer(ik)                :: nc
    ! coordinates of the CI
    real(drk), allocatable     :: qci(:)
    ! width of the gaussian basis functions
    real(drk), allocatable     :: alpha(:)
    ! 2*widths in matrix form: general form for passing
    ! to integration routines
    real(drk), allocatable     :: alpha_mat(:,:)
    ! characteristic frequencies for LVC model
    real(drk), allocatable     :: omega(:)
    ! second-order coefficients (for now: diagonal matrix)
    real(drk), allocatable     :: w_mat(:,:)
    ! average energy contributions, first-order
    real(drk), allocatable     :: w_vec(:)
    ! energy difference contribtion, first-order
    real(drk), allocatable     :: z_vec(:)
    ! off-diagonal first order coefficients
    real(drk), allocatable     :: x_vec(:)
    ! matrix to transform first order coefficients to bs coordinates
    real(drk), allocatable     :: At(:,:)
    ! kappa/lamda parameters transformed to bs coordinates
    real(drk)                  :: b_alpha(2,2)
    ! bs parameters (g/h)
    real(drk)                  :: d_alpha(2)
    ! scalar contributions to various energy contributions
    real(drk)                  :: ex, ez, ew

    contains

      procedure, public       :: init
      procedure, public       :: force

      procedure, private      :: determine_qci
      procedure, private      :: determine_bspace
      procedure, private      :: shift_poly
      procedure, private      :: shift_params
    end type lvc

  contains

    subroutine init(self, widths, freq, focons, scalars)
      class(lvc)                         :: self
      real(drk), intent(in)              :: widths(:)
      real(drk), intent(in)              :: freq(:)
      real(drk), intent(in)              :: focons(:,:)
      real(drk), intent(in)              :: scalars(:)

      integer(ik)                        :: i

      self%nc = size(widths)

      ! set the Hamiltonian parameters
      allocate(self%alpha(self%nc))
      allocate(self%omega(self%nc))
      allocate(self%alpha_mat(self%nc, self%nc))
      allocate(self%w_mat(self%nc, self%nc))
      allocate(self%w_vec(self%nc))
      allocate(self%z_vec(self%nc))
      allocate(self%x_vec(self%nc))

      allocate(self%qci(self%nc))
      allocate(self%At(2, self%nc))

      self%omega = freq(1:self%nc)
      self%alpha = widths(1:self%nc)

      self%w_mat     = zero_drk
      self%alpha_mat = zero_drk
      do i = 1,self%nc
        self%w_mat(i,i)     = self%omega(i)
        self%alpha_mat(i,i) = 2*self%alpha(i)
      enddo
      self%w_vec(1:self%nc) = 0.5d0 * (focons(1:self%nc, 1) + focons(1:self%nc, 2))
      self%z_vec(1:self%nc) = 0.5d0 * (focons(1:self%nc, 1) - focons(1:self%nc, 2))
      self%x_vec(1:self%nc) = focons(1:self%nc, 3)

      self%ew   = 0.5d0 * ( scalars(1) + scalars(2) )
      self%ez   = 0.5d0 * ( scalars(1) - scalars(2) )
      self%ex   = scalars(3)

      ! find the CI position
      call determine_qci(self)

      ! shift parameters so CI is origin
      call shift_params(self, self%qci)

      ! determine branching space parameters
      call determine_bspace(self)
       
      !print *,'At=',self%At

    end subroutine init

    !
    !
    function force(self, q, state) result(lvc_force)
      class(lvc)                  :: self
      real(drk), intent(in)       :: q(self%nc)
      integer(ik), intent(in)     :: state

      real(drk)                   :: lvc_force(self%nc)
      real(drk)                   :: ave_grad(self%nc)
      real(drk)                   :: ss_grad(self%nc)
      real(drk)                   :: x1, z1, denom
      real(drk)                   :: qt(self%nc)
      integer(ik)                 :: sgn

      qt  = q - self%qci

      ! average energy component of the gradient
      ave_grad = self%omega * qt + self%w_vec

      ! state-specific component of the gradient
      z1 = dot_product(qt, self%z_vec)
      x1 = dot_product(qt, self%x_vec)

      denom = sqrt(z1**2 + x1**2)
      if( abs(denom) == 0.d0 ) then
        ss_grad = 0.d0
      else
        ss_grad = (self%z_vec*z1 + self%x_vec*x1) / denom
      endif

      sgn = -3 + 2*state
      lvc_force = -(ave_grad + sgn*ss_grad)

      return
    end function force

!-------------------------------------------------------------------
!
! Private functions
!

    !
    !
    subroutine determine_qci(self)
      class(lvc)                 :: self

      real(drk)                  :: T(self%nc+2, self%nc+2)
      real(drk)                  :: Tinv(self%nc+2, self%nc+2)
      real(drk)                  :: v(self%nc+2)
      integer(ik)                :: n

      T = zero_drk
      n = self%nc

      T(:self%nc, :self%nc) = self%w_mat
      v(:self%nc)           = self%w_vec

      if(norm2(self%z_vec) > zero_drk) then
        n = n  + 1
        T(:self%nc, n) = -self%z_vec
        T(n, :self%nc) =  self%z_vec
        v(n)           =  self%ez
      endif

      if(norm2(self%x_vec) > zero_drk) then
        n = n + 1
        T(:self%nc, n) = -self%x_vec
        T(n, :self%nc) =  self%x_vec
        v(n)           =  self%ex
      endif

      Tinv(:n,:n) = inverse_gelss(n, T(:n,:n))
      T(:n,1)     = matvec_prod(Tinv(:n,:n), -v)
      self%qci    = T(:self%nc, 1)

    end subroutine determine_qci


    !
    !
    subroutine determine_bspace(self)
      class(lvc)      :: self

      real(drk)       :: ainv(self%nc, self%nc)
      real(drk)       :: B(self%nc, 2)
      real(drk)       :: U(2,2), Uvec(2,2)
      real(rk)        :: dm12(2,2)
      real(drk)       :: work(5)
      integer(ik)     :: i, info

      ainv   = inverse_gelss(self%nc, self%alpha_mat)
      B(:,1) = self%z_vec
      B(:,2) = self%x_vec
      U      = matmul(matmul(transpose(B), ainv), B)

      call sym_eigen(U, Uvec, self%d_alpha)

      ! employ convention that dy > dx   
      if(self%d_alpha(1) > self%d_alpha(2)) stop 'eigenvalues not in ascending order -- check sym_eigen'

      dm12 = zero_drk
      do i = 1,2
        if(sqrt(self%d_alpha(i)) > mp_drk) then
          dm12(i,i) = 1.d0 / sqrt(self%d_alpha(i))
        else
          dm12(i,i) = 0.d0
        endif
      enddo

      ! transform matrix to bs coordinates
      self%At = matmul(matmul(dm12, transpose(Uvec)), matmul(transpose(B), ainv))

      ! transformed kappa and lambda parameters
      self%b_alpha = matmul(self%At, B)

      return
    end subroutine determine_bspace

    !
    !
    !
    subroutine shift_params(self, dx)
      class(lvc)              :: self
      real(drk), intent(in)   :: dx(self%nc)

      real(drk)               :: vec(self%nc)
      real(drk)               :: con

      call self%shift_poly(dx, con, vec, self%ew, self%w_vec, mat=self%w_mat)
      self%ew    = con
      self%w_vec = vec

      call self%shift_poly(dx, con, vec, self%ez, self%z_vec)
      self%ez    = con
      self%z_vec = vec

      call self%shift_poly(dx, con, vec, self%ex, self%x_vec)
      self%ex    = con
      self%x_vec = vec

      return
    end subroutine shift_params

    !
    !
    subroutine shift_poly(self, x_shft, new_con, new_vec, con, vec, mat)
      class(lvc)                         :: self

      real(drk), intent(in)              :: x_shft(:)
      real(drk), intent(out)             :: new_con
      real(drk), intent(out)             :: new_vec(:)
      real(drk), intent(in)              :: con
      real(drk), intent(in)              :: vec(:)
      real(drk), intent(in), optional    :: mat(:, :)

      new_vec = vec
      new_con = con + dot_product(x_shft, vec)

      if(present(mat)) then
        new_vec = new_vec + 2*matvec_prod(mat, x_shft)
        new_con = new_con + dot_product(x_shft, matvec_prod(mat, x_shft))
      endif

      return
    end subroutine shift_poly


end module lvcmod
