!
! libden
!
!
module libden
  use accuracy
  use math
  use lib_traj
  implicit none

  private
  public init_density 
  public evaluate_density

  ! fuzzy integration function recursion order
  integer(ik), parameter        :: fuzzy_order = 3
  ! maximum number of iterations
  integer(ik)                   :: iter_max = 100000
  ! number of internal coordinates
  integer(ik)                   :: n_intl  
  ! N-dimensional density grid
  real(drk), allocatable        :: den_grid(:)
  ! density that lies off-grid
  real(drk)                     :: off_grid
  ! number of simulations to average over
  integer(ik)                   :: n_batch
  ! states in to include
  integer(ik), allocatable      :: state_list(:)
  ! bin boundary locations for each integration dimension
  real(drk), allocatable        :: grid_bnds(:,:)
  ! number of grid pts
  integer(ik), allocatable      :: grid_npts(:)
  ! width of grid bins
  real(drk), allocatable        :: grid_width(:)

  type intcoord
    integer(ik)                      :: n_prim
    character(len=5)                 :: op
    character(len=7), allocatable    :: coord_types(:)                 
    real(ik), allocatable            :: cf(:)
    integer(ik), allocatable         :: atoms(:,:)
  end type intcoord

  type(intcoord), allocatable        :: int_coords(:)

  character(len=7), dimension(8)     :: coord_types  = (/ 'stre   ', 'bend   ', 'tors   ', '|tors| ', &
                                                          'oop    ', '|oop|  ', 'plane  ', '|plane|' /)
  character(len=5), dimension(3)     :: op_types     = (/ 'sum  ', 'max  ', '|sum|' /)
  integer(ik), dimension(8)          :: n_atms_coord = (/  2, 3, 4, 4, 4, 4, 6, 6 /)
  integer(ik), parameter             :: max_states  = 10
  integer(ik), parameter             :: max_prim    = 8
  integer(ik), parameter             :: max_atoms   = 6*8

  contains

    !
    !
    !
    subroutine init_density(rseed, n_wfn, n_coord, state_lst, coord_lst, cf_lst, atm_lst, op_lst, bounds, npts) bind(c, name='init_density')
      integer(ik), intent(in)                  :: rseed
      integer(ik), intent(in)                  :: n_wfn
      integer(ik), intent(in)                  :: n_coord
      integer(ik), intent(in)                  :: state_lst(max_states)
      integer(ik), intent(in)                  :: coord_lst(max_prim*n_coord)
      real(drk), intent(in)                    :: cf_lst(max_prim*n_coord)
      integer(ik), intent(in)                  :: atm_lst(max_atoms*n_coord)
      integer(ik), intent(in)                  :: op_lst(n_coord)
      real(drk), intent(in)                    :: bounds(2*n_coord)
      integer(ik), intent(in)                  :: npts(n_coord)

      integer(ik)                              :: i,j,n_prim
      integer(ik)                              :: scr_vec(max_prim)

      call seed(rseed)

      ! simulation averaging parameters
      n_batch = n_wfn
 
      ! internal coordinate definitions  
      n_intl = n_coord
      allocate(int_coords(n_intl))
      do i = 1,n_intl
        scr_vec = coord_lst(max_prim*(i-1)+1:max_prim*i)
        n_prim  = count(scr_vec > 0 .and. scr_vec < len(coord_types))

        int_coords(i)%n_prim = n_prim
        int_coords(i)%op     = op_types(op_lst(i))
        allocate(int_coords(i)%coord_types(n_prim))
        allocate(int_coords(i)%cf(n_prim))
        allocate(int_coords(i)%atoms(max_atoms, n_prim))
        do j = 1,n_prim
          int_coords(i)%coord_types(j) = coord_types(scr_vec(j))
          int_coords(i)%cf(j)          = cf_lst(max_prim*(i-1)+j)
          int_coords(i)%atoms(:,j)     = atm_lst(max_atoms*(i-1)+1:max_atoms*i)
        enddo
      enddo

      ! integration grid initialization
      allocate(den_grid(product(npts)))
      allocate(grid_bnds(2, n_intl))
      allocate(grid_npts(n_intl))
      allocate(grid_width(n_intl))

      den_grid   = zero_drk
      grid_bnds  = reshape(bounds, (/ 2, n_coord /))
      grid_npts  = npts
      grid_width = (grid_bnds(2,:) - grid_bnds(1,:)) / grid_npts

      return
    end subroutine init_density

    !
    !
    !
    subroutine evaluate_density(time, npts, density) bind(c, name='evaluate_density')
      real(drk), intent(in)                   :: time
      integer(ik), intent(in)                 :: npts
      real(drk), intent(out)                  :: density(npts)

      type(trajectory), allocatable           :: traj_list(:)
      real(drk), allocatable                  :: wgt_list(:)
      real(drk),allocatable                   :: geom(:)
      complex(drk), allocatable               :: smat(:,:)
      integer(ik), allocatable                :: labels(:)
      real(drk)                               :: igeom(n_intl)
      real(drk)                               :: den
      real(drk)                               :: wts(9)
      real(drk)                               :: off_grid
      integer(ik)                             :: i, j, iter
      integer(ik)                             :: n_traj
      integer(ik)                             :: n_cart
      integer(ik)                             :: n_indx
      integer(ik)                             :: itraj
      integer(ik)                             :: ibatch
      integer(ik)                             :: indices(9)
      logical                                 :: converged
 

      if(npts /= product(grid_npts))stop 'grid mismatch in evaluate_density'
      den_grid = zero_drk

      do ibatch = 1,n_batch

        ! get list of trajectories for this time and batch
        call locate_trajectories(time, ibatch, labels, traj_list)
        n_traj = size(labels) 
        if(n_traj == 0)cycle
        n_cart = size(traj_list(1)%x)

        ! allocate data structures
        if(allocated(wgt_list))deallocate(wgt_list)
        allocate(wgt_list(0:n_traj))
        allocate(geom(n_cart))
        allocate(smat(n_traj, n_traj))

        ! make weighted list of trajectories to do importance sampling
        call amp_weight_list(n_traj, traj_list, wgt_list)        

        ! construct overlap matrix
        smat = zero_c
        do i = 1,n_traj
          do j = 1,i
            if(traj_list(i)%state == traj_list(j)%state) then 
              smat(i,j) = nuc_overlap(traj_list(i), traj_list(j))
            endif
          enddo
        enddo

        ! perform MC integration of the density
        converged = .false.
        iter      = 1
        do while(.not.converged .and. iter <= iter_max)
          itraj     =  select_trajectory(wgt_list)
          geom      =  generate_cartesian_geom(n_cart, traj_list(itraj)%x, traj_list(itraj)%width)
          igeom     =  internal_coordinates(geom)
          den       =  compute_density(geom, n_traj, traj_list)

          call grid_indices(igeom, n_indx, indices, wts) 
          do i = 1,n_indx
            den_grid(indices(i)) = den_grid(indices(i)) + wts(i) * den
          enddo
          if(n_indx == 0)off_grid = off_grid + den

          iter = iter + 1
        enddo

      enddo  
   
      density = den_grid
   
      print *,'off_grid = ',off_grid

      return
    end subroutine evaluate_density

    !#################################################
    ! Density routines
    !
    !#################################################

    !
    !
    !
    function compute_density(geom, n, traj_list) result(den)
      real(drk), intent(in)             :: geom(:)
      integer(ik), intent(in)           :: n
      type(trajectory), intent(in)      :: traj_list(n)

      real(drk)                         :: den
      complex(drk)                      :: dpt
      integer(ik)                       :: i,j

      den = zero_drk
      do i = 1,n
        if(.not.any(state_list == traj_list(i)%state))cycle
        do j = 1,i
          if(traj_list(j)%state /= traj_list(i)%state) cycle
          dpt = conjg(traj_list(i)%amplitude)*traj_list(j)%amplitude* &
                exp(I_drk*(traj_list(j)%phase-traj_list(i)%phase))*   &
                nuc_density(traj_list(i), traj_list(j), geom)
          den = den + dpt
          if(i /= j) den = den + conjg(dpt)
        enddo
      enddo
        
      return
    end function compute_density

    !
    !
    !
    function select_trajectory(wgt_list) result(traj_indx)
      real(drk),intent(in)              :: wgt_list(:)

      integer(ik)                       :: traj_indx
      real(drk)                         :: num
      
      call random_number(num)
      traj_indx = 1
      do while(traj_indx < size(wgt_list) )
        if( num > wgt_list(traj_indx-1) .and. num < wgt_list(traj_indx))exit
        traj_indx = traj_indx + 1
      enddo
  
      return
    end function select_trajectory

    !
    !
    !
    function generate_cartesian_geom(n, x, width) result(geom)
      integer(ik), intent(in)             :: n
      real(drk), intent(in)               :: x(n)
      real(drk), intent(in)               :: width(n)

      real(drk)                           :: geom(n)
      real(drk)                           :: dx(n)
      real(drk)                           :: sig, rsq
      real(drk)                           :: dx1, dx2
      integer(ik)                         :: i

      do i = 1,n
        sig = sqrt(1./(4.*width(i)))
        rsq = 2.
        do
          if(rsq.lt.1.d0.and.rsq.ne.0.d0)exit
          call random_number(dx1)
          call random_number(dx2)
          dx1=2.d0*dx1-1.d0
          dx2=2.d0*dx2-1.d0
          rsq=dx1*dx1+dx2*dx2
        enddo
        dx(i) = sig * dx1 * sqrt(-2.d0*log(rsq)/rsq)
      enddo

      geom = x + dx

      return
    end function generate_cartesian_geom

    !
    !
    !
    function internal_coordinates(geom) result(intgeom)
      real(drk), intent(in)               :: geom(:)

      real(drk)                           :: intgeom(n_intl)
      real(drk)                           :: intvec(max_prim)
      integer(ik)                         :: i
      integer(ik)                         :: i_prim

      intgeom = zero_drk

      do i = 1,n_intl

        do i_prim = 1,int_coords(i)%n_prim
          select case(trim(int_coords(i)%coord_types(i_prim)))          
            ! distance
            case('stre')
              intvec(i_prim) = intgeom(i) + int_coords(i)%cf(i_prim) * &
                               dist(geom, int_coords(i)%atoms(1,i_prim), int_coords(i)%atoms(2,i_prim))
            ! angle
            case('bend')
              intvec(i_prim) = intgeom(i) +int_coords(i)%cf(i_prim) * &
                               angle(geom, int_coords(i)%atoms(1,i_prim), int_coords(i)%atoms(2,i_prim), &
                               int_coords(i)%atoms(3,i_prim))
            ! torsion
            case('tors')
              intvec(i_prim) = intgeom(i) + int_coords(i)%cf(i_prim) * &
                               tors(geom, int_coords(i)%atoms(1,i_prim), int_coords(i)%atoms(2,i_prim), &
                               int_coords(i)%atoms(3,i_prim), int_coords(i)%atoms(4,i_prim))
            ! torsion
            case('|tors|')
              intvec(i_prim) = intgeom(i) + int_coords(i)%cf(i_prim) * &
                               abs(tors(geom, int_coords(i)%atoms(1,i_prim), int_coords(i)%atoms(2,i_prim), &
                               int_coords(i)%atoms(3,i_prim), int_coords(i)%atoms(4,i_prim)))
            ! out of plane angle
            case('oop')
              intvec(i_prim) = intgeom(i) + int_coords(i)%cf(i_prim) * &
                               oop(geom, int_coords(i)%atoms(1,i_prim), int_coords(i)%atoms(2,i_prim), &
                               int_coords(i)%atoms(3,i_prim), int_coords(i)%atoms(4,i_prim))
            case('|oop|')
              intvec(i_prim) = intgeom(i) + int_coords(i)%cf(i_prim) * &
                               abs(oop(geom, int_coords(i)%atoms(1,i_prim), int_coords(i)%atoms(2,i_prim), &
                               int_coords(i)%atoms(3,i_prim), int_coords(i)%atoms(4,i_prim)))
            ! plane angle
            case('plane')
              intvec(i_prim) = intgeom(i) + int_coords(i)%cf(i_prim) * &
                               plane(geom, int_coords(i)%atoms(1,i_prim), int_coords(i)%atoms(2,i_prim), &
                               int_coords(i)%atoms(3,i_prim), int_coords(i)%atoms(4,i_prim), &
                               int_coords(i)%atoms(5,i_prim), int_coords(i)%atoms(6,i_prim))
            case('|plane|')
              intvec(i_prim) = intgeom(i) + int_coords(i)%cf(i_prim) * &
                               abs(plane(geom, int_coords(i)%atoms(1,i_prim), int_coords(i)%atoms(2,i_prim), &
                               int_coords(i)%atoms(3,i_prim), int_coords(i)%atoms(4,i_prim), &
                               int_coords(i)%atoms(5,i_prim), int_coords(i)%atoms(6,i_prim)))
          end select
        enddo

        select case(trim(int_coords(i)%op))
          ! simple sum
          case('sum')
            intgeom(i) = sum(intvec(1:int_coords(i)%n_prim))
 
          ! maximum value
          case('max')
            intgeom(i) = maxval(intvec(1:int_coords(i)%n_prim))

          ! absolute value of the sum
          case('|sum|')
            intgeom(i) = abs(sum(intvec(1:int_coords(i)%n_prim)))
        end select

      enddo

      return
    end function internal_coordinates

    !
    !
    !
    subroutine amp_weight_list(n, traj_list, wgt_lst)
      integer(ik), intent(in)             :: n
      type(trajectory), intent(in)        :: traj_list(n)
      real(drk), intent(out)              :: wgt_lst(n+1)

      complex(drk),allocatable            :: amps(:)
      real(drk),allocatable               :: norms(:)
      integer(ik)                         :: i  

      allocate(amps(n))
      allocate(norms(n))

      do i = 1,n
        amps(i) = traj_list(i)%amplitude
      enddo
      
      norms   = real(conjg(amps)*amps / dot_product(amps,amps))
      wgt_lst = zero_drk
      do i = 1,n
        wgt_lst(i) = sum(norms(1:i))
      enddo

      return
    end subroutine amp_weight_list

    !
    !
    !
    subroutine grid_indices(igeom, n_indx, indices, wgts)
      real(drk), intent(in)               :: igeom(n_intl)
      integer(ik), intent(out)            :: n_indx
      integer(ik), intent(out)            :: indices(:)
      real(drk), intent(out)              :: wgts(:)

      integer(ik)                         :: i
      integer(ik)                         :: indx
      integer(ik)                         :: iaddr
      integer(ik)                         :: center_bin(n_intl)
      integer(ik)                         :: step_bin(n_intl)
      real(drk)                           :: grid_loc(n_intl)
      integer(ik)                         :: delta(n_intl)
      integer(ik)                         :: max_step = 1

      indices = -1
      wgts    = -1
      n_indx  = 0

      do i = 1,n_intl
        grid_loc(i)   = (igeom(i) - grid_bnds(1,i))/grid_width(i)
        center_bin(i) = ceiling(grid_loc(i))
      enddo 

      delta    = -max_step
      delta(1) = delta(1) - 1
      do while(sum(delta) < n_intl*max_step)

        ! iterate the delta counter
        indx = 1
        do while(delta(indx) == max_step)
          delta(indx) = -max_step
          indx        = indx - 1
        enddo
        delta(indx) = delta(indx) + 1

        step_bin = center_bin + delta
        iaddr    = grid_address(step_bin)
        if(iaddr /= -1) then
          n_indx          = n_indx + 1
          indices(n_indx) = iaddr
          wgts(n_indx)    = bin_wgt(delta, grid_loc)
        endif

      enddo

      return
    end subroutine 
  
    !
    !
    !
    function grid_address(bin_vals) result(addr)
      integer(ik), intent(in)            :: bin_vals(n_intl)

      integer(ik)                        :: addr
      integer(ik)                        :: i
     
      if(any(bin_vals < 1) .or. any(bin_vals > grid_npts)) then
        addr = -1
      else
        addr = 0
        do i = 1,n_intl-1
          addr = addr + bin_vals(i) * product(grid_npts(i+1:n_intl))
        enddo
        addr = addr + bin_vals(n_intl)
      endif

      return
    end function grid_address

    !
    !
    !
    function bin_wgt(step, grid_val) result(wgt)
      integer(ik), intent(in)            :: step(:)
      real(drk), intent(in)              :: grid_val(:)

      real(drk)                          :: distance(n_intl)
      real(drk)                          :: wgt
      real(drk)                          :: r
      integer(ik)                        :: i

      do i = 1,n_intl
        distance(i) = grid_val(i) - (ceiling(grid_val(i))+step(i)-0.5)
      enddo

      r   = sqrt(dot_product(distance, distance))
      wgt = wgt_function(r, fuzzy_order)

      return
    end function bin_wgt

    !
    ! the value of the region function for a specified recursion level
    !
    recursive function wgt_function(x, n) result(f)
      integer(ik), intent(in)            :: n
      real(drk), intent(in)              :: x

      real(drk)                          :: f

      if(n.eq.0) then
        f = x
      else
        f = wgt_function(1.5*x-0.5*x**3,n-1)
      endif

      return
    end function wgt_function



    !#################################################
    ! Coordinate routines
    !
    !#################################################

    ! 
    ! distance between two atoms
    !
    function dist(geom, atom1, atom2) result(r)
      real(drk), intent(in)             :: geom(:)
      integer(ik), intent(in)           :: atom1
      integer(ik), intent(in)           :: atom2

      real(drk)                         :: r
      real(drk)                         :: rvec(3)      

      rvec = geom(3*(atom1-1)+1:3*atom1) - geom(3*(atom2-1)+1:3*atom2)
      r = sqrt(dot_product(rvec, rvec)) 

      return
    end function dist

    !
    ! distance between the centroid of atoms1 and atom2
    ! and atom3
    !
    function centroid_dist(geom, atom1, atom2, atom3) result(r)
      real(drk), intent(in)            :: geom(:)
      integer(ik), intent(in)          :: atom1
      integer(ik), intent(in)          :: atom2
      integer(ik), intent(in)          :: atom3

      real(drk)                        :: r
      real(drk)                        :: centroid(3)
      real(drk)                        :: rvec(3)

      centroid = 0.5 * (geom(3*(atom1-1)+1:3*atom1) + geom(3*(atom2-1)+1:3*atom2))
      rvec     = centroid - geom(3*(atom3-1)+1:3*atom3)
      r        = sqrt(dot_product(rvec, rvec))
      return
    end function centroid_dist

    !
    ! angle between three atoms, center atom 2nd
    !
    function angle(geom, atom1, atom2, atom3) result(theta)
      real(drk), intent(in)            :: geom(:)
      integer(ik), intent(in)          :: atom1
      integer(ik), intent(in)          :: atom2
      integer(ik), intent(in)          :: atom3

      real(drk)                        :: theta
      real(drk)                        :: v1(3), v2(3)
      real(drk)                        :: n1, n2

      v1 = geom(3*(atom1-1)+1:3*atom1) - geom(3*(atom2-1)+1:3*atom2)
      v2 = geom(3*(atom3-1)+1:3*atom3) - geom(3*(atom2-1)+1:3*atom2)
      n1 = sqrt(dot_product(v1, v1))
      n2 = sqrt(dot_product(v2, v2))

      theta = acos(dot_product(v1,v2)/(n1*n2))
      
      return
    end function angle
    
    !
    ! torsional angle 
    !
    function tors(geom, atom1, atom2, atom3, atom4) result(tau)
      real(drk), intent(in)            :: geom(:)
      integer(ik), intent(in)          :: atom1,atom2,atom3,atom4

      real(drk)                        :: tau
      real(drk)                        :: x,y
!      real(drk)                        :: shft=0.5*pi
      real(drk)                        :: e1(3), e2(3), e3(3)
      real(drk)                        :: cp1(3),cp2(3),cp3(3)

      e1  = vec( geom(3*(atom1-1)+1:3*atom1)-geom(3*(atom2-1)+1:3*atom2) )
      e3  = vec( geom(3*(atom3-1)+1:3*atom3)-geom(3*(atom2-1)+1:3*atom2) )
      e2  = vec( geom(3*(atom3-1)+1:3*atom3)-geom(3*(atom4-1)+1:3*atom4) )

      cp1 = cross_prod(e1, e3)
      cp2 = cross_prod(e2, e3)

      e1  = vec(cp1)
      e2  = vec(cp2)
      cp3 = cross_prod(e1, e2)

      x   = -dot_product(e1,e2)
      y   = sqrt(dot_product(cp3, cp3))
      tau = atan2(y,x)

      return
    end function tors

    !
    ! compute out-plane angle: atom1(oop), atom2/atom3 (in plane)
    !                          atom4 (center)
    !
    function oop(geom, atom1, atom2, atom3, atom4) result(alpha)
      real(drk), intent(in)            :: geom(:)
      integer(ik), intent(in)          :: atom1
      integer(ik), intent(in)          :: atom2
      integer(ik), intent(in)          :: atom3
      integer(ik), intent(in)          :: atom4

      real(drk)                        :: alpha
      real(drk)                        :: e1(3),e2(3),e3(3)
      real(drk)                        :: cp(3),stheta,ctheta

      e1 = vec(geom(3*(atom1-1)+1:3*atom1)-geom(3*(atom4-1)+1:3*atom4))
      e2 = vec(geom(3*(atom2-1)+1:3*atom2)-geom(3*(atom4-1)+1:3*atom4))
      e3 = vec(geom(3*(atom3-1)+1:3*atom3)-geom(3*(atom4-1)+1:3*atom4))

      cp = cross_prod(e2, e3)
      e2 = vec(cp)

      stheta = dot_product(e1,e2)
      ctheta = sqrt(1 - stheta**2)
      alpha  = acos(ctheta)
      if(stheta.lt.0)alpha = -alpha

      return
    end function oop

    !
    ! compute the angle between two planes
    !
    function plane(geom, atom1, atom2, atom3, atom4, atom5, atom6) result(theta)
      real(drk), intent(in)            :: geom(:)
      integer(ik), intent(in)          :: atom1
      integer(ik), intent(in)          :: atom2
      integer(ik), intent(in)          :: atom3
      integer(ik), intent(in)          :: atom4
      integer(ik), intent(in)          :: atom5
      integer(ik), intent(in)          :: atom6

      real(drk)                        :: theta
      real(drk)                        :: m(3),n(3)
      real(drk)                        :: m1(3),m2(3)
      real(drk)                        :: n1(3),n2(3)
      real(drk)                        :: m_norm, n_norm

      m1 = vec(geom(3*(atom1-1)+1:3*atom1)-geom(3*(atom2-1)+1:3*atom2))
      m2 = vec(geom(3*(atom3-1)+1:3*atom3)-geom(3*(atom2-1)+1:3*atom2))

      n1 = vec(geom(3*(atom4-1)+1:3*atom4)-geom(3*(atom5-1)+1:3*atom5))
      n2 = vec(geom(3*(atom6-1)+1:3*atom6)-geom(3*(atom5-1)+1:3*atom5))

      m  = cross_prod(m1,m2)
      n  = cross_prod(n1,n2)
      m_norm = sqrt(dot_product(m, m))
      n_norm = sqrt(dot_product(n, n))

      theta = abs(acos( dot_product(m,n) / (m_norm*n_norm) ))

      return
    end function plane

    !
    !
    !
    function vec(v) result(u)
      real(drk), intent(in)              :: v(3)
   
      real(drk)                          :: u(3)
      real(drk)                          :: nrm
 
      u   = zero_drk     
      nrm = sqrt(dot_product(v, v))
      if(nrm > mp_drk) u = v / nrm

      return
    end function

    !
    !
    !
    function cross_prod(v1, v2) result(u)
      real(drk), intent(in)               :: v1(3)
      real(drk), intent(in)               :: v2(3)

      real(drk)                           :: u(3)

      u(1) = v1(2)*v2(3) - v1(3)*v2(2)
      u(2) = v1(3)*v2(1) - v1(1)*v2(3)
      u(3) = v1(1)*v2(2) - v1(2)*v2(1)

      return
    end function cross_prod


end module libden
