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
  ! a numerical parameter that defines the extent of the 
  ! penetration of the "fuzziness" into neighboring cells
  integer(ik), parameter        :: fuzzy_depth = 1 
  ! number of iterations to include in rolling average 
  integer(ik), parameter        :: n_rolling = 100
  ! number of grid indices set for a single sampling,
  ! equal to (2*fuzzy_depth+1)**n_intl
  integer(ik)                   :: n_fuzzy
  ! maximum number of iterations
  integer(ik)                   :: iter_max = 25000
  ! number of internal coordinates
  integer(ik)                   :: n_intl  
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

      ! initialize random number generator
      call seed(rseed)

      ! simulation averaging parameters
      n_batch = n_wfn

      ! internal coordinate definitions  
      n_intl = n_coord
      allocate(int_coords(n_intl))
      allocate(state_list(count(state_lst /= -1)))
      state_list = state_lst(1:size(state_list))
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
      allocate(grid_bnds(2, n_intl))
      allocate(grid_npts(n_intl))
      allocate(grid_width(n_intl))

      grid_bnds  = reshape(bounds, (/ 2, n_coord /))
      grid_npts  = npts
      grid_width = (grid_bnds(2,:) - grid_bnds(1,:)) / grid_npts
      if(fuzzy_depth > 0) then
        n_fuzzy    = (2*fuzzy_depth+1)**n_intl
      else
        n_fuzzy    = 1
      endif 

      return
    end subroutine init_density

    !
    !
    !
    subroutine evaluate_density(time, npts, density, converge, total_iter, total_basis, norm_on, norm_off) bind(c, name='evaluate_density')
      real(drk), intent(in)                   :: time
      integer(ik), intent(in)                 :: npts
      real(drk), intent(out)                  :: density(npts)
      real(drk), intent(out)                  :: converge
      integer(ik), intent(out)                :: total_iter
      integer(ik), intent(out)                :: total_basis
      real(drk), intent(out)                  :: norm_on
      real(drk), intent(out)                  :: norm_off

      type(trajectory), allocatable           :: traj_list(:)
      real(drk), allocatable                  :: wgt_list(:)
      real(drk), allocatable                  :: geom(:)
      real(drk)                               :: wts(n_fuzzy)
      integer(ik), allocatable                :: labels(:)
      integer(ik)                             :: indices(n_fuzzy)
      real(drk)                               :: batch_den(npts)
      real(drk)                               :: igeom(n_intl)
      real(drk)                               :: den
      real(drk)                               :: off_wt
      real(drk)                               :: volume_element
      real(drk)                               :: total_pop
      real(drk)                               :: batch_pop
      real(drk)                               :: batch_on
      real(drk)                               :: batch_off
      real(drk)                               :: max_bin
      real(drk)                               :: integral_value
      real(drk)                               :: rolling_ave
      integer(ik)                             :: i, iter
      integer(ik)                             :: n_traj
      integer(ik)                             :: n_cart
      integer(ik)                             :: n_indx
      integer(ik)                             :: itraj
      integer(ik)                             :: ibatch
      logical                                 :: converged
      logical                                 :: first_run

      if(npts /= product(grid_npts))stop 'grid mismatch in evaluate_density'
      density        = zero_drk
      total_pop      = zero_drk
      norm_on        = zero_drk
      norm_off       = zero_drk
      volume_element = product(grid_width)
      total_iter     = 0
      total_basis    = 0

      do ibatch = 1,n_batch

        ! zero out the initial trial density
        batch_den = zero_drk

        ! get list of trajectories for this time and batch
        call locate_trajectories(time, ibatch, labels, traj_list)
        n_traj = size(labels) 
        if(n_traj == 0)cycle

        total_basis = total_basis + n_traj
        n_cart      = size(traj_list(1)%x)

        ! allocate data structures
        if(allocated(wgt_list))deallocate(wgt_list)
        allocate(wgt_list(0:n_traj))
        allocate(geom(n_cart))

        ! make weighted list of trajectories to do importance sampling
        call amp_weight_list(n_traj, traj_list, wgt_list)        

        ! determine the total population on the states to be integrated
        batch_pop = state_population(n_traj, traj_list)
        total_pop = total_pop + batch_pop

        ! perform MC integration of the density
        converged      = .false.
        first_run      = .true.
        iter           = 1
        batch_on       = zero_drk
        batch_off      = zero_drk
        max_bin        = zero_drk
        integral_value = zero_drk
        rolling_ave    = zero_drk
        do while(.not.converged .and. iter <= iter_max)
          ! select a trajectory, weighted by magnitude of amplitude
          itraj     =  select_trajectory(wgt_list)
          ! generate a trial geometry via sampling about cartesian geometry
          ! using a distribution obtained from basis function width
          geom      =  generate_cartesian_geom(n_cart, traj_list(itraj)%x, traj_list(itraj)%width)
          ! determine internal coordiante values corresponding to sampled
          ! geometry
          igeom     =  internal_coordinates(geom)
          ! determine the value of the density at this point
          den       =  compute_density(geom, n_traj, traj_list)

          ! determine which grid points to update. Since we're using fuzzy
          ! integration bounds, this will involve a "region" around the 
          ! central grid pt.
          call grid_indices(igeom, n_indx, indices, wts, off_wt) 
          do i = 1,n_indx
            batch_den(indices(i)) = batch_den(indices(i)) + wts(i) * den 
            max_bin               = max(max_bin, batch_den(indices(i)))
            batch_on              = batch_on + wts(i) * den
          enddo
          batch_off = batch_off + off_wt * den 
          
          ! the converge criteria is the value of the integral, where the max grid pt is 
          ! normlized to "1". We use a rolling average of the difference to smooth out the noise
          if(n_indx > 0) then
            call update_average(abs(integral_value - (batch_on/max_bin)*volume_element), rolling_ave, first_run)
            integral_value = (batch_on/max_bin)*volume_element
            first_run      = .false.
          endif

          iter = iter + 1 
          converged = (rolling_ave <= 1.e-5) .and. (iter > n_rolling)
        enddo

        ! add the density from this batch
        density    = density  + batch_den * (batch_pop / (batch_on + batch_off))
        norm_off   = norm_off + batch_off * (batch_pop / (batch_on + batch_off))    
        converge   = max(converge, rolling_ave)
        total_iter = total_iter + iter

      enddo  
  
      ! total density is incoherent sum of nbatch simulations -- renormalize to "1"
      if(total_pop > mp_drk) then
        density  = density / total_pop
        norm_off = norm_off / total_pop
      endif

      norm_on  = sum(density)

      return
    end subroutine evaluate_density

    !#################################################
    ! Density routines
    !
    !#################################################

    !
    ! compute the density at a specific coordinate specificed
    ! by geom.
    !
    function compute_density(geom, n, traj_list) result(den)
      real(drk), intent(in)             :: geom(:)
      integer(ik), intent(in)           :: n
      type(trajectory), intent(in)      :: traj_list(n)

      real(drk)                         :: den
      complex(drk)                      :: den_cmplx
      complex(drk)                      :: den_ij
      integer(ik)                       :: i,j

      den_cmplx = zero_c
      do i = 1,n
        if(.not.any(state_list == traj_list(i)%state))cycle
        do j = 1,i
          if(traj_list(j)%state /= traj_list(i)%state) cycle
          den_ij = conjg(traj_list(i)%amplitude) * traj_list(j)%amplitude * &
                         nuc_density(traj_list(i), traj_list(j), geom)
          den_cmplx = den_cmplx + den_ij
          if(i /= j) den_cmplx = den_cmplx + conjg(den_ij)
        enddo
      enddo
       
      den = real(den_cmplx)
      if(aimag(den_cmplx) > 100.*mp_drk) stop 'error in compute density -- imag non-zero'
      return
    end function compute_density

    !
    ! compute the total wfn amplitude on states enumerated
    ! in state_list
    !
    function state_population(n, traj_list) result(pop)
      integer(ik), intent(in)           :: n
      type(trajectory), intent(in)      :: traj_list(n)

      real(drk)                         :: pop
      complex(drk)                      :: pop_ij
      complex(drk)                      :: pop_cmplx
      integer(ik)                       :: i,j

      pop_cmplx = zero_c
      do i = 1,n
        if(.not.any(state_list == traj_list(i)%state))cycle
        do j = 1,i
          if(traj_list(j)%state /= traj_list(i)%state) cycle
          pop_ij    = conjg(traj_list(i)%amplitude) * &
                            traj_list(j)%amplitude *  &
                            nuc_overlap(traj_list(i), traj_list(j))
          pop_cmplx = pop_cmplx + pop_ij
          if(i/=j) pop_cmplx = pop_cmplx + conjg(pop_ij)
        enddo
      enddo

      pop = real(pop_cmplx)
      if(aimag(pop_cmplx) > 100.*mp_drk) stop 'error in state_population'
 
      return
    end function state_population

    !
    ! randomly select a trajectory from the trajectory list, 
    ! with a probability weighted by the norms of the trajectories
    !
    function select_trajectory(wgt_list) result(traj_indx)
      real(drk),intent(in)              :: wgt_list(0:)

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
    ! generate a cartesian geometry by sampling about a pt "x"
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
    ! compute a set of internal coordinates corresponding to
    ! to cartesian geometry geom
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
    !  create the sets of weights for the trajectories in
    !  traj_list corresponding to the trajectory amplitudes
    !
    subroutine amp_weight_list(n, traj_list, wgt_lst)
      integer(ik), intent(in)             :: n
      type(trajectory), intent(in)        :: traj_list(n)
      real(drk), intent(out)              :: wgt_lst(0:n)

      complex(drk),allocatable            :: amps(:)
      real(drk),allocatable               :: norms(:)
      real(drk)                           :: nrm
      integer(ik)                         :: i  

      allocate(amps(n))
      allocate(norms(n))
      norms = zero_drk

      do i = 1,n
        amps(i) = traj_list(i)%amplitude
      enddo
 
      nrm     = real(dot_product(amps,amps))
      if(nrm > mp_drk)norms = real(conjg(amps)*amps) / nrm
      wgt_lst = zero_drk
      do i = 1,n
        wgt_lst(i) = sum(norms(1:i))
      enddo

      return
    end subroutine amp_weight_list

    !
    ! compute the grid indices, and corresponding weights
    ! that correspond to the internal coordinates in igeom
    !
    subroutine grid_indices(igeom, n_indx, indices, wgts, wgt_off)
      real(drk), intent(in)               :: igeom(n_intl)
      integer(ik), intent(out)            :: n_indx
      integer(ik), intent(out)            :: indices(n_fuzzy)
      real(drk), intent(out)              :: wgts(n_fuzzy)
      real(drk), intent(out)              :: wgt_off

      integer(ik)                         :: i
      integer(ik)                         :: indx
      integer(ik)                         :: iaddr
      integer(ik)                         :: center_bin(n_intl)
      integer(ik)                         :: step_bin(n_intl)
      integer(ik)                         :: delta(n_intl)
      real(drk)                           :: grid_loc(n_intl)
      real(drk)                           :: wgt_sum

      indices = -1
      wgts    = -1
      n_indx  =  0
      wgt_off =  zero_drk

      do i = 1,n_intl
        grid_loc(i)   = (igeom(i) - grid_bnds(1,i))/grid_width(i)
        center_bin(i) = ceiling(grid_loc(i))
      enddo 

      delta    = -fuzzy_depth
      delta(1) = delta(1) - 1
      do while(sum(delta) < n_intl*fuzzy_depth)

        ! iterate the delta counter
        indx = 1
        do while(delta(indx) == fuzzy_depth)
          delta(indx) = -fuzzy_depth
          indx        = indx - 1
        enddo
        delta(indx) = delta(indx) + 1

        step_bin = center_bin + delta
        iaddr    = grid_address(step_bin)
        if(iaddr /= -1) then
          n_indx          = n_indx + 1
          indices(n_indx) = iaddr
          wgts(n_indx)    = bin_wgt(delta, grid_loc)
        else
          wgt_off         = wgt_off + bin_wgt(delta, grid_loc)
        endif

      enddo

      ! normalize the weights so the total density added is equal to "1"
      wgt_sum = sum(wgts(1:n_indx))+wgt_off
      if(wgt_sum > mp_drk) then
        wgts    = wgts / wgt_sum
        wgt_off = wgt_off / wgt_sum 
      endif

      return
    end subroutine 
  
    !
    ! given a set of grid indices, compute the address in the
    ! density vector
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
    ! compute the grid weight given the distance from
    ! a particular internal coordiante value
    !
    function bin_wgt(step, grid_val) result(wgt)
      integer(ik), intent(in)            :: step(:)
      real(drk), intent(in)              :: grid_val(:)

      real(drk)                          :: distance(n_intl)
      real(drk)                          :: wgt
      real(drk)                          :: r
      real(drk)                          :: r_scale
      integer(ik)                        :: i

      do i = 1,n_intl
        distance(i) = grid_val(i) - (ceiling(grid_val(i))-0.5 + step(i))
      enddo

      ! nascent r defined on range 0 -> sqrt(n_intl)*fuzzy_depth
      ! scale so that r define on range 0 -> 2, then subtract one so range is -1,1
      ! this is range that Becke function defined on
      r_scale = max(1., sqrt(dble(n_intl)) * fuzzy_depth)      
      r       = sqrt(dot_product(distance, distance)) * 2. / r_scale - 1
      wgt     = 0.5 * (1 - wgt_function(r, fuzzy_order))

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
        f = wgt_function(1.5*x - 0.5*x**3,n-1)
      endif

      return
    end function wgt_function

    !
    ! update the rolling average corresponding to the iteration-
    ! by-iteration change to the density integral 
    !
    subroutine update_average(new_value, average, re_init)
      real(drk), intent(in)             :: new_value
      real(drk), intent(out)            :: average
      logical, intent(in)               :: re_init

      real(drk), save                   :: vec(n_rolling)
      real(drk), save                   :: current_sum
      integer(ik), save                 :: current_indx
      integer(ik), save                 :: n_indx 

      if(re_init) then
        vec          = zero_drk
        current_sum  = zero_drk
        current_indx = 0
        n_indx       = 0
      endif

      current_indx      = current_indx + 1
      if(current_indx > n_rolling) current_indx = 1
      n_indx            = min(n_indx+1, n_rolling)
      current_sum       = current_sum - vec(current_indx) + new_value
      vec(current_indx) = new_value

      average           = current_sum / n_indx

    end subroutine update_average



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

      e1  = normalized( geom(3*(atom1-1)+1:3*atom1)-geom(3*(atom2-1)+1:3*atom2) )
      e3  = normalized( geom(3*(atom3-1)+1:3*atom3)-geom(3*(atom2-1)+1:3*atom2) )
      e2  = normalized( geom(3*(atom3-1)+1:3*atom3)-geom(3*(atom4-1)+1:3*atom4) )

      cp1 = cross_prod(e1, e3)
      cp2 = cross_prod(e2, e3)

      e1  = normalized(cp1)
      e2  = normalized(cp2)
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

      e1 = normalized(geom(3*(atom1-1)+1:3*atom1)-geom(3*(atom4-1)+1:3*atom4))
      e2 = normalized(geom(3*(atom2-1)+1:3*atom2)-geom(3*(atom4-1)+1:3*atom4))
      e3 = normalized(geom(3*(atom3-1)+1:3*atom3)-geom(3*(atom4-1)+1:3*atom4))

      cp = cross_prod(e2, e3)
      e2 = normalized(cp)

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

      m1 = normalized(geom(3*(atom1-1)+1:3*atom1)-geom(3*(atom2-1)+1:3*atom2))
      m2 = normalized(geom(3*(atom3-1)+1:3*atom3)-geom(3*(atom2-1)+1:3*atom2))

      n1 = normalized(geom(3*(atom4-1)+1:3*atom4)-geom(3*(atom5-1)+1:3*atom5))
      n2 = normalized(geom(3*(atom6-1)+1:3*atom6)-geom(3*(atom5-1)+1:3*atom5))

      m  = cross_prod(m1,m2)
      n  = cross_prod(n1,n2)
      m_norm = sqrt(dot_product(m, m))
      n_norm = sqrt(dot_product(n, n))

      theta = abs(acos( dot_product(m,n) / (m_norm*n_norm) ))

      return
    end function plane

    !
    ! return a normalized vector u from v
    !
    function normalized(v) result(u)
      real(drk), intent(in)              :: v(3)
   
      real(drk)                          :: u(3)
      real(drk)                          :: nrm
 
      u   = zero_drk     
      nrm = sqrt(dot_product(v, v))
      if(nrm > mp_drk) u = v / nrm

      return
    end function

    !
    !  compute cross-product between to 3D vectors
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
