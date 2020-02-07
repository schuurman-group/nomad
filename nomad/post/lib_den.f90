!
! libden
!
!
module libden
  use accuracy
  use math
  use libtraj
  implicit none

  private
  public load_parameters
  public evaluate_density

  ! maximum number of iterations
  integer(ik)                   :: iter_max = 100000
  ! number of internal coordinates
  integer(ik)                   :: n_intl  
  ! N-dimensional density grid
  real(drk), allocatable        :: den_grid(:)
  ! density that lies off-grid
  real(drk)                     :: off_grid
  ! number of batches to average over
  integer(ik),allocatable       :: batch_list(:)
  ! states in to include
  integer(ik), allocatable      :: state_list(:)


  contains

    !
    !
    !
    subroutine load_parameters(rseed, n_batch, batch_lst, n_coord, coord_lst, cf_lst, atm_lst, grid_bnds, n_grid) bind(c, name='load_parameters')
      integer(ik), intent(in)                  :: rseed
      integer(ik), intent(in)                  :: n_batch
      integer(ik), intent(in)                  :: batch_lst(n_batch)
      integer(ik), intent(in)                  :: n_coord
      integer(ik), intent(in)                  :: coord_lst(8, n_coord)
      real(drk), intent(in)                    :: cf_lst(8, n_coord)
      integer(ik), intent(in)                  :: atm_lst(16, n_coord)
      real(drk), intent(in)                    :: grid_bnds(2, n_coord)
      integer(ik), intent(in)                  :: n_grid(n_coord)

      call seed(rseed)

      allocate(batch_list(n_batch)
      batch_list(1:n_batch) = batch_lst(1:n_batch)

      n_intl = n_coord
      allocate(den_grid(product(n_grid)))
      

      return
    end subroutine load_parameters

    !
    !
    !
    subroutine evaluate_density(time) bind(c, name='load_parameters')
      real(drk)                               :: time
      integer(ik)                             :: batch = 0
      integer(ik), allocatable                :: traj(:)
      complex(drk), allocatable               :: smat(:,:)
      integer(ik)                             :: i
      integer(ik)                             :: itraj, ntraj
      integer(ik)                             :: nindx
      real(drk)                               :: geom(n_cart)
      real(drk)                               :: igeom(n_intl)
      real(drk), allocatable                  :: wgt_list(:)

      do ibatch = 1,size(batch_list)

        call collect_trajectories(time, batch_list(ibatch), traj)
        ntraj = size(traj)

        ! make weighted list of trajectories to do importance sampling
        if(allocated(wgt_list))deallocate(wgt_list)
        allocate(wgt_list(0:ntraj))
        call amp_weight_list(wgt_list)        

        ! construct overlap matrix
        allocate(smat(ntraj, ntraj))
        smat = zero_c
        do i = 1,ntraj
          do j = 1,i
            if(traj_list(i)%state == traj_list(j)%state)smat(i,j) = nuc_overlap(traj_lst(i), traj_lst(j))
          enddo
        enddo

        converged = .false.
        iter      = 1
        do while(.not.converged .and. iter <= iter_max)
          itraj     =  select_trajectory(wgt_list)
          geom      =  generate_cartesian_geom(itraj)
          igeom     =  internal_coordinates(geom)
          den       =  compute_density(geom, smat)

          call grid_indices(igeom, nindx, indices, wts) 
          do i = 1,nindx
            den_grid(indices(i)) = den_grid(indices(i) + wts(i) * den
          enddo
          if(nindx == 0)off_grid = off_grid + den

          iter = iter + 1
        enddo

      enddo  
   
      return
    end subroutine evaluate_density

!#################################################
! Density routines
!
!#################################################
    !
    !
    !
    function compute_density(geom, smat) result(den)
      real(drk), intent(in)             :: geom(:)
      complex(drk), intent(in)          :: smat(:,:)

      real(drk)                         :: den
      complex(drk)                      :: dpt
      integer(ik)                       :: i,j

      den = zero
      do i = 1,size(traj_list)
        if(.not.any(state_list) == traj_list(i)%state)cycle
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
      integer(ik),intent(out)           :: traj_indx

      integer(ik)                       :: i
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
    function generate_cartesian_geom(itraj) result(geom)
      integer(ik), intent(in)             :: itraj

      real(drk)                           :: geom(n_crd)
      real(drk)                           :: dx(n_crd)
      real(drk)                           :: sig, rsq
      real(drk)                           :: dx1, dx2

      do i = 1,n_crd
        sig = sqrt(1./(4.*widths(i)))
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

      geom = traj_list(itraj)%x + dx

      return
    end function generate_cartesian_geom

    !
    !
    !
    function internal_coordinates(geom) result(intgeom)
      real(drk), intent(in)               :: geom(n_crd)

      real(drk)                           :: intgeom(n_intl)
      real(drk)                           :: intvec(8)
      integer(ik)                         :: i

      intgeom = zero_drk

      do i = 1,n_intl

        icrd = 1
        do while(coord_lst(icrd, i) /= 0)

          select case(coord_lst(icrd,i))          
            ! distance
            case(1):
              intvec(icrd) = intgeom(i) + cf_lst(icrd,i) * &
                          dist(geom, atm_lst(1, icrd, i), atm_lst(2, icrd, i))
            ! angle
            case(2):
              intvec(icrd) = intgeom(i) + cf_lst(icrd,i) * &
                          angle(geom, atm_lst(1, icrd, i), atm_lst(2, icrd, i), atm_lst(3, icrd, i))
            ! torsion
            case(3):
              intvec(icrd) = intgeom(i) + cf_lst(icrd,i) * &
                          tors(geom, atm_lst(1, icrd, i), atm_lst(2, icrd, i), atm_lst(3, icrd, i), atm_lst(4, icrd, i))
            ! out of plane angle
            case(4):
              intvec(icrd) = intgeom(i) + cf_lst(icrd,i) * &
                          oops(geom, atm_lst(1, icrd, i), atm_lst(2, icrd, i), atm_lst(3, icrd, i), atm_lst(4, icrd, i))
            ! plane angle
            case(5):
              intvec(icrd) = intgeom(i) + cf_lst(icrd,i) * &
                          plane(geom, atm_lst(1, icrd, i), atm_lst(2, icrd, i), atm_lst(3, icrd, i), atm_lst(4, icrd, i), atm_lst(5, icrd, i), atm_lst(6, icrd, i))
          end select

          icrd = icrd + 1
        enddo

        select case(coord_op(i))
          ! simple sum
          case(1):
            intgeom(i) = sum(intvec(1:icrd))
 
          ! maximum value
          case(2)
            intgeom(i) = maxval(intvec(1:icrd))

          ! absolute value of the sum
          case(3):
            integeom(i) = abs(sum(intvec(1:icrd)))
        end select

      enddo

      return
    end function internal_coordinates

    !
    !
    !
    subroutine amp_weight_list(wgt_lst)
      real(ik), intent(out)               :: wgt_lst(:)

      complex(drk),allocatable            :: amps(:)
      real(drk),allocatable               :: norms(:)
      integer(ik)                         :: i

      if(size(traj_list) > size(wgt_lst))stop 'ddn < size(traj_list) in amp_weighted_list'
      allocate(amps(size(traj_list)))
      allocate(norms(size(traj_list)))

      do i = 1,size(traj_list)
        amps(i) = traj_list(i)%amplitude
      enddo
      
      norms   = real(conjg(amps)*amps / dot_product(amps,amps))
      do i = 1,size(traj_list)
        wgt_lst(i) = sum(norms(1:i))
      enddo

      return
    end subroutine amp_weighted_list

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

      centroid = 0.5 * (geom(geom(3*(atom1-1)+1:3*atom1) + geom(3*(atom2-1)+1:3*atom2))
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
    






end module libden
