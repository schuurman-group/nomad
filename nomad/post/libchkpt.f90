!
! lib_chkpt -- a library to work with nomad HDF5 checkpoint files
!
! M. S. Schuurman -- Oct. 11, 2018
!
module libchkpt
 use hdf5
 use lookup

 implicit none
 
 public read_params
 public read_chkpt

  type traj
    real,allocatable     :: x(:)
    real,allocatable     :: p(:)
    real,allocatable     :: widths(:)
    real,allocatable     :: masses(:)
    real                 :: phase
    integer,allocatable  :: states(:)
  end type traj

  type traj_basis
   type(traj), allocatable :: t_basis(:,:)
   real,allocatable        :: times(:)
  end type traj_basis

  type integrals
   type(traj), allocatable :: centroids(:,:)
   real, allocatable       :: times(:)
  end type integrals 

  integer,parameter                        :: link_max = 1000 
  type grp_info
    character(len=100)                     :: root_name
    integer                                :: ngrp
    integer(hid_t), dimension(link_max)    :: grp_ids
    character(len=100),dimension(link_max) :: grp_names
    integer                                :: ndset
    integer(hid_t),dimension(link_max)     :: dset_ids
    character(len=100),dimension(link_max) :: dset_names
  end type grp_info
  type(grp_info)                           :: grp_data

 contains

 !
 !
 !
 subroutine read_params(file_list, params)
   implicit none
   character(len=120),intent(in)    :: file_list(:)
   type(keyword_list),intent(inout) :: params

   integer                          :: nfiles
   integer                          :: error
   type(c_ptr)                      :: params_ptr
   type(c_funptr)                   :: parse_params_ptr
   type(h5o_info_t)                 :: infobuf
   integer(hid_t)                   :: file_id
   integer(hsize_t)                 :: idx
   integer                          :: ret_value
   integer                          :: nkey
   integer                          :: igrp,ikey

   ! total number of hdf5 chkpoint files to scan
   !
   nfiles = size(file_list)

   ! open first file and read in parameter list
   !
   call h5fopen_f (file_list(1), H5F_ACC_RDONLY_F, file_id, error)
   
   ! Open simulation group
   !
   call h5oget_info_by_name_f(file_id, "/", infobuf, error)

   ! initialize group data pointer
   grp_data%ngrp      = 0
   grp_data%ndset     = 0
   grp_data%root_name = "/"
   ! Iterate over the groups in root
   !
   idx              = 0
   parse_params_ptr = c_funloc(parse_params)
   null_ptr         = c_loc(params)
   call h5literate_f(file_id, H5_INDEX_NAME_F, H5_ITER_NATIVE_F, idx, parse_params_ptr, params_ptr, ret_value, error) 

   call h5fclose_f(file_id, error)

 end subroutine read_params

 subroutine read_trajectories(file_list, ti, tf)
   implicit none
   character(len=120),intent(in)  :: file_list(:)
   double precision, intent(in)               :: ti
   double precision, intent(in)               :: tf

   integer                        :: nfiles
   integer                        :: ret_value, error
   type(c_ptr)                    :: null_ptr
   type(c_funptr)                 :: parse_wfn_ptr
   type(h5o_info_t)               :: infobuf
   integer(hid_t)                 :: file_id, grp_id
   integer(hsize_t)               :: idx
   integer                        :: ifile, igrp


   ! total number of hdf5 chkpoint files to scan
   !
   nfiles = size(file_list)

   do ifile = 1,nfiles

    ! open first file and read in parameter list
    call h5fopen_f (file_list(1), H5F_ACC_RDONLY_F, file_id, error)

    ! open wavefunction group
    call h5gopen_f(file_id, "/wavefunction", grp_id, error)

    ! initialize group data pointer
    grp_data%ngrp      = 0
    grp_data%ndset     = 0
    grp_data%root_name = "/wavefunction"
    ! Iterate over the groups in root
    !
    idx             = 0
    parse_wfn_ptr   = c_funloc(parse_wfn)
    null_ptr        = C_NULL_PTR
    call h5literate_f(grp_id, H5_INDEX_NAME_F, H5_ITER_NATIVE_F, idx, parse_wfn_ptr, null_ptr, ret_value, error)
   
    do igrp = 1,grp_data%ngrp
     ! if this is a keyword section.
     if(index(grp_data%grp_names(igrp),'trajectory').ne.0) then
      print *,'trajectory=',grp_data%grp_names(igrp)
 !     call h5dopen_f(file, dataset, dset, hdferr)
 !     call h5aopen_f(grp_data%grp_ids(igrp), attribute, attr, error)
 
      ! get the number of keywords
 !     call h5aget_num_attrs_f(grp_data%grp_ids(igrp), nkey, error)
 !     print *,'nattr=',nkey
      ! loop over attribles
 !    do ikey = 1,nkey  
 !     call 

     endif
    enddo

    call h5gclose_f(grp_id,error)
    call h5fclose_f(file_id, error)
   enddo 

 end subroutine read_trajectories

 !
 !
 !
 function parse_params(obj_id, item_name, info, params_ptr) result(return_val) bind(c)
   use hdf5
   use iso_c_binding
   implicit none

   integer(hid_t), value, intent(in)           :: obj_id
   character(len=1),dimension(1:20),intent(in) :: item_name ! Must have LEN=1 for bind(C) strings   
   type(c_ptr),value, intent(in)               :: info
   type(c_ptr),value, intent(in)               :: params_ptr

   integer                                     :: stat, return_val
   integer                                     :: indx, len
   type(keyword_list),pointer                  :: keywords
   type(h5o_info_t), target                    :: infobuf
   character(len=20)                           :: name_string
   integer(hsize_t)                            :: nkey,ikey

    !
    ! Initialize FORTRAN interface.
    !
    CALL h5open_f(stat)

    return_val = stat
    
    name_string(1:20) = " "
    len = 0
    do 
       len = len + 1
       if(item_name(len)(1:1).eq.C_NULL_CHAR) exit
       name_string(len:len) = item_name(len)(1:1)
    enddo
    len = len - 1 ! subtract NULL character
    print *,'name_string=|',name_string(1:len),'|'

    call c_f_pointer(params_ptr, keywords)
    !
    ! Get type of the object and display its name and type.
    ! The name of the object is passed to this function by
    ! the Library.
    !
    call H5Oget_info_by_name_f(obj_id, name_string(1:len), infobuf, stat)
    return_val = return_val.and.stat

    ! if this is a keyword group, read all the attributes
    if(infobuf%type.eq.H5O_TYPE_GROUP_F .and. index(name_string(1:len),'keywords').ne.0) then
     nkey = infobuf%num_attrs
     do ikey = 1,nkey
       call h5aopen_by_idx_f(obj_id, name_string(1:len), H5_INDEX_NAME_F, H5_ITER_NATIVE_F, ikey, attr_id, stat)  
       call h5aget_name_f(attr_id, size, buf, hdferr)
       call 

     enddo
    elseif(infobuf%type.eq.H5O_TYPE_DATASET_F) then
      return
    else 
      stop 'ERROR -- cannot identify section...'
    endif

    return
 end function parse_simulation

 function parse_wfn(obj_id, item_name, info, null_ptr) result(return_val) bind(c)
   use hdf5
   use iso_c_binding
   implicit none

   integer(hid_t), value, intent(in)           :: obj_id
   character(len=1),dimension(1:20),intent(in) :: item_name ! Must have LEN=1 for bind(C) strings   
   type(c_ptr),value, intent(in)               :: info
   type(c_ptr),value, intent(in)               :: null_ptr

   integer(hsize_t)                            :: dset_mem, dset_size
   integer                                     :: stat, return_val
   integer                                     :: indx, len
   type(h5o_info_t), target                    :: infobuf
   character(len=20)                           :: name_string
   double precision, dimension(:), allocatable :: tbuf(:)

    !
    ! Initialize FORTRAN interface.
    !
    CALL h5open_f(stat)

    return_val = stat

    name_string(1:20) = " "
    len = 0
    do
       len = len + 1
       if(item_name(len)(1:1).eq.C_NULL_CHAR) exit
       name_string(len:len) = item_name(len)(1:1)
    enddo
    len = len - 1 ! subtract NULL character
    print *,'name_string=|',name_string(1:len),'|'

    call H5Oget_info_by_name_f(obj_id, name_string(1:len), infobuf, stat)
    return_val = return_val.and.stat

    ! check if this group is a trajectory
    if(infobuf%type.eq.H5O_TYPE_GROUP_F) then
 
     call h5literate_f(obj_id, H5_INDEX_NAME_F, H5_ITER_NATIVE_F, idx, parse_traj_ptr, null_ptr, ret_value, error)

    elseif(infobuf%type.eq.H5O_TYPE_DATASET_F) then
      
      ! if this is time list, slurp in times
      if(index(name_string, 'time') .ne. 0) then
        call h5dget_storage_size_f(obj_id, dset_mem, stat)
        dset_size = int(dset_mem/8.)
        print *,'time size, mem=',dset_size,dset_mem
        allocate(tbuf(dset_size))
        call h5dread_f(obj_id, H5T_NATIVE_FLOAT, tbuf, (/ dset_size /), stat)
        print *,'read complete'

      elseif(index(name_string,'energy')) then
        ! don't worry about this now
        return

      elseif(index(name_string,'pop')) then
        ! don't worry about this now
        return

      else 
        stop 'error: do not recognize wavefunction group data set'
  
    endif

    if(allocated)deallocate(tbuf)
    return
 end function parse_wfn

 !
 !
 !
 subroutine parse_traj(traj_id, traj_name)
   implicit none
   integer(hid_t), value, intent(in)           :: traj_id
   character(len=20),,intent(in)               :: traj_name
   type(h5o_info_t), target                    :: info

   print *,'should be traj name: ',traj_name
   
 end subroutine

 !
 !
 !
 subroutine read_chkpt(t_min, t_max, chkpt_file, traj_list )
   implicit none
   real, intent(in)               :: t_min
   real, intent(in)               :: t_max
   character(len=120), intent(in) :: chkpt_file
   type(traj_basis),intent(inout) :: traj_list

   print *,'chkpt_file=',chkpt_file

   if(allocated(traj_list%t_basis))deallocate(traj_list%t_basis)
   if(allocated(traj_list%times))deallocate(traj_list%times)

   return
 end subroutine read_chkpt

 end module libchkpt



