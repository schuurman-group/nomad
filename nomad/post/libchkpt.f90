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

 !
 !
 !
 subroutine read_params(chkpt, params)
   implicit none
   character(len=120),intent(in)    :: chkpt
   type(keyword_list),intent(inout) :: params

   integer                          :: error
   type(c_ptr)                      :: params_ptr
   type(c_funptr)                   :: parse_kwd_ptr
   type(h5o_info_t)                 :: infobuf
   integer(hid_t)                   :: file_id
   integer(hsize_t)                 :: idx
   integer                          :: ret_value
   integer                          :: nkey
   integer                          :: igrp,ikey

   ! open first file and read in parameter list
   !
   call h5fopen_f (chkpt, H5F_ACC_RDONLY_F, file_id, error)
   
   ! Open simulation group
   !
   call h5oget_info_by_name_f(file_id, "/", infobuf, error)

   idx           = 0
   parse_kwd_ptr = c_funloc(parse_keywords)
   params_ptr    = c_loc(params)
   call h5literate_f(file_id, H5_INDEX_NAME_F, H5_ITER_NATIVE_F, idx, parse_kwd_ptr, params_ptr, ret_value, error) 

!   params%n_keys = params%n_keys+1

   call h5fclose_f(file_id, error)

 end subroutine read_params

 !
 !
 !
 subroutine read_trajectories(chkpt, ti, tf)
   implicit none
   character(len=120),intent(in)  :: chkpt 
   double precision, intent(in)   :: ti
   double precision, intent(in)   :: tf

   integer                        :: ret_value, error
   type(c_ptr)                    :: null_ptr

 end subroutine read_trajectories

 !
 !
 !
 function parse_keywords(obj_id, item_name, info, params_ptr) result(return_val) bind(c)
   use iso_c_binding
   implicit none

   integer(hid_t), value, intent(in)           :: obj_id
   character(len=1),dimension(1:20),intent(in) :: item_name ! Must have LEN=1 for bind(C) strings   
   type(c_ptr),value                           :: info
   type(c_ptr),value                           :: params_ptr

   integer                                     :: stat, return_val
   integer                                     :: indx, len
   type(keyword_list),pointer                  :: keywords
   type(h5o_info_t), target                    :: infobuf
   character(len=20)                           :: name_string
   integer(hsize_t)                            :: nkey,ikey,stor_size
   integer(hid_t)                              :: attr_id, lapl_id, data_type
   integer(size_t)                             :: n_size
   character(len=120)                          :: n_buf
   character(len=120),target                   :: a_val
   integer(hsize_t),dimension(1)               :: dims
   type(c_ptr)                                 :: read_ptr

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
     print *,'section: ',name_string(1:len),' is a keyword section'
     nkey = infobuf%num_attrs
     print *,'number of attributes: ',nkey
     do ikey = 0,nkey-1,1
       call h5aget_name_by_idx_f(obj_id, name_string(1:len), H5_INDEX_NAME_F, H5_ITER_NATIVE_F, ikey, n_buf, stat)
       print *,"attr_name=",trim(adjustl(n_buf))
       call h5aopen_by_name_f(obj_id, name_string(1:len), trim(adjustl(n_buf)), attr_id, stat)
       print *,"stat,attr_id=",stat,attr_id 
       call h5aget_type_f(attr_id, data_type, stat) 
       print *,"data_type=",data_type
       call h5aget_storage_size_f(attr_id, stor_size, stat)
       dims(1) = 120
       print *,'storage_size=',stor_size
       read_ptr = c_loc(a_val(1:1))
       call h5aread_f(attr_id, data_type, read_ptr, stat)
       print *,'read stat=',stat
       print *,"attr_value=",trim(adjustl(a_val))
       call h5aclose_f(attr_id, stat)
     enddo
    elseif(infobuf%type.eq.H5O_TYPE_DATASET_F) then
      return
    else
      stop 'ERROR -- cannot identify section...'
    endif

    return
 end function parse_keywords


 !
 !
 !
 function parse_wfn(obj_id, item_name, info, null_ptr) result(ret_val) bind(c)
   use iso_c_binding
   implicit none

   integer(hid_t), value, intent(in)           :: obj_id
   character(len=1),dimension(1:20),intent(in) :: item_name ! Must have LEN=1 for bind(C) strings   
   type(c_ptr),value                           :: info
   type(c_ptr),value                           :: null_ptr
   integer                                     :: ret_val

   type(c_funptr)                              :: parse_traj_ptr
   integer(hsize_t)                            :: idx, dset_mem, dset_size
   integer                                     :: stat 
   integer                                     :: indx, len
   type(h5o_info_t), target                    :: infobuf
   character(len=20)                           :: name_string
   double precision, dimension(:), allocatable :: tbuf(:)

    !
    ! Initialize FORTRAN interface.
    !
    CALL h5open_f(stat)

    parse_traj_ptr = c_funloc(parse_traj)
    ret_val = stat

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
    ret_val = ret_val.and.stat

    ! check if this group is a trajectory
    if(infobuf%type.eq.H5O_TYPE_GROUP_F) then
 
     call h5literate_f(obj_id, H5_INDEX_NAME_F, H5_ITER_NATIVE_F, idx, parse_traj_ptr, null_ptr, ret_val, stat)

    elseif(infobuf%type.eq.H5O_TYPE_DATASET_F) then
      
      ! if this is time list, slurp in times
      if(index(name_string, 'time') .ne. 0) then
        call h5dget_storage_size_f(obj_id, dset_mem, stat)
        dset_size = int(dset_mem/8.)
        print *,'time size, mem=',dset_size,dset_mem
        allocate(tbuf(dset_size))
!        call h5dread_f(obj_id, H5T_NATIVE_FLOAT, tbuf, (/ dset_size /), stat)
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
     
    else
      stop 'ERROR: do not recognize data type...'
    endif

    if(allocated(tbuf))deallocate(tbuf)
    return
 end function parse_wfn

 !
 !
 !
 function parse_traj(obj_id, item_name, info, null_ptr) result(ret_val) bind(c)
   use iso_c_binding
   implicit none

   integer(hid_t), value, intent(in)           :: obj_id
   character(len=1),dimension(1:20),intent(in) :: item_name ! Must have LEN=1 for bind(C) strings   
   type(c_ptr),value                           :: info
   type(c_ptr),value                           :: null_ptr
   integer                                     :: ret_val

   ret_val = 1
   
 end function parse_traj

end module libchkpt



