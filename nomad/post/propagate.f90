!
!
!
module propagate_module
 use hdf5
 use lookup
 use libchkpt

 public keyword_list
 public read_options
 public initialize_prop
 public propagate

 double precision               :: ti, tf
 character(len=120)             :: ansatz
 character(len=120)             :: chkpt               
  
 integer                        :: n_sim
 character(len=120),allocatable :: nomad_files(:)
 type(keyword_list)             :: nomad_params


 contains


 !
 !
 !
 subroutine read_options()
  implicit none
  integer                   :: n_arg, i_arg
  character(len=120)        :: abuf

  n_arg = iargc()

  if(n_arg.lt.1)stop 'need to specify ansatz'

  ! read in ansatz
  call getarg(1,abuf)
  ansatz = adjustl(abuf)

  i_arg = 2
  do while (i_arg < n_arg)
    call getarg(i_arg,abuf)

    if (trim(adjustl(abuf)) == '-ti') then
       i_arg = i_arg + 1
       call getarg(i_arg,abuf)
       abuf = trim(adjustl(abuf))
       read(abuf,*) ti

    elseif (trim(adjustl(abuf)) == '-tf') then
       i_arg = i_arg + 1
       call getarg(i_arg,abuf)
       abuf = trim(adjustl(abuf))
       read(abuf,*) tf

    elseif (trim(adjustl(abuf)) == '-file') then
       i_arg = i_arg + 1
       call getarg(i_arg,abuf)
       abuf = trim(adjustl(abuf))
       read(abuf,*) chkpt

    else
       stop ' command line argument argument not recognized...'

    endif

    i_arg = i_arg + 1
  enddo

 end subroutine read_options

 !
 !
 !
 subroutine initialize_prop()
  implicit none
  integer         :: ierr  
 
  ! initialize HDF5 system
  call h5open_f(ierr)

  ! identify all the checkpont files that are going to be read
  call build_file_list()

  ! read the simulation parameters, check that all files are
  ! consistent in the ways that matter
  call init_table(nomad_params, 100)
  call read_params(nomad_files, nomad_params)

  ! read in the trajectory information
  call read_trajectories(nomad_files, ti, tf)


 end subroutine initialize_prop

 !
 !
 !
 subroutine propagate()
  implicit none



 end subroutine propagate


!------------------------------------------------

 !
 !
 !
 subroutine build_file_list()
  implicit none
  character(len=5)        :: file_index
  character(len=120)      :: file_name
  logical                 :: file_exists
  integer                 :: i

  i = 1
  file_loop: do
    write(file_index,'(i5)')i
    file_name   = trim(adjustl(chkpt))//'.'//trim(adjustl(file_index))
    inquire(file=file_name, exist=file_exists)
 
    if(.not.file_exists) exit file_loop
    i = i + 1
  enddo file_loop
 
  n_sim = i-1
  allocate(nomad_files(n_sim))
 
  do i=1,n_sim
    write(file_index,'(i5)')i
    file_name      = trim(adjustl(chkpt))//'.'//trim(adjustl(file_index))
    nomad_files(i) = file_name
  enddo


 end subroutine build_file_list

end module propagate_module

