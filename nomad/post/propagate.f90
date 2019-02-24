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
 type(keyword_list)             :: params


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
  call init_table(params, 100)
  call read_params(chkpt, params)

  ! read in the trajectory information
  call read_trajectories(chkpt, ti, tf)


 end subroutine initialize_prop

 !
 !
 !
 subroutine propagate()
  implicit none



 end subroutine propagate


!------------------------------------------------

