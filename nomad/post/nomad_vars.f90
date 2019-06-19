!
!
!
module nomad_vars
 use hdf5

 implicit none
 public init_table
 public set_entry 
 public get_entry
 public var_type

  interface set_entry
    module procedure set_entry_string
    module procedure set_entry_integer
    module procedure set_entry_real
  end interface set_entry

  interface get_entry
    module procedure get_entry_string
    module procedure get_entry_integer
    module procedure get_entry_real
  end interface get_entry

  integer                         :: n_str = 21
  integer                         :: n_int = 11
  integer                         :: n_real = 15
  character(len=30),dimension(21) :: str_keywords =                                                             &
                                     (/ 'cwd', 'log_file', 'chkpt_file', 'parallel', 'ansatz', 'opfile',          &
                                        'adapt_basis', 'init_conds', 'integral_eval', 'interface', 'propagator', &
                                        'surface', 'init_brightest', 'virtual_basis', 'use_atom_lib', 'restart', &
                                        'renorm', 'auto', 'phase_prop', 'matching_pursuit', 'init_amp_overlap' /)
  character(len=30),dimension(11) :: int_keywords =                                                              &
                                     (/ 'comm', 'rank', 'nproc', 'n_states', 'n_init_traj', 'ordr_max',          &
                                        'init_state', 'init_mode_min_olap', 'seed', 'integral_order', 'print_level' /)
  character(len=30),dimension(15) :: real_keywords =                                                                         &
                                     (/ 'init_mode_min_olap', 'distrib_compression', 'simulation_time', 'default_time_step', &
                                        'coupled_time_step', 'energy_jump_toler', 'pop_jump_toler', 'pot_shift',             &
                                        'sinv_thrsh', 'norm_thresh', 'sij_thresh', 'hij_coup_thresh',                        &
                                        'spawn_pop_thresh', 'spawn_coup_thresh', 'continuous_min_overlap' /)

 public 
  type keyword_list
    character(len=120),allocatable     :: str_val(:)
    integer,allocatable                :: int_val(:)
    double precision,allocatable       :: real_val(:)          
  end type keyword_list


 contains

 !
 !
 !
 subroutine init_table(klist) 
  implicit none
  type(keyword_list), intent(inout) :: klist
 
  if(allocated(klist%str_val)) deallocate(klist%str_val)
  if(allocated(klist%int_val)) deallocate(klist%int_val)
  if(allocated(klist%real_val)) deallocate(klist%real_val)

  allocate(klist%str_val(n_str))
  allocate(klist%int_val(n_int))
  allocate(klist%real_val(n_real))

  klist%str_val  = ''                
  klist%int_val  = 0
  klist%real_val = 0.

 end subroutine init_table

 !
 !
 !
 subroutine set_entry_string(klist, key, val)
  implicit none
  type(keyword_list), intent(inout) :: klist
  character(len=30), intent(in)     :: key  
  character(len=120), intent(in)    :: val
  integer                           :: i
  logical                           :: found

  i = 1
  found = .false.
  search_keys: do 
    if (i.gt.n_str) exit
    if (str_keywords(i) == key) then
        found = .true.
        klist%str_val(i) = val
        exit
    endif
    i = i + 1
  enddo search_keys

  if(.not.found)write(*,*)'Key: '//key//' not found. Skipping.'

 end subroutine set_entry_string

 !
 !
 !
 subroutine set_entry_integer(klist, key, val)
   implicit none
   type(keyword_list), intent(inout) :: klist
   character(len=30), intent(in)     :: key
   integer,intent(in)                :: val
   integer                           :: i
   logical                           :: found

  i = 1
  found = .false.
  search_keys: do
    if (i.gt.n_int) exit
    if (int_keywords(i) == key) then
        found = .true.
        klist%int_val(i) = val
        exit
    endif
    i = i + 1
  enddo search_keys

  if(.not.found)write(*,*)'Key: '//key//' not found. Skipping.'

 end subroutine set_entry_integer

 !
 !
 !
 subroutine set_entry_real(klist, key, val)
   implicit none
   type(keyword_list), intent(inout) :: klist
   character(len=30), intent(in)     :: key
   real,intent(in)                   :: val
   integer                           :: i
   logical                           :: found
   
  i = 1
  found = .false.
  search_keys: do
    if (i.gt.n_real) exit
    if (real_keywords(i) == key) then
        found = .true.
        klist%real_val(i) = val
        exit
    endif
    i = i + 1
  enddo search_keys

  if(.not.found)write(*,*)'Key: '//key//' not found. Skipping.'

 end subroutine set_entry_real

 !
 !
 ! 
 subroutine get_entry_string (klist, key, val)
  implicit none
  type(keyword_list), intent(in)     :: klist
  character(len=30), intent(in)      :: key
  character(len=120),intent(out)     :: val
  integer                            :: i
  
  i = 1
  search_keys: do 
    if(i.gt.n_real) stop 'key not found'
    if (str_keywords(i) == key) then
        val = klist%str_val(i)
        exit search_keys 
    endif
    i = i + 1
  enddo search_keys

 end subroutine get_entry_string

 !
 ! 
 subroutine get_entry_integer (klist, key, val)
  implicit none
  type(keyword_list), intent(in)     :: klist
  character(len=30), intent(in)      :: key
  integer,intent(out)                :: val
  integer                            :: i

  i = 1
  search_keys: do
    if(i.gt.n_int) stop 'key  not found'
    if (int_keywords(i) == key) then
        val = klist%int_val(i)
        exit search_keys 
    endif
    i = i + 1
  enddo search_keys

 end subroutine get_entry_integer

 !
 ! 
 subroutine get_entry_real (klist, key, val)
  implicit none
  type(keyword_list), intent(in)     :: klist
  character(len=30), intent(in)      :: key
  double precision,intent(out)       :: val
  integer                            :: i

  i = 1
  search_keys: do
    if(i.gt.n_real) stop 'key not found'
    if (real_keywords(i) == key) then
        val = klist%real_val(i)
        exit search_keys 
    endif
    i = i + 1
  enddo search_keys

 end subroutine get_entry_real

 !
 !
 !
 function var_type(key)
  implicit none
  character(len=30), intent(in)      :: key
  integer(hid_t)                     :: var_type 

  if(any(str_keywords==key)) then
    var_type = H5T_STRING_F
  elseif(any(int_keywords==key)) then
    var_type = H5T_INTEGER_F
  elseif(any(real_keywords==key)) then
    var_type = H5T_FLOAT_F
  else 
    var_type = -1
    write(*,*) 'key='//key//' not recognized...'
  endif

  return
 end function var_type

 !
 !
 !
 function var_valid(key)
  implicit none
  character(len=30), intent(in)  :: key
  logical                        :: var_valid

  var_valid = any(str_keywords == key).or. &
              any(int_keywords==key).or.   &
              any(real_keywords==key)

  return 
 end function var_valid

end module nomad_vars



