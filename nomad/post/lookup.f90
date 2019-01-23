!
!
!
module lookup

 implicit none
 public init_table
 public set_entry 
 public get_entry

  interface set_entry
    module procedure set_entry_string
    module procedure set_entry_real
    module procedure set_entry_integer
  end interface set_entry

 public 
  type keyword_list
    integer                            :: n_keys
    integer                            :: max_keys
    character(len=30),allocatable      :: keys(:)
    character(len=120),allocatable     :: values(:)
  end type keyword_list

 contains

 !
 !
 !
 subroutine init_table(klist, max_keys) 
  implicit none
  type(keyword_list), intent(inout) :: klist
  integer,intent(in)                :: max_keys

  klist%max_keys = max_keys
  klist%n_keys   = 0
  allocate(klist%keys(max_keys))
  allocate(klist%values(max_keys))

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
    if (i.gt.klist%n_keys) exit
    if (klist%keys(i) == key) then
        found = .true.
        exit
    endif
    i = i + 1
  enddo search_keys

  if (found) then
    klist%values(i) = val
  else
    if (klist%n_keys+1.gt.klist%max_keys)stop 'n_keys > max_keys'
    klist%n_keys               = klist%n_keys + 1
    klist%keys(klist%n_keys)   = key
    klist%values(klist%n_keys) = val
  endif

 end subroutine set_entry_string

 !
 !
 !
 subroutine set_entry_real(klist, key, val)
   implicit none
   type(keyword_list), intent(inout) :: klist
   character(len=30), intent(in)     :: key
   real,intent(in)                   :: val
   character(len=120)                :: val_str
   
   write(val_str,*)val
   call set_entry_string(klist, key, val_str)

 end subroutine set_entry_real

 !
 !
 !
 subroutine set_entry_integer(klist, key, val)
   implicit none
   type(keyword_list), intent(inout) :: klist
   character(len=30), intent(in)     :: key
   integer,intent(in)                :: val
   character(len=120)                :: val_str

   write(val_str,'(i120)')val
   call set_entry_string(klist, key, val_str)
 end subroutine set_entry_integer

 !
 !
 ! 
 subroutine get_entry (klist, key, val)
  implicit none
  type(keyword_list), intent(in)     :: klist
  character(len=30), intent(in)      :: key
  character(len=120),intent(out)     :: val
  integer                            :: i
  
  i = 1
  search_keys: do 
    if(i.gt.klist%n_keys) stop 'key not found'
    if (klist%keys(i) == key) then
        val = klist%values(i)
        exit search_keys 
    endif
    i = i + 1
  enddo search_keys

 end subroutine get_entry

end module lookup



