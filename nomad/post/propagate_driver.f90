!
!
!
program nomad_prop
  use propagate_module

  call read_options
  call initialize_prop
  call propagate
end program nomad_prop
