goal: propagate.x

#compilers
#F90 = gfortran -fbounds-check
#F90=ifort -warn -nofor_main -ipo -O2 -no-prec-div -openmp -assume cc_omp -complex_limited_range -debug extended -traceback -static-intel
F90=/globalhome/schuurm/software/hdf5/bin/h5fc -warn -nofor_main -ipo -O2 -no-prec-div -qopenmp -assume cc_omp -complex_limited_range -debug extended -traceback

#additional libraries
LIBS= -L /software/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64 -mkl=sequential -lpthread -L /home/schuurm/software/hdf5/lib -lhdf5_fortran -lhdf5 -L /usr/lib64 -lz

#object files
OBJS=propagate.o libchkpt.o nomad_vars.o

######## executables ##########
propagate.x:   propagate_driver.o $(OBJS)
	$(F90) -o propagate.x propagate_driver.o $(LIBS) $(OBJS)

######## source ###############

propagate_driver.o: propagate_driver.f90 propagate.o
	$(F90) -c propagate_driver.f90

propagate.o:        propagate.f90 libchkpt.o nomad_vars.o
	$(F90) -c propagate.f90

libchkpt.o:         libchkpt.f90 nomad_vars.o
	$(F90) -c libchkpt.f90

lookup.o:           nomad_vars.f90
	$(F90) -c nomad_vars.f90

clean:
	-rm *.o *.mod propagate.x hamiltonian.x
