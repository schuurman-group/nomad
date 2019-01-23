goal: propagate.x

#compilers
#F90 = gfortran -fbounds-check
#F90=ifort -warn -nofor_main -ipo -O2 -no-prec-div -openmp -assume cc_omp -complex_limited_range -debug extended -traceback -static-intel
F90=/home/schuurm/software/hdf5/bin/h5fc -warn -nofor_main -ipo -O2 -no-prec-div -openmp -assume cc_omp -complex_limited_range -debug extended -traceback

#additional libraries
LIBS= -L /usr/local/intel/ics/2013.0.028/composer_xe_2013.1.117/mkl/lib/intel64 -mkl=sequential -lpthread -L /home/schuurm/software/hdf5/lib -lhdf5_fortran -lhdf5 -L /usr/lib64 -lz

#object files
OBJS=propagate.o libchkpt.o lookup.o

######## executables ##########
propagate.x:   propagate_driver.o $(OBJS)
	$(F90) -o propagate.x propagate_driver.o $(LIBS) $(OBJS)

######## source ###############

propagate_driver.o: propagate_driver.f90 propagate.o
	$(F90) -c propagate_driver.f90

propagate.o:        propagate.f90 libchkpt.o lookup.o
	$(F90) -c propagate.f90

libchkpt.o:         libchkpt.f90 lookup.o
	$(F90) -c libchkpt.f90

lookup.o:           lookup.f90
	$(F90) -c lookup.f90

clean:
	-rm *.o *.mod propagate.x hamiltonian.x
