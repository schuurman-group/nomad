goal: fms.so

#compilers
F90=ifort -warn -nofor_main -ipo -O2 -no-prec-div -qopenmp -assume cc_omp -complex_limited_range -debug extended -traceback -check all -mkl:serial
#F90=ifort -i8 -warn -nofor_main -ipo -O2 -no-prec-div -qopenmp -assume cc_omp -complex_limited_range -debug extended -traceback -check all
#F90=gfortran -fdefault-integer-8 -fcheck=all -fbacktrace

#additional libraries
#LIBS= -L /software/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64_lin -mkl=sequential -lpthread
#LIBS= -L/software/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64 libmkl_scalapack_ilp64.a libmkl_intel_ilp64.a libmkl_intel_thread.a libmkl_core.a libiomp5.a libpthread.a libm.a libdl.a
#LIBS= -L /usr/lib -L /usr/lib64
LIBS=''

#object files
OBJS=accuracy.o math.o lib_traj.o lib_amp.o fms.o

######## library ######
#fms.so:	fms.o libprop.o
#	ifort -shared fms.so fms.o libprop.o

fms.so:	fms.f90 
	$(F90) -fPIC -shared -o fms.so accuracy.f90 math.f90 lib_traj.f90 lib_amp.f90 fms.f90 $(LIBS)

#libprop.o: libprop.f90
#	$(F90) -fpic -c libprop.f90

clean:
	-rm *.o *.mod *.so
