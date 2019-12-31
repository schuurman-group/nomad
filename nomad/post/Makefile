goal: fms.so

#compilers
F90=ifort -i8 -warn -nofor_main -ipo -O2 -no-prec-div -qopenmp -assume cc_omp -complex_limited_range -debug extended -traceback

#additional libraries
LIBS= -L /software/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64_lin -mkl=sequential -lpthread -L /usr/lib64 -lz

#object files
OBJS=libprop.o fms.o

######## library ######
#fms.so:	fms.o libprop.o
#	ifort -shared fms.so fms.o libprop.o

fms.so:	fms.f90 
	$(F90) -fpic -shared -o fms.so libprop.f90 fms.f90

#libprop.o: libprop.f90
#	$(F90) -fpic -c libprop.f90

clean:
	-rm *.o *.mod *.so
