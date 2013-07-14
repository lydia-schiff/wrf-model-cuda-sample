CC=gcc
#CFLAGS = -std=c99 -O3 -ftree-vectorize -ftree-loop-linear -funroll-loops -lm -lgomp -fopenmp
CFLAGS=-gdwarf-2 -g3 -O0 -W -Wall -std=c99 -lm # debug with array macro support
#CFLAGS += -g

FC=gfortran
#FFLAGS=-g -O0 -pg #debug
FFLAGS = -O3 -ftree-vectorize -ftree-loop-linear -funroll-loops -ffree-line-length-none -lgomp -fopenmp
#FFLAGS += -Wall -Wextra
#FFLAGS += -g

CUFLAGS= -O3 --ptxas-options=-v -arch sm_20 -m64 -fmad=false
#-use_fast_math -g -G -DEXACT_MATH -use_fast_math  
CULIBS = -lcudart -lcuda

CSRC = advance_mu_t.c advance_mu_t_driver.c ulps.c

ALL = advance_mu_t_fortran advance_mu_t_driver advance_mu_t_cuda_no_coalesced_no_async advance_mu_t_cuda_with_coalesced_no_async

all: $(ALL)

advance_mu_t_fortran: advance_mu_t_driver.f90 module_small_step_em.f90 module_configure.f90
	$(FC) $(FFLAGS) -c module_configure.f90
	$(FC) $(FFLAGS) -c module_small_step_em.f90
	$(FC) $(FFLAGS) advance_mu_t_driver.f90 module_small_step_em.o -o $@

advance_mu_t_driver: $(CSRC)
	$(CC) $(CFLAGS) $^ -o $@

advance_mu_t_cuda_no_coalesced_no_async: advance_mu_t_driver.cu advance_mu_t_no_async.cu advance_mu_t_kernel.cu common.cu 
	nvcc $(CUFLAGS)  -o $@ $^ $(CULIBS)
	#nvcc $(CUFLAGS) -c advance_mu_t_driver.cu $(CULIBS)
	#nvcc $(CUFLAGS) -c common.cu $(CULIBS)
	#nvcc $(CUFLAGS) -c advance_mu_t_no_async.cu $(CULIBS)
	#nvcc $(CUFLAGS) -c advance_mu_t_kernel.cu $(CULIBS)
	
	#nvcc $(CUFLAGS) -o $@ advance_mu_t_driver.o common.o advance_mu_t_no_async.o advance_mu_t_kernel.o 

advance_mu_t_cuda_with_coalesced_no_async: advance_mu_t_driver.cu advance_mu_t_no_async.cu advance_mu_t_kernel.cu common.cu 
	nvcc $(CUFLAGS) -o $@ $^ $(CULIBS) -DCOALESCED

#wsm5_cuda_with_coalesced_with_async: wsm5_driver.o module_mp_wsm5_with_async.cu module_mp_wsm5_kernel.cu
#	nvcc $(CUFLAGS) -o $@ $^ $(CULIBS) -DCOALESCED

#wsm5_cuda_no_coalesced_with_async: wsm5_driver.o module_mp_wsm5_with_async.cu module_mp_wsm5_kernel.cu
#	nvcc $(CUFLAGS) -o $@ $^ $(CULIBS)

#wsm5_driver.o: wsm5_driver.cu
#	nvcc $(CUFLAGS) -c $^

analyze:
	clang --analyze $(CSRC)

clean:
	rm -fr *.mod *.plist *.o $(ALL)

