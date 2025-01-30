/* 
 * ------------------------------------------------------------------------------
 *
 * MIT License
 *
 * Copyright (c) 2021 Parallel Applications Modelling Group - GMAP
 *      GMAP website: https://gmap.pucrs.br
 *
 * Pontifical Catholic University of Rio Grande do Sul (PUCRS)
 * Av. Ipiranga, 6681, Porto Alegre - Brazil, 90619-900
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ------------------------------------------------------------------------------
 *
 * The original NPB 3.4 version was written in Fortran and belongs to:
 *      http://www.nas.nasa.gov/Software/NPB/
 *
 * Authors of the Fortran code:
 *      D. Bailey
 *      W. Saphir
 *
 * ------------------------------------------------------------------------------
 *
 * The serial C++ version is a translation of the original NPB 3.4
 * Serial C++ version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-SER
 *
 * Authors of the C++ code:
 *      Dalvan Griebler <dalvangriebler@gmail.com>
 *      Gabriell Araujo <hexenoften@gmail.com>
 *      Júnior Löff <loffjh@gmail.com>
 *
 * ------------------------------------------------------------------------------
 *
 * The hip version is a parallel implementation of the serial C++ version
 * hip version: https://github.com/GMAP/NPB-GPU/tree/master/hip
 *
 * Authors of the hip code:
 *      Gabriell Araujo <hexenoften@gmail.com>
 *
 * ------------------------------------------------------------------------------
 */

#include <omp.h>
#include <hip/hip_runtime.h>
#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

/*
 * ---------------------------------------------------------------------
 * u0, u1, u2 are the main arrays in the problem. 
 * depending on the decomposition, these arrays will have different 
 * dimensions. to accomodate all possibilities, we allocate them as 
 * one-dimensional arrays and pass them to subroutines for different 
 * views
 * - u0 contains the initial (transformed) initial condition
 * - u1 and u2 are working arrays
 * - twiddle contains exponents for the time evolution operator. 
 * ---------------------------------------------------------------------
 * large arrays are in common so that they are allocated on the
 * heap rather than the stack. this common block is not
 * referenced directly anywhere else. padding is to avoid accidental 
 * cache problems, since all array sizes are powers of two.
 * ---------------------------------------------------------------------
 * we need a bunch of logic to keep track of how
 * arrays are laid out. 
 *
 * note: this serial version is the derived from the parallel 0D case
 * of the ft NPB.
 * the computation proceeds logically as
 *
 * set up initial conditions
 * fftx(1)
 * transpose (1->2)
 * ffty(2)
 * transpose (2->3)
 * fftz(3)
 * time evolution
 * fftz(3)
 * transpose (3->2)
 * ffty(2)
 * transpose (2->1)
 * fftx(1)
 * compute residual(1)
 * 
 * for the 0D, 1D, 2D strategies, the layouts look like xxx
 *
 *            0D        1D        2D
 * 1:        xyz       xyz       xyz
 * 2:        xyz       xyz       yxz
 * 3:        xyz       zyx       zxy
 * the array dimensions are stored in dims(coord, phase)
 * ---------------------------------------------------------------------
 * if processor array is 1x1 -> 0D grid decomposition
 * 
 * cache blocking params. these values are good for most
 * RISC processors.  
 * FFT parameters:
 * fftblock controls how many ffts are done at a time. 
 * the default is appropriate for most cache-based machines
 * on vector machines, the FFT can be vectorized with vector
 * length equal to the block size, so the block size should
 * be as large as possible. this is the size of the smallest
 * dimension of the problem: 128 for class A, 256 for class B
 * and 512 for class C.
 * ---------------------------------------------------------------------
 */
#define	FFTBLOCK_DEFAULT (DEFAULT_BEHAVIOR)
#define	FFTBLOCKPAD_DEFAULT (DEFAULT_BEHAVIOR)
#define FFTBLOCK (FFTBLOCK_DEFAULT)
#define FFTBLOCKPAD (FFTBLOCKPAD_DEFAULT)
#define	SEED (314159265.0)
#define	A (1220703125.0)
#define	PI (3.141592653589793238)
#define	ALPHA (1.0e-6)
#define AP (-4.0*ALPHA*PI*PI)
#define OMP_THREADS (3)
#define TASK_INDEXMAP (0)
#define TASK_INITIAL_CONDITIONS (1)
#define TASK_INIT_UI (2)
#define PROFILING_TOTAL_TIME (0)
#define PROFILING_INDEXMAP (1)
#define PROFILING_INITIAL_CONDITIONS (2)
#define PROFILING_INIT_UI (3)
#define PROFILING_EVOLVE (4)
#define PROFILING_FFTX_1 (5)
#define PROFILING_FFTX_2 (6)
#define PROFILING_FFTX_3 (7)
#define PROFILING_FFTY_1 (8)
#define PROFILING_FFTY_2 (9)
#define PROFILING_FFTY_3 (10)
#define PROFILING_FFTZ_1 (11)
#define PROFILING_FFTZ_2 (12)
#define PROFILING_FFTZ_3 (13)
#define PROFILING_CHECKSUM (14)
#define PROFILING_INIT (15)
#define CHECKSUM_TASKS (1024)

/* global variables */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static dcomplex sums[NITER_DEFAULT+1];
static double twiddle[NTOTAL];
static dcomplex u[MAXDIM];
static dcomplex u0[NTOTAL];
static dcomplex u1[NTOTAL];
static int dims[3];
#else
static dcomplex (*sums)=(dcomplex*)malloc(sizeof(dcomplex)*(NITER_DEFAULT+1));
static double (*twiddle)=(double*)malloc(sizeof(double)*(NTOTAL));
static dcomplex (*u)=(dcomplex*)malloc(sizeof(dcomplex)*(MAXDIM));
static dcomplex (*u0)=(dcomplex*)malloc(sizeof(dcomplex)*(NTOTAL));
static dcomplex (*u1)=(dcomplex*)malloc(sizeof(dcomplex)*(NTOTAL));
static int (*dims)=(int*)malloc(sizeof(int)*(3));
#endif
static int niter;
/* gpu variables */
double* starts_device;
double* twiddle_device;
dcomplex* sums_device;
dcomplex* u_device;
dcomplex* u0_device;
dcomplex* u1_device;
dcomplex* u2_device;
dcomplex* y0_device;
dcomplex* y1_device;
size_t size_sums_device;
size_t size_starts_device;
size_t size_twiddle_device;
size_t size_u_device;
size_t size_u0_device;
size_t size_u1_device;
size_t size_y0_device;
size_t size_y1_device;
size_t size_shared_data;
int blocks_per_grid_on_compute_indexmap;
int blocks_per_grid_on_compute_initial_conditions;
int blocks_per_grid_on_init_ui;
int blocks_per_grid_on_evolve;
int blocks_per_grid_on_fftx_1;
int blocks_per_grid_on_fftx_2;
int blocks_per_grid_on_fftx_3;
int blocks_per_grid_on_ffty_1;
int blocks_per_grid_on_ffty_2;
int blocks_per_grid_on_ffty_3;
int blocks_per_grid_on_fftz_1;
int blocks_per_grid_on_fftz_2;
int blocks_per_grid_on_fftz_3;
int blocks_per_grid_on_checksum;
int threads_per_block_on_compute_indexmap;
int threads_per_block_on_compute_initial_conditions;
int threads_per_block_on_init_ui;
int threads_per_block_on_evolve;
int threads_per_block_on_fftx_1;
int threads_per_block_on_fftx_2;
int threads_per_block_on_fftx_3;
int threads_per_block_on_ffty_1;
int threads_per_block_on_ffty_2;
int threads_per_block_on_ffty_3;
int threads_per_block_on_fftz_1;
int threads_per_block_on_fftz_2;
int threads_per_block_on_fftz_3;
int threads_per_block_on_checksum;
int gpu_device_id;
int total_devices;
hipDeviceProp_t gpu_device_properties;
extern __shared__ double extern_share_data[];

/* function declarations */
static void cffts1_gpu(const int is, 
		dcomplex u[], 
		dcomplex x_in[], 
		dcomplex x_out[], 
		dcomplex y0[], 
		dcomplex y1[]);
__global__ void cffts1_gpu_kernel_1(dcomplex x_in[], 
		dcomplex y0[]);
__global__ void cffts1_gpu_kernel_2(const int is, 
		dcomplex y0[], 
		dcomplex y1[], 
		dcomplex u_device[]);
__global__ void cffts1_gpu_kernel_3(dcomplex x_out[], 
		dcomplex y0[]);
static void cffts2_gpu(int is, 
		dcomplex u[], 
		dcomplex x_in[], 
		dcomplex x_out[], 
		dcomplex y0[], 
		dcomplex y1[]);
__global__ void cffts2_gpu_kernel_1(dcomplex x_in[], 
		dcomplex y0[]);
__global__ void cffts2_gpu_kernel_2(const int is, 
		dcomplex y0[], 
		dcomplex y1[], 
		dcomplex u_device[]);
__global__ void cffts2_gpu_kernel_3(dcomplex x_out[], 
		dcomplex y0[]);
static void cffts3_gpu(int is, 
		dcomplex u[], 
		dcomplex x_in[], 
		dcomplex x_out[], 
		dcomplex y0[], 
		dcomplex y1[]);
__device__ void cffts3_gpu_cfftz_device(const int is, 
		int m, 
		int n, 
		dcomplex x[], 
		dcomplex y[], 
		dcomplex u_device[], 
		int index_arg, 
		int size_arg);
__device__ void cffts3_gpu_fftz2_device(const int is, 
		int l, 
		int m, 
		int n, 
		dcomplex u[], 
		dcomplex x[], 
		dcomplex y[], 
		int index_arg, 
		int size_arg);
__global__ void cffts3_gpu_kernel_1(dcomplex x_in[], 
		dcomplex y0[]);
__global__ void cffts3_gpu_kernel_2(const int is, 
		dcomplex y0[], 
		dcomplex y1[], 
		dcomplex u_device[]);
__global__ void cffts3_gpu_kernel_3(dcomplex x_out[], 
		dcomplex y0[]);
static void checksum_gpu(int iteration, 
		dcomplex u1[]);				
__global__ void checksum_gpu_kernel(int iteration, 
		dcomplex u1[], 
		dcomplex sums[]);
static void compute_indexmap_gpu(double twiddle[]);
__global__ void compute_indexmap_gpu_kernel(double twiddle[]);
static void compute_initial_conditions_gpu(dcomplex u0[]);
__global__ void compute_initial_conditions_gpu_kernel(dcomplex u0[], 
		double starts[]);
static void evolve_gpu(dcomplex u0[], 
		dcomplex u1[], 
		double twiddle[]);
__global__ void evolve_gpu_kernel(dcomplex u0[], 
		dcomplex u1[], 
		double twiddle[]);
static void fft_gpu(int dir,
		dcomplex x1[],
		dcomplex x2[]);
static void fft_init_gpu(int n);
static int ilog2(int n);
__device__ int ilog2_device(int n);
static void init_ui_gpu(dcomplex u0[],
		dcomplex u1[],
		double twiddle[]);
__global__ void init_ui_gpu_kernel(dcomplex u0[],
		dcomplex u1[],
		double twiddle[]);
static void ipow46(double a, 
		int exponent, 
		double* result);
__device__ void ipow46_device(double a, 
		int exponent, 
		double* result);
__device__ double randlc_device(double* x, 
		double a);
static void release_gpu();
static void setup();
static void setup_gpu();
static void verify (int d1, 
		int d2, 
		int d3, 
		int nt, 
		boolean* verified, 
		char* class_npb);
__device__ void vranlc_device(int n, 
		double* x_seed, 
		double a, 
		double y[]);

/* ft */
int main(int argc, char** argv){
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
	printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif
#if defined(PROFILING)
	printf(" PROFILING mode on\n");
#endif
	int iter=0;
	double total_time, mflops;
	boolean verified;
	char class_npb;	

	/*
	 * ---------------------------------------------------------------------
	 * run the entire problem once to make sure all data is touched. 
	 * this reduces variable startup costs, which is important for such a 
	 * short benchmark. the other NPB 2 implementations are similar. 
	 * ---------------------------------------------------------------------
	 */	
	setup();
	setup_gpu();
	init_ui_gpu(u0_device, u1_device, twiddle_device);
#pragma omp parallel
	{
		if(omp_get_thread_num()==TASK_INDEXMAP){
			compute_indexmap_gpu(twiddle_device);
		}else if(omp_get_thread_num()==TASK_INITIAL_CONDITIONS){
			compute_initial_conditions_gpu(u1_device);
		}else if(omp_get_thread_num()==TASK_INIT_UI){
			fft_init_gpu(MAXDIM);
		}		
	}hipDeviceSynchronize();
	fft_gpu(1, u1_device, u0_device);

	/*
	 * ---------------------------------------------------------------------
	 * start over from the beginning. note that all operations must
	 * be timed, in contrast to other benchmarks. 
	 * ---------------------------------------------------------------------
	 */
	timer_clear(PROFILING_TOTAL_TIME);
#if defined(PROFILING)
	timer_clear(PROFILING_INDEXMAP);
	timer_clear(PROFILING_INITIAL_CONDITIONS);
	timer_clear(PROFILING_INITIAL_CONDITIONS);
	timer_clear(PROFILING_EVOLVE);
	timer_clear(PROFILING_FFTX_1);
	timer_clear(PROFILING_FFTX_2);
	timer_clear(PROFILING_FFTX_3);
	timer_clear(PROFILING_FFTY_1);
	timer_clear(PROFILING_FFTY_2);
	timer_clear(PROFILING_FFTY_3);
	timer_clear(PROFILING_FFTZ_1);
	timer_clear(PROFILING_FFTZ_2);
	timer_clear(PROFILING_FFTZ_3);
	timer_clear(PROFILING_CHECKSUM);
#endif

	timer_start(PROFILING_TOTAL_TIME);
#pragma omp parallel
	{
		if(omp_get_thread_num()==TASK_INDEXMAP){
			compute_indexmap_gpu(twiddle_device);
		}else if(omp_get_thread_num()==TASK_INITIAL_CONDITIONS){
			compute_initial_conditions_gpu(u1_device);
		}else if(omp_get_thread_num()==TASK_INIT_UI){
			fft_init_gpu(MAXDIM);
		}		
	}hipDeviceSynchronize();
	fft_gpu(1, u1_device, u0_device);
	for(iter=1; iter<=niter; iter++){
		evolve_gpu(u0_device, u1_device, twiddle_device);
		fft_gpu(-1, u1_device, u1_device);
		checksum_gpu(iter, u1_device);
	}

	hipMemcpy(sums, sums_device, size_sums_device, hipMemcpyDeviceToHost);
	for(iter=1; iter<=niter; iter++){
		printf("T = %5d     Checksum = %22.12e %22.12e\n", iter, sums[iter].real, sums[iter].imag);
	}		

	verify(NX, NY, NZ, niter, &verified, &class_npb);

	timer_stop(PROFILING_TOTAL_TIME);
	total_time = timer_read(PROFILING_TOTAL_TIME);		

	if(total_time != 0.0){
		mflops = 1.0e-6 * ((double)(NTOTAL)) *
			(14.8157 + 7.19641 * log((double)(NTOTAL))
			 + (5.23518 + 7.21113 * log((double)(NTOTAL)))*niter)
			/ total_time;
	}else{
		mflops = 0.0;
	}

	char gpu_config[256];
	char gpu_config_string[2048];
#if defined(PROFILING)
	sprintf(gpu_config, "%5s\t%25s\t%25s\t%25s\n", "GPU Kernel", "Threads Per Block", "Time in Seconds", "Time in Percentage");
	strcpy(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " indexmap", threads_per_block_on_compute_indexmap, timer_read(PROFILING_INDEXMAP), (timer_read(PROFILING_INDEXMAP)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " initial conditions", threads_per_block_on_compute_initial_conditions, timer_read(PROFILING_INITIAL_CONDITIONS), (timer_read(PROFILING_INITIAL_CONDITIONS)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " init ui", threads_per_block_on_init_ui, timer_read(PROFILING_INIT_UI), (timer_read(PROFILING_INIT_UI)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " evolve", threads_per_block_on_evolve, timer_read(PROFILING_EVOLVE), (timer_read(PROFILING_EVOLVE)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " fftx 1", threads_per_block_on_fftx_1, timer_read(PROFILING_FFTX_1), (timer_read(PROFILING_FFTX_1)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " fftx 2", threads_per_block_on_fftx_2, timer_read(PROFILING_FFTX_2), (timer_read(PROFILING_FFTX_2)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " fftx 3", threads_per_block_on_fftx_3, timer_read(PROFILING_FFTX_3), (timer_read(PROFILING_FFTX_3)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " ffty 1", threads_per_block_on_ffty_1, timer_read(PROFILING_FFTY_1), (timer_read(PROFILING_FFTY_1)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " ffty 2", threads_per_block_on_ffty_2, timer_read(PROFILING_FFTY_2), (timer_read(PROFILING_FFTY_2)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " ffty 3", threads_per_block_on_ffty_3, timer_read(PROFILING_FFTY_3), (timer_read(PROFILING_FFTY_3)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " fftz 1", threads_per_block_on_fftz_1, timer_read(PROFILING_FFTZ_1), (timer_read(PROFILING_FFTZ_1)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " fftz 2", threads_per_block_on_fftz_2, timer_read(PROFILING_FFTZ_2), (timer_read(PROFILING_FFTZ_2)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " fftz 3", threads_per_block_on_fftz_3, timer_read(PROFILING_FFTZ_3), (timer_read(PROFILING_FFTZ_3)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " checksum", threads_per_block_on_checksum, timer_read(PROFILING_CHECKSUM), (timer_read(PROFILING_CHECKSUM)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
#else
	sprintf(gpu_config, "%5s\t%25s\n", "GPU Kernel", "Threads Per Block");
	strcpy(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " indexmap", threads_per_block_on_compute_indexmap);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " initial conditions", threads_per_block_on_compute_initial_conditions);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " init ui", threads_per_block_on_init_ui);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " evolve", threads_per_block_on_evolve);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " fftx 1", threads_per_block_on_fftx_1);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " fftx 2", threads_per_block_on_fftx_2);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " fftx 3", threads_per_block_on_fftx_3);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " ffty 1", threads_per_block_on_ffty_1);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " ffty 2", threads_per_block_on_ffty_2);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " ffty 3", threads_per_block_on_ffty_3);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " fftz 1", threads_per_block_on_fftz_1);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " fftz 2", threads_per_block_on_fftz_2);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " fftz 3", threads_per_block_on_fftz_3);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " checksum", threads_per_block_on_checksum);
	strcat(gpu_config_string, gpu_config);
#endif

	c_print_results((char*)"FT", 
			class_npb, 
			NX, 
			NY, 
			NZ, 
			niter, 
			total_time, 
			mflops, 
			(char*)"          floating point", 
			verified, 
			(char*)NPBVERSION, 
			(char*)COMPILETIME,
			(char*)COMPILERVERSION,
			(char*)LIBVERSION,
			(char*)CPU_MODEL,
			(char*)gpu_device_properties.name,
			(char*)gpu_config_string,
			(char*)CS1, 
			(char*)CS2, 
			(char*)CS3, 
			(char*)CS4, 
			(char*)CS5, 
			(char*)CS6, 
			(char*)CS7);	

	release_gpu();

	return 0;
}

static void cffts1_gpu(const int is, 
		dcomplex u[], 
		dcomplex x_in[], 
		dcomplex x_out[], 
		dcomplex y0[], 
		dcomplex y1[]){
#if defined(PROFILING)
	timer_start(PROFILING_FFTX_1);
#endif
	cffts1_gpu_kernel_1<<<blocks_per_grid_on_fftx_1,
		threads_per_block_on_fftx_1>>>(x_in, 
				y0);
	hipDeviceSynchronize();
#if defined(PROFILING)
	timer_stop(PROFILING_FFTX_1);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_FFTX_2);
#endif
	cffts1_gpu_kernel_2<<<blocks_per_grid_on_fftx_2,
		threads_per_block_on_fftx_2>>>(is, 
				y0, 
				y1, 
				u);
	hipDeviceSynchronize();
#if defined(PROFILING)
	timer_stop(PROFILING_FFTX_2);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_FFTX_3);
#endif
	cffts1_gpu_kernel_3<<<blocks_per_grid_on_fftx_3,
		threads_per_block_on_fftx_3>>>(x_out, 
				y0);
	hipDeviceSynchronize();
#if defined(PROFILING)
	timer_stop(PROFILING_FFTX_3);
#endif
}

/*
 * ----------------------------------------------------------------------
 * y0[z][x][y] = x_in[z][y][x] 
 *
 * y0[y + x*NY + z*NX*NY] = x_in[x + y*NX + z*NX*NY] 
 * ----------------------------------------------------------------------
 */
__global__ void cffts1_gpu_kernel_1(dcomplex x_in[], 
		dcomplex y0[]){
	int x_y_z = blockIdx.x * blockDim.x + threadIdx.x;
	if(x_y_z >= (NX*NY*NZ)){
		return;
	}
	int x = x_y_z % NX;
	int y = (x_y_z / NX) % NY;
	int z = x_y_z / (NX * NY);
	y0[y+(x*NY)+(z*NX*NY)].real = x_in[x_y_z].real;
	y0[y+(x*NY)+(z*NX*NY)].imag = x_in[x_y_z].imag;
}

/*
 * ----------------------------------------------------------------------
 * pattern = j + variable*NY + k*NX*NY | variable is i and transforms x axis
 * ----------------------------------------------------------------------
 */
__global__ void cffts1_gpu_kernel_2(const int is, 
		dcomplex gty1[], 
		dcomplex gty2[], 
		dcomplex u_device[]){	
	int y_z = blockIdx.x * blockDim.x + threadIdx.x;

	if(y_z >= (NY*NZ)){
		return;
	}

	int j, k;
	int l, j1, i1, k1;
	int n1, li, lj, lk, ku, i11, i12, i21, i22;

	j = y_z % NY; /* j = y */
	k = (y_z / NY) % NZ; /* k = z */

	const int logd1 = ilog2_device(NX);

	double uu1_real, x11_real, x21_real;
	double uu1_imag, x11_imag, x21_imag;
	double uu2_real, x12_real, x22_real;
	double uu2_imag, x12_imag, x22_imag;
	double temp_real, temp2_real;
	double temp_imag, temp2_imag;

	for(l=1; l<=logd1; l+=2){
		n1 = NX / 2;
		lk = 1 << (l - 1);
		li = 1 << (logd1 - l);
		lj = 2 * lk;
		ku = li;
		for(i1=0; i1<=li-1; i1++){		    
			for(k1=0; k1<=lk-1; k1++){
				i11 = i1 * lk;
				i12 = i11 + n1;
				i21 = i1 * lj;
				i22 = i21 + lk;

				uu1_real = u_device[ku+i1].real;
				uu1_imag = is*u_device[ku+i1].imag;

				/* gty1[k][i11+k1][j] */
				x11_real = gty1[j + (i11+k1)*NY + k*NX*NY].real;
				x11_imag = gty1[j + (i11+k1)*NY + k*NX*NY].imag;

				/* gty1[k][i12+k1][j] */
				x21_real = gty1[j + (i12+k1)*NY + k*NX*NY].real;
				x21_imag = gty1[j + (i12+k1)*NY + k*NX*NY].imag;

				/* gty2[k][i21+k1][j] */
				gty2[j + (i21+k1)*NY + k*NX*NY].real = x11_real + x21_real;
				gty2[j + (i21+k1)*NY + k*NX*NY].imag = x11_imag + x21_imag;

				temp_real = x11_real - x21_real;
				temp_imag = x11_imag - x21_imag;

				/* gty2[k][i22+k1][j] */
				gty2[j + (i22+k1)*NY + k*NX*NY].real = (uu1_real)*(temp_real) - (uu1_imag)*(temp_imag);
				gty2[j + (i22+k1)*NY + k*NX*NY].imag = (uu1_real)*(temp_imag) + (uu1_imag)*(temp_real);
			}
		}
		if(l==logd1){
			for(j1=0; j1<NX; j1++){
				/* gty1[k][j1][j] */
				gty1[j + j1*NY + k*NX*NY].real = gty2[j + j1*NY + k*NX*NY].real;
				gty1[j + j1*NY + k*NX*NY].imag = gty2[j + j1*NY + k*NX*NY].imag;
			}
		}else{
			n1 = NX / 2;
			lk = 1 << (l+1 - 1);
			li = 1 << (logd1 - (l+1));
			lj = 2 * lk;
			ku = li;
			for(i1=0; i1<=li-1; i1++){			    
				for(k1=0; k1<=lk-1; k1++){
					i11 = i1 * lk;
					i12 = i11 + n1;
					i21 = i1 * lj;
					i22 = i21 + lk;

					uu2_real = u_device[ku+i1].real;
					uu2_imag = is*u_device[ku+i1].imag;

					/* gty2[k][i11+k1][j] */
					x12_real = gty2[j + (i11+k1)*NY + k*NX*NY].real;
					x12_imag = gty2[j + (i11+k1)*NY + k*NX*NY].imag;

					/* gty2[k][i12+k1][j] */
					x22_real = gty2[j + (i12+k1)*NY + k*NX*NY].real;
					x22_imag = gty2[j + (i12+k1)*NY + k*NX*NY].imag;

					/* gty2[k][i21+k1][j] */
					gty1[j + (i21+k1)*NY + k*NX*NY].real = x12_real + x22_real;
					gty1[j + (i21+k1)*NY + k*NX*NY].imag = x12_imag + x22_imag;

					temp2_real = x12_real - x22_real;
					temp2_imag = x12_imag - x22_imag;

					/* gty1[k][i22+k1][j] */
					gty1[j + (i22+k1)*NY + k*NX*NY].real = (uu2_real)*(temp2_real) - (uu2_imag)*(temp2_imag);
					gty1[j + (i22+k1)*NY + k*NX*NY].imag = (uu2_real)*(temp2_imag) + (uu2_imag)*(temp2_real);
				}
			}
		}
	} 	
}

/*
 * ----------------------------------------------------------------------
 * x_out[z][y][x] = y0[z][x][y] 
 *
 * x_out[x + y*NX + z*NX*NY] = y0[y + x*NY + z*NX*NY]  
 * ----------------------------------------------------------------------
 */
__global__ void cffts1_gpu_kernel_3(dcomplex x_out[], 
		dcomplex y0[]){
	int x_y_z = blockIdx.x * blockDim.x + threadIdx.x;
	if(x_y_z >= (NX*NY*NZ)){
		return;
	}
	int x = x_y_z % NX;
	int y = (x_y_z / NX) % NY;
	int z = x_y_z / (NX * NY);
	x_out[x_y_z].real = y0[y+(x*NY)+(z*NX*NY)].real;
	x_out[x_y_z].imag = y0[y+(x*NY)+(z*NX*NY)].imag;
}

static void cffts2_gpu(int is, 
		dcomplex u[], 
		dcomplex x_in[], 
		dcomplex x_out[], 
		dcomplex y0[], 
		dcomplex y1[]){
#if defined(PROFILING)
	timer_start(PROFILING_FFTY_1);
#endif
	cffts2_gpu_kernel_1<<<blocks_per_grid_on_ffty_1,
		threads_per_block_on_ffty_1>>>(x_in, 
				y0);
	hipDeviceSynchronize();
#if defined(PROFILING)
	timer_stop(PROFILING_FFTY_1);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_FFTY_2);
#endif
	cffts2_gpu_kernel_2<<<blocks_per_grid_on_ffty_2,
		threads_per_block_on_ffty_2>>>(is, 
				y0, 
				y1, 
				u);
	hipDeviceSynchronize();
#if defined(PROFILING)
	timer_stop(PROFILING_FFTY_2);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_FFTY_3);
#endif
	cffts2_gpu_kernel_3<<<blocks_per_grid_on_ffty_3,
		threads_per_block_on_ffty_3>>>(x_out, 
				y0);
	hipDeviceSynchronize();
#if defined(PROFILING)
	timer_stop(PROFILING_FFTY_3);
#endif
}

/*
 * ----------------------------------------------------------------------
 * y0[z][y][x] = x_in[z][y][x] 
 *
 * y0[x + y*NX + z*NX*NY]  = x_in[x + y*NX + z*NX*NY] 
 * ----------------------------------------------------------------------
 */
__global__ void cffts2_gpu_kernel_1(dcomplex x_in[], 
		dcomplex y0[]){
	int x_y_z = blockIdx.x * blockDim.x + threadIdx.x;
	if(x_y_z >= (NX*NY*NZ)){
		return;
	}
	y0[x_y_z].real = x_in[x_y_z].real;
	y0[x_y_z].imag = x_in[x_y_z].imag;
}

/*
 * ----------------------------------------------------------------------
 * pattern = i + variable*NX + k*NX*NY | variable is j and transforms y axis
 * ----------------------------------------------------------------------
 */
__global__ void cffts2_gpu_kernel_2(const int is, 
		dcomplex gty1[], 
		dcomplex gty2[], 
		dcomplex u_device[]){
	int x_z = blockIdx.x * blockDim.x + threadIdx.x;

	if(x_z >= (NX*NZ)){
		return;
	}

	int i, k;
	int l, j1, i1, k1;
	int n1, li, lj, lk, ku, i11, i12, i21, i22;

	i = x_z % NX; /* i = x */
	k = (x_z / NX) % NZ; /* k = z */

	const int logd2 = ilog2_device(NY);

	double uu1_real, x11_real, x21_real;
	double uu1_imag, x11_imag, x21_imag;
	double uu2_real, x12_real, x22_real;
	double uu2_imag, x12_imag, x22_imag;
	double temp_real, temp2_real;
	double temp_imag, temp2_imag;	

	for(l=1; l<=logd2; l+=2){
		n1 = NY / 2;
		lk = 1 << (l - 1);
		li = 1 << (logd2 - l);
		lj = 2 * lk;
		ku = li;
		for(i1=0; i1<=li-1; i1++){
			for(k1=0; k1<=lk-1; k1++){
				i11 = i1 * lk;
				i12 = i11 + n1;
				i21 = i1 * lj;
				i22 = i21 + lk;

				uu1_real = u_device[ku+i1].real;
				uu1_imag = is*u_device[ku+i1].imag;

				/* gty1[k][i11+k1][i] */
				x11_real = gty1[i + (i11+k1)*NX + k*NX*NY].real;
				x11_imag = gty1[i + (i11+k1)*NX + k*NX*NY].imag;

				/* gty1[k][i12+k1][i] */
				x21_real = gty1[i + (i12+k1)*NX + k*NX*NY].real;
				x21_imag = gty1[i + (i12+k1)*NX + k*NX*NY].imag;

				/* gty2[k][i21+k1][i] */
				gty2[i + (i21+k1)*NX + k*NX*NY].real = x11_real + x21_real;
				gty2[i + (i21+k1)*NX + k*NX*NY].imag = x11_imag + x21_imag;

				temp_real = x11_real - x21_real;
				temp_imag = x11_imag - x21_imag;

				/* gty2[k][i22+k1][i] */
				gty2[i + (i22+k1)*NX + k*NX*NY].real = (uu1_real)*(temp_real) - (uu1_imag)*(temp_imag);
				gty2[i + (i22+k1)*NX + k*NX*NY].imag = (uu1_real)*(temp_imag) + (uu1_imag)*(temp_real);

			}
		}
		if(l==logd2){
			for(j1=0; j1<NY; j1++){
				/* gty1[k][j1][i] */
				gty1[i + j1*NX + k*NX*NY].real = gty2[i + j1*NX + k*NX*NY].real;
				gty1[i + j1*NX + k*NX*NY].imag = gty2[i + j1*NX + k*NX*NY].imag;
			}
		}
		else{
			n1 = NY / 2;
			lk = 1 << (l+1 - 1);
			li = 1 << (logd2 - (l+1));
			lj = 2 * lk;
			ku = li;
			for(i1=0; i1<=li-1; i1++){
				for(k1=0; k1<=lk-1; k1++){
					i11 = i1 * lk;
					i12 = i11 + n1;
					i21 = i1 * lj;
					i22 = i21 + lk;

					uu2_real = u_device[ku+i1].real;
					uu2_imag = is*u_device[ku+i1].imag;

					/* gty2[k][i11+k1][i] */
					x12_real = gty2[i + (i11+k1)*NX + k*NX*NY].real;
					x12_imag = gty2[i + (i11+k1)*NX + k*NX*NY].imag;

					/* gty2[k][i12+k1][i] */
					x22_real = gty2[i + (i12+k1)*NX + k*NX*NY].real;
					x22_imag = gty2[i + (i12+k1)*NX + k*NX*NY].imag;

					/* gty1[k][i21+k1][i] */
					gty1[i + (i21+k1)*NX + k*NX*NY].real = x12_real + x22_real;
					gty1[i + (i21+k1)*NX + k*NX*NY].imag = x12_imag + x22_imag;

					temp2_real = x12_real - x22_real;
					temp2_imag = x12_imag - x22_imag;

					/* gty1[k][i22+k1][i] */
					gty1[i + (i22+k1)*NX + k*NX*NY].real = (uu2_real)*(temp2_real) - (uu2_imag)*(temp2_imag);
					gty1[i + (i22+k1)*NX + k*NX*NY].imag = (uu2_real)*(temp2_imag) + (uu2_imag)*(temp2_real);
				}
			}
		}
	}
}

/*
 * ----------------------------------------------------------------------
 * x_out[z][y][x] = y0[z][y][x]
 *
 * x_out[x + y*NX + z*NX*NY] = y0[x + y*NX + z*NX*NY] 
 * ----------------------------------------------------------------------
 */
__global__ void cffts2_gpu_kernel_3(dcomplex x_out[], 
		dcomplex y0[]){
	int x_y_z = blockIdx.x * blockDim.x + threadIdx.x;
	if(x_y_z >= (NX*NY*NZ)){
		return;
	}
	x_out[x_y_z].real = y0[x_y_z].real;
	x_out[x_y_z].imag = y0[x_y_z].imag;
}

static void cffts3_gpu(int is, 
		dcomplex u[], 
		dcomplex x_in[], 
		dcomplex x_out[], 
		dcomplex y0[], 
		dcomplex y1[]){
#if defined(PROFILING)
	timer_start(PROFILING_FFTZ_1);
#endif
	cffts3_gpu_kernel_1<<<blocks_per_grid_on_fftz_1,
		threads_per_block_on_fftz_1>>>(x_in, 
				y0);
	hipDeviceSynchronize();
#if defined(PROFILING)
	timer_stop(PROFILING_FFTZ_1);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_FFTZ_2);
#endif
	cffts3_gpu_kernel_2<<<blocks_per_grid_on_fftz_2,
		threads_per_block_on_fftz_2>>>(is, 
				y0, 
				y1, 
				u);
	hipDeviceSynchronize();
#if defined(PROFILING)
	timer_stop(PROFILING_FFTZ_2);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_FFTZ_3);
#endif
	cffts3_gpu_kernel_3<<<blocks_per_grid_on_fftz_3,
		threads_per_block_on_fftz_3>>>(x_out, 
				y0);
	hipDeviceSynchronize();
#if defined(PROFILING)
	timer_stop(PROFILING_FFTZ_3);
#endif	
}

/*
 * ----------------------------------------------------------------------
 * pattern = i + j*NX + variable*NX*NY | variable is z and transforms z axis
 *
 * index_arg = i + j*NX
 *
 * size_arg = NX*NY
 * ----------------------------------------------------------------------
 */
__device__ void cffts3_gpu_cfftz_device(const int is, 
		int m, 
		int n, 
		dcomplex x[], 
		dcomplex y[], 
		dcomplex u_device[], 
		int index_arg, 
		int size_arg){
	int j,l;
	/*
	 * ---------------------------------------------------------------------
	 * perform one variant of the Stockham FFT.
	 * ---------------------------------------------------------------------
	 */
	for(l=1; l<=m; l+=2){
		cffts3_gpu_fftz2_device(is, l, m, n, u_device, x, y, index_arg, size_arg);
		if(l==m){break;}
		cffts3_gpu_fftz2_device(is, l + 1, m, n, u_device, y, x, index_arg, size_arg);
	}
	/*
	 * ---------------------------------------------------------------------
	 * copy Y to X.
	 * ---------------------------------------------------------------------
	 */
	if(m%2==1){
		for(j=0; j<n; j++){
			x[j*size_arg+index_arg].real = y[j*size_arg+index_arg].real;
			x[j*size_arg+index_arg].imag = y[j*size_arg+index_arg].imag;
		}
	}
}

/*
 * ----------------------------------------------------------------------
 * pattern = i + j*NX + variable*NX*NY | variable is z and transforms z axis
 *
 * index_arg = i + j*NX
 *
 * size_arg = NX*NY
 * ----------------------------------------------------------------------
 */
__device__ void cffts3_gpu_fftz2_device(const int is, 
		int l, 
		int m, 
		int n, 
		dcomplex u[], 
		dcomplex x[], 
		dcomplex y[], 
		int index_arg, 
		int size_arg){
	int k,n1,li,lj,lk,ku,i,i11,i12,i21,i22;
	double x11real, x11imag;
	double x21real, x21imag;
	dcomplex u1;
	/*
	 * ---------------------------------------------------------------------
	 * set initial parameters.
	 * ---------------------------------------------------------------------
	 */
	n1 = n / 2;
	lk = 1 << (l - 1);
	li = 1 << (m - l);
	lj = 2 * lk;
	ku = li;
	for(i=0; i<li; i++){
		i11 = i * lk;
		i12 = i11 + n1;
		i21 = i * lj;
		i22 = i21 + lk;
		if(is>=1){
			u1.real = u[ku+i].real;
			u1.imag = u[ku+i].imag;
		}else{
			u1.real = u[ku+i].real;
			u1.imag = -u[ku+i].imag;
		}
		for(k=0; k<lk; k++){
			x11real = x[(i11+k)*size_arg+index_arg].real;
			x11imag = x[(i11+k)*size_arg+index_arg].imag;
			x21real = x[(i12+k)*size_arg+index_arg].real;
			x21imag = x[(i12+k)*size_arg+index_arg].imag;
			y[(i21+k)*size_arg+index_arg].real = x11real + x21real;
			y[(i21+k)*size_arg+index_arg].imag = x11imag + x21imag;
			y[(i22+k)*size_arg+index_arg].real = u1.real * (x11real - x21real) - u1.imag * (x11imag - x21imag);
			y[(i22+k)*size_arg+index_arg].imag = u1.real * (x11imag - x21imag) + u1.imag * (x11real - x21real);
		}
	}
}

/*
 * ----------------------------------------------------------------------
 * y0[z][y][x] = x_in[z][y][x] 
 *
 * y0[x + y*NX + z*NX*NY]  = x_in[x + y*NX + z*NX*NY] 
 * ----------------------------------------------------------------------
 */
__global__ void cffts3_gpu_kernel_1(dcomplex x_in[], 
		dcomplex y0[]){
	int x_y_z = blockIdx.x * blockDim.x + threadIdx.x;
	if(x_y_z >= (NX*NY*NZ)){
		return;
	}
	y0[x_y_z].real = x_in[x_y_z].real;
	y0[x_y_z].imag = x_in[x_y_z].imag;
}

/*
 * ----------------------------------------------------------------------
 * pattern = i + j*NX + variable*NX*NY | variable is z and transforms z axis
 * ----------------------------------------------------------------------
 */
__global__ void cffts3_gpu_kernel_2(const int is, 
		dcomplex gty1[], 
		dcomplex gty2[], 
		dcomplex u_device[]){
	int x_y = blockIdx.x * blockDim.x + threadIdx.x;
	if(x_y >= (NX*NY)){
		return;
	}
	cffts3_gpu_cfftz_device(is, 
			ilog2_device(NZ), 
			NZ, 
			gty1 , 
			gty2, 
			u_device, 
			x_y /* index_arg */, 
			NX*NY /* size_arg */);
}

/*
 * ----------------------------------------------------------------------
 * x_out[z][y][x] = y0[z][y][x]
 *
 * x_out[x + y*NX + z*NX*NY] = y0[x + y*NX + z*NX*NY] 
 * ----------------------------------------------------------------------
 */
__global__ void cffts3_gpu_kernel_3(dcomplex x_out[], 
		dcomplex y0[]){
	int x_y_z = blockIdx.x * blockDim.x + threadIdx.x;
	if(x_y_z >= (NX*NY*NZ)){
		return;
	}
	x_out[x_y_z].real = y0[x_y_z].real;
	x_out[x_y_z].imag = y0[x_y_z].imag;
}

static void checksum_gpu(int iteration,
		dcomplex u1[]){
#if defined(PROFILING)
	timer_start(PROFILING_CHECKSUM);
#endif
	checksum_gpu_kernel<<<blocks_per_grid_on_checksum,
		threads_per_block_on_checksum,
		size_shared_data>>>(iteration, 
				u1, 
				sums_device);
#if defined(PROFILING)
	timer_stop(PROFILING_CHECKSUM);
#endif
}

__global__ void checksum_gpu_kernel(int iteration, 
		dcomplex u1[], 
		dcomplex sums[]){
	dcomplex* share_sums = (dcomplex*)(extern_share_data);
	int j = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
	int q, r, s;

	if(j<=CHECKSUM_TASKS){
		q = j % NX;
		r = 3*j % NY;
		s = 5*j % NZ;
		share_sums[threadIdx.x] = u1[ q + r*NX + s*NX*NY ];
	}else{
		share_sums[threadIdx.x] = dcomplex_create(0.0, 0.0);
	}

	__syncthreads();
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(threadIdx.x<i){
			share_sums[threadIdx.x] = dcomplex_add(share_sums[threadIdx.x], share_sums[threadIdx.x+i]);
		}
		__syncthreads();
	}
	if(threadIdx.x==0){
		share_sums[0].real = share_sums[0].real/(double)(NTOTAL);
		atomicAdd(&sums[iteration].real,share_sums[0].real);
		share_sums[0].imag = share_sums[0].imag/(double)(NTOTAL);
		atomicAdd(&sums[iteration].imag,share_sums[0].imag);
	}
}

static void compute_indexmap_gpu(double twiddle[]){
#if defined(PROFILING)
	timer_start(PROFILING_INDEXMAP);
#endif
	compute_indexmap_gpu_kernel<<<blocks_per_grid_on_compute_indexmap,
		threads_per_block_on_compute_indexmap>>>(twiddle);
#if defined(PROFILING)
	timer_stop(PROFILING_INDEXMAP);
#endif
}

__global__ void compute_indexmap_gpu_kernel(double twiddle[]){
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	if(thread_id>=NTOTAL){
		return;
	}

	int i = thread_id % NX;
	int j = (thread_id / NX) % NY;
	int k = thread_id / (NX * NY);

	int kk, kk2, jj, kj2, ii;

	kk = ((k+NZ/2) % NZ) - NZ/2;
	kk2 = kk*kk;
	jj = ((j+NY/2) % NY) - NY/2;
	kj2 = jj*jj+kk2;
	ii = ((i+NX/2) % NX) - NX/2;

	twiddle[thread_id] = exp(AP*(double)(ii*ii+kj2));
}

static void compute_initial_conditions_gpu(dcomplex u0[]){  
#if defined(PROFILING)
	timer_start(PROFILING_INITIAL_CONDITIONS);
#endif  
	int z;
	double start, an, starts[NZ];

	start = SEED;

	ipow46(A, 0, &an);
	randlc(&start, an);
	ipow46(A, 2*NX*NY, &an);

	starts[0] = start;
	for(z=1; z<NZ; z++){
		randlc(&start, an);
		starts[z] = start;
	}

	hipMemcpy(starts_device, starts, size_starts_device, hipMemcpyHostToDevice);

	compute_initial_conditions_gpu_kernel<<<blocks_per_grid_on_compute_initial_conditions,
		threads_per_block_on_compute_initial_conditions>>>(u0, 
				starts_device);
#if defined(PROFILING)
	timer_stop(PROFILING_INITIAL_CONDITIONS);
#endif  
}

__global__ void compute_initial_conditions_gpu_kernel(dcomplex u0[], 
		double starts[]){    
	int z = blockIdx.x * blockDim.x + threadIdx.x;

	if(z>=NZ){return;}

	double x0 = starts[z];	
	for(int y=0; y<NY; y++){
		vranlc_device(2*NX, &x0, A, (double*)&u0[ 0 + y*NX + z*NX*NY ]);
	}
}

static void evolve_gpu(dcomplex u0[], 
		dcomplex u1[],
		double twiddle[]){
#if defined(PROFILING)
	timer_start(PROFILING_EVOLVE);
#endif  
	evolve_gpu_kernel<<<blocks_per_grid_on_evolve,
		threads_per_block_on_evolve>>>(u0, 
				u1,
				twiddle);
	hipDeviceSynchronize();
#if defined(PROFILING)
	timer_stop(PROFILING_EVOLVE);
#endif  
}

__global__ void evolve_gpu_kernel(dcomplex u0[], 
		dcomplex u1[],
		double twiddle[]){
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	if(thread_id>=(NZ*NY*NX)){
		return;
	}	

	u0[thread_id] = dcomplex_mul2(u0[thread_id], twiddle[thread_id]);
	u1[thread_id] = u0[thread_id];
}

static void fft_gpu(int dir,
		dcomplex x1[],
		dcomplex x2[]){
	/*
	 * ---------------------------------------------------------------------
	 * note: args x1, x2 must be different arrays
	 * note: args for cfftsx are (direction, layout, xin, xout, scratch)
	 * xin/xout may be the same and it can be somewhat faster
	 * if they are
	 * ---------------------------------------------------------------------
	 */
	if(dir==1){
		cffts1_gpu(1, u_device, x1, x1, y0_device, y1_device);
		cffts2_gpu(1, u_device, x1, x1, y0_device, y1_device);
		cffts3_gpu(1, u_device, x1, x2, y0_device, y1_device);
	}else{
		cffts3_gpu(-1, u_device, x1, x1, y0_device, y1_device);
		cffts2_gpu(-1, u_device, x1, x1, y0_device, y1_device);
		cffts1_gpu(-1, u_device, x1, x2, y0_device, y1_device);
	}
}

static void fft_init_gpu(int n){
#if defined(PROFILING)
	timer_start(PROFILING_INIT);
#endif  
	int m,ku,i,j,ln;
	double t, ti;
	/*
	 * ---------------------------------------------------------------------
	 * initialize the U array with sines and cosines in a manner that permits
	 * stride one access at each FFT iteration.
	 * ---------------------------------------------------------------------	
	 */
	m = ilog2(n);
	u[0] = dcomplex_create((double)m, 0.0);
	ku = 2;
	ln = 1;
	for(j=1; j<=m; j++){
		t = PI / ln;
		for(i=0; i<=ln-1; i++){
			ti = i * t;
			u[i+ku-1] = dcomplex_create(cos(ti), sin(ti));
		}
		ku = ku + ln;
		ln = 2 * ln;
	}
	hipMemcpy(u_device, u, size_u_device, hipMemcpyHostToDevice);
#if defined(PROFILING)
	timer_stop(PROFILING_INIT);
#endif 
}

static int ilog2(int n){
	int nn, lg;
	if(n==1){
		return 0;
	}
	lg = 1;
	nn = 2;
	while(nn<n){
		nn = nn << 1;
		lg++;
	}
	return lg;
}

__device__ int ilog2_device(int n){
	int nn, lg;
	if(n==1){
		return 0;
	}
	lg = 1;
	nn = 2;
	while(nn<n){
		nn = nn << 1;
		lg++;
	}
	return lg;
}

static void init_ui_gpu(dcomplex u0[],
		dcomplex u1[],
		double twiddle[]){
#if defined(PROFILING)
	timer_start(PROFILING_INIT_UI);
#endif  
	init_ui_gpu_kernel<<<blocks_per_grid_on_init_ui,
		threads_per_block_on_init_ui>>>(u0, 
				u1,
				twiddle);
	hipDeviceSynchronize();
#if defined(PROFILING)
	timer_stop(PROFILING_INIT_UI);
#endif  
}

__global__ void init_ui_gpu_kernel(dcomplex u0[],
		dcomplex u1[],
		double twiddle[]){
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	if(thread_id>=NTOTAL){
		return;
	}	

	u0[thread_id] = dcomplex_create(0.0, 0.0);
	u1[thread_id] = dcomplex_create(0.0, 0.0);
	twiddle[thread_id] = 0.0;
}

static void ipow46(double a, 
		int exponent, 
		double* result){
	double q, r;
	int n, n2;
	/*
	 * --------------------------------------------------------------------
	 * use
	 * a^n = a^(n/2)*a^(n/2) if n even else
	 * a^n = a*a^(n-1)       if n odd
	 * -------------------------------------------------------------------
	 */
	*result = 1;
	if(exponent==0){return;}
	q = a;
	r = 1;
	n = exponent;
	while(n>1){
		n2 = n/2;
		if(n2*2==n){
			randlc(&q, q);
			n = n2;
		}else{
			randlc(&r, q);
			n = n-1;
		}
	}
	randlc(&r, q);
	*result = r;
}

__device__ void ipow46_device(double a, 
		int exponent, 
		double* result){
	double q, r;
	int n, n2;
	/*
	 * --------------------------------------------------------------------
	 * use
	 * a^n = a^(n/2)*a^(n/2) if n even else
	 * a^n = a*a^(n-1)       if n odd
	 * -------------------------------------------------------------------
	 */
	*result = 1;
	if(exponent==0){return;}
	q = a;
	r = 1;
	n = exponent;
	while(n>1){
		n2 = n/2;
		if(n2*2==n){
			randlc_device(&q, q);
			n = n2;
		}else{
			randlc_device(&r, q);
			n = n-1;
		}
	}
	randlc_device(&r, q);
	*result = r;
}

__device__ double randlc_device(double* x, 
		double a){
	double t1,t2,t3,t4,a1,a2,x1,x2,z;
	t1 = R23 * a;
	a1 = (int)t1;
	a2 = a - T23 * a1;
	t1 = R23 * (*x);
	x1 = (int)t1;
	x2 = (*x) - T23 * x1;
	t1 = a1 * x2 + a2 * x1;
	t2 = (int)(R23 * t1);
	z = t1 - T23 * t2;
	t3 = T23 * z + a2 * x2;
	t4 = (int)(R46 * t3);
	(*x) = t3 - T46 * t4;
	return (R46 * (*x));
}

static void release_gpu(){
	hipFree(sums_device);
	hipFree(starts_device);
	hipFree(twiddle_device);
	hipFree(u_device);
	hipFree(u0_device);
	hipFree(u1_device);
	hipFree(y0_device);
	hipFree(y1_device);
}

static void setup(){
	niter = NITER_DEFAULT;

	printf("\n\n NAS Parallel Benchmarks 4.1 HIP C++ version - FT Benchmark\n\n");
	printf(" Size                : %4dx%4dx%4d\n", NX, NY, NZ);
	printf(" Iterations                  :%7d\n", niter);
	printf("\n");
}

static void setup_gpu(){
	/*
	 * struct hipDeviceProp{
	 *  char name[256];
	 *  size_t totalGlobalMem;
	 *  size_t sharedMemPerBlock;
	 *  int regsPerBlock;
	 *  int warpSize;
	 *  size_t memPitch;
	 *  int maxThreadsPerBlock;
	 *  int maxThreadsDim[3];
	 *  int maxGridSize[3];
	 *  size_t totalConstMem;
	 *  int major;
	 *  int minor;
	 *  int clockRate;
	 *  size_t textureAlignment;
	 *  int deviceOverlap;
	 *  int multiProcessorCount;
	 *  int kernelExecTimeoutEnabled;
	 *  int integrated;
	 *  int canMapHostMemory;
	 *  int computeMode;
	 *  int concurrentKernels;
	 *  int ECCEnabled;
	 *  int pciBusID;
	 *  int pciDeviceID;
	 *  int tccDriver;
	 * }
	 */
	/* amount of available devices */ 
	hipGetDeviceCount(&total_devices);

	/* define gpu_device */
	if(total_devices==0){
		printf("\n\n\nNo Nvidia GPU found!\n\n\n");
		exit(-1);
	}else if((GPU_DEVICE>=0)&&
			(GPU_DEVICE<total_devices)){
		gpu_device_id = GPU_DEVICE;
	}else{
		gpu_device_id = 0;
	}
	hipSetDevice(gpu_device_id);	
	hipGetDeviceProperties(&gpu_device_properties, gpu_device_id);

	/* define threads_per_block */
	if((FT_THREADS_PER_BLOCK_ON_COMPUTE_INDEXMAP>=1)&&
			(FT_THREADS_PER_BLOCK_ON_COMPUTE_INDEXMAP<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_compute_indexmap = FT_THREADS_PER_BLOCK_ON_COMPUTE_INDEXMAP;
	}else{
		threads_per_block_on_compute_indexmap = gpu_device_properties.warpSize;
	}
	if((FT_THREADS_PER_BLOCK_ON_COMPUTE_INITIAL_CONDITIONS>=1)&&
			(FT_THREADS_PER_BLOCK_ON_COMPUTE_INITIAL_CONDITIONS<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_compute_initial_conditions = FT_THREADS_PER_BLOCK_ON_COMPUTE_INITIAL_CONDITIONS;
	}else{
		threads_per_block_on_compute_initial_conditions = gpu_device_properties.warpSize;
	}
	if((FT_THREADS_PER_BLOCK_ON_INIT_UI>=1)&&
			(FT_THREADS_PER_BLOCK_ON_INIT_UI<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_init_ui = FT_THREADS_PER_BLOCK_ON_INIT_UI;
	}else{
		threads_per_block_on_init_ui=gpu_device_properties.warpSize;
	}
	if((FT_THREADS_PER_BLOCK_ON_EVOLVE>=1)&&
			(FT_THREADS_PER_BLOCK_ON_EVOLVE<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_evolve = FT_THREADS_PER_BLOCK_ON_EVOLVE;
	}else{
		threads_per_block_on_evolve=gpu_device_properties.warpSize;
	}
	if((FT_THREADS_PER_BLOCK_ON_FFTX_1>=1)&&
			(FT_THREADS_PER_BLOCK_ON_FFTX_1<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_fftx_1 = FT_THREADS_PER_BLOCK_ON_FFTX_1;
	}else{
		threads_per_block_on_fftx_1 = gpu_device_properties.warpSize;
	}
	if((FT_THREADS_PER_BLOCK_ON_FFTX_2>=1)&&
			(FT_THREADS_PER_BLOCK_ON_FFTX_2<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_fftx_2 = FT_THREADS_PER_BLOCK_ON_FFTX_2;
	}else{
		threads_per_block_on_fftx_2 = gpu_device_properties.warpSize;
	}
	if((FT_THREADS_PER_BLOCK_ON_FFTX_3>=1)&&
			(FT_THREADS_PER_BLOCK_ON_FFTX_3<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_fftx_3 = FT_THREADS_PER_BLOCK_ON_FFTX_3;
	}else{
		threads_per_block_on_fftx_3 = gpu_device_properties.warpSize;
	}
	if((FT_THREADS_PER_BLOCK_ON_FFTY_1>=1)&&
			(FT_THREADS_PER_BLOCK_ON_FFTY_1<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_ffty_1 = FT_THREADS_PER_BLOCK_ON_FFTY_1;
	}else{
		threads_per_block_on_ffty_1 = gpu_device_properties.warpSize;
	}
	if((FT_THREADS_PER_BLOCK_ON_FFTY_2>=1)&&
			(FT_THREADS_PER_BLOCK_ON_FFTY_2<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_ffty_2 = FT_THREADS_PER_BLOCK_ON_FFTY_2;
	}else{
		threads_per_block_on_ffty_2 = gpu_device_properties.warpSize;
	}
	if((FT_THREADS_PER_BLOCK_ON_FFTY_3>=1)&&
			(FT_THREADS_PER_BLOCK_ON_FFTY_3<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_ffty_3 = FT_THREADS_PER_BLOCK_ON_FFTY_3;
	}else{
		threads_per_block_on_ffty_3 = gpu_device_properties.warpSize;
	}
	if((FT_THREADS_PER_BLOCK_ON_FFTZ_1>=1)&&
			(FT_THREADS_PER_BLOCK_ON_FFTZ_1<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_fftz_1 = FT_THREADS_PER_BLOCK_ON_FFTZ_1;
	}else{
		threads_per_block_on_fftz_1 = gpu_device_properties.warpSize;
	}
	if((FT_THREADS_PER_BLOCK_ON_FFTZ_2>=1)&&
			(FT_THREADS_PER_BLOCK_ON_FFTZ_2<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_fftz_2 = FT_THREADS_PER_BLOCK_ON_FFTZ_2;
	}else{
		threads_per_block_on_fftz_2 = gpu_device_properties.warpSize;
	}
	if((FT_THREADS_PER_BLOCK_ON_FFTZ_3>=1)&&
			(FT_THREADS_PER_BLOCK_ON_FFTZ_3<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_fftz_3 = FT_THREADS_PER_BLOCK_ON_FFTZ_3;
	}else{
		threads_per_block_on_fftz_3 = gpu_device_properties.warpSize;
	}
	if((FT_THREADS_PER_BLOCK_ON_CHECKSUM>=1)&&
			(FT_THREADS_PER_BLOCK_ON_CHECKSUM<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_checksum = FT_THREADS_PER_BLOCK_ON_CHECKSUM;
	}else{
		threads_per_block_on_checksum = gpu_device_properties.warpSize;
	}	

	blocks_per_grid_on_compute_indexmap=ceil(double(NTOTAL)/double(threads_per_block_on_compute_indexmap));
	blocks_per_grid_on_compute_initial_conditions=ceil(double(NZ)/double(threads_per_block_on_compute_initial_conditions));
	blocks_per_grid_on_init_ui=ceil(double(NTOTAL)/double(threads_per_block_on_init_ui));
	blocks_per_grid_on_evolve=ceil(double(NTOTAL)/double(threads_per_block_on_evolve));
	blocks_per_grid_on_fftx_1=ceil(double(NX*NY*NZ)/double(threads_per_block_on_fftx_1));
	blocks_per_grid_on_fftx_2=ceil(double(NY*NZ)/double(threads_per_block_on_fftx_2));
	blocks_per_grid_on_fftx_3=ceil(double(NX*NY*NZ)/double(threads_per_block_on_fftx_3));
	blocks_per_grid_on_ffty_1=ceil(double(NX*NY*NZ)/double(threads_per_block_on_ffty_1));
	blocks_per_grid_on_ffty_2=ceil(double(NX*NZ)/double(threads_per_block_on_ffty_2));
	blocks_per_grid_on_ffty_3=ceil(double(NX*NY*NZ)/double(threads_per_block_on_ffty_3));
	blocks_per_grid_on_fftz_1=ceil(double(NX*NY*NZ)/double(threads_per_block_on_fftz_1));
	blocks_per_grid_on_fftz_2=ceil(double(NX*NY)/double(threads_per_block_on_fftz_2));
	blocks_per_grid_on_fftz_3=ceil(double(NX*NY*NZ)/double(threads_per_block_on_fftz_3));
	blocks_per_grid_on_checksum=ceil(double(CHECKSUM_TASKS)/double(threads_per_block_on_checksum));

	size_sums_device=(NITER_DEFAULT+1)*sizeof(dcomplex);
	size_starts_device=NZ*sizeof(double);
	size_twiddle_device=NTOTAL*sizeof(double);
	size_u_device=MAXDIM*sizeof(dcomplex);
	size_u0_device=NTOTAL*sizeof(dcomplex);
	size_u1_device=NTOTAL*sizeof(dcomplex);
	size_y0_device=NTOTAL*sizeof(dcomplex);
	size_y1_device=NTOTAL*sizeof(dcomplex);
	size_shared_data=threads_per_block_on_checksum*sizeof(dcomplex);

	hipMalloc(&sums_device, size_sums_device);
	hipMalloc(&starts_device, size_starts_device);
	hipMalloc(&twiddle_device, size_twiddle_device);
	hipMalloc(&u_device, size_u_device);
	hipMalloc(&u0_device, size_u0_device);
	hipMalloc(&u1_device, size_u1_device);
	hipMalloc(&y0_device, size_y0_device);
	hipMalloc(&y1_device, size_y1_device);

	omp_set_num_threads(OMP_THREADS);	
}

static void verify(int d1,
		int d2,
		int d3,
		int nt,
		boolean* verified,
		char* class_npb){
	int i;
	double err, epsilon;
	/*
	 * ---------------------------------------------------------------------
	 * reference checksums
	 * ---------------------------------------------------------------------
	 */
	dcomplex csum_ref[25+1];
	*class_npb = 'U';
	epsilon = 1.0e-12;
	*verified = false;
	if(d1 == 64 && d2 == 64 && d3 == 64 && nt == 6){
		/*
		 * ---------------------------------------------------------------------
		 * sample size reference checksums
		 * ---------------------------------------------------------------------
		 */
		*class_npb = 'S';
		csum_ref[1] = dcomplex_create(5.546087004964E+02, 4.845363331978E+02);
		csum_ref[2] = dcomplex_create(5.546385409189E+02, 4.865304269511E+02);
		csum_ref[3] = dcomplex_create(5.546148406171E+02, 4.883910722336E+02);
		csum_ref[4] = dcomplex_create(5.545423607415E+02, 4.901273169046E+02);
		csum_ref[5] = dcomplex_create(5.544255039624E+02, 4.917475857993E+02);
		csum_ref[6] = dcomplex_create(5.542683411902E+02, 4.932597244941E+02);
	}else if(d1 == 128 && d2 == 128 && d3 == 32 && nt == 6){
		/*
		 * ---------------------------------------------------------------------
		 * class_npb W size reference checksums
		 * ---------------------------------------------------------------------
		 */
		*class_npb = 'W';
		csum_ref[1] = dcomplex_create(5.673612178944E+02, 5.293246849175E+02);
		csum_ref[2] = dcomplex_create(5.631436885271E+02, 5.282149986629E+02);
		csum_ref[3] = dcomplex_create(5.594024089970E+02, 5.270996558037E+02);
		csum_ref[4] = dcomplex_create(5.560698047020E+02, 5.260027904925E+02);
		csum_ref[5] = dcomplex_create(5.530898991250E+02, 5.249400845633E+02);
		csum_ref[6] = dcomplex_create(5.504159734538E+02, 5.239212247086E+02);
	}else if(d1 == 256 && d2 == 256 && d3 == 128 && nt == 6){
		/*
		 * ---------------------------------------------------------------------
		 * class_npb A size reference checksums
		 * ---------------------------------------------------------------------
		 */
		*class_npb = 'A';
		csum_ref[1] = dcomplex_create(5.046735008193E+02, 5.114047905510E+02);
		csum_ref[2] = dcomplex_create(5.059412319734E+02, 5.098809666433E+02);
		csum_ref[3] = dcomplex_create(5.069376896287E+02, 5.098144042213E+02);
		csum_ref[4] = dcomplex_create(5.077892868474E+02, 5.101336130759E+02);
		csum_ref[5] = dcomplex_create(5.085233095391E+02, 5.104914655194E+02);
		csum_ref[6] = dcomplex_create(5.091487099959E+02, 5.107917842803E+02);
	}else if(d1 == 512 && d2 == 256 && d3 == 256 && nt == 20){
		/*
		 * --------------------------------------------------------------------
		 * class_npb B size reference checksums
		 * ---------------------------------------------------------------------
		 */
		*class_npb = 'B';
		csum_ref[1]  = dcomplex_create(5.177643571579E+02, 5.077803458597E+02);
		csum_ref[2]  = dcomplex_create(5.154521291263E+02, 5.088249431599E+02);
		csum_ref[3]  = dcomplex_create(5.146409228649E+02, 5.096208912659E+02);
		csum_ref[4]  = dcomplex_create(5.142378756213E+02, 5.101023387619E+02);
		csum_ref[5]  = dcomplex_create(5.139626667737E+02, 5.103976610617E+02);
		csum_ref[6]  = dcomplex_create(5.137423460082E+02, 5.105948019802E+02);
		csum_ref[7]  = dcomplex_create(5.135547056878E+02, 5.107404165783E+02);
		csum_ref[8]  = dcomplex_create(5.133910925466E+02, 5.108576573661E+02);
		csum_ref[9]  = dcomplex_create(5.132470705390E+02, 5.109577278523E+02);
		csum_ref[10] = dcomplex_create(5.131197729984E+02, 5.110460304483E+02);
		csum_ref[11] = dcomplex_create(5.130070319283E+02, 5.111252433800E+02);
		csum_ref[12] = dcomplex_create(5.129070537032E+02, 5.111968077718E+02);
		csum_ref[13] = dcomplex_create(5.128182883502E+02, 5.112616233064E+02);
		csum_ref[14] = dcomplex_create(5.127393733383E+02, 5.113203605551E+02);
		csum_ref[15] = dcomplex_create(5.126691062020E+02, 5.113735928093E+02);
		csum_ref[16] = dcomplex_create(5.126064276004E+02, 5.114218460548E+02);
		csum_ref[17] = dcomplex_create(5.125504076570E+02, 5.114656139760E+02);
		csum_ref[18] = dcomplex_create(5.125002331720E+02, 5.115053595966E+02);
		csum_ref[19] = dcomplex_create(5.124551951846E+02, 5.115415130407E+02);
		csum_ref[20] = dcomplex_create(5.124146770029E+02, 5.115744692211E+02);
	}else if(d1 == 512 && d2 == 512 && d3 == 512 && nt == 20){
		/*
		 * ---------------------------------------------------------------------
		 * class_npb C size reference checksums
		 * ---------------------------------------------------------------------
		 */
		*class_npb = 'C';
		csum_ref[1]  = dcomplex_create(5.195078707457E+02, 5.149019699238E+02);
		csum_ref[2]  = dcomplex_create(5.155422171134E+02, 5.127578201997E+02);
		csum_ref[3]  = dcomplex_create(5.144678022222E+02, 5.122251847514E+02);
		csum_ref[4]  = dcomplex_create(5.140150594328E+02, 5.121090289018E+02);
		csum_ref[5]  = dcomplex_create(5.137550426810E+02, 5.121143685824E+02);
		csum_ref[6]  = dcomplex_create(5.135811056728E+02, 5.121496764568E+02);
		csum_ref[7]  = dcomplex_create(5.134569343165E+02, 5.121870921893E+02);
		csum_ref[8]  = dcomplex_create(5.133651975661E+02, 5.122193250322E+02);
		csum_ref[9]  = dcomplex_create(5.132955192805E+02, 5.122454735794E+02);
		csum_ref[10] = dcomplex_create(5.132410471738E+02, 5.122663649603E+02);
		csum_ref[11] = dcomplex_create(5.131971141679E+02, 5.122830879827E+02);
		csum_ref[12] = dcomplex_create(5.131605205716E+02, 5.122965869718E+02);
		csum_ref[13] = dcomplex_create(5.131290734194E+02, 5.123075927445E+02);
		csum_ref[14] = dcomplex_create(5.131012720314E+02, 5.123166486553E+02);
		csum_ref[15] = dcomplex_create(5.130760908195E+02, 5.123241541685E+02);
		csum_ref[16] = dcomplex_create(5.130528295923E+02, 5.123304037599E+02);
		csum_ref[17] = dcomplex_create(5.130310107773E+02, 5.123356167976E+02);
		csum_ref[18] = dcomplex_create(5.130103090133E+02, 5.123399592211E+02);
		csum_ref[19] = dcomplex_create(5.129905029333E+02, 5.123435588985E+02);
		csum_ref[20] = dcomplex_create(5.129714421109E+02, 5.123465164008E+02);
	}else if(d1 == 2048 && d2 == 1024 && d3 == 1024 && nt == 25){
		/*
		 * ---------------------------------------------------------------------
		 * class_npb D size reference checksums
		 * ---------------------------------------------------------------------
		 */
		*class_npb = 'D';
		csum_ref[1]  = dcomplex_create(5.122230065252E+02, 5.118534037109E+02);
		csum_ref[2]  = dcomplex_create(5.120463975765E+02, 5.117061181082E+02);
		csum_ref[3]  = dcomplex_create(5.119865766760E+02, 5.117096364601E+02);
		csum_ref[4]  = dcomplex_create(5.119518799488E+02, 5.117373863950E+02);
		csum_ref[5]  = dcomplex_create(5.119269088223E+02, 5.117680347632E+02);
		csum_ref[6]  = dcomplex_create(5.119082416858E+02, 5.117967875532E+02);
		csum_ref[7]  = dcomplex_create(5.118943814638E+02, 5.118225281841E+02);
		csum_ref[8]  = dcomplex_create(5.118842385057E+02, 5.118451629348E+02);
		csum_ref[9]  = dcomplex_create(5.118769435632E+02, 5.118649119387E+02);
		csum_ref[10] = dcomplex_create(5.118718203448E+02, 5.118820803844E+02);
		csum_ref[11] = dcomplex_create(5.118683569061E+02, 5.118969781011E+02);
		csum_ref[12] = dcomplex_create(5.118661708593E+02, 5.119098918835E+02);
		csum_ref[13] = dcomplex_create(5.118649768950E+02, 5.119210777066E+02);
		csum_ref[14] = dcomplex_create(5.118645605626E+02, 5.119307604484E+02);
		csum_ref[15] = dcomplex_create(5.118647586618E+02, 5.119391362671E+02);
		csum_ref[16] = dcomplex_create(5.118654451572E+02, 5.119463757241E+02);
		csum_ref[17] = dcomplex_create(5.118665212451E+02, 5.119526269238E+02);
		csum_ref[18] = dcomplex_create(5.118679083821E+02, 5.119580184108E+02);
		csum_ref[19] = dcomplex_create(5.118695433664E+02, 5.119626617538E+02);
		csum_ref[20] = dcomplex_create(5.118713748264E+02, 5.119666538138E+02);
		csum_ref[21] = dcomplex_create(5.118733606701E+02, 5.119700787219E+02);
		csum_ref[22] = dcomplex_create(5.118754661974E+02, 5.119730095953E+02);
		csum_ref[23] = dcomplex_create(5.118776626738E+02, 5.119755100241E+02);
		csum_ref[24] = dcomplex_create(5.118799262314E+02, 5.119776353561E+02);
		csum_ref[25] = dcomplex_create(5.118822370068E+02, 5.119794338060E+02);
	}else if(d1 == 4096 && d2 == 2048 && d3 == 2048 && nt == 25){
		/*
		 * ---------------------------------------------------------------------
		 * class_npb E size reference checksums
		 * ---------------------------------------------------------------------
		 */
		*class_npb = 'E';
		csum_ref[1]  = dcomplex_create(5.121601045346E+02, 5.117395998266E+02);
		csum_ref[2]  = dcomplex_create(5.120905403678E+02, 5.118614716182E+02);
		csum_ref[3]  = dcomplex_create(5.120623229306E+02, 5.119074203747E+02);
		csum_ref[4]  = dcomplex_create(5.120438418997E+02, 5.119345900733E+02);
		csum_ref[5]  = dcomplex_create(5.120311521872E+02, 5.119551325550E+02);
		csum_ref[6]  = dcomplex_create(5.120226088809E+02, 5.119720179919E+02);
		csum_ref[7]  = dcomplex_create(5.120169296534E+02, 5.119861371665E+02);
		csum_ref[8]  = dcomplex_create(5.120131225172E+02, 5.119979364402E+02);
		csum_ref[9]  = dcomplex_create(5.120104767108E+02, 5.120077674092E+02);
		csum_ref[10] = dcomplex_create(5.120085127969E+02, 5.120159443121E+02);
		csum_ref[11] = dcomplex_create(5.120069224127E+02, 5.120227453670E+02);
		csum_ref[12] = dcomplex_create(5.120055158164E+02, 5.120284096041E+02);
		csum_ref[13] = dcomplex_create(5.120041820159E+02, 5.120331373793E+02);
		csum_ref[14] = dcomplex_create(5.120028605402E+02, 5.120370938679E+02);
		csum_ref[15] = dcomplex_create(5.120015223011E+02, 5.120404138831E+02);
		csum_ref[16] = dcomplex_create(5.120001570022E+02, 5.120432068837E+02);
		csum_ref[17] = dcomplex_create(5.119987650555E+02, 5.120455615860E+02);
		csum_ref[18] = dcomplex_create(5.119973525091E+02, 5.120475499442E+02);
		csum_ref[19] = dcomplex_create(5.119959279472E+02, 5.120492304629E+02);
		csum_ref[20] = dcomplex_create(5.119945006558E+02, 5.120506508902E+02);
		csum_ref[21] = dcomplex_create(5.119930795911E+02, 5.120518503782E+02);
		csum_ref[22] = dcomplex_create(5.119916728462E+02, 5.120528612016E+02);
		csum_ref[23] = dcomplex_create(5.119902874185E+02, 5.120537101195E+02);
		csum_ref[24] = dcomplex_create(5.119889291565E+02, 5.120544194514E+02);
		csum_ref[25] = dcomplex_create(5.119876028049E+02, 5.120550079284E+02);
	}
	if(*class_npb != 'U'){
		*verified = TRUE;
		for(i = 1; i <= nt; i++){
			err = dcomplex_abs(dcomplex_div(dcomplex_sub(sums[i], csum_ref[i]),
						csum_ref[i]));
			if(!(err <= epsilon)){
				*verified = FALSE;
				break;
			}
		}
	}
	if(*class_npb != 'U'){
		if(*verified){
			printf(" Result verification successful\n");
		}else{
			printf(" Result verification failed\n");
		}
	}
	printf(" class_npb = %c\n", *class_npb);
}

__device__ void vranlc_device(int n, 
		double* x_seed, 
		double a, 
		double y[]){
	int i;
	double x,t1,t2,t3,t4,a1,a2,x1,x2,z;
	t1 = R23 * a;
	a1 = (int)t1;
	a2 = a - T23 * a1;
	x = *x_seed;
	for(i=0; i<n; i++){
		t1 = R23 * x;
		x1 = (int)t1;
		x2 = x - T23 * x1;
		t1 = a1 * x2 + a2 * x1;
		t2 = (int)(R23 * t1);
		z = t1 - T23 * t2;
		t3 = T23 * z + a2 * x2;
		t4 = (int)(R46 * t3);
		x = t3 - T46 * t4;
		y[i] = R46 * x;
	}
	*x_seed = x;
}
