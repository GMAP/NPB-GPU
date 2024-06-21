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
 * The GSParLib version is a parallel implementation of the serial C++ version
 * GSParLib version: https://github.com/GMAP/NPB-GPU/tree/master/GSParLib
 *
 * Authors of the GSParLib code:
 *      Gabriell Araujo <hexenoften@gmail.com>
 *
 * ------------------------------------------------------------------------------
 */

// export LD_LIBRARY_PATH=../lib/gspar/bin:$LD_LIBRARY_PATH
// clear && make clean && make ep CLASS=S GPU_DRIVER=CUDA && bin/ep.S 
// clear && make clean && make ep CLASS=S GPU_DRIVER=OPENCL && bin/ep.S 

#include <iostream>
#include <chrono>

#include "../common/npb.hpp"
#include "npbparams.hpp"

#ifdef GSPARDRIVER_CUDA
#include "GSPar_CUDA.hpp"
using namespace GSPar::Driver::CUDA;
#else
/* GSPARDRIVER_OPENCL */
#include "GSPar_OpenCL.hpp"
using namespace GSPar::Driver::OpenCL;
#endif

using namespace std;

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
#define CHECKSUM_TASKS (1024)
#define MAX_THREADS_PER_BLOCK (1024)
#define THREADS_PER_BLOCK (32)

/* global variables */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static dcomplex sums[NITER_DEFAULT+1];
static double twiddle[NTOTAL];
static dcomplex u[MAXDIM];
static dcomplex u0[NTOTAL];
static dcomplex u1[NTOTAL];
static dcomplex y0_host[NTOTAL];
static dcomplex y1_host[NTOTAL];
static int dims[3];
static double starts[NZ];
#else
static dcomplex *sums=(dcomplex*)malloc(sizeof(dcomplex)*(NITER_DEFAULT+1));
static double *twiddle=(double*)malloc(sizeof(double)*(NTOTAL));
static dcomplex *u=(dcomplex*)malloc(sizeof(dcomplex)*(MAXDIM));
static dcomplex *u0=(dcomplex*)malloc(sizeof(dcomplex)*(NTOTAL));
static dcomplex *u1=(dcomplex*)malloc(sizeof(dcomplex)*(NTOTAL));
static dcomplex *y0_host=(dcomplex*)malloc(sizeof(dcomplex)*(NTOTAL));
static dcomplex *y1_host=(dcomplex*)malloc(sizeof(dcomplex)*(NTOTAL));
static int *dims=(int*)malloc(sizeof(int)*(3));
static double *starts=(double*)malloc(sizeof(double)*(NZ));
#endif
static int niter;

/* gpu variables */
MemoryObject* starts_device;
MemoryObject* twiddle_device;
MemoryObject* sums_device;
MemoryObject* u_device;
MemoryObject* u0_device;
MemoryObject* u1_device;
MemoryObject* u2_device;
MemoryObject* y0_device;
MemoryObject* y1_device;
//
size_t size_sums_device;
size_t size_starts_device;
size_t size_twiddle_device;
size_t size_u_device;
size_t size_u0_device;
size_t size_u1_device;
size_t size_y0_device;
size_t size_y1_device;
size_t size_shared_data;
//
int amount_of_work_on_compute_indexmap;
int amount_of_work_on_compute_initial_conditions;
int amount_of_work_on_init_ui;
int amount_of_work_on_evolve;
int amount_of_work_on_fftx_1;
int amount_of_work_on_fftx_2;
int amount_of_work_on_fftx_3;
int amount_of_work_on_ffty_1;
int amount_of_work_on_ffty_2;
int amount_of_work_on_ffty_3;
int amount_of_work_on_fftz_1;
int amount_of_work_on_fftz_2;
int amount_of_work_on_fftz_3;
int amount_of_work_on_checksum;
//
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
//
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
//
Kernel* kernel_compute_indexmap;
Kernel* kernel_compute_initial_conditions;
Kernel* kernel_init_ui;
Kernel* kernel_evolve;
Kernel* kernel_fftx_1;
Kernel* kernel_fftx_2;
Kernel* kernel_fftx_3;
Kernel* kernel_ffty_1;
Kernel* kernel_ffty_2;
Kernel* kernel_ffty_3;
Kernel* kernel_fftz_1;
Kernel* kernel_fftz_2;
Kernel* kernel_fftz_3;
Kernel* kernel_checksum;
//
extern std::string kernel_source_compute_indexmap;
extern std::string kernel_source_compute_initial_conditions;
extern std::string kernel_source_init_ui;
extern std::string kernel_source_evolve;
extern std::string kernel_source_fftx_1;
extern std::string kernel_source_fftx_2;
extern std::string kernel_source_fftx_3;
extern std::string kernel_source_ffty_1;
extern std::string kernel_source_ffty_2;
extern std::string kernel_source_ffty_3;
extern std::string kernel_source_fftz_1;
extern std::string kernel_source_fftz_2;
extern std::string kernel_source_fftz_3;
extern std::string kernel_source_checksum;
extern std::string source_additional_routines_1;
extern std::string source_additional_routines_2;
//
string DEVICE_NAME;
Instance* driver;

/* function declarations for the cpu */
static void cffts1(int is,
		int d1,
		int d2,
		int d3,
		void* pointer_x,
		void* pointer_xout,
		dcomplex y1[][FFTBLOCKPAD],
		dcomplex y2[][FFTBLOCKPAD]);
static void cffts2(int is,
		int d1,
		int d2,
		int d3,
		void* pointer_x,
		void* pointer_xout,
		dcomplex y1[][FFTBLOCKPAD],
		dcomplex y2[][FFTBLOCKPAD]);
static void cffts3(int is,
		int d1,
		int d2,
		int d3,
		void* pointer_x,
		void* pointer_xout,
		dcomplex y1[][FFTBLOCKPAD],
		dcomplex y2[][FFTBLOCKPAD]);
static void cfftz(int is,
		int m,
		int n,
		dcomplex x[][FFTBLOCKPAD],
		dcomplex y[][FFTBLOCKPAD]);
static void checksum(int i,
		void* pointer_u1,
		int d1,
		int d2,
		int d3);
static void compute_indexmap(void* pointer_twiddle,
		int d1,
		int d2,
		int d3);
static void compute_initial_conditions(void* pointer_u0,
		int d1,
		int d2,
		int d3);
static void evolve(void* pointer_u0,
		void* pointer_u1,
		void* pointer_twiddle,
		int d1,
		int d2,
		int d3);
static void fft(int dir,
		void* pointer_x1,
		void* pointer_x2);
static void fft_init(int n);
static void fftz2(int is,
		int l,
		int m,
		int n,
		int ny,
		int ny1,
		dcomplex u[],
		dcomplex x[][FFTBLOCKPAD],
		dcomplex y[][FFTBLOCKPAD]);
static int ilog2(int n);
static void init_ui(void* pointer_u0,
		void* pointer_u1,
		void* pointer_twiddle,
		int d1,
		int d2,
		int d3);
static void ipow46(double a,
		int exponent,
		double* result);
static void setup();
static void verify(int d1,
		int d2,
		int d3,
		int nt,
		boolean* verified,
		char* class_npb);

/* function declarations for the gpu */
static void cffts1_gpu(
		const int is, 
		MemoryObject u[], 
		MemoryObject x_in[], 
		MemoryObject x_out[], 
		MemoryObject y0[], 
		MemoryObject y1[]);
static void cffts2_gpu(
		int is, 
		MemoryObject u[], 
		MemoryObject x_in[], 
		MemoryObject x_out[], 
		MemoryObject y0[], 
		MemoryObject y1[]);
static void cffts3_gpu(
		int is, 
		MemoryObject u[], 
		MemoryObject x_in[], 
		MemoryObject x_out[], 
		MemoryObject y0[], 
		MemoryObject y1[]);
static void checksum_gpu(
		int iteration, 
		MemoryObject u1[]);				
static void compute_indexmap_gpu(
		MemoryObject twiddle[]);
static void compute_initial_conditions_gpu(
		MemoryObject u0[]);
static void evolve_gpu(
		MemoryObject u0[], 
		MemoryObject u1[], 
		MemoryObject twiddle[]);
static void fft_gpu(
		int dir,
		MemoryObject x1[],
		MemoryObject x2[]);
static void fft_init_gpu(
		int n);
static void init_ui_gpu(
		MemoryObject u0[],
		MemoryObject u1[],
		MemoryObject twiddle[]);
static void setup();
static void setup_gpu();

/* ft */
int main(int argc, char** argv){
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
	printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif
#ifdef GSPARDRIVER_CUDA
	printf(" Performing GSParLib with CUDA\n");
#else
/* GSPARDRIVER_OPENCL */
	printf(" Performing GSParLib with OpenCL\n");
#endif
	int iter=0;
	double total_time, mflops;
	boolean verified;
	char class_npb;	

	for(int i=0; i<(NITER_DEFAULT+1); i++){
		dcomplex aux;
		aux.real = 0.0;
		aux.imag = 0.0;
		sums[i]=aux;
	}

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
	
	compute_indexmap_gpu(twiddle_device);
	
	compute_initial_conditions_gpu(u1_device);
	
	fft_init_gpu(MAXDIM);
	
	fft_gpu(1, u1_device, u0_device);	

	/*
	 * ---------------------------------------------------------------------
	 * start over from the beginning. note that all operations must
	 * be timed, in contrast to other benchmarks. 
	 * ---------------------------------------------------------------------
	 */
	timer_clear(PROFILING_TOTAL_TIME);
	timer_start(PROFILING_TOTAL_TIME);
	
	compute_indexmap_gpu(twiddle_device);
	
	compute_initial_conditions_gpu(u1_device);
	
	fft_init_gpu(MAXDIM);
	
	fft_gpu(1, u1_device, u0_device);

	for(iter=1; iter<=niter; iter++){
		evolve_gpu(u0_device, u1_device, twiddle_device);

		fft_gpu(-1, u1_device, u1_device);
		
		checksum_gpu(iter, u1_device);	
	}
	sums_device->copyOut();

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
			(char*) DEVICE_NAME.c_str(),
			(char*)CS1, 
			(char*)CS2, 
			(char*)CS3, 
			(char*)CS4, 
			(char*)CS5, 
			(char*)CS6, 
			(char*)CS7);

	return 0;
}

static void cffts1(int is,
		int d1,
		int d2,
		int d3,
		void* pointer_x,
		void* pointer_xout,
		dcomplex y1[][FFTBLOCKPAD],
		dcomplex y2[][FFTBLOCKPAD]){
	dcomplex (*x)[NY][NX] = (dcomplex(*)[NY][NX])pointer_x;
	dcomplex (*xout)[NY][NX] = (dcomplex(*)[NY][NX])pointer_xout;

	int logd1;
	int i, j, k, jj;

	logd1 = ilog2(d1);
	
	for(k=0; k<d3; k++){
		for(jj=0; jj<=d2-FFTBLOCK; jj+=FFTBLOCK){
			for(j=0; j<FFTBLOCK; j++){
				for(i=0; i<d1; i++){
					y1[i][j] = x[k][j+jj][i];
				}
			}
			cfftz(is, logd1, d1, y1, y2);
			for(j=0; j<FFTBLOCK; j++){
				for(i=0; i<d1; i++){
					xout[k][j+jj][i] = y1[i][j];
				}
			}
		}
	}
}

static void cffts2(int is,
		int d1,
		int d2,
		int d3,
		void* pointer_x,
		void* pointer_xout,
		dcomplex y1[][FFTBLOCKPAD],
		dcomplex y2[][FFTBLOCKPAD]){
	dcomplex (*x)[NY][NX] = (dcomplex(*)[NY][NX])pointer_x;
	dcomplex (*xout)[NY][NX] = (dcomplex(*)[NY][NX])pointer_xout;

	int logd2;
	int i, j, k, ii;

	logd2 = ilog2(d2);
	
	for(k=0; k<d3; k++){
		for(ii=0; ii<=d1-FFTBLOCK; ii+=FFTBLOCK){
			for(j=0; j<d2; j++){
				for(i=0; i<FFTBLOCK; i++){
					y1[j][i] = x[k][j][i+ii];
				}
			}
			cfftz(is, logd2, d2, y1, y2);
			for(j=0; j<d2; j++){
				for(i=0; i<FFTBLOCK; i++){
					xout[k][j][i+ii] = y1[j][i];
				}
			}
		}
	}
}

static void cffts3(int is,
		int d1,
		int d2,
		int d3,
		void* pointer_x,
		void* pointer_xout,
		dcomplex y1[][FFTBLOCKPAD],
		dcomplex y2[][FFTBLOCKPAD]){
	dcomplex (*x)[NY][NX] = (dcomplex(*)[NY][NX])pointer_x;
	dcomplex (*xout)[NY][NX] = (dcomplex(*)[NY][NX])pointer_xout;

	int logd3;
	int i, j, k, ii;

	logd3 = ilog2(d3);
	
	for(j=0; j<d2; j++){
		for(ii=0; ii<=d1-FFTBLOCK; ii+=FFTBLOCK){
			for(k=0; k<d3; k++){
				for(i=0; i<FFTBLOCK; i++){
					y1[k][i] = x[k][j][i+ii];
				}
			}
			cfftz(is, logd3, d3, y1, y2);
			for(k=0; k<d3; k++){
				for(i=0; i<FFTBLOCK; i++){
					xout[k][j][i+ii] = y1[k][i];
				}
			}
		}
	}
}

/*
 * ---------------------------------------------------------------------
 * computes NY N-point complex-to-complex FFTs of X using an algorithm due
 * to swarztrauber. X is both the input and the output array, while Y is a 
 * scratch array. it is assumed that N = 2^M. before calling CFFTZ to 
 * perform FFTs, the array U must be initialized by calling CFFTZ with is 
 * set to 0 and M set to MX, where MX is the maximum value of M for any 
 * subsequent call.
 * ---------------------------------------------------------------------
 */
static void cfftz(int is,
		int m,
		int n,
		dcomplex x[][FFTBLOCKPAD],
		dcomplex y[][FFTBLOCKPAD]){
	int i,j,l,mx;

	/*
	 * ---------------------------------------------------------------------
	 * check if input parameters are invalid.
	 * ---------------------------------------------------------------------
	 */
	mx = (int)(u[0].real);
	if((is != 1 && is != -1) || m < 1 || m > mx){
		printf("CFFTZ: Either U has not been initialized, or else\n"    
				"one of the input parameters is invalid%5d%5d%5d\n", is, m, mx);
		exit(EXIT_FAILURE); 
	}

	/*
	 * ---------------------------------------------------------------------
	 * perform one variant of the Stockham FFT.
	 * ---------------------------------------------------------------------
	 */
	for(l=1; l<=m; l+=2){
		fftz2(is, l, m, n, FFTBLOCK, FFTBLOCKPAD, u, x, y);
		if(l==m){
			/*
			 * ---------------------------------------------------------------------
			 * copy Y to X.
			 * ---------------------------------------------------------------------
			 */
			for(j=0; j<n; j++){
				for(i=0; i<FFTBLOCK; i++){
					x[j][i] = y[j][i];
				}
			}
			break;
		}
		fftz2(is, l + 1, m, n, FFTBLOCK, FFTBLOCKPAD, u, y, x);
	}	
}

static void checksum(int i,
		void* pointer_u1,
		int d1,
		int d2,
		int d3){
	dcomplex (*u1)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u1;
	int j,q,r,s;
	dcomplex chk = dcomplex_create(0.0, 0.0);
	for(j=1; j<=1024; j++){
		q = j % NX;
		r = (3*j) % NY;
		s = (5*j) % NZ;
		chk = dcomplex_add(chk, u1[s][r][q]);
	}
	chk = dcomplex_div2(chk, (double)(NTOTAL));
	sums[i] = chk;
}

/*
 * compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2 
 * for time evolution exponent. 
 */
static void compute_indexmap(void* pointer_twiddle,
		int d1,
		int d2,
		int d3){
	double (*twiddle)[NY][NX] = (double(*)[NY][NX])pointer_twiddle;

	int i, j, k, kk, kk2, jj, kj2, ii;

	/*
	 * ---------------------------------------------------------------------
	 * basically we want to convert the fortran indices 
	 * 1 2 3 4 5 6 7 8 
	 * to 
	 * 0 1 2 3 -4 -3 -2 -1
	 * the following magic formula does the trick:
	 * mod(i-1+n/2, n) - n/2
	 * ---------------------------------------------------------------------
	 */
	for(k=0; k<d3; k++){
		kk = ((k+NZ/2) % NZ) - NZ/2;
		kk2 = kk*kk;
		for(j=0; j<d2; j++){
			jj = ((j+NY/2) % NY) - NY/2;
			kj2 = jj*jj+kk2;
			for(i=0; i<d1; i++){
				ii = ((i+NX/2) % NX) - NX/2;
				twiddle[k][j][i] = exp(AP*(double)(ii*ii+kj2));
			}
		}
	}
}

/*
 * ---------------------------------------------------------------------
 * fill in array u0 with initial conditions from 
 * random number generator 
 * ---------------------------------------------------------------------
 */
static void compute_initial_conditions(void* pointer_u0,
		int d1,
		int d2,
		int d3){
	dcomplex (*u0)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u0;

	int k, j;
	double x0, start, an, starts[NZ];
	start = SEED;

	/*
	 * ---------------------------------------------------------------------
	 * jump to the starting element for our first plane.
	 * ---------------------------------------------------------------------
	 */
	ipow46(A, 0, &an);
	randlc(&start, an);
	ipow46(A, 2*NX*NY, &an);

	starts[0] = start;
	for(int k=1; k<dims[2]; k++){
		randlc(&start, an);
		starts[k] = start;
	}

	/*
	 * ---------------------------------------------------------------------
	 * go through by z planes filling in one square at a time.
	 * ---------------------------------------------------------------------
	 */
	for(k=0; k<dims[2]; k++){
		x0 = starts[k];
		for(j=0; j<dims[1]; j++){			
			vranlc(2*NX, &x0, A, (double*)&u0[k][j][0]);
		}
	}
}

/*
 * ---------------------------------------------------------------------
 * evolve u0 -> u1 (t time steps) in fourier space
 * ---------------------------------------------------------------------
 */
static void evolve(void* pointer_u0,
		void* pointer_u1,
		void* pointer_twiddle,
		int d1,
		int d2,
		int d3){
	dcomplex (*u0)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u0;
	dcomplex (*u1)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u1;
	double (*twiddle)[NY][NX] = (double(*)[NY][NX])pointer_twiddle;

	int i, j, k;
	for(k=0; k<d3; k++){
		for(j=0; j<d2; j++){
			for(i=0; i<d1; i++){
				u0[k][j][i] = dcomplex_mul2(u0[k][j][i], twiddle[k][j][i]);
				u1[k][j][i] = u0[k][j][i];
			}
		}
	}
}

static void fft(int dir,
		void* pointer_x1,
		void* pointer_x2){
	dcomplex y1[MAXDIM*FFTBLOCKPAD];
	dcomplex y2[MAXDIM*FFTBLOCKPAD];

	/*
	 * ---------------------------------------------------------------------
	 * note: args x1, x2 must be different arrays
	 * note: args for cfftsx are (direction, layout, xin, xout, scratch)
	 * xin/xout may be the same and it can be somewhat faster
	 * if they are
	 * ---------------------------------------------------------------------
	 */
	if(dir==1){
		cffts1(1, dims[0], dims[1], dims[2], pointer_x1, pointer_x1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y2);
		cffts2(1, dims[0], dims[1], dims[2], pointer_x1, pointer_x1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y2);
		cffts3(1, dims[0], dims[1], dims[2], pointer_x1, pointer_x2,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y2);
	}else{
		cffts3(-1, dims[0], dims[1], dims[2], pointer_x1, pointer_x1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y2);
		cffts2(-1, dims[0], dims[1], dims[2], pointer_x1, pointer_x1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y2);
		cffts1(-1, dims[0], dims[1], dims[2], pointer_x1, pointer_x2,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y2);
	}
}

/*
 * ---------------------------------------------------------------------
 * compute the roots-of-unity array that will be used for subsequent FFTs. 
 * ---------------------------------------------------------------------
 */
static void fft_init(int n){
	int m,nu,ku,i,j,ln;
	double t, ti;
	/*
	 * ---------------------------------------------------------------------
	 * initialize the U array with sines and cosines in a manner that permits
	 * stride one access at each FFT iteration.
	 * ---------------------------------------------------------------------	
	 */	
	nu = n;
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
}

/*
 * ---------------------------------------------------------------------
 * performs the l-th iteration of the second variant of the stockham FFT
 * ---------------------------------------------------------------------
 */
static void fftz2(int is,
		int l,
		int m,
		int n,
		int ny,
		int ny1,
		dcomplex u[],
		dcomplex x[][FFTBLOCKPAD],
		dcomplex y[][FFTBLOCKPAD]){
	int k,n1,li,lj,lk,ku,i,j,i11,i12,i21,i22;
	dcomplex u1,x11,x21;

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

	for(i=0; i<=li-1; i++){
		i11 = i * lk;
		i12 = i11 + n1;
		i21 = i * lj;
		i22 = i21 + lk;
		if(is>=1){
			u1 = u[ku+i];
		}else{
			u1 = dconjg(u[ku+i]);
		}

		/*
		 * ---------------------------------------------------------------------
		 * this loop is vectorizable.
		 * ---------------------------------------------------------------------
		 */
		for(k=0; k<=lk-1; k++){
			for(j=0; j<ny; j++){
				x11 = x[i11+k][j];
				x21 = x[i12+k][j];
				y[i21+k][j] = dcomplex_add(x11, x21);
				y[i22+k][j] = dcomplex_mul(u1, dcomplex_sub(x11, x21));
			}
		}
	}
}

static int ilog2(int n){
	int nn, lg;
	if(n==1){return 0;}
	lg = 1;
	nn = 2;
	while(nn<n){
		nn*=2;
		lg+=1;
	}
	return lg;
}

/*
 * ---------------------------------------------------------------------
 * touch all the big data
 * ---------------------------------------------------------------------  
 */
static void init_ui(void* pointer_u0,
		void* pointer_u1,
		void* pointer_twiddle,
		int d1,
		int d2,
		int d3){
	dcomplex (*u0)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u0;
	dcomplex (*u1)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u1;
	double (*twiddle)[NY][NX] = (double(*)[NY][NX])pointer_twiddle;

	int i, j, k;
	for(k=0; k<d3; k++){
		for(j=0; j<d2; j++){
			for(i=0; i<d1; i++){
				u0[k][j][i] = dcomplex_create(0.0, 0.0);
				u1[k][j][i] = dcomplex_create(0.0, 0.0);
				twiddle[k][j][i] = 0.0;
			}
		}
	}
}

/*
 * ---------------------------------------------------------------------
 * compute a^exponent mod 2^46
 * ---------------------------------------------------------------------
 */
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

	while(n > 1){
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

static void setup(){
	FILE* fp;

	niter = NITER_DEFAULT;

	printf("\n\n NAS Parallel Benchmarks 4.1 Serial C++ version - FT Benchmark\n\n");
	printf(" Size                : %4dx%4dx%4d\n", NX, NY, NZ);
	printf(" Iterations                  :%7d\n", niter);
	printf("\n");

	dims[0] = NX;
	dims[1] = NY;
	dims[2] = NZ;

	/*
	 * ---------------------------------------------------------------------
	 * set up info for blocking of ffts and transposes. this improves
	 * performance on cache-based systems. blocking involves
	 * working on a chunk of the problem at a time, taking chunks
	 * along the first, second, or third dimension. 
	 * 
	 * - in cffts1 blocking is on 2nd dimension (with fft on 1st dim)
	 * - in cffts2/3 blocking is on 1st dimension (with fft on 2nd and 3rd dims)
	 * 
	 * since 1st dim is always in processor, we'll assume it's long enough 
	 * (default blocking factor is 16 so min size for 1st dim is 16)
	 * the only case we have to worry about is cffts1 in a 2d decomposition. 
	 * so the blocking factor should not be larger than the 2nd dimension. 
	 * ---------------------------------------------------------------------
	 */
	/* block values were already set */
	/* fftblock = FFTBLOCK_DEFAULT; */
	/* fftblockpad = FFTBLOCKPAD_DEFAULT; */
	/* if(fftblock!=FFTBLOCK_DEFAULT){fftblockpad=fftblock+3;} */
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

static void cffts1_gpu(const int is, 
		MemoryObject u[], 
		MemoryObject x_in[], 
		MemoryObject x_out[], 
		MemoryObject y0[], 
		MemoryObject y1[]){
	int nx_aux = NX;
	int ny_aux = NY;
	int nz_aux = NZ;	

	auto gpu = driver->getGpu(0);

	try {
		/* kernel_fftx_1 */		
		kernel_fftx_1->clearParameters();
		kernel_fftx_1->setNumThreadsPerBlockForX(threads_per_block_on_fftx_1);
		kernel_fftx_1->setParameter(x_in);
		kernel_fftx_1->setParameter(y0);
		kernel_fftx_1->setParameter(sizeof(int), &nx_aux);
		kernel_fftx_1->setParameter(sizeof(int), &ny_aux);
		kernel_fftx_1->setParameter(sizeof(int), &nz_aux);

		unsigned long dimensions_on_kernel_fftx_1[3] = {(unsigned long)amount_of_work_on_fftx_1, 0, 0};
		kernel_fftx_1->runAsync(dimensions_on_kernel_fftx_1);
		kernel_fftx_1->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	try {	
		/* kernel_fftx_2 */
		kernel_fftx_2->clearParameters();
		kernel_fftx_2->setNumThreadsPerBlockForX(threads_per_block_on_fftx_2);
		kernel_fftx_2->setParameter(sizeof(int), &is);
		kernel_fftx_2->setParameter(y0);
		kernel_fftx_2->setParameter(y1);
		kernel_fftx_2->setParameter(u);
		kernel_fftx_2->setParameter(sizeof(int), &nx_aux);
		kernel_fftx_2->setParameter(sizeof(int), &ny_aux);
		kernel_fftx_2->setParameter(sizeof(int), &nz_aux);

		unsigned long dimensions_on_kernel_fftx_2[3] = {(unsigned long)amount_of_work_on_fftx_2, 0, 0};
		kernel_fftx_2->runAsync(dimensions_on_kernel_fftx_2);
		kernel_fftx_2->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	try {
		/* kernel_fftx_3 */
		kernel_fftx_3->clearParameters();
		kernel_fftx_3->setNumThreadsPerBlockForX(threads_per_block_on_fftx_3);
		kernel_fftx_3->setParameter(x_out);
		kernel_fftx_3->setParameter(y0);
		kernel_fftx_3->setParameter(sizeof(int), &nx_aux);
		kernel_fftx_3->setParameter(sizeof(int), &ny_aux);
		kernel_fftx_3->setParameter(sizeof(int), &nz_aux);

		unsigned long dimensions_on_kernel_fftx_3[3] = {(unsigned long)amount_of_work_on_fftx_3, 0, 0};
		kernel_fftx_3->runAsync(dimensions_on_kernel_fftx_3);
		kernel_fftx_3->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

static void cffts2_gpu(int is, 
		MemoryObject u[], 
		MemoryObject x_in[], 
		MemoryObject x_out[], 
		MemoryObject y0[], 
		MemoryObject y1[]){
	int nx_aux = NX;
	int ny_aux = NY;
	int nz_aux = NZ;

	try {
		/* kernel_ffty_1 */
		kernel_ffty_1->clearParameters();
		kernel_ffty_1->setNumThreadsPerBlockForX(threads_per_block_on_ffty_1);
		kernel_ffty_1->setParameter(x_in);
		kernel_ffty_1->setParameter(y0);
		kernel_ffty_1->setParameter(sizeof(int), &nx_aux);
		kernel_ffty_1->setParameter(sizeof(int), &ny_aux);
		kernel_ffty_1->setParameter(sizeof(int), &nz_aux);

		unsigned long dimensions_on_kernel_ffty_1[3] = {(unsigned long)amount_of_work_on_ffty_1, 0, 0};
		kernel_ffty_1->runAsync(dimensions_on_kernel_ffty_1);
		kernel_ffty_1->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	try {
		/* kernel_ffty_2 */
		kernel_ffty_2->clearParameters();
		kernel_ffty_2->setNumThreadsPerBlockForX(threads_per_block_on_ffty_2);
		kernel_ffty_2->setParameter(sizeof(int), &is);
		kernel_ffty_2->setParameter(y0);
		kernel_ffty_2->setParameter(y1);
		kernel_ffty_2->setParameter(u);
		kernel_ffty_2->setParameter(sizeof(int), &nx_aux);
		kernel_ffty_2->setParameter(sizeof(int), &ny_aux);
		kernel_ffty_2->setParameter(sizeof(int), &nz_aux);

		unsigned long dimensions_on_kernel_ffty_2[3] = {(unsigned long)amount_of_work_on_ffty_2, 0, 0};
		kernel_ffty_2->runAsync(dimensions_on_kernel_ffty_2);
		kernel_ffty_2->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	try {
		/* kernel_ffty_3 */
		kernel_ffty_3->clearParameters();
		kernel_ffty_3->setNumThreadsPerBlockForX(threads_per_block_on_ffty_3);
		kernel_ffty_3->setParameter(x_out);
		kernel_ffty_3->setParameter(y0);
		kernel_ffty_3->setParameter(sizeof(int), &nx_aux);
		kernel_ffty_3->setParameter(sizeof(int), &ny_aux);
		kernel_ffty_3->setParameter(sizeof(int), &nz_aux);

		unsigned long dimensions_on_kernel_ffty_3[3] = {(unsigned long)amount_of_work_on_ffty_3, 0, 0};
		kernel_ffty_3->runAsync(dimensions_on_kernel_ffty_3);
		kernel_ffty_3->waitAsync();

	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

static void cffts3_gpu(int is, 
		MemoryObject u[], 
		MemoryObject x_in[], 
		MemoryObject x_out[], 
		MemoryObject y0[], 
		MemoryObject y1[]){
	int nx_aux = NX;
	int ny_aux = NY;
	int nz_aux = NZ;

	try {
		/* kernel_fftz_1 */
		kernel_fftz_1->clearParameters();
		kernel_fftz_1->setNumThreadsPerBlockForX(threads_per_block_on_fftz_1);
		kernel_fftz_1->setParameter(x_in);
		kernel_fftz_1->setParameter(y0);
		kernel_fftz_1->setParameter(sizeof(int), &nx_aux);
		kernel_fftz_1->setParameter(sizeof(int), &ny_aux);
		kernel_fftz_1->setParameter(sizeof(int), &nz_aux);

		unsigned long dimensions_on_kernel_fftz_1[3] = {(unsigned long)amount_of_work_on_fftz_1, 0, 0};
		kernel_fftz_1->runAsync(dimensions_on_kernel_fftz_1);
		kernel_fftz_1->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	try {

		/* kernel_fftz_2 */
		kernel_fftz_2->clearParameters();
		kernel_fftz_2->clearParameters();
		kernel_fftz_2->setNumThreadsPerBlockForX(threads_per_block_on_fftz_2);
		kernel_fftz_2->setParameter(sizeof(int), &is);
		kernel_fftz_2->setParameter(y0);
		kernel_fftz_2->setParameter(y1);
		kernel_fftz_2->setParameter(u);
		kernel_fftz_2->setParameter(sizeof(int), &nx_aux);
		kernel_fftz_2->setParameter(sizeof(int), &ny_aux);
		kernel_fftz_2->setParameter(sizeof(int), &nz_aux);

		unsigned long dimensions_on_kernel_fftz_2[3] = {(unsigned long)amount_of_work_on_fftz_2, 0, 0};
		kernel_fftz_2->runAsync(dimensions_on_kernel_fftz_2);		kernel_fftz_2->waitAsync();

	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	try {
		/* kernel_fftz_3 */
		kernel_fftz_3->clearParameters();
		kernel_fftz_3->setNumThreadsPerBlockForX(threads_per_block_on_fftz_3);
		kernel_fftz_3->setParameter(x_out);
		kernel_fftz_3->setParameter(y0);
		kernel_fftz_3->setParameter(sizeof(int), &nx_aux);
		kernel_fftz_3->setParameter(sizeof(int), &ny_aux);
		kernel_fftz_3->setParameter(sizeof(int), &nz_aux);

		unsigned long dimensions_on_kernel_fftz_3[3] = {(unsigned long)amount_of_work_on_fftz_3, 0, 0};
		kernel_fftz_3->runAsync(dimensions_on_kernel_fftz_3);
		kernel_fftz_3->waitAsync();

	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

static void checksum_gpu(
		int iteration,
		MemoryObject u1[]){
	int nx_aux = NX;
	int ny_aux = NY;
	int nz_aux = NZ;
	int ntotal_aux = NTOTAL;

	try {	
		/* kernel_checksum */
		kernel_checksum->clearParameters();
		kernel_checksum->setNumThreadsPerBlockForX(32);
		kernel_checksum->setParameter(sizeof(int), &iteration);
		kernel_checksum->setParameter(u1);
		kernel_checksum->setParameter(sums_device);
		kernel_checksum->setParameter(sizeof(int), &nx_aux);
		kernel_checksum->setParameter(sizeof(int), &ny_aux);
		kernel_checksum->setParameter(sizeof(int), &nz_aux);
		kernel_checksum->setParameter(sizeof(int), &ntotal_aux);

		unsigned long dimensions_on_kernel_checksum[3] = {(unsigned long)CHECKSUM_TASKS, 0, 0};
		kernel_checksum->runAsync(dimensions_on_kernel_checksum);
		kernel_checksum->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

static void compute_indexmap_gpu(
		MemoryObject twiddle[]){
	int nx_aux = NX;
	int ny_aux = NY;
	int nz_aux = NZ;
	int ntotal_aux = NTOTAL;
	double ap_aux = AP;	

	try {
		/* kernel_compute_indexmap */
		kernel_compute_indexmap->clearParameters();
		kernel_compute_indexmap->setNumThreadsPerBlockForX(threads_per_block_on_compute_indexmap);
		kernel_compute_indexmap->setParameter(twiddle);
		kernel_compute_indexmap->setParameter(sizeof(int), &nx_aux);
		kernel_compute_indexmap->setParameter(sizeof(int), &ny_aux);
		kernel_compute_indexmap->setParameter(sizeof(int), &nz_aux);
		kernel_compute_indexmap->setParameter(sizeof(int), &ntotal_aux);
		kernel_compute_indexmap->setParameter(sizeof(double), &ap_aux);		

		unsigned long dimensions_on_kernel_compute_indexmap[3] = {(unsigned long)amount_of_work_on_compute_indexmap, 0, 0};
		kernel_compute_indexmap->runAsync(dimensions_on_kernel_compute_indexmap);
		kernel_compute_indexmap->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

static void compute_initial_conditions_gpu(
		MemoryObject u0[]){  
	int nx_aux = NX;
	int ny_aux = NY;
	int nz_aux = NZ;
	double a_aux = A;

	int z;
	double start, an;

	start = SEED;

	ipow46(A, 0, &an);
	randlc(&start, an);
	ipow46(A, 2*NX*NY, &an);

	starts[0] = start;
	for(z=1; z<NZ; z++){
		randlc(&start, an);
		starts[z] = start;
	}
	
	starts_device->copyIn();

	try {
		/* kernel_compute_initial_conditions */
		kernel_compute_initial_conditions->clearParameters();
		kernel_compute_initial_conditions->setNumThreadsPerBlockForX(threads_per_block_on_compute_initial_conditions);
		kernel_compute_initial_conditions->setParameter(u0);
		kernel_compute_initial_conditions->setParameter(starts_device);
		kernel_compute_initial_conditions->setParameter(sizeof(int), &nx_aux);
		kernel_compute_initial_conditions->setParameter(sizeof(int), &ny_aux);
		kernel_compute_initial_conditions->setParameter(sizeof(int), &nz_aux);
		kernel_compute_initial_conditions->setParameter(sizeof(double), &a_aux);

		unsigned long dimensions_on_kernel_compute_initial_conditions[3] = {(unsigned long)amount_of_work_on_compute_initial_conditions, 0, 0};
		kernel_compute_initial_conditions->runAsync(dimensions_on_kernel_compute_initial_conditions);
		kernel_compute_initial_conditions->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

static void evolve_gpu(
		MemoryObject u0[], 
		MemoryObject u1[],
		MemoryObject twiddle[]){
	int nx_aux = NX;
	int ny_aux = NY;
	int nz_aux = NZ;

	try {
		/* kernel_evolve */
		kernel_evolve->clearParameters();
		kernel_evolve->setNumThreadsPerBlockForX(threads_per_block_on_evolve);
		kernel_evolve->setParameter(u0);
		kernel_evolve->setParameter(u1);
		kernel_evolve->setParameter(twiddle);
		kernel_evolve->setParameter(sizeof(int), &nx_aux);
		kernel_evolve->setParameter(sizeof(int), &ny_aux);
		kernel_evolve->setParameter(sizeof(int), &nz_aux);

		unsigned long dimensions_on_kernel_evolve[3] = {(unsigned long)amount_of_work_on_evolve, 0, 0};
		kernel_evolve->runAsync(dimensions_on_kernel_evolve);
		kernel_evolve->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

static void fft_gpu(
		int dir,
		MemoryObject x1[],
		MemoryObject x2[]){
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
	
	u_device->copyIn();
}

static void init_ui_gpu(
		MemoryObject u0[],
		MemoryObject u1[],
		MemoryObject twiddle[]){
	int nx_aux = NX;
	int ny_aux = NY;
	int nz_aux = NZ;

	try {
		/* kernel_init_ui */
		kernel_init_ui->clearParameters();
		kernel_init_ui->setNumThreadsPerBlockForX(threads_per_block_on_init_ui);
		kernel_init_ui->setParameter(u0);
		kernel_init_ui->setParameter(u1);
		kernel_init_ui->setParameter(twiddle);
		kernel_init_ui->setParameter(sizeof(int), &nx_aux);
		kernel_init_ui->setParameter(sizeof(int), &ny_aux);
		kernel_init_ui->setParameter(sizeof(int), &nz_aux);

		unsigned long dimensions_on_kernel_init_ui[3] = {(unsigned long)amount_of_work_on_init_ui, 0, 0};
		kernel_init_ui->runAsync(dimensions_on_kernel_init_ui);
		kernel_init_ui->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	} 
}

static void setup_gpu(){
	driver = Instance::getInstance();
	driver->init();

	int numGpus = driver->getGpuCount();
	if (numGpus == 0) {
		std::cout << "No GPU found, interrupting the benchmark" << std::endl;
		exit(-1);
	}

	auto gpus = driver->getGpuList();

	DEVICE_NAME = gpus[0]->getName();	

	auto gpu = driver->getGpu(0);	

	threads_per_block_on_compute_indexmap = THREADS_PER_BLOCK;
	threads_per_block_on_compute_initial_conditions = THREADS_PER_BLOCK;
	threads_per_block_on_init_ui = THREADS_PER_BLOCK;
	threads_per_block_on_evolve = THREADS_PER_BLOCK;
	threads_per_block_on_fftx_1 = THREADS_PER_BLOCK;
	threads_per_block_on_fftx_2 = THREADS_PER_BLOCK;
	threads_per_block_on_fftx_3 = THREADS_PER_BLOCK;
	threads_per_block_on_ffty_1 = THREADS_PER_BLOCK;
	threads_per_block_on_ffty_2 = THREADS_PER_BLOCK;
	threads_per_block_on_ffty_3 = THREADS_PER_BLOCK;
	threads_per_block_on_fftz_1 = THREADS_PER_BLOCK;
	threads_per_block_on_fftz_2 = THREADS_PER_BLOCK;
	threads_per_block_on_fftz_3 = THREADS_PER_BLOCK;
	threads_per_block_on_checksum = THREADS_PER_BLOCK;

	amount_of_work_on_compute_indexmap = NX*NY*NZ;
	amount_of_work_on_compute_initial_conditions = NZ;
	amount_of_work_on_init_ui = NX*NY*NZ;
	amount_of_work_on_evolve = NX*NY*NZ;
	amount_of_work_on_fftx_1 = NX*NY*NZ;
	amount_of_work_on_fftx_2 = NY*NZ;
	amount_of_work_on_fftx_3 = NX*NY*NZ;
	amount_of_work_on_ffty_1 = NX*NY*NZ;
	amount_of_work_on_ffty_2 = NX*NZ;
	amount_of_work_on_ffty_3 = NX*NY*NZ;
	amount_of_work_on_fftz_1 = NX*NY*NZ;
	amount_of_work_on_fftz_2 = NX*NY;
	amount_of_work_on_fftz_3 = NX*NY*NZ;
	amount_of_work_on_checksum = CHECKSUM_TASKS;

	blocks_per_grid_on_compute_indexmap=ceil(double(NX*NY*NZ)/double(threads_per_block_on_compute_indexmap));
	blocks_per_grid_on_compute_initial_conditions=ceil(double(NZ)/double(threads_per_block_on_compute_initial_conditions));
	blocks_per_grid_on_init_ui=ceil(double(NX*NY*NZ)/double(threads_per_block_on_init_ui));
	blocks_per_grid_on_evolve=ceil(double(NX*NY*NZ)/double(threads_per_block_on_evolve));
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
	size_twiddle_device=NX*NY*NZ*sizeof(double);
	size_u_device=MAXDIM*sizeof(dcomplex);
	size_u0_device=NX*NY*NZ*sizeof(dcomplex);
	size_u1_device=NX*NY*NZ*sizeof(dcomplex);
	size_y0_device=NX*NY*NZ*sizeof(dcomplex);
	size_y1_device=NX*NY*NZ*sizeof(dcomplex);

	sums_device = gpu->malloc(size_sums_device, sums);
	starts_device = gpu->malloc(size_starts_device, starts);
	twiddle_device = gpu->malloc(size_twiddle_device, twiddle);
	u_device = gpu->malloc(size_u_device, u);
	u0_device = gpu->malloc(size_u0_device, u0);
	u1_device = gpu->malloc(size_u1_device, u1);
	y0_device = gpu->malloc(size_y0_device, y0_host);
	y1_device = gpu->malloc(size_y1_device, y1_host);

	for(int i=0; i<(NITER_DEFAULT+1); i++){
		dcomplex aux;
		aux.real = 0.0;
		aux.imag = 0.0;
		sums[i]=aux;
	}

	sums_device->copyIn();
	starts_device->copyIn();
	twiddle_device->copyIn();
	u_device->copyIn();
	u0_device->copyIn();
	u1_device->copyIn();
	y0_device->copyIn();
	y1_device->copyIn();

	try {
		std::string kernel_source_compute_indexmap_complete = "\n";
		kernel_source_compute_indexmap_complete.append(source_additional_routines_1);
		kernel_source_compute_indexmap_complete.append(source_additional_routines_2);
		kernel_source_compute_indexmap_complete.append(kernel_source_compute_indexmap);
		kernel_compute_indexmap = new Kernel(gpu, kernel_source_compute_indexmap_complete, "compute_indexmap_gpu_kernel");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_compute_initial_conditions_complete = "\n";
		kernel_source_compute_initial_conditions_complete.append(source_additional_routines_1);
		kernel_source_compute_initial_conditions_complete.append(source_additional_routines_2);
		kernel_source_compute_initial_conditions_complete.append(kernel_source_compute_initial_conditions);
		kernel_compute_initial_conditions = new Kernel(gpu, kernel_source_compute_initial_conditions_complete, "compute_initial_conditions_gpu_kernel");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_init_ui_complete = "\n";
		kernel_source_init_ui_complete.append(source_additional_routines_1);
		kernel_source_init_ui_complete.append(source_additional_routines_2);
		kernel_source_init_ui_complete.append(kernel_source_init_ui);
		kernel_init_ui = new Kernel(gpu, kernel_source_init_ui_complete, "init_ui_gpu_kernel");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_evolve_complete = "\n";
		kernel_source_evolve_complete.append(source_additional_routines_1);
		kernel_source_evolve_complete.append(source_additional_routines_2);
		kernel_source_evolve_complete.append(kernel_source_evolve);
		kernel_evolve = new Kernel(gpu, kernel_source_evolve_complete, "evolve_gpu_kernel");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	try {
		std::string kernel_source_fftx_1_complete = "\n";
		kernel_source_fftx_1_complete.append(source_additional_routines_1);
		kernel_source_fftx_1_complete.append(source_additional_routines_2);
		kernel_source_fftx_1_complete.append(kernel_source_fftx_1);
		kernel_fftx_1 = new Kernel(gpu, kernel_source_fftx_1_complete, "cffts1_gpu_kernel_1");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_fftx_2_complete = "\n";
		kernel_source_fftx_2_complete.append(source_additional_routines_1);
		kernel_source_fftx_2_complete.append(source_additional_routines_2);
		kernel_source_fftx_2_complete.append(kernel_source_fftx_2);
		kernel_fftx_2 = new Kernel(gpu, kernel_source_fftx_2_complete, "cffts1_gpu_kernel_2");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_fftx_3_complete = "\n";
		kernel_source_fftx_3_complete.append(source_additional_routines_1);
		kernel_source_fftx_3_complete.append(source_additional_routines_2);
		kernel_source_fftx_3_complete.append(kernel_source_fftx_3);
		kernel_fftx_3 = new Kernel(gpu, kernel_source_fftx_3_complete, "cffts1_gpu_kernel_3");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_ffty_1_complete = "\n";
		kernel_source_ffty_1_complete.append(source_additional_routines_1);
		kernel_source_ffty_1_complete.append(source_additional_routines_2);
		kernel_source_ffty_1_complete.append(kernel_source_ffty_1);
		kernel_ffty_1 = new Kernel(gpu, kernel_source_ffty_1_complete, "cffts2_gpu_kernel_1");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_ffty_2_complete = "\n";
		kernel_source_ffty_2_complete.append(source_additional_routines_1);
		kernel_source_ffty_2_complete.append(source_additional_routines_2);
		kernel_source_ffty_2_complete.append(kernel_source_ffty_2);
		kernel_ffty_2 = new Kernel(gpu, kernel_source_ffty_2_complete, "cffts2_gpu_kernel_2");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_ffty_3_complete = "\n";
		kernel_source_ffty_3_complete.append(source_additional_routines_1);
		kernel_source_ffty_3_complete.append(source_additional_routines_2);
		kernel_source_ffty_3_complete.append(kernel_source_ffty_3);
		kernel_ffty_3 = new Kernel(gpu, kernel_source_ffty_3_complete, "cffts2_gpu_kernel_3");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_fftz_1_complete = "\n";
		kernel_source_fftz_1_complete.append(source_additional_routines_1);
		kernel_source_fftz_1_complete.append(source_additional_routines_2);
		kernel_source_fftz_1_complete.append(kernel_source_fftz_1);
		kernel_fftz_1 = new Kernel(gpu, kernel_source_fftz_1_complete, "cffts3_gpu_kernel_1");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_fftz_2_complete = "\n";
		kernel_source_fftz_2_complete.append(source_additional_routines_1);
		kernel_source_fftz_2_complete.append(source_additional_routines_2);
		kernel_source_fftz_2_complete.append(kernel_source_fftz_2);
		kernel_fftz_2 = new Kernel(gpu, kernel_source_fftz_2_complete, "cffts3_gpu_kernel_2");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_fftz_3_complete = "\n";
		kernel_source_fftz_3_complete.append(source_additional_routines_1);
		kernel_source_fftz_3_complete.append(source_additional_routines_2);
		kernel_source_fftz_3_complete.append(kernel_source_fftz_3);
		kernel_fftz_3 = new Kernel(gpu, kernel_source_fftz_3_complete, "cffts3_gpu_kernel_3");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_checksum_complete = "\n";
		kernel_source_checksum_complete.append(source_additional_routines_1);
		kernel_source_checksum_complete.append(source_additional_routines_2);
		kernel_source_checksum_complete.append(kernel_source_checksum);
		kernel_checksum = new Kernel(gpu, kernel_source_checksum_complete, "checksum_gpu_kernel");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}			
}

std::string kernel_source_fftx_1 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void cffts1_gpu_kernel_1(
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex x_in[], 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex y0[],
		int NX,
		int NY,
		int NZ){
	//BEGIN 

	int x_y_z = gspar_get_global_id(0);
	if(x_y_z >= (NX*NY*NZ)){
		return;
	}
	int x = x_y_z % NX;
	int y = (x_y_z / NX) % NY;
	int z = x_y_z / (NX * NY);
	y0[y+(x*NY)+(z*NX*NY)].real = x_in[x_y_z].real;
	y0[y+(x*NY)+(z*NX*NY)].imag = x_in[x_y_z].imag;

	//END
});

std::string kernel_source_fftx_2 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void cffts1_gpu_kernel_2(
		const int is, 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex gty1[], 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex gty2[], 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex u_device[],
		int NX,
		int NY,
		int NZ){	
	// BEGIN

	int y_z = gspar_get_global_id(0);

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

	// END
});

std::string kernel_source_fftx_3 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void cffts1_gpu_kernel_3(
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex x_out[], 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex y0[],
		int NX,
		int NY,
		int NZ){
	// BEGIN

	int x_y_z = gspar_get_global_id(0);
	if(x_y_z >= (NX*NY*NZ)){
		return;
	}
	int x = x_y_z % NX;
	int y = (x_y_z / NX) % NY;
	int z = x_y_z / (NX * NY);
	x_out[x_y_z].real = y0[y+(x*NY)+(z*NX*NY)].real;
	x_out[x_y_z].imag = y0[y+(x*NY)+(z*NX*NY)].imag;

	// END
});

std::string kernel_source_ffty_1 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void cffts2_gpu_kernel_1(
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex x_in[], 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex y0[],
		int NX,
		int NY,
		int NZ){
	// BEGIN

	int x_y_z = gspar_get_global_id(0);
	if(x_y_z >= (NX*NY*NZ)){
		return;
	}
	y0[x_y_z].real = x_in[x_y_z].real;
	y0[x_y_z].imag = x_in[x_y_z].imag;

	// END
});

std::string kernel_source_ffty_2 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void cffts2_gpu_kernel_2(
		const int is, 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex gty1[], 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex gty2[], 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex u_device[],
		int NX,
		int NY,
		int NZ){
	// BEGIN

	int x_z = gspar_get_global_id(0);

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

	// END
});

std::string kernel_source_ffty_3 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void cffts2_gpu_kernel_3(
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex x_out[], 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex y0[],
		int NX,
		int NY,
		int NZ){
	// BEGIN

	int x_y_z = gspar_get_global_id(0);
	if(x_y_z >= (NX*NY*NZ)){
		return;
	}
	x_out[x_y_z].real = y0[x_y_z].real;
	x_out[x_y_z].imag = y0[x_y_z].imag;

	//END
});

std::string kernel_source_fftz_1 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void cffts3_gpu_kernel_1(
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex x_in[], 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex y0[],
		int NX,
		int NY,
		int NZ){
	// BEGIN

	int x_y_z = gspar_get_global_id(0);
	if(x_y_z >= (NX*NY*NZ)){
		return;
	}
	y0[x_y_z].real = x_in[x_y_z].real;
	y0[x_y_z].imag = x_in[x_y_z].imag;

	//END
});

std::string kernel_source_fftz_2 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void cffts3_gpu_kernel_2(
		const int is, 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex gty1[], 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex gty2[], 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex u_device[],
		int NX,
		int NY,
		int NZ){
	// BEGIN

	int x_y = gspar_get_global_id(0);
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

	//END
});

std::string kernel_source_fftz_3 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void cffts3_gpu_kernel_3(
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex x_out[], 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex y0[],
		int NX,
		int NY,
		int NZ){
	// BEGIN

	int x_y_z = gspar_get_global_id(0);
	if(x_y_z >= (NX*NY*NZ)){
		return;
	}
	x_out[x_y_z].real = y0[x_y_z].real;
	x_out[x_y_z].imag = y0[x_y_z].imag;

	// END
});

std::string kernel_source_checksum = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void checksum_gpu_kernel(
		int iteration, 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex u1[], 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex sums[],
		int NX,
		int NY,
		int NZ,
		int NTOTAL){
		// BEGIN
		
	int thread_id = gspar_get_global_id(0);
	
	GSPAR_DEVICE_SHARED_MEMORY dcomplex share_sums[THREADS_PER_BLOCK];
	int j = thread_id + 1;
	int q, r, s;

	if(j<=CHECKSUM_TASKS){
		q = j % NX;
		r = 3*j % NY;
		s = 5*j % NZ;
		share_sums[gspar_get_thread_id(0)] = u1[ q + r*NX + s*NX*NY ];
	}else{
		dcomplex aux;
		aux.real = 0.0;
		aux.imag = 0.0;
		share_sums[gspar_get_thread_id(0)] = aux;
	}

	gspar_synchronize_local_threads();
	for(int i=gspar_get_block_size(0)/2; i>0; i>>=1){
		if(gspar_get_thread_id(0)<i){
			share_sums[gspar_get_thread_id(0)] = 
			dcomplex_add(share_sums[gspar_get_thread_id(0)], share_sums[gspar_get_thread_id(0)+i]);
		}
		gspar_synchronize_local_threads();
	}
	if(gspar_get_thread_id(0)==0){
		share_sums[0].real = share_sums[0].real/(double)(NTOTAL);
		gspar_atomic_add_double(&sums[iteration].real,share_sums[0].real);
		share_sums[0].imag = share_sums[0].imag/(double)(NTOTAL);
		gspar_atomic_add_double(&sums[iteration].imag,share_sums[0].imag);			
	}	

	// END
});

std::string kernel_source_compute_indexmap = GSPAR_STRINGIZE_SOURCE(
/* gpu kernel */
GSPAR_DEVICE_KERNEL void compute_indexmap_gpu_kernel(
		GSPAR_DEVICE_GLOBAL_MEMORY double twiddle[],
		int NX,
		int NY,
		int NZ,		
		int NTOTAL,
		double AP){

	// BEGIN

	int thread_id = gspar_get_global_id(0);

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

	// END
});

std::string kernel_source_compute_initial_conditions = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void compute_initial_conditions_gpu_kernel(
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex u0[], 
		GSPAR_DEVICE_GLOBAL_MEMORY double starts[],
		int NX,
		int NY,
		int NZ,
		double A){ 
	// BEGIN

	int z = gspar_get_global_id(0);

	if(z>=NZ){return;}

	double x0 = starts[z];	
	for(int y=0; y<NY; y++){
		vranlc_device(2*NX, &x0, A, (double*)&u0[ 0 + y*NX + z*NX*NY ]);
	}

	//END
});

std::string kernel_source_evolve = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void evolve_gpu_kernel(
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex u0[], 
		GSPAR_DEVICE_GLOBAL_MEMORY dcomplex u1[],
		GSPAR_DEVICE_GLOBAL_MEMORY double twiddle[],
		int NX,
		int NY,
		int NZ){
	// BEGIN

	int thread_id = gspar_get_global_id(0);

	if(thread_id>=(NZ*NY*NX)){
		return;
	}	

	u0[thread_id] = dcomplex_mul2(u0[thread_id], twiddle[thread_id]);
	u1[thread_id] = u0[thread_id];

	// END
});

std::string kernel_source_init_ui = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL
void init_ui_gpu_kernel(
	GSPAR_DEVICE_GLOBAL_MEMORY dcomplex u0[],
	GSPAR_DEVICE_GLOBAL_MEMORY dcomplex u1[],
	GSPAR_DEVICE_GLOBAL_MEMORY double twiddle[],
	int NX,
	int NY,
	int NZ){
	// BEGIN

	int thread_id = gspar_get_global_id(0);

	if(thread_id>=(NX*NY*NZ)){
		return;
	}

	dcomplex aux;
	aux.real = 0.0;
	aux.imag = 0.0;
	u0[thread_id] = aux;
	u1[thread_id] = aux;
	twiddle[thread_id] = 0.0;

	// END
});

std::string source_additional_routines_1 = 
"\n"
"#define CHECKSUM_TASKS 1024\n"
"#define WARP_SIZE 32\n"
"#define MAX_THREADS_PER_BLOCK 1024\n"
"#define THREADS_PER_BLOCK 32\n"
"\n";

std::string source_additional_routines_2 = GSPAR_STRINGIZE_SOURCE(
/* dcomplex struct */
typedef struct tdcomplex { double real; double imag; } dcomplex;
/* dcomplex multiplication */
dcomplex dcomplex_mul2(
	dcomplex a, 
	double b){
	dcomplex aux;
	aux.real = a.real*b;
	aux.imag = a.imag*b;
	return aux;
}
/* dcomplex add */
dcomplex dcomplex_add(
	dcomplex a, 
	dcomplex b){
	dcomplex aux;
	aux.real = a.real + b.real;
	aux.imag = a.imag + b.imag;
	return aux;
}
/* ilog function */
GSPAR_DEVICE_FUNCTION int ilog2_device(int n){
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
/* fftz2 function */
GSPAR_DEVICE_FUNCTION void cffts3_gpu_fftz2_device(
	const int is, 
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
/* cfftz function */
GSPAR_DEVICE_FUNCTION void cffts3_gpu_cfftz_device(
	const int is, 
	int m, 
	int n, 
	dcomplex x[], 
	dcomplex y[], 
	dcomplex u_device[], 
	int index_arg, 
	int size_arg){
	int j,l;
	for(l=1; l<=m; l+=2){
		cffts3_gpu_fftz2_device(is, l, m, n, u_device, x, y, index_arg, size_arg);
		if(l==m){break;}
		cffts3_gpu_fftz2_device(is, l + 1, m, n, u_device, y, x, index_arg, size_arg);
	}
	if(m%2==1){
		for(j=0; j<n; j++){
			x[j*size_arg+index_arg].real = y[j*size_arg+index_arg].real;
			x[j*size_arg+index_arg].imag = y[j*size_arg+index_arg].imag;
		}
	}
}
/* vranlc function */
GSPAR_DEVICE_FUNCTION void vranlc_device(
	int n, 
	double* x_seed, 
	double a, 
	double y[]){
	const double R23 = (0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5);
	const double R46 = ((0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5)*(0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5));
	const double T23 = (2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0);
	const double T46 = ((2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0)*(2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0));
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
);
