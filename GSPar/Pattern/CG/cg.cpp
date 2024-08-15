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
 *      M. Yarrow
 *      C. Kuszmaul
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
 * 
 * How to run:
 *      export LD_LIBRARY_PATH=../lib/gspar/bin:$LD_LIBRARY_PATH
 *      clear && make clean && make cg CLASS=S GPU_DRIVER=CUDA && bin/cg.S 
 *      clear && make clean && make cg CLASS=S GPU_DRIVER=OPENCL && bin/cg.S 
 * 
 * ------------------------------------------------------------------------------
 */

#include <iostream>
#include <chrono>

#include "../common/npb.hpp"
#include "npbparams.hpp"

#include "GSPar_PatternMap.hpp"
#include "GSPar_PatternReduce.hpp"
#include "GSPar_PatternComposition.hpp"

#ifdef GSPARDRIVER_CUDA
#include "GSPar_CUDA.hpp"
using namespace GSPar::Driver::CUDA;
#else
/* GSPARDRIVER_OPENCL */
#include "GSPar_OpenCL.hpp"
using namespace GSPar::Driver::OpenCL;
#endif

using namespace GSPar::Pattern;
using namespace std;

/*
 * ---------------------------------------------------------------------
 * note: please observe that in the routine conj_grad three 
 * implementations of the sparse matrix-vector multiply have
 * been supplied. the default matrix-vector multiply is not
 * loop unrolled. the alternate implementations are unrolled
 * to a depth of 2 and unrolled to a depth of 8. please
 * experiment with these to find the fastest for your particular
 * architecture. if reporting timing results, any of these three may
 * be used without penalty.
 * ---------------------------------------------------------------------
 * class specific parameters: 
 * it appears here for reference only.
 * these are their values, however, this info is imported in the npbparams.h
 * include file, which is written by the sys/setparams.c program.
 * ---------------------------------------------------------------------
 */
#define NZ (NA*(NONZER+1)*(NONZER+1))
#define NAZ (NA*(NONZER+1))
#define PROFILING_TOTAL_TIME (0)
#define CG_THREADS_PER_BLOCK_ON_KERNEL_ONE 1024
#define CG_THREADS_PER_BLOCK_ON_KERNEL_TWO 256
#define CG_THREADS_PER_BLOCK_ON_KERNEL_THREE 64
#define CG_THREADS_PER_BLOCK_ON_KERNEL_FOUR 256
#define CG_THREADS_PER_BLOCK_ON_KERNEL_FIVE 64
#define CG_THREADS_PER_BLOCK_ON_KERNEL_SIX 256
#define CG_THREADS_PER_BLOCK_ON_KERNEL_SEVEN 512
#define CG_THREADS_PER_BLOCK_ON_KERNEL_EIGHT 64
#define CG_THREADS_PER_BLOCK_ON_KERNEL_NINE 512
#define CG_THREADS_PER_BLOCK_ON_KERNEL_TEN 256
#define CG_THREADS_PER_BLOCK_ON_KERNEL_ELEVEN 512

/* global variables */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static int colidx[NZ];
static int rowstr[NA+1];
static int iv[NA];
static int arow[NA];
static int acol[NAZ];
static double aelt[NAZ];
static double a[NZ];
static double x[NA+2];
static double z[NA+2];
static double p[NA+2];
static double q[NA+2];
static double r[NA+2];
#else
static int (*colidx)=(int*)malloc(sizeof(int)*(NZ));
static int (*rowstr)=(int*)malloc(sizeof(int)*(NA+1));
static int (*iv)=(int*)malloc(sizeof(int)*(NA));
static int (*arow)=(int*)malloc(sizeof(int)*(NA));
static int (*acol)=(int*)malloc(sizeof(int)*(NAZ));
static double (*aelt)=(double*)malloc(sizeof(double)*(NAZ));
static double (*a)=(double*)malloc(sizeof(double)*(NZ));
static double (*x)=(double*)malloc(sizeof(double)*(NA+2));
static double (*z)=(double*)malloc(sizeof(double)*(NA+2));
static double (*p)=(double*)malloc(sizeof(double)*(NA+2));
static double (*q)=(double*)malloc(sizeof(double)*(NA+2));
static double (*r)=(double*)malloc(sizeof(double)*(NA+2));
#endif
static int naa;
static int nzz;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;
static double amult;
static double tran;

/* gpu data */
/* gpu patterns */
Map* kernel_one_map;
Map* kernel_two_map;
Map* kernel_three_map;
Map* kernel_four_map;
Map* kernel_five_map;
Map* kernel_six_map;
Map* kernel_seven_map;
Map* kernel_eight_map;
Map* kernel_nine_map;
Map* kernel_ten1_map;
Map* kernel_ten2_map;
Map* kernel_eleven_map;
/* gpu kernel sources */
extern std::string source_kernel_one_map;
extern std::string source_kernel_two_map;
extern std::string source_kernel_three_map;
extern std::string source_kernel_four_map;
extern std::string source_kernel_five_map;
extern std::string source_kernel_six_map;
extern std::string source_kernel_seven_map;
extern std::string source_kernel_eight_map;
extern std::string source_kernel_nine_map;
extern std::string source_kernel_ten1_map;
extern std::string source_kernel_ten2_map;
extern std::string source_kernel_eleven_map;
extern std::string source_extra_code;
/* gpu driver */
Instance* driver;
/* gpu name */
string DEVICE_NAME;
/* gpu memory objects */
MemoryObject* colidx_device;
MemoryObject* rowstr_device;
MemoryObject* a_device;
MemoryObject* p_device;
MemoryObject* q_device;
MemoryObject* r_device;
MemoryObject* x_device;
MemoryObject* z_device;
MemoryObject* global_data_device;
MemoryObject* global_data_two_device;
/* gpu parameters */
int amount_of_work_on_kernel_one;
int amount_of_work_on_kernel_two;
int amount_of_work_on_kernel_three;
int amount_of_work_on_kernel_four;
int amount_of_work_on_kernel_five;
int amount_of_work_on_kernel_six;
int amount_of_work_on_kernel_seven;
int amount_of_work_on_kernel_eight;
int amount_of_work_on_kernel_nine;
int amount_of_work_on_kernel_ten;
int amount_of_work_on_kernel_eleven;
int blocks_per_grid_on_kernel_one;
int blocks_per_grid_on_kernel_two;
int blocks_per_grid_on_kernel_three;
int blocks_per_grid_on_kernel_four;
int blocks_per_grid_on_kernel_five;
int blocks_per_grid_on_kernel_six;
int blocks_per_grid_on_kernel_seven;
int blocks_per_grid_on_kernel_eight;
int blocks_per_grid_on_kernel_nine;
int blocks_per_grid_on_kernel_ten;
int blocks_per_grid_on_kernel_eleven;
int threads_per_block_on_kernel_one;
int threads_per_block_on_kernel_two;
int threads_per_block_on_kernel_three;
int threads_per_block_on_kernel_four;
int threads_per_block_on_kernel_five;
int threads_per_block_on_kernel_six;
int threads_per_block_on_kernel_seven;
int threads_per_block_on_kernel_eight;
int threads_per_block_on_kernel_nine;
int threads_per_block_on_kernel_ten;
int threads_per_block_on_kernel_eleven;
/* other gpu variables */
int global_data_elements;
size_t size_global_data;
size_t size_colidx_device;
size_t size_rowstr_device;
size_t size_iv_device;
size_t size_arow_device;
size_t size_acol_device;
size_t size_aelt_device;
size_t size_a_device;
size_t size_x_device;
size_t size_z_device;
size_t size_p_device;
size_t size_q_device;
size_t size_r_device;
extern std::string source_additional_routines;
/* other stuff */
double* global_data;
double* global_data_two;

/* function prototypes */
static void conj_grad(int colidx[],
		int rowstr[],
		double x[],
		double z[],
		double a[],
		double p[],
		double q[],
		double r[],
		double* rnorm);
static int icnvrt(double x,
		int ipwr2);
static void makea(int n,
		int nz,
		double a[],
		int colidx[],
		int rowstr[],
		int firstrow,
		int lastrow,
		int firstcol,
		int lastcol,
		int arow[],
		int acol[][NONZER+1],
		double aelt[][NONZER+1],
		int iv[]);
static void sparse(double a[],
		int colidx[],
		int rowstr[],
		int n,
		int nz,
		int nozer,
		int arow[],
		int acol[][NONZER+1],
		double aelt[][NONZER+1],
		int firstrow,
		int lastrow,
		int nzloc[],
		double rcond,
		double shift);
static void sprnvc(int n,
		int nz,
		int nn1,
		double v[],
		int iv[]);
static void vecset(int n,
		double v[],
		int iv[],
		int* nzv,
		int i,
		double val);
static void setup_gpu();
static void conj_grad_gpu(int colidx[],
		int rowstr[],
		double x[],
		double z[],
		double a[],
		double p[],
		double q[],
		double r[],
		double* rnorm);

/* gpu function prototypes */
void kernel_one(double* q, 
		double* z, 
		double* r, 
		double* p,
		double* x);
double kernel_two(double* global_data,
		double* r);
void kernel_three(int* rowstr, 
		double* a, 
		double* p, 
		int* colidx,
		double* q);
double kernel_four(double* global_data,
		double* p,
		double* q);
void kernel_five(double* z, 
		double* r, 
		double* p, 
		double* q,
		double alpha);
double kernel_six(double* buffer,
		double* r);
void kernel_seven(double* p, 
		double* r,
		double beta);
void kernel_eight(int* rowstr, 
		double* a, 
		double* z, 
		int* colidx,
		double* r);
double kernel_nine(double* buffer,
		double* x,
		double* r);
double kernel_ten_1(double* buffer,
		double* x,
		double* z);
double kernel_ten_2(double* buffer,
		double* z);
void kernel_eleven(double* x, 
		double* z, 
		double norm_temp2);

/* cg */
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
	int	i, j, k, it;
	double zeta;
	double rnorm;
	double norm_temp1, norm_temp2;
	double t, mflops, tmax;
	char class_npb;
	boolean verified;
	double zeta_verify_value, epsilon, err;

	firstrow = 0;
	lastrow  = NA-1;
	firstcol = 0;
	lastcol  = NA-1;

	if(NA == 1400 && NONZER == 7 && NITER == 15 && SHIFT == 10.0){
		class_npb = 'S';
		zeta_verify_value = 8.5971775078648;
	}else if(NA == 7000 && NONZER == 8 && NITER == 15 && SHIFT == 12.0){
		class_npb = 'W';
		zeta_verify_value = 10.362595087124;
	}else if(NA == 14000 && NONZER == 11 && NITER == 15 && SHIFT == 20.0){
		class_npb = 'A';
		zeta_verify_value = 17.130235054029;
	}else if(NA == 75000 && NONZER == 13 && NITER == 75 && SHIFT == 60.0){
		class_npb = 'B';
		zeta_verify_value = 22.712745482631;
	}else if(NA == 150000 && NONZER == 15 && NITER == 75 && SHIFT == 110.0){
		class_npb = 'C';
		zeta_verify_value = 28.973605592845;
	}else if(NA == 1500000 && NONZER == 21 && NITER == 100 && SHIFT == 500.0){
		class_npb = 'D';
		zeta_verify_value = 52.514532105794;
	}else if(NA == 9000000 && NONZER == 26 && NITER == 100 && SHIFT == 1500.0){
		class_npb = 'E';
		zeta_verify_value = 77.522164599383;
	}else{
		class_npb = 'U';
	}

	printf("\n\n NAS Parallel Benchmarks 4.1 Serial C++ version - CG Benchmark\n\n");
	printf(" Size: %11d\n", NA);
	printf(" Iterations: %5d\n", NITER);

	naa = NA;
	nzz = NZ;

	/* initialize random number generator */
	tran    = 314159265.0;
	amult   = 1220703125.0;
	zeta    = randlc( &tran, amult );

	makea(naa, 
			nzz, 
			a, 
			colidx, 
			rowstr, 
			firstrow, 
			lastrow, 
			firstcol, 
			lastcol, 
			arow, 
			(int(*)[NONZER+1])(void*)acol, 
			(double(*)[NONZER+1])(void*)aelt,
			iv);

	/*
	 * ---------------------------------------------------------------------
	 * note: as a result of the above call to makea:
	 * values of j used in indexing rowstr go from 0 --> lastrow-firstrow
	 * values of colidx which are col indexes go from firstcol --> lastcol
	 * so:
	 * shift the col index vals from actual (firstcol --> lastcol) 
	 * to local, i.e., (0 --> lastcol-firstcol)
	 * ---------------------------------------------------------------------
	 */
	for(j = 0; j < lastrow - firstrow + 1; j++){
		for(k = rowstr[j]; k < rowstr[j+1]; k++){
			colidx[k] = colidx[k] - firstcol;
		}
	}

	/* set starting vector to (1, 1, .... 1) */
	for(i = 0; i < NA+1; i++){
		x[i] = 1.0;
	}
	for(j = 0; j<lastcol-firstcol+1; j++){
		q[j] = 0.0;
		z[j] = 0.0;
		r[j] = 0.0;
		p[j] = 0.0;
	}
	zeta = 0.0;

	/*
	 * -------------------------------------------------------------------
	 * ---->
	 * do one iteration untimed to init all code and data page tables
	 * ----> (then reinit, start timing, to niter its)
	 * -------------------------------------------------------------------*/
	for(it = 1; it <= 1; it++){
		/* the call to the conjugate gradient routine */
		conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);

		/*
		 * --------------------------------------------------------------------
		 * zeta = shift + 1/(x.z)
		 * so, first: (x.z)
		 * also, find norm of z
		 * so, first: (z.z)
		 * --------------------------------------------------------------------
		 */
		norm_temp1 = 0.0;
		norm_temp2 = 0.0;
		for(j = 0; j < lastcol - firstcol + 1; j++){
			norm_temp1 = norm_temp1 + x[j] * z[j];
			norm_temp2 = norm_temp2 + z[j] * z[j];
		}
		norm_temp2 = 1.0 / sqrt(norm_temp2);

		/* normalize z to obtain x */
		for(j = 0; j < lastcol - firstcol + 1; j++){     
			x[j] = norm_temp2 * z[j];
		}
	} /* end of do one iteration untimed */

	/* set starting vector to (1, 1, .... 1) */	
	for(i = 0; i < NA+1; i++){
		x[i] = 1.0;
	}
	zeta = 0.0;

	setup_gpu();

	timer_clear(PROFILING_TOTAL_TIME);
	timer_start(PROFILING_TOTAL_TIME);

	/*
	 * --------------------------------------------------------------------
	 * ---->
	 * main iteration for inverse power method
	 * ---->
	 * --------------------------------------------------------------------
	 */
	for(it = 1; it <= NITER; it++){
		/* the call to the conjugate gradient routine */
		conj_grad_gpu(colidx, rowstr, x, z, a, p, q, r, &rnorm);

		/*
		 * --------------------------------------------------------------------
		 * zeta = shift + 1/(x.z)
		 * so, first: (x.z)
		 * also, find norm of z
		 * so, first: (z.z)
		 * --------------------------------------------------------------------
		 */		
		/* kernel ten */
		norm_temp1 = kernel_ten_1(global_data, x, z);
		norm_temp2 = kernel_ten_2(global_data, z);

		norm_temp2 = (double) 1.0 / (double) sqrt(norm_temp2);
		zeta = (double) SHIFT + (double) 1.0 / (double) norm_temp1;
		if(it==1){printf("\n   iteration           ||r||                 zeta\n");}
		printf("    %5d       %20.14e%20.13e\n", it, rnorm, zeta);

		/* normalize z to obtain x */
		kernel_eleven(x, r, norm_temp2);
	} /* end of main iter inv pow meth */

	/*
	 * --------------------------------------------------------------------
	 * end of timed section
	 * --------------------------------------------------------------------
	 */
	timer_stop(PROFILING_TOTAL_TIME);
	t = timer_read(PROFILING_TOTAL_TIME);

	printf(" Benchmark completed\n");

	epsilon = 1.0e-10;
	if(class_npb != 'U'){
		err = fabs(zeta - zeta_verify_value) / zeta_verify_value;
		if(err <= epsilon){
			verified = TRUE;
			printf(" VERIFICATION SUCCESSFUL\n");
			printf(" Zeta is    %20.13e\n", zeta);
			printf(" Error is   %20.13e\n", err);
		}else{
			verified = FALSE;
			printf(" VERIFICATION FAILED\n");
			printf(" Zeta                %20.13e\n", zeta);
			printf(" The correct zeta is %20.13e\n", zeta_verify_value);
		}
	}else{
		verified = FALSE;
		printf(" Problem size unknown\n");
		printf(" NO VERIFICATION PERFORMED\n");
	}
	if(t != 0.0){
		mflops = (double)(2.0*NITER*NA)
			* (3.0+(double)(NONZER*(NONZER+1))
					+ 25.0
					* (5.0+(double)(NONZER*(NONZER+1)))+3.0)
			/ t / 1000000.0;
	}else{
		mflops = 0.0;
	}
	c_print_results((char*)"CG",
			class_npb,
			NA,
			0,
			0,
			NITER,
			t,
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

/*
 * ---------------------------------------------------------------------
 * floating point arrays here are named as in NPB1 spec discussion of 
 * CG algorithm
 * ---------------------------------------------------------------------
 */
static void conj_grad(int colidx[],
		int rowstr[],
		double x[],
		double z[],
		double a[],
		double p[],
		double q[],
		double r[],
		double* rnorm){
	int j, k;
	int cgit, cgitmax;
	double d, sum, rho, rho0, alpha, beta;

	cgitmax = 25;

	rho = 0.0;

	/* initialize the CG algorithm */
	for(j = 0; j < naa+1; j++){
		q[j] = 0.0;
		z[j] = 0.0;
		r[j] = x[j];
		p[j] = r[j];
	}

	/*
	 * --------------------------------------------------------------------
	 * rho = r.r
	 * now, obtain the norm of r: First, sum squares of r elements locally...
	 * --------------------------------------------------------------------
	 */
	for(j = 0; j < lastcol - firstcol + 1; j++){
		rho = rho + r[j]*r[j];
	}

	/* the conj grad iteration loop */
	for(cgit = 1; cgit <= cgitmax; cgit++){
		/*
		 * ---------------------------------------------------------------------
		 * q = A.p
		 * the partition submatrix-vector multiply: use workspace w
		 * ---------------------------------------------------------------------
		 * 
		 * note: this version of the multiply is actually (slightly: maybe %5) 
		 * faster on the sp2 on 16 nodes than is the unrolled-by-2 version 
		 * below. on the Cray t3d, the reverse is TRUE, i.e., the 
		 * unrolled-by-two version is some 10% faster.  
		 * the unrolled-by-8 version below is significantly faster
		 * on the Cray t3d - overall speed of code is 1.5 times faster.
		 */
		for(j = 0; j < lastrow - firstrow + 1; j++){
			sum = 0.0;
			for(k = rowstr[j]; k < rowstr[j+1]; k++){
				sum = sum + a[k]*p[colidx[k]];
			}
			q[j] = sum;
		}

		/*
		 * --------------------------------------------------------------------
		 * obtain p.q
		 * --------------------------------------------------------------------
		 */
		d = 0.0;
		for (j = 0; j < lastcol - firstcol + 1; j++) {
			d = d + p[j]*q[j];
		}

		/*
		 * --------------------------------------------------------------------
		 * obtain alpha = rho / (p.q)
		 * -------------------------------------------------------------------
		 */
		alpha = rho / d;

		/*
		 * --------------------------------------------------------------------
		 * save a temporary of rho
		 * --------------------------------------------------------------------
		 */
		rho0 = rho;

		/*
		 * ---------------------------------------------------------------------
		 * obtain z = z + alpha*p
		 * and    r = r - alpha*q
		 * ---------------------------------------------------------------------
		 */
		rho = 0.0;
		for(j = 0; j < lastcol - firstcol + 1; j++){
			z[j] = z[j] + alpha*p[j];
			r[j] = r[j] - alpha*q[j];
		}

		/*
		 * ---------------------------------------------------------------------
		 * rho = r.r
		 * now, obtain the norm of r: first, sum squares of r elements locally...
		 * ---------------------------------------------------------------------
		 */
		for(j = 0; j < lastcol - firstcol + 1; j++){
			rho = rho + r[j]*r[j];
		}

		/*
		 * ---------------------------------------------------------------------
		 * obtain beta
		 * ---------------------------------------------------------------------
		 */
		beta = rho / rho0;

		/*
		 * ---------------------------------------------------------------------
		 * p = r + beta*p
		 * ---------------------------------------------------------------------
		 */
		for(j = 0; j < lastcol - firstcol + 1; j++){
			p[j] = r[j] + beta*p[j];
		}
	} /* end of do cgit=1, cgitmax */

	/*
	 * ---------------------------------------------------------------------
	 * compute residual norm explicitly: ||r|| = ||x - A.z||
	 * first, form A.z
	 * the partition submatrix-vector multiply
	 * ---------------------------------------------------------------------
	 */
	sum = 0.0;
	for(j = 0; j < lastrow - firstrow + 1; j++){
		d = 0.0;
		for(k = rowstr[j]; k < rowstr[j+1]; k++){
			d = d + a[k]*z[colidx[k]];
		}
		r[j] = d;
	}

	/*
	 * ---------------------------------------------------------------------
	 * at this point, r contains A.z
	 * ---------------------------------------------------------------------
	 */
	for(j = 0; j < lastcol-firstcol+1; j++){
		d   = x[j] - r[j];
		sum = sum + d*d;
	}

	*rnorm = sqrt(sum);
}

/*
 * ---------------------------------------------------------------------
 * scale a double precision number x in (0,1) by a power of 2 and chop it
 * ---------------------------------------------------------------------
 */
static int icnvrt(double x, int ipwr2){
	return (int)(ipwr2 * x);
}

/*
 * ---------------------------------------------------------------------
 * generate the test problem for benchmark 6
 * makea generates a sparse matrix with a
 * prescribed sparsity distribution
 *
 * parameter    type        usage
 *
 * input
 *
 * n            i           number of cols/rows of matrix
 * nz           i           nonzeros as declared array size
 * rcond        r*8         condition number
 * shift        r*8         main diagonal shift
 *
 * output
 *
 * a            r*8         array for nonzeros
 * colidx       i           col indices
 * rowstr       i           row pointers
 *
 * workspace
 *
 * iv, arow, acol i
 * aelt           r*8
 * ---------------------------------------------------------------------
 */
static void makea(int n,
		int nz,
		double a[],
		int colidx[],
		int rowstr[],
		int firstrow,
		int lastrow,
		int firstcol,
		int lastcol,
		int arow[],
		int acol[][NONZER+1],
		double aelt[][NONZER+1],
		int iv[]){
	int iouter, ivelt, nzv, nn1;
	int ivc[NONZER+1];
	double vc[NONZER+1];

	/*
	 * --------------------------------------------------------------------
	 * nonzer is approximately  (int(sqrt(nnza /n)));
	 * --------------------------------------------------------------------
	 * nn1 is the smallest power of two not less than n
	 * --------------------------------------------------------------------
	 */
	nn1 = 1;
	do{
		nn1 = 2 * nn1;
	}while(nn1 < n);

	/*
	 * -------------------------------------------------------------------
	 * generate nonzero positions and save for the use in sparse
	 * -------------------------------------------------------------------
	 */
	for(iouter = 0; iouter < n; iouter++){
		nzv = NONZER;
		sprnvc(n, nzv, nn1, vc, ivc);
		vecset(n, vc, ivc, &nzv, iouter+1, 0.5);
		arow[iouter] = nzv;
		for(ivelt = 0; ivelt < nzv; ivelt++){
			acol[iouter][ivelt] = ivc[ivelt] - 1;
			aelt[iouter][ivelt] = vc[ivelt];
		}
	}

	/*
	 * ---------------------------------------------------------------------
	 * ... make the sparse matrix from list of elements with duplicates
	 * (iv is used as  workspace)
	 * ---------------------------------------------------------------------
	 */
	sparse(a,
			colidx,
			rowstr,
			n,
			nz,
			NONZER,
			arow,
			acol,
			aelt,
			firstrow,
			lastrow,
			iv,
			RCOND,
			SHIFT);
}

/*
 * ---------------------------------------------------------------------
 * rows range from firstrow to lastrow
 * the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
 * ---------------------------------------------------------------------
 */
static void sparse(double a[],
		int colidx[],
		int rowstr[],
		int n,
		int nz,
		int nozer,
		int arow[],
		int acol[][NONZER+1],
		double aelt[][NONZER+1],
		int firstrow,
		int lastrow,
		int nzloc[],
		double rcond,
		double shift){	
	int nrows;

	/*
	 * ---------------------------------------------------
	 * generate a sparse matrix from a list of
	 * [col, row, element] tri
	 * ---------------------------------------------------
	 */
	int i, j, j1, j2, nza, k, kk, nzrow, jcol;
	double size, scale, ratio, va;
	boolean goto_40;

	/*
	 * --------------------------------------------------------------------
	 * how many rows of result
	 * --------------------------------------------------------------------
	 */
	nrows = lastrow - firstrow + 1;

	/*
	 * --------------------------------------------------------------------
	 * ...count the number of triples in each row
	 * --------------------------------------------------------------------
	 */
	for(j = 0; j < nrows+1; j++){
		rowstr[j] = 0;
	}
	for(i = 0; i < n; i++){
		for(nza = 0; nza < arow[i]; nza++){
			j = acol[i][nza] + 1;
			rowstr[j] = rowstr[j] + arow[i];
		}
	}
	rowstr[0] = 0;
	for(j = 1; j < nrows+1; j++){
		rowstr[j] = rowstr[j] + rowstr[j-1];
	}
	nza = rowstr[nrows] - 1;

	/*
	 * ---------------------------------------------------------------------
	 * ... rowstr(j) now is the location of the first nonzero
	 * of row j of a
	 * ---------------------------------------------------------------------
	 */
	if(nza > nz){
		printf("Space for matrix elements exceeded in sparse\n");
		printf("nza, nzmax = %d, %d\n", nza, nz);
		exit(EXIT_FAILURE);
	}

	/*
	 * ---------------------------------------------------------------------
	 * ... preload data pages
	 * ---------------------------------------------------------------------
	 */
	for(j = 0; j < nrows; j++){
		for(k = rowstr[j]; k < rowstr[j+1]; k++){
			a[k] = 0.0;
			colidx[k] = -1;
		}
		nzloc[j] = 0;
	}

	/*
	 * ---------------------------------------------------------------------
	 * ... generate actual values by summing duplicates
	 * ---------------------------------------------------------------------
	 */
	size = 1.0;
	ratio = pow(rcond, (1.0 / (double)(n)));
	for(i = 0; i < n; i++){
		for(nza = 0; nza < arow[i]; nza++){
			j = acol[i][nza];

			scale = size * aelt[i][nza];
			for(nzrow = 0; nzrow < arow[i]; nzrow++){
				jcol = acol[i][nzrow];
				va = aelt[i][nzrow] * scale;

				/*
				 * --------------------------------------------------------------------
				 * ... add the identity * rcond to the generated matrix to bound
				 * the smallest eigenvalue from below by rcond
				 * --------------------------------------------------------------------
				 */
				if(jcol == j && j == i){
					va = va + rcond - shift;
				}

				goto_40 = FALSE;
				for(k = rowstr[j]; k < rowstr[j+1]; k++){
					if(colidx[k] > jcol){
						/*
						 * ----------------------------------------------------------------
						 * ... insert colidx here orderly
						 * ----------------------------------------------------------------
						 */
						for(kk = rowstr[j+1]-2; kk >= k; kk--){
							if(colidx[kk] > -1){
								a[kk+1] = a[kk];
								colidx[kk+1] = colidx[kk];
							}
						}
						colidx[k] = jcol;
						a[k]  = 0.0;
						goto_40 = TRUE;
						break;
					}else if(colidx[k] == -1){
						colidx[k] = jcol;
						goto_40 = TRUE;
						break;
					}else if(colidx[k] == jcol){
						/*
						 * --------------------------------------------------------------
						 * ... mark the duplicated entry
						 * -------------------------------------------------------------
						 */
						nzloc[j] = nzloc[j] + 1;
						goto_40 = TRUE;
						break;
					}
				}
				if(goto_40 == FALSE){
					printf("internal error in sparse: i=%d\n", i);
					exit(EXIT_FAILURE);
				}
				a[k] = a[k] + va;
			}
		}
		size = size * ratio;
	}

	/*
	 * ---------------------------------------------------------------------
	 * ... remove empty entries and generate final results
	 * ---------------------------------------------------------------------
	 */
	for(j = 1; j < nrows; j++){
		nzloc[j] = nzloc[j] + nzloc[j-1];
	}

	for(j = 0; j < nrows; j++){
		if(j > 0){
			j1 = rowstr[j] - nzloc[j-1];
		}else{
			j1 = 0;
		}
		j2 = rowstr[j+1] - nzloc[j];
		nza = rowstr[j];
		for(k = j1; k < j2; k++){
			a[k] = a[nza];
			colidx[k] = colidx[nza];
			nza = nza + 1;
		}
	}
	for(j = 1; j < nrows+1; j++){
		rowstr[j] = rowstr[j] - nzloc[j-1];
	}
	nza = rowstr[nrows] - 1;
}

/*
 * ---------------------------------------------------------------------
 * generate a sparse n-vector (v, iv)
 * having nzv nonzeros
 *
 * mark(i) is set to 1 if position i is nonzero.
 * mark is all zero on entry and is reset to all zero before exit
 * this corrects a performance bug found by John G. Lewis, caused by
 * reinitialization of mark on every one of the n calls to sprnvc
 * ---------------------------------------------------------------------
 */
static void sprnvc(int n, int nz, int nn1, double v[], int iv[]){
	int nzv, ii, i;
	double vecelt, vecloc;

	nzv = 0;

	while(nzv < nz){
		vecelt = randlc(&tran, amult);

		/*
		 * --------------------------------------------------------------------
		 * generate an integer between 1 and n in a portable manner
		 * --------------------------------------------------------------------
		 */
		vecloc = randlc(&tran, amult);
		i = icnvrt(vecloc, nn1) + 1;
		if(i>n){continue;}

		/*
		 * --------------------------------------------------------------------
		 * was this integer generated already?
		 * --------------------------------------------------------------------
		 */
		boolean was_gen = FALSE;
		for(ii = 0; ii < nzv; ii++){
			if(iv[ii] == i){
				was_gen = TRUE;
				break;
			}
		}
		if(was_gen){continue;}
		v[nzv] = vecelt;
		iv[nzv] = i;
		nzv = nzv + 1;
	}
}

/*
 * --------------------------------------------------------------------
 * set ith element of sparse vector (v, iv) with
 * nzv nonzeros to val
 * --------------------------------------------------------------------
 */
static void vecset(int n, double v[], int iv[], int* nzv, int i, double val){
	int k;
	boolean set;

	set = FALSE;
	for(k = 0; k < *nzv; k++){
		if(iv[k] == i){
			v[k] = val;
			set  = TRUE;
		}
	}
	if(set == FALSE){
		v[*nzv]  = val;
		iv[*nzv] = i;
		*nzv     = *nzv + 1;
	}
}

/* gpu setup */
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
	
	threads_per_block_on_kernel_one=CG_THREADS_PER_BLOCK_ON_KERNEL_ONE;
	threads_per_block_on_kernel_two=CG_THREADS_PER_BLOCK_ON_KERNEL_TWO;
	threads_per_block_on_kernel_three=CG_THREADS_PER_BLOCK_ON_KERNEL_THREE;
	threads_per_block_on_kernel_four=CG_THREADS_PER_BLOCK_ON_KERNEL_FOUR;
	threads_per_block_on_kernel_five=CG_THREADS_PER_BLOCK_ON_KERNEL_FIVE;
	threads_per_block_on_kernel_six=CG_THREADS_PER_BLOCK_ON_KERNEL_SIX;
	threads_per_block_on_kernel_seven=CG_THREADS_PER_BLOCK_ON_KERNEL_SEVEN;
	threads_per_block_on_kernel_eight=CG_THREADS_PER_BLOCK_ON_KERNEL_EIGHT;
	threads_per_block_on_kernel_nine=CG_THREADS_PER_BLOCK_ON_KERNEL_NINE;
	threads_per_block_on_kernel_ten=CG_THREADS_PER_BLOCK_ON_KERNEL_TEN;
	threads_per_block_on_kernel_eleven=CG_THREADS_PER_BLOCK_ON_KERNEL_ELEVEN;

	amount_of_work_on_kernel_one = (ceil(double(NA)/double(CG_THREADS_PER_BLOCK_ON_KERNEL_ONE))*CG_THREADS_PER_BLOCK_ON_KERNEL_ONE);
	amount_of_work_on_kernel_two = (ceil(double(NA)/double(CG_THREADS_PER_BLOCK_ON_KERNEL_TWO))*CG_THREADS_PER_BLOCK_ON_KERNEL_TWO);
	amount_of_work_on_kernel_three = NA * threads_per_block_on_kernel_three;
	amount_of_work_on_kernel_four = (ceil(double(NA)/double(CG_THREADS_PER_BLOCK_ON_KERNEL_FOUR))*CG_THREADS_PER_BLOCK_ON_KERNEL_FOUR);
	amount_of_work_on_kernel_five = (ceil(double(NA)/double(CG_THREADS_PER_BLOCK_ON_KERNEL_FIVE))*CG_THREADS_PER_BLOCK_ON_KERNEL_FIVE);
	amount_of_work_on_kernel_six = (ceil(double(NA)/double(CG_THREADS_PER_BLOCK_ON_KERNEL_SIX))*CG_THREADS_PER_BLOCK_ON_KERNEL_SIX);
	amount_of_work_on_kernel_seven = (ceil(double(NA)/double(CG_THREADS_PER_BLOCK_ON_KERNEL_SEVEN))*CG_THREADS_PER_BLOCK_ON_KERNEL_SEVEN);
	amount_of_work_on_kernel_eight = NA * threads_per_block_on_kernel_eight;
	amount_of_work_on_kernel_nine = (ceil(double(NA)/double(CG_THREADS_PER_BLOCK_ON_KERNEL_NINE))*CG_THREADS_PER_BLOCK_ON_KERNEL_NINE);
	amount_of_work_on_kernel_ten = (ceil(double(NA)/double(CG_THREADS_PER_BLOCK_ON_KERNEL_TEN))*CG_THREADS_PER_BLOCK_ON_KERNEL_TEN);
	amount_of_work_on_kernel_eleven = (ceil(double(NA)/double(CG_THREADS_PER_BLOCK_ON_KERNEL_ELEVEN))*CG_THREADS_PER_BLOCK_ON_KERNEL_ELEVEN);

	blocks_per_grid_on_kernel_one=(ceil((double)NA/(double)threads_per_block_on_kernel_one));
	blocks_per_grid_on_kernel_two=(ceil((double)NA/(double)threads_per_block_on_kernel_two));   
	blocks_per_grid_on_kernel_three=NA;
	blocks_per_grid_on_kernel_four=(ceil((double)NA/(double)threads_per_block_on_kernel_four));
	blocks_per_grid_on_kernel_five=(ceil((double)NA/(double)threads_per_block_on_kernel_five));
	blocks_per_grid_on_kernel_six=(ceil((double)NA/(double)threads_per_block_on_kernel_six));
	blocks_per_grid_on_kernel_seven=(ceil((double)NA/threads_per_block_on_kernel_seven));
	blocks_per_grid_on_kernel_eight=NA;
	blocks_per_grid_on_kernel_nine=(ceil((double)NA/(double)threads_per_block_on_kernel_nine));
	blocks_per_grid_on_kernel_ten=(ceil((double)NA/(double)threads_per_block_on_kernel_ten));
	blocks_per_grid_on_kernel_eleven=(ceil((double)NA/(double)threads_per_block_on_kernel_eleven));

	global_data_elements=ceil(double(NA)/double(gpu->getWarpSize()));

	size_global_data=global_data_elements*sizeof(double);
	size_colidx_device=NZ*sizeof(int);
	size_rowstr_device=(NA+1)*sizeof(int);
	size_iv_device=NA*sizeof(int);
	size_arow_device=NA*sizeof(int);
	size_acol_device=NAZ*sizeof(int);
	size_aelt_device=NAZ*sizeof(double);
	size_a_device=NZ*sizeof(double);
	size_x_device=(NA+2)*sizeof(double);
	size_z_device=(NA+2)*sizeof(double);
	size_p_device=(NA+2)*sizeof(double);
	size_q_device=(NA+2)*sizeof(double);
	size_r_device=(NA+2)*sizeof(double);

	size_global_data=global_data_elements*sizeof(double);
	size_colidx_device=NZ*sizeof(int);
	size_rowstr_device=(NA+1)*sizeof(int);
	size_iv_device=NA*sizeof(int);
	size_arow_device=NA*sizeof(int);
	size_acol_device=NAZ*sizeof(int);
	size_aelt_device=NAZ*sizeof(double);
	size_a_device=NZ*sizeof(double);
	size_x_device=(NA+2)*sizeof(double);
	size_z_device=(NA+2)*sizeof(double);
	size_p_device=(NA+2)*sizeof(double);
	size_q_device=(NA+2)*sizeof(double);
	size_r_device=(NA+2)*sizeof(double);

	global_data=(double*)malloc(size_global_data);
	global_data_two=(double*)malloc(size_global_data);	

	for(int i=0; i<global_data_elements; i++){
		global_data[i] = 0.0;
		global_data_two[i] = 0.0;
	}

	colidx_device = gpu->malloc(size_colidx_device, colidx);
	rowstr_device = gpu->malloc(size_rowstr_device, rowstr);
	a_device = gpu->malloc(size_a_device, a);
	p_device = gpu->malloc(size_p_device, p);
	q_device = gpu->malloc(size_q_device, q);
	r_device = gpu->malloc(size_r_device, r);
	x_device = gpu->malloc(size_x_device, x);
	z_device = gpu->malloc(size_z_device, z);	

	global_data_device = gpu->malloc(size_global_data, global_data);
	global_data_two_device = gpu->malloc(size_global_data, global_data_two);

	colidx_device->copyIn();
	rowstr_device->copyIn();
	a_device->copyIn();
	p_device->copyIn();
	q_device->copyIn();
	r_device->copyIn();
	x_device->copyIn();
	z_device->copyIn();	
	global_data_device->copyIn();
	global_data_two_device->copyIn();

	double alpha, beta, norm_temp2;

	/* compiling each pattern */
	/* kernel one */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_kernel_one, 0, 0}; 
		kernel_one_map = new Map(source_kernel_one_map);
		kernel_one_map->setStdVarNames({"gspar_thread_id"});
		kernel_one_map->setParameter<double*>("q", q_device, GSPAR_PARAM_PRESENT);
		kernel_one_map->setParameter<double*>("z", z_device, GSPAR_PARAM_PRESENT);
		kernel_one_map->setParameter<double*>("r", r_device, GSPAR_PARAM_PRESENT);
		kernel_one_map->setParameter<double*>("p", p_device, GSPAR_PARAM_PRESENT);
		kernel_one_map->setParameter<double*>("x", x_device, GSPAR_PARAM_PRESENT);	
		kernel_one_map->setParameter("NA", NA);
		kernel_one_map->setNumThreadsPerBlockForX(threads_per_block_on_kernel_one);
		kernel_one_map->addExtraKernelCode(source_additional_routines);
		kernel_one_map->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel two */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_kernel_two, 0, 0};
		kernel_two_map = new Map(source_kernel_two_map);
		kernel_two_map->setStdVarNames({"gspar_thread_id"});
		kernel_two_map->setParameter<double*>("r", r_device, GSPAR_PARAM_PRESENT);
		kernel_two_map->setParameter<double*>("global_data", global_data_device, GSPAR_PARAM_PRESENT);
		kernel_two_map->setParameter("NA", NA);
		kernel_two_map->setNumThreadsPerBlockForX(threads_per_block_on_kernel_two);
		kernel_two_map->addExtraKernelCode(source_additional_routines);
		kernel_two_map->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel three */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_kernel_three, 0, 0};
		kernel_three_map = new Map(source_kernel_three_map);
		kernel_three_map->setStdVarNames({"gspar_thread_id"});
		kernel_three_map->setParameter<int*>("rowstr", rowstr_device, GSPAR_PARAM_PRESENT);
		kernel_three_map->setParameter<double*>("a", a_device, GSPAR_PARAM_PRESENT);
		kernel_three_map->setParameter<double*>("p", p_device, GSPAR_PARAM_PRESENT);
		kernel_three_map->setParameter<int*>("colidx", colidx_device, GSPAR_PARAM_PRESENT);
		kernel_three_map->setParameter<double*>("q", q_device, GSPAR_PARAM_PRESENT);		
		kernel_three_map->setParameter("NA", NA);	
		kernel_three_map->setNumThreadsPerBlockForX(threads_per_block_on_kernel_three);
		kernel_three_map->addExtraKernelCode(source_additional_routines);
		kernel_three_map->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel four */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_kernel_four, 0, 0};
		kernel_four_map = new Map(source_kernel_four_map);
		kernel_four_map->setStdVarNames({"gspar_thread_id"});		
		kernel_four_map->setParameter<double*>("p", p_device, GSPAR_PARAM_PRESENT);
		kernel_four_map->setParameter<double*>("q", q_device, GSPAR_PARAM_PRESENT);
		kernel_four_map->setParameter<double*>("global_data", global_data_device, GSPAR_PARAM_PRESENT);
		kernel_four_map->setParameter("NA", NA);	
		kernel_four_map->setNumThreadsPerBlockForX(threads_per_block_on_kernel_four);
		kernel_four_map->addExtraKernelCode(source_additional_routines);
		kernel_four_map->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel five */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_kernel_five, 0, 0}; 
		kernel_five_map = new Map(source_kernel_five_map);
		kernel_five_map->setStdVarNames({"gspar_thread_id"});			
		kernel_five_map->setParameter<double*>("p", p_device, GSPAR_PARAM_PRESENT);
		kernel_five_map->setParameter<double*>("q", q_device, GSPAR_PARAM_PRESENT);
		kernel_five_map->setParameter<double*>("r", r_device, GSPAR_PARAM_PRESENT);
		kernel_five_map->setParameter<double*>("z", z_device, GSPAR_PARAM_PRESENT);		
		kernel_five_map->setParameter("alpha", alpha);
		kernel_five_map->setParameter("NA", NA);	
		kernel_five_map->setNumThreadsPerBlockForX(threads_per_block_on_kernel_five);
		kernel_five_map->addExtraKernelCode(source_additional_routines);
		kernel_five_map->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel six */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_kernel_six, 0, 0}; 
		kernel_six_map = new Map(source_kernel_six_map);
		kernel_six_map->setStdVarNames({"gspar_thread_id"});
		kernel_six_map->setParameter<double*>("r", r_device, GSPAR_PARAM_PRESENT);
		kernel_six_map->setParameter<double*>("global_data", global_data_device, GSPAR_PARAM_PRESENT);
		kernel_six_map->setParameter("NA", NA);	
		kernel_six_map->setNumThreadsPerBlockForX(threads_per_block_on_kernel_six);
		kernel_six_map->addExtraKernelCode(source_additional_routines);
		kernel_six_map->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel seven */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_kernel_seven, 0, 0}; 
		kernel_seven_map = new Map(source_kernel_seven_map);
		kernel_seven_map->setStdVarNames({"gspar_thread_id"});	
		kernel_seven_map->setParameter<double*>("p", p_device, GSPAR_PARAM_PRESENT);
		kernel_seven_map->setParameter<double*>("r", r_device, GSPAR_PARAM_PRESENT);
		kernel_seven_map->setParameter("beta", beta);
		kernel_seven_map->setParameter("NA", NA);	
		kernel_seven_map->setNumThreadsPerBlockForX(threads_per_block_on_kernel_seven);
		kernel_seven_map->addExtraKernelCode(source_additional_routines);
		kernel_seven_map->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel eight */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_kernel_eight, 0, 0}; 
		kernel_eight_map = new Map(source_kernel_eight_map);
		kernel_eight_map->setStdVarNames({"gspar_thread_id"});		
		kernel_eight_map->setParameter<int*>("rowstr", rowstr_device, GSPAR_PARAM_PRESENT);
		kernel_eight_map->setParameter<double*>("a", a_device, GSPAR_PARAM_PRESENT);
		kernel_eight_map->setParameter<double*>("z", z_device, GSPAR_PARAM_PRESENT);
		kernel_eight_map->setParameter<int*>("colidx", colidx_device, GSPAR_PARAM_PRESENT);
		kernel_eight_map->setParameter<double*>("r", r_device, GSPAR_PARAM_PRESENT);
		kernel_eight_map->setParameter("NA", NA);	
		kernel_eight_map->setNumThreadsPerBlockForX(threads_per_block_on_kernel_eight);
		kernel_eight_map->addExtraKernelCode(source_additional_routines);
		kernel_eight_map->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel nine */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_kernel_nine, 0, 0}; 
		kernel_nine_map = new Map(source_kernel_nine_map);
		kernel_nine_map->setStdVarNames({"gspar_thread_id"});			
		kernel_nine_map->setParameter<double*>("r", r_device, GSPAR_PARAM_PRESENT);
		kernel_nine_map->setParameter<double*>("x", x_device, GSPAR_PARAM_PRESENT);
		kernel_nine_map->setParameter<double*>("global_data", global_data_device, GSPAR_PARAM_PRESENT);
		kernel_nine_map->setParameter("NA", NA);	
		kernel_nine_map->setNumThreadsPerBlockForX(threads_per_block_on_kernel_nine);
		kernel_nine_map->addExtraKernelCode(source_additional_routines);
		kernel_nine_map->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel ten1 */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_kernel_ten, 0, 0}; 
		kernel_ten1_map = new Map(source_kernel_ten1_map);
		kernel_ten1_map->setStdVarNames({"gspar_thread_id"});		
		kernel_ten1_map->setParameter<double*>("x", x_device, GSPAR_PARAM_PRESENT);
		kernel_ten1_map->setParameter<double*>("z", z_device, GSPAR_PARAM_PRESENT);
		kernel_ten1_map->setParameter<double*>("norm_temp1", global_data_device, GSPAR_PARAM_PRESENT);
		kernel_ten1_map->setParameter<double*>("norm_temp2", global_data_two_device, GSPAR_PARAM_PRESENT);
		kernel_ten1_map->setParameter("NA", NA);	
		kernel_ten1_map->setNumThreadsPerBlockForX(threads_per_block_on_kernel_ten);
		kernel_ten1_map->addExtraKernelCode(source_additional_routines);
		kernel_ten1_map->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel ten2 */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_kernel_ten, 0, 0};
		kernel_ten2_map = new Map(source_kernel_ten2_map);
		kernel_ten2_map->setStdVarNames({"gspar_thread_id"});			
		kernel_ten2_map->setParameter<double*>("x", x_device, GSPAR_PARAM_PRESENT);
		kernel_ten2_map->setParameter<double*>("z", z_device, GSPAR_PARAM_PRESENT);
		kernel_ten2_map->setParameter<double*>("norm_temp1", global_data_device, GSPAR_PARAM_PRESENT);
		kernel_ten2_map->setParameter<double*>("norm_temp2", global_data_two_device, GSPAR_PARAM_PRESENT);
		kernel_ten2_map->setParameter("NA", NA);	
		kernel_ten2_map->setNumThreadsPerBlockForX(threads_per_block_on_kernel_ten);
		kernel_ten2_map->addExtraKernelCode(source_additional_routines);
		kernel_ten2_map->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/* kernel eleven */
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work_on_kernel_eleven, 0, 0}; 
		kernel_eleven_map = new Map(source_kernel_eleven_map);
		kernel_eleven_map->setStdVarNames({"gspar_thread_id"});			
		kernel_eleven_map->setParameter<double*>("x", x_device, GSPAR_PARAM_PRESENT);
		kernel_eleven_map->setParameter<double*>("z", z_device, GSPAR_PARAM_PRESENT);
		kernel_eleven_map->setParameter("norm_temp2", norm_temp2);	
		kernel_eleven_map->setParameter("NA", NA);	
		kernel_eleven_map->setNumThreadsPerBlockForX(threads_per_block_on_kernel_eleven);
		kernel_eleven_map->addExtraKernelCode(source_additional_routines);
		kernel_eleven_map->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

/*
 * ---------------------------------------------------------------------
 * floating point arrays here are named as in NPB1 spec discussion of 
 * CG algorithm
 * ---------------------------------------------------------------------
 */
static void conj_grad_gpu(int colidx[],
		int rowstr[],
		double x[],
		double z[],
		double a[],
		double p[],
		double q[],
		double r[],
		double* rnorm){
	int j, k;
	int cgit, cgitmax;
	double d, sum, rho, rho0, alpha, beta;

	cgitmax = 25;

	/* initialize the CG algorithm */
	kernel_one(q, z, r, p, x);
	
	/*
	 * --------------------------------------------------------------------
	 * rho = r.r
	 * now, obtain the norm of r: First, sum squares of r elements locally...
	 * --------------------------------------------------------------------
	 */
	rho = kernel_two(global_data, r);

	/* the conj grad iteration loop */
	for(cgit = 1; cgit <= cgitmax; cgit++){
		/*
		 * ---------------------------------------------------------------------
		 * q = A.p
		 * the partition submatrix-vector multiply: use workspace w
		 * ---------------------------------------------------------------------
		 * 
		 * note: this version of the multiply is actually (slightly: maybe %5) 
		 * faster on the sp2 on 16 nodes than is the unrolled-by-2 version 
		 * below. on the Cray t3d, the reverse is TRUE, i.e., the 
		 * unrolled-by-two version is some 10% faster.  
		 * the unrolled-by-8 version below is significantly faster
		 * on the Cray t3d - overall speed of code is 1.5 times faster.
		 */
		kernel_three(rowstr, a, p, colidx, q);

		/*
		 * --------------------------------------------------------------------
		 * obtain p.q
		 * --------------------------------------------------------------------
		 */
		d = kernel_four(global_data, p, q);

		/*
		 * --------------------------------------------------------------------
		 * obtain alpha = rho / (p.q)
		 * -------------------------------------------------------------------
		 */
		alpha = rho / d;

		/*
		 * --------------------------------------------------------------------
		 * save a temporary of rho
		 * --------------------------------------------------------------------
		 */
		rho0 = rho;

		/*
		 * ---------------------------------------------------------------------
		 * obtain z = z + alpha*p
		 * and    r = r - alpha*q
		 * ---------------------------------------------------------------------
		 */
		kernel_five(z, r, p, q, alpha);

		/*
		 * ---------------------------------------------------------------------
		 * rho = r.r
		 * now, obtain the norm of r: first, sum squares of r elements locally...
		 * ---------------------------------------------------------------------
		 */
		rho = kernel_six(global_data, r);

		/*
		 * ---------------------------------------------------------------------
		 * obtain beta
		 * ---------------------------------------------------------------------
		 */
		beta = rho / rho0;

		/*
		 * ---------------------------------------------------------------------
		 * p = r + beta*p
		 * ---------------------------------------------------------------------
		 */
		kernel_seven(p, r, beta);
	} /* end of do cgit=1, cgitmax */

	/*
	 * ---------------------------------------------------------------------
	 * compute residual norm explicitly: ||r|| = ||x - A.z||
	 * first, form A.z
	 * the partition submatrix-vector multiply
	 * ---------------------------------------------------------------------
	 */	
	kernel_eight(rowstr, a, z, colidx, r);

	/*
	 * ---------------------------------------------------------------------
	 * at this point, r contains A.z
	 * ---------------------------------------------------------------------
	 */
	sum = kernel_nine(global_data, x, r);

	*rnorm = sqrt(sum);
}

/* initialize the CG algorithm */
void kernel_one(double* q, 
		double* z, 
		double* r, 
		double* p,
		double* x){
	try {
		kernel_one_map->setParameter<double*>("q", q_device, GSPAR_PARAM_PRESENT);
		kernel_one_map->setParameter<double*>("z", z_device, GSPAR_PARAM_PRESENT);
		kernel_one_map->setParameter<double*>("r", r_device, GSPAR_PARAM_PRESENT);
		kernel_one_map->setParameter<double*>("p", p_device, GSPAR_PARAM_PRESENT);
		kernel_one_map->setParameter<double*>("x", x_device, GSPAR_PARAM_PRESENT);	
		kernel_one_map->setParameter("NA", NA);	

		kernel_one_map->run<Instance>();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

/* obtain the norm of r */
double kernel_two(double* global_data,
		double* r){
	try {
		kernel_two_map->setParameter<double*>("r", r_device, GSPAR_PARAM_PRESENT);
		kernel_two_map->setParameter<double*>("global_data", global_data_device, GSPAR_PARAM_PRESENT);
		kernel_two_map->setParameter("NA", NA);

		kernel_two_map->run<Instance>();

		double global_data_reduce=0.0; 		
		global_data_device->copyOut();
		for(int i=0; i<blocks_per_grid_on_kernel_two; i++){global_data_reduce+=global_data[i];}

		return global_data_reduce;
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

/* q = A.p */
void kernel_three(int* rowstr, 
		double* a, 
		double* p, 
		int* colidx,
		double* q){
	try {
		kernel_three_map->setParameter<int*>("rowstr", rowstr_device, GSPAR_PARAM_PRESENT);
		kernel_three_map->setParameter<double*>("a", a_device, GSPAR_PARAM_PRESENT);
		kernel_three_map->setParameter<double*>("p", p_device, GSPAR_PARAM_PRESENT);
		kernel_three_map->setParameter<int*>("colidx", colidx_device, GSPAR_PARAM_PRESENT);
		kernel_three_map->setParameter<double*>("q", q_device, GSPAR_PARAM_PRESENT);		
		kernel_three_map->setParameter("NA", NA);

		kernel_three_map->run<Instance>();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

/* obtain p.q */
double kernel_four(double* global_data,
		double* p,
		double* q){
	try {
		kernel_four_map->setParameter<double*>("p", p_device, GSPAR_PARAM_PRESENT);
		kernel_four_map->setParameter<double*>("q", q_device, GSPAR_PARAM_PRESENT);
		kernel_four_map->setParameter<double*>("global_data", global_data_device, GSPAR_PARAM_PRESENT);
		kernel_four_map->setParameter("NA", NA);

		kernel_four_map->run<Instance>();

		global_data_device->copyOut();

		double global_data_reduce=0.0; 			
		
		for(int i=0; i<blocks_per_grid_on_kernel_four; i++){global_data_reduce+=global_data[i];}

		return global_data_reduce;
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

/* obtain z and r */
void kernel_five(double* z, 
		double* r, 
		double* p, 
		double* q,
		double alpha){
	try {
		kernel_five_map->setParameter<double*>("p", p_device, GSPAR_PARAM_PRESENT);
		kernel_five_map->setParameter<double*>("q", q_device, GSPAR_PARAM_PRESENT);
		kernel_five_map->setParameter<double*>("r", r_device, GSPAR_PARAM_PRESENT);
		kernel_five_map->setParameter<double*>("z", z_device, GSPAR_PARAM_PRESENT);		
		kernel_five_map->setParameter("alpha", alpha);
		kernel_five_map->setParameter("NA", NA);

		kernel_five_map->run<Instance>();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

/* obtain the norm of r */
double kernel_six(double* buffer,
		double* r){
	try {
		kernel_six_map->setParameter<double*>("r", r_device, GSPAR_PARAM_PRESENT);
		kernel_six_map->setParameter<double*>("global_data", global_data_device, GSPAR_PARAM_PRESENT);
		kernel_six_map->setParameter("NA", NA);	

		kernel_six_map->run<Instance>();

		double global_data_reduce=0.0;	
		global_data_device->copyOut();
		for(int i=0; i<blocks_per_grid_on_kernel_six; i++){global_data_reduce+=global_data[i];}
		return global_data_reduce;
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

/* p = r + beta*p*/
void kernel_seven(double* p, 
		double* r,
		double beta){
	try {
		kernel_seven_map->setParameter<double*>("p", p_device, GSPAR_PARAM_PRESENT);
		kernel_seven_map->setParameter<double*>("r", r_device, GSPAR_PARAM_PRESENT);
		kernel_seven_map->setParameter("beta", beta);
		kernel_seven_map->setParameter("NA", NA);

		kernel_seven_map->run<Instance>();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

/* ||r|| = ||x - A.z|| */
void kernel_eight(int* rowstr, 
		double* a, 
		double* z, 
		int* colidx,
		double* r){
	try {
		kernel_eight_map->setParameter<int*>("rowstr", rowstr_device, GSPAR_PARAM_PRESENT);
		kernel_eight_map->setParameter<double*>("a", a_device, GSPAR_PARAM_PRESENT);
		kernel_eight_map->setParameter<double*>("z", z_device, GSPAR_PARAM_PRESENT);
		kernel_eight_map->setParameter<int*>("colidx", colidx_device, GSPAR_PARAM_PRESENT);
		kernel_eight_map->setParameter<double*>("r", r_device, GSPAR_PARAM_PRESENT);
		kernel_eight_map->setParameter("NA", NA);	

		kernel_eight_map->run<Instance>();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

/* r contains A.z */
double kernel_nine(double* buffer,
		double* x,
		double* r){
	try {
		kernel_nine_map->setParameter<double*>("r", r_device, GSPAR_PARAM_PRESENT);
		kernel_nine_map->setParameter<double*>("x", x_device, GSPAR_PARAM_PRESENT);
		kernel_nine_map->setParameter<double*>("global_data", global_data_device, GSPAR_PARAM_PRESENT);
		kernel_nine_map->setParameter("NA", NA);

		kernel_nine_map->run<Instance>();

		double global_data_reduce=0.0;	
		global_data_device->copyOut();
		for(int i=0; i<blocks_per_grid_on_kernel_nine; i++){global_data_reduce+=global_data[i];}
		return global_data_reduce;
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

/* find the norm of z */
double kernel_ten_1(double* buffer,
		double* x,
		double* z){
	try {
		kernel_ten1_map->setParameter<double*>("x", x_device, GSPAR_PARAM_PRESENT);
		kernel_ten1_map->setParameter<double*>("z", z_device, GSPAR_PARAM_PRESENT);
		kernel_ten1_map->setParameter<double*>("norm_temp1", global_data_device, GSPAR_PARAM_PRESENT);
		kernel_ten1_map->setParameter<double*>("norm_temp2", global_data_two_device, GSPAR_PARAM_PRESENT);
		kernel_ten1_map->setParameter("NA", NA);

		kernel_ten1_map->run<Instance>();

		double global_data_reduce=0.0;  		
		global_data_device->copyOut();
		for(int i=0; i<blocks_per_grid_on_kernel_ten; i++){
			global_data_reduce+=global_data[i];
		}
		return global_data_reduce;
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

/* find the norm of z */
double kernel_ten_2(double* buffer,
		double* z){
	try {
		kernel_ten2_map->setParameter<double*>("x", x_device, GSPAR_PARAM_PRESENT);
		kernel_ten2_map->setParameter<double*>("z", z_device, GSPAR_PARAM_PRESENT);
		kernel_ten2_map->setParameter<double*>("norm_temp1", global_data_device, GSPAR_PARAM_PRESENT);
		kernel_ten2_map->setParameter<double*>("norm_temp2", global_data_two_device, GSPAR_PARAM_PRESENT);
		kernel_ten2_map->setParameter("NA", NA);	

		kernel_ten2_map->run<Instance>();
		
		double global_data_two_reduce=0.0;
		global_data_two_device->copyOut();
		for(int i=0; i<blocks_per_grid_on_kernel_ten; i++){
			global_data_two_reduce+=global_data_two[i];
		}
		return global_data_two_reduce;
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

/* normalize z */
void kernel_eleven(double* x, 
		double* z, 
		double norm_temp2){
	try {
		kernel_eleven_map->setParameter<double*>("x", x_device, GSPAR_PARAM_PRESENT);
		kernel_eleven_map->setParameter<double*>("z", z_device, GSPAR_PARAM_PRESENT);
		kernel_eleven_map->setParameter("norm_temp2", norm_temp2);	
		kernel_eleven_map->setParameter("NA", NA);	

		kernel_eleven_map->run<Instance>();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

/* gpu kernel sources */
std::string source_kernel_one_map = GSPAR_STRINGIZE_SOURCE(
	int thread_id = gspar_get_global_id(0);
	if(thread_id >= NA){return;}
	q[thread_id] = 0.0;
	z[thread_id] = 0.0;
	r[thread_id] = x[thread_id];
	p[thread_id] = r[thread_id];
);

std::string source_kernel_two_map = GSPAR_STRINGIZE_SOURCE(
	GSPAR_DEVICE_SHARED_MEMORY double share_data[CG_THREADS_PER_BLOCK_ON_KERNEL_TWO];

	int thread_id = gspar_get_global_id(0);
	int local_id = gspar_get_thread_id(0);

	share_data[local_id] = 0.0;

	if(thread_id >= NA){return;}

	double r_value = r[thread_id];
	share_data[local_id] = r_value * r_value;

	gspar_synchronize_local_threads();
	for(int i=gspar_get_block_size(0)/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		gspar_synchronize_local_threads();
	}
	if(local_id==0){global_data[gspar_get_block_id(0)]=share_data[0];}	
);

std::string source_kernel_three_map = GSPAR_STRINGIZE_SOURCE(
	GSPAR_DEVICE_SHARED_MEMORY double share_data[CG_THREADS_PER_BLOCK_ON_KERNEL_THREE];
	
	int j = gspar_get_block_id(0);
	int local_id = gspar_get_thread_id(0);

	int begin = rowstr[j];
	int end = rowstr[j+1];
	double sum = 0.0;
	for(int k=begin+local_id; k<end; k+=gspar_get_block_size(0)){
		sum = sum + a[k]*p[colidx[k]];
	}
	share_data[local_id] = sum;

	gspar_synchronize_local_threads();
	for(int i=gspar_get_block_size(0)/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		gspar_synchronize_local_threads();
	}
	if(local_id==0){q[j]=share_data[0];}
);

std::string source_kernel_four_map = GSPAR_STRINGIZE_SOURCE(
	GSPAR_DEVICE_SHARED_MEMORY double share_data[CG_THREADS_PER_BLOCK_ON_KERNEL_FOUR];

	int thread_id = gspar_get_global_id(0);
	int local_id = gspar_get_thread_id(0);

	share_data[local_id] = 0.0;

	if(thread_id >= NA){return;}

	share_data[local_id] = p[thread_id] * q[thread_id];

	gspar_synchronize_local_threads();
	for(int i=gspar_get_block_size(0)/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		gspar_synchronize_local_threads();
	}
	if(local_id==0){global_data[gspar_get_block_id(0)]=share_data[0];}
);

std::string source_kernel_five_map = GSPAR_STRINGIZE_SOURCE(
	int thread_id = gspar_get_global_id(0);
	if(thread_id >= NA){return;}
	z[thread_id] = z[thread_id] + alpha*p[thread_id];
	r[thread_id] = r[thread_id] - alpha*q[thread_id];
);

std::string source_kernel_six_map = GSPAR_STRINGIZE_SOURCE(
	GSPAR_DEVICE_SHARED_MEMORY double share_data[CG_THREADS_PER_BLOCK_ON_KERNEL_SIX];

	int thread_id = gspar_get_global_id(0);
	int local_id = gspar_get_thread_id(0);
	share_data[local_id] = 0.0;

	if(thread_id >= NA){return;}

	double r_value = r[thread_id];
	share_data[local_id] = r_value * r_value;

	gspar_synchronize_local_threads();
	for(int i=gspar_get_block_size(0)/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		gspar_synchronize_local_threads();
	}
	if(local_id==0){global_data[gspar_get_block_id(0)]=share_data[0];}
);

std::string source_kernel_seven_map = GSPAR_STRINGIZE_SOURCE(
	int thread_id = gspar_get_global_id(0);
	if(thread_id >= NA){return;}
	p[thread_id] = r[thread_id] + beta*p[thread_id];
);

std::string source_kernel_eight_map = GSPAR_STRINGIZE_SOURCE(
	GSPAR_DEVICE_SHARED_MEMORY double share_data[CG_THREADS_PER_BLOCK_ON_KERNEL_EIGHT];

	int j = gspar_get_block_id(0);
	int local_id = gspar_get_thread_id(0);

	int begin = rowstr[j];
	int end = rowstr[j+1];
	double sum = 0.0;
	for(int k=begin+local_id; k<end; k+=gspar_get_block_size(0)){
		sum = sum + a[k]*z[colidx[k]];
	}
	share_data[local_id] = sum;

	gspar_synchronize_local_threads();
	for(int i=gspar_get_block_size(0)/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		gspar_synchronize_local_threads();
	}
	if(local_id==0){r[j]=share_data[0];}
);

std::string source_kernel_nine_map = GSPAR_STRINGIZE_SOURCE(
	GSPAR_DEVICE_SHARED_MEMORY double share_data[CG_THREADS_PER_BLOCK_ON_KERNEL_NINE];

	int thread_id = gspar_get_global_id(0);
	int local_id = gspar_get_thread_id(0);

	share_data[local_id] = 0.0;

	if(thread_id >= NA){return;}

	share_data[local_id] = x[thread_id] - r[thread_id];
	share_data[local_id] = share_data[local_id] * share_data[local_id];

	gspar_synchronize_local_threads();
	for(int i=gspar_get_block_size(0)/2; i>0; i>>=1) {
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		gspar_synchronize_local_threads();
	}
	if(local_id==0){global_data[gspar_get_block_id(0)]=share_data[0];}
);

std::string source_kernel_ten1_map = GSPAR_STRINGIZE_SOURCE(
	GSPAR_DEVICE_SHARED_MEMORY double share_data_1[CG_THREADS_PER_BLOCK_ON_KERNEL_TEN];

	int thread_id = gspar_get_global_id(0);
	int local_id = gspar_get_thread_id(0);

	share_data_1[gspar_get_thread_id(0)] = 0.0;

	if(thread_id >= NA){return;}

	share_data_1[local_id] = x[thread_id]*z[thread_id];

	gspar_synchronize_local_threads();
	for(int i=gspar_get_block_size(0)/2; i>0; i>>=1){
		if(local_id<i){
			share_data_1[local_id]+=share_data_1[local_id+i];}
		gspar_synchronize_local_threads();
	}
	if(local_id==0){
		norm_temp1[gspar_get_block_id(0)]=share_data_1[0];}
);

std::string source_kernel_ten2_map = GSPAR_STRINGIZE_SOURCE(
	GSPAR_DEVICE_SHARED_MEMORY double share_data_2[CG_THREADS_PER_BLOCK_ON_KERNEL_TEN];

	int thread_id = gspar_get_global_id(0);
	int local_id = gspar_get_thread_id(0);
	
	share_data_2[gspar_get_thread_id(0)] = 0.0;

	if(thread_id >= NA){return;}
	
	share_data_2[local_id] = z[thread_id]*z[thread_id];

	gspar_synchronize_local_threads();
	for(int i=gspar_get_block_size(0)/2; i>0; i>>=1){
		if(local_id<i){
			share_data_2[local_id]+=share_data_2[local_id+i];}
		gspar_synchronize_local_threads();
	}
	if(local_id==0){
		norm_temp2[gspar_get_block_id(0)]=share_data_2[0];}
);

std::string source_kernel_eleven_map = GSPAR_STRINGIZE_SOURCE(
	int thread_id = gspar_get_global_id(0);
	x[thread_id] = norm_temp2 * z[thread_id];
);

std::string source_additional_routines =
    "#define CG_THREADS_PER_BLOCK_ON_KERNEL_ONE " + std::to_string(CG_THREADS_PER_BLOCK_ON_KERNEL_ONE) + "\n" +
    "#define CG_THREADS_PER_BLOCK_ON_KERNEL_TWO " + std::to_string(CG_THREADS_PER_BLOCK_ON_KERNEL_TWO) + "\n" +
    "#define CG_THREADS_PER_BLOCK_ON_KERNEL_THREE " + std::to_string(CG_THREADS_PER_BLOCK_ON_KERNEL_THREE) + "\n" +
    "#define CG_THREADS_PER_BLOCK_ON_KERNEL_FOUR " + std::to_string(CG_THREADS_PER_BLOCK_ON_KERNEL_FOUR) + "\n" +
    "#define CG_THREADS_PER_BLOCK_ON_KERNEL_FIVE " + std::to_string(CG_THREADS_PER_BLOCK_ON_KERNEL_FIVE) + "\n" +
    "#define CG_THREADS_PER_BLOCK_ON_KERNEL_SIX " + std::to_string(CG_THREADS_PER_BLOCK_ON_KERNEL_SIX) + "\n" +
    "#define CG_THREADS_PER_BLOCK_ON_KERNEL_SEVEN " + std::to_string(CG_THREADS_PER_BLOCK_ON_KERNEL_SEVEN) + "\n" +
    "#define CG_THREADS_PER_BLOCK_ON_KERNEL_EIGHT " + std::to_string(CG_THREADS_PER_BLOCK_ON_KERNEL_EIGHT) + "\n" +
    "#define CG_THREADS_PER_BLOCK_ON_KERNEL_NINE " + std::to_string(CG_THREADS_PER_BLOCK_ON_KERNEL_NINE) + "\n" +
    "#define CG_THREADS_PER_BLOCK_ON_KERNEL_TEN " + std::to_string(CG_THREADS_PER_BLOCK_ON_KERNEL_TEN) + "\n" +
    "#define CG_THREADS_PER_BLOCK_ON_KERNEL_ELEVEN " + std::to_string(CG_THREADS_PER_BLOCK_ON_KERNEL_ELEVEN) + "\n";
