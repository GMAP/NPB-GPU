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
 * The hip version is a parallel implementation of the serial C++ version
 * hip version: https://github.com/GMAP/NPB-GPU/tree/master/hip
 *
 * Authors of the hip code:
 *      Gabriell Araujo <hexenoften@gmail.com>
 *
 * ------------------------------------------------------------------------------
 */

#include <hip/hip_runtime.h>
#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

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
#define PROFILING_KERNEL_ONE (1)
#define PROFILING_KERNEL_TWO (2)
#define PROFILING_KERNEL_THREE (3)
#define PROFILING_KERNEL_FOUR (4)
#define PROFILING_KERNEL_FIVE (5)
#define PROFILING_KERNEL_SIX (6)
#define PROFILING_KERNEL_SEVEN (7)
#define PROFILING_KERNEL_EIGHT (8)
#define PROFILING_KERNEL_NINE (9)
#define PROFILING_KERNEL_TEN (10)
#define PROFILING_KERNEL_ELEVEN (11)

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
/* gpu variables */
int* colidx_device;
int* rowstr_device;
double* a_device;
double* p_device;
double* q_device;
double* r_device;
double* x_device;
double* z_device;
double* rho_device;
double* d_device;
double* alpha_device;
double* beta_device;
double* sum_device;
double* norm_temp1_device;
double* norm_temp2_device;
double* global_data;
double* global_data_two;
double* global_data_device;
double* global_data_two_device;
double global_data_reduce;
double global_data_two_reduce;
size_t global_data_elements;
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
size_t size_rho_device;
size_t size_d_device;
size_t size_alpha_device;
size_t size_beta_device;
size_t size_sum_device;
size_t size_norm_temp1_device;
size_t size_norm_temp2_device;
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
size_t size_shared_data_on_kernel_one;
size_t size_shared_data_on_kernel_two;
size_t size_shared_data_on_kernel_three;
size_t size_shared_data_on_kernel_four;
size_t size_shared_data_on_kernel_five;
size_t size_shared_data_on_kernel_six;
size_t size_shared_data_on_kernel_seven;
size_t size_shared_data_on_kernel_eight;
size_t size_shared_data_on_kernel_nine;
size_t size_shared_data_on_kernel_ten;
size_t size_shared_data_on_kernel_eleven;
size_t size_reduce_memory_on_kernel_one;
size_t size_reduce_memory_on_kernel_two;
size_t size_reduce_memory_on_kernel_three;
size_t size_reduce_memory_on_kernel_four;
size_t size_reduce_memory_on_kernel_five;
size_t size_reduce_memory_on_kernel_six;
size_t size_reduce_memory_on_kernel_seven;
size_t size_reduce_memory_on_kernel_eight;
size_t size_reduce_memory_on_kernel_nine;
size_t size_reduce_memory_on_kernel_ten;
size_t size_reduce_memory_on_kernel_eleven;
int gpu_device_id;
int total_devices;
hipDeviceProp_t gpu_device_properties;
extern __shared__ double extern_share_data[];

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
static void conj_grad_gpu(double* rnorm);
static void gpu_kernel_one();
__global__ void gpu_kernel_one(double p[], 
		double q[], 
		double r[], 
		double x[], 
		double z[]);
static void gpu_kernel_two(double* rho_host);
__global__ void gpu_kernel_two(double r[],
		double* rho, 
		double global_data[]);
static void gpu_kernel_three();
__global__ void gpu_kernel_three(int colidx[], 
		int rowstr[], 
		double a[], 
		double p[], 
		double q[]);
static void gpu_kernel_four(double* d_host);
__global__ void gpu_kernel_four(double* d, 
		double* p, 
		double* q, 
		double global_data[]);
static void gpu_kernel_five(double alpha_host);
__global__ void gpu_kernel_five_1(double alpha, 
		double* p, 
		double* z);
__global__ void gpu_kernel_five_2(double alpha, 
		double* q, 
		double* r);
static void gpu_kernel_six(double* rho_host);
__global__ void gpu_kernel_six(double r[],
		double global_data[]);
static void gpu_kernel_seven(double beta_host);
__global__ void gpu_kernel_seven(double beta, 
		double* p, 
		double* r);
static void gpu_kernel_eight();
__global__ void gpu_kernel_eight(int colidx[], 
		int rowstr[], 
		double a[], 
		double r[], 
		double* z);
static void gpu_kernel_nine(double* sum_host);
__global__ void gpu_kernel_nine(double r[],
		double x[], 
		double* sum, 
		double global_data[]);
static void gpu_kernel_ten(double* norm_temp1, 
		double* norm_temp2);
__global__ void gpu_kernel_ten_1(double* norm_temp, 
		double x[], 
		double z[]);
__global__ void gpu_kernel_ten_2(double* norm_temp, 
		double x[], 
		double z[]);
static void gpu_kernel_eleven(double norm_temp2);
__global__ void gpu_kernel_eleven(double norm_temp2, 
		double x[], 
		double z[]);
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
static void release_gpu();
static void setup_gpu();
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

/* cg */
int main(int argc, char** argv){
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
	printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif
#if defined(PROFILING)
	printf(" PROFILING mode on\n");
#endif
	int	i, j, k, it;
	double zeta;
	double rnorm;
	double norm_temp1, norm_temp2;
	double t, mflops;
	char class_npb;
	boolean verified;
	double zeta_verify_value, epsilon, err;

	timer_clear(PROFILING_TOTAL_TIME);
#if defined(PROFILING)
	timer_clear(PROFILING_KERNEL_ONE);
	timer_clear(PROFILING_KERNEL_TWO);
	timer_clear(PROFILING_KERNEL_THREE);
	timer_clear(PROFILING_KERNEL_FOUR);
	timer_clear(PROFILING_KERNEL_FIVE);
	timer_clear(PROFILING_KERNEL_SIX);
	timer_clear(PROFILING_KERNEL_SEVEN);
	timer_clear(PROFILING_KERNEL_EIGHT);
	timer_clear(PROFILING_KERNEL_NINE);
	timer_clear(PROFILING_KERNEL_TEN);
	timer_clear(PROFILING_KERNEL_ELEVEN);
#endif

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

	printf("\n\n NAS Parallel Benchmarks 4.1 HIP C++ version - CG Benchmark\n\n");
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
		conj_grad_gpu(&rnorm);

		/*
		 * --------------------------------------------------------------------
		 * zeta = shift + 1/(x.z)
		 * so, first: (x.z)
		 * also, find norm of z
		 * so, first: (z.z)
		 * --------------------------------------------------------------------
		 */
		gpu_kernel_ten(&norm_temp1, &norm_temp2);
		norm_temp2 = 1.0 / sqrt(norm_temp2);
		zeta = SHIFT + 1.0 / norm_temp1;
		if(it==1){printf("\n   iteration           ||r||                 zeta\n");}
		printf("    %5d       %20.14e%20.13e\n", it, rnorm, zeta);

		/* normalize z to obtain x */
		gpu_kernel_eleven(norm_temp2);
	} /* end of main iter inv pow meth */

	timer_stop(PROFILING_TOTAL_TIME);

	/*
	 * --------------------------------------------------------------------
	 * end of timed section
	 * --------------------------------------------------------------------
	 */

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

	char gpu_config[256];
	char gpu_config_string[2048];
#if defined(PROFILING)
	sprintf(gpu_config, "%5s\t%25s\t%25s\t%25s\n", "GPU Kernel", "Threads Per Block", "Time in Seconds", "Time in Percentage");
	strcpy(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " one", threads_per_block_on_kernel_one, timer_read(PROFILING_KERNEL_ONE), (timer_read(PROFILING_KERNEL_ONE)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " two", threads_per_block_on_kernel_two, timer_read(PROFILING_KERNEL_TWO), (timer_read(PROFILING_KERNEL_TWO)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " three", threads_per_block_on_kernel_three, timer_read(PROFILING_KERNEL_THREE), (timer_read(PROFILING_KERNEL_THREE)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " four", threads_per_block_on_kernel_four, timer_read(PROFILING_KERNEL_FOUR), (timer_read(PROFILING_KERNEL_FOUR)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " five", threads_per_block_on_kernel_five, timer_read(PROFILING_KERNEL_FIVE), (timer_read(PROFILING_KERNEL_FIVE)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " six", threads_per_block_on_kernel_six, timer_read(PROFILING_KERNEL_SIX), (timer_read(PROFILING_KERNEL_SIX)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " seven", threads_per_block_on_kernel_seven, timer_read(PROFILING_KERNEL_SEVEN), (timer_read(PROFILING_KERNEL_SEVEN)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " eight", threads_per_block_on_kernel_eight, timer_read(PROFILING_KERNEL_EIGHT), (timer_read(PROFILING_KERNEL_EIGHT)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " nine", threads_per_block_on_kernel_nine, timer_read(PROFILING_KERNEL_NINE), (timer_read(PROFILING_KERNEL_NINE)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " ten", threads_per_block_on_kernel_ten, timer_read(PROFILING_KERNEL_TEN), (timer_read(PROFILING_KERNEL_TEN)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " eleven", threads_per_block_on_kernel_eleven, timer_read(PROFILING_KERNEL_ELEVEN), (timer_read(PROFILING_KERNEL_ELEVEN)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
#else
	sprintf(gpu_config, "%5s\t%25s\n", "GPU Kernel", "Threads Per Block");
	strcpy(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " one", threads_per_block_on_kernel_one);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " two", threads_per_block_on_kernel_two);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " three", threads_per_block_on_kernel_three);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " four", threads_per_block_on_kernel_four);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " five", threads_per_block_on_kernel_five);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " six", threads_per_block_on_kernel_six);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " seven", threads_per_block_on_kernel_seven);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " eight", threads_per_block_on_kernel_eight);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " nine", threads_per_block_on_kernel_nine);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " ten", threads_per_block_on_kernel_ten);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " eleven", threads_per_block_on_kernel_eleven);
	strcat(gpu_config_string, gpu_config);
#endif

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

static void conj_grad_gpu(double* rnorm){
	double d, sum, rho, rho0, alpha, beta;
	int cgit, cgitmax = 25;

	/* initialize the CG algorithm */
	gpu_kernel_one();

	/* rho = r.r - now, obtain the norm of r: first, sum squares of r elements locally */
	gpu_kernel_two(&rho);

	/* the conj grad iteration loop */
	for(cgit = 1; cgit <= cgitmax; cgit++){
		/* q = A.p */
		gpu_kernel_three();

		/* obtain p.q */
		gpu_kernel_four(&d);

		alpha = rho / d;

		/* save a temporary of rho */
		rho0 = rho;

		/* obtain (z = z + alpha*p) and (r = r - alpha*q) */
		gpu_kernel_five(alpha);

		/* rho = r.r - now, obtain the norm of r: first, sum squares of r elements locally */
		gpu_kernel_six(&rho);

		/* obtain beta */
		beta = rho / rho0;

		/* p = r + beta*p */
		gpu_kernel_seven(beta);
	} /* end of do cgit=1, cgitmax */

	/* compute residual norm explicitly:  ||r|| = ||x - A.z|| */
	gpu_kernel_eight();

	/* at this point, r contains A.z */
	gpu_kernel_nine(&sum);

	*rnorm = sqrt(sum);
}

static void gpu_kernel_one(){   
#if defined(PROFILING)
	timer_start(PROFILING_KERNEL_ONE);
#endif
	gpu_kernel_one<<<blocks_per_grid_on_kernel_one,
		threads_per_block_on_kernel_one>>>(
				p_device, 
				q_device, 
				r_device, 
				x_device, 
				z_device);
#if defined(PROFILING)
	timer_stop(PROFILING_KERNEL_ONE);
#endif
}

__global__ void gpu_kernel_one(double p[], 
		double q[], 
		double r[], 
		double x[], 
		double z[]){
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread_id >= NA){return;}
	q[thread_id] = 0.0;
	z[thread_id] = 0.0;
	double x_value = x[thread_id];
	r[thread_id] = x_value;
	p[thread_id] = x_value;
}

static void gpu_kernel_two(double* rho_host){  
#if defined(PROFILING)
	timer_start(PROFILING_KERNEL_TWO);
#endif
	gpu_kernel_two<<<blocks_per_grid_on_kernel_two,
		threads_per_block_on_kernel_two,
		size_shared_data_on_kernel_two>>>(
				r_device, 
				rho_device, 
				global_data_device);
	global_data_reduce=0.0; 
	hipMemcpy(global_data, global_data_device, size_reduce_memory_on_kernel_two, hipMemcpyDeviceToHost);	
	for(int i=0; i<blocks_per_grid_on_kernel_two; i++){global_data_reduce+=global_data[i];}
	*rho_host=global_data_reduce;
#if defined(PROFILING)
	timer_stop(PROFILING_KERNEL_TWO);
#endif
}

__global__ void gpu_kernel_two(double r[],
		double* rho, 
		double global_data[]){
	double* share_data = (double*)extern_share_data;

	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int local_id = threadIdx.x;

	share_data[local_id] = 0.0;

	if(thread_id >= NA){return;}

	double r_value = r[thread_id];
	share_data[local_id] = r_value * r_value;

	__syncthreads();
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		__syncthreads();
	}
	if(local_id==0){global_data[blockIdx.x]=share_data[0];}
}

static void gpu_kernel_three(){
#if defined(PROFILING)
	timer_start(PROFILING_KERNEL_THREE);
#endif
	gpu_kernel_three<<<blocks_per_grid_on_kernel_three,
		threads_per_block_on_kernel_three,
		size_shared_data_on_kernel_three>>>(
				colidx_device,
				rowstr_device,
				a_device,
				p_device,
				q_device);
#if defined(PROFILING)
	timer_stop(PROFILING_KERNEL_THREE);
#endif
}

__global__ void gpu_kernel_three(int colidx[], 
		int rowstr[], 
		double a[], 
		double p[], 
		double q[]){
	double* share_data = (double*)extern_share_data;

	int j = (int) ((blockIdx.x*blockDim.x+threadIdx.x) / blockDim.x);
	int local_id = threadIdx.x;

	int begin = rowstr[j];
	int end = rowstr[j+1];
	double sum = 0.0;
	for(int k=begin+local_id; k<end; k+=blockDim.x){
		sum = sum + a[k]*p[colidx[k]];
	}
	share_data[local_id] = sum;

	__syncthreads();
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		__syncthreads();
	}
	if(local_id==0){q[j]=share_data[0];}
}

static void gpu_kernel_four(double* d_host){   
#if defined(PROFILING)
	timer_start(PROFILING_KERNEL_FOUR);
#endif
	gpu_kernel_four<<<blocks_per_grid_on_kernel_four,
		threads_per_block_on_kernel_four,
		size_shared_data_on_kernel_four>>>(
				d_device, 
				p_device,
				q_device,
				global_data_device);
	global_data_reduce=0.0; 
	hipMemcpy(global_data, global_data_device, size_reduce_memory_on_kernel_four, hipMemcpyDeviceToHost);
	for(int i=0; i<blocks_per_grid_on_kernel_four; i++){global_data_reduce+=global_data[i];}
	*d_host=global_data_reduce;
#if defined(PROFILING)
	timer_stop(PROFILING_KERNEL_FOUR);
#endif
}

__global__ void gpu_kernel_four(double* d, 
		double* p, 
		double* q, 
		double global_data[]){
	double* share_data = (double*)extern_share_data; 

	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int local_id = threadIdx.x;

	share_data[local_id] = 0.0;

	if(thread_id >= NA){return;}

	share_data[threadIdx.x] = p[thread_id] * q[thread_id];

	__syncthreads();
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		__syncthreads();
	}
	if(local_id==0){global_data[blockIdx.x]=share_data[0];}
}

static void gpu_kernel_five(double alpha_host){
#if defined(PROFILING)
	timer_start(PROFILING_KERNEL_FIVE);
#endif
	gpu_kernel_five_1<<<blocks_per_grid_on_kernel_five,
		threads_per_block_on_kernel_five>>>(
				alpha_host,
				p_device,
				z_device);
	gpu_kernel_five_2<<<blocks_per_grid_on_kernel_five,
		threads_per_block_on_kernel_five>>>(
				alpha_host,
				q_device,
				r_device);
#if defined(PROFILING)
	timer_stop(PROFILING_KERNEL_FIVE);
#endif
} 

__global__ void gpu_kernel_five_1(double alpha, 
		double* p, 
		double* z){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j >= NA){return;}
	z[j] += alpha * p[j];
}

__global__ void gpu_kernel_five_2(double alpha, 
		double* q, 
		double* r){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j >= NA){return;}
	r[j] -= alpha * q[j];
}

static void gpu_kernel_six(double* rho_host){
#if defined(PROFILING)
	timer_start(PROFILING_KERNEL_SIX);
#endif
	gpu_kernel_six<<<blocks_per_grid_on_kernel_six,
		threads_per_block_on_kernel_six,
		size_shared_data_on_kernel_six>>>(
				r_device, 
				global_data_device);
	global_data_reduce=0.0;
	hipMemcpy(global_data, global_data_device, size_reduce_memory_on_kernel_six, hipMemcpyDeviceToHost);
	for(int i=0; i<blocks_per_grid_on_kernel_six; i++){global_data_reduce+=global_data[i];}
	*rho_host=global_data_reduce;
#if defined(PROFILING)
	timer_stop(PROFILING_KERNEL_SIX);
#endif
} 

__global__ void gpu_kernel_six(double r[], 
		double global_data[]){
	double* share_data = (double*)extern_share_data;
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int local_id = threadIdx.x;
	share_data[local_id] = 0.0;
	if(thread_id >= NA){return;}
	double r_value = r[thread_id];
	share_data[local_id] = r_value * r_value;
	__syncthreads();
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		__syncthreads();
	}
	if(local_id==0){global_data[blockIdx.x]=share_data[0];}
}

static void gpu_kernel_seven(double beta_host){
#if defined(PROFILING)
	timer_start(PROFILING_KERNEL_SEVEN);
#endif
	gpu_kernel_seven<<<blocks_per_grid_on_kernel_seven,
		threads_per_block_on_kernel_seven>>>(
				beta_host,
				p_device,
				r_device);
#if defined(PROFILING)
	timer_stop(PROFILING_KERNEL_SEVEN);
#endif
}

__global__ void gpu_kernel_seven(double beta, 
		double* p, 
		double* r){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j >= NA){return;}
	p[j] = r[j] + beta*p[j];
}

static void gpu_kernel_eight(){
#if defined(PROFILING)
	timer_start(PROFILING_KERNEL_EIGHT);
#endif
	gpu_kernel_eight<<<blocks_per_grid_on_kernel_eight,
		threads_per_block_on_kernel_eight,
		size_shared_data_on_kernel_eight>>>(
				colidx_device, 
				rowstr_device, 
				a_device, 
				r_device, 
				z_device);
#if defined(PROFILING)
	timer_stop(PROFILING_KERNEL_EIGHT);
#endif
}

__global__ void gpu_kernel_eight(int colidx[], 
		int rowstr[], 
		double a[], 
		double r[], 
		double* z){
	double* share_data = (double*)extern_share_data;

	int j = (int) ((blockIdx.x*blockDim.x+threadIdx.x) / blockDim.x);
	int local_id = threadIdx.x;

	int begin = rowstr[j];
	int end = rowstr[j+1];
	double sum = 0.0;
	for(int k=begin+local_id; k<end; k+=blockDim.x){
		sum = sum + a[k]*z[colidx[k]];
	}
	share_data[local_id] = sum;

	__syncthreads();
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		__syncthreads();
	}
	if(local_id==0){r[j]=share_data[0];}
}

static void gpu_kernel_nine(double* sum_host){ 
#if defined(PROFILING)
	timer_start(PROFILING_KERNEL_NINE);
#endif 
	gpu_kernel_nine<<<blocks_per_grid_on_kernel_nine,
		threads_per_block_on_kernel_nine,
		size_shared_data_on_kernel_nine>>>(
				r_device, 
				x_device, 
				sum_device,
				global_data_device);
	global_data_reduce=0.0;
	hipMemcpy(global_data, global_data_device, size_reduce_memory_on_kernel_nine, hipMemcpyDeviceToHost);
	for(int i=0; i<blocks_per_grid_on_kernel_nine; i++){global_data_reduce+=global_data[i];}
	*sum_host=global_data_reduce;
#if defined(PROFILING)
	timer_stop(PROFILING_KERNEL_NINE);
#endif
}

__global__ void gpu_kernel_nine(double r[], double x[], double* sum, double global_data[]){
	double* share_data = (double*)extern_share_data;

	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int local_id = threadIdx.x;

	share_data[local_id] = 0.0;

	if(thread_id >= NA){return;}

	share_data[local_id] = x[thread_id] - r[thread_id];
	share_data[local_id] = share_data[local_id] * share_data[local_id];

	__syncthreads();
	for(int i=blockDim.x/2; i>0; i>>=1) {
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		__syncthreads();
	}
	if(local_id==0){global_data[blockIdx.x]=share_data[0];}
}

static void gpu_kernel_ten(double* norm_temp1, 
		double* norm_temp2){
#if defined(PROFILING)
	timer_start(PROFILING_KERNEL_TEN);
#endif
	gpu_kernel_ten_1<<<blocks_per_grid_on_kernel_ten,threads_per_block_on_kernel_ten,size_shared_data_on_kernel_ten>>>(global_data_device,x_device,z_device);
	gpu_kernel_ten_2<<<blocks_per_grid_on_kernel_ten,threads_per_block_on_kernel_ten,size_shared_data_on_kernel_ten>>>(global_data_two_device,x_device,z_device);

	global_data_reduce=0.0; 
	global_data_two_reduce=0.0; 
	hipMemcpy(global_data, global_data_device, size_reduce_memory_on_kernel_ten, hipMemcpyDeviceToHost);
	hipMemcpy(global_data_two, global_data_two_device, size_reduce_memory_on_kernel_ten, hipMemcpyDeviceToHost);

	for(int i=0; i<blocks_per_grid_on_kernel_ten; i++){global_data_reduce+=global_data[i];global_data_two_reduce+=global_data_two[i];}
	*norm_temp1=global_data_reduce;
	*norm_temp2=global_data_two_reduce;
#if defined(PROFILING)
	timer_stop(PROFILING_KERNEL_TEN);
#endif
}

__global__ void gpu_kernel_ten_1(double* norm_temp, 
		double x[], 
		double z[]){
	double* share_data = (double*)extern_share_data;	  

	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int local_id = threadIdx.x;	  

	share_data[threadIdx.x] = 0.0;

	if(thread_id >= NA){return;}	 

	share_data[threadIdx.x] = x[thread_id]*z[thread_id];

	__syncthreads();
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		__syncthreads();
	}
	if(local_id==0){norm_temp[blockIdx.x]=share_data[0];}
}

__global__ void gpu_kernel_ten_2(double* norm_temp, 
		double x[], 
		double z[]){
	double* share_data = (double*)extern_share_data;

	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int local_id = threadIdx.x;

	share_data[threadIdx.x] = 0.0;

	if(thread_id >= NA){return;}

	share_data[threadIdx.x] = z[thread_id]*z[thread_id];

	__syncthreads();
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		__syncthreads();
	}
	if(local_id==0){norm_temp[blockIdx.x]=share_data[0];}
}

static void gpu_kernel_eleven(double norm_temp2){   
#if defined(PROFILING)
	timer_start(PROFILING_KERNEL_ELEVEN);
#endif
	gpu_kernel_eleven<<<blocks_per_grid_on_kernel_eleven,
		threads_per_block_on_kernel_eleven>>>(
				norm_temp2,
				x_device,
				z_device);
#if defined(PROFILING)
	timer_stop(PROFILING_KERNEL_ELEVEN);
#endif
}

__global__ void gpu_kernel_eleven(double norm_temp2, double x[], double z[]){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j >= NA){return;}
	x[j]=norm_temp2*z[j];
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

static void release_gpu(){
	hipFree(colidx_device);
	hipFree(rowstr_device);
	hipFree(a_device);
	hipFree(p_device);
	hipFree(q_device);
	hipFree(r_device);
	hipFree(x_device);
	hipFree(z_device);
	hipFree(rho_device);
	hipFree(d_device);
	hipFree(alpha_device);
	hipFree(beta_device);
	hipFree(sum_device);
	hipFree(norm_temp1_device);
	hipFree(norm_temp2_device);
	hipFree(global_data_device);
	hipFree(global_data_two_device);
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
	if((CG_THREADS_PER_BLOCK_ON_KERNEL_ONE>=1)&&
			(CG_THREADS_PER_BLOCK_ON_KERNEL_ONE<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_kernel_one = CG_THREADS_PER_BLOCK_ON_KERNEL_ONE;
	}
	else{
		threads_per_block_on_kernel_one = gpu_device_properties.warpSize;
	}
	if((CG_THREADS_PER_BLOCK_ON_KERNEL_TWO>=1)&&
			(CG_THREADS_PER_BLOCK_ON_KERNEL_TWO<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_kernel_two = CG_THREADS_PER_BLOCK_ON_KERNEL_TWO;
	}
	else{
		threads_per_block_on_kernel_two = gpu_device_properties.warpSize;
	}
	if((CG_THREADS_PER_BLOCK_ON_KERNEL_THREE>=1)&&
			(CG_THREADS_PER_BLOCK_ON_KERNEL_THREE<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_kernel_three = CG_THREADS_PER_BLOCK_ON_KERNEL_THREE;
	}
	else{
		threads_per_block_on_kernel_three = gpu_device_properties.warpSize;
	}	
	if((CG_THREADS_PER_BLOCK_ON_KERNEL_FOUR>=1)&&
			(CG_THREADS_PER_BLOCK_ON_KERNEL_FOUR<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_kernel_four = CG_THREADS_PER_BLOCK_ON_KERNEL_FOUR;
	}
	else{
		threads_per_block_on_kernel_four = gpu_device_properties.warpSize;
	}
	if((CG_THREADS_PER_BLOCK_ON_KERNEL_FIVE>=1)&&
			(CG_THREADS_PER_BLOCK_ON_KERNEL_FIVE<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_kernel_five = CG_THREADS_PER_BLOCK_ON_KERNEL_FIVE;
	}
	else{
		threads_per_block_on_kernel_five = gpu_device_properties.warpSize;
	}
	if((CG_THREADS_PER_BLOCK_ON_KERNEL_SIX>=1)&&
			(CG_THREADS_PER_BLOCK_ON_KERNEL_SIX<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_kernel_six = CG_THREADS_PER_BLOCK_ON_KERNEL_SIX;
	}
	else{
		threads_per_block_on_kernel_six = gpu_device_properties.warpSize;
	}
	if((CG_THREADS_PER_BLOCK_ON_KERNEL_SEVEN>=1)&&
			(CG_THREADS_PER_BLOCK_ON_KERNEL_SEVEN<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_kernel_seven = CG_THREADS_PER_BLOCK_ON_KERNEL_SEVEN;
	}
	else{
		threads_per_block_on_kernel_seven = gpu_device_properties.warpSize;
	}
	if((CG_THREADS_PER_BLOCK_ON_KERNEL_EIGHT>=1)&&
			(CG_THREADS_PER_BLOCK_ON_KERNEL_EIGHT<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_kernel_eight = CG_THREADS_PER_BLOCK_ON_KERNEL_EIGHT;
	}
	else{
		threads_per_block_on_kernel_eight = gpu_device_properties.warpSize;
	}
	if((CG_THREADS_PER_BLOCK_ON_KERNEL_NINE>=1)&&
			(CG_THREADS_PER_BLOCK_ON_KERNEL_NINE<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_kernel_nine = CG_THREADS_PER_BLOCK_ON_KERNEL_NINE;
	}
	else{
		threads_per_block_on_kernel_nine=gpu_device_properties.warpSize;
	}
	if((CG_THREADS_PER_BLOCK_ON_KERNEL_TEN>=1)&&
			(CG_THREADS_PER_BLOCK_ON_KERNEL_TEN<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_kernel_ten = CG_THREADS_PER_BLOCK_ON_KERNEL_TEN;
	}
	else{
		threads_per_block_on_kernel_ten=gpu_device_properties.warpSize;
	}
	if((CG_THREADS_PER_BLOCK_ON_KERNEL_ELEVEN>=1)&&
			(CG_THREADS_PER_BLOCK_ON_KERNEL_ELEVEN<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_kernel_eleven = CG_THREADS_PER_BLOCK_ON_KERNEL_ELEVEN;
	}
	else{
		threads_per_block_on_kernel_eleven = gpu_device_properties.warpSize;
	}	

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

	global_data_elements=ceil(double(NA)/double(gpu_device_properties.warpSize));

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
	size_rho_device=sizeof(double);
	size_d_device=sizeof(double);
	size_alpha_device=sizeof(double);
	size_beta_device=sizeof(double);
	size_sum_device=sizeof(double);
	size_norm_temp1_device=sizeof(double);
	size_norm_temp2_device=sizeof(double);

	global_data=(double*)malloc(size_global_data);
	global_data_two=(double*)malloc(size_global_data);	

	hipMalloc(&colidx_device, size_colidx_device);
	hipMalloc(&rowstr_device, size_rowstr_device);
	hipMalloc(&a_device, size_a_device);
	hipMalloc(&p_device, size_p_device);
	hipMalloc(&q_device, size_q_device);
	hipMalloc(&r_device, size_r_device);
	hipMalloc(&x_device, size_x_device);
	hipMalloc(&z_device, size_z_device);
	hipMalloc(&rho_device, size_rho_device);
	hipMalloc(&d_device, size_d_device);
	hipMalloc(&alpha_device, size_alpha_device);
	hipMalloc(&beta_device, size_beta_device);
	hipMalloc(&sum_device, size_sum_device);
	hipMalloc(&norm_temp1_device, size_norm_temp1_device);
	hipMalloc(&norm_temp2_device, size_norm_temp2_device);
	hipMalloc(&global_data_device, size_global_data);
	hipMalloc(&global_data_two_device, size_global_data);

	hipMemcpy(colidx_device, colidx, size_colidx_device, hipMemcpyHostToDevice);
	hipMemcpy(rowstr_device, rowstr, size_rowstr_device, hipMemcpyHostToDevice);
	hipMemcpy(a_device, a, size_a_device, hipMemcpyHostToDevice);
	hipMemcpy(p_device, p, size_p_device, hipMemcpyHostToDevice);
	hipMemcpy(q_device, q, size_q_device, hipMemcpyHostToDevice);
	hipMemcpy(r_device, r, size_r_device, hipMemcpyHostToDevice);
	hipMemcpy(x_device, x, size_x_device, hipMemcpyHostToDevice);
	hipMemcpy(z_device, z, size_z_device, hipMemcpyHostToDevice);	

	size_shared_data_on_kernel_one=threads_per_block_on_kernel_one*sizeof(double);
	size_shared_data_on_kernel_two=threads_per_block_on_kernel_two*sizeof(double);
	size_shared_data_on_kernel_three=threads_per_block_on_kernel_three*sizeof(double);
	size_shared_data_on_kernel_four=threads_per_block_on_kernel_four*sizeof(double);
	size_shared_data_on_kernel_five=threads_per_block_on_kernel_five*sizeof(double);
	size_shared_data_on_kernel_six=threads_per_block_on_kernel_six*sizeof(double);
	size_shared_data_on_kernel_seven=threads_per_block_on_kernel_seven*sizeof(double);
	size_shared_data_on_kernel_eight=threads_per_block_on_kernel_eight*sizeof(double);
	size_shared_data_on_kernel_nine=threads_per_block_on_kernel_nine*sizeof(double);
	size_shared_data_on_kernel_ten=threads_per_block_on_kernel_ten*sizeof(double);
	size_shared_data_on_kernel_eleven=threads_per_block_on_kernel_eleven*sizeof(double);

	size_reduce_memory_on_kernel_one=blocks_per_grid_on_kernel_one*sizeof(double);
	size_reduce_memory_on_kernel_two=blocks_per_grid_on_kernel_two*sizeof(double);
	size_reduce_memory_on_kernel_three=blocks_per_grid_on_kernel_three*sizeof(double);
	size_reduce_memory_on_kernel_four=blocks_per_grid_on_kernel_four*sizeof(double);
	size_reduce_memory_on_kernel_five=blocks_per_grid_on_kernel_five*sizeof(double);
	size_reduce_memory_on_kernel_six=blocks_per_grid_on_kernel_six*sizeof(double);
	size_reduce_memory_on_kernel_seven=blocks_per_grid_on_kernel_seven*sizeof(double);
	size_reduce_memory_on_kernel_eight=blocks_per_grid_on_kernel_eight*sizeof(double);
	size_reduce_memory_on_kernel_nine=blocks_per_grid_on_kernel_nine*sizeof(double);
	size_reduce_memory_on_kernel_ten=blocks_per_grid_on_kernel_ten*sizeof(double);
	size_reduce_memory_on_kernel_eleven=blocks_per_grid_on_kernel_eleven*sizeof(double);
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
