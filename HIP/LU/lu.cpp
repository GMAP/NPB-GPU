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
 *      S. Weeratunga
 *      V. Venkatakrishnan
 *      E. Barszcz
 *      M. Yarrow
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
 * driver for the performance evaluation of the solver for
 * five coupled parabolic/elliptic partial differential equations
 * ---------------------------------------------------------------------
 * parameters which can be overridden in runtime config file
 * isiz1,isiz2,isiz3 give the maximum size
 * ipr = 1 to print out verbose information
 * omega = 2.0 is correct for all classes
 * tolrsd is tolerance levels for steady state residuals
 * ---------------------------------------------------------------------
 * field variables and residuals
 * to improve cache performance, second two dimensions padded by 1 
 * for even number sizes only.
 * note: corresponding array (called "v") in routines blts, buts, 
 * and l2norm are similarly padded
 * ---------------------------------------------------------------------
 */
#define IPR_DEFAULT (1)
#define OMEGA_DEFAULT (1.2)
#define TOLRSD1_DEF (1.0e-08)
#define TOLRSD2_DEF (1.0e-08)
#define TOLRSD3_DEF (1.0e-08)
#define TOLRSD4_DEF (1.0e-08)
#define TOLRSD5_DEF (1.0e-08)
#define C1 (1.40e+00)
#define C2 (0.40e+00)
#define C3 (1.00e-01)
#define C4 (1.00e+00)
#define C5 (1.40e+00)
#define PROFILING_TOTAL_TIME (0)

#define PROFILING_ERHS_1 (1)
#define PROFILING_ERHS_2 (2)
#define PROFILING_ERHS_3 (3)
#define PROFILING_ERHS_4 (4)
#define PROFILING_ERROR (5)
#define PROFILING_NORM (6)
#define PROFILING_JACLD_BLTS (7)
#define PROFILING_JACU_BUTS (8)
#define PROFILING_L2NORM (9)
#define PROFILING_PINTGR_1 (10)
#define PROFILING_PINTGR_2 (11)
#define PROFILING_PINTGR_3 (12)
#define PROFILING_PINTGR_4 (13)
#define PROFILING_RHS_1 (14)
#define PROFILING_RHS_2 (15)
#define PROFILING_RHS_3 (16)
#define PROFILING_RHS_4 (17)
#define PROFILING_SETBV_1 (18)
#define PROFILING_SETBV_2 (19)
#define PROFILING_SETBV_3 (20)
#define PROFILING_SETIV (21)
#define PROFILING_SSOR_1 (22)
#define PROFILING_SSOR_2 (23)

/* gpu linear pattern */
#define u(m,i,j,k) u[(m)+5*((i)+nx*((j)+ny*(k)))]
#define v(m,i,j,k) v[(m)+5*((i)+nx*((j)+ny*(k)))]
#define rsd(m,i,j,k) rsd[(m)+5*((i)+nx*((j)+ny*(k)))]
#define frct(m,i,j,k) frct[(m)+5*((i)+nx*((j)+ny*(k)))]
#define rho_i(i,j,k) rho_i[(i)+nx*((j)+ny*(k))]
#define qs(i,j,k) qs[(i)+nx*((j)+ny*(k))]

/* global variables */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static double u_host[ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1][5];
static double rsd_host[ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1][5];
static double frct_host[ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1][5];
static double flux_host[ISIZ1][5];
static double qs_host[ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1];
static double rho_i_host[ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1];
static double a_host[ISIZ2][ISIZ1/2*2+1][5][5];
static double b_host[ISIZ2][ISIZ1/2*2+1][5][5];
static double c_host[ISIZ2][ISIZ1/2*2+1][5][5];
static double d_host[ISIZ2][ISIZ1/2*2+1][5][5];
static double ce_host[13][5];
#else
static double (*u_host)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]=(double(*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])malloc(sizeof(double)*((ISIZ3)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*(5)));
static double (*rsd_host)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]=(double(*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])malloc(sizeof(double)*((ISIZ3)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*(5)));
static double (*frct_host)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]=(double(*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])malloc(sizeof(double)*((ISIZ3)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*(5)));
static double (*flux_host)[5]=(double(*)[5])malloc(sizeof(double)*((ISIZ1)*(5)));
static double (*qs_host)[ISIZ2/2*2+1][ISIZ1/2*2+1]=(double(*)[ISIZ2/2*2+1][ISIZ1/2*2+1])malloc(sizeof(double)*((ISIZ3)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)));
static double (*rho_i_host)[ISIZ2/2*2+1][ISIZ1/2*2+1]=(double(*)[ISIZ2/2*2+1][ISIZ1/2*2+1])malloc(sizeof(double)*((ISIZ3)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)));
static double (*a_host)[ISIZ1/2*2+1][5][5]=(double(*)[ISIZ1/2*2+1][5][5])malloc(sizeof(double)*((ISIZ2)*(ISIZ1/2*2+1)*(5)*(5)));
static double (*b_host)[ISIZ1/2*2+1][5][5]=(double(*)[ISIZ1/2*2+1][5][5])malloc(sizeof(double)*((ISIZ2)*(ISIZ1/2*2+1)*(5)*(5)));
static double (*c_host)[ISIZ1/2*2+1][5][5]=(double(*)[ISIZ1/2*2+1][5][5])malloc(sizeof(double)*((ISIZ2)*(ISIZ1/2*2+1)*(5)*(5)));
static double (*d_host)[ISIZ1/2*2+1][5][5]=(double(*)[ISIZ1/2*2+1][5][5])malloc(sizeof(double)*((ISIZ2)*(ISIZ1/2*2+1)*(5)*(5)));
static double (*ce_host)[5]=(double(*)[5])malloc(sizeof(double)*((13)*(5)));
#endif
/* output control parameters */
static int ipr, inorm;
/* newton-raphson iteration control parameters */
static double dt_host, omega_host, tolrsd[5], rsdnm[5], errnm[5], frc;
static int itmax;
/* timer */
static double maxtime;
/* gpu variables */
static double* u_device;
static double* rsd_device;
static double* frct_device;
static double* rho_i_device;
static double* qs_device;
static double* norm_buffer_device;
static size_t size_u_device;
static size_t size_rsd_device;
static size_t size_frct_device;
static size_t size_rho_i_device;
static size_t size_qs_device;
static size_t size_norm_buffer_device;
static int nx;
static int ny;
static int nz;
static int THREADS_PER_BLOCK_ON_ERHS_1;
static int THREADS_PER_BLOCK_ON_ERHS_2;
static int THREADS_PER_BLOCK_ON_ERHS_3;
static int THREADS_PER_BLOCK_ON_ERHS_4;
static int THREADS_PER_BLOCK_ON_ERROR;
static int THREADS_PER_BLOCK_ON_NORM;
static int THREADS_PER_BLOCK_ON_JACLD_BLTS;
static int THREADS_PER_BLOCK_ON_JACU_BUTS;
static int THREADS_PER_BLOCK_ON_L2NORM;
static int THREADS_PER_BLOCK_ON_PINTGR_1;
static int THREADS_PER_BLOCK_ON_PINTGR_2;
static int THREADS_PER_BLOCK_ON_PINTGR_3;
static int THREADS_PER_BLOCK_ON_PINTGR_4;
static int THREADS_PER_BLOCK_ON_RHS_1;
static int THREADS_PER_BLOCK_ON_RHS_2;
static int THREADS_PER_BLOCK_ON_RHS_3;
static int THREADS_PER_BLOCK_ON_RHS_4;
static int THREADS_PER_BLOCK_ON_SETBV_1;
static int THREADS_PER_BLOCK_ON_SETBV_2;
static int THREADS_PER_BLOCK_ON_SETBV_3;
static int THREADS_PER_BLOCK_ON_SETIV;
static int THREADS_PER_BLOCK_ON_SSOR_1;
static int THREADS_PER_BLOCK_ON_SSOR_2;
int gpu_device_id;
int total_devices;
hipDeviceProp_t gpu_device_properties;
extern __shared__ double extern_share_data[];

namespace constants_device{
	/* coefficients of the exact solution */
	__constant__ double ce[13][5];
	/* grid */
	__constant__ double dxi, deta, dzeta;
	__constant__ double tx1, tx2, tx3;
	__constant__ double ty1, ty2, ty3;
	__constant__ double tz1, tz2, tz3;	
	/* dissipation */
	__constant__ double dx1, dx2, dx3, dx4, dx5;
	__constant__ double dy1, dy2, dy3, dy4, dy5;
	__constant__ double dz1, dz2, dz3, dz4, dz5;
	__constant__ double dssp;
	/* newton-raphson iteration control parameters */
	__constant__ double dt, omega;
}

/* function prototypes */
static void erhs_gpu();
__global__ static void erhs_gpu_kernel_1(double* frct,
		double* rsd,
		const int nx,
		const int ny,
		const int nz);
__global__ static void erhs_gpu_kernel_2(double* frct,
		const double* rsd,
		const int nx,
		const int ny,
		const int nz);
__global__ static void erhs_gpu_kernel_3(double* frct,
		const double* rsd,
		const int nx,
		const int ny,
		const int nz);
__global__ static void erhs_gpu_kernel_4(double* frct,
		const double* rsd,
		const int nx,
		const int ny,
		const int nz);
static void error_gpu();
__global__ static void error_gpu_kernel(const double* u,
		double* errnm,
		const int nx,
		const int ny,
		const int nz);
__device__ static void exact_gpu_device(const int i,
		const int j,
		const int k,
		double* u000ijk,
		const int nx,
		const int ny,
		const int nz);
__global__ static void jacld_blts_gpu_kernel(const int plane,
		const int klower,
		const int jlower,
		const double* u,
		const double* rho_i,
		const double* qs,
		double* v,
		const int nx,
		const int ny,
		const int nz);
__global__ static void jacu_buts_gpu_kernel(const int plane,
		const int klower,
		const int jlower,
		const double* u,
		const double* rho_i,
		const double* qs,
		double* v,
		const int nx,
		const int ny,
		const int nz);
static void l2norm_gpu(const double* v, 
		double* sum);
__global__ static void l2norm_gpu_kernel(const double* v,
		double* sum,
		const int nx,
		const int ny,
		const int nz);
__global__ static void norm_gpu_kernel(double* rms,
		const int size);
static void pintgr_gpu();
__global__ static void pintgr_gpu_kernel_1(const double* u,
		double* frc,
		const int nx,
		const int ny,
		const int nz);
__global__ static void pintgr_gpu_kernel_2(const double* u,
		double* frc,
		const int nx,
		const int ny,
		const int nz);
__global__ static void pintgr_gpu_kernel_3(const double* u,
		double* frc,
		const int nx,
		const int ny,
		const int nz);
__global__ static void pintgr_gpu_kernel_4(double* frc,
		const int num);
static void read_input();
static void release_gpu();
static void rhs_gpu();
__global__ static void rhs_gpu_kernel_1(const double* u,
		double* rsd,
		const double* frct,
		double* qs,
		double* rho_i,
		const int nx,
		const int ny,
		const int nz);
__global__ static void rhs_gpu_kernel_2(const double* u,
		double* rsd,
		const double* qs,
		const double* rho_i,
		const int nx,
		const int ny,
		const int nz);
__global__ static void rhs_gpu_kernel_3(const double* u,
		double* rsd,
		const double* qs,
		const double* rho_i,
		const int nx,
		const int ny,
		const int nz);
__global__ static void rhs_gpu_kernel_4(const double* u,
		double* rsd,
		const double* qs,
		const double* rho_i,
		const int nx,
		const int ny,
		const int nz);
static void setbv_gpu();
__global__ static void setbv_gpu_kernel_1(double* u,
		const int nx,
		const int ny,
		const int nz);
__global__ static void setbv_gpu_kernel_2(double* u,
		const int nx,
		const int ny,
		const int nz);
__global__ static void setbv_gpu_kernel_3(double* u,
		const int nx,
		const int ny,
		const int nz);
static void setcoeff_gpu();
static void setiv_gpu();
__global__ static void setiv_gpu_kernel(double* u,
		const int nx,
		const int ny,
		const int nz);
static void setup_gpu();
static void ssor_gpu(int niter);
__global__ static void ssor_gpu_kernel_1(double* rsd,
		const int nx,
		const int ny,
		const int nz);
__global__ static void ssor_gpu_kernel_2(double* u,
		double* rsd,
		const double tmp,
		const int nx,
		const int ny,
		const int nz);
static void verify_gpu(double xcr[],
		double xce[],
		double xci,
		char* class_npb,
		boolean* verified);

/* lu */
int main(int argc, char** argv){
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
	printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif
#if defined(PROFILING)
	printf(" PROFILING mode on\n");
#endif
	char class_npb;
	boolean verified;
	double mflops;	
	/*
	 * ---------------------------------------------------------------------
	 * read input data
	 * ---------------------------------------------------------------------
	 */
	read_input();	
	/*
	 * ---------------------------------------------------------------------
	 * set up coefficients
	 * ---------------------------------------------------------------------
	 */
	setup_gpu();
	setcoeff_gpu();
	/*
	 * ---------------------------------------------------------------------
	 * set the boundary values for dependent variables
	 * ---------------------------------------------------------------------
	 */
	setbv_gpu();
	/*
	 * ---------------------------------------------------------------------
	 * set the initial values for dependent variables
	 * ---------------------------------------------------------------------
	 */
	setiv_gpu();
	/*
	 * ---------------------------------------------------------------------
	 * compute the forcing term based on prescribed exact solution
	 * ---------------------------------------------------------------------
	 */
	erhs_gpu();
	/*
	 * ---------------------------------------------------------------------
	 * perform one SSOR iteration to touch all pages
	 * ---------------------------------------------------------------------
	 */
	ssor_gpu(1);
	/*
	 * ---------------------------------------------------------------------
	 * reset the boundary and initial values
	 * ---------------------------------------------------------------------
	 */
	setbv_gpu();
	setiv_gpu();
	/*
	 * ---------------------------------------------------------------------
	 * perform the SSOR iterations
	 * ---------------------------------------------------------------------
	 */
	ssor_gpu(itmax);
	/*
	 * ---------------------------------------------------------------------
	 * compute the solution error
	 * ---------------------------------------------------------------------
	 */
	error_gpu();
	/*
	 * ---------------------------------------------------------------------
	 * compute the surface integral
	 * ---------------------------------------------------------------------
	 */
	pintgr_gpu();
	/*
	 * ---------------------------------------------------------------------
	 * verification test
	 * ---------------------------------------------------------------------
	 */
	verify_gpu(rsdnm, errnm, frc, &class_npb, &verified);
	mflops=(double)itmax*(1984.77*(double)nx
			*(double)ny
			*(double)nz
			-10923.3*pow(((double)(nx+ny+nz)/3.0),2.0) 
			+27770.9*(double)(nx+ny+nz)/3.0
			-144010.0)
		/(maxtime*1000000.0);	

	char gpu_config[256];
	char gpu_config_string[4096];
#if defined(PROFILING)
	sprintf(gpu_config, "%5s\t%25s\t%25s\t%25s\n", "GPU Kernel", "Threads Per Block", "Time in Seconds", "Time in Percentage");
	strcpy(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-erhs-1", THREADS_PER_BLOCK_ON_ERHS_1, timer_read(PROFILING_ERHS_1), (timer_read(PROFILING_ERHS_1)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-erhs-2", THREADS_PER_BLOCK_ON_ERHS_2, timer_read(PROFILING_ERHS_2), (timer_read(PROFILING_ERHS_2)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-erhs-3", THREADS_PER_BLOCK_ON_ERHS_3, timer_read(PROFILING_ERHS_3), (timer_read(PROFILING_ERHS_3)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-erhs-4", THREADS_PER_BLOCK_ON_ERHS_4, timer_read(PROFILING_ERHS_4), (timer_read(PROFILING_ERHS_4)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-error", THREADS_PER_BLOCK_ON_ERROR, timer_read(PROFILING_ERROR), (timer_read(PROFILING_ERROR)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-norm", THREADS_PER_BLOCK_ON_NORM, timer_read(PROFILING_NORM), (timer_read(PROFILING_NORM)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-jacld-blts", THREADS_PER_BLOCK_ON_JACLD_BLTS, timer_read(PROFILING_JACLD_BLTS), (timer_read(PROFILING_JACLD_BLTS)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-jacu-buts", THREADS_PER_BLOCK_ON_JACU_BUTS, timer_read(PROFILING_JACU_BUTS), (timer_read(PROFILING_JACU_BUTS)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-l2norm", THREADS_PER_BLOCK_ON_L2NORM, timer_read(PROFILING_L2NORM), (timer_read(PROFILING_L2NORM)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-pintgr-1", THREADS_PER_BLOCK_ON_PINTGR_1, timer_read(PROFILING_PINTGR_1), (timer_read(PROFILING_PINTGR_1)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-pintgr-2", THREADS_PER_BLOCK_ON_PINTGR_2, timer_read(PROFILING_PINTGR_2), (timer_read(PROFILING_PINTGR_2)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-pintgr-3", THREADS_PER_BLOCK_ON_PINTGR_3, timer_read(PROFILING_PINTGR_3), (timer_read(PROFILING_PINTGR_3)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-pintgr-4", THREADS_PER_BLOCK_ON_PINTGR_4, timer_read(PROFILING_PINTGR_4), (timer_read(PROFILING_PINTGR_4)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-rhs-1", THREADS_PER_BLOCK_ON_RHS_1, timer_read(PROFILING_RHS_1), (timer_read(PROFILING_RHS_1)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-rhs-2", THREADS_PER_BLOCK_ON_RHS_2, timer_read(PROFILING_RHS_2), (timer_read(PROFILING_RHS_2)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-rhs-3", THREADS_PER_BLOCK_ON_RHS_3, timer_read(PROFILING_RHS_3), (timer_read(PROFILING_RHS_3)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-rhs-4", THREADS_PER_BLOCK_ON_RHS_4, timer_read(PROFILING_RHS_4), (timer_read(PROFILING_RHS_4)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-setbv-1", THREADS_PER_BLOCK_ON_SETBV_1, timer_read(PROFILING_SETBV_1), (timer_read(PROFILING_SETBV_1)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-setbv-2", THREADS_PER_BLOCK_ON_SETBV_2, timer_read(PROFILING_SETBV_2), (timer_read(PROFILING_SETBV_2)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-setbv-3", THREADS_PER_BLOCK_ON_SETBV_3, timer_read(PROFILING_SETBV_3), (timer_read(PROFILING_SETBV_3)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-setiv", THREADS_PER_BLOCK_ON_SETIV, timer_read(PROFILING_SETIV), (timer_read(PROFILING_SETIV)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-ssor-1", THREADS_PER_BLOCK_ON_SSOR_1, timer_read(PROFILING_SSOR_1), (timer_read(PROFILING_SSOR_1)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " lu-ssor-2", THREADS_PER_BLOCK_ON_SSOR_2, timer_read(PROFILING_SSOR_2), (timer_read(PROFILING_SSOR_2)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
#else
	sprintf(gpu_config, "%5s\t%25s\n", "GPU Kernel", "Threads Per Block");
	strcpy(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-erhs-1", THREADS_PER_BLOCK_ON_ERHS_1);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-erhs-2", THREADS_PER_BLOCK_ON_ERHS_2);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-erhs-3", THREADS_PER_BLOCK_ON_ERHS_3);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-erhs-4", THREADS_PER_BLOCK_ON_ERHS_4);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-error", THREADS_PER_BLOCK_ON_ERROR);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-norm", THREADS_PER_BLOCK_ON_NORM);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-jacld-blts", THREADS_PER_BLOCK_ON_JACLD_BLTS);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-jacu-buts", THREADS_PER_BLOCK_ON_JACU_BUTS);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-l2norm", THREADS_PER_BLOCK_ON_L2NORM);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-pintgr-1", THREADS_PER_BLOCK_ON_PINTGR_1);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-pintgr-2", THREADS_PER_BLOCK_ON_PINTGR_2);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-pintgr-3", THREADS_PER_BLOCK_ON_PINTGR_3);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-pintgr-4", THREADS_PER_BLOCK_ON_PINTGR_4);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-rhs-1", THREADS_PER_BLOCK_ON_RHS_1);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-rhs-2", THREADS_PER_BLOCK_ON_RHS_2);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-rhs-3", THREADS_PER_BLOCK_ON_RHS_3);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-rhs-4", THREADS_PER_BLOCK_ON_RHS_4);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-setbv-1", THREADS_PER_BLOCK_ON_SETBV_1);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-setbv-2", THREADS_PER_BLOCK_ON_SETBV_2);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-setbv-3", THREADS_PER_BLOCK_ON_SETBV_3);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-setiv", THREADS_PER_BLOCK_ON_SETIV);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-ssor-1", THREADS_PER_BLOCK_ON_SSOR_1);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " lu-ssor-2", THREADS_PER_BLOCK_ON_SSOR_2);
	strcat(gpu_config_string, gpu_config);
#endif
	c_print_results((char*)"LU",
			class_npb,
			nx,
			ny,
			nz,
			itmax,
			maxtime,
			mflops,
			(char*)"          floating point",
			verified,
			(char*)NPBVERSION,
			(char*)COMPILETIME,
			(char*)COMPILERVERSION,
			(char*)LIBVERSION,
			(char*)CPU_MODEL,
			(char*)gpu_device_properties.name,
			gpu_config_string,
			(char*)CS1,
			(char*)CS2,
			(char*)CS3,
			(char*)CS4,
			(char*)CS5,
			(char*)CS6,
			(char*)"(none)");
	release_gpu();
	return 0;
}

/*
 * ---------------------------------------------------------------------
 * compute the right hand side based on exact solution
 * ---------------------------------------------------------------------
 */
static void erhs_gpu(){
#if defined(PROFILING)
	timer_start(PROFILING_ERHS_1);
#endif
	/* #KERNEL ERHS 1 */
	int erhs_1_workload = nx * ny * nz;
	int erhs_1_threads_per_block = THREADS_PER_BLOCK_ON_ERHS_1;
	int erhs_1_blocks_per_grid = (ceil((double)erhs_1_workload/(double)erhs_1_threads_per_block));

	erhs_gpu_kernel_1<<<
		erhs_1_blocks_per_grid, 
		erhs_1_threads_per_block>>>(
				frct_device, 
				rsd_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_ERHS_1);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_ERHS_2);
#endif
	/* #KERNEL ERHS 2 */
	int erhs_2_threads_per_block;
	dim3 erhs_2_blocks_per_grid(nz-2, ny-2);
	if(THREADS_PER_BLOCK_ON_ERHS_2 != gpu_device_properties.warpSize){
		erhs_2_threads_per_block = gpu_device_properties.warpSize;
	}
	else{
		erhs_2_threads_per_block = THREADS_PER_BLOCK_ON_ERHS_2;
	}

	erhs_gpu_kernel_2<<<
		erhs_2_blocks_per_grid, 
		(min(nx,erhs_2_threads_per_block)),
		sizeof(double)*((2*(min(nx,erhs_2_threads_per_block))*5)+(4*(min(nx,erhs_2_threads_per_block))))>>>(
				frct_device, 
				rsd_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_ERHS_2);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_ERHS_3);
#endif
	/* #KERNEL ERHS 3 */
	int erhs_3_threads_per_block;
	dim3 erhs_3_blocks_per_grid(nz-2, nx-2);
	if(THREADS_PER_BLOCK_ON_ERHS_3 != gpu_device_properties.warpSize){
		erhs_3_threads_per_block = gpu_device_properties.warpSize;
	}
	else{
		erhs_3_threads_per_block = THREADS_PER_BLOCK_ON_ERHS_3;
	}

	erhs_gpu_kernel_3<<<
		erhs_3_blocks_per_grid, 
		(min(ny,erhs_3_threads_per_block)),
		sizeof(double)*((2*(min(ny,erhs_3_threads_per_block))*5)+(4*(min(ny,erhs_3_threads_per_block))))>>>(
				frct_device, 
				rsd_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_ERHS_3);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_ERHS_4);
#endif
	/* #KERNEL ERHS 4 */
	int erhs_4_threads_per_block;
	dim3 erhs_4_blocks_per_grid(ny-2, nx-2);
	if(THREADS_PER_BLOCK_ON_ERHS_4 != gpu_device_properties.warpSize){
		erhs_4_threads_per_block = gpu_device_properties.warpSize;
	}
	else{
		erhs_4_threads_per_block = THREADS_PER_BLOCK_ON_ERHS_4;
	}

	erhs_gpu_kernel_4<<<
		erhs_4_blocks_per_grid, 
		(min(nz,erhs_4_threads_per_block)),
		sizeof(double)*((2*(min(nz,erhs_4_threads_per_block))*5)+(4*(min(nz,erhs_4_threads_per_block))))>>>(
				frct_device, 
				rsd_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_ERHS_4);
#endif
}

__global__ static void erhs_gpu_kernel_1(double* frct,
		double* rsd,
		const int nx,
		const int ny,
		const int nz){
	int i_j_k, i, j, k, m;
	double xi, eta, zeta;

	i_j_k = blockIdx.x * blockDim.x + threadIdx.x;

	i = i_j_k % nx;
	j = (i_j_k / nx) % ny;
	k = i_j_k / (nx * ny);

	if(i_j_k >= (nx*ny*nz)){
		return;
	}

	using namespace constants_device;
	for(m=0;m<5;m++){frct(m,i,j,k)=0.0;}
	zeta=(double)k/((double)(nz-1));
	eta=(double)j/((double)(ny-1));
	xi=(double)i/((double)(nx-1));	
	for(m=0;m<5;m++){rsd(m,i,j,k)=ce[0][m]+
		(ce[1][m]+
		 (ce[4][m]+
		  (ce[7][m]+
		   ce[10][m]*xi)*xi)*xi)*xi+
			(ce[2][m]+
			 (ce[5][m]+
			  (ce[8][m]+
			   ce[11][m]*eta)*eta)*eta)*eta+
			(ce[3][m]+
			 (ce[6][m]+
			  (ce[9][m]+
			   ce[12][m]*zeta)*zeta)*zeta)*zeta;
	}
}

__global__ static void erhs_gpu_kernel_2(double* frct,
		const double* rsd,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m, nthreads;
	double q, u21;

	double* flux = (double*)extern_share_data;
	double* rtmp = (double*)flux+(blockDim.x*5);
	double* u21i = (double*)rtmp+(blockDim.x*5);
	double* u31i = (double*)u21i+(blockDim.x);
	double* u41i = (double*)u31i+(blockDim.x);
	double* u51i = (double*)u41i+(blockDim.x);

	double utmp[5];

	k=blockIdx.x+1;
	j=blockIdx.y+1;
	i=threadIdx.x;

	using namespace constants_device;
	while(i < nx){
		nthreads=nx-(i-threadIdx.x);
		if(nthreads>blockDim.x){nthreads=blockDim.x;}
		m=threadIdx.x;
		rtmp[m]=rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m+=nthreads;
		rtmp[m]=rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m+=nthreads;
		rtmp[m]=rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m+=nthreads;
		rtmp[m]=rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m+=nthreads;
		rtmp[m]=rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		__syncthreads();
		/*
		 * ---------------------------------------------------------------------
		 * xi-direction flux differences
		 * ---------------------------------------------------------------------
		 */
		flux[threadIdx.x+(0*blockDim.x)]=rtmp[threadIdx.x*5+1];
		u21=rtmp[threadIdx.x*5+1]/rtmp[threadIdx.x*5+0];
		q=0.5*(rtmp[threadIdx.x*5+1]*rtmp[threadIdx.x*5+1]+rtmp[threadIdx.x*5+2]*rtmp[threadIdx.x*5+2]+rtmp[threadIdx.x*5+3]*rtmp[threadIdx.x*5+3])/rtmp[threadIdx.x*5+0];
		flux[threadIdx.x+(1*blockDim.x)]=rtmp[threadIdx.x*5+1]*u21+C2*(rtmp[threadIdx.x*5+4]-q);
		flux[threadIdx.x+(2*blockDim.x)]=rtmp[threadIdx.x*5+2]*u21;
		flux[threadIdx.x+(3*blockDim.x)]=rtmp[threadIdx.x*5+3]*u21;
		flux[threadIdx.x+(4*blockDim.x)]=(C1*rtmp[threadIdx.x*5+4]-C2*q)*u21;
		__syncthreads();
		if((threadIdx.x>=1)&&(threadIdx.x<(blockDim.x-1))&&(i<(nx-1))){for(m=0;m<5;m++){utmp[m]=frct(m,i,j,k)-tx2*(flux[(threadIdx.x+1)+(m*blockDim.x)]-flux[(threadIdx.x-1)+(m*blockDim.x)]);}}
		u21=1.0/rtmp[threadIdx.x*5+0];
		u21i[threadIdx.x]=u21*rtmp[threadIdx.x*5+1];
		u31i[threadIdx.x]=u21*rtmp[threadIdx.x*5+2];
		u41i[threadIdx.x]=u21*rtmp[threadIdx.x*5+3];
		u51i[threadIdx.x]=u21*rtmp[threadIdx.x*5+4];
		__syncthreads();
		if(threadIdx.x>=1){
			flux[threadIdx.x+(1*blockDim.x)]=(4.0/3.0)*tx3*(u21i[threadIdx.x]-u21i[threadIdx.x-1]);
			flux[threadIdx.x+(2*blockDim.x)]=tx3*(u31i[threadIdx.x]-u31i[threadIdx.x-1]);
			flux[threadIdx.x+(3*blockDim.x)]=tx3*(u41i[threadIdx.x]-u41i[threadIdx.x-1]);
			flux[threadIdx.x+(4*blockDim.x)]=0.5*(1.0-C1*C5)*tx3*((u21i[threadIdx.x]*u21i[threadIdx.x]+u31i[threadIdx.x]*u31i[threadIdx.x]+u41i[threadIdx.x]*u41i[threadIdx.x])-(u21i[threadIdx.x-1]*u21i[threadIdx.x-1]+u31i[threadIdx.x-1]*u31i[threadIdx.x-1]+u41i[threadIdx.x-1]*u41i[threadIdx.x-1]))+(1.0/6.0)*tx3*(u21i[threadIdx.x]*u21i[threadIdx.x]-u21i[threadIdx.x-1]*u21i[threadIdx.x-1])+C1*C5*tx3*(u51i[threadIdx.x]-u51i[threadIdx.x-1]);
		}
		__syncthreads();
		if((threadIdx.x>=1)&&(threadIdx.x<(blockDim.x-1))&&(i<nx-1)){
			utmp[0]+=dx1*tx1*(rtmp[threadIdx.x*5-5]-2.0*rtmp[threadIdx.x*5+0]+rtmp[threadIdx.x*5+5]);
			utmp[1]+=tx3*C3*C4*(flux[(threadIdx.x+1)+(1*blockDim.x)]-flux[threadIdx.x+(1*blockDim.x)])+dx2*tx1*(rtmp[threadIdx.x*5-4]-2.0*rtmp[threadIdx.x*5+1]+rtmp[threadIdx.x*5+6]);
			utmp[2]+=tx3*C3*C4*(flux[(threadIdx.x+1)+(2*blockDim.x)]-flux[threadIdx.x+(2*blockDim.x)])+dx3*tx1*(rtmp[threadIdx.x*5-3]-2.0*rtmp[threadIdx.x*5+2]+rtmp[threadIdx.x*5+7]);
			utmp[3]+=tx3*C3*C4*(flux[(threadIdx.x+1)+(3*blockDim.x)]-flux[threadIdx.x+(3*blockDim.x)])+dx4*tx1*(rtmp[threadIdx.x*5-2]-2.0*rtmp[threadIdx.x*5+3]+rtmp[threadIdx.x*5+8]);
			utmp[4]+=tx3*C3*C4*(flux[(threadIdx.x+1)+(4*blockDim.x)]-flux[threadIdx.x+(4*blockDim.x)])+dx5*tx1*(rtmp[threadIdx.x*5-1]-2.0*rtmp[threadIdx.x*5+4]+rtmp[threadIdx.x*5+9]);
			/*
			 * ---------------------------------------------------------------------
			 * fourth-order dissipation
			 * ---------------------------------------------------------------------
			 */
			if(i==1){for(m=0;m<5;m++){frct(m,1,j,k)=utmp[m]-dssp*(+5.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,3,j,k));}}
			if(i==2){for(m=0;m<5;m++){frct(m,2,j,k)=utmp[m]-dssp*(-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,4,j,k));}}
			if((i>=3)&&(i<(nx-3))){for(m=0;m<5;m++){frct(m,i,j,k)=utmp[m]-dssp*(rsd(m,i-2,j,k)-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i+2,j,k));}}
			if(i==(nx-3)){for(m=0;m<5;m++){frct(m,nx-3,j,k)=utmp[m]-dssp*(rsd(m,nx-5,j,k)-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]);}}
			if(i==(nx-2)){for(m=0;m<5;m++){frct(m,nx-2,j,k)=utmp[m]-dssp*(rsd(m,nx-4,j,k)-4.0*rtmp[threadIdx.x*5+m-5]+5.0*rtmp[threadIdx.x*5+m]);}}
		}
		i+=blockDim.x-2;
	}
}

__global__ static void erhs_gpu_kernel_3(double* frct,
		const double* rsd,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m, nthreads;
	double q, u31;

	double* flux = (double*)extern_share_data;
	double* rtmp = (double*)flux+(blockDim.x*5);
	double* u21j = (double*)rtmp+(blockDim.x*5);
	double* u31j = (double*)u21j+(blockDim.x);
	double* u41j = (double*)u31j+(blockDim.x);
	double* u51j = (double*)u41j+(blockDim.x);

	double utmp[5];

	k=blockIdx.x+1;
	i=blockIdx.y+1;
	j=threadIdx.x;

	using namespace constants_device;
	while(j<ny){ 
		nthreads=ny-(j-threadIdx.x);
		if(nthreads>blockDim.x){nthreads=blockDim.x;}
		m=threadIdx.x;
		rtmp[m]=rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m+=nthreads;
		rtmp[m]=rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m+=nthreads;
		rtmp[m]=rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m+=nthreads;
		rtmp[m]=rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m+=nthreads;
		rtmp[m]=rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		__syncthreads();
		/*
		 * ---------------------------------------------------------------------
		 * eta-direction flux differences
		 * ---------------------------------------------------------------------
		 */
		flux[threadIdx.x+(0*blockDim.x)]=rtmp[threadIdx.x*5+2];
		u31=rtmp[threadIdx.x*5+2]/rtmp[threadIdx.x*5+0];
		q=0.5*(rtmp[threadIdx.x*5+1]*rtmp[threadIdx.x*5+1]+rtmp[threadIdx.x*5+2]*rtmp[threadIdx.x*5+2]+rtmp[threadIdx.x*5+3]*rtmp[threadIdx.x*5+3])/rtmp[threadIdx.x*5+0];
		flux[threadIdx.x+(1*blockDim.x)]=rtmp[threadIdx.x*5+1]*u31;
		flux[threadIdx.x+(2*blockDim.x)]=rtmp[threadIdx.x*5+2]*u31+C2*(rtmp[threadIdx.x*5+4]-q);
		flux[threadIdx.x+(3*blockDim.x)]=rtmp[threadIdx.x*5+3]*u31;
		flux[threadIdx.x+(4*blockDim.x)]=(C1*rtmp[threadIdx.x*5+4]-C2*q)*u31;
		__syncthreads();
		if((threadIdx.x>=1)&&(threadIdx.x<(blockDim.x-1))&&(j<(ny-1))){for(m=0;m<5;m++){utmp[m]=frct(m,i,j,k)-ty2*(flux[(threadIdx.x+1)+(m*blockDim.x)]-flux[(threadIdx.x-1)+(m*blockDim.x)]);}}
		u31=1.0/rtmp[threadIdx.x*5+0];
		u21j[threadIdx.x]=u31*rtmp[threadIdx.x*5+1];
		u31j[threadIdx.x]=u31*rtmp[threadIdx.x*5+2];
		u41j[threadIdx.x]=u31*rtmp[threadIdx.x*5+3];
		u51j[threadIdx.x]=u31*rtmp[threadIdx.x*5+4];
		__syncthreads();
		if(threadIdx.x>=1){
			flux[threadIdx.x+(1*blockDim.x)]=ty3*(u21j[threadIdx.x]-u21j[threadIdx.x-1]);
			flux[threadIdx.x+(2*blockDim.x)]=(4.0/3.0)*ty3*(u31j[threadIdx.x]-u31j[threadIdx.x-1]);
			flux[threadIdx.x+(3*blockDim.x)]=ty3*(u41j[threadIdx.x]-u41j[threadIdx.x-1]);
			flux[threadIdx.x+(4*blockDim.x)]=0.5*(1.0-C1*C5)*ty3*((u21j[threadIdx.x]*u21j[threadIdx.x]+u31j[threadIdx.x]*u31j[threadIdx.x]+u41j[threadIdx.x]*u41j[threadIdx.x])-(u21j[threadIdx.x-1]*u21j[threadIdx.x-1]+u31j[threadIdx.x-1]*u31j[threadIdx.x-1]+u41j[threadIdx.x-1]*u41j[threadIdx.x-1]))+(1.0/6.0)*ty3*(u31j[threadIdx.x]*u31j[threadIdx.x]-u31j[threadIdx.x-1]*u31j[threadIdx.x-1])+C1*C5*ty3*(u51j[threadIdx.x]-u51j[threadIdx.x-1]);
		}
		__syncthreads();
		if((threadIdx.x>=1)&&(threadIdx.x<(blockDim.x-1))&&(j<(ny-1))){
			utmp[0]+=dy1*ty1*(rtmp[threadIdx.x*5-5]-2.0*rtmp[threadIdx.x*5+0]+rtmp[threadIdx.x*5+5]);
			utmp[1]+=ty3*C3*C4*(flux[(threadIdx.x+1)+(1*blockDim.x)]-flux[threadIdx.x+(1*blockDim.x)])+dy2*ty1*(rtmp[threadIdx.x*5-4]-2.0*rtmp[threadIdx.x*5+1]+rtmp[threadIdx.x*5+6]);
			utmp[2]+=ty3*C3*C4*(flux[(threadIdx.x+1)+(2*blockDim.x)]-flux[threadIdx.x+(2*blockDim.x)])+dy3*ty1*(rtmp[threadIdx.x*5-3]-2.0*rtmp[threadIdx.x*5+2]+rtmp[threadIdx.x*5+7]);
			utmp[3]+=ty3*C3*C4*(flux[(threadIdx.x+1)+(3*blockDim.x)]-flux[threadIdx.x+(3*blockDim.x)])+dy4*ty1*(rtmp[threadIdx.x*5-2]-2.0*rtmp[threadIdx.x*5+3]+rtmp[threadIdx.x*5+8]);
			utmp[4]+=ty3*C3*C4*(flux[(threadIdx.x+1)+(4*blockDim.x)]-flux[threadIdx.x+(4*blockDim.x)])+dy5*ty1*(rtmp[threadIdx.x*5-1]-2.0*rtmp[threadIdx.x*5+4]+rtmp[threadIdx.x*5+9]);
			/*
			 * ---------------------------------------------------------------------
			 * fourth-order dissipation
			 * ---------------------------------------------------------------------
			 */
			if(j==1){for(m=0;m<5;m++){frct(m,i,1,k)=utmp[m]-dssp*(+5.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i,3,k));}}
			if(j==2){for(m=0;m<5;m++){frct(m,i,2,k)=utmp[m]-dssp*(-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i,4,k));}}
			if((j>=3)&&(j<(ny-3))){for(m=0;m<5;m++){frct(m,i,j,k)=utmp[m]-dssp*(rsd(m,i,j-2,k)-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i,j+2,k));}}
			if(j==(ny-3)){for(m=0;m<5;m++){frct(m,i,ny-3,k)=utmp[m]-dssp*(rsd(m,i,ny-5,k)-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]);}}
			if(j==(ny-2)){for(m=0;m<5;m++){frct(m,i,ny-2,k)=utmp[m]-dssp*(rsd(m,i,ny-4,k)-4.0*rtmp[threadIdx.x*5+m-5]+5.0*rtmp[threadIdx.x*5+m]);}}
		}
		j += blockDim.x-2;
	}
}

__global__ static void erhs_gpu_kernel_4(double* frct,
		const double* rsd,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m, nthreads;
	double q, u41;

	double* flux = (double*)extern_share_data;
	double* rtmp = (double*)flux+(blockDim.x*5);
	double* u21k = (double*)rtmp+(blockDim.x*5);
	double* u31k = (double*)u21k+(blockDim.x);
	double* u41k = (double*)u31k+(blockDim.x);
	double* u51k = (double*)u41k+(blockDim.x);

	double utmp[5];

	j=blockIdx.x+1;
	i=blockIdx.y+1;
	k=threadIdx.x;

	using namespace constants_device;
	while(k<nz){
		nthreads=(nz-(k-threadIdx.x));
		if(nthreads>blockDim.x){nthreads=blockDim.x;}
		m=threadIdx.x;
		rtmp[m]=rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m+=nthreads;
		rtmp[m]=rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m+=nthreads;
		rtmp[m]=rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m+=nthreads;
		rtmp[m]=rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m+=nthreads;
		rtmp[m]=rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		__syncthreads();
		/*
		 * ---------------------------------------------------------------------
		 * zeta-direction flux differences
		 * ---------------------------------------------------------------------
		 */
		flux[threadIdx.x+(0*blockDim.x)]=rtmp[threadIdx.x*5+3];
		u41=rtmp[threadIdx.x*5+3]/rtmp[threadIdx.x*5+0];
		q=0.5*(rtmp[threadIdx.x*5+1]*rtmp[threadIdx.x*5+1]+rtmp[threadIdx.x*5+2]*rtmp[threadIdx.x*5+2]+rtmp[threadIdx.x*5+3]*rtmp[threadIdx.x*5+3])/rtmp[threadIdx.x*5+0];
		flux[threadIdx.x+(1*blockDim.x)]=rtmp[threadIdx.x*5+1]*u41;
		flux[threadIdx.x+(2*blockDim.x)]=rtmp[threadIdx.x*5+2]*u41;
		flux[threadIdx.x+(3*blockDim.x)]=rtmp[threadIdx.x*5+3]*u41+C2*(rtmp[threadIdx.x*5+4]-q);
		flux[threadIdx.x+(4*blockDim.x)]=(C1*rtmp[threadIdx.x*5+4]-C2*q)*u41;
		__syncthreads();
		if((threadIdx.x>=1)&&(threadIdx.x<(blockDim.x-1))&&(k<(nz-1))){for(m=0;m<5;m++){utmp[m]=frct(m,i,j,k)-tz2*(flux[(threadIdx.x+1)+(m*blockDim.x)]-flux[(threadIdx.x-1)+(m*blockDim.x)]);}}
		u41=1.0/rtmp[threadIdx.x*5+0];
		u21k[threadIdx.x]=u41*rtmp[threadIdx.x*5+1];
		u31k[threadIdx.x]=u41*rtmp[threadIdx.x*5+2];
		u41k[threadIdx.x]=u41*rtmp[threadIdx.x*5+3];
		u51k[threadIdx.x]=u41*rtmp[threadIdx.x*5+4];
		__syncthreads();
		if(threadIdx.x>=1){
			flux[threadIdx.x+(1*blockDim.x)]=tz3*(u21k[threadIdx.x]-u21k[threadIdx.x-1]);
			flux[threadIdx.x+(2*blockDim.x)]=tz3*(u31k[threadIdx.x]-u31k[threadIdx.x-1]);
			flux[threadIdx.x+(3*blockDim.x)]=(4.0/3.0)*tz3*(u41k[threadIdx.x]-u41k[threadIdx.x-1]);
			flux[threadIdx.x+(4*blockDim.x)]=0.5*(1.0-C1*C5)*tz3*((u21k[threadIdx.x]*u21k[threadIdx.x]+u31k[threadIdx.x]*u31k[threadIdx.x]+u41k[threadIdx.x]*u41k[threadIdx.x])-(u21k[threadIdx.x-1]*u21k[threadIdx.x-1]+u31k[threadIdx.x-1]*u31k[threadIdx.x-1]+u41k[threadIdx.x-1]*u41k[threadIdx.x-1]))+(1.0/6.0)*tz3*(u41k[threadIdx.x]*u41k[threadIdx.x]-u41k[threadIdx.x-1]*u41k[threadIdx.x-1])+C1*C5*tz3*(u51k[threadIdx.x]-u51k[threadIdx.x-1]);
		}
		__syncthreads();
		if((threadIdx.x>=1)&&(threadIdx.x<(blockDim.x-1))&&(k<(nz-1))){
			utmp[0]+=dz1*tz1*(rtmp[threadIdx.x*5-5]-2.0*rtmp[threadIdx.x*5+0]+rtmp[threadIdx.x*5+5]);
			utmp[1]+=tz3*C3*C4*(flux[(threadIdx.x+1)+(1*blockDim.x)]-flux[threadIdx.x+(1*blockDim.x)])+dz2*tz1*(rtmp[threadIdx.x*5-4]-2.0*rtmp[threadIdx.x*5+1]+rtmp[threadIdx.x*5+6]);
			utmp[2]+=tz3*C3*C4*(flux[(threadIdx.x+1)+(2*blockDim.x)]-flux[threadIdx.x+(2*blockDim.x)])+dz3*tz1*(rtmp[threadIdx.x*5-3]-2.0*rtmp[threadIdx.x*5+2]+rtmp[threadIdx.x*5+7]);
			utmp[3]+=tz3*C3*C4*(flux[(threadIdx.x+1)+(3*blockDim.x)]-flux[threadIdx.x+(3*blockDim.x)])+dz4*tz1*(rtmp[threadIdx.x*5-2]-2.0*rtmp[threadIdx.x*5+3]+rtmp[threadIdx.x*5+8]);
			utmp[4]+=tz3*C3*C4*(flux[(threadIdx.x+1)+(4*blockDim.x)]-flux[threadIdx.x+(4*blockDim.x)])+dz5*tz1*(rtmp[threadIdx.x*5-1]-2.0*rtmp[threadIdx.x*5+4]+rtmp[threadIdx.x*5+9]);
			/*
			 * ---------------------------------------------------------------------
			 * fourth-order dissipation
			 * ---------------------------------------------------------------------
			 */
			if(k==1){for(m=0;m<5;m++){frct(m,i,j,1)=utmp[m]-dssp*(+5.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i,j,3));}}
			if(k==2){for(m=0;m<5;m++){frct(m,i,j,2)=utmp[m]-dssp*(-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i,j,4));}}
			if((k>=3)&&(k<(nz-3))){for(m=0;m<5;m++){frct(m,i,j,k)=utmp[m]-dssp*(rsd(m,i,j,k-2)-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i,j,k+2));}}
			if(k==(nz-3)){for(m=0;m<5;m++){frct(m,i,j,nz-3)=utmp[m]-dssp*(rsd(m,i,j,nz-5)-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]);}}
			if(k==(nz-2)){for(m=0;m<5;m++){frct(m,i,j,nz-2)=utmp[m]-dssp*(rsd(m,i,j,nz-4)-4.0*rtmp[threadIdx.x*5+m-5]+5.0*rtmp[threadIdx.x*5+m]);}}
		}
		k+=blockDim.x-2;
	}
}

/*
 * ---------------------------------------------------------------------
 * compute the solution error
 * ---------------------------------------------------------------------
 */
static void error_gpu(){
	dim3 grid(nz-2, ny-2);

#if defined(PROFILING)
	timer_start(PROFILING_ERROR);
#endif
	/* #KERNEL ERROR */
	int error_threads_per_block=THREADS_PER_BLOCK_ON_ERROR;
	dim3 error_blocks_per_grid(nz-2, ny-2);

	/* shared memory must fit the gpu */
	while((sizeof(double)*5*error_threads_per_block) > gpu_device_properties.sharedMemPerBlock){
		error_threads_per_block = error_threads_per_block / 2;
	}

	error_gpu_kernel<<<
		error_blocks_per_grid,
		(min(nx-2,error_threads_per_block)),
		sizeof(double)*5*(min(nx-2,error_threads_per_block))>>>(
				u_device, 
				norm_buffer_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_ERROR);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_NORM);
#endif
	/* #KERNEL NORM */
	int norm_threads_per_block=THREADS_PER_BLOCK_ON_NORM;
	dim3 norm_blocks_per_grid(1);

	/* shared memory must fit the gpu */
	while((sizeof(double)*5*norm_threads_per_block) > gpu_device_properties.sharedMemPerBlock){
		norm_threads_per_block = norm_threads_per_block / 2;
	}

	norm_gpu_kernel<<<
		norm_blocks_per_grid, 
		norm_threads_per_block,
		(sizeof(double)*5*norm_threads_per_block)>>>(
				norm_buffer_device, 
				(nz-2)*(ny-2));
#if defined(PROFILING)
	timer_stop(PROFILING_NORM);
#endif

	hipMemcpy(errnm, norm_buffer_device, 5*sizeof(double), hipMemcpyDeviceToHost);
	for(int m=0;m<5;m++){errnm[m]=sqrt(errnm[m]/((double)(nz-2)*(double)(ny-2)*(double)(nx-2)));}
}

/*
 * ---------------------------------------------------------------------
 * compute the solution error
 * ---------------------------------------------------------------------
 */
__global__ static void error_gpu_kernel(const double* u,
		double* errnm,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m;
	double tmp, u000ijk[5];

	double* errnm_loc = (double*)extern_share_data;

	k=blockIdx.x+1;
	j=blockIdx.y+1;
	i=threadIdx.x+1;

	for(m=0;m<5;m++){errnm_loc[m+5*threadIdx.x]=0.0;}
	while(i<nx-1){
		exact_gpu_device(i, j, k, u000ijk, nx, ny, nz);
		for(m=0; m<5; m++){
			tmp=u000ijk[m]-u(m,i,j,k);
			errnm_loc[m+5*threadIdx.x]+=tmp*tmp;
		}
		i+=blockDim.x;
	}
	i=threadIdx.x;
	int loc_max=blockDim.x;
	int dist=(loc_max+1)/2;
	__syncthreads();
	while(loc_max>1){
		if(i<dist && i+dist<loc_max){for(m=0;m<5;m++){errnm_loc[m+5*i]+=errnm_loc[m+5*(i+dist)];}}
		loc_max=dist;
		dist=(dist+1)/2;
		__syncthreads();
	}
	if(i==0){for(m=0;m<5;m++){errnm[m+5*(blockIdx.y+gridDim.y*blockIdx.x)]=errnm_loc[m];}}
}

/*
 * ---------------------------------------------------------------------
 * compute the exact solution at (i,j,k)
 * ---------------------------------------------------------------------
 */
__device__ static void exact_gpu_device(const int i,
		const int j,
		const int k,
		double* u000ijk,
		const int nx,
		const int ny,
		const int nz){
	int m;
	double xi, eta, zeta;
	using namespace constants_device;
	xi=(double)i/(double)(nx-1);
	eta=(double)j/(double)(ny-1);
	zeta=(double)k/(double)(nz-1);
	for(m=0; m<5; m++){
		u000ijk[m]=ce[0][m]+
			(ce[1][m]+
			 (ce[4][m]+
			  (ce[7][m]+
			   ce[10][m]*xi)*xi)*xi)*xi+ 
			(ce[2][m]+
			 (ce[5][m]+
			  (ce[8][m]+
			   ce[11][m]*eta)*eta)*eta)*eta+ 
			(ce[3][m]+
			 (ce[6][m]+
			  (ce[9][m]+
			   ce[12][m]*zeta)*zeta)*zeta)*zeta;
	}
}

__global__ static void jacld_blts_gpu_kernel(const int plane,
		const int klower,
		const int jlower,
		const double* u,
		const double* rho_i,
		const double* qs,
		double* v,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m;
	double tmp1, tmp2, tmp3, tmat[5*5], tv[5];
	double r43, c1345, c34;

	k=klower+blockIdx.x+1;
	j=jlower+threadIdx.x+1;

	i=plane-k-j+3;

	if((j>(ny-2))||(i>(nx-2))||(i<1)){return;}

	r43=4.0/3.0;
	c1345=C1*C3*C4*C5;
	c34=C3*C4;
	using namespace constants_device;
	/*
	 * ---------------------------------------------------------------------
	 * form the first block sub-diagonal
	 * ---------------------------------------------------------------------
	 */
	tmp1=rho_i(i,j,k-1);
	tmp2=tmp1*tmp1;
	tmp3=tmp1*tmp2;
	tmat[0+5*0]= -dt*tz1*dz1;
	tmat[0+5*1]=0.0;
	tmat[0+5*2]=0.0;
	tmat[0+5*3]=-dt*tz2;
	tmat[0+5*4]=0.0;
	tmat[1+5*0]=-dt*tz2*(-(u(1,i,j,k-1)*u(3,i,j,k-1))*tmp2)-dt*tz1*(-c34*tmp2*u(1,i,j,k-1));
	tmat[1+5*1]=-dt*tz2*(u(3,i,j,k-1)*tmp1)-dt*tz1*c34*tmp1-dt*tz1*dz2;
	tmat[1+5*2]=0.0;
	tmat[1+5*3]=-dt*tz2*(u(1,i,j,k-1)*tmp1);
	tmat[1+5*4]=0.0;
	tmat[2+5*0]=-dt*tz2*(-(u(2,i,j,k-1)*u(3,i,j,k-1))*tmp2)-dt*tz1*(-c34*tmp2*u(2,i,j,k-1));
	tmat[2+5*1]=0.0;
	tmat[2+5*2]=-dt*tz2*(u(3,i,j,k-1)*tmp1)-dt*tz1*(c34*tmp1)-dt*tz1*dz3;
	tmat[2+5*3]=-dt*tz2*(u(2,i,j,k-1)*tmp1);
	tmat[2+5*4]=0.0;
	tmat[3+5*0]=-dt*tz2*(-(u(3,i,j,k-1)*tmp1)*(u(3,i,j,k-1)*tmp1)+C2*qs(i,j,k-1)*tmp1)-dt*tz1*(-r43*c34*tmp2*u(3,i,j,k-1));
	tmat[3+5*1]=-dt*tz2*(-C2*(u(1,i,j,k-1)*tmp1));
	tmat[3+5*2]=-dt*tz2*(-C2*(u(2,i,j,k-1)*tmp1));
	tmat[3+5*3]=-dt*tz2*(2.0-C2)*(u(3,i,j,k-1)*tmp1)-dt*tz1*(r43*c34*tmp1)-dt*tz1*dz4;
	tmat[3+5*4]=-dt*tz2*C2;
	tmat[4+5*0]=-dt*tz2*((C2*2.0*qs(i,j,k-1)-C1*u(4,i,j,k-1))*u(3,i,j,k-1)*tmp2)-dt*tz1*(-(c34-c1345)*tmp3*(u(1,i,j,k-1)*u(1,i,j,k-1))-(c34-c1345)*tmp3*(u(2,i,j,k-1)*u(2,i,j,k-1))-(r43*c34-c1345)*tmp3*(u(3,i,j,k-1)*u(3,i,j,k-1))-c1345*tmp2*u(4,i,j,k-1));
	tmat[4+5*1]=-dt*tz2*(-C2*(u(1,i,j,k-1)*u(3,i,j,k-1))*tmp2)-dt*tz1*(c34-c1345)*tmp2*u(1,i,j,k-1);
	tmat[4+5*2]=-dt*tz2*(-C2*(u(2,i,j,k-1)*u(3,i,j,k-1))*tmp2)-dt*tz1*(c34-c1345)*tmp2*u(2,i,j,k-1);
	tmat[4+5*3]=-dt*tz2*(C1*(u(4,i,j,k-1)*tmp1)-C2*(qs(i,j,k-1)*tmp1+u(3,i,j,k-1)*u(3,i,j,k-1)*tmp2))-dt*tz1*(r43*c34-c1345)*tmp2*u(3,i,j,k-1);
	tmat[4+5*4]=-dt*tz2*(C1*(u(3,i,j,k-1)*tmp1))-dt*tz1*c1345*tmp1-dt*tz1*dz5;
	for(m=0;m<5;m++){tv[m]=v(m,i,j,k)-omega*(tmat[m+5*0]*v(0,i,j,k-1)+tmat[m+5*1]*v(1,i,j,k-1)+tmat[m+5*2]*v(2,i,j,k-1)+tmat[m+5*3]*v(3,i,j,k-1)+tmat[m+5*4]*v(4,i,j,k-1));}
	/*
	 * ---------------------------------------------------------------------
	 * form the second block sub-diagonal
	 * ---------------------------------------------------------------------
	 */
	tmp1=rho_i(i,j-1,k);
	tmp2=tmp1*tmp1;
	tmp3=tmp1*tmp2;
	tmat[0+5*0]=-dt*ty1*dy1;
	tmat[0+5*1]=0.0;
	tmat[0+5*2]=-dt*ty2;
	tmat[0+5*3]=0.0;
	tmat[0+5*4]=0.0;
	tmat[1+5*0]=-dt*ty2*(-(u(1,i,j-1,k)*u(2,i,j-1,k))*tmp2)-dt*ty1*(-c34*tmp2*u(1,i,j-1,k));
	tmat[1+5*1]=-dt*ty2*(u(2,i,j-1,k)*tmp1)-dt*ty1*(c34*tmp1)-dt*ty1*dy2;
	tmat[1+5*2]=-dt*ty2*(u(1,i,j-1,k)*tmp1);
	tmat[1+5*3]=0.0;
	tmat[1+5*4]=0.0;
	tmat[2+5*0]=-dt*ty2*(-(u(2,i,j-1,k)*tmp1)*(u(2,i,j-1,k)*tmp1)+C2*(qs(i,j-1,k)*tmp1))-dt*ty1*(-r43*c34*tmp2*u(2,i,j-1,k));
	tmat[2+5*1]=-dt*ty2*(-C2*(u(1,i,j-1,k)*tmp1));
	tmat[2+5*2]=-dt*ty2*((2.0-C2)*(u(2,i,j-1,k)*tmp1))-dt*ty1*(r43*c34*tmp1)-dt*ty1*dy3;
	tmat[2+5*3]=-dt*ty2*(-C2*(u(3,i,j-1,k)*tmp1));
	tmat[2+5*4]=-dt*ty2*C2;
	tmat[3+5*0]=-dt*ty2*(-(u(2,i,j-1,k)*u(3,i,j-1,k))*tmp2)-dt*ty1*(-c34*tmp2*u(3,i,j-1,k));
	tmat[3+5*1]=0.0;
	tmat[3+5*2]=-dt*ty2*(u(3,i,j-1,k)*tmp1);
	tmat[3+5*3]=-dt*ty2*(u(2,i,j-1,k)*tmp1)-dt*ty1*(c34*tmp1)-dt*ty1*dy4;
	tmat[3+5*4]=0.0;
	tmat[4+5*0]=-dt*ty2*((C2*2.0*qs(i,j-1,k)-C1*u(4,i,j-1,k))*(u(2,i,j-1,k)*tmp2))-dt*ty1*(-(c34-c1345)*tmp3*(u(1,i,j-1,k)*u(1,i,j-1,k))-(r43*c34-c1345)*tmp3*(u(2,i,j-1,k)*u(2,i,j-1,k))-(c34-c1345)*tmp3*(u(3,i,j-1,k)*u(3,i,j-1,k))-c1345*tmp2*u(4,i,j-1,k));
	tmat[4+5*1]=-dt*ty2*(-C2*(u(1,i,j-1,k)*u(2,i,j-1,k))*tmp2)-dt*ty1*(c34-c1345)*tmp2*u(1,i,j-1,k);
	tmat[4+5*2]=-dt*ty2*(C1*(u(4,i,j-1,k)*tmp1)-C2*(qs(i,j-1,k)*tmp1+u(2,i,j-1,k)*u(2,i,j-1,k)*tmp2))-dt*ty1*(r43*c34-c1345)*tmp2*u(2,i,j-1,k);
	tmat[4+5*3]=-dt*ty2*(-C2*(u(2,i,j-1,k)*u(3,i,j-1,k))*tmp2) - dt*ty1*(c34-c1345)*tmp2*u(3,i,j-1,k);
	tmat[4+5*4]=-dt*ty2*(C1*(u(2,i,j-1,k)*tmp1))-dt*ty1*c1345*tmp1-dt*ty1*dy5;
	for(m=0;m<5;m++){tv[m]=tv[m]-omega*(tmat[m+5*0]*v(0,i,j-1,k)+tmat[m+5*1]*v(1,i,j-1,k)+tmat[m+5*2]*v(2,i,j-1,k)+tmat[m+5*3]*v(3,i,j-1,k)+tmat[m+5*4]*v(4,i,j-1,k));}
	/*
	 * ---------------------------------------------------------------------
	 * form the third block sub-diagonal
	 * ---------------------------------------------------------------------
	 */
	tmp1=rho_i(i-1,j,k);
	tmp2=tmp1*tmp1;
	tmp3=tmp1*tmp2;
	tmat[0+5*0]=-dt*tx1*dx1;
	tmat[0+5*1]=-dt*tx2;
	tmat[0+5*2]=0.0;
	tmat[0+5*3]=0.0;
	tmat[0+5*4]=0.0;
	tmat[1+5*0]=-dt*tx2*(-(u(1,i-1,j,k)*tmp1)*(u(1,i-1,j,k)*tmp1)+C2*qs(i-1,j,k)*tmp1)-dt*tx1*(-r43*c34*tmp2*u(1,i-1,j,k));
	tmat[1+5*1]=-dt*tx2*((2.0-C2)*(u(1,i-1,j,k)*tmp1))-dt*tx1*(r43*c34*tmp1)-dt*tx1*dx2;
	tmat[1+5*2]=-dt*tx2*(-C2*(u(2,i-1,j,k)*tmp1));
	tmat[1+5*3]=-dt*tx2*(-C2*(u(3,i-1,j,k)*tmp1));
	tmat[1+5*4]=-dt*tx2*C2;
	tmat[2+5*0]=-dt*tx2*(-(u(1,i-1,j,k)*u(2,i-1,j,k))*tmp2)-dt*tx1*(-c34*tmp2*u(2,i-1,j,k));
	tmat[2+5*1]=-dt*tx2*(u(2,i-1,j,k)*tmp1);
	tmat[2+5*2]=-dt*tx2*(u(1,i-1,j,k)*tmp1)-dt*tx1*(c34*tmp1)-dt*tx1*dx3;
	tmat[2+5*3]=0.0;
	tmat[2+5*4]=0.0;
	tmat[3+5*0]=-dt*tx2*(-(u(1,i-1,j,k)*u(3,i-1,j,k))*tmp2)-dt*tx1*(-c34*tmp2*u(3,i-1,j,k));
	tmat[3+5*1]=-dt*tx2*(u(3,i-1,j,k)*tmp1);
	tmat[3+5*2]=0.0;
	tmat[3+5*3]=-dt*tx2*(u(1,i-1,j,k)*tmp1)-dt*tx1*(c34*tmp1)-dt*tx1*dx4;
	tmat[3+5*4]=0.0;
	tmat[4+5*0]=-dt*tx2*((C2*2.0*qs(i-1,j,k)-C1*u(4,i-1,j,k))*u(1,i-1,j,k)*tmp2)-dt*tx1*(-(r43*c34-c1345)*tmp3*(u(1,i-1,j,k)*u(1,i-1,j,k))-(c34-c1345)*tmp3*(u(2,i-1,j,k)*u(2,i-1,j,k))-(c34-c1345)*tmp3*(u(3,i-1,j,k)*u(3,i-1,j,k))-c1345*tmp2*u(4,i-1,j,k));
	tmat[4+5*1]=-dt*tx2*(C1*(u(4,i-1,j,k)*tmp1)-C2*(u(1,i-1,j,k)*u(1,i-1,j,k)*tmp2+qs(i-1,j,k)*tmp1))-dt*tx1*(r43*c34-c1345)*tmp2*u(1,i-1,j,k);
	tmat[4+5*2]=-dt*tx2*(-C2*(u(2,i-1,j,k)*u(1,i-1,j,k))*tmp2)-dt*tx1*(c34-c1345)*tmp2*u(2,i-1,j,k);
	tmat[4+5*3]=-dt*tx2*(-C2*(u(3,i-1,j,k)*u(1,i-1,j,k))*tmp2)-dt*tx1*(c34-c1345)*tmp2*u(3,i-1,j,k);
	tmat[4+5*4]=-dt*tx2*(C1*(u(1,i-1,j,k)*tmp1))-dt*tx1*c1345*tmp1-dt*tx1*dx5;
	for(m=0;m<5;m++){tv[m]=tv[m]-omega*(tmat[m+0*5]*v(0,i-1,j,k)+tmat[m+5*1]*v(1,i-1,j,k)+tmat[m+5*2]*v(2,i-1,j,k)+tmat[m+5*3]*v(3,i-1,j,k)+tmat[m+5*4]*v(4,i-1,j,k));}
	/*
	 * ---------------------------------------------------------------------
	 * form the block diagonal
	 * ---------------------------------------------------------------------
	 */
	tmp1=rho_i(i,j,k);
	tmp2=tmp1*tmp1;
	tmp3=tmp1*tmp2;
	tmat[0+5*0]=1.0+dt*2.0*(tx1*dx1+ty1*dy1+tz1*dz1);
	tmat[0+5*1]=0.0;
	tmat[0+5*2]=0.0;
	tmat[0+5*3]=0.0;
	tmat[0+5*4]=0.0;
	tmat[1+5*0]=-dt*2.0*(tx1*r43+ty1+tz1)*c34*tmp2*u(1,i,j,k);
	tmat[1+5*1]=1.0+dt*2.0*c34*tmp1*(tx1*r43+ty1+tz1) + dt*2.0*(tx1*dx2+ty1*dy2+tz1*dz2);
	tmat[1+5*2]=0.0;
	tmat[1+5*3]=0.0;
	tmat[1+5*4]=0.0;
	tmat[2+5*0]=-dt*2.0*(tx1+ty1*r43+tz1)*c34*tmp2*u(2,i,j,k);
	tmat[2+5*1]=0.0;
	tmat[2+5*2]=1.0+dt*2.0*c34*tmp1*(tx1+ty1*r43+tz1)+dt*2.0*(tx1*dx3+ty1*dy3+tz1*dz3);
	tmat[2+5*3]=0.0;
	tmat[2+5*4]=0.0;
	tmat[3+5*0]=-dt*2.0*(tx1+ty1+tz1*r43)*c34*tmp2*u(3,i,j,k);
	tmat[3+5*1]=0.0;
	tmat[3+5*2]=0.0;
	tmat[3+5*3]=1.0+dt*2.0*c34*tmp1*(tx1+ty1+tz1*r43)+dt*2.0*(tx1*dx4+ty1*dy4+tz1*dz4);
	tmat[3+5*4]=0.0;
	tmat[4+5*0]=-dt*2.0*(((tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345))*(u(1,i,j,k)*u(1,i,j,k))+(tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345))*(u(2,i,j,k)*u(2,i,j,k))+(tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345))*(u(3,i,j,k)*u(3,i,j,k)))*tmp3+(tx1+ty1+tz1)*c1345*tmp2*u(4,i,j,k));
	tmat[4+5*1]=dt*2.0*tmp2*u(1,i,j,k)*(tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345));
	tmat[4+5*2]=dt*2.0*tmp2*u(2,i,j,k)*(tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345));
	tmat[4+5*3]=dt*2.0*tmp2*u(3,i,j,k)*(tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345));
	tmat[4+5*4]=1.0+dt*2.0*(tx1+ty1+tz1)*c1345*tmp1+dt*2.0*(tx1*dx5+ty1*dy5+tz1*dz5);
	/*
	 * ---------------------------------------------------------------------
	 * diagonal block inversion
	 * --------------------------------------------------------------------- 
	 * forward elimination
	 * ---------------------------------------------------------------------
	 */
	tmp1=1.0/tmat[0+0*5];
	tmp2=tmp1*tmat[1+0*5];
	tmat[1+1*5]-=tmp2*tmat[0+1*5];
	tmat[1+2*5]-=tmp2*tmat[0+2*5];
	tmat[1+3*5]-=tmp2*tmat[0+3*5];
	tmat[1+4*5]-=tmp2*tmat[0+4*5];
	tv[1]-=tmp2*tv[0];
	tmp2=tmp1*tmat[2+0*5];
	tmat[2+1*5]-=tmp2*tmat[0+1*5];
	tmat[2+2*5]-=tmp2*tmat[0+2*5];
	tmat[2+3*5]-=tmp2*tmat[0+3*5];
	tmat[2+4*5]-=tmp2*tmat[0+4*5];
	tv[2]-=tmp2*tv[0];
	tmp2=tmp1*tmat[3+0*5];
	tmat[3+1*5]-=tmp2*tmat[0+1*5];
	tmat[3+2*5]-=tmp2*tmat[0+2*5];
	tmat[3+3*5]-=tmp2*tmat[0+3*5];
	tmat[3+4*5]-=tmp2*tmat[0+4*5];
	tv[3]-=tmp2*tv[0];
	tmp2=tmp1*tmat[4+0*5];
	tmat[4+1*5]-=tmp2*tmat[0+1*5];
	tmat[4+2*5]-=tmp2*tmat[0+2*5];
	tmat[4+3*5]-=tmp2*tmat[0+3*5];
	tmat[4+4*5]-=tmp2*tmat[0+4*5];
	tv[4]-=tmp2*tv[0];
	tmp1=1.0/tmat[1+1*5];
	tmp2=tmp1*tmat[2+1*5];
	tmat[2+2*5]-=tmp2*tmat[1+2*5];
	tmat[2+3*5]-=tmp2*tmat[1+3*5];
	tmat[2+4*5]-=tmp2*tmat[1+4*5];
	tv[2]-=tmp2*tv[1];
	tmp2=tmp1*tmat[3+1*5];
	tmat[3+2*5]-=tmp2*tmat[1+2*5];
	tmat[3+3*5]-=tmp2*tmat[1+3*5];
	tmat[3+4*5]-=tmp2*tmat[1+4*5];
	tv[3]-=tmp2*tv[1];
	tmp2=tmp1*tmat[4+1*5];
	tmat[4+2*5]-=tmp2*tmat[1+2*5];
	tmat[4+3*5]-=tmp2*tmat[1+3*5];
	tmat[4+4*5]-=tmp2*tmat[1+4*5];
	tv[4]-=tmp2*tv[1];
	tmp1=1.0/tmat[2+2*5];
	tmp2=tmp1*tmat[3+2*5];
	tmat[3+3*5]-=tmp2*tmat[2+3*5];
	tmat[3+4*5]-=tmp2*tmat[2+4*5];
	tv[3]-=tmp2*tv[2];
	tmp2=tmp1*tmat[4+2*5];
	tmat[4+3*5]-=tmp2*tmat[2+3*5];
	tmat[4+4*5]-=tmp2*tmat[2+4*5];
	tv[4]-=tmp2*tv[2];
	tmp1=1.0/tmat[3+3*5];
	tmp2=tmp1*tmat[4+3*5];
	tmat[4+4*5]-=tmp2*tmat[3+4*5];
	tv[4]-=tmp2*tv[3];
	/*
	 * ---------------------------------------------------------------------
	 * back substitution
	 * ---------------------------------------------------------------------
	 */
	v(4,i,j,k)=tv[4]/tmat[4+4*5];
	tv[3]=tv[3]-tmat[3+4*5]*v(4,i,j,k);
	v(3,i,j,k)=tv[3]/tmat[3+3*5];
	tv[2]=tv[2]-tmat[2+3*5]*v(3,i,j,k)-tmat[2+4*5]*v(4,i,j,k);
	v(2,i,j,k)=tv[2]/tmat[2+2*5];
	tv[1]=tv[1]-tmat[1+2*5]*v(2,i,j,k)-tmat[1+3*5]*v(3,i,j,k)-tmat[1+4*5]*v(4,i,j,k);
	v(1,i,j,k)=tv[1]/tmat[1+1*5];
	tv[0]=tv[0]-tmat[0+1*5]*v(1,i,j,k)-tmat[0+2*5]*v(2,i,j,k)-tmat[0+3*5]*v(3,i,j,k)-tmat[0+4*5]*v(4,i,j,k);
	v(0,i,j,k)=tv[0]/tmat[0+0*5];
}

__global__ static void jacu_buts_gpu_kernel(const int plane,
		const int klower,
		const int jlower,
		const double* u,
		const double* rho_i,
		const double* qs,
		double* v,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m;
	double tmp, tmp1, tmp2, tmp3, tmat[5*5], tv[5];
	double r43, c1345, c34;

	k=klower+blockIdx.x+1;
	j=jlower+threadIdx.x+1;

	i=plane-j-k+3;

	if((i<1)||(i>(nx-2))||(j>(ny-2))){return;}

	using namespace constants_device;
	r43=4.0/3.0;
	c1345=C1*C3*C4*C5;
	c34=C3*C4;
	/*
	 * ---------------------------------------------------------------------
	 * form the first block sub-diagonal
	 * ---------------------------------------------------------------------
	 */
	tmp1=rho_i(i+1,j,k);
	tmp2=tmp1*tmp1;
	tmp3=tmp1*tmp2;
	tmat[0+5*0]=-dt*tx1*dx1;
	tmat[0+5*1]=dt*tx2;
	tmat[0+5*2]=0.0;
	tmat[0+5*3]=0.0;
	tmat[0+5*4]=0.0;
	tmat[1+5*0]=dt*tx2*(-(u(1,i+1,j,k)*tmp1)*(u(1,i+1,j,k)*tmp1)+C2*qs(i+1,j,k)*tmp1)-dt*tx1*(-r43*c34*tmp2*u(1,i+1,j,k));
	tmat[1+5*1]=dt*tx2*((2.0-C2)*(u(1,i+1,j,k)*tmp1))-dt*tx1*(r43*c34*tmp1)-dt*tx1*dx2;
	tmat[1+5*2]=dt*tx2*(-C2*(u(2,i+1,j,k)*tmp1));
	tmat[1+5*3]=dt*tx2*(-C2*(u(3,i+1,j,k)*tmp1));
	tmat[1+5*4]=dt*tx2*C2;
	tmat[2+5*0]=dt*tx2*(-(u(1,i+1,j,k)*u(2,i+1,j,k))*tmp2)-dt*tx1*(-c34*tmp2*u(2,i+1,j,k));
	tmat[2+5*1]=dt*tx2*(u(2,i+1,j,k)*tmp1);
	tmat[2+5*2]=dt*tx2*(u(1,i+1,j,k)*tmp1)-dt*tx1*(c34*tmp1)-dt*tx1*dx3;
	tmat[2+5*3]=0.0;
	tmat[2+5*4]=0.0;
	tmat[3+5*0]=dt*tx2*(-(u(1,i+1,j,k)*u(3,i+1,j,k))*tmp2)-dt*tx1*(-c34*tmp2*u(3,i+1,j,k));
	tmat[3+5*1]=dt*tx2*(u(3,i+1,j,k)*tmp1);
	tmat[3+5*2]=0.0;
	tmat[3+5*3]=dt*tx2*(u(1,i+1,j,k)*tmp1)-dt*tx1*(c34*tmp1)-dt*tx1*dx4;
	tmat[3+5*4]=0.0;
	tmat[4+5*0]=dt*tx2*((C2*2.0*qs(i+1,j,k)-C1*u(4,i+1,j,k))*(u(1,i+1,j,k)*tmp2))-dt*tx1*(-(r43*c34-c1345)*tmp3*(u(1,i+1,j,k)*u(1,i+1,j,k))-(c34-c1345)*tmp3*(u(2,i+1,j,k)*u(2,i+1,j,k))-(c34-c1345)*tmp3*(u(3,i+1,j,k)*u(3,i+1,j,k))-c1345*tmp2*u(4,i+1,j,k));
	tmat[4+5*1]=dt*tx2*(C1*(u(4,i+1,j,k)*tmp1)-C2*(u(1,i+1,j,k)*u(1,i+1,j,k)*tmp2+qs(i+1,j,k)*tmp1))-dt*tx1*(r43*c34-c1345)*tmp2*u(1,i+1,j,k);
	tmat[4+5*2]=dt*tx2*(-C2*(u(2,i+1,j,k)*u(1,i+1,j,k))*tmp2)-dt*tx1*(c34-c1345)*tmp2*u(2,i+1,j,k);
	tmat[4+5*3]=dt*tx2*(-C2*(u(3,i+1,j,k)*u(1,i+1,j,k))*tmp2)-dt*tx1*(c34-c1345)*tmp2*u(3,i+1,j,k);
	tmat[4+5*4]=dt*tx2*(C1*(u(1,i+1,j,k)*tmp1))-dt*tx1*c1345*tmp1-dt*tx1*dx5;
	for(m=0;m<5;m++){tv[m]=omega*(tmat[m+5*0]*v(0,i+1,j,k)+tmat[m+5*1]*v(1,i+1,j,k)+tmat[m+5*2]*v(2,i+1,j,k)+tmat[m+5*3]*v(3,i+1,j,k)+tmat[m+5*4]*v(4,i+1,j,k));}
	/*
	 * ---------------------------------------------------------------------
	 * form the second block sub-diagonal
	 * ---------------------------------------------------------------------
	 */
	tmp1=rho_i(i,j+1,k);
	tmp2=tmp1*tmp1;
	tmp3=tmp1*tmp2;
	tmat[0+5*0]=-dt*ty1*dy1;
	tmat[0+5*1]=0.0;
	tmat[0+5*2]=dt*ty2;
	tmat[0+5*3]=0.0;
	tmat[0+5*4]=0.0;
	tmat[1+5*0]=dt*ty2*(-(u(1,i,j+1,k)*u(2,i,j+1,k))*tmp2)-dt*ty1*(-c34*tmp2*u(1,i,j+1,k));
	tmat[1+5*1]=dt*ty2*(u(2,i,j+1,k)*tmp1)-dt*ty1*(c34*tmp1)-dt*ty1*dy2;
	tmat[1+5*2]=dt*ty2*(u(1,i,j+1,k)*tmp1);
	tmat[1+5*3]=0.0;
	tmat[1+5*4]=0.0;
	tmat[2+5*0]=dt*ty2*(-(u(2,i,j+1,k)*tmp1)*(u(2,i,j+1,k)*tmp1)+C2*(qs(i,j+1,k)*tmp1))-dt*ty1*(-r43*c34*tmp2*u(2,i,j+1,k));
	tmat[2+5*1]=dt*ty2*(-C2*(u(1,i,j+1,k)*tmp1));
	tmat[2+5*2]=dt*ty2*((2.0-C2)*(u(2,i,j+1,k)*tmp1))-dt*ty1*(r43*c34*tmp1)-dt*ty1*dy3;
	tmat[2+5*3]=dt*ty2*(-C2*(u(3,i,j+1,k)*tmp1));
	tmat[2+5*4]=dt*ty2*C2;
	tmat[3+5*0]=dt*ty2*(-(u(2,i,j+1,k)*u(3,i,j+1,k))*tmp2)-dt*ty1*(-c34*tmp2*u(3,i,j+1,k));
	tmat[3+5*1]=0.0;
	tmat[3+5*2]=dt*ty2*(u(3,i,j+1,k)*tmp1);
	tmat[3+5*3]=dt*ty2*(u(2,i,j+1,k)*tmp1)-dt*ty1*(c34*tmp1)-dt*ty1*dy4;
	tmat[3+5*4]=0.0;
	tmat[4+5*0]=dt*ty2*((C2*2.0*qs(i,j+1,k)-C1*u(4,i,j+1,k))*(u(2,i,j+1,k)*tmp2))-dt*ty1*(-(c34-c1345)*tmp3*(u(1,i,j+1,k)*u(1,i,j+1,k))-(r43*c34-c1345)*tmp3*(u(2,i,j+1,k)*u(2,i,j+1,k))-(c34-c1345)*tmp3*(u(3,i,j+1,k)*u(3,i,j+1,k))-c1345*tmp2*u(4,i,j+1,k));
	tmat[4+5*1]=dt*ty2*(-C2*(u(1,i,j+1,k)*u(2,i,j+1,k))*tmp2)-dt*ty1*(c34-c1345)*tmp2*u(1,i,j+1,k);
	tmat[4+5*2]=dt*ty2*(C1*(u(4,i,j+1,k)*tmp1)-C2*(qs(i,j+1,k)*tmp1+u(2,i,j+1,k)*u(2,i,j+1,k)*tmp2))-dt*ty1*(r43*c34-c1345)*tmp2*u(2,i,j+1,k);
	tmat[4+5*3]=dt*ty2*(-C2*(u(2,i,j+1,k)*u(3,i,j+1,k))*tmp2)-dt*ty1*(c34-c1345)*tmp2*u(3,i,j+1,k);
	tmat[4+5*4]=dt*ty2*(C1*(u(2,i,j+1,k)*tmp1))-dt*ty1*c1345*tmp1-dt*ty1*dy5;
	for(m=0;m<5;m++){tv[m]=tv[m]+omega*(tmat[m+5*0]*v(0,i,j+1,k)+tmat[m+5*1]*v(1,i,j+1,k)+tmat[m+5*2]*v(2,i,j+1,k)+tmat[m+5*3]*v(3,i,j+1,k)+tmat[m+5*4]*v(4,i,j+1,k));}
	/*
	 * ---------------------------------------------------------------------
	 * form the third block sub-diagonal
	 * ---------------------------------------------------------------------
	 */
	tmp1=rho_i(i,j,k+1);
	tmp2=tmp1*tmp1;
	tmp3=tmp1*tmp2;
	tmat[0+5*0]=-dt*tz1*dz1;
	tmat[0+5*1]=0.0;
	tmat[0+5*2]=0.0;
	tmat[0+5*3]=dt*tz2;
	tmat[0+5*4]=0.0;
	tmat[1+5*0]=dt*tz2*(-(u(1,i,j,k+1)*u(3,i,j,k+1))*tmp2)-dt*tz1*(-c34*tmp2*u(1,i,j,k+1));
	tmat[1+5*1]=dt*tz2*(u(3,i,j,k+1)*tmp1)-dt*tz1*c34*tmp1-dt*tz1*dz2;
	tmat[1+5*2]=0.0;
	tmat[1+5*3]=dt*tz2*(u(1,i,j,k+1)*tmp1);
	tmat[1+5*4]=0.0;
	tmat[2+5*0]=dt*tz2*(-(u(2,i,j,k+1)*u(3,i,j,k+1))*tmp2)-dt*tz1*(-c34*tmp2*u(2,i,j,k+1));
	tmat[2+5*1]=0.0;
	tmat[2+5*2]=dt*tz2*(u(3,i,j,k+1)*tmp1)-dt*tz1*(c34*tmp1)-dt*tz1*dz3;
	tmat[2+5*3]=dt*tz2*(u(2,i,j,k+1)*tmp1);
	tmat[2+5*4]=0.0;
	tmat[3+5*0]=dt*tz2*(-(u(3,i,j,k+1)*tmp1)*(u(3,i,j,k+1)*tmp1)+C2*(qs(i,j,k+1)*tmp1))-dt*tz1*(-r43*c34*tmp2*u(3,i,j,k+1));
	tmat[3+5*1]=dt*tz2*(-C2*(u(1,i,j,k+1)*tmp1));
	tmat[3+5*2]=dt*tz2*(-C2*(u(2,i,j,k+1)*tmp1));
	tmat[3+5*3]=dt*tz2*(2.0-C2)*(u(3,i,j,k+1)*tmp1)-dt*tz1*(r43*c34*tmp1)-dt*tz1*dz4;
	tmat[3+5*4]=dt*tz2*C2;
	tmat[4+5*0]=dt*tz2*((C2*2.0*qs(i,j,k+1)-C1*u(4,i,j,k+1))*(u(3,i,j,k+1)*tmp2))-dt*tz1*(-(c34-c1345)*tmp3*(u(1,i,j,k+1)*u(1,i,j,k+1))-(c34-c1345)*tmp3*(u(2,i,j,k+1)*u(2,i,j,k+1))-(r43*c34-c1345)*tmp3*(u(3,i,j,k+1)*u(3,i,j,k+1))-c1345*tmp2*u(4,i,j,k+1));
	tmat[4+5*1]=dt*tz2*(-C2*(u(1,i,j,k+1)*u(3,i,j,k+1))*tmp2)-dt*tz1*(c34-c1345)*tmp2*u(1,i,j,k+1);
	tmat[4+5*2]=dt*tz2*(-C2*(u(2,i,j,k+1)*u(3,i,j,k+1))*tmp2)-dt*tz1*(c34-c1345)*tmp2*u(2,i,j,k+1);
	tmat[4+5*3]=dt*tz2*(C1*(u(4,i,j,k+1)*tmp1)-C2*(qs(i,j,k+1)*tmp1+u(3,i,j,k+1)*u(3,i,j,k+1)*tmp2))-dt*tz1*(r43*c34-c1345)*tmp2*u(3,i,j,k+1);
	tmat[4+5*4]=dt*tz2*(C1*(u(3,i,j,k+1)*tmp1))-dt*tz1*c1345*tmp1-dt*tz1*dz5;
	for(m=0;m<5;m++){tv[m]=tv[m]+omega*(tmat[m+5*0]*v(0,i,j,k+1)+tmat[m+5*1]*v(1,i,j,k+1)+tmat[m+5*2]*v(2,i,j,k+1)+tmat[m+5*3]*v(3,i,j,k+1)+tmat[m+5*4]*v(4,i,j,k+1));}
	/*
	 * ---------------------------------------------------------------------
	 * form the block daigonal
	 * ---------------------------------------------------------------------
	 */
	tmp1=rho_i(i,j,k);
	tmp2=tmp1*tmp1;
	tmp3=tmp1*tmp2;
	tmat[0+5*0]=1.0+dt*2.0*(tx1*dx1+ty1*dy1+tz1*dz1);
	tmat[0+5*1]=0.0;
	tmat[0+5*2]=0.0;
	tmat[0+5*3]=0.0;
	tmat[0+5*4]=0.0;
	tmat[1+5*0]=dt*2.0*(-tx1*r43-ty1-tz1)*(c34*tmp2*u(1,i,j,k));
	tmat[1+5*1]=1.0+dt*2.0*c34*tmp1*(tx1*r43+ty1+tz1)+dt*2.0*(tx1*dx2+ty1*dy2+tz1*dz2);
	tmat[1+5*2]=0.0;
	tmat[1+5*3]=0.0;
	tmat[1+5*4]=0.0;
	tmat[2+5*0]=dt*2.0*(-tx1-ty1*r43-tz1)*(c34*tmp2*u(2,i,j,k));
	tmat[2+5*1]=0.0;
	tmat[2+5*2]=1.0+dt*2.0*c34*tmp1*(tx1+ty1*r43+tz1)+dt*2.0*(tx1*dx3+ty1*dy3+tz1*dz3);
	tmat[2+5*3]=0.0;
	tmat[2+5*4]=0.0;
	tmat[3+5*0]=dt*2.0*(-tx1-ty1-tz1*r43)*(c34*tmp2*u(3,i,j,k));
	tmat[3+5*1]=0.0;
	tmat[3+5*2]=0.0;
	tmat[3+5*3]=1.0+dt*2.0*c34*tmp1*(tx1+ty1+tz1*r43)+dt*2.0*(tx1*dx4+ty1*dy4+tz1*dz4);
	tmat[3+5*4]=0.0;
	tmat[4+5*0]=-dt*2.0*(((tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345))*(u(1,i,j,k)*u(1,i,j,k))+(tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345))*(u(2,i,j,k)*u(2,i,j,k))+(tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345))*(u(3,i,j,k)*u(3,i,j,k)))*tmp3+(tx1+ty1+tz1)*c1345*tmp2*u(4,i,j,k));
	tmat[4+5*1]=dt*2.0*(tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345))*tmp2*u(1,i,j,k);
	tmat[4+5*2]=dt*2.0*(tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345))*tmp2*u(2,i,j,k);
	tmat[4+5*3]=dt*2.0*(tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345))*tmp2*u(3,i,j,k);
	tmat[4+5*4]=1.0 + dt*2.0*(tx1+ty1+tz1)*c1345*tmp1+dt*2.0*(tx1*dx5+ty1*dy5+tz1*dz5);
	/*
	 * ---------------------------------------------------------------------
	 * diagonal block inversion
	 * ---------------------------------------------------------------------
	 */
	tmp1=1.0/tmat[0+0*5];
	tmp=tmp1*tmat[1+0*5];
	tmat[1+1*5]-=tmp*tmat[0+1*5];
	tmat[1+2*5]-=tmp*tmat[0+2*5];
	tmat[1+3*5]-=tmp*tmat[0+3*5];
	tmat[1+4*5]-=tmp*tmat[0+4*5];
	tv[1]-=tmp*tv[0];
	tmp=tmp1*tmat[2+0*5];
	tmat[2+1*5]-=tmp*tmat[0+1*5];
	tmat[2+2*5]-=tmp*tmat[0+2*5];
	tmat[2+3*5]-=tmp*tmat[0+3*5];
	tmat[2+4*5]-=tmp*tmat[0+4*5];
	tv[2]-=tmp*tv[0];
	tmp=tmp1*tmat[3+0*5];
	tmat[3+1*5]-=tmp*tmat[0+1*5];
	tmat[3+2*5]-=tmp*tmat[0+2*5];
	tmat[3+3*5]-=tmp*tmat[0+3*5];
	tmat[3+4*5]-=tmp*tmat[0+4*5];
	tv[3]-=tmp*tv[0];
	tmp=tmp1*tmat[4+0*5];
	tmat[4+1*5]-=tmp*tmat[0+1*5];
	tmat[4+2*5]-=tmp*tmat[0+2*5];
	tmat[4+3*5]-=tmp*tmat[0+3*5];
	tmat[4+4*5]-=tmp*tmat[0+4*5];
	tv[4]-=tmp*tv[0];
	tmp1=1.0/tmat[1+1*5];
	tmp=tmp1*tmat[2+1*5];
	tmat[2+2*5]-=tmp*tmat[1+2*5];
	tmat[2+3*5]-=tmp*tmat[1+3*5];
	tmat[2+4*5]-=tmp*tmat[1+4*5];
	tv[2]-=tmp*tv[1];
	tmp=tmp1*tmat[3+1*5];
	tmat[3+2*5]-=tmp*tmat[1+2*5];
	tmat[3+3*5]-=tmp*tmat[1+3*5];
	tmat[3+4*5]-=tmp*tmat[1+4*5];
	tv[3]-=tmp*tv[1];
	tmp=tmp1*tmat[4+1*5];
	tmat[4+2*5]-=tmp*tmat[1+2*5];
	tmat[4+3*5]-=tmp*tmat[1+3*5];
	tmat[4+4*5]-=tmp*tmat[1+4*5];
	tv[4]-=tmp*tv[1];
	tmp1=1.0/tmat[2+2*5];
	tmp=tmp1*tmat[3+2*5];
	tmat[3+3*5]-=tmp*tmat[2+3*5];
	tmat[3+4*5]-=tmp*tmat[2+4*5];
	tv[3]-=tmp*tv[2];
	tmp=tmp1*tmat[4+2*5];
	tmat[4+3*5]-=tmp*tmat[2+3*5];
	tmat[4+4*5]-=tmp*tmat[2+4*5];
	tv[4]-=tmp*tv[2];
	tmp1=1.0/tmat[3+3*5];
	tmp=tmp1*tmat[4+3*5];
	tmat[4+4*5]-=tmp*tmat[3+4*5];
	tv[4]-=tmp*tv[3];
	/*
	 * ---------------------------------------------------------------------
	 * back substitution
	 * ---------------------------------------------------------------------
	 */
	tv[4]=tv[4]/tmat[4+4*5];
	tv[3]=tv[3]-tmat[3+4*5]*tv[4];
	tv[3]=tv[3]/tmat[3+3*5];
	tv[2]=tv[2]-tmat[2+3*5]*tv[3]-tmat[2+4*5]*tv[4];
	tv[2]=tv[2]/tmat[2+2*5];
	tv[1]=tv[1]-tmat[1+2*5]*tv[2]-tmat[1+3*5]*tv[3]-tmat[1+4*5]*tv[4];
	tv[1]=tv[1]/tmat[1+1*5];
	tv[0]=tv[0]-tmat[0+1*5]*tv[1]-tmat[0+2*5]*tv[2]-tmat[0+3*5]*tv[3]-tmat[0+4*5]*tv[4];
	tv[0]=tv[0]/tmat[0+0*5];
	v(0,i,j,k)-=tv[0];
	v(1,i,j,k)-=tv[1];
	v(2,i,j,k)-=tv[2];
	v(3,i,j,k)-=tv[3];
	v(4,i,j,k)-=tv[4];
}

/*
 * ---------------------------------------------------------------------
 * to compute the l2-norm of vector v.
 * ---------------------------------------------------------------------
 * to improve cache performance, second two dimensions padded by 1 
 * for even number sizes only.  Only needed in v.
 * ---------------------------------------------------------------------
 */
static void l2norm_gpu(const double* v,
		double* sum){
#if defined(PROFILING)
	timer_start(PROFILING_L2NORM);
#endif
	/* #KERNEL L2NORM */
	int l2norm_threads_per_block=THREADS_PER_BLOCK_ON_L2NORM;
	dim3 l2norm_blocks_per_grid(nz-2, ny-2);

	/* shared memory must fit the gpu */
	while((sizeof(double)*5*l2norm_threads_per_block) > gpu_device_properties.sharedMemPerBlock){
		l2norm_threads_per_block = l2norm_threads_per_block / 2;
	}

	l2norm_gpu_kernel<<<
		l2norm_blocks_per_grid, 
		min(nx-2,l2norm_threads_per_block),
		sizeof(double)*5*min(nx-2,l2norm_threads_per_block)>>>(
				v, 
				norm_buffer_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_L2NORM);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_NORM);
#endif
	/* #KERNEL NORM */
	int norm_threads_per_block=THREADS_PER_BLOCK_ON_NORM;
	dim3 norm_blocks_per_grid(1);

	/* shared memory must fit the gpu */
	while((sizeof(double)*5*norm_threads_per_block) > gpu_device_properties.sharedMemPerBlock){
		norm_threads_per_block = norm_threads_per_block / 2;
	}

	norm_gpu_kernel<<<
		norm_blocks_per_grid, 
		norm_threads_per_block,
		(sizeof(double)*5*norm_threads_per_block)>>>(
				norm_buffer_device, 
				(nz-2)*(ny-2));
#if defined(PROFILING)
	timer_stop(PROFILING_NORM);
#endif

	hipMemcpy(sum, norm_buffer_device, 5*sizeof(double), hipMemcpyDeviceToHost);
	for(int m=0;m<5;m++){sum[m]=sqrt(sum[m]/((double)(nz-2)*(double)(ny-2)*(double)(nx-2)));}
}

__global__ static void l2norm_gpu_kernel(const double* v,
		double* sum,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m;

	double* sum_loc = (double*)extern_share_data;

	k=blockIdx.x+1;
	j=blockIdx.y+1;
	i=threadIdx.x+1;

	for(m=0;m<5;m++){sum_loc[m+5*threadIdx.x]=0.0;}
	while(i<(nx-1)){
		for(m=0;m<5;m++){sum_loc[m+5*threadIdx.x]+=v(m,i,j,k)*v(m,i,j,k);}
		i+=blockDim.x;
	}
	i=threadIdx.x;
	int loc_max=blockDim.x;
	int dist=(loc_max+1)/2;
	__syncthreads();
	while(loc_max>1){
		if((i<dist)&&(i+dist<loc_max)){
			for(m=0;m<5;m++){sum_loc[m+5*i]+=sum_loc[m+5*(i+dist)];}
		}
		loc_max=dist;
		dist=(dist+1)/2;
		__syncthreads();
	}
	if(i==0){for(m=0;m<5;m++){sum[m+5*(blockIdx.y+gridDim.y*blockIdx.x)]=sum_loc[m];}}
}

__global__ static void norm_gpu_kernel(double* rms,
		const int size){
	int i, m, loc_max, dist;

	double* buffer = (double*)extern_share_data;

	i=threadIdx.x;

	for(m=0;m<5;m++){buffer[m+5*i]=0.0;}
	while(i<size){
		for(m=0;m<5;m++){buffer[m+5*threadIdx.x]+=rms[m+5*i];}
		i+=blockDim.x;
	}
	loc_max=blockDim.x;
	dist=(loc_max+1)/2;
	i=threadIdx.x;
	__syncthreads();
	while(loc_max>1){
		if((i<dist)&&((i+dist)<loc_max)){for(m=0;m<5;m++){buffer[m+5*i]+=buffer[m+5*(i+dist)];}}
		loc_max=dist;
		dist=(dist+1)/2;
		__syncthreads();
	}
	if(threadIdx.x<5){rms[threadIdx.x]=buffer[threadIdx.x];}
}

static void pintgr_gpu(){
#if defined(PROFILING)
	timer_start(PROFILING_PINTGR_1);
#endif
	/* #KERNEL PINTGR 1 */
	int pintgr_1_threads_per_block=THREADS_PER_BLOCK_ON_PINTGR_1;
	dim3 pintgr_1_blocks_per_grid(nx, ny);
	int grid_1_size=nx*ny;

	/* dimensions must fit the gpu */
	while((pintgr_1_threads_per_block*pintgr_1_threads_per_block) > gpu_device_properties.maxThreadsPerBlock){
		pintgr_1_threads_per_block = ceil(sqrt(pintgr_1_threads_per_block));
	}

	/* shared memory must fit the gpu */
	while((sizeof(double)*3*(pintgr_1_threads_per_block*pintgr_1_threads_per_block)) > gpu_device_properties.sharedMemPerBlock){
		pintgr_1_threads_per_block = ceil((double)pintgr_1_threads_per_block/2.0);
	}

	dim3 final_pintgr_1_threads_per_block(pintgr_1_threads_per_block, pintgr_1_threads_per_block);

	pintgr_gpu_kernel_1<<<
		pintgr_1_blocks_per_grid, 
		final_pintgr_1_threads_per_block,
		(sizeof(double)*3*(pintgr_1_threads_per_block*pintgr_1_threads_per_block))>>>(
				u_device, 
				norm_buffer_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_PINTGR_1);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_PINTGR_2);
#endif
	/* #KERNEL PINTGR 2 */
	int pintgr_2_threads_per_block=THREADS_PER_BLOCK_ON_PINTGR_2;
	dim3 pintgr_2_blocks_per_grid(nx, nz);
	int grid_2_size=nx*nz;

	/* dimensions must fit the gpu */
	while((pintgr_2_threads_per_block*pintgr_2_threads_per_block) > gpu_device_properties.maxThreadsPerBlock){
		pintgr_2_threads_per_block = ceil(sqrt(pintgr_2_threads_per_block));
	}

	/* shared memory must fit the gpu */
	while((sizeof(double)*3*(pintgr_2_threads_per_block*pintgr_2_threads_per_block)) > gpu_device_properties.sharedMemPerBlock){
		pintgr_2_threads_per_block = ceil((double)pintgr_2_threads_per_block/2.0);
	}

	dim3 final_pintgr_2_threads_per_block(pintgr_2_threads_per_block, pintgr_2_threads_per_block);

	pintgr_gpu_kernel_2<<<
		pintgr_2_blocks_per_grid, 
		final_pintgr_2_threads_per_block,
		(sizeof(double)*3*(pintgr_2_threads_per_block*pintgr_2_threads_per_block))>>>(
				u_device, 
				norm_buffer_device+grid_1_size, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_PINTGR_2);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_PINTGR_3);
#endif
	/* #KERNEL PINTGR 3 */
	int pintgr_3_threads_per_block=THREADS_PER_BLOCK_ON_PINTGR_3;
	dim3 pintgr_3_blocks_per_grid(ny, nz);
	int grid_3_size=ny*nz;

	/* dimensions must fit the gpu */
	while((pintgr_3_threads_per_block*pintgr_3_threads_per_block) > gpu_device_properties.maxThreadsPerBlock){
		pintgr_3_threads_per_block = ceil(sqrt(pintgr_3_threads_per_block));
	}

	/* shared memory must fit the gpu */
	while((sizeof(double)*3*(pintgr_3_threads_per_block*pintgr_3_threads_per_block)) > gpu_device_properties.sharedMemPerBlock){
		pintgr_3_threads_per_block = ceil((double)pintgr_3_threads_per_block/2.0);
	}

	dim3 final_pintgr_3_threads_per_block(pintgr_3_threads_per_block, pintgr_3_threads_per_block);

	pintgr_gpu_kernel_3<<<
		pintgr_3_blocks_per_grid, 
		final_pintgr_3_threads_per_block,
		(sizeof(double)*3*(pintgr_3_threads_per_block*pintgr_3_threads_per_block))>>>(
				u_device, 
				norm_buffer_device+grid_1_size+grid_2_size, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_PINTGR_3);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_PINTGR_4);
#endif
	/* #KERNEL PINTGR 4 */
	int pintgr_4_threads_per_block=THREADS_PER_BLOCK_ON_PINTGR_4;
	int pintgr_4_blocks_per_grid = 1;

	/* shared memory must fit the gpu */
	while((sizeof(double)*pintgr_4_threads_per_block) > gpu_device_properties.sharedMemPerBlock){
		pintgr_4_threads_per_block = pintgr_4_threads_per_block / 2;
	}

	pintgr_gpu_kernel_4<<<
		pintgr_4_blocks_per_grid, 
		pintgr_4_threads_per_block,
		(sizeof(double)*pintgr_4_threads_per_block)>>>(
				norm_buffer_device,
				grid_1_size+grid_2_size+grid_3_size);
#if defined(PROFILING)
	timer_stop(PROFILING_PINTGR_4);
#endif

	hipMemcpy(&frc, norm_buffer_device, sizeof(double), hipMemcpyDeviceToHost);
}

__global__ static void pintgr_gpu_kernel_1(const double* u,
		double* frc,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k;

	double* phi1 = (double*)extern_share_data;
	double* phi2 = (double*)phi1+(blockDim.x*blockDim.y);
	double* frc1 = (double*)phi2+(blockDim.x*blockDim.y);

	i=blockIdx.x*(blockDim.x-1)+threadIdx.x+1;
	j=blockIdx.y*(blockDim.x-1)+threadIdx.y+1;

	using namespace constants_device;
	/*
	 * ---------------------------------------------------------------------
	 * initialize
	 * ---------------------------------------------------------------------
	 */
	if(j<ny-2 && i<nx-1){
		k=2;
		phi1[threadIdx.x+(threadIdx.y*blockDim.x)]=C2*(u(4,i,j,k)-0.5*(u(1,i,j,k)*u(1,i,j,k)+u(2,i,j,k)*u(2,i,j,k)+u(3,i,j,k)*u(3,i,j,k))/u(0,i,j,k));
		k=nz-2;
		phi2[threadIdx.x+(threadIdx.y*blockDim.x)]=C2*(u(4,i,j,k)-0.5*(u(1,i,j,k)*u(1,i,j,k)+u(2,i,j,k)*u(2,i,j,k)+u(3,i,j,k)*u(3,i,j,k))/u(0,i,j,k));
	}
	__syncthreads();
	frc1[threadIdx.y*blockDim.x+threadIdx.x]=0.0;
	if((j<(ny-3))&&(i<(nx-2))&&(threadIdx.x<(blockDim.x-1))&&(threadIdx.y<(blockDim.x-1))){ 
		frc1[threadIdx.y*blockDim.x+threadIdx.x]=phi1[threadIdx.x+(threadIdx.y*blockDim.x)]+phi1[(threadIdx.x+1)+(threadIdx.y*blockDim.x)]+phi1[threadIdx.x+((threadIdx.y+1)*blockDim.x)]+phi1[(threadIdx.x+1)+((threadIdx.y+1)*blockDim.x)]+phi2[threadIdx.x+(threadIdx.y*blockDim.x)]+phi2[(threadIdx.x+1)+(threadIdx.y*blockDim.x)]+phi2[(threadIdx.x)+((threadIdx.y+1)*blockDim.x)]+phi2[(threadIdx.x+1)+((threadIdx.y+1)*blockDim.x)];}
	int loc_max=blockDim.x*blockDim.y;
	int dist=(loc_max+1)/2;
	i=threadIdx.y*blockDim.x+threadIdx.x;
	__syncthreads();
	while(loc_max>1){
		if((i<dist)&&((i+dist)<loc_max)){frc1[i]+=frc1[i+dist];}
		loc_max=dist;
		dist=(dist+1)/2;
		__syncthreads();
	}
	if(i==0){frc[blockIdx.y*gridDim.x+blockIdx.x]=frc1[0]*dxi*deta;}
}

__global__ static void pintgr_gpu_kernel_2(const double* u,
		double* frc,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, kp, ip;

	double* phi1 = (double*)extern_share_data;
	double* phi2 = (double*)phi1+(blockDim.x*blockDim.y);
	double* frc2 = (double*)phi2+(blockDim.x*blockDim.y);

	i=blockIdx.x*(blockDim.x-1)+1;
	k=blockIdx.y*(blockDim.x-1)+2;
	kp=threadIdx.y;
	ip=threadIdx.x;

	using namespace constants_device;
	/*
	 * ---------------------------------------------------------------------
	 * initialize
	 * ---------------------------------------------------------------------
	 */
	if(((k+kp)<(nz-1))&&((i+ip)<(nx-1))){
		j=1;
		phi1[kp+(ip*blockDim.x)]=C2*(u(4,i+ip,j,k+kp)-0.5*(u(1,i+ip,j,k+kp)*u(1,i+ip,j,k+kp)+u(2,i+ip,j,k+kp)*u(2,i+ip,j,k+kp)+u(3,i+ip,j,k+kp)*u(3,i+ip,j,k+kp))/u(0,i+ip,j,k+kp));
		j=ny-3;
		phi2[kp+(ip*blockDim.x)]=C2*(u(4,i+ip,j,k+kp)-0.5*(u(1,i+ip,j,k+kp)*u(1,i+ip,j,k+kp)+u(2,i+ip,j,k+kp)*u(2,i+ip,j,k+kp)+u(3,i+ip,j,k+kp)*u(3,i+ip,j,k+kp))/u(0,i+ip,j,k+kp));
	}
	__syncthreads();
	frc2[kp*blockDim.x+ip]=0.0;
	if(((k+kp)<(nz-2))&&((i+ip)<(nx-2))&&(kp<(blockDim.x-1))&&(ip<(blockDim.x-1))){frc2[kp*blockDim.x+ip]+=phi1[kp+(ip*blockDim.x)]+phi1[(kp+1)+(ip*blockDim.x)]+phi1[kp+((ip+1)*blockDim.x)]+phi1[(kp+1)+((ip+1)*blockDim.x)]+phi2[kp+(ip*blockDim.x)]+phi2[(kp+1)+(ip*blockDim.x)]+phi2[kp+((ip+1)*blockDim.x)]+phi2[(kp+1)+((ip+1)*blockDim.x)];}
	int loc_max=blockDim.x*blockDim.y;
	int dist=(loc_max+1)/2;
	i=threadIdx.y*blockDim.x+threadIdx.x;
	__syncthreads();
	while(loc_max>1){
		if((i<dist)&&((i+dist)<loc_max)){frc2[i]+=frc2[i+dist];}
		loc_max=dist;
		dist=(dist+1)/2;
		__syncthreads();
	}
	if(i==0){frc[blockIdx.y*gridDim.x+blockIdx.x]=frc2[0]*dxi*dzeta;}
}

__global__ static void pintgr_gpu_kernel_3(const double* u,
		double* frc,
		const int nx,
		const int ny,
		const int nz){
	int j, k, jp, kp;

	double* phi1 = (double*)extern_share_data;
	double* phi2 = (double*)phi1+(blockDim.x*blockDim.y);
	double* frc3 = (double*)phi2+(blockDim.x*blockDim.y);

	j=blockIdx.x*(blockDim.x-1)+1;
	k=blockIdx.y*(blockDim.x-1)+2;
	kp=threadIdx.y;
	jp=threadIdx.x;

	using namespace constants_device;
	/*
	 * ---------------------------------------------------------------------
	 * initialize
	 * ---------------------------------------------------------------------
	 */
	if(((k+kp)<(nz-1))&&((j+jp)<(ny-2))){
		phi1[kp+(jp*blockDim.x)]=C2*(u(4,1,j+jp,k+kp)-0.5*(u(1,1,j+jp,k+kp)*u(1,1,j+jp,k+kp)+u(2,1,j+jp,k+kp)*u(2,1,j+jp,k+kp)+u(3,1,j+jp,k+kp)*u(3,1,j+jp,k+kp))/u(0,1,j+jp,k+kp));
		phi2[kp+(jp*blockDim.x)]=C2*(u(4,nx-2,j+jp,k+kp)-0.5*(u(1,nx-2,j+jp,k+kp)*u(1,nx-2,j+jp,k+kp)+u(2,nx-2,j+jp,k+kp)*u(2,nx-2,j+jp,k+kp)+u(3,nx-2,j+jp,k+kp)*u(3,nx-2,j+jp,k+kp))/u(0,nx-2,j+jp,k+kp));
	}
	__syncthreads();
	frc3[kp*blockDim.x+jp]=0.0;
	if(((k+kp)<(nz-2))&&((j+jp)<(ny-3))&&(kp<(blockDim.x-1))&&(jp<blockDim.x-1)){frc3[kp*blockDim.x+jp]=phi1[kp+(jp*blockDim.x)]+phi1[(kp+1)+(jp*blockDim.x)]+phi1[kp+((jp+1)*blockDim.x)]+phi1[(kp+1)+((jp+1)*blockDim.x)]+phi2[kp+(jp*blockDim.x)]+phi2[(kp+1)+(jp*blockDim.x)]+phi2[(kp)+((jp+1)*blockDim.x)]+phi2[(kp+1)+((jp+1)*blockDim.x)];}
	int loc_max=blockDim.x*blockDim.y;
	int dist=(loc_max+1)/2;
	j=threadIdx.y*blockDim.x+threadIdx.x;
	__syncthreads();
	while(loc_max>1){
		if((j<dist)&&((j+dist)<loc_max)){frc3[j]+=frc3[j+dist];}
		loc_max=dist;
		dist=(dist+1)/2;
		__syncthreads();
	}
	if(j==0){frc[blockIdx.y*gridDim.x+blockIdx.x]=frc3[0]*deta*dzeta;}
}

__global__ static void pintgr_gpu_kernel_4(double* frc,
		const int num){
	int i, loc_max, dist;

	double* buffer = (double*)extern_share_data;

	i=threadIdx.x;

	buffer[i]=0.0;
	while(i<num){
		buffer[threadIdx.x]+=frc[i];
		i+=blockDim.x;
	}
	loc_max=blockDim.x;
	dist=(loc_max+1)/2;
	i=threadIdx.x;
	__syncthreads();
	while(loc_max>1){
		if((i<dist)&&((i+dist)<loc_max)){buffer[i]+=buffer[i+dist];}
		loc_max=dist;
		dist=(dist+1)/2;
		__syncthreads();
	}
	if(i==0){frc[0]=.25*buffer[0];}
}

void read_input(){
	/*
	 * ---------------------------------------------------------------------
	 * if input file does not exist, it uses defaults
	 * ipr = 1 for detailed progress output
	 * inorm = how often the norm is printed (once every inorm iterations)
	 * itmax = number of pseudo time steps
	 * dt = time step
	 * omega 1 over-relaxation factor for SSOR
	 * tolrsd = steady state residual tolerance levels
	 * nx, ny, nz = number of grid points in x, y, z directions
	 * ---------------------------------------------------------------------
	 */	
	FILE* fp; int avoid_warning;
	if((fp=fopen("inputlu.data","r"))!=NULL){
		printf("Reading from input file inputlu.data\n");
		while(fgetc(fp)!='\n');
		while(fgetc(fp)!='\n');
		avoid_warning=fscanf(fp,"%d%d",&ipr,&inorm); 
		while(fgetc(fp)!='\n');
		while(fgetc(fp)!='\n');
		while(fgetc(fp)!='\n');
		avoid_warning=fscanf(fp,"%d",&itmax);
		while(fgetc(fp)!='\n');
		while(fgetc(fp)!='\n');
		while(fgetc(fp)!='\n');
		avoid_warning=fscanf(fp,"%lf",&dt_host);
		while(fgetc(fp)!='\n');
		while(fgetc(fp)!='\n');
		while(fgetc(fp)!='\n');
		avoid_warning=fscanf(fp,"%lf",&omega_host);
		while(fgetc(fp)!='\n');
		while(fgetc(fp)!='\n');
		while(fgetc(fp)!='\n');
		avoid_warning=fscanf(fp,"%lf%lf%lf%lf%lf",&tolrsd[0],
				&tolrsd[1],
				&tolrsd[2],
				&tolrsd[3],
				&tolrsd[4]);
		while(fgetc(fp)!='\n');
		while(fgetc(fp)!='\n');
		avoid_warning=fscanf(fp,"%d%d%d",&nx,&ny,&nz);
		avoid_warning++;
		fclose(fp);
	}else{
		ipr=IPR_DEFAULT;
		inorm=INORM_DEFAULT;
		itmax=ITMAX_DEFAULT;
		dt_host=DT_DEFAULT;
		omega_host=OMEGA_DEFAULT;
		tolrsd[0]=TOLRSD1_DEF;
		tolrsd[1]=TOLRSD2_DEF;
		tolrsd[2]=TOLRSD3_DEF;
		tolrsd[3]=TOLRSD4_DEF;
		tolrsd[4]=TOLRSD5_DEF;
		nx=ISIZ1;
		ny=ISIZ2;
		nz=ISIZ3;
	}
	/*
	 * ---------------------------------------------------------------------
	 * check problem size
	 * ---------------------------------------------------------------------
	 */
	if((nx<4)||(ny<4)||(nz<4)){
		printf("     PROBLEM SIZE IS TOO SMALL - \n"
				"     SET EACH OF NX, NY AND NZ AT LEAST EQUAL TO 5\n");
		exit(EXIT_FAILURE);
	}
	if((nx>ISIZ1)||(ny>ISIZ2)||(nz>ISIZ3)){
		printf("     PROBLEM SIZE IS TOO LARGE - \n"
				"     NX, NY AND NZ SHOULD BE EQUAL TO \n"
				"     ISIZ1, ISIZ2 AND ISIZ3 RESPECTIVELY\n");
		exit(EXIT_FAILURE);
	}
	printf("\n\n NAS Parallel Benchmarks 4.1 HIP C++ version - LU Benchmark\n\n");
	printf(" Size: %4dx%4dx%4d\n",nx,ny,nz);
	printf(" Iterations: %4d\n",itmax);
	printf("\n");
}

static void release_gpu(){
	hipFree(u_device);
	hipFree(rsd_device);
	hipFree(frct_device);
	hipFree(rho_i_device);
	hipFree(qs_device);
	hipFree(norm_buffer_device);
}

static void rhs_gpu(){
#if defined(PROFILING)
	timer_start(PROFILING_RHS_1);
#endif
	/* #KERNEL RHS 1 */
	int rhs_1_workload = nx * ny * nz;
	int rhs_1_threads_per_block = THREADS_PER_BLOCK_ON_RHS_1;
	int rhs_1_blocks_per_grid = (ceil((double)rhs_1_workload/(double)rhs_1_threads_per_block));

	rhs_gpu_kernel_1<<<
		rhs_1_blocks_per_grid, 
		rhs_1_threads_per_block>>>(
				u_device, 
				rsd_device, 
				frct_device, 
				qs_device, 
				rho_i_device,
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_RHS_1);
#endif

	/*
	 * ---------------------------------------------------------------------
	 * xi-direction flux differences
	 * ---------------------------------------------------------------------
	 */
#if defined(PROFILING)
	timer_start(PROFILING_RHS_2);
#endif
	/* #KERNEL RHS 2 */
	int rhs_2_threads_per_block;
	dim3 rhs_2_blocks_per_grid(nz-2, ny-2);
	if(THREADS_PER_BLOCK_ON_RHS_2 != gpu_device_properties.warpSize){
		rhs_2_threads_per_block = gpu_device_properties.warpSize;
	}
	else{
		rhs_2_threads_per_block = THREADS_PER_BLOCK_ON_RHS_2;
	}

	rhs_gpu_kernel_2<<<
		rhs_2_blocks_per_grid, 
		(min(nx,rhs_2_threads_per_block)),
		sizeof(double)*((3*(min(nx,rhs_2_threads_per_block))*5)+(5*(min(nx,rhs_2_threads_per_block))))>>>(
				u_device, 
				rsd_device, 
				qs_device, 
				rho_i_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_RHS_2);
#endif

	/*
	 * ---------------------------------------------------------------------
	 * eta-direction flux differences
	 * ---------------------------------------------------------------------
	 */
#if defined(PROFILING)
	timer_start(PROFILING_RHS_3);
#endif
	/* #KERNEL RHS 3 */
	int rhs_3_threads_per_block;
	dim3 rhs_3_blocks_per_grid(nz-2, nx-2);
	if(THREADS_PER_BLOCK_ON_RHS_3 != gpu_device_properties.warpSize){
		rhs_3_threads_per_block = gpu_device_properties.warpSize;
	}
	else{
		rhs_3_threads_per_block = THREADS_PER_BLOCK_ON_RHS_3;
	}

	rhs_gpu_kernel_3<<<
		rhs_3_blocks_per_grid, 
		(min(ny,rhs_3_threads_per_block)),
		sizeof(double)*((3*(min(ny,rhs_3_threads_per_block))*5)+(5*(min(ny,rhs_3_threads_per_block))))>>>(
				u_device, 
				rsd_device, 
				qs_device, 
				rho_i_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_RHS_3);
#endif

	/*
	 * ---------------------------------------------------------------------
	 * zeta-direction flux differences
	 * ---------------------------------------------------------------------
	 */
#if defined(PROFILING)
	timer_start(PROFILING_RHS_4);
#endif
	/* #KERNEL RHS 4 */
	int rhs_4_threads_per_block;
	dim3 rhs_4_blocks_per_grid(ny-2, nx-2);
	if(THREADS_PER_BLOCK_ON_RHS_4 != gpu_device_properties.warpSize){
		rhs_4_threads_per_block = gpu_device_properties.warpSize;
	}
	else{
		rhs_4_threads_per_block = THREADS_PER_BLOCK_ON_RHS_4;
	}

	rhs_gpu_kernel_4<<<
		rhs_4_blocks_per_grid, 
		(min(nz,rhs_4_threads_per_block)),
		sizeof(double)*((3*(min(nz,rhs_4_threads_per_block))*5)+(5*(min(nz,rhs_4_threads_per_block))))>>>(
				u_device, 
				rsd_device, 
				qs_device, 
				rho_i_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_RHS_4);
#endif
}

__global__ static void rhs_gpu_kernel_1(const double* u,
		double* rsd,
		const double* frct,
		double* qs,
		double* rho_i,
		const int nx,
		const int ny,
		const int nz){
	int i_j_k, i, j, k, m;
	double tmp;

	i_j_k = blockIdx.x * blockDim.x + threadIdx.x;

	i = i_j_k % nx;
	j = (i_j_k / nx) % ny;
	k = i_j_k / (nx * ny);

	if(i_j_k >= (nx*ny*nz)){
		return;
	}

	for(m=0;m<5;m++){rsd(m,i,j,k)=-frct(m,i,j,k);}
	rho_i(i,j,k)=tmp=1.0/u(0,i,j,k);
	qs(i,j,k)=0.5*(u(1,i,j,k)*u(1,i,j,k)+u(2,i,j,k)*u(2,i,j,k)+u(3,i,j,k)*u(3,i,j,k))*tmp;
}

__global__ static void rhs_gpu_kernel_2(const double* u,
		double* rsd,
		const double* qs,
		const double* rho_i,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m, nthreads;
	double q, u21;

	double* flux = (double*)extern_share_data;
	double* utmp = (double*)flux+(blockDim.x*5);
	double* rtmp = (double*)utmp+(blockDim.x*5);
	double* rhotmp = (double*)rtmp+(blockDim.x*5);
	double* u21i = (double*)rhotmp+(blockDim.x);
	double* u31i = (double*)u21i+(blockDim.x);
	double* u41i = (double*)u31i+(blockDim.x);
	double* u51i = (double*)u41i+(blockDim.x);

	k=blockIdx.x+1;
	j=blockIdx.y+1;
	i=threadIdx.x;

	using namespace constants_device;
	while(i<nx){
		nthreads=nx-(i-threadIdx.x);
		if(nthreads>blockDim.x){nthreads=blockDim.x;}
		m=threadIdx.x;
		utmp[m]=u(m%5, (i-threadIdx.x)+m/5, j, k);
		rtmp[m]=rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m+=nthreads;
		utmp[m]=u(m%5, (i-threadIdx.x)+m/5, j, k);
		rtmp[m]=rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m+=nthreads;
		utmp[m]=u(m%5, (i-threadIdx.x)+m/5, j, k);
		rtmp[m]=rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m+=nthreads;
		utmp[m]=u(m%5, (i-threadIdx.x)+m/5, j, k);
		rtmp[m]=rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m+=nthreads;
		utmp[m]=u(m%5, (i-threadIdx.x)+m/5, j, k);
		rtmp[m]=rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		rhotmp[threadIdx.x]=rho_i(i,j,k);
		__syncthreads();
		/*
		 * ---------------------------------------------------------------------
		 * xi-direction flux differences
		 * ---------------------------------------------------------------------
		 */
		flux[threadIdx.x+(0*blockDim.x)]=utmp[threadIdx.x*5+1];
		u21=utmp[threadIdx.x*5+1]*rhotmp[threadIdx.x];
		q=qs(i,j,k);
		flux[threadIdx.x+(1*blockDim.x)]=utmp[threadIdx.x*5+1]*u21+C2*(utmp[threadIdx.x*5+4]-q);
		flux[threadIdx.x+(2*blockDim.x)]=utmp[threadIdx.x*5+2]*u21;
		flux[threadIdx.x+(3*blockDim.x)]=utmp[threadIdx.x*5+3]*u21;
		flux[threadIdx.x+(4*blockDim.x)]=(C1*utmp[threadIdx.x*5+4]-C2*q)*u21;
		__syncthreads();
		if((threadIdx.x>=1)&&(threadIdx.x<(blockDim.x-1))&&(i<(nx-1))){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]=rtmp[threadIdx.x*5+m]-tx2*(flux[(threadIdx.x+1)+(m*blockDim.x)]-flux[(threadIdx.x-1)+(m*blockDim.x)]);}}
		u21i[threadIdx.x]=rhotmp[threadIdx.x]*utmp[threadIdx.x*5+1];
		u31i[threadIdx.x]=rhotmp[threadIdx.x]*utmp[threadIdx.x*5+2];
		u41i[threadIdx.x]=rhotmp[threadIdx.x]*utmp[threadIdx.x*5+3];
		u51i[threadIdx.x]=rhotmp[threadIdx.x]*utmp[threadIdx.x*5+4];
		__syncthreads();
		if(threadIdx.x>=1){
			flux[threadIdx.x+(1*blockDim.x)]=(4.0/3.0)*tx3*(u21i[threadIdx.x]-u21i[threadIdx.x-1]);
			flux[threadIdx.x+(2*blockDim.x)]=tx3*(u31i[threadIdx.x]-u31i[threadIdx.x-1]);
			flux[threadIdx.x+(3*blockDim.x)]=tx3*(u41i[threadIdx.x]-u41i[threadIdx.x-1]);
			flux[threadIdx.x+(4*blockDim.x)]=0.5*(1.0-C1*C5)*tx3*((u21i[threadIdx.x]*u21i[threadIdx.x]+u31i[threadIdx.x]*u31i[threadIdx.x]+u41i[threadIdx.x]*u41i[threadIdx.x])-(u21i[threadIdx.x-1]*u21i[threadIdx.x-1]+u31i[threadIdx.x-1]*u31i[threadIdx.x-1]+u41i[threadIdx.x-1]*u41i[threadIdx.x-1]))+(1.0/6.0)*tx3*(u21i[threadIdx.x]*u21i[threadIdx.x]-u21i[threadIdx.x-1]*u21i[threadIdx.x-1]) + C1*C5*tx3*(u51i[threadIdx.x]-u51i[threadIdx.x-1]);
		}
		__syncthreads();
		if((threadIdx.x>=1)&&(threadIdx.x<(blockDim.x-1))&&(i<(nx-1))){
			rtmp[threadIdx.x*5+0]+=dx1*tx1*(utmp[threadIdx.x*5-5]-2.0*utmp[threadIdx.x*5+0]+utmp[threadIdx.x*5+5]);
			rtmp[threadIdx.x*5+1]+=tx3*C3*C4*(flux[(threadIdx.x+1)+(1*blockDim.x)]-flux[threadIdx.x+(1*blockDim.x)])+dx2*tx1*(utmp[threadIdx.x*5-4]-2.0*utmp[threadIdx.x*5+1]+utmp[threadIdx.x*5+6]);
			rtmp[threadIdx.x*5+2]+=tx3*C3*C4*(flux[(threadIdx.x+1)+(2*blockDim.x)]-flux[threadIdx.x+(2*blockDim.x)])+dx3*tx1*(utmp[threadIdx.x*5-3]-2.0*utmp[threadIdx.x*5+2]+utmp[threadIdx.x*5+7]);
			rtmp[threadIdx.x*5+3]+=tx3*C3*C4*(flux[(threadIdx.x+1)+(3*blockDim.x)]-flux[threadIdx.x+(3*blockDim.x)])+dx4*tx1*(utmp[threadIdx.x*5-2]-2.0*utmp[threadIdx.x*5+3]+utmp[threadIdx.x*5+8]);
			rtmp[threadIdx.x*5+4]+=tx3*C3*C4*(flux[(threadIdx.x+1)+(4*blockDim.x)]-flux[threadIdx.x+(4*blockDim.x)])+dx5*tx1*(utmp[threadIdx.x*5-1]-2.0*utmp[threadIdx.x*5+4]+utmp[threadIdx.x*5+9]);
			/*
			 * ---------------------------------------------------------------------
			 * fourth-order dissipation
			 * ---------------------------------------------------------------------
			 */
			if(i==1){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]-=dssp*(5.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,3,j,k));}}
			if(i==2){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]-=dssp*(-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,4,j,k));}}
			if((i>=3)&&(i<(nx-3))){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]-=dssp*(u(m,i-2,j,k)-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,i+2,j,k));}}
			if(i==(nx-3)){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]-=dssp*(u(m,nx-5,j,k)-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]);}}
			if(i==(nx-2)){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]-=dssp*(u(m,nx-4,j,k)-4.0*utmp[threadIdx.x*5+m-5]+5.0*utmp[threadIdx.x*5+m]);}}
		}
		m=threadIdx.x;
		rsd(m%5, (i-threadIdx.x)+m/5, j, k)=rtmp[m];
		m+=nthreads;
		rsd(m%5, (i-threadIdx.x)+m/5, j, k)=rtmp[m];
		m+=nthreads;
		rsd(m%5, (i-threadIdx.x)+m/5, j, k)=rtmp[m];
		m+=nthreads;
		rsd(m%5, (i-threadIdx.x)+m/5, j, k)=rtmp[m];
		m+=nthreads;
		rsd(m%5, (i-threadIdx.x)+m/5, j, k)=rtmp[m];
		i+=blockDim.x-2;
	}
}

__global__ static void rhs_gpu_kernel_3(const double* u,
		double* rsd,
		const double* qs,
		const double* rho_i,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m, nthreads;
	double q, u31;

	double* flux = (double*)extern_share_data;
	double* utmp = (double*)flux+(blockDim.x*5);
	double* rtmp = (double*)utmp+(blockDim.x*5);
	double* rhotmp = (double*)rtmp+(blockDim.x*5);
	double* u21j = (double*)rhotmp+(blockDim.x);
	double* u31j = (double*)u21j+(blockDim.x);
	double* u41j = (double*)u31j+(blockDim.x);
	double* u51j = (double*)u41j+(blockDim.x);

	k=blockIdx.x+1;
	i=blockIdx.y+1;
	j=threadIdx.x;

	using namespace constants_device;
	while(j<ny){
		nthreads=ny-(j-threadIdx.x);
		if(nthreads>blockDim.x){nthreads=blockDim.x;}
		m=threadIdx.x;
		utmp[m]=u(m%5, i, (j-threadIdx.x)+m/5, k);
		rtmp[m]=rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m+=nthreads;
		utmp[m]=u(m%5, i, (j-threadIdx.x)+m/5, k);
		rtmp[m]=rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m+=nthreads;
		utmp[m]=u(m%5, i, (j-threadIdx.x)+m/5, k);
		rtmp[m]=rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m+=nthreads;
		utmp[m]=u(m%5, i, (j-threadIdx.x)+m/5, k);
		rtmp[m]=rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m+=nthreads;
		utmp[m]=u(m%5, i, (j-threadIdx.x)+m/5, k);
		rtmp[m]=rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		rhotmp[threadIdx.x]=rho_i(i,j,k);
		__syncthreads();
		/*
		 * ---------------------------------------------------------------------
		 * eta-direction flux differences
		 * ---------------------------------------------------------------------
		 */
		flux[threadIdx.x+(0*blockDim.x)]=utmp[threadIdx.x*5+2];
		u31=utmp[threadIdx.x*5+2]*rhotmp[threadIdx.x];
		q=qs(i,j,k);
		flux[threadIdx.x+(1*blockDim.x)]=utmp[threadIdx.x*5+1]*u31;
		flux[threadIdx.x+(2*blockDim.x)]=utmp[threadIdx.x*5+2]*u31+C2*(utmp[threadIdx.x*5+4]-q);
		flux[threadIdx.x+(3*blockDim.x)]=utmp[threadIdx.x*5+3]*u31;
		flux[threadIdx.x+(4*blockDim.x)]=(C1*utmp[threadIdx.x*5+4]-C2*q)*u31;
		__syncthreads();
		if((threadIdx.x>=1)&&(threadIdx.x<(blockDim.x-1))&&(j<(ny-1))){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]=rtmp[threadIdx.x*5+m]-ty2*(flux[(threadIdx.x+1)+(m*blockDim.x)]-flux[(threadIdx.x-1)+(m*blockDim.x)]);}}
		u21j[threadIdx.x]=rhotmp[threadIdx.x]*utmp[threadIdx.x*5+1];
		u31j[threadIdx.x]=rhotmp[threadIdx.x]*utmp[threadIdx.x*5+2];
		u41j[threadIdx.x]=rhotmp[threadIdx.x]*utmp[threadIdx.x*5+3];
		u51j[threadIdx.x]=rhotmp[threadIdx.x]*utmp[threadIdx.x*5+4];
		__syncthreads();
		if(threadIdx.x>=1){
			flux[threadIdx.x+(1*blockDim.x)]=ty3*(u21j[threadIdx.x]-u21j[threadIdx.x-1]);
			flux[threadIdx.x+(2*blockDim.x)]=(4.0/3.0)*ty3*(u31j[threadIdx.x]-u31j[threadIdx.x-1]);
			flux[threadIdx.x+(3*blockDim.x)]=ty3*(u41j[threadIdx.x]-u41j[threadIdx.x-1]);
			flux[threadIdx.x+(4*blockDim.x)]=0.5*(1.0-C1*C5)*ty3*((u21j[threadIdx.x]*u21j[threadIdx.x]+u31j[threadIdx.x]*u31j[threadIdx.x]+u41j[threadIdx.x]*u41j[threadIdx.x])-(u21j[threadIdx.x-1]*u21j[threadIdx.x-1]+u31j[threadIdx.x-1]*u31j[threadIdx.x-1]+u41j[threadIdx.x-1]*u41j[threadIdx.x-1]))+(1.0/6.0)*ty3*(u31j[threadIdx.x]*u31j[threadIdx.x]-u31j[threadIdx.x-1]*u31j[threadIdx.x-1])+C1*C5*ty3*(u51j[threadIdx.x]-u51j[threadIdx.x-1]);
		}
		__syncthreads();
		if((threadIdx.x>=1)&&(threadIdx.x<(blockDim.x-1)&&(j<(ny-1)))){
			rtmp[threadIdx.x*5+0]+=dy1*ty1*(utmp[5*(threadIdx.x-1)]-2.0*utmp[threadIdx.x*5+0]+utmp[5*(threadIdx.x+1)]);
			rtmp[threadIdx.x*5+1]+=ty3*C3*C4*(flux[(threadIdx.x+1)+(1*blockDim.x)]-flux[threadIdx.x+(1*blockDim.x)])+dy2*ty1*(utmp[5*threadIdx.x-4]-2.0*utmp[threadIdx.x*5+1]+utmp[5*threadIdx.x+6]);
			rtmp[threadIdx.x*5+2]+=ty3*C3*C4*(flux[(threadIdx.x+1)+(2*blockDim.x)]-flux[threadIdx.x+(2*blockDim.x)])+dy3*ty1*(utmp[5*threadIdx.x-3]-2.0*utmp[threadIdx.x*5+2]+utmp[5*threadIdx.x+7]);
			rtmp[threadIdx.x*5+3]+=ty3*C3*C4*(flux[(threadIdx.x+1)+(3*blockDim.x)]-flux[threadIdx.x+(3*blockDim.x)])+dy4*ty1*(utmp[5*threadIdx.x-2]-2.0*utmp[threadIdx.x*5+3]+utmp[5*threadIdx.x+8]);
			rtmp[threadIdx.x*5+4]+=ty3*C3*C4*(flux[(threadIdx.x+1)+(4*blockDim.x)]-flux[threadIdx.x+(4*blockDim.x)])+dy5*ty1*(utmp[5*threadIdx.x-1]-2.0*utmp[threadIdx.x*5+4]+utmp[5*threadIdx.x+9]);
			/*
			 * ---------------------------------------------------------------------
			 * fourth-order dissipation
			 * ---------------------------------------------------------------------
			 */
			if(j==1){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]-=dssp*(5.0*utmp[threadIdx.x*5+m]-4.0*utmp[5*threadIdx.x+m+5]+u(m,i,3,k));}}
			if(j==2){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]-=dssp*(-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,i,4,k));}}
			if((j>=3)&&(j<(ny-3))){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]-=dssp*(u(m,i,j-2,k)-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,i,j+2,k));}}
			if(j==(ny-3)){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]-=dssp*(u(m,i,ny-5,k)-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]);}}
			if(j==(ny-2)){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]-=dssp*(u(m,i,ny-4,k)-4.0*utmp[threadIdx.x*5+m-5]+5.0*utmp[threadIdx.x*5+m]);}}
		}
		m=threadIdx.x;
		rsd(m%5, i, (j-threadIdx.x)+m/5, k)=rtmp[m];
		m+=nthreads;
		rsd(m%5, i, (j-threadIdx.x)+m/5, k)=rtmp[m];
		m+=nthreads;
		rsd(m%5, i, (j-threadIdx.x)+m/5, k)=rtmp[m];
		m+=nthreads;
		rsd(m%5, i, (j-threadIdx.x)+m/5, k)=rtmp[m];
		m+=nthreads;
		rsd(m%5, i, (j-threadIdx.x)+m/5, k)=rtmp[m];
		j+=blockDim.x-2;
	}
}

__global__ static void rhs_gpu_kernel_4(const double* u,
		double* rsd,
		const double* qs,
		const double* rho_i,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m, nthreads;
	double q, u41;

	double* flux = (double*)extern_share_data;
	double* utmp = (double*)flux+(blockDim.x*5);
	double* rtmp = (double*)utmp+(blockDim.x*5);
	double* rhotmp = (double*)rtmp+(blockDim.x*5);
	double* u21k = (double*)rhotmp+(blockDim.x);
	double* u31k = (double*)u21k+(blockDim.x);
	double* u41k = (double*)u31k+(blockDim.x);
	double* u51k = (double*)u41k+(blockDim.x);

	j=blockIdx.x+1;
	i=blockIdx.y+1;
	k=threadIdx.x;

	using namespace constants_device;
	while(k<nz){
		nthreads=(nz-(k-threadIdx.x));
		if(nthreads>blockDim.x){nthreads=blockDim.x;}
		m=threadIdx.x;
		utmp[m]=u(m%5, i, j, (k-threadIdx.x)+m/5);
		rtmp[m]=rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m+=nthreads;
		utmp[m]=u(m%5, i, j, (k-threadIdx.x)+m/5);
		rtmp[m]=rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m+=nthreads;
		utmp[m]=u(m%5, i, j, (k-threadIdx.x)+m/5);
		rtmp[m]=rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m+=nthreads;
		utmp[m]=u(m%5, i, j, (k-threadIdx.x)+m/5);
		rtmp[m]=rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m+=nthreads;
		utmp[m]=u(m%5, i, j, (k-threadIdx.x)+m/5);
		rtmp[m]=rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		rhotmp[threadIdx.x]=rho_i(i,j,k);
		__syncthreads();
		/*
		 * ---------------------------------------------------------------------
		 * zeta-direction flux differences
		 * ---------------------------------------------------------------------
		 */
		flux[threadIdx.x+(0*blockDim.x)]=utmp[threadIdx.x*5+3];
		u41=utmp[threadIdx.x*5+3]*rhotmp[threadIdx.x];
		q=qs(i,j,k);
		flux[threadIdx.x+(1*blockDim.x)]=utmp[threadIdx.x*5+1]*u41;
		flux[threadIdx.x+(2*blockDim.x)]=utmp[threadIdx.x*5+2]*u41;
		flux[threadIdx.x+(3*blockDim.x)]=utmp[threadIdx.x*5+3]*u41+C2*(utmp[threadIdx.x*5+4]-q);
		flux[threadIdx.x+(4*blockDim.x)]=(C1*utmp[threadIdx.x*5+4]-C2*q)*u41;
		__syncthreads();
		if((threadIdx.x>=1)&&(threadIdx.x<(blockDim.x-1))&&(k<(nz-1))){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]=rtmp[threadIdx.x*5+m]-tz2*(flux[(threadIdx.x+1)+(m*blockDim.x)]-flux[(threadIdx.x-1)+(m*blockDim.x)]);}}
		u21k[threadIdx.x]=rhotmp[threadIdx.x]*utmp[threadIdx.x*5+1];
		u31k[threadIdx.x]=rhotmp[threadIdx.x]*utmp[threadIdx.x*5+2];
		u41k[threadIdx.x]=rhotmp[threadIdx.x]*utmp[threadIdx.x*5+3];
		u51k[threadIdx.x]=rhotmp[threadIdx.x]*utmp[threadIdx.x*5+4];
		__syncthreads();
		if(threadIdx.x>=1){
			flux[threadIdx.x+(1*blockDim.x)]=tz3*(u21k[threadIdx.x]-u21k[threadIdx.x-1]);
			flux[threadIdx.x+(2*blockDim.x)]=tz3*(u31k[threadIdx.x]-u31k[threadIdx.x-1]);
			flux[threadIdx.x+(3*blockDim.x)]=(4.0/3.0)*tz3*(u41k[threadIdx.x]-u41k[threadIdx.x-1]);
			flux[threadIdx.x+(4*blockDim.x)]=0.5*(1.0-C1*C5)*tz3*((u21k[threadIdx.x]*u21k[threadIdx.x]+u31k[threadIdx.x]*u31k[threadIdx.x]+u41k[threadIdx.x]*u41k[threadIdx.x])-(u21k[threadIdx.x-1]*u21k[threadIdx.x-1]+u31k[threadIdx.x-1]*u31k[threadIdx.x-1]+u41k[threadIdx.x-1]*u41k[threadIdx.x-1]))+(1.0/6.0)*tz3*(u41k[threadIdx.x]*u41k[threadIdx.x]-u41k[threadIdx.x-1]*u41k[threadIdx.x-1])+C1*C5*tz3*(u51k[threadIdx.x]-u51k[threadIdx.x-1]);
		}
		__syncthreads();
		if((threadIdx.x>=1)&&(threadIdx.x<(blockDim.x-1))&&(k<(nz-1))){
			rtmp[threadIdx.x*5+0]+=dz1*tz1*(utmp[threadIdx.x*5-5]-2.0*utmp[threadIdx.x*5+0]+utmp[threadIdx.x*5+5]);
			rtmp[threadIdx.x*5+1]+=tz3*C3*C4*(flux[(threadIdx.x+1)+(1*blockDim.x)]-flux[threadIdx.x+(1*blockDim.x)])+dz2*tz1*(utmp[5*threadIdx.x-4]-2.0*utmp[threadIdx.x*5+1]+utmp[threadIdx.x*5+6]);
			rtmp[threadIdx.x*5+2]+=tz3*C3*C4*(flux[(threadIdx.x+1)+(2*blockDim.x)]-flux[threadIdx.x+(2*blockDim.x)])+dz3*tz1*(utmp[5*threadIdx.x-3]-2.0*utmp[threadIdx.x*5+2]+utmp[threadIdx.x*5+7]);
			rtmp[threadIdx.x*5+3]+=tz3*C3*C4*(flux[(threadIdx.x+1)+(3*blockDim.x)]-flux[threadIdx.x+(3*blockDim.x)])+dz4*tz1*(utmp[5*threadIdx.x-2]-2.0*utmp[threadIdx.x*5+3]+utmp[threadIdx.x*5+8]);
			rtmp[threadIdx.x*5+4]+=tz3*C3*C4*(flux[(threadIdx.x+1)+(4*blockDim.x)]-flux[threadIdx.x+(4*blockDim.x)])+dz5*tz1*(utmp[5*threadIdx.x-1]-2.0*utmp[threadIdx.x*5+4]+utmp[threadIdx.x*5+9]);
			/*
			 * ---------------------------------------------------------------------
			 * fourth-order dissipation
			 * ---------------------------------------------------------------------
			 */
			if(k==1){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]-=dssp*(5.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,i,j,3));}}
			if(k==2){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]-=dssp*(-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,i,j,4));}}
			if((k>=3)&&(k<(nz-3))){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]-=dssp*(u(m,i,j,k-2)-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,i,j,k+2));}}
			if(k==(nz-3)){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]-=dssp*(u(m,i,j,nz-5)-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]);}}
			if(k==(nz-2)){for(m=0;m<5;m++){rtmp[threadIdx.x*5+m]-=dssp*(u(m,i,j,nz-4)-4.0*utmp[threadIdx.x*5+m-5]+5.0*utmp[threadIdx.x*5+m]);}}
		}
		m=threadIdx.x;
		rsd(m%5, i, j, (k-threadIdx.x)+m/5)=rtmp[m];
		m+=nthreads;
		rsd(m%5, i, j, (k-threadIdx.x)+m/5)=rtmp[m];
		m+=nthreads;
		rsd(m%5, i, j, (k-threadIdx.x)+m/5)=rtmp[m];
		m+=nthreads;
		rsd(m%5, i, j, (k-threadIdx.x)+m/5)=rtmp[m];
		m+=nthreads;
		rsd(m%5, i, j, (k-threadIdx.x)+m/5)=rtmp[m];
		k+=blockDim.x-2;
	}
}

/*
 * ---------------------------------------------------------------------
 * set the boundary values of dependent variables
 * ---------------------------------------------------------------------
 */
static void setbv_gpu(){
#if defined(PROFILING)
	timer_start(PROFILING_SETBV_3);
#endif
	/* #KERNEL SETBV 3 */
	int setbv_3_workload = nx * ny;
	int setbv_3_threads_per_block = THREADS_PER_BLOCK_ON_SETBV_3;
	int setbv_3_blocks_per_grid = (ceil((double)setbv_3_workload/(double)setbv_3_threads_per_block));

	setbv_gpu_kernel_3<<<
		setbv_3_blocks_per_grid, 
		setbv_3_threads_per_block>>>(
				u_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_SETBV_3);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_SETBV_2);
#endif
	/* #KERNEL SETBV 2 */
	int setbv_2_workload = nx * nz;
	int setbv_2_threads_per_block = THREADS_PER_BLOCK_ON_SETBV_2;
	int setbv_2_blocks_per_grid = (ceil((double)setbv_2_workload/(double)setbv_2_threads_per_block));

	setbv_gpu_kernel_2<<<
		setbv_2_blocks_per_grid, 
		setbv_2_threads_per_block>>>(
				u_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_SETBV_2);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_SETBV_1);
#endif
	/* #KERNEL SETBV 1 */
	int setbv_1_workload = ny * nz;
	int setbv_1_threads_per_block = THREADS_PER_BLOCK_ON_SETBV_1;
	int setbv_1_blocks_per_grid = (ceil((double)setbv_1_workload/(double)setbv_1_threads_per_block));

	setbv_gpu_kernel_1<<<
		setbv_1_blocks_per_grid, 
		setbv_1_threads_per_block>>>(
				u_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_SETBV_1);
#endif
}

__global__ static void setbv_gpu_kernel_1(double* u,
		const int nx,
		const int ny,
		const int nz){
	int j_k, j, k, m;
	double temp1[5], temp2[5];

	j_k = blockIdx.x * blockDim.x + threadIdx.x;

	j = j_k % ny;
	k = (j_k / ny) % nz;

	if(j_k >= (nz*ny)){
		return;
	}

	/*
	 * ---------------------------------------------------------------------
	 * set the dependent variable values along east and west faces
	 * ---------------------------------------------------------------------
	 */
	exact_gpu_device(0, 
			j, 
			k, 
			temp1, 
			nx, 
			ny, 
			nz);
	exact_gpu_device(nx-1, 
			j, 
			k, 
			temp2, 
			nx, 
			ny, 
			nz);
	for(m=0; m<5; m++){
		u(m,0,j,k)=temp1[m];
		u(m,nx-1,j,k)=temp2[m];
	}
}

__global__ static void setbv_gpu_kernel_2(double* u,
		const int nx,
		const int ny,
		const int nz){
	int i_k, i, k, m;
	double temp1[5], temp2[5];

	i_k = blockIdx.x * blockDim.x + threadIdx.x;

	i = i_k % nx;
	k = (i_k / nx) % nz;

	if(i_k >= (nx*nz)){
		return;
	}

	/*
	 * ---------------------------------------------------------------------
	 * set the dependent variable values along north and south faces
	 * ---------------------------------------------------------------------
	 */
	exact_gpu_device(i, 
			0, 
			k, 
			temp1, 
			nx, 
			ny, 
			nz);
	exact_gpu_device(i, 
			ny-1, 
			k, 
			temp2, 
			nx, 
			ny, 
			nz);
	for(m=0; m<5; m++){
		u(m,i,0,k)=temp1[m];
		u(m,i,ny-1,k)=temp2[m];
	}
}

__global__ static void setbv_gpu_kernel_3(double* u,
		const int nx,
		const int ny,
		const int nz){
	int i_j, i, j, m;
	double temp1[5], temp2[5];

	i_j = blockIdx.x * blockDim.x + threadIdx.x;

	i = i_j % nx;
	j = (i_j / nx) % ny;

	if(i_j >= (nx*ny)){
		return;
	}

	/*
	 * ---------------------------------------------------------------------
	 * set the dependent variable values along the top and bottom faces
	 * ---------------------------------------------------------------------
	 */
	exact_gpu_device(i, 
			j, 
			0, 
			temp1, 
			nx, 
			ny, 
			nz);
	exact_gpu_device(i, 
			j, 
			nz-1, 
			temp2, 
			nx, 
			ny, 
			nz);
	for(m=0; m<5; m++){
		u(m,i,j,0)=temp1[m];
		u(m,i,j,nz-1)=temp2[m];
	}
}

static void setcoeff_gpu(){
	double dt=dt_host;
	double omega=omega_host;
	/* grid */
	static double dxi, deta, dzeta;
	static double tx1, tx2, tx3;
	static double ty1, ty2, ty3;
	static double tz1, tz2, tz3;	
	/* dissipation */
	static double dx1, dx2, dx3, dx4, dx5;
	static double dy1, dy2, dy3, dy4, dy5;
	static double dz1, dz2, dz3, dz4, dz5;
	static double dssp;	
	/*
	 * ---------------------------------------------------------------------
	 * local variables
	 * ---------------------------------------------------------------------
	 * set up coefficients
	 * ---------------------------------------------------------------------
	 */
	dxi=1.0/(nx-1);
	deta=1.0/(ny-1);
	dzeta=1.0/(nz-1);
	tx1=1.0/(dxi*dxi);
	tx2=1.0/(2.0*dxi);
	tx3=1.0/dxi;
	ty1=1.0/(deta*deta);
	ty2=1.0/(2.0*deta);
	ty3=1.0/deta;
	tz1=1.0/(dzeta*dzeta);
	tz2=1.0/(2.0*dzeta);
	tz3=1.0/dzeta;
	/*
	 * ---------------------------------------------------------------------
	 * diffusion coefficients
	 * ---------------------------------------------------------------------
	 */
	dx1=0.75;
	dx2=dx1;
	dx3=dx1;
	dx4=dx1;
	dx5=dx1;
	dy1=0.75;
	dy2=dy1;
	dy3=dy1;
	dy4=dy1;
	dy5=dy1;
	dz1=1.00;
	dz2=dz1;
	dz3=dz1;
	dz4=dz1;
	dz5=dz1;
	/*
	 * ---------------------------------------------------------------------
	 * fourth difference dissipation
	 * ---------------------------------------------------------------------
	 */
	dssp=(max(max(dx1,dy1),dz1))/4.0;	 
	/*
	 * ---------------------------------------------------------------------
	 * coefficients of the exact solution to the first pde
	 * ---------------------------------------------------------------------
	 */
	double ce[13][5];
	ce[0][0]=2.0;
	ce[1][0]=0.0;
	ce[2][0]=0.0;
	ce[3][0]=4.0;
	ce[4][0]=5.0;
	ce[5][0]=3.0;
	ce[6][0]=0.5;
	ce[7][0]=0.02;
	ce[8][0]=0.01;
	ce[9][0]=0.03;
	ce[10][0]=0.5;
	ce[11][0]=0.4;
	ce[12][0]=0.3;
	/*
	 * ---------------------------------------------------------------------
	 * coefficients of the exact solution to the second pde
	 * ---------------------------------------------------------------------
	 */
	ce[0][1]=1.0;
	ce[1][1]=0.0;
	ce[2][1]=0.0;
	ce[3][1]=0.0;
	ce[4][1]=1.0;
	ce[5][1]=2.0;
	ce[6][1]=3.0;
	ce[7][1]=0.01;
	ce[8][1]=0.03;
	ce[9][1]=0.02;
	ce[10][1]=0.4;
	ce[11][1]=0.3;
	ce[12][1]=0.5;
	/*
	 * ---------------------------------------------------------------------
	 * coefficients of the exact solution to the third pde
	 * ---------------------------------------------------------------------
	 */
	ce[0][2]=2.0;
	ce[1][2]=2.0;
	ce[2][2]=0.0;
	ce[3][2]=0.0;
	ce[4][2]=0.0;
	ce[5][2]=2.0;
	ce[6][2]=3.0;
	ce[7][2]=0.04;
	ce[8][2]=0.03;
	ce[9][2]=0.05;
	ce[10][2]=0.3;
	ce[11][2]=0.5;
	ce[12][2]=0.4;
	/*
	 * ---------------------------------------------------------------------
	 * coefficients of the exact solution to the fourth pde
	 * ---------------------------------------------------------------------
	 */
	ce[0][3]=2.0;
	ce[1][3]=2.0;
	ce[2][3]=0.0;
	ce[3][3]=0.0;
	ce[4][3]=0.0;
	ce[5][3]=2.0;
	ce[6][3]=3.0;
	ce[7][3]=0.03;
	ce[8][3]=0.05;
	ce[9][3]=0.04;
	ce[10][3]=0.2;
	ce[11][3]=0.1;
	ce[12][3]=0.3;
	/*
	 * ---------------------------------------------------------------------
	 * coefficients of the exact solution to the fifth pde
	 * ---------------------------------------------------------------------
	 */
	ce[0][4]=5.0;
	ce[1][4]=4.0;
	ce[2][4]=3.0;
	ce[3][4]=2.0;
	ce[4][4]=0.1;
	ce[5][4]=0.4;
	ce[6][4]=0.3;
	ce[7][4]=0.05;
	ce[8][4]=0.04;
	ce[9][4]=0.03;
	ce[10][4]=0.1;
	ce[11][4]=0.3;
	ce[12][4]=0.2;
	/* */
	hipMemcpyToSymbol(constants_device::dx1, &dx1, sizeof(double));
	hipMemcpyToSymbol(constants_device::dx2, &dx2, sizeof(double));
	hipMemcpyToSymbol(constants_device::dx3, &dx3, sizeof(double));
	hipMemcpyToSymbol(constants_device::dx4, &dx4, sizeof(double));
	hipMemcpyToSymbol(constants_device::dx5, &dx5, sizeof(double));
	hipMemcpyToSymbol(constants_device::dy1, &dy1, sizeof(double));
	hipMemcpyToSymbol(constants_device::dy2, &dy2, sizeof(double));
	hipMemcpyToSymbol(constants_device::dy3, &dy3, sizeof(double));
	hipMemcpyToSymbol(constants_device::dy4, &dy4, sizeof(double));
	hipMemcpyToSymbol(constants_device::dy5, &dy5, sizeof(double));
	hipMemcpyToSymbol(constants_device::dz1, &dz1, sizeof(double));
	hipMemcpyToSymbol(constants_device::dz2, &dz2, sizeof(double));
	hipMemcpyToSymbol(constants_device::dz3, &dz3, sizeof(double));
	hipMemcpyToSymbol(constants_device::dz4, &dz4, sizeof(double));
	hipMemcpyToSymbol(constants_device::dz5, &dz5, sizeof(double));
	hipMemcpyToSymbol(constants_device::dssp, &dssp, sizeof(double));
	hipMemcpyToSymbol(constants_device::dxi, &dxi, sizeof(double));
	hipMemcpyToSymbol(constants_device::deta, &deta, sizeof(double));
	hipMemcpyToSymbol(constants_device::dzeta, &dzeta, sizeof(double));
	hipMemcpyToSymbol(constants_device::tx1, &tx1, sizeof(double));
	hipMemcpyToSymbol(constants_device::tx2, &tx2, sizeof(double));
	hipMemcpyToSymbol(constants_device::tx3, &tx3, sizeof(double));
	hipMemcpyToSymbol(constants_device::ty1, &ty1, sizeof(double));
	hipMemcpyToSymbol(constants_device::ty2, &ty2, sizeof(double));
	hipMemcpyToSymbol(constants_device::ty3, &ty3, sizeof(double));
	hipMemcpyToSymbol(constants_device::tz1, &tz1, sizeof(double));
	hipMemcpyToSymbol(constants_device::tz2, &tz2, sizeof(double));
	hipMemcpyToSymbol(constants_device::tz3, &tz3, sizeof(double));
	hipMemcpyToSymbol(constants_device::ce, &ce, 13*5*sizeof(double));
	hipMemcpyToSymbol(constants_device::dt, &dt, sizeof(double));
	hipMemcpyToSymbol(constants_device::omega, &omega, sizeof(double));
}

/*
 * ---------------------------------------------------------------------
 * set the initial values of independent variables based on tri-linear
 * interpolation of boundary values in the computational space.
 * ---------------------------------------------------------------------
 */
static void setiv_gpu(){
#if defined(PROFILING)
	timer_start(PROFILING_SETIV);
#endif
	/* #KERNEL SETIV */
	int setiv_threads_per_block;
	dim3 setiv_blocks_per_grid(nz-2, ny-2);
	if(THREADS_PER_BLOCK_ON_SETIV != nx-2){
		setiv_threads_per_block = nx-2;
	}
	else{
		setiv_threads_per_block = THREADS_PER_BLOCK_ON_SETIV;
	}

	setiv_gpu_kernel<<<
		setiv_blocks_per_grid, 
		setiv_threads_per_block>>>(
				u_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_SETIV);
#endif
}

__global__ static void setiv_gpu_kernel(double* u,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m;
	double xi, eta, zeta, pxi, peta, pzeta;
	double ue_1jk[5], ue_nx0jk[5], ue_i1k[5], ue_iny0k[5], ue_ij1[5], ue_ijnz[5];	

	k=blockIdx.x+1;
	j=blockIdx.y+1;
	i=threadIdx.x+1;

	zeta=(double)k/(double)(nz-1);
	eta=(double)j/(double)(ny-1);
	xi=(double)i/(double)(nx-1);
	exact_gpu_device(0, j, k, ue_1jk, nx, ny, nz);
	exact_gpu_device(nx-1, j, k, ue_nx0jk, nx, ny, nz);
	exact_gpu_device(i, 0, k, ue_i1k, nx, ny, nz);
	exact_gpu_device(i, ny-1, k, ue_iny0k, nx, ny, nz);
	exact_gpu_device(i, j, 0, ue_ij1, nx, ny, nz);
	exact_gpu_device(i, j, nz-1, ue_ijnz, nx, ny, nz);
	for(m=0; m<5; m++){
		pxi=(1.0-xi)*ue_1jk[m]+xi*ue_nx0jk[m];
		peta=(1.0-eta)*ue_i1k[m]+eta*ue_iny0k[m];
		pzeta=(1.0-zeta)*ue_ij1[m]+zeta*ue_ijnz[m];
		u(m,i,j,k)=pxi+peta+pzeta-pxi*peta-peta*pzeta-pzeta*pxi+pxi*peta*pzeta;
	}
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
	if((LU_THREADS_PER_BLOCK_ON_ERHS_1>=1)&&
			(LU_THREADS_PER_BLOCK_ON_ERHS_1<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_ERHS_1 = LU_THREADS_PER_BLOCK_ON_ERHS_1;
	}
	else{
		THREADS_PER_BLOCK_ON_ERHS_1 = gpu_device_properties.warpSize;
	}
	if((LU_THREADS_PER_BLOCK_ON_ERHS_2>=1)&&
			(LU_THREADS_PER_BLOCK_ON_ERHS_2<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_ERHS_2 = LU_THREADS_PER_BLOCK_ON_ERHS_2;
	}
	else{
		THREADS_PER_BLOCK_ON_ERHS_2 = gpu_device_properties.warpSize;
	}
	if((LU_THREADS_PER_BLOCK_ON_ERHS_3>=1)&&
			(LU_THREADS_PER_BLOCK_ON_ERHS_3<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_ERHS_3 = LU_THREADS_PER_BLOCK_ON_ERHS_3;
	}
	else{
		THREADS_PER_BLOCK_ON_ERHS_3 = gpu_device_properties.warpSize;
	}	
	if((LU_THREADS_PER_BLOCK_ON_ERHS_4>=1)&&
			(LU_THREADS_PER_BLOCK_ON_ERHS_4<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_ERHS_4 = LU_THREADS_PER_BLOCK_ON_ERHS_4;
	}
	else{
		THREADS_PER_BLOCK_ON_ERHS_4 = gpu_device_properties.warpSize;
	}
	if((LU_THREADS_PER_BLOCK_ON_ERROR>=1)&&
			(LU_THREADS_PER_BLOCK_ON_ERROR<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_ERROR = LU_THREADS_PER_BLOCK_ON_ERROR;
	}
	else{
		THREADS_PER_BLOCK_ON_ERROR = gpu_device_properties.warpSize;
	}
	if((LU_THREADS_PER_BLOCK_ON_NORM>=1)&&
			(LU_THREADS_PER_BLOCK_ON_NORM<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_NORM = LU_THREADS_PER_BLOCK_ON_NORM;
	}
	else{
		THREADS_PER_BLOCK_ON_NORM = gpu_device_properties.warpSize;
	}
	if((LU_THREADS_PER_BLOCK_ON_JACLD_BLTS>=1)&&
			(LU_THREADS_PER_BLOCK_ON_JACLD_BLTS<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_JACLD_BLTS = LU_THREADS_PER_BLOCK_ON_JACLD_BLTS;
	}
	else{
		THREADS_PER_BLOCK_ON_JACLD_BLTS = gpu_device_properties.warpSize;
	}
	if((LU_THREADS_PER_BLOCK_ON_JACU_BUTS>=1)&&
			(LU_THREADS_PER_BLOCK_ON_JACU_BUTS<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_JACU_BUTS = LU_THREADS_PER_BLOCK_ON_JACU_BUTS;
	}
	else{
		THREADS_PER_BLOCK_ON_JACU_BUTS = gpu_device_properties.warpSize;
	}
	if((LU_THREADS_PER_BLOCK_ON_L2NORM>=1)&&
			(LU_THREADS_PER_BLOCK_ON_L2NORM<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_L2NORM = LU_THREADS_PER_BLOCK_ON_L2NORM;
	}
	else{
		THREADS_PER_BLOCK_ON_L2NORM=gpu_device_properties.warpSize;
	}
	if((LU_THREADS_PER_BLOCK_ON_PINTGR_1>=1)&&
			(LU_THREADS_PER_BLOCK_ON_PINTGR_1<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_PINTGR_1 = LU_THREADS_PER_BLOCK_ON_PINTGR_1;
	}
	else{
		THREADS_PER_BLOCK_ON_PINTGR_1=gpu_device_properties.warpSize;
	}
	if((LU_THREADS_PER_BLOCK_ON_PINTGR_2>=1)&&
			(LU_THREADS_PER_BLOCK_ON_PINTGR_2<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_PINTGR_2 = LU_THREADS_PER_BLOCK_ON_PINTGR_2;
	}
	else{
		THREADS_PER_BLOCK_ON_PINTGR_2 = gpu_device_properties.warpSize;
	}	
	if((LU_THREADS_PER_BLOCK_ON_PINTGR_3>=1)&&
			(LU_THREADS_PER_BLOCK_ON_PINTGR_3<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_PINTGR_3 = LU_THREADS_PER_BLOCK_ON_PINTGR_3;
	}
	else{
		THREADS_PER_BLOCK_ON_PINTGR_3 = gpu_device_properties.warpSize;
	}	
	if((LU_THREADS_PER_BLOCK_ON_PINTGR_4>=1)&&
			(LU_THREADS_PER_BLOCK_ON_PINTGR_4<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_PINTGR_4 = LU_THREADS_PER_BLOCK_ON_PINTGR_4;
	}
	else{
		THREADS_PER_BLOCK_ON_PINTGR_4 = gpu_device_properties.warpSize;
	}	
	if((LU_THREADS_PER_BLOCK_ON_RHS_1>=1)&&
			(LU_THREADS_PER_BLOCK_ON_RHS_1<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_RHS_1 = LU_THREADS_PER_BLOCK_ON_RHS_1;
	}
	else{
		THREADS_PER_BLOCK_ON_RHS_1 = gpu_device_properties.warpSize;
	}	
	if((LU_THREADS_PER_BLOCK_ON_RHS_2>=1)&&
			(LU_THREADS_PER_BLOCK_ON_RHS_2<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_RHS_2 = LU_THREADS_PER_BLOCK_ON_RHS_2;
	}
	else{
		THREADS_PER_BLOCK_ON_RHS_2 = gpu_device_properties.warpSize;
	}	
	if((LU_THREADS_PER_BLOCK_ON_RHS_3>=1)&&
			(LU_THREADS_PER_BLOCK_ON_RHS_3<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_RHS_3 = LU_THREADS_PER_BLOCK_ON_RHS_3;
	}
	else{
		THREADS_PER_BLOCK_ON_RHS_3 = gpu_device_properties.warpSize;
	}	
	if((LU_THREADS_PER_BLOCK_ON_RHS_4>=1)&&
			(LU_THREADS_PER_BLOCK_ON_RHS_4<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_RHS_4 = LU_THREADS_PER_BLOCK_ON_RHS_4;
	}
	else{
		THREADS_PER_BLOCK_ON_RHS_4 = gpu_device_properties.warpSize;
	}	
	if((LU_THREADS_PER_BLOCK_ON_SETBV_1>=1)&&
			(LU_THREADS_PER_BLOCK_ON_SETBV_1<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_SETBV_1 = LU_THREADS_PER_BLOCK_ON_SETBV_1;
	}
	else{
		THREADS_PER_BLOCK_ON_SETBV_1 = gpu_device_properties.warpSize;
	}
	if((LU_THREADS_PER_BLOCK_ON_SETBV_2>=1)&&
			(LU_THREADS_PER_BLOCK_ON_SETBV_2<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_SETBV_2 = LU_THREADS_PER_BLOCK_ON_SETBV_2;
	}
	else{
		THREADS_PER_BLOCK_ON_SETBV_2 = gpu_device_properties.warpSize;
	}
	if((LU_THREADS_PER_BLOCK_ON_SETBV_3>=1)&&
			(LU_THREADS_PER_BLOCK_ON_SETBV_3<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_SETBV_3 = LU_THREADS_PER_BLOCK_ON_SETBV_3;
	}
	else{
		THREADS_PER_BLOCK_ON_SETBV_3 = gpu_device_properties.warpSize;
	}
	if((LU_THREADS_PER_BLOCK_ON_SETIV>=1)&&
			(LU_THREADS_PER_BLOCK_ON_SETIV<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_SETIV = LU_THREADS_PER_BLOCK_ON_SETIV;
	}
	else{
		THREADS_PER_BLOCK_ON_SETIV = gpu_device_properties.warpSize;
	}
	if((LU_THREADS_PER_BLOCK_ON_SSOR_1>=1)&&
			(LU_THREADS_PER_BLOCK_ON_SSOR_1<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_SSOR_1 = LU_THREADS_PER_BLOCK_ON_SSOR_1;
	}
	else{
		THREADS_PER_BLOCK_ON_SSOR_1 = gpu_device_properties.warpSize;
	}
	if((LU_THREADS_PER_BLOCK_ON_SSOR_2>=1)&&
			(LU_THREADS_PER_BLOCK_ON_SSOR_2<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_SSOR_2 = LU_THREADS_PER_BLOCK_ON_SSOR_2;
	}
	else{
		THREADS_PER_BLOCK_ON_SSOR_2 = gpu_device_properties.warpSize;
	}

	int gridsize=nx*ny*nz;
	int norm_buf_size=max(5*(ny-2)*(nz-2), ((nx-3)*(ny-3)+(nx-3)*(nz-3)+(ny-3)*(nz-3))/((gpu_device_properties.maxThreadsPerBlock-1)*(gpu_device_properties.maxThreadsPerBlock-1))+3);
	size_u_device=sizeof(double)*(5*gridsize);
	size_rsd_device=sizeof(double)*(5*gridsize);
	size_frct_device=sizeof(double)*(5*gridsize);
	size_rho_i_device=sizeof(double)*(gridsize);
	size_qs_device=sizeof(double)*(gridsize);
	size_norm_buffer_device=sizeof(double)*(norm_buf_size);
	hipMalloc(&u_device, size_u_device);
	hipMalloc(&rsd_device, size_rsd_device);
	hipMalloc(&frct_device, size_frct_device);
	hipMalloc(&rho_i_device, size_rho_i_device);
	hipMalloc(&qs_device, size_qs_device);
	hipMalloc(&norm_buffer_device, size_norm_buffer_device);
}

/*
 * ---------------------------------------------------------------------
 * to perform pseudo-time stepping SSOR iterations
 * for five nonlinear pde's.
 * ---------------------------------------------------------------------
 */
static void ssor_gpu(int niter){
	double omega=omega_host;
	double tmp=1.0/(omega*(2.0-omega));
	/*
	 * ---------------------------------------------------------------------
	 * compute the steady-state residuals
	 * ---------------------------------------------------------------------
	 */
	rhs_gpu();
	/*
	 * ---------------------------------------------------------------------
	 * compute the L2 norms of newton iteration residuals
	 * ---------------------------------------------------------------------
	 */
	l2norm_gpu(rsd_device, rsdnm);
	timer_clear(PROFILING_TOTAL_TIME);
#if defined(PROFILING)
	timer_clear(PROFILING_ERHS_1);
	timer_clear(PROFILING_ERHS_2);
	timer_clear(PROFILING_ERHS_3);
	timer_clear(PROFILING_ERHS_4);
	timer_clear(PROFILING_ERROR);
	timer_clear(PROFILING_NORM);
	timer_clear(PROFILING_JACLD_BLTS);
	timer_clear(PROFILING_JACU_BUTS);
	timer_clear(PROFILING_L2NORM);
	timer_clear(PROFILING_PINTGR_1);
	timer_clear(PROFILING_PINTGR_2);
	timer_clear(PROFILING_PINTGR_3);
	timer_clear(PROFILING_PINTGR_4);
	timer_clear(PROFILING_RHS_1);
	timer_clear(PROFILING_RHS_2);
	timer_clear(PROFILING_RHS_3);
	timer_clear(PROFILING_RHS_4);
	timer_clear(PROFILING_SETBV_1);
	timer_clear(PROFILING_SETBV_2);
	timer_clear(PROFILING_SETBV_3);
	timer_clear(PROFILING_SETIV);
	timer_clear(PROFILING_SSOR_1);
	timer_clear(PROFILING_SSOR_2);
#endif
	timer_start(PROFILING_TOTAL_TIME);/*#start_timer*/
	/*
	 * ---------------------------------------------------------------------
	 * the timestep loop
	 * ---------------------------------------------------------------------
	 */
	for(int istep=1; istep<=niter; istep++){
		if((istep%20)==0||istep==itmax||istep==1){if(niter>1){printf(" Time step %4d\n",istep);}}
		/*
		 * ---------------------------------------------------------------------
		 * perform SSOR iteration
		 * ---------------------------------------------------------------------
		 */
#if defined(PROFILING)
		timer_start(PROFILING_SSOR_1);
#endif
		int ssor_1_threads_per_block=THREADS_PER_BLOCK_ON_SSOR_1;
		dim3 ssor_1_blocks_per_grid(nz-2, ny-2);

		ssor_gpu_kernel_1<<<
			ssor_1_blocks_per_grid, 
			max(nx-2, ssor_1_threads_per_block)>>>(
					rsd_device, 
					nx, 
					ny, 
					nz);
#if defined(PROFILING)
		timer_stop(PROFILING_SSOR_1);
#endif
		/*
		 * ---------------------------------------------------------------------
		 * form the lower triangular part of the jacobian matrix
		 * ---------------------------------------------------------------------
		 * perform the lower triangular solution
		 * ---------------------------------------------------------------------
		 */
#if defined(PROFILING)
		timer_start(PROFILING_JACLD_BLTS);
#endif
		for(int plane=0; plane<=nx+ny+nz-9; plane++){
			int klower=max(0, plane-(nx-3)-(ny-3));
			int kupper=min(plane, nz-3);
			int jlowermin=max(0, plane-kupper-(nx-3));
			int juppermax=min(plane, ny-3);

			/* #KERNEL JACLD BLTS */
			int jacld_blts_threads_per_block=THREADS_PER_BLOCK_ON_JACLD_BLTS;
			int jacld_blts_blocks_per_grid = kupper-klower+1;
			if(THREADS_PER_BLOCK_ON_JACLD_BLTS != (juppermax-jlowermin+1)){
				jacld_blts_threads_per_block = juppermax-jlowermin+1;
			}
			else{
				jacld_blts_threads_per_block = THREADS_PER_BLOCK_ON_JACLD_BLTS;
			}

			jacld_blts_gpu_kernel<<<
				jacld_blts_blocks_per_grid, 
				jacld_blts_threads_per_block>>>(
						plane, 
						klower, 
						jlowermin, 
						u_device, 
						rho_i_device, 
						qs_device, 
						rsd_device, 
						nx, 
						ny, 
						nz);
		}
#if defined(PROFILING)
		timer_stop(PROFILING_JACLD_BLTS);
#endif  
		/*
		 * ---------------------------------------------------------------------
		 * form the strictly upper triangular part of the jacobian matrix
		 * ---------------------------------------------------------------------
		 * perform the upper triangular solution
		 * ---------------------------------------------------------------------
		 */
#if defined(PROFILING)
		timer_start(PROFILING_JACU_BUTS);
#endif
		for(int plane=nx+ny+nz-9; plane>=0; plane--){
			int klower=max(0, plane-(nx-3)-(ny-3));
			int kupper=min(plane, nz-3);
			int jlowermin=max(0, plane-kupper-(nx-3));
			int juppermax=min(plane, ny-3);

			/* #KERNEL JACLD BLTS */
			int jacu_buts_threads_per_block=THREADS_PER_BLOCK_ON_JACU_BUTS;
			int jacu_buts_blocks_per_grid = kupper-klower+1;
			if(THREADS_PER_BLOCK_ON_JACU_BUTS != (juppermax-jlowermin+1)){
				jacu_buts_threads_per_block = juppermax-jlowermin+1;
			}
			else{
				jacu_buts_threads_per_block = THREADS_PER_BLOCK_ON_JACU_BUTS;
			}

			jacu_buts_gpu_kernel<<<
				jacu_buts_blocks_per_grid, 
				jacu_buts_threads_per_block>>>(
						plane, 
						klower, 
						jlowermin, 
						u_device, 
						rho_i_device, 
						qs_device, 
						rsd_device, 
						nx, 
						ny, 
						nz);
		}
#if defined(PROFILING)
		timer_stop(PROFILING_JACU_BUTS);
#endif  
		/*
		 * ---------------------------------------------------------------------
		 * update the variables
		 * ---------------------------------------------------------------------
		 */
#if defined(PROFILING)
		timer_start(PROFILING_SSOR_2);
#endif
		int ssor_2_threads_per_block=THREADS_PER_BLOCK_ON_SSOR_2;
		dim3 ssor_2_blocks_per_grid(nz-2, ny-2);

		ssor_gpu_kernel_2<<<
			ssor_2_blocks_per_grid, 
			max(nx-2, ssor_2_threads_per_block)>>>(
					u_device, 
					rsd_device, 
					tmp, 
					nx, 
					ny, 
					nz);
#if defined(PROFILING)
		timer_stop(PROFILING_SSOR_2);
#endif
		/*
		 * ---------------------------------------------------------------------
		 * compute the max-norms of newton iteration corrections
		 * ---------------------------------------------------------------------
		 */
		if(istep%inorm==0){
			double delunm[5];
			l2norm_gpu(rsd_device, delunm);
		}
		/*
		 * ---------------------------------------------------------------------
		 * compute the steady-state residuals
		 * ---------------------------------------------------------------------
		 */
		rhs_gpu();
		/*
		 * ---------------------------------------------------------------------
		 * compute the max-norms of newton iteration residuals
		 * ---------------------------------------------------------------------
		 */
		if(istep%inorm==0){
			l2norm_gpu(rsd_device, rsdnm);
		}
		/*
		 * ---------------------------------------------------------------------
		 * check the newton-iteration residuals against the tolerance levels
		 * ---------------------------------------------------------------------
		 */
		if((rsdnm[0]<tolrsd[0])&&(rsdnm[1]<tolrsd[1])&&(rsdnm[2]<tolrsd[2])&&(rsdnm[3]<tolrsd[3])&&(rsdnm[4]<tolrsd[4])){
			printf("\n convergence was achieved after %4d pseudo-time steps\n", istep);
			break;
		}
	}
	timer_stop(PROFILING_TOTAL_TIME);/*#stop_timer*/
	maxtime=timer_read(PROFILING_TOTAL_TIME);
}

__global__ static void ssor_gpu_kernel_1(double* rsd,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k;

	if(threadIdx.x >= (nx-2)){
		return;
	}

	i=threadIdx.x+1;
	j=blockIdx.y+1;
	k=blockIdx.x+1;

	using namespace constants_device;
	rsd(0,i,j,k)*=dt;
	rsd(1,i,j,k)*=dt;
	rsd(2,i,j,k)*=dt;
	rsd(3,i,j,k)*=dt;
	rsd(4,i,j,k)*=dt;
}

__global__ static void ssor_gpu_kernel_2(double* u,
		double* rsd,
		const double tmp,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k;

	if(threadIdx.x >= (nx-2)){
		return;
	}

	i=threadIdx.x+1;
	j=blockIdx.y+1;
	k=blockIdx.x+1;

	u(0,i,j,k)+=tmp*rsd(0,i,j,k);
	u(1,i,j,k)+=tmp*rsd(1,i,j,k);
	u(2,i,j,k)+=tmp*rsd(2,i,j,k);
	u(3,i,j,k)+=tmp*rsd(3,i,j,k);
	u(4,i,j,k)+=tmp*rsd(4,i,j,k);
}

/*
 * ---------------------------------------------------------------------
 * verification routine                         
 * ---------------------------------------------------------------------
 */
static void verify_gpu(double xcr[],
		double xce[],
		double xci,
		char* class_npb,
		boolean* verified){
	double dt=dt_host;
	double xcrref[5], xceref[5], xciref;
	double xcrdif[5], xcedif[5], xcidif;
	double epsilon, dtref=0.0;
	int m;
	/*
	 * ---------------------------------------------------------------------
	 * tolerance level
	 * ---------------------------------------------------------------------
	 */
	epsilon=1.0e-08;
	*class_npb='U';
	*verified=TRUE;
	for(m=0; m<5; m++){
		xcrref[m]=1.0;
		xceref[m]=1.0;
	}
	xciref=1.0;
	if((nx==12)&&(ny==12)&&(nz==12)&&(itmax==50)){
		*class_npb='S';
		dtref=5.0e-1;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of residual, for the (12X12X12) grid,
		 * after 50 time steps, with DT = 5.0d-01
		 * ---------------------------------------------------------------------
		 */
		xcrref[0]=1.6196343210976702e-02;
		xcrref[1]=2.1976745164821318e-03;
		xcrref[2]=1.5179927653399185e-03;
		xcrref[3]=1.5029584435994323e-03;
		xcrref[4]=3.4264073155896461e-02;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of solution error, for the (12X12X12) grid,
		 * after 50 time steps, with DT = 5.0d-01
		 * ---------------------------------------------------------------------
		 */
		xceref[0]=6.4223319957960924e-04;
		xceref[1]=8.4144342047347926e-05;
		xceref[2]=5.8588269616485186e-05;
		xceref[3]=5.8474222595157350e-05;
		xceref[4]=1.3103347914111294e-03;
		/*
		 * ---------------------------------------------------------------------
		 * reference value of surface integral, for the (12X12X12) grid,
		 * after 50 time steps, with DT = 5.0d-01
		 * ---------------------------------------------------------------------
		 */
		xciref=7.8418928865937083e+00;
	}else if((nx==33)&&(ny==33)&&(nz==33)&&(itmax==300)){
		*class_npb='W'; /* SPEC95fp size */
		dtref=1.5e-3;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of residual, for the (33x33x33) grid,
		 * after 300 time steps, with DT = 1.5d-3
		 * ---------------------------------------------------------------------
		 */
		xcrref[0]=0.1236511638192e+02;
		xcrref[1]=0.1317228477799e+01;
		xcrref[2]=0.2550120713095e+01;
		xcrref[3]=0.2326187750252e+01;
		xcrref[4]=0.2826799444189e+02;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of solution error, for the (33X33X33) grid,
		 * ---------------------------------------------------------------------
		 */
		xceref[0]=0.4867877144216e+00;
		xceref[1]=0.5064652880982e-01;
		xceref[2]=0.9281818101960e-01;
		xceref[3]=0.8570126542733e-01;
		xceref[4]=0.1084277417792e+01;
		/*
		 * ---------------------------------------------------------------------
		 * rReference value of surface integral, for the (33X33X33) grid,
		 * after 300 time steps, with DT = 1.5d-3
		 * ---------------------------------------------------------------------
		 */
		xciref=0.1161399311023e+02;
	}else if((nx==64)&&(ny==64)&&(nz==64)&&(itmax==250)){
		*class_npb='A';
		dtref=2.0e+0;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of residual, for the (64X64X64) grid,
		 * after 250 time steps, with DT = 2.0d+00
		 * ---------------------------------------------------------------------
		 */
		xcrref[0]=7.7902107606689367e+02;
		xcrref[1]=6.3402765259692870e+01;
		xcrref[2]=1.9499249727292479e+02;
		xcrref[3]=1.7845301160418537e+02;
		xcrref[4]=1.8384760349464247e+03;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of solution error, for the (64X64X64) grid,
		 * after 250 time steps, with DT = 2.0d+00
		 * ---------------------------------------------------------------------
		 */
		xceref[0]=2.9964085685471943e+01;
		xceref[1]=2.8194576365003349e+00;
		xceref[2]=7.3473412698774742e+00;
		xceref[3]=6.7139225687777051e+00;
		xceref[4]=7.0715315688392578e+01;
		/*
		 * ---------------------------------------------------------------------
		 * reference value of surface integral, for the (64X64X64) grid,
		 * after 250 time steps, with DT = 2.0d+00
		 * ---------------------------------------------------------------------
		 */
		xciref=2.6030925604886277e+01;
	}else if((nx==102)&&(ny==102)&&(nz==102)&&(itmax==250)){
		*class_npb='B';
		dtref=2.0e+0;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of residual, for the (102X102X102) grid,
		 * after 250 time steps, with DT = 2.0d+00
		 * ---------------------------------------------------------------------
		 */
		xcrref[0]=3.5532672969982736e+03;
		xcrref[1]=2.6214750795310692e+02;
		xcrref[2]=8.8333721850952190e+02;
		xcrref[3]=7.7812774739425265e+02;
		xcrref[4]=7.3087969592545314e+03;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of solution error, for the (102X102X102) 
		 * grid, after 250 time steps, with DT = 2.0d+00
		 * ---------------------------------------------------------------------
		 */
		xceref[0]=1.1401176380212709e+02;
		xceref[1]=8.1098963655421574e+00;
		xceref[2]=2.8480597317698308e+01;
		xceref[3]=2.5905394567832939e+01;
		xceref[4]=2.6054907504857413e+02;
		/*
		   c---------------------------------------------------------------------
		 * reference value of surface integral, for the (102X102X102) grid,
		 * after 250 time steps, with DT = 2.0d+00
		 * ---------------------------------------------------------------------
		 */
		xciref=4.7887162703308227e+01;
	}else if((nx==162)&&(ny==162)&&(nz==162)&&(itmax==250)){
		*class_npb='C';
		dtref=2.0e+0;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of residual, for the (162X162X162) grid,
		 * after 250 time steps, with DT = 2.0d+00
		 * ---------------------------------------------------------------------
		 */
		xcrref[0]=1.03766980323537846e+04;
		xcrref[1]=8.92212458801008552e+02;
		xcrref[2]=2.56238814582660871e+03;
		xcrref[3]=2.19194343857831427e+03;
		xcrref[4]=1.78078057261061185e+04;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of solution error, for the (162X162X162) 
		 * grid, after 250 time steps, with DT = 2.0d+00
		 * ---------------------------------------------------------------------
		 */
		xceref[0]=2.15986399716949279e+02;
		xceref[1]=1.55789559239863600e+01;
		xceref[2]=5.41318863077207766e+01;
		xceref[3]=4.82262643154045421e+01;
		xceref[4]=4.55902910043250358e+02;
		/*
		 * ---------------------------------------------------------------------
		 * reference value of surface integral, for the (162X162X162) grid,
		 * after 250 time steps, with DT = 2.0d+00
		 * ---------------------------------------------------------------------
		 */
		xciref=6.66404553572181300e+01;
		/*
		 * ---------------------------------------------------------------------
		 * reference value of surface integral, for the (162X162X162) grid,
		 * after 250 time steps, with DT = 2.0d+00
		 * ---------------------------------------------------------------------
		 */
		xciref=6.66404553572181300e+01;
	}else if((nx==408)&&(ny==408)&&(nz==408)&&(itmax== 300)){
		*class_npb='D';
		dtref=1.0e+0;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of residual, for the (408X408X408) grid,
		 * after 300 time steps, with DT = 1.0d+00
		 * ---------------------------------------------------------------------
		 */
		xcrref[0]=0.4868417937025e+05;
		xcrref[1]=0.4696371050071e+04;
		xcrref[2]=0.1218114549776e+05;
		xcrref[3]=0.1033801493461e+05;
		xcrref[4]=0.7142398413817e+05;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of solution error, for the (408X408X408) 
		 * grid, after 300 time steps, with DT = 1.0d+00
		 * ---------------------------------------------------------------------
		 */
		xceref[0]=0.3752393004482e+03;
		xceref[1]=0.3084128893659e+02;
		xceref[2]=0.9434276905469e+02;
		xceref[3]=0.8230686681928e+02;
		xceref[4]=0.7002620636210e+03;
		/*
		 * ---------------------------------------------------------------------
		 * reference value of surface integral, for the (408X408X408) grid,
		 * after 300 time steps, with DT = 1.0d+00
		 * ---------------------------------------------------------------------
		 */
		xciref=0.8334101392503e+02;
	}else if((nx==1020)&&(ny==1020)&&(nz==1020)&&(itmax==300)){
		*class_npb='E';
		dtref=0.5e+0;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of residual, for the (1020X1020X1020) grid,
		 * after 300 time steps, with DT = 0.5d+00
		 * ---------------------------------------------------------------------
		 */
		xcrref[0]=0.2099641687874e+06;
		xcrref[1]=0.2130403143165e+05;
		xcrref[2]=0.5319228789371e+05;
		xcrref[3]=0.4509761639833e+05;
		xcrref[4]=0.2932360006590e+06;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of solution error, for the (1020X1020X1020) 
		 * grid, after 300 time steps, with DT = 0.5d+00
		 * ---------------------------------------------------------------------
		 */
		xceref[0]=0.4800572578333e+03;
		xceref[1]=0.4221993400184e+02;
		xceref[2]=0.1210851906824e+03;
		xceref[3]=0.1047888986770e+03;
		xceref[4]=0.8363028257389e+03;
		/*
		 * ---------------------------------------------------------------------
		 * reference value of surface integral, for the (1020X1020X1020) grid,
		 * after 300 time steps, with DT = 0.5d+00
		 * ---------------------------------------------------------------------
		 */
		xciref=0.9512163272273e+02;
	}else{
		*verified=FALSE;
	}
	/*
	 * ---------------------------------------------------------------------
	 * verification test for residuals if gridsize is one of 
	 * the defined grid sizes above (class .ne. 'U')
	 * ---------------------------------------------------------------------
	 * compute the difference of solution values and the known reference values.
	 * ---------------------------------------------------------------------
	 */
	for(m=0; m<5; m++){
		xcrdif[m]=fabs((xcr[m]-xcrref[m])/xcrref[m]);
		xcedif[m]=fabs((xce[m]-xceref[m])/xceref[m]);
	}
	xcidif=fabs((xci-xciref)/xciref);
	/*
	 * ---------------------------------------------------------------------
	 * output the comparison of computed results to known cases.
	 * ---------------------------------------------------------------------
	 */
	if(*class_npb!='U'){
		printf("\n Verification being performed for class_npb %c\n",*class_npb);
		printf(" Accuracy setting for epsilon = %20.13E\n",epsilon);
		*verified=(fabs(dt-dtref)<=epsilon);
		if(!(*verified)){
			*class_npb='U';
			printf(" DT does not match the reference value of %15.8E\n",dtref);
		}
	}else{ 
		printf(" Unknown class_npb\n");
	}
	if(*class_npb!='U'){
		printf(" Comparison of RMS-norms of residual\n");
	}else{
		printf(" RMS-norms of residual\n");
	}
	for(m=0; m<5; m++){
		if(*class_npb=='U'){
			printf("          %2d  %20.13E\n",m+1,xcr[m]);
		}else if(xcrdif[m]<=epsilon){
			printf("          %2d  %20.13E%20.13E%20.13E\n",m+1,xcr[m],xcrref[m],xcrdif[m]);
		}else{ 
			*verified=FALSE;
			printf(" FAILURE: %2d  %20.13E%20.13E%20.13E\n",m+1,xcr[m],xcrref[m],xcrdif[m]);
		}
	}
	if(*class_npb!='U'){
		printf(" Comparison of RMS-norms of solution error\n");
	}else{
		printf(" RMS-norms of solution error\n");
	}
	for(m=0; m<5; m++){
		if(*class_npb=='U'){
			printf("          %2d  %20.13E\n",m+1,xce[m]);
		}else if(xcedif[m]<=epsilon){
			printf("          %2d  %20.13E%20.13E%20.13E\n",m+1,xce[m],xceref[m],xcedif[m]);
		}else{
			*verified=FALSE;
			printf(" FAILURE: %2d  %20.13E%20.13E%20.13E\n",m+1,xce[m],xceref[m],xcedif[m]);
		}
	}
	if(*class_npb!='U'){
		printf(" Comparison of surface integral\n");
	}else{
		printf(" Surface integral\n");
	}
	if(*class_npb=='U'){
		printf("              %20.13E\n",xci);
	}else if(xcidif<=epsilon){
		printf("              %20.13E%20.13E%20.13E\n",xci,xciref,xcidif);
	}else{
		*verified=FALSE;
		printf(" FAILURE:     %20.13E%20.13E%20.13E\n",xci,xciref,xcidif);
	}
	if(*class_npb=='U'){
		printf(" No reference values provided\n");
		printf("No verification performed\n");
	}else if(*verified){
		printf(" Verification Successful\n");
	}else{
		printf(" Verification failed\n");
	}
}
