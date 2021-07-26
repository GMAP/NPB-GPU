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
 *      E. Barszcz
 *      P. Frederickson
 *      A. Woo
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
 * The CUDA version is a parallel implementation of the serial C++ version
 * CUDA version: https://github.com/GMAP/NPB-GPU/tree/master/CUDA
 *
 * Authors of the CUDA code:
 *      Gabriell Araujo <hexenoften@gmail.com>
 *
 * ------------------------------------------------------------------------------
 */

#include <cuda.h>
#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

#define NM (2+(1<<LM)) /* actual dimension including ghost cells for communications */
#define NV (ONE*(2+(1<<NDIM1))*(2+(1<<NDIM2))*(2+(1<<NDIM3))) /* size of rhs array */
#define NR (((NV+NM*NM+5*NM+7*LM+6)/7)*8) /* size of residual array */
#define MAXLEVEL (LT_DEFAULT+1) /* maximum number of levels */
#define M (NM+1) /* set at m=1024, can handle cases up to 1024^3 case */
#define MM (10)
#define	A (pow(5.0,13.0))
#define	X (314159265.0)
#define PROFILING_TOTAL_TIME (0)
#define PROFILING_COMM3 (1)
#define PROFILING_INTERP (2)
#define PROFILING_NORM2U3 (3)
#define PROFILING_PSINV (4)
#define PROFILING_RESID (5)
#define PROFILING_RPRJ3 (6)
#define PROFILING_ZERO3 (7)

/* global variables */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static int nx[MAXLEVEL+1];
static int ny[MAXLEVEL+1];
static int nz[MAXLEVEL+1];
static int m1[MAXLEVEL+1];
static int m2[MAXLEVEL+1];
static int m3[MAXLEVEL+1];
static int ir[MAXLEVEL+1];
static int debug_vec[8];
static double u[NR];
static double v[NV];
static double r[NR];
#else
static int (*nx)=(int*)malloc(sizeof(int)*(MAXLEVEL+1));
static int (*ny)=(int*)malloc(sizeof(int)*(MAXLEVEL+1));
static int (*nz)=(int*)malloc(sizeof(int)*(MAXLEVEL+1));
static int (*m1)=(int*)malloc(sizeof(int)*(MAXLEVEL+1));
static int (*m2)=(int*)malloc(sizeof(int)*(MAXLEVEL+1));
static int (*m3)=(int*)malloc(sizeof(int)*(MAXLEVEL+1));
static int (*ir)=(int*)malloc(sizeof(int)*(MAXLEVEL+1));
static int (*debug_vec)=(int*)malloc(sizeof(int)*(8));
static double (*u)=(double*)malloc(sizeof(double)*(NR));
static double (*v)=(double*)malloc(sizeof(double)*(NV));
static double (*r)=(double*)malloc(sizeof(double)*(NR));
#endif
static int is1, is2, is3, ie1, ie2, ie3, lt, lb;
/* gpu variables */
size_t size_a_device;
size_t size_c_device;
size_t size_u_device;
size_t size_v_device;
size_t size_r_device;
double* a_device;
double* c_device;
double* u_device;
double* v_device;
double* r_device;
int threads_per_block_on_comm3;
int threads_per_block_on_interp;
int threads_per_block_on_norm2u3;
int threads_per_block_on_psinv;
int threads_per_block_on_resid;
int threads_per_block_on_rprj3;
int threads_per_block_on_zero3;
size_t size_shared_data_on_interp;
size_t size_shared_data_on_norm2u3;
size_t size_shared_data_on_psinv;
size_t size_shared_data_on_resid;
size_t size_shared_data_on_rprj3;
int gpu_device_id;
int total_devices;
cudaDeviceProp gpu_device_properties;
extern __shared__ double extern_share_data[];

/* function prototypes */
static void bubble(double ten[][MM], 
		int j1[][MM], 
		int j2[][MM], 
		int j3[][MM], 
		int m, 
		int ind);
static void comm3(void* pointer_u, 
		int n1, 
		int n2, 
		int n3, 
		int kk);
static void comm3_gpu(double* u_device, 
		int n1, 
		int n2, 
		int n3, 
		int kk);
__global__ void comm3_gpu_kernel_1(double* u, 
		int n1, 
		int n2, 
		int n3, 
		int amount_of_work);
__global__ void comm3_gpu_kernel_2(double* u,
		int n1,
		int n2,
		int n3,
		int amount_of_work);
__global__ void comm3_gpu_kernel_3(double* u,
		int n1, 
		int n2, 
		int n3, 
		int amount_of_work);
static void interp(void* pointer_z, 
		int mm1, 
		int mm2, 
		int mm3, 
		void* pointer_u, 
		int n1, 
		int n2, int n3, 
		int k);
static void interp_gpu(double* z_device, 
		int mm1, 
		int mm2, 
		int mm3, 
		double* u_device, 
		int n1, 
		int n2, 
		int n3, 
		int k);
__global__ void interp_gpu_kernel(double* z_device,
		double* u_device,
		int mm1, 
		int mm2, 
		int mm3,
		int n1, 
		int n2, 
		int n3,
		int amount_of_work);
static void mg3P(double u[], 
		double v[], 
		double r[], 
		double a[4], 
		double c[4], 
		int n1, 
		int n2, 
		int n3, 
		int k);
static void mg3P_gpu(double* u_device, 
		double* v_device, 
		double* r_device, 
		double a[4], 
		double c[4], 
		int n1, 
		int n2, 
		int n3, 
		int k);
static void norm2u3(void* pointer_r, 
		int n1, 
		int n2, 
		int n3, 
		double* rnm2, 
		double* rnmu, 
		int nx, 
		int ny, 
		int nz);
static void norm2u3_gpu(double* r_device, 
		int n1, 
		int n2, 
		int n3, 
		double* rnm2, 
		double* rnmu, 
		int nx, 
		int ny, 
		int nz);
__global__ void norm2u3_gpu_kernel(double* r,
		const int n1, 
		const int n2, 
		const int n3,
		double* res_sum,
		double* res_max,
		int number_of_blocks,
		int amount_of_work);
static double power(double a, 
		int n);
static void psinv(void* pointer_r, 
		void* pointer_u, 
		int n1, 
		int n2, 
		int n3, 
		double c[4], 
		int k);
static void psinv_gpu(double* r_device, 
		double* u_device, 
		int n1, 
		int n2, 
		int n3, 
		double* c_device, 
		int k);
__global__ void psinv_gpu_kernel(double* r,
		double* u,
		double* c,
		int n1,
		int n2,
		int n3,
		int amount_of_work);
static void release_gpu();
static void rep_nrm(void* pointer_u, 
		int n1, 
		int n2, 
		int n3, 
		char* title, 
		int kk);
static void resid(void* pointer_u, 
		void* pointer_v, 
		void* pointer_r, 
		int n1, 
		int n2, 
		int n3, 
		double a[4], 
		int k);
static void resid_gpu(double* u_device,
		double* v_device,
		double* r_device,
		int n1,
		int n2,
		int n3,
		double* a_device,
		int k);
__global__ void resid_gpu_kernel(double* r,
		double* u,
		double* v,
		double* a,
		int n1,
		int n2,
		int n3,
		int amount_of_work);
static void rprj3(void* pointer_r, 
		int m1k, 
		int m2k, 
		int m3k, 
		void* pointer_s, 
		int m1j, 
		int m2j, 
		int m3j, 
		int k);
static void rprj3_gpu(double* r_device, 
		int m1k, 
		int m2k, 
		int m3k, 
		double* s_device, 
		int m1j, 
		int m2j, 
		int m3j, 
		int k);
__global__ void rprj3_gpu_kernel(double* r_device,
		double* s_device,
		int m1k,
		int m2k,
		int m3k,
		int m1j,
		int m2j,
		int m3j,
		int d1, 
		int d2, 
		int d3,
		int amount_of_work);
static void setup(int* n1, 
		int* n2, 
		int* n3, 
		int k);
static void setup_gpu(double* a, 
		double* c);
static void showall(void* pointer_z, 
		int n1, 
		int n2, 
		int n3);
static void zero3_gpu(double* z_device, 
		int n1, 
		int n2, 
		int n3);
__global__ void zero3_gpu_kernel(double* z, 
		int n1, 
		int n2, 
		int n3, 
		int amount_of_work);
static void zero3(void* pointer_z, 
		int n1, 
		int n2, 
		int n3);
static void zran3(void* pointer_z, 
		int n1, 
		int n2, 
		int n3, 
		int nx, 
		int ny, 
		int k);

/* mg */
int main(int argc, char** argv){
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
	printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif
#if defined(PROFILING)
	printf(" PROFILING mode on\n");
#endif
	/*
	 * -------------------------------------------------------------------------
	 * k is the current level. it is passed down through subroutine args
	 * and is not global. it is the current iteration
	 * -------------------------------------------------------------------------
	 */
	int k, it;
	double t, mflops;

	double a[4], c[4];

	double rnm2, rnmu, epsilon;
	int n1, n2, n3, nit;
	double nn, verify_value, err;
	boolean verified;
	char class_npb;

	int i;

	/*
	 * ----------------------------------------------------------------------
	 * read in and broadcast input data
	 * ----------------------------------------------------------------------
	 */
	lt = LT_DEFAULT;
	nit = NIT_DEFAULT;
	nx[lt] = NX_DEFAULT;
	ny[lt] = NY_DEFAULT;
	nz[lt] = NZ_DEFAULT;
	for(i = 0; i <= 7; i++){
		debug_vec[i] = DEBUG_DEFAULT;
	}
	if((nx[lt] != ny[lt]) || (nx[lt] != nz[lt])){
		class_npb = 'U';
	}else if(nx[lt] == 32 && nit == 4){
		class_npb = 'S';
	}else if(nx[lt] == 128 && nit == 4){
		class_npb = 'W';
	}else if(nx[lt] == 256 && nit == 4){
		class_npb = 'A';
	}else if(nx[lt] == 256 && nit == 20){
		class_npb = 'B';
	}else if(nx[lt] == 512 && nit == 20){
		class_npb = 'C';
	}else if(nx[lt] == 1024 && nit == 50){  
		class_npb = 'D';
	}else if(nx[lt] == 2048 && nit == 50){  
		class_npb = 'E';	
	}else{
		class_npb = 'U';
	}

	/*
	 * ---------------------------------------------------------------------
	 * use these for debug info:
	 * ---------------------------------------------------------------------
	 * debug_vec(0) = 1 !=> report all norms
	 * debug_vec(1) = 1 !=> some setup information
	 * debug_vec(1) = 2 !=> more setup information
	 * debug_vec(2) = k => at level k or below, show result of resid
	 * debug_vec(3) = k => at level k or below, show result of psinv
	 * debug_vec(4) = k => at level k or below, show result of rprj
	 * debug_vec(5) = k => at level k or below, show result of interp
	 * debug_vec(6) = 1 => (unused)
	 * debug_vec(7) = 1 => (unused)
	 * ---------------------------------------------------------------------
	 */
	a[0] = -8.0/3.0;
	a[1] =  0.0;
	a[2] =  1.0/6.0;
	a[3] =  1.0/12.0;

	if(class_npb == 'A' || class_npb == 'S' || class_npb =='W'){
		/* coefficients for the s(a) smoother */
		c[0] =  -3.0/8.0;
		c[1] =  +1.0/32.0;
		c[2] =  -1.0/64.0;
		c[3] =   0.0;
	}else{
		/* coefficients for the s(b) smoother */
		c[0] =  -3.0/17.0;
		c[1] =  +1.0/33.0;
		c[2] =  -1.0/61.0;
		c[3] =   0.0;
	}

	lb = 1;
	k = lt;

	setup(&n1,&n2,&n3,k);

	zero3(u,n1,n2,n3);
	zran3(v,n1,n2,n3,nx[lt],ny[lt],k);

	norm2u3(v,n1,n2,n3,&rnm2,&rnmu,nx[lt],ny[lt],nz[lt]);	

	printf("\n\n NAS Parallel Benchmarks 4.1 CUDA C++ version - MG Benchmark\n\n");
	printf(" Size: %3dx%3dx%3d (class_npb %1c)\n", nx[lt], ny[lt], nz[lt], class_npb);
	printf(" Iterations: %3d\n", nit);

	resid(u,v,r,n1,n2,n3,a,k);
	norm2u3(r,n1,n2,n3,&rnm2,&rnmu,nx[lt],ny[lt],nz[lt]);

	/*
	 * ---------------------------------------------------------------------
	 * one iteration for startup
	 * ---------------------------------------------------------------------
	 */
	mg3P(u,v,r,a,c,n1,n2,n3,k);
	resid(u,v,r,n1,n2,n3,a,k);

	setup(&n1,&n2,&n3,k);

	zero3(u,n1,n2,n3);
	zran3(v,n1,n2,n3,nx[lt],ny[lt],k);

	setup_gpu(a,c);

	timer_clear(PROFILING_TOTAL_TIME);
#if defined(PROFILING)
	timer_clear(PROFILING_COMM3);
	timer_clear(PROFILING_INTERP);
	timer_clear(PROFILING_NORM2U3);
	timer_clear(PROFILING_PSINV);
	timer_clear(PROFILING_RESID);
	timer_clear(PROFILING_RPRJ3);
	timer_clear(PROFILING_ZERO3);
#endif

	timer_start(PROFILING_TOTAL_TIME);

	resid_gpu(u_device,v_device,r_device,n1,n2,n3,a_device,k);	
	norm2u3_gpu(r_device,n1,n2,n3,&rnm2,&rnmu,nx[lt],ny[lt],nz[lt]);
	for(it = 1; it <= nit; it++){
		mg3P_gpu(u_device,v_device,r_device,a_device,c_device,n1,n2,n3,k);
		resid_gpu(u_device,v_device,r_device,n1,n2,n3,a_device,k);
	}
	norm2u3_gpu(r_device,n1,n2,n3,&rnm2,&rnmu,nx[lt],ny[lt],nz[lt]);

	timer_stop(PROFILING_TOTAL_TIME);
	t = timer_read(PROFILING_TOTAL_TIME);  	

	verified = FALSE;
	verify_value = 0.0;	

	printf(" Benchmark completed\n");

	epsilon = 1.0e-8;
	if(class_npb != 'U'){
		if(class_npb == 'S'){
			verify_value = 0.5307707005734e-04;
		}else if(class_npb == 'W'){
			verify_value = 0.6467329375339e-05;
		}else if(class_npb == 'A'){
			verify_value = 0.2433365309069e-05;
		}else if(class_npb == 'B'){
			verify_value = 0.1800564401355e-05;
		}else if(class_npb == 'C'){
			verify_value = 0.5706732285740e-06;
		}else if(class_npb == 'D'){
			verify_value = 0.1583275060440e-09;
		}else if(class_npb == 'E'){
			verify_value = 0.8157592357404e-10; 
		}
		err = fabs(rnm2-verify_value) / verify_value;
		if(err <= epsilon){
			verified = TRUE;
			printf(" VERIFICATION SUCCESSFUL\n");
			printf(" L2 Norm is %20.13e\n", rnm2);
			printf(" Error is   %20.13e\n", err);
		}else{
			verified = FALSE;
			printf(" VERIFICATION FAILED\n");
			printf(" L2 Norm is             %20.13e\n", rnm2);
			printf(" The correct L2 Norm is %20.13e\n", verify_value);
		}
	}else{
		verified = FALSE;
		printf(" Problem size unknown\n");
		printf(" NO VERIFICATION PERFORMED\n");
	}

	nn = 1.0*nx[lt]*ny[lt]*nz[lt];

	if(t!=0.0){
		mflops = 58.0*nit*nn*1.0e-6/t;
	}else{
		mflops = 0.0;
	}

	char gpu_config[256];
	char gpu_config_string[2048];
#if defined(PROFILING)
	sprintf(gpu_config, "%5s\t%25s\t%25s\t%25s\n", "GPU Kernel", "Threads Per Block", "Time in Seconds", "Time in Percentage");
	strcpy(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " comm3", threads_per_block_on_comm3, timer_read(PROFILING_COMM3), (timer_read(PROFILING_COMM3)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " interp", threads_per_block_on_interp, timer_read(PROFILING_INTERP), (timer_read(PROFILING_INTERP)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " norm2u3", threads_per_block_on_norm2u3, timer_read(PROFILING_NORM2U3), (timer_read(PROFILING_NORM2U3)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " psinv", threads_per_block_on_psinv, timer_read(PROFILING_PSINV), (timer_read(PROFILING_PSINV)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " resid", threads_per_block_on_resid, timer_read(PROFILING_RESID), (timer_read(PROFILING_RESID)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " rprj3", threads_per_block_on_rprj3, timer_read(PROFILING_RPRJ3), (timer_read(PROFILING_RPRJ3)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " zero3", threads_per_block_on_zero3, timer_read(PROFILING_ZERO3), (timer_read(PROFILING_ZERO3)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
#else
	sprintf(gpu_config, "%5s\t%25s\n", "GPU Kernel", "Threads Per Block");
	strcpy(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " comm3", threads_per_block_on_comm3);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " interp", threads_per_block_on_interp);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " norm2u3", threads_per_block_on_norm2u3);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " psinv", threads_per_block_on_psinv);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " resid", threads_per_block_on_resid);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " rprj3", threads_per_block_on_rprj3);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " zero3", threads_per_block_on_zero3);
	strcat(gpu_config_string, gpu_config);
#endif

	c_print_results((char*)"MG",
			class_npb,
			nx[lt],
			ny[lt],
			nz[lt],
			nit,
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
 * bubble does a bubble sort in direction dir
 * ---------------------------------------------------------------------
 */
static void bubble(double ten[][MM], 
		int j1[][MM], 
		int j2[][MM], 
		int j3[][MM], 
		int m, 
		int ind){
	double temp;
	int i, j_temp;

	if(ind == 1){
		for(i = 0; i < m-1; i++){
			if(ten[ind][i] > ten[ind][i+1]){
				temp = ten[ind][i+1];
				ten[ind][i+1] = ten[ind][i];
				ten[ind][i] = temp;

				j_temp = j1[ind][i+1];
				j1[ind][i+1] = j1[ind][i];
				j1[ind][i] = j_temp;

				j_temp = j2[ind][i+1];
				j2[ind][i+1] = j2[ind][i];
				j2[ind][i] = j_temp;

				j_temp = j3[ind][i+1];
				j3[ind][i+1] = j3[ind][i];
				j3[ind][i] = j_temp;
			}else{
				return;
			}
		}
	}else{
		for(i = 0; i < m-1; i++){
			if(ten[ind][i] < ten[ind][i+1]){
				temp = ten[ind][i+1];
				ten[ind][i+1] = ten[ind][i];
				ten[ind][i] = temp;

				j_temp = j1[ind][i+1];
				j1[ind][i+1] = j1[ind][i];
				j1[ind][i] = j_temp;

				j_temp = j2[ind][i+1];
				j2[ind][i+1] = j2[ind][i];
				j2[ind][i] = j_temp;

				j_temp = j3[ind][i+1];
				j3[ind][i+1] = j3[ind][i];
				j3[ind][i] = j_temp;
			}else{
				return;
			}
		}
	}
}

/*
 * ---------------------------------------------------------------------
 * comm3 organizes the communication on all borders 
 * ---------------------------------------------------------------------
 */
static void comm3(void* pointer_u, 
		int n1, 
		int n2, 
		int n3, 
		int kk){
	double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;

	int i1, i2, i3;
	/* axis = 1 */
	for(i3 = 1; i3 < n3-1; i3++){
		for(i2 = 1; i2 < n2-1; i2++){
			u[i3][i2][0] = u[i3][i2][n1-2];
			u[i3][i2][n1-1] = u[i3][i2][1];			
		}
	}
	/* axis = 2 */
	for(i3 = 1; i3 < n3-1; i3++){
		for(i1 = 0; i1 < n1; i1++){
			u[i3][0][i1] = u[i3][n2-2][i1];
			u[i3][n2-1][i1] = u[i3][1][i1];			
		}
	}
	/* axis = 3 */
	for(i2 = 0; i2 < n2; i2++){
		for(i1 = 0; i1 < n1; i1++){
			u[0][i2][i1] = u[n3-2][i2][i1];
			u[n3-1][i2][i1] = u[1][i2][i1];			
		}
	}
}

static void comm3_gpu(double* u_device, 
		int n1, 
		int n2, 
		int n3, 
		int kk){
#if defined(PROFILING)
	timer_start(PROFILING_COMM3);
#endif

	/* axis = 1 */
	int threads_per_block_x=threads_per_block_on_comm3;
	int threads_per_block_y=1;
	int amount_of_work_x=(ceil(double(n2-2)/double(threads_per_block_x)))*threads_per_block_x;
	int amount_of_work_y=n3-2;
	dim3 threadsPerBlock(threads_per_block_x,
			threads_per_block_y,
			1);
	dim3 blocksPerGrid(ceil(double(amount_of_work_x)/double(threadsPerBlock.x)),
			ceil(double(amount_of_work_y)/double(threadsPerBlock.y)),
			1);
	comm3_gpu_kernel_1<<<blocksPerGrid,
		threadsPerBlock>>>(u_device,
				n1,
				n2,
				n3,
				amount_of_work_x);

	/* axis = 2 */
	threads_per_block_x=threads_per_block_on_comm3;
	threads_per_block_y=1;
	amount_of_work_x=(ceil(double(n1)/double(threads_per_block_x)))*threads_per_block_x;
	amount_of_work_y=n3-2;
	threadsPerBlock.x=threads_per_block_x;
	threadsPerBlock.y=threads_per_block_y;
	blocksPerGrid.x=ceil(double(amount_of_work_x)/double(threadsPerBlock.x));
	blocksPerGrid.y=ceil(double(amount_of_work_y)/double(threadsPerBlock.y));
	comm3_gpu_kernel_2<<<blocksPerGrid,	
		threadsPerBlock>>>(u_device,
				n1,
				n2,
				n3,
				amount_of_work_x);

	/* axis = 3 */
	threads_per_block_x=threads_per_block_on_comm3;
	threads_per_block_y=1;
	amount_of_work_x=(ceil(double(n1)/double(threads_per_block_x)))*threads_per_block_x;
	amount_of_work_y=n2;
	threadsPerBlock.x=threads_per_block_x;
	threadsPerBlock.y=threads_per_block_y;
	blocksPerGrid.x=ceil(double(amount_of_work_x) / double(threadsPerBlock.x));
	blocksPerGrid.y=ceil(double(amount_of_work_y) / double(threadsPerBlock.y));
	comm3_gpu_kernel_3<<<blocksPerGrid,
		threadsPerBlock>>>(u_device,
				n1,
				n2,
				n3,
				amount_of_work_x);

#if defined(PROFILING)
	timer_stop(PROFILING_COMM3);
#endif
}

__global__ void comm3_gpu_kernel_1(double* u, 
		int n1, 
		int n2, 
		int n3, 
		int amount_of_work){
	int i3=blockIdx.y*blockDim.y+threadIdx.y+1;
	int i2=blockIdx.x*blockDim.x+threadIdx.x+1;

	if(i2>=n2-1){return;}

	u[i3*n2*n1+i2*n1+0]=u[i3*n2*n1+i2*n1+n1-2];
	u[i3*n2*n1+i2*n1+n1-1]=u[i3*n2*n1+i2*n1+1];
}

__global__ void comm3_gpu_kernel_2(double* u,
		int n1,
		int n2,
		int n3,
		int amount_of_work){
	int i3=blockIdx.y*blockDim.y+threadIdx.y+1;
	int i1=blockIdx.x*blockDim.x+threadIdx.x;

	if(i1>=n1){return;}

	u[i3*n2*n1+0*n1+i1]=u[i3*n2*n1+(n2-2)*n1+i1];
	u[i3*n2*n1+(n2-1)*n1+i1]=u[i3*n2*n1+1*n1+i1];
}

__global__ void comm3_gpu_kernel_3(double* u,
		int n1, 
		int n2, 
		int n3, 
		int amount_of_work){
	int i2=blockIdx.y*blockDim.y+threadIdx.y;
	int i1=blockIdx.x*blockDim.x+threadIdx.x;

	if(i1>=n1){return;}

	u[0*n2*n1+i2*n1+i1]=u[(n3-2)*n2*n1+i2*n1+i1];
	u[(n3-1)*n2*n1+i2*n1+i1]=u[1*n2*n1+i2*n1+i1];
}

/*
 * --------------------------------------------------------------------
 * interp adds the trilinear interpolation of the correction
 * from the coarser grid to the current approximation: u = u + Qu'
 *     
 * observe that this  implementation costs  16A + 4M, where
 * A and M denote the costs of addition and multiplication.  
 * note that this vectorizes, and is also fine for cache 
 * based machines. vector machines may get slightly better 
 * performance however, with 8 separate "do i1" loops, rather than 4.
 * --------------------------------------------------------------------
 */
static void interp(void* pointer_z, 
		int mm1, 
		int mm2, 
		int mm3, 
		void* pointer_u, 
		int n1, 
		int n2, 
		int n3, 
		int k){
	double (*z)[mm2][mm1] = (double (*)[mm2][mm1])pointer_z;
	double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;	

	int i3, i2, i1, d1, d2, d3, t1, t2, t3;

	/* 
	 * --------------------------------------------------------------------
	 * note that m = 1037 in globals.h but for this only need to be
	 * 535 to handle up to 1024^3
	 * integer m
	 * parameter( m=535 )
	 * --------------------------------------------------------------------
	 */
	double z1[M], z2[M], z3[M];

	if(n1 != 3 && n2 != 3 && n3 != 3){
		for(i3 = 0; i3 < mm3-1; i3++){
			for(i2 = 0; i2 < mm2-1; i2++){
				for(i1 = 0; i1 < mm1; i1++){
					z1[i1] = z[i3][i2+1][i1] + z[i3][i2][i1];
					z2[i1] = z[i3+1][i2][i1] + z[i3][i2][i1];
					z3[i1] = z[i3+1][i2+1][i1] + z[i3+1][i2][i1] + z1[i1];
				}
				for(i1 = 0; i1 < mm1-1; i1++){
					u[2*i3][2*i2][2*i1] = u[2*i3][2*i2][2*i1]
						+z[i3][i2][i1];
					u[2*i3][2*i2][2*i1+1] = u[2*i3][2*i2][2*i1+1]
						+0.5*(z[i3][i2][i1+1]+z[i3][i2][i1]);
				}
				for(i1 = 0; i1 < mm1-1; i1++){
					u[2*i3][2*i2+1][2*i1] = u[2*i3][2*i2+1][2*i1]
						+0.5 * z1[i1];
					u[2*i3][2*i2+1][2*i1+1] = u[2*i3][2*i2+1][2*i1+1]
						+0.25*( z1[i1] + z1[i1+1] );
				}
				for(i1 = 0; i1 < mm1-1; i1++){
					u[2*i3+1][2*i2][2*i1] = u[2*i3+1][2*i2][2*i1]
						+0.5 * z2[i1];
					u[2*i3+1][2*i2][2*i1+1] = u[2*i3+1][2*i2][2*i1+1]
						+0.25*( z2[i1] + z2[i1+1] );
				}
				for(i1 = 0; i1 < mm1-1; i1++){
					u[2*i3+1][2*i2+1][2*i1] = u[2*i3+1][2*i2+1][2*i1]
						+0.25* z3[i1];
					u[2*i3+1][2*i2+1][2*i1+1] = u[2*i3+1][2*i2+1][2*i1+1]
						+0.125*( z3[i1] + z3[i1+1] );
				}
			}
		}
	}else{
		if(n1 == 3){
			d1 = 2;
			t1 = 1;
		}else{
			d1 = 1;
			t1 = 0;
		}      
		if(n2 == 3){
			d2 = 2;
			t2 = 1;
		}else{
			d2 = 1;
			t2 = 0;
		}          
		if(n3 == 3){
			d3 = 2;
			t3 = 1;
		}else{
			d3 = 1;
			t3 = 0;
		}
		for(i3 = d3; i3 <= mm3-1; i3++){
			for(i2 = d2; i2 <= mm2-1; i2++){
				for(i1 = d1; i1 <= mm1-1; i1++){
					u[2*i3-d3-1][2*i2-d2-1][2*i1-d1-1] =
						u[2*i3-d3-1][2*i2-d2-1][2*i1-d1-1]
						+z[i3-1][i2-1][i1-1];
				}
				for(i1 = 1; i1 <= mm1-1; i1++){
					u[2*i3-d3-1][2*i2-d2-1][2*i1-t1-1] =
						u[2*i3-d3-1][2*i2-d2-1][2*i1-t1-1]
						+0.5*(z[i3-1][i2-1][i1]+z[i3-1][i2-1][i1-1]);
				}
			}
			for(i2 = 1; i2 <= mm2-1; i2++){
				for ( i1 = d1; i1 <= mm1-1; i1++) {
					u[2*i3-d3-1][2*i2-t2-1][2*i1-d1-1] =
						u[2*i3-d3-1][2*i2-t2-1][2*i1-d1-1]
						+0.5*(z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1]);
				}
				for(i1 = 1; i1 <= mm1-1; i1++){
					u[2*i3-d3-1][2*i2-t2-1][2*i1-t1-1] =
						u[2*i3-d3-1][2*i2-t2-1][2*i1-t1-1]
						+0.25*(z[i3-1][i2][i1]+z[i3-1][i2-1][i1]
								+z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1]);
				}
			}
		}
		for(i3 = 1; i3 <= mm3-1; i3++){
			for(i2 = d2; i2 <= mm2-1; i2++){
				for(i1 = d1; i1 <= mm1-1; i1++){
					u[2*i3-t3-1][2*i2-d2-1][2*i1-d1-1] =
						u[2*i3-t3-1][2*i2-d2-1][2*i1-d1-1]
						+0.5*(z[i3][i2-1][i1-1]+z[i3-1][i2-1][i1-1]);
				}
				for(i1 = 1; i1 <= mm1-1; i1++){
					u[2*i3-t3-1][2*i2-d2-1][2*i1-t1-1] =
						u[2*i3-t3-1][2*i2-d2-1][2*i1-t1-1]
						+0.25*(z[i3][i2-1][i1]+z[i3][i2-1][i1-1]
								+z[i3-1][i2-1][i1]+z[i3-1][i2-1][i1-1]);
				}
			}
			for(i2 = 1; i2 <= mm2-1; i2++){
				for (i1 = d1; i1 <= mm1-1; i1++){
					u[2*i3-t3-1][2*i2-t2-1][2*i1-d1-1] =
						u[2*i3-t3-1][2*i2-t2-1][2*i1-d1-1]
						+0.25*(z[i3][i2][i1-1]+z[i3][i2-1][i1-1]
								+z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1]);
				}
				for(i1 = 1; i1 <= mm1-1; i1++){
					u[2*i3-t3-1][2*i2-t2-1][2*i1-t1-1] =
						u[2*i3-t3-1][2*i2-t2-1][2*i1-t1-1]
						+0.125*(z[i3][i2][i1]+z[i3][i2-1][i1]
								+z[i3][i2][i1-1]+z[i3][i2-1][i1-1]
								+z[i3-1][i2][i1]+z[i3-1][i2-1][i1]
								+z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1]);
				}
			}
		}
	}

	if(debug_vec[0] >= 1){
		rep_nrm(z,mm1,mm2,mm3,(char*)"z: inter",k-1);
		rep_nrm(u,n1,n2,n3,(char*)"u: inter",k);
	}
	if(debug_vec[5] >= k){
		showall(z,mm1,mm2,mm3);
		showall(u,n1,n2,n3);
	}
}

static void interp_gpu(double* z_device, 
		int mm1, 
		int mm2, 
		int mm3, 
		double* u_device, 
		int n1, 
		int n2, 
		int n3, 
		int k){
#if defined(PROFILING)
	timer_start(PROFILING_INTERP);
#endif

	if(n1 != 3 && n2 != 3 && n3 != 3){
		int amount_of_work_x=(mm2-1)*mm1;
		int amount_of_work_y=mm3-1;
		int threads_per_block_x;
		int threads_per_block_y=1;
		if(threads_per_block_on_interp != mm1){
			threads_per_block_x = mm1;
		}
		else{
			threads_per_block_x = threads_per_block_on_interp;
		}
		dim3 threadsPerBlock(threads_per_block_x,
				threads_per_block_y,
				1);
		dim3 blocksPerGrid(ceil(double(amount_of_work_x)/double(threadsPerBlock.x)),
				ceil(double(amount_of_work_y)/double(threadsPerBlock.y)),
				1);
		interp_gpu_kernel<<<blocksPerGrid, 
			threadsPerBlock,
			size_shared_data_on_interp>>>(
					z_device,
					u_device,
					mm1,
					mm2,
					mm3,
					n1,
					n2,
					n3,
					amount_of_work_x);		
	}

#if defined(PROFILING)
	timer_stop(PROFILING_INTERP);
#endif
}

__global__ void interp_gpu_kernel(double* z_device,
		double* u_device,
		int mm1, 
		int mm2, 
		int mm3,
		int n1, 
		int n2, 
		int n3,
		int amount_of_work){
	int check=blockIdx.x*blockDim.x+threadIdx.x;
	if(check>=amount_of_work){return;}

	double* z1 = (double*)(extern_share_data);
	double* z2 = (double*)(z1+M);
	double* z3 = (double*)(z2+M);

	int i3=blockIdx.y*blockDim.y+threadIdx.y;
	int i2=blockIdx.x;
	int i1=threadIdx.x;

	z1[i1]=z_device[i3*mm2*mm1+(i2+1)*mm1+i1]+z_device[i3*mm2*mm1+i2*mm1+i1];
	z2[i1]=z_device[(i3+1)*mm2*mm1+i2*mm1+i1]+z_device[i3*mm2*mm1+i2*mm1+i1];
	z3[i1]=z_device[(i3+1)*mm2*mm1+(i2+1)*mm1+i1]+z_device[(i3+1)*mm2*mm1+i2*mm1+i1]+z1[i1];

	__syncthreads();
	if(i1<mm1-1){
		double z321=z_device[i3*mm2*mm1+i2*mm1+i1];
		u_device[2*i3*n2*n1+2*i2*n1+2*i1]+=z321;
		u_device[2*i3*n2*n1+2*i2*n1+2*i1+1]+=0.5*(z_device[i3*mm2*mm1+i2*mm1+i1+1]+z321);
		u_device[2*i3*n2*n1+(2*i2+1)*n1+2*i1]+=0.5*z1[i1];
		u_device[2*i3*n2*n1+(2*i2+1)*n1+2*i1+1]+=0.25*(z1[i1]+z1[i1+1]);
		u_device[(2*i3+1)*n2*n1+2*i2*n1+2*i1]+=0.5*z2[i1];
		u_device[(2*i3+1)*n2*n1+2*i2*n1+2*i1+1]+=0.25*(z2[i1]+z2[i1+1]);
		u_device[(2*i3+1)*n2*n1+(2*i2+1)*n1+2*i1]+=0.25*z3[i1];
		u_device[(2*i3+1)*n2*n1+(2*i2+1)*n1+2*i1+1]+=0.125*(z3[i1]+z3[i1+1]);
	}
}

/* 
 * --------------------------------------------------------------------
 * multigrid v-cycle routine
 * --------------------------------------------------------------------
 */
static void mg3P(double u[], 
		double v[], 
		double r[], 
		double a[4], 
		double c[4], 
		int n1, 
		int n2, 
		int n3, 
		int k){
	int j;
	/*
	 * --------------------------------------------------------------------
	 * down cycle.
	 * restrict the residual from the find grid to the coarse
	 * -------------------------------------------------------------------
	 */
	for(k = lt; k >= lb+1; k--){
		j = k-1;
		rprj3(&r[ir[k]], m1[k], m2[k], m3[k], &r[ir[j]], m1[j], m2[j], m3[j], k);
	}
	k = lb;
	/*
	 * --------------------------------------------------------------------
	 * compute an approximate solution on the coarsest grid
	 * --------------------------------------------------------------------
	 */
	zero3(&u[ir[k]], m1[k], m2[k], m3[k]);
	psinv(&r[ir[k]], &u[ir[k]], m1[k], m2[k], m3[k], c, k);
	for(k = lb+1; k <= lt-1; k++){
		j = k-1;
		/*
		 * --------------------------------------------------------------------
		 * prolongate from level k-1  to k
		 * -------------------------------------------------------------------
		 */
		zero3(&u[ir[k]], m1[k], m2[k], m3[k]);
		interp(&u[ir[j]], m1[j], m2[j], m3[j], &u[ir[k]], m1[k], m2[k], m3[k], k);
		/*
		 * --------------------------------------------------------------------
		 * compute residual for level k
		 * --------------------------------------------------------------------
		 */
		resid(&u[ir[k]], &r[ir[k]], &r[ir[k]], m1[k], m2[k], m3[k], a, k);
		/*
		 * --------------------------------------------------------------------
		 * apply smoother
		 * --------------------------------------------------------------------
		 */
		psinv(&r[ir[k]], &u[ir[k]], m1[k], m2[k], m3[k], c, k);
	}
	j = lt - 1; 
	k = lt;
	interp(&u[ir[j]], m1[j], m2[j], m3[j], u, n1, n2, n3, k);
	resid(u, v, r, n1, n2, n3, a, k);
	psinv(r, u, n1, n2, n3, c, k);
}

static void mg3P_gpu(double* u_device, 
		double* v_device, 
		double* r_device, 
		double* a_device, 
		double* c_device, 
		int n1, 
		int n2, 
		int n3, 
		int k){
	int j;
	/*
	 * --------------------------------------------------------------------
	 * down cycle.
	 * restrict the residual from the find grid to the coarse
	 * -------------------------------------------------------------------
	 */
	for(k = lt; k >= lb+1; k--){
		j = k-1;
		rprj3_gpu(r_device+ir[k], m1[k], m2[k], m3[k], r_device+ir[j], m1[j], m2[j], m3[j],	k);
	}
	k = lb;
	/*
	 * --------------------------------------------------------------------
	 * compute an approximate solution on the coarsest grid
	 * --------------------------------------------------------------------
	 */
	zero3_gpu(u_device+ir[k], m1[k], m2[k], m3[k]);
	psinv_gpu(r_device+ir[k], u_device+ir[k], m1[k], m2[k], m3[k], c_device, k);
	for(k = lb+1; k <= lt-1; k++){
		j = k-1;
		/*
		 * --------------------------------------------------------------------
		 * prolongate from level k-1  to k
		 * -------------------------------------------------------------------
		 */
		zero3_gpu(u_device+ir[k], m1[k], m2[k], m3[k]);
		interp_gpu(u_device+ir[j], m1[j], m2[j], m3[j], u_device+ir[k], m1[k], m2[k], m3[k], k);
		/*
		 * --------------------------------------------------------------------
		 * compute residual for level k
		 * --------------------------------------------------------------------
		 */
		resid_gpu(u_device+ir[k], r_device+ir[k], r_device+ir[k], m1[k], m2[k], m3[k], a_device, k);
		/*
		 * --------------------------------------------------------------------
		 * apply smoother
		 * --------------------------------------------------------------------
		 */
		psinv_gpu(r_device+ir[k], u_device+ir[k], m1[k], m2[k], m3[k], c_device, k);
	}
	j = lt - 1; 
	k = lt;
	interp_gpu(u_device+ir[j], m1[j], m2[j], m3[j], u_device, n1, n2, n3, k);	
	resid_gpu(u_device, v_device, r_device, n1, n2, n3, a_device, k);
	psinv_gpu(r_device, u_device, n1, n2, n3, c_device, k);
}

/*
 * ---------------------------------------------------------------------
 * norm2u3 evaluates approximations to the l2 norm and the
 * uniform (or l-infinity or chebyshev) norm, under the
 * assumption that the boundaries are periodic or zero. add the
 * boundaries in with half weight (quarter weight on the edges
 * and eighth weight at the corners) for inhomogeneous boundaries.
 * ---------------------------------------------------------------------
 */
static void norm2u3(void* pointer_r, 
		int n1, 
		int n2, 
		int n3, 
		double* rnm2, 
		double* rnmu, 
		int nx, 
		int ny, 
		int nz){
	double (*r)[n2][n1] = (double (*)[n2][n1])pointer_r;

	double s, a;
	int i3, i2, i1;

	double dn;

	dn = 1.0*nx*ny*nz;

	s = 0.0;
	*rnmu = 0.0;
	for(i3 = 1; i3 < n3-1; i3++){
		for(i2 = 1; i2 < n2-1; i2++){
			for(i1 = 1; i1 < n1-1; i1++){
				s = s + r[i3][i2][i1] * r[i3][i2][i1];
				a = fabs(r[i3][i2][i1]);
				if(a > *rnmu){*rnmu = a;}
			}
		}
	}

	*rnm2 = sqrt(s/dn);
}

static void norm2u3_gpu(double* r_device, 
		int n1, 
		int n2, 
		int n3, 
		double* rnm2, 
		double* rnmu, 
		int nx, 
		int ny, 
		int nz){
#if defined(PROFILING)
	timer_start(PROFILING_NORM2U3);
#endif

	int temp_size;
	double* data_host;
	double* data_device;
	double* sum_host;
	double* sum_device;
	double* max_host;
	double* max_device;	

	double dn=1.0*nx*ny*nz;
	double s=0.0;
	double max_rnmu=0.0;

	int threads_per_block_x=threads_per_block_on_norm2u3;
	int threads_per_block_y=1;
	int amount_of_work_x=(n2-2)*threads_per_block_x;
	int amount_of_work_y=n3-2;	
	dim3 threadsPerBlock(threads_per_block_x,
			threads_per_block_y,
			1);
	dim3 blocksPerGrid(ceil(double(amount_of_work_x)/double(threadsPerBlock.x)),
			ceil(double(amount_of_work_y)/double(threadsPerBlock.y)),
			1);

	temp_size=(amount_of_work_x*amount_of_work_y)/(threads_per_block_x*threads_per_block_y);	
	cudaMalloc(&data_device,2*temp_size*sizeof(double));
	sum_device=data_device;
	max_device=data_device+temp_size;

	norm2u3_gpu_kernel<<<blocksPerGrid, 
		threadsPerBlock,
		size_shared_data_on_norm2u3>>>(
				r_device,
				n1,
				n2,
				n3,
				sum_device,
				max_device,
				blocksPerGrid.x,
				amount_of_work_x);

	data_host=(double*)malloc(2*temp_size*sizeof(double));
	cudaMemcpy(data_host, data_device, 2*temp_size*sizeof(double), cudaMemcpyDeviceToHost);
	sum_host=data_host;
	max_host=(data_host+temp_size);

	for(int j=0; j<temp_size; j++){
		s=s+sum_host[j];
		if(max_rnmu<max_host[j]){max_rnmu=max_host[j];}
	}

	cudaFree(data_device);
	free(data_host);

	*rnmu=max_rnmu;
	*rnm2=sqrt(s/dn);

#if defined(PROFILING)
	timer_stop(PROFILING_NORM2U3);
#endif
}

__global__ void norm2u3_gpu_kernel(double* r,
		const int n1, 
		const int n2, 
		const int n3,
		double* res_sum,
		double* res_max,
		int number_of_blocks_on_x_axis,
		int amount_of_work){
	int check=blockIdx.x*blockDim.x+threadIdx.x;
	if(check>=amount_of_work){return;}

	double* scratch_sum = (double*)(extern_share_data);
	double* scratch_max = (double*)(scratch_sum+blockDim.x);

	int i3=blockIdx.y*blockDim.y+threadIdx.y+1;
	int i2=blockIdx.x+1;
	int i1=threadIdx.x+1;

	double s=0.0;
	double my_rnmu=0.0;
	double a;

	while(i1<n1-1){
		double r321=r[i3*n2*n1+i2*n1+i1];
		s=s+r321*r321;
		a=fabs(r321);
		my_rnmu=(a>my_rnmu)?a:my_rnmu;
		i1+=blockDim.x;
	}

	int lid=threadIdx.x;
	scratch_sum[lid]=s;
	scratch_max[lid]=my_rnmu;

	__syncthreads();
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(lid<i){
			scratch_sum[lid]+=scratch_sum[lid+i];
			scratch_max[lid]=(scratch_max[lid]>scratch_max[lid+i])?scratch_max[lid]:scratch_max[lid+i];
		}
		__syncthreads();
	}
	if(lid == 0){
		int idx=blockIdx.y*number_of_blocks_on_x_axis+blockIdx.x;
		res_sum[idx]=scratch_sum[0];
		res_max[idx]=scratch_max[0];
	}
}

/*
 * ---------------------------------------------------------------------
 * power raises an integer, disguised as a double
 * precision real, to an integer power
 * ---------------------------------------------------------------------
 */
static double power(double a, 
		int n){
	double aj;
	int nj;
	double power;

	power = 1.0;
	nj = n;
	aj = a;

	while(nj != 0){
		if((nj%2)==1){randlc(&power, aj);}
		randlc(&aj, aj);
		nj = nj/2;
	}

	return power;
}

/*
 * --------------------------------------------------------------------
 * psinv applies an approximate inverse as smoother: u = u + Cr
 * 
 * this  implementation costs  15A + 4M per result, where
 * A and M denote the costs of Addition and Multiplication.  
 * presuming coefficient c(3) is zero (the NPB assumes this,
 * but it is thus not a general case), 2A + 1M may be eliminated,
 * resulting in 13A + 3M.
 * note that this vectorizes, and is also fine for cache 
 * based machines.  
 * --------------------------------------------------------------------
 */
static void psinv(void* pointer_r, 
		void* pointer_u, 
		int n1, 
		int n2, 
		int n3, 
		double c[4], 
		int k){
	double (*r)[n2][n1] = (double (*)[n2][n1])pointer_r;
	double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;	

	int i3, i2, i1;
	double r1[M], r2[M];

	for(i3 = 1; i3 < n3-1; i3++){
		for(i2 = 1; i2 < n2-1; i2++){
			for(i1 = 0; i1 < n1; i1++){
				r1[i1] = r[i3][i2-1][i1] + r[i3][i2+1][i1]
					+ r[i3-1][i2][i1] + r[i3+1][i2][i1];
				r2[i1] = r[i3-1][i2-1][i1] + r[i3-1][i2+1][i1]
					+ r[i3+1][i2-1][i1] + r[i3+1][i2+1][i1];
			}
			for(i1 = 1; i1 < n1-1; i1++){
				u[i3][i2][i1] = u[i3][i2][i1]
					+ c[0] * r[i3][i2][i1]
					+ c[1] * ( r[i3][i2][i1-1] + r[i3][i2][i1+1]
							+ r1[i1] )
					+ c[2] * ( r2[i1] + r1[i1-1] + r1[i1+1] );
				/*
				 * --------------------------------------------------------------------
				 * assume c(3) = 0    (enable line below if c(3) not= 0)
				 * --------------------------------------------------------------------
				 * > + c(3) * ( r2(i1-1) + r2(i1+1) )
				 * --------------------------------------------------------------------
				 */
			}
		}
	}

	/*
	 * --------------------------------------------------------------------
	 * exchange boundary points
	 * --------------------------------------------------------------------
	 */
	comm3(u,n1,n2,n3,k);

	if(debug_vec[0] >= 1){
		rep_nrm(u,n1,n2,n3,(char*)"   psinv",k);
	}

	if(debug_vec[3] >= k){
		showall(u,n1,n2,n3);
	}
}

static void psinv_gpu(double* r_device, 
		double* u_device, 
		int n1, 
		int n2, 
		int n3, 
		double* c_device, 
		int k){
#if defined(PROFILING)
	timer_start(PROFILING_PSINV);
#endif

	int threads_per_block = n1 > threads_per_block_on_psinv ? threads_per_block_on_psinv : n1;
	int amount_of_work_x=(n2-2)*threads_per_block;
	int amount_of_work_y=n3-2;
	int threads_per_block_x=threads_per_block;
	int threads_per_block_y=1;
	dim3 threadsPerBlock(threads_per_block_x,
			threads_per_block_y,
			1);
	dim3 blocksPerGrid(ceil(double(amount_of_work_x) / double(threadsPerBlock.x)),
			ceil(double(amount_of_work_y) / double(threadsPerBlock.y)),
			1);

	psinv_gpu_kernel<<<blocksPerGrid, 
		threadsPerBlock,
		size_shared_data_on_psinv>>>(
				r_device,
				u_device,
				c_device,
				n1,
				n2,
				n3,
				amount_of_work_x);

#if defined(PROFILING)
	timer_stop(PROFILING_PSINV);
#endif

	/*
	 * --------------------------------------------------------------------
	 * exchange boundary points
	 * --------------------------------------------------------------------
	 */
	comm3_gpu(u_device,n1,n2,n3,k);
}

__global__ void psinv_gpu_kernel(double* r,
		double* u,
		double* c,
		int n1,
		int n2,
		int n3,
		int amount_of_work){
	int check=blockIdx.x*blockDim.x+threadIdx.x;
	if(check>=amount_of_work){return;}

	double* r1 = (double*)(extern_share_data);
	double* r2 = (double*)(r1+M);

	int i3=blockIdx.y*blockDim.y+threadIdx.y+1;
	int i2=blockIdx.x+1;
	int lid=threadIdx.x;
	int i1;	

	for(i1=lid; i1<n1; i1+=blockDim.x){
		r1[i1]=r[i3*n2*n1+(i2-1)*n2+i1]
			+r[i3*n2*n1+(i2+1)*n1+i1]
			+r[(i3-1)*n2*n1+i2*n1+i1]
			+r[(i3+1)*n2*n1+i2*n1+i1];
		r2[i1]=r[(i3-1)*n2*n1+(i2-1)*n1+i1]
			+r[(i3-1)*n2*n1+(i2+1)*n1+i1]
			+r[(i3+1)*n2*n1+(i2-1)*n1+i1]
			+r[(i3+1)*n2*n1+(i2+1)*n1+i1];
	} __syncthreads();
	for(i1=lid+1; i1<n1-1; i1+=blockDim.x){
		u[i3*n2*n1+i2*n1+i1]=u[i3*n2*n1+i2*n1+i1]
			+c[0]*r[i3*n2*n1+i2*n1+i1]
			+c[1]*(r[i3*n2*n1+i2*n1+i1-1]
					+r[i3*n2*n1+i2*n1+i1+1]
					+r1[i1])
			+c[2]*(r2[i1]+r1[i1-1]+r1[i1+1] );
	}
}

static void release_gpu(){
	cudaFree(a_device);
	cudaFree(c_device);
	cudaFree(u_device);
	cudaFree(v_device);
	cudaFree(r_device);
}

/*
 * ---------------------------------------------------------------------
 * report on norm
 * ---------------------------------------------------------------------
 */
static void rep_nrm(void* pointer_u, 
		int n1, 
		int n2, 
		int n3, 
		char* title, 
		int kk){
	double rnm2, rnmu;

	norm2u3(pointer_u,n1,n2,n3,&rnm2,&rnmu,nx[kk],ny[kk],nz[kk]);
	printf(" Level%2d in %8s: norms =%21.14e%21.14e\n", kk, title, rnm2, rnmu);
}

/*
 * --------------------------------------------------------------------
 * resid computes the residual: r = v - Au
 *
 * this  implementation costs  15A + 4M per result, where
 * A and M denote the costs of addition (or subtraction) and 
 * multiplication, respectively. 
 * presuming coefficient a(1) is zero (the NPB assumes this,
 * but it is thus not a general case), 3A + 1M may be eliminated,
 * resulting in 12A + 3M.
 * note that this vectorizes, and is also fine for cache 
 * based machines.  
 * --------------------------------------------------------------------
 */
static void resid(void* pointer_u, 
		void* pointer_v, 
		void* pointer_r, 
		int n1, 
		int n2, 
		int n3, 
		double a[4], 
		int k){
	double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;
	double (*v)[n2][n1] = (double (*)[n2][n1])pointer_v;
	double (*r)[n2][n1] = (double (*)[n2][n1])pointer_r;	

	int i3, i2, i1;
	double u1[M], u2[M];

	for(i3 = 1; i3 < n3-1; i3++){
		for(i2 = 1; i2 < n2-1; i2++){
			for(i1 = 0; i1 < n1; i1++){
				u1[i1] = u[i3][i2-1][i1] + u[i3][i2+1][i1]
					+ u[i3-1][i2][i1] + u[i3+1][i2][i1];
				u2[i1] = u[i3-1][i2-1][i1] + u[i3-1][i2+1][i1]
					+ u[i3+1][i2-1][i1] + u[i3+1][i2+1][i1];
			}
			for(i1 = 1; i1 < n1-1; i1++){
				r[i3][i2][i1] = v[i3][i2][i1]
					- a[0] * u[i3][i2][i1]
					/*
					 * ---------------------------------------------------------------------
					 * assume a(1) = 0 (enable 2 lines below if a(1) not= 0)
					 * ---------------------------------------------------------------------
					 * > - a(1) * ( u(i1-1,i2,i3) + u(i1+1,i2,i3)
					 * > + u1(i1) )
					 * ---------------------------------------------------------------------
					 */
					- a[2] * ( u2[i1] + u1[i1-1] + u1[i1+1] )
					- a[3] * ( u2[i1-1] + u2[i1+1] );
			}
		}
	}

	/*
	 * --------------------------------------------------------------------
	 * exchange boundary data
	 * --------------------------------------------------------------------
	 */
	comm3(r,n1,n2,n3,k);

	if(debug_vec[0] >= 1){
		rep_nrm(r,n1,n2,n3,(char*)"   resid",k);
	}

	if(debug_vec[2] >= k){
		showall(r,n1,n2,n3);
	}
}

static void resid_gpu(double* u_device,
		double* v_device,
		double* r_device,
		int n1,
		int n2,
		int n3,
		double* a_device,
		int k){
#if defined(PROFILING)
	timer_start(PROFILING_RESID);
#endif

	int threads_per_block = n1 > threads_per_block_on_resid ? threads_per_block_on_resid : n1;
	int amount_of_work_x=(n2-2)*threads_per_block;
	int amount_of_work_y=n3-2;
	int threads_per_block_x=threads_per_block;
	int threads_per_block_y=1;
	dim3 threadsPerBlock(threads_per_block_x,
			threads_per_block_y,
			1);
	dim3 blocksPerGrid(ceil(double(amount_of_work_x) / double(threadsPerBlock.x)),
			ceil(double(amount_of_work_y) / double(threadsPerBlock.y)),
			1);

	resid_gpu_kernel<<<blocksPerGrid, 
		threadsPerBlock,
		size_shared_data_on_resid>>>(
				u_device,
				v_device,
				r_device,
				a_device,
				n1,
				n2,
				n3,
				amount_of_work_x);

#if defined(PROFILING)
	timer_stop(PROFILING_RESID);
#endif

	/*
	 * --------------------------------------------------------------------
	 * exchange boundary data
	 * --------------------------------------------------------------------
	 */
	comm3_gpu(r_device,n1,n2,n3,k);
}

__global__ void resid_gpu_kernel(double* u,
		double* v,
		double* r,
		double* a,
		int n1,
		int n2,
		int n3,
		int amount_of_work){
	int check=blockIdx.x*blockDim.x+threadIdx.x;
	if(check>=amount_of_work){return;}

	double* u1=(double*)(extern_share_data);
	double* u2=(double*)(u1+M);

	int i3=blockIdx.y*blockDim.y+threadIdx.y+1;
	int i2=blockIdx.x+1;
	int lid=threadIdx.x;
	int i1;	

	for(i1=lid; i1<n1; i1+=blockDim.x){
		u1[i1]=u[i3*n2*n1+(i2-1)*n1+i1]
			+u[i3*n2*n1+(i2+1)*n1+i1]
			+u[(i3-1)*n2*n1+i2*n1+i1]
			+u[(i3+1)*n2*n1+i2*n1+i1];
		u2[i1]=u[(i3-1)*n2*n1+(i2-1)*n1+i1]
			+u[(i3-1)*n2*n1+(i2+1)*n1+i1]
			+u[(i3+1)*n2*n1+(i2-1)*n1+i1]
			+u[(i3+1)*n2*n1+(i2+1)*n1+i1];
	} __syncthreads();
	for(i1=lid+1; i1<n1-1; i1+=blockDim.x){
		r[i3*n2*n1+i2*n1+i1]=v[i3*n2*n1+i2*n1+i1]
			-a[0]*u[i3*n2*n1+i2*n1+i1]
			-a[2]*(u2[i1]+u1[i1-1]+u1[i1+1])
			-a[3]*(u2[i1-1]+u2[i1+1] );
	}
}

/*
 * --------------------------------------------------------------------
 * rprj3 projects onto the next coarser grid, 
 * using a trilinear finite element projection: s = r' = P r
 *     
 * this  implementation costs 20A + 4M per result, where
 * A and M denote the costs of addition and multiplication.  
 * note that this vectorizes, and is also fine for cache 
 * based machines.  
 * --------------------------------------------------------------------
 */
static void rprj3(void* pointer_r, 
		int m1k, 
		int m2k, 
		int m3k, 
		void* pointer_s, 
		int m1j, 
		int m2j, 
		int m3j, 
		int k){
	double (*r)[m2k][m1k] = (double (*)[m2k][m1k])pointer_r;
	double (*s)[m2j][m1j] = (double (*)[m2j][m1j])pointer_s;	

	int j3, j2, j1, i3, i2, i1, d1, d2, d3, j;

	double x1[M], y1[M], x2, y2;

	if(m1k == 3){
		d1 = 2;
	}else{
		d1 = 1;
	}
	if(m2k == 3){
		d2 = 2;
	}else{
		d2 = 1;
	}
	if(m3k == 3){
		d3 = 2;
	}else{
		d3 = 1;
	}
	for(j3 = 1; j3 < m3j-1; j3++){
		i3 = 2*j3-d3;		
		for(j2 = 1; j2 < m2j-1; j2++){
			i2 = 2*j2-d2;
			for(j1 = 1; j1 < m1j; j1++){
				i1 = 2*j1-d1;				
				x1[i1] = r[i3+1][i2][i1] + r[i3+1][i2+2][i1]
					+ r[i3][i2+1][i1] + r[i3+2][i2+1][i1];
				y1[i1] = r[i3][i2][i1] + r[i3+2][i2][i1]
					+ r[i3][i2+2][i1] + r[i3+2][i2+2][i1];
			}
			for(j1 = 1; j1 < m1j-1; j1++){
				i1 = 2*j1-d1;				
				y2 = r[i3][i2][i1+1] + r[i3+2][i2][i1+1]
					+ r[i3][i2+2][i1+1] + r[i3+2][i2+2][i1+1];
				x2 = r[i3+1][i2][i1+1] + r[i3+1][i2+2][i1+1]
					+ r[i3][i2+1][i1+1] + r[i3+2][i2+1][i1+1];
				s[j3][j2][j1] =
					0.5 * r[i3+1][i2+1][i1+1]
					+ 0.25 * ( r[i3+1][i2+1][i1] + r[i3+1][i2+1][i1+2] + x2)
					+ 0.125 * ( x1[i1] + x1[i1+2] + y2)
					+ 0.0625 * ( y1[i1] + y1[i1+2] );
			}
		}
	}

	j=k-1;
	comm3(s,m1j,m2j,m3j,j);

	if(debug_vec[0] >= 1){
		rep_nrm(s,m1j,m2j,m3j,(char*)"   rprj3",k-1);	
	}

	if(debug_vec[4] >= k){
		showall(s,m1j,m2j,m3j);
	}
}

static void rprj3_gpu(double* r_device, 
		int m1k, 
		int m2k, 
		int m3k, 
		double* s_device, 
		int m1j, 
		int m2j, 
		int m3j, 
		int k){
#if defined(PROFILING)
	timer_start(PROFILING_RPRJ3);
#endif

	int d1,d2,d3,j;

	if(m1k==3){
		d1=2;
	}else{
		d1=1;
	}
	if(m2k==3){
		d2=2;
	}else{
		d2=1;
	}
	if(m3k==3){
		d3=2;
	}else{
		d3=1;
	}

	int amount_of_work_x=(m2j-2)*(m1j-1);
	int amount_of_work_y=m3j-2;
	int threads_per_block_x;
	int threads_per_block_y=1;
	if(threads_per_block_on_rprj3 != m1j-1){
		threads_per_block_x = m1j-1;
	}
	else{
		threads_per_block_x = threads_per_block_on_interp;
	}
	dim3 threadsPerBlock(threads_per_block_x,
			threads_per_block_y,
			1);
	dim3 blocksPerGrid(ceil(double(amount_of_work_x) / double(threadsPerBlock.x)),
			ceil(double(amount_of_work_y) / double(threadsPerBlock.y)),
			1);

	rprj3_gpu_kernel<<<blocksPerGrid, 
		threadsPerBlock,
		size_shared_data_on_rprj3>>>(
				r_device,
				s_device,
				m1k,
				m2k,
				m3k,
				m1j,
				m2j,
				m3j,
				d1,
				d2,
				d3,
				amount_of_work_x);

#if defined(PROFILING)
	timer_stop(PROFILING_RPRJ3);
#endif

	j=k-1;
	comm3_gpu(s_device,m1j,m2j,m3j,j);
}

__global__ void rprj3_gpu_kernel(double* r_device,
		double* s_device,
		int m1k,
		int m2k,
		int m3k,
		int m1j,
		int m2j,
		int m3j,
		int d1, 
		int d2, 
		int d3,
		int amount_of_work){
	int check=blockIdx.x*blockDim.x+threadIdx.x;
	if(check>=amount_of_work){return;}

	int j3,j2,j1,i3,i2,i1;
	double x2,y2;

	double* x1 = (double*)(extern_share_data);
	double* y1 = (double*)(x1+M);

	j3=blockIdx.y*blockDim.y+threadIdx.y+1;
	j2=blockIdx.x+1;
	j1=threadIdx.x+1;

	i3=2*j3-d3;
	i2=2*j2-d2;
	i1=2*j1-d1;
	x1[i1]=r_device[(i3+1)*m2k*m1k+i2*m1k+i1]
		+r_device[(i3+1)*m2k*m1k+(i2+2)*m1k+i1]
		+r_device[i3*m2k*m1k+(i2+1)*m1k+i1]
		+r_device[(i3+2)*m2k*m1k+(i2+1)*m1k+i1];
	y1[i1]=r_device[i3*m2k*m1k+i2*m1k+i1]
		+r_device[(i3+2)*m2k*m1k+i2*m1k+i1]
		+r_device[i3*m2k*m1k+(i2+2)*m1k+i1]
		+r_device[(i3+2)*m2k*m1k+(i2+2)*m1k+i1];		
	__syncthreads();
	if(j1<m1j-1){
		i1=2*j1-d1;
		y2=r_device[i3*m2k*m1k+i2*m1k+i1+1]
			+r_device[(i3+2)*m2k*m1k+i2*m1k+i1+1]
			+r_device[i3*m2k*m1k+(i2+2)*m1k+i1+1]
			+r_device[(i3+2)*m2k*m1k+(i2+2)*m1k+i1+1];
		x2=r_device[(i3+1)*m2k*m1k+i2*m1k+i1+1]
			+r_device[(i3+1)*m2k*m1k+(i2+2)*m1k+i1+1]
			+r_device[i3*m2k*m1k+(i2+1)*m1k+i1+1]
			+r_device[(i3+2)*m2k*m1k+(i2+1)*m1k+i1+1];
		s_device[j3*m2j*m1j+j2*m1j+j1]=
			0.5*r_device[(i3+1)*m2k*m1k+(i2+1)*m1k+i1+1]
			+0.25*(r_device[(i3+1)*m2k*m1k+(i2+1)*m1k+i1]
					+r_device[(i3+1)*m2k*m1k+(i2+1)*m1k+i1+2]+x2)
			+0.125*(x1[i1]+x1[i1+2]+y2)
			+0.0625*(y1[i1]+y1[i1+2]);
	}
}

static void setup(int* n1, 
		int* n2, 
		int* n3, 
		int k){
	int j;

	int ax, mi[MAXLEVEL+1][3];
	int ng[MAXLEVEL+1][3];

	ng[lt][0] = nx[lt];
	ng[lt][1] = ny[lt];
	ng[lt][2] = nz[lt];
	for(ax = 0; ax < 3; ax++){
		for(k = lt-1; k >= 1; k--){
			ng[k][ax] = ng[k+1][ax]/2;
		}
	}
	for(k = lt; k >= 1; k--){
		nx[k] = ng[k][0];
		ny[k] = ng[k][1];
		nz[k] = ng[k][2];
	}

	for(k = lt; k >= 1; k--){
		for (ax = 0; ax < 3; ax++){
			mi[k][ax] = 2 + ng[k][ax];
		}

		m1[k] = mi[k][0];
		m2[k] = mi[k][1];
		m3[k] = mi[k][2];
	}

	k = lt;
	is1 = 2 + ng[k][0] - ng[lt][0];
	ie1 = 1 + ng[k][0];
	*n1 = 3 + ie1 - is1;
	is2 = 2 + ng[k][1] - ng[lt][1];
	ie2 = 1 + ng[k][1];
	*n2 = 3 + ie2 - is2;
	is3 = 2 + ng[k][2] - ng[lt][2];
	ie3 = 1 + ng[k][2];
	*n3 = 3 + ie3 - is3;

	ir[lt] = 0;
	for(j = lt-1; j >= 1; j--){
		ir[j] = ir[j+1]+ONE*m1[j+1]*m2[j+1]*m3[j+1];
	}

	if(debug_vec[1] >= 1){
		printf(" in setup, \n");
		printf("   k  lt  nx  ny  nz  n1  n2  n3 is1 is2 is3 ie1 ie2 ie3\n");
		printf("%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d\n", 
				k,lt,ng[k][0],ng[k][1],ng[k][2],*n1,*n2,*n3,is1,is2,is3,ie1,ie2,ie3);
	}
}

static void setup_gpu(double* a, 
		double* c){
	/*
	 * struct cudaDeviceProp{
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
	/* define gpu_device */
	cudaGetDeviceCount(&total_devices);
	if(total_devices==0){
		printf("\n\n\n No Nvidia GPU found!!! \n\n\n");
		exit(-1);
	}else if((GPU_DEVICE>=0)&&
			(GPU_DEVICE<total_devices)){
		gpu_device_id = GPU_DEVICE;
	}else{
		gpu_device_id = 0;
	}
	cudaSetDevice(gpu_device_id);	
	cudaGetDeviceProperties(&gpu_device_properties, gpu_device_id);

	/* define threads_per_block */
	if((MG_THREADS_PER_BLOCK_ON_COMM3>=1)&&
			(MG_THREADS_PER_BLOCK_ON_COMM3<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_comm3 = MG_THREADS_PER_BLOCK_ON_COMM3;
	}else{
		threads_per_block_on_comm3 = gpu_device_properties.warpSize;
	}	

	if((MG_THREADS_PER_BLOCK_ON_INTERP>=1)&&
			(MG_THREADS_PER_BLOCK_ON_INTERP<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_interp = MG_THREADS_PER_BLOCK_ON_INTERP;
	}else{
		threads_per_block_on_interp = gpu_device_properties.warpSize;
	}	

	if((MG_THREADS_PER_BLOCK_ON_NORM2U3>=1)&&
			(MG_THREADS_PER_BLOCK_ON_NORM2U3<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_norm2u3 = MG_THREADS_PER_BLOCK_ON_NORM2U3;
	}else{
		threads_per_block_on_norm2u3 = gpu_device_properties.warpSize;
	}
	if((MG_THREADS_PER_BLOCK_ON_PSINV>=1)&&
			(MG_THREADS_PER_BLOCK_ON_PSINV<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_psinv = MG_THREADS_PER_BLOCK_ON_PSINV;
	}else{
		threads_per_block_on_psinv = gpu_device_properties.warpSize;
	}	
	if((MG_THREADS_PER_BLOCK_ON_RESID>=1)&&
			(MG_THREADS_PER_BLOCK_ON_RESID<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_resid = MG_THREADS_PER_BLOCK_ON_RESID;
	}else{
		threads_per_block_on_resid = gpu_device_properties.warpSize;
	}
	if((MG_THREADS_PER_BLOCK_ON_RPRJ3>=1)&&
			(MG_THREADS_PER_BLOCK_ON_RPRJ3<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_rprj3 = MG_THREADS_PER_BLOCK_ON_RPRJ3;
	}else{
		threads_per_block_on_rprj3 = gpu_device_properties.warpSize;
	}
	if((MG_THREADS_PER_BLOCK_ON_ZERO3>=1)&&
			(MG_THREADS_PER_BLOCK_ON_ZERO3<=gpu_device_properties.maxThreadsPerBlock)){
		threads_per_block_on_zero3 = MG_THREADS_PER_BLOCK_ON_ZERO3;
	}else{
		threads_per_block_on_zero3 = gpu_device_properties.warpSize;
	}	

	size_a_device=sizeof(double)*(4);
	size_c_device=sizeof(double)*(4);
	size_u_device=sizeof(double)*(NR);
	size_v_device=sizeof(double)*(NV);
	size_r_device=sizeof(double)*(NR);
	size_shared_data_on_interp=3*M*sizeof(double);
	size_shared_data_on_norm2u3=2*threads_per_block_on_norm2u3*sizeof(double);
	size_shared_data_on_psinv=2*M*sizeof(double);
	size_shared_data_on_resid=2*M*sizeof(double);
	size_shared_data_on_rprj3=2*M*sizeof(double);

	cudaMalloc(&a_device, size_a_device);
	cudaMalloc(&c_device, size_c_device);
	cudaMalloc(&u_device, size_u_device);
	cudaMalloc(&v_device, size_v_device);
	cudaMalloc(&r_device, size_r_device);
	cudaMemcpy(a_device, a, size_a_device, cudaMemcpyHostToDevice);
	cudaMemcpy(c_device, c, size_c_device, cudaMemcpyHostToDevice);
	cudaMemcpy(u_device, u, size_u_device, cudaMemcpyHostToDevice);
	cudaMemcpy(v_device, v, size_v_device, cudaMemcpyHostToDevice);
	cudaMemcpy(r_device, r, size_r_device, cudaMemcpyHostToDevice);		
}

static void showall(void* pointer_z, 
		int n1, 
		int n2, 
		int n3){
	double (*z)[n2][n1] = (double (*)[n2][n1])pointer_z;

	int i1,i2,i3;
	int m1, m2, m3;

	m1 = min(n1,18);
	m2 = min(n2,14);
	m3 = min(n3,18);

	printf("\n");
	for(i3 = 0; i3 < m3; i3++){
		for(i2 = 0; i2 < m2; i2++){
			for(i1 = 0; i1 < m1; i1++){			
				printf("%6.3f", z[i3][i2][i1]);
			}
			printf("\n");
		}
		printf(" - - - - - - - \n");
	}
	printf("\n");
}

static void zero3(void* pointer_z, 
		int n1, 
		int n2, 
		int n3){
	double (*z)[n2][n1] = (double (*)[n2][n1])pointer_z;

	int i1, i2, i3;
	for(i3 = 0;i3 < n3; i3++){
		for(i2 = 0; i2 < n2; i2++){
			for(i1 = 0; i1 < n1; i1++){
				z[i3][i2][i1] = 0.0;
			}
		}
	}
}

static void zero3_gpu(double* z_device, 
		int n1, 
		int n2, 
		int n3){
#if defined(PROFILING)
	timer_start(PROFILING_ZERO3);
#endif

	int threads_per_block = threads_per_block_on_zero3;
	int amount_of_work = n1*n2*n3;	
	int blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));

	zero3_gpu_kernel<<<blocks_per_grid, threads_per_block>>>(z_device,
			n1,
			n2,
			n3,
			amount_of_work);

#if defined(PROFILING)
	timer_stop(PROFILING_ZERO3);
#endif
}

__global__ void zero3_gpu_kernel(double* z, 
		int n1, 
		int n2, 
		int n3, 
		int amount_of_work){
	int thread_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(thread_id>=(n1*n2*n3)){return;}
	z[thread_id]=0.0;
}

/*
 * ---------------------------------------------------------------------
 * zran3 loads +1 at ten randomly chosen points,
 * loads -1 at a different ten random points,
 * and zero elsewhere.
 * ---------------------------------------------------------------------
 */
static void zran3(void* pointer_z, 
		int n1, 
		int n2, 
		int n3, 
		int nx, 
		int ny, 
		int k){
	double (*z)[n2][n1] = (double (*)[n2][n1])pointer_z;

	int i0, m0, m1;

	int i1, i2, i3, d1, e2, e3;
	double xx, x0, x1, a1, a2, ai;

	double ten[2][MM], best;
	int i, j1[2][MM], j2[2][MM], j3[2][MM];
	int jg[2][MM][4];

	a1 = power(A, nx);
	a2 = power(A, nx*ny);

	zero3(z, n1, n2, n3);

	i = is1-2+nx*(is2-2+ny*(is3-2));

	ai = power(A, i);
	d1 = ie1 - is1 + 1;
	e2 = ie2 - is2 + 2;
	e3 = ie3 - is3 + 2;
	x0 = X;
	randlc(&x0, ai);
	for(i3 = 1; i3 < e3; i3++){
		x1 = x0;
		for(i2 = 1; i2 < e2; i2++){
			xx = x1;
			vranlc(d1, &xx, A, &(z[i3][i2][1]));
			randlc(&x1,a1);
		}
		randlc(&x0, a2);
	}

	/*
	 * ---------------------------------------------------------------------
	 * each processor looks for twenty candidates
	 * ---------------------------------------------------------------------
	 */	
	for(i = 0; i < MM; i++){
		ten[1][i] = 0.0;
		j1[1][i] = 0;
		j2[1][i] = 0;
		j3[1][i] = 0;
		ten[0][i] = 1.0;
		j1[0][i] = 0;
		j2[0][i] = 0;
		j3[0][i] = 0;
	}
	for(i3 = 1; i3 < n3-1; i3++){
		for(i2 = 1; i2 < n2-1; i2++){
			for(i1 = 1; i1 < n1-1; i1++){
				if(z[i3][i2][i1] > ten[1][0]){
					ten[1][0] = z[i3][i2][i1];
					j1[1][0] = i1;
					j2[1][0] = i2;
					j3[1][0] = i3;
					bubble(ten, j1, j2, j3, MM, 1);
				}
				if(z[i3][i2][i1] < ten[0][0]){
					ten[0][0] = z[i3][i2][i1];
					j1[0][0] = i1;
					j2[0][0] = i2;
					j3[0][0] = i3;
					bubble(ten, j1, j2, j3, MM, 0);
				}
			}
		}
	}

	/*
	 * ---------------------------------------------------------------------
	 * now which of these are globally best?
	 * ---------------------------------------------------------------------
	 */	
	i1 = MM - 1;
	i0 = MM - 1; 
	for(i = MM - 1; i >= 0; i--){
		best = 0.0;
		if(best < ten[1][i1]){
			jg[1][i][0] = 0;
			jg[1][i][1] = is1 - 2 + j1[1][i1];
			jg[1][i][2] = is2 - 2 + j2[1][i1];
			jg[1][i][3] = is3 - 2 + j3[1][i1];
			i1 = i1-1;
		}else{
			jg[1][i][0] = 0;
			jg[1][i][1] = 0;
			jg[1][i][2] = 0;
			jg[1][i][3] = 0;
		}
		best = 1.0;
		if(best > ten[0][i0]){
			jg[0][i][0] = 0;
			jg[0][i][1] = is1 - 2 + j1[0][i0];
			jg[0][i][2] = is2 - 2 + j2[0][i0];
			jg[0][i][3] = is3 - 2 + j3[0][i0];
			i0 = i0-1;
		}else{
			jg[0][i][0] = 0;
			jg[0][i][1] = 0;
			jg[0][i][2] = 0;
			jg[0][i][3] = 0;
		}
	}
	m1 = 0;
	m0 = 0;

	for(i3 = 0; i3 < n3; i3++){
		for(i2 = 0; i2 < n2; i2++){
			for(i1 = 0; i1 < n1; i1++){
				z[i3][i2][i1] = 0.0;
			}
		}
	}
	for (i = MM-1; i >= m0; i--){
		z[jg[0][i][3]][jg[0][i][2]][jg[0][i][1]] = -1.0;
	}
	for(i = MM-1; i >= m1; i--){
		z[jg[1][i][3]][jg[1][i][2]][jg[1][i][1]] = +1.0;
	}
	comm3(z, n1, n2, n3, k);
}
