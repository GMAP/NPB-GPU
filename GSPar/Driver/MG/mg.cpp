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

#define NM (2+(1<<LM)) /* actual dimension including ghost cells for communications */
#define NV (ONE*(2+(1<<NDIM1))*(2+(1<<NDIM2))*(2+(1<<NDIM3))) /* size of rhs array */
#define NR (((NV+NM*NM+5*NM+7*LM+6)/7)*8) /* size of residual array */
#define MAXLEVEL (LT_DEFAULT+1) /* maximum number of levels */
#define M (NM+1) /* set at m=1024, can handle cases up to 1024^3 case */
#define MM (10)
#define	A (pow(5.0,13.0))
#define	X (314159265.0)
#define PROFILING_TOTAL_TIME (0)
#define THREADS_PER_BLOCK (1024)

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
double a[4];
double c[4];

/* gpu variables */
int threads_per_block;
int blocks_per_grid;
int amount_of_work;
//
size_t size_a_device;
size_t size_c_device;
size_t size_u_device;
size_t size_v_device;
size_t size_r_device;
//
MemoryObject* a_device;
MemoryObject* c_device;
MemoryObject* u_device;
MemoryObject* v_device;
MemoryObject* r_device;
//
Kernel* kernel_comm3_1;
Kernel* kernel_comm3_2;
Kernel* kernel_comm3_3;
Kernel* kernel_interp;
Kernel* kernel_norm2u3;
Kernel* kernel_psinv;
Kernel* kernel_resid;
Kernel* kernel_rprj3;
Kernel* kernel_zero3;
//
extern std::string kernel_source_comm3_1;
extern std::string kernel_source_comm3_2;
extern std::string kernel_source_comm3_3;
extern std::string kernel_source_interp;
extern std::string kernel_source_norm2u3;
extern std::string kernel_source_psinv;
extern std::string kernel_source_resid;
extern std::string kernel_source_rprj3;
extern std::string kernel_source_zero3;
extern std::string source_additional_routines;
extern std::string source_additional_routines_complete;
//
string DEVICE_NAME;
Instance* driver;

/* function prototypes */
static void bubble(
		double ten[][MM], 
		int j1[][MM], 
		int j2[][MM], 
		int j3[][MM], 
		int m, 
		int ind);
static void comm3(
		void* pointer_base_u, 
		int n1, 
		int n2, 
		int n3, 
		int kk,
		int offset);
static void comm3_gpu(
		MemoryObject* base_u_device, 
		int n1, 
		int n2, 
		int n3, 
		int kk,
		int offset);
static void interp(
		void* pointer_base_z, 
		int mm1, 
		int mm2, 
		int mm3, 
		void* pointer_base_u, 
		int n1, 
		int n2, 
		int n3, 
		int k,
		int offset_1,
		int offset_2);
static void interp_gpu(
		MemoryObject* base_z_device, 
		int mm1, 
		int mm2, 
		int mm3, 
		MemoryObject* base_u_device, 
		int n1, 
		int n2, 
		int n3, 
		int k,
		int offset_1,
		int offset_2);
static void mg3P(
		double u[], 
		double v[], 
		double r[], 
		double a[4], 
		double c[4], 
		int n1, 
		int n2, 
		int n3, 
		int k);
static void mg3P_gpu(
		MemoryObject* u_device, 
		MemoryObject* v_device, 
		MemoryObject* r_device, 
		MemoryObject a[4], 
		MemoryObject c[4], 
		int n1, 
		int n2, 
		int n3, 
		int k);
static void norm2u3(
		void* pointer_r, 
		int n1, 
		int n2, 
		int n3, 
		double* rnm2, 
		double* rnmu, 
		int nx, 
		int ny, 
		int nz);
static void norm2u3_gpu(
		MemoryObject* r_device, 
		int n1, 
		int n2, 
		int n3, 
		double* rnm2, 
		double* rnmu, 
		int nx, 
		int ny, 
		int nz);
static double power(
		double a, 
		int n);
static void psinv(
		void* pointer_base_r, 
		void* pointer_base_u, 
		int n1, 
		int n2, 
		int n3, 
		double c[4], 
		int k,
		int offset_1,
		int offset_2);
static void psinv_gpu(
		MemoryObject* base_r_device, 
		MemoryObject* base_u_device, 
		int n1, 
		int n2, 
		int n3, 
		MemoryObject* c_device, 
		int k,
		int offset_1,
		int offset_2);
static void rep_nrm(
		void* pointer_u, 
		int n1, 
		int n2, 
		int n3, 
		char* title, 
		int kk);
static void resid(
		void* pointer_base_u, 
		void* pointer_base_v, 
		void* pointer_base_r, 
		int n1, 
		int n2, 
		int n3, 
		double a[4], 
		int k,
		int offset_1,
		int offset_2,
		int offset_3);
static void resid_gpu(
		MemoryObject* base_u_device,
		MemoryObject* base_v_device,
		MemoryObject* base_r_device,
		int n1,
		int n2,
		int n3,
		MemoryObject* a_device,
		int k,
		int offset_1,
		int offset_2,
		int offset_3);
static void rprj3(
		void* pointer_base_r, 
		int m1k, 
		int m2k, 
		int m3k, 
		void* pointer_base_s, 
		int m1j, 
		int m2j, 
		int m3j, 
		int k,
		int offset_1,
		int offset_2);
static void rprj3_gpu(
		MemoryObject* base_r_device, 
		int m1k, 
		int m2k, 
		int m3k, 
		MemoryObject* base_s_device, 
		int m1j, 
		int m2j, 
		int m3j, 
		int k,
		int offset_1,
		int offset_2);
static void setup(
		int* n1, 
		int* n2, 
		int* n3, 
		int k);
static void setup_gpu(
		double* a, 
		double* c);
static void showall(
		void* pointer_z, 
		int n1, 
		int n2, 
		int n3);
static void zero3_gpu(
		MemoryObject* base_z_device, 
		int n1, 
		int n2, 
		int n3,
		int offset);
static void zero3(
		void* pointer_base_z, 
		int n1, 
		int n2, 
		int n3,
		int offset);
static void zran3(
		void* pointer_z, 
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
#ifdef GSPARDRIVER_CUDA
	printf(" Performing GSParLib with CUDA\n");
#else
/* GSPARDRIVER_OPENCL */
	printf(" Performing GSParLib with OpenCL\n");
#endif
	/*
	 * -------------------------------------------------------------------------
	 * k is the current level. it is passed down through subroutine args
	 * and is not global. it is the current iteration
	 * -------------------------------------------------------------------------
	 */
	int k, it;
	double t, tinit, mflops;

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
	FILE* fp;
	fp = fopen("mg.input", "r");
	if(fp != NULL){
		printf(" Reading from input file mg.input\n");
		if(fscanf(fp, "%d", &lt) != 1){
			printf(" Error in reading elements\n");
			exit(1);
		}
		while(fgetc(fp) != '\n');
		if(fscanf(fp, "%d%d%d", &nx[lt], &ny[lt], &nz[lt]) != 3){
			printf(" Error in reading elements\n");
			exit(1);
		}
		while(fgetc(fp) != '\n');
		if(fscanf(fp, "%d", &nit) != 1){
			printf(" Error in reading elements\n");
			exit(1);
		}
		while(fgetc(fp) != '\n');
		for(i = 0; i <= 7; i++) {
			if(fscanf(fp, "%d", &debug_vec[i]) != 1){
				printf(" Error in reading elements\n");
				exit(1);
			}
		}
		fclose(fp);
	}else{
		printf(" No input file. Using compiled defaults\n");
		lt = LT_DEFAULT;
		nit = NIT_DEFAULT;
		nx[lt] = NX_DEFAULT;
		ny[lt] = NY_DEFAULT;
		nz[lt] = NZ_DEFAULT;
		for(i = 0; i <= 7; i++){
			debug_vec[i] = DEBUG_DEFAULT;
		}
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
	
	zero3(u,n1,n2,n3, 0 /*offset*/);
	zran3(v,n1,n2,n3,nx[lt],ny[lt],k);

	norm2u3(v,n1,n2,n3,&rnm2,&rnmu,nx[lt],ny[lt],nz[lt]);	

	printf("\n\n NAS Parallel Benchmarks 4.1 CUDA C++ version - MG Benchmark\n\n");
	printf(" Size: %3dx%3dx%3d (class_npb %1c)\n", nx[lt], ny[lt], nz[lt], class_npb);
	printf(" Iterations: %3d\n", nit);
	
	resid(u,v,r,n1,n2,n3,a,k, 0 /*offset_1*/, 0 /*offset_2*/, 0 /*offset_3*/);
	norm2u3(r,n1,n2,n3,&rnm2,&rnmu,nx[lt],ny[lt],nz[lt]);

	/*
	 * ---------------------------------------------------------------------
	 * one iteration for startup
	 * ---------------------------------------------------------------------
	 */
	mg3P(u,v,r,a,c,n1,n2,n3,k);	
	resid(u,v,r,n1,n2,n3,a,k, 0 /*offset_1*/, 0 /*offset_2*/, 0 /*offset_3*/);

	setup(&n1,&n2,&n3,k);
	
	zero3(u,n1,n2,n3, 0/*offset*/);
	zran3(v,n1,n2,n3,nx[lt],ny[lt],k);

	setup_gpu(a,c);

	timer_clear(PROFILING_TOTAL_TIME);
	timer_start(PROFILING_TOTAL_TIME);
	
	resid_gpu(u_device,v_device,r_device,n1,n2,n3,a_device,k, 0, 0, 0);
	norm2u3_gpu(r_device,n1,n2,n3,&rnm2,&rnmu,nx[lt],ny[lt],nz[lt]);

	for(it = 1; it <= nit; it++){
		//if((it==1)||(it==nit)||((it%5)==0)){printf("  iter %3d\n",it);}
		mg3P_gpu(u_device,v_device,r_device,a_device,c_device,n1,n2,n3,k);
		resid_gpu(u_device,v_device,r_device,n1,n2,n3,a_device,k, 0, 0, 0);
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
static void comm3(void* pointer_base_u, 
		int n1, 
		int n2, 
		int n3, 
		int kk,
		int offset){
	void* pointer_u = pointer_base_u + offset;

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

static void comm3_gpu(
		MemoryObject* base_u_device, 
		int n1, 
		int n2, 
		int n3, 
		int kk,
		int offset){
	threads_per_block = THREADS_PER_BLOCK;
	amount_of_work = (n3-2) * THREADS_PER_BLOCK;
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));

	try {	
		/* kernel_comm3_1 */		
		kernel_comm3_1->clearParameters();
		kernel_comm3_1->setNumThreadsPerBlockForX(threads_per_block);
		kernel_comm3_1->setParameter(base_u_device);		
		kernel_comm3_1->setParameter(sizeof(int), &n1);
		kernel_comm3_1->setParameter(sizeof(int), &n2);
		kernel_comm3_1->setParameter(sizeof(int), &n3);
		kernel_comm3_1->setParameter(sizeof(int), &amount_of_work);
		kernel_comm3_1->setParameter(sizeof(int), &offset);

		unsigned long dimensions_comm3_1[3] = {(unsigned long)amount_of_work, 0, 0};
		kernel_comm3_1->runAsync(dimensions_comm3_1);
		kernel_comm3_1->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	threads_per_block = THREADS_PER_BLOCK;
	amount_of_work = (n3-2) * THREADS_PER_BLOCK;	
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));

	try {	
		/* kernel_comm3_2 */		
		kernel_comm3_2->clearParameters();
		kernel_comm3_2->setNumThreadsPerBlockForX(threads_per_block);
		kernel_comm3_2->setParameter(base_u_device);		
		kernel_comm3_2->setParameter(sizeof(int), &n1);
		kernel_comm3_2->setParameter(sizeof(int), &n2);
		kernel_comm3_2->setParameter(sizeof(int), &n3);
		kernel_comm3_2->setParameter(sizeof(int), &amount_of_work);
		kernel_comm3_2->setParameter(sizeof(int), &offset);

		unsigned long dimensions_comm3_2[3] = {(unsigned long)amount_of_work, 0, 0};
		kernel_comm3_2->runAsync(dimensions_comm3_2);
		kernel_comm3_2->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	threads_per_block = THREADS_PER_BLOCK;
	amount_of_work = n2 * THREADS_PER_BLOCK;
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));

	try {	
		/* kernel_comm3_ */		
		kernel_comm3_3->clearParameters();
		kernel_comm3_3->setNumThreadsPerBlockForX(threads_per_block);
		kernel_comm3_3->setParameter(base_u_device);		
		kernel_comm3_3->setParameter(sizeof(int), &n1);
		kernel_comm3_3->setParameter(sizeof(int), &n2);
		kernel_comm3_3->setParameter(sizeof(int), &n3);
		kernel_comm3_3->setParameter(sizeof(int), &amount_of_work);
		kernel_comm3_3->setParameter(sizeof(int), &offset);

		unsigned long dimensions_comm3_3[3] = {(unsigned long)amount_of_work, 0, 0};
		kernel_comm3_3->runAsync(dimensions_comm3_3);
		kernel_comm3_3->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
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
static void interp(void* pointer_base_z, 
		int mm1, 
		int mm2, 
		int mm3, 
		void* pointer_base_u, 
		int n1, 
		int n2, 
		int n3, 
		int k,
		int offset_1,
		int offset_2){
	void* pointer_z = pointer_base_z + offset_1;
	void* pointer_u = pointer_base_u + offset_2;

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

static void interp_gpu(
		MemoryObject* base_z_device, 
		int mm1, 
		int mm2, 
		int mm3, 
		MemoryObject* base_u_device, 
		int n1, 
		int n2, 
		int n3, 
		int k,
		int offset_1,
		int offset_2){
	if(n1 != 3 && n2 != 3 && n3 != 3){
		threads_per_block = mm1;
		amount_of_work = (mm3-1) * (mm2-1) * mm1;	
		blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));

		try {		
			/* kernel_interp */		
			kernel_interp->clearParameters();
			kernel_interp->setNumThreadsPerBlockForX(threads_per_block);
			kernel_interp->setParameter(base_z_device);
			kernel_interp->setParameter(base_u_device);
			kernel_interp->setParameter(sizeof(int), &mm1);
			kernel_interp->setParameter(sizeof(int), &mm2);
			kernel_interp->setParameter(sizeof(int), &mm3);
			kernel_interp->setParameter(sizeof(int), &n1);
			kernel_interp->setParameter(sizeof(int), &n2);
			kernel_interp->setParameter(sizeof(int), &n3);
			kernel_interp->setParameter(sizeof(int), &amount_of_work);
			kernel_interp->setParameter(sizeof(int), &offset_1);
			kernel_interp->setParameter(sizeof(int), &offset_2);

			unsigned long dimensions_comm3_1[3] = {(unsigned long)amount_of_work, 0, 0};
			kernel_interp->runAsync(dimensions_comm3_1);
			kernel_interp->waitAsync();
		} catch (GSPar::GSParException &ex) {
			std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
			exit(-1);
		}
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
		rprj3(r, m1[k], m2[k], m3[k], r, m1[j], m2[j], m3[j], k, ir[k] /*offset_1*/, ir[j] /*offset_2*/);
	}
	k = lb;
	/*
	 * --------------------------------------------------------------------
	 * compute an approximate solution on the coarsest grid
	 * --------------------------------------------------------------------
	 */
	zero3(u, m1[k], m2[k], m3[k], ir[k] /*offset*/);
	psinv(r, u, m1[k], m2[k], m3[k], c, k, ir[k] /*offset_1*/, ir[k] /*offset_2*/);
	for(k = lb+1; k <= lt-1; k++){
		j = k-1;
		/*
		 * --------------------------------------------------------------------
		 * prolongate from level k-1  to k
		 * -------------------------------------------------------------------
		 */
		zero3(u, m1[k], m2[k], m3[k], ir[k] /*offset*/);
		interp(u, m1[j], m2[j], m3[j], u, m1[k], m2[k], m3[k], k, ir[j] /*offset_1*/, ir[k] /*offset_2*/);
		/*
		 * --------------------------------------------------------------------
		 * compute residual for level k
		 * --------------------------------------------------------------------
		 */
		resid(u, r, r, m1[k], m2[k], m3[k], a, k, ir[k] /*offset_1*/, ir[k] /*offset_2*/, ir[k] /*offset_3*/);
		/*
		 * --------------------------------------------------------------------
		 * apply smoother
		 * --------------------------------------------------------------------
		 */
		psinv(r, u, m1[k], m2[k], m3[k], c, k, ir[k] /*offset_1*/, ir[k] /*offset_2*/);
	}
	j = lt - 1; 
	k = lt;
	interp(u, m1[j], m2[j], m3[j], u, n1, n2, n3, k, ir[j] /*offset_1*/, 0 /*offset_2*/);
	
	resid(u, v, r, n1, n2, n3, a, k, 0 /*offset_1*/, 0 /*offset_2*/, 0 /*offset_3*/);
	psinv(r, u, n1, n2, n3, c, k, 0 /*offset_1*/, 0 /*offset_2*/);
}

static void mg3P_gpu(
		MemoryObject* u_device, 
		MemoryObject* v_device, 
		MemoryObject* r_device, 
		MemoryObject* a_device, 
		MemoryObject* c_device, 
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
		rprj3_gpu(r_device, m1[k], m2[k], m3[k], r_device, m1[j], m2[j], m3[j],	k, ir[k] /*offset_1*/, ir[j] /*offset_2*/);
	}
	k = lb;
	/*
	 * --------------------------------------------------------------------
	 * compute an approximate solution on the coarsest grid
	 * --------------------------------------------------------------------
	 */
	zero3_gpu(u_device, m1[k], m2[k], m3[k], ir[k] /*offset*/);
	psinv_gpu(r_device, u_device, m1[k], m2[k], m3[k], c_device, k, ir[k] /*offset_1*/, ir[k] /*offset_2*/);
	for(k = lb+1; k <= lt-1; k++){
		j = k-1;
		/*
		 * --------------------------------------------------------------------
		 * prolongate from level k-1  to k
		 * -------------------------------------------------------------------
		 */
		zero3_gpu(u_device, m1[k], m2[k], m3[k], ir[k] /*offset*/);
		interp_gpu(u_device, m1[j], m2[j], m3[j], u_device, m1[k], m2[k], m3[k], k, ir[j] /*offset_1*/, ir[k] /*offset_2*/);
		/*
		 * --------------------------------------------------------------------
		 * compute residual for level k
		 * --------------------------------------------------------------------
		 */
		resid_gpu(u_device, r_device, r_device, m1[k], m2[k], m3[k], a_device, k, ir[k], ir[k], ir[k]);
		/*
		 * --------------------------------------------------------------------
		 * apply smoother
		 * --------------------------------------------------------------------
		 */
		psinv_gpu(r_device, u_device, m1[k], m2[k], m3[k], c_device, k, ir[k] /*offset_1*/, ir[k] /*offset_2*/);
	}
	j = lt - 1; 
	k = lt;
	interp_gpu(u_device, m1[j], m2[j], m3[j], u_device, n1, n2, n3, k, ir[j] /*offset_1*/, 0 /*offset_2*/);
	resid_gpu(u_device, v_device, r_device, n1, n2, n3, a_device, k, 0, 0, 0);
	psinv_gpu(r_device, u_device, n1, n2, n3, c_device, k, 0 /*offset_1*/, 0 /*offset_2*/);
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

static void norm2u3_gpu(
		MemoryObject* r_device, 
		int n1, 
		int n2, 
		int n3, 
		double* rnm2, 
		double* rnmu, 
		int nx, 
		int ny, 
		int nz){
	auto gpus = driver->getGpuList();
	DEVICE_NAME = gpus[0]->getName();	
	auto gpu = driver->getGpu(0);	

	double s;
	double dn, max_rnmu;
	int temp_size, j;

	dn=1.0*nx*ny*nz;
	s=0.0;
	max_rnmu=0.0;

	threads_per_block = THREADS_PER_BLOCK;
	amount_of_work = (n2-2) * (n3-2) * threads_per_block;
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));

	temp_size = amount_of_work / threads_per_block;	

	double* sum_host;	
	double* max_host;

	MemoryObject* sum_device;
	MemoryObject* max_device;	

	sum_host=(double*)malloc(temp_size*sizeof(double));
	max_host=(double*)malloc(temp_size*sizeof(double));

	sum_device = gpu->malloc(temp_size*sizeof(double), sum_host);
	max_device = gpu->malloc(temp_size*sizeof(double), max_host);

	try {		
		/* kernel_norm2u3 */		
		kernel_norm2u3->clearParameters();
		kernel_norm2u3->setNumThreadsPerBlockForX(threads_per_block);
		kernel_norm2u3->setParameter(r_device);
		kernel_norm2u3->setParameter(sizeof(int), &n1);
		kernel_norm2u3->setParameter(sizeof(int), &n2);
		kernel_norm2u3->setParameter(sizeof(int), &n3);
		kernel_norm2u3->setParameter(sum_device);
		kernel_norm2u3->setParameter(max_device);
		kernel_norm2u3->setParameter(sizeof(int), &blocks_per_grid);
		kernel_norm2u3->setParameter(sizeof(int), &amount_of_work);

		unsigned long dimensions_norm2u3[3] = {(unsigned long)amount_of_work, 0, 0};
		kernel_norm2u3->runAsync(dimensions_norm2u3);
		kernel_norm2u3->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	sum_device->copyOut();
	max_device->copyOut();

	for(j=0; j<temp_size; j++){
		s=s+sum_host[j];
		if(max_rnmu<max_host[j]){max_rnmu=max_host[j];}
	}

	delete sum_device;
	delete max_device;
	free(sum_host);
	free(max_host);

	*rnmu=max_rnmu;
	*rnm2=sqrt(s/dn);
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
static void psinv(
		void* pointer_base_r, 
		void* pointer_base_u, 
		int n1, 
		int n2, 
		int n3, 
		double c[4], 
		int k,
		int offset_1,
		int offset_2){
	void* pointer_r = pointer_base_r + offset_1;
	void* pointer_u = pointer_base_u + offset_2;

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
	comm3(pointer_base_u, n1, n2, n3, k, offset_2 /*offset*/);

	if(debug_vec[0] >= 1){
		rep_nrm(u,n1,n2,n3,(char*)"   psinv",k);
	}

	if(debug_vec[3] >= k){
		showall(u,n1,n2,n3);
	}
}

static void psinv_gpu(
		MemoryObject* base_r_device, 
		MemoryObject* base_u_device, 
		int n1, 
		int n2, 
		int n3, 
		MemoryObject* c_device, 
		int k,
		int offset_1,
		int offset_2){
	threads_per_block = n1 > THREADS_PER_BLOCK ? THREADS_PER_BLOCK : n1;
	amount_of_work = (n3-2) * (n2-2) * threads_per_block;
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));

	try {	
		/* kernel_psinv */		
		kernel_psinv->clearParameters();
		kernel_psinv->setNumThreadsPerBlockForX(threads_per_block);
		kernel_psinv->setParameter(base_r_device);	
		kernel_psinv->setParameter(base_u_device);	
		kernel_psinv->setParameter(c_device);	
		kernel_psinv->setParameter(sizeof(int), &n1);
		kernel_psinv->setParameter(sizeof(int), &n2);
		kernel_psinv->setParameter(sizeof(int), &n3);
		kernel_psinv->setParameter(sizeof(int), &amount_of_work);
		kernel_psinv->setParameter(sizeof(int), &offset_1);
		kernel_psinv->setParameter(sizeof(int), &offset_2);

		unsigned long dimensions_kernel_psinv[3] = {(unsigned long)amount_of_work, 0, 0};
		kernel_psinv->runAsync(dimensions_kernel_psinv);
		kernel_psinv->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/*
	 * --------------------------------------------------------------------
	 * exchange boundary points
	 * --------------------------------------------------------------------
	 */
	comm3_gpu(base_u_device,n1,n2,n3,k, offset_2);
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
static void resid(
		void* pointer_base_u, 
		void* pointer_base_v, 
		void* pointer_base_r, 
		int n1, 
		int n2, 
		int n3, 
		double a[4], 
		int k,
		int offset_1,
		int offset_2,
		int offset_3){
	void* pointer_u = pointer_base_u + offset_1;
	void* pointer_v = pointer_base_v + offset_2;
	void* pointer_r = pointer_base_r + offset_3;

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
	comm3(pointer_base_r, n1, n2, n3, k, offset_3 /*offset*/);

	if(debug_vec[0] >= 1){
		rep_nrm(r,n1,n2,n3,(char*)"   resid",k);
	}

	if(debug_vec[2] >= k){
		showall(r,n1,n2,n3);
	}
}

static void resid_gpu(
		MemoryObject* base_u_device,
		MemoryObject* base_v_device,
		MemoryObject* base_r_device,
		int n1,
		int n2,
		int n3,
		MemoryObject* a_device,
		int k,
		int offset_1,
		int offset_2,
		int offset_3){
	threads_per_block = n1 > THREADS_PER_BLOCK ? THREADS_PER_BLOCK : n1;
	amount_of_work = (n3-2) * (n2-2) * threads_per_block;
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));

	try {
		/* kernel_resid */		
		kernel_resid->clearParameters();
		kernel_resid->setNumThreadsPerBlockForX(threads_per_block);
		kernel_resid->setParameter(base_u_device);
		kernel_resid->setParameter(base_v_device);
		kernel_resid->setParameter(base_r_device);
		kernel_resid->setParameter(a_device);
		kernel_resid->setParameter(sizeof(int), &n1);
		kernel_resid->setParameter(sizeof(int), &n2);
		kernel_resid->setParameter(sizeof(int), &n3);
		kernel_resid->setParameter(sizeof(int), &amount_of_work);
		kernel_resid->setParameter(sizeof(int), &offset_1);
		kernel_resid->setParameter(sizeof(int), &offset_2);
		kernel_resid->setParameter(sizeof(int), &offset_3);

		unsigned long dimensions_resid[3] = {(unsigned long)amount_of_work, 0, 0};
		kernel_resid->runAsync(dimensions_resid);
		kernel_resid->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	/*
	 * --------------------------------------------------------------------
	 * exchange boundary data
	 * --------------------------------------------------------------------
	 */
	comm3_gpu(base_r_device, n1, n2, n3, k, offset_3);
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
static void rprj3(
		void* pointer_base_r, 
		int m1k, 
		int m2k, 
		int m3k, 
		void* pointer_base_s, 
		int m1j, 
		int m2j, 
		int m3j, 
		int k,
		int offset_1,
		int offset_2){
	void* pointer_r = pointer_base_r + offset_1; 
	void* pointer_s = pointer_base_s + offset_2;

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
	comm3(pointer_base_s, m1j, m2j, m3j, j, offset_2 /*offset*/);

	if(debug_vec[0] >= 1){
		rep_nrm(s,m1j,m2j,m3j,(char*)"   rprj3",k-1);	
	}

	if(debug_vec[4] >= k){
		showall(s,m1j,m2j,m3j);
	}
}

static void rprj3_gpu(
		MemoryObject* base_r_device, 
		int m1k, 
		int m2k, 
		int m3k, 
		MemoryObject* base_s_device, 
		int m1j, 
		int m2j, 
		int m3j, 
		int k,
		int offset_1,
		int offset_2){
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

	threads_per_block = m1j-1;
	amount_of_work = (m3j-2) * (m2j-2) * (m1j-1);
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));

	try {		
		/* kernel_rprj3 */		
		kernel_rprj3->clearParameters();
		kernel_rprj3->setNumThreadsPerBlockForX(threads_per_block);
		kernel_rprj3->setParameter(base_r_device);
		kernel_rprj3->setParameter(base_s_device);
		kernel_rprj3->setParameter(sizeof(int), &m1k);
		kernel_rprj3->setParameter(sizeof(int), &m2k);
		kernel_rprj3->setParameter(sizeof(int), &m3k);
		kernel_rprj3->setParameter(sizeof(int), &m1j);
		kernel_rprj3->setParameter(sizeof(int), &m2j);
		kernel_rprj3->setParameter(sizeof(int), &m3j);
		kernel_rprj3->setParameter(sizeof(int), &d1);
		kernel_rprj3->setParameter(sizeof(int), &d2);
		kernel_rprj3->setParameter(sizeof(int), &d3);
		kernel_rprj3->setParameter(sizeof(int), &amount_of_work);
		kernel_rprj3->setParameter(sizeof(int), &offset_1);
		kernel_rprj3->setParameter(sizeof(int), &offset_2);

		unsigned long dimensions_resid[3] = {(unsigned long)amount_of_work, 0, 0};
		kernel_rprj3->runAsync(dimensions_resid);
		kernel_rprj3->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
	
	j=k-1;
	comm3_gpu(base_s_device,m1j,m2j,m3j,j, offset_2);
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

static void setup_gpu(
		double* a, 
		double* c){
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

	size_a_device=sizeof(double)*(4);
	size_c_device=sizeof(double)*(4);
	size_u_device=sizeof(double)*(NR);
	size_v_device=sizeof(double)*(NV);
	size_r_device=sizeof(double)*(NR);

	a_device = gpu->malloc(size_a_device, a);
	c_device = gpu->malloc(size_c_device, c);
	u_device = gpu->malloc(size_u_device, u);
	v_device = gpu->malloc(size_v_device, v);
	r_device = gpu->malloc(size_r_device, r);		

	a_device->copyIn();
	c_device->copyIn();
	u_device->copyIn();
	v_device->copyIn();
	r_device->copyIn();	

	source_additional_routines_complete = source_additional_routines + std::to_string(M) + "\n";

	try {
		std::string kernel_source_comm3_1_complete = "\n";
		kernel_source_comm3_1_complete.append(source_additional_routines_complete);
		kernel_source_comm3_1_complete.append(kernel_source_comm3_1);
		kernel_comm3_1 = new Kernel(gpu, kernel_source_comm3_1_complete, "comm3_gpu_kernel_1");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_comm3_2_complete = "\n";
		kernel_source_comm3_2_complete.append(source_additional_routines_complete);
		kernel_source_comm3_2_complete.append(kernel_source_comm3_2);
		kernel_comm3_2 = new Kernel(gpu, kernel_source_comm3_2_complete, "comm3_gpu_kernel_2");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_comm3_3_complete = "\n";
		kernel_source_comm3_3_complete.append(source_additional_routines_complete);
		kernel_source_comm3_3_complete.append(kernel_source_comm3_3);
		kernel_comm3_3 = new Kernel(gpu, kernel_source_comm3_3_complete, "comm3_gpu_kernel_3");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_interp_complete = "\n";
		kernel_source_interp_complete.append(source_additional_routines_complete);
		kernel_source_interp_complete.append(kernel_source_interp);
		kernel_interp = new Kernel(gpu, kernel_source_interp_complete, "interp_gpu_kernel");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_norm2u3_complete = "\n";
		kernel_source_norm2u3_complete.append(source_additional_routines_complete);
		kernel_source_norm2u3_complete.append(kernel_source_norm2u3);
		kernel_norm2u3 = new Kernel(gpu, kernel_source_norm2u3_complete, "norm2u3_gpu_kernel");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_psinv_complete = "\n";
		kernel_source_psinv_complete.append(source_additional_routines_complete);
		kernel_source_psinv_complete.append(kernel_source_psinv);
		kernel_psinv = new Kernel(gpu, kernel_source_psinv_complete, "psinv_gpu_kernel");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}	

	try {
		std::string kernel_source_resid_complete = "\n";
		kernel_source_resid_complete.append(source_additional_routines_complete);
		kernel_source_resid_complete.append(kernel_source_resid);
		kernel_resid = new Kernel(gpu, kernel_source_resid_complete, "resid_gpu_kernel");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	try {
		std::string kernel_source_rprj3_complete = "\n";
		kernel_source_rprj3_complete.append(source_additional_routines_complete);
		kernel_source_rprj3_complete.append(kernel_source_rprj3);
		kernel_rprj3 = new Kernel(gpu, kernel_source_rprj3_complete, "rprj3_gpu_kernel");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	try {
		std::string kernel_source_zero3_complete = "\n";
		kernel_source_zero3_complete.append(source_additional_routines_complete);
		kernel_source_zero3_complete.append(kernel_source_zero3);
		kernel_zero3 = new Kernel(gpu, kernel_source_zero3_complete, "zero3_gpu_kernel");
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
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

static void zero3(void* pointer_base_z, 
		int n1, 
		int n2, 
		int n3,
		int offset){
	void* pointer_z = pointer_base_z + offset;

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

static void zero3_gpu(MemoryObject* base_z_device, 
		int n1, 
		int n2, 
		int n3,
		int offset){
	threads_per_block = THREADS_PER_BLOCK;
	amount_of_work = n1*n2*n3;	
	blocks_per_grid = (ceil((double)(amount_of_work)/(double)(threads_per_block)));

	try {		
		/* kernel_zero3 */		
		kernel_zero3->clearParameters();
		kernel_zero3->setNumThreadsPerBlockForX(threads_per_block);
		kernel_zero3->setParameter(base_z_device);		
		kernel_zero3->setParameter(sizeof(int), &n1);
		kernel_zero3->setParameter(sizeof(int), &n2);
		kernel_zero3->setParameter(sizeof(int), &n3);
		kernel_zero3->setParameter(sizeof(int), &amount_of_work);
		kernel_zero3->setParameter(sizeof(int), &offset);

		unsigned long dimensions_zero3[3] = {(unsigned long)amount_of_work, 0, 0};
		kernel_zero3->runAsync(dimensions_zero3);
		kernel_zero3->waitAsync();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
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
	
	zero3(z, n1, n2, n3, 0 /*offset*/);

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
	comm3(z, n1, n2, n3, k, 0 /*offset*/);
}

std::string kernel_source_comm3_1 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void comm3_gpu_kernel_1(
		GSPAR_DEVICE_GLOBAL_MEMORY double* base_u, 
		int n1, 
		int n2, 
		int n3, 
		int amount_of_work,
		int offset){
	// BEGIN

	int check=gspar_get_global_id(0);
	if(check>=amount_of_work){return;}

	GSPAR_DEVICE_GLOBAL_MEMORY double* u = base_u + offset;	

	int i3=gspar_get_block_id(0)+1;
	int i2=gspar_get_thread_id(0)+1;

	while(i2<n2-1){
		u[i3*n2*n1+i2*n1+0]=u[i3*n2*n1+i2*n1+n1-2];
		u[i3*n2*n1+i2*n1+n1-1]=u[i3*n2*n1+i2*n1+1];
		i2+=gspar_get_block_size(0);
	}

	// END
});

std::string kernel_source_comm3_2 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void comm3_gpu_kernel_2(
		GSPAR_DEVICE_GLOBAL_MEMORY double* base_u,
		int n1,
		int n2,
		int n3,
		int amount_of_work,
		int offset){
	// BEGIN

	int check=gspar_get_global_id(0);
	if(check>=amount_of_work){return;}

	GSPAR_DEVICE_GLOBAL_MEMORY double* u = base_u + offset;

	int i3=gspar_get_block_id(0) + 1;
	int i1=gspar_get_thread_id(0);

	while(i1<n1){
		u[i3*n2*n1+0*n1+i1]=u[i3*n2*n1+(n2-2)*n1+i1];
		u[i3*n2*n1+(n2-1)*n1+i1]=u[i3*n2*n1+1*n1+i1];
		i1+=gspar_get_block_size(0);
	}

	// END
});

std::string kernel_source_comm3_3 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void comm3_gpu_kernel_3(
		GSPAR_DEVICE_GLOBAL_MEMORY double* base_u,
		int n1, 
		int n2, 
		int n3, 
		int amount_of_work,
		int offset){
	// BEGIN

	int check=gspar_get_global_id(0);
	if(check>=amount_of_work){return;}

	GSPAR_DEVICE_GLOBAL_MEMORY double* u = base_u + offset;

	int i2=gspar_get_block_id(0);
	int i1=gspar_get_thread_id(0);

	while(i1<n1){
		u[0*n2*n1+i2*n1+i1]=u[(n3-2)*n2*n1+i2*n1+i1];
		u[(n3-1)*n2*n1+i2*n1+i1]=u[1*n2*n1+i2*n1+i1];
		i1+=gspar_get_block_size(0);
	}

	// END
});

std::string kernel_source_interp = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void interp_gpu_kernel(
		GSPAR_DEVICE_GLOBAL_MEMORY double* base_z,
		GSPAR_DEVICE_GLOBAL_MEMORY double* base_u,
		int mm1, 
		int mm2, 
		int mm3,
		int n1, 
		int n2, 
		int n3,
		int amount_of_work,
		int offset_1,
		int offset_2){
	// BEGIN

	int check=gspar_get_global_id(0);
	if(check>=amount_of_work){return;}	

	int i3,i2,i1;
	
	GSPAR_DEVICE_SHARED_MEMORY double z1[M],z2[M],z3[M];

	GSPAR_DEVICE_GLOBAL_MEMORY double (*z) = base_z + offset_1;
	GSPAR_DEVICE_GLOBAL_MEMORY double (*u) = base_u + offset_2;

	i3=gspar_get_block_id(0)/(mm2-1);
	i2=gspar_get_block_id(0)%(mm2-1);
	i1=gspar_get_thread_id(0);

	z1[i1]=z[i3*mm2*mm1+(i2+1)*mm1+i1]+z[i3*mm2*mm1+i2*mm1+i1];
	z2[i1]=z[(i3+1)*mm2*mm1+i2*mm1+i1]+z[i3*mm2*mm1+i2*mm1+i1];
	z3[i1]=z[(i3+1)*mm2*mm1+(i2+1)*mm1+i1] 
		+z[(i3+1)*mm2*mm1+i2*mm1+i1]+z1[i1];

	gspar_synchronize_local_threads();
	if(i1<mm1-1){
		double z321=z[i3*mm2*mm1+i2*mm1+i1];
		u[2*i3*n2*n1+2*i2*n1+2*i1]+=z321;
		u[2*i3*n2*n1+2*i2*n1+2*i1+1]+=0.5*(z[i3*mm2*mm1+i2*mm1+i1+1]+z321);
		u[2*i3*n2*n1+(2*i2+1)*n1+2*i1]+=0.5*z1[i1];
		u[2*i3*n2*n1+(2*i2+1)*n1+2*i1+1]+=0.25*(z1[i1]+z1[i1+1]);
		u[(2*i3+1)*n2*n1+2*i2*n1+2*i1]+=0.5*z2[i1];
		u[(2*i3+1)*n2*n1+2*i2*n1+2*i1+1]+=0.25*(z2[i1]+z2[i1+1]);
		u[(2*i3+1)*n2*n1+(2*i2+1)*n1+2*i1]+=0.25*z3[i1];
		u[(2*i3+1)*n2*n1+(2*i2+1)*n1+2*i1+1]+=0.125*(z3[i1]+z3[i1+1]);
	}

	// END
});

std::string kernel_source_norm2u3 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void norm2u3_gpu_kernel(
		GSPAR_DEVICE_GLOBAL_MEMORY double* r,
		const int n1, 
		const int n2, 
		const int n3,
		GSPAR_DEVICE_GLOBAL_MEMORY double* res_sum,
		GSPAR_DEVICE_GLOBAL_MEMORY double* res_max,
		int number_of_blocks,
		int amount_of_work){
	// BEGIN 

	int check=gspar_get_global_id(0);
	if(check>=amount_of_work){return;}
	
	GSPAR_DEVICE_SHARED_MEMORY double scratch_sum[THREADS_PER_BLOCK];
	GSPAR_DEVICE_SHARED_MEMORY double scratch_max[THREADS_PER_BLOCK];

	int i3=gspar_get_block_id(0)/(n2-2)+1;
	int i2=gspar_get_block_id(0)%(n2-2)+1;
	int i1=gspar_get_thread_id(0)+1;

	double s=0.0;
	double my_rnmu=0.0;
	double a;

	while(i1<n1-1){
		double r321=r[i3*n2*n1+i2*n1+i1];
		s=s+r321*r321;
		a=fabs(r321);
		my_rnmu=(a>my_rnmu)?a:my_rnmu;
		i1+=gspar_get_block_size(0);
	}

	int lid=gspar_get_thread_id(0);
	scratch_sum[lid]=s;
	scratch_max[lid]=my_rnmu;

	gspar_synchronize_local_threads();
	for(int i=gspar_get_block_size(0)/2; i>0; i>>=1){
		if(lid<i){
			scratch_sum[lid]+=scratch_sum[lid+i];
			scratch_max[lid]=(scratch_max[lid]>scratch_max[lid+i])?scratch_max[lid]:scratch_max[lid+i];
		}
		gspar_synchronize_local_threads();
	}
	if(lid == 0){
		int idx=gspar_get_block_id(0);
		res_sum[idx]=scratch_sum[0];
		res_max[idx]=scratch_max[0];
	}

	// END
});

std::string kernel_source_psinv = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void psinv_gpu_kernel(
		GSPAR_DEVICE_GLOBAL_MEMORY double* base_r,
		GSPAR_DEVICE_GLOBAL_MEMORY double* base_u,
		GSPAR_DEVICE_GLOBAL_MEMORY double* c,
		int n1,
		int n2,
		int n3,
		int amount_of_work,
		int offset_1,
		int offset_2){
	// BEGIN

	int check=gspar_get_global_id(0);
	if(check>=amount_of_work){return;}

	GSPAR_DEVICE_GLOBAL_MEMORY double* r = base_r + offset_1;
	GSPAR_DEVICE_GLOBAL_MEMORY double* u = base_u + offset_2;
	
	GSPAR_DEVICE_SHARED_MEMORY double r1[M],r2[M];

	int i3=gspar_get_block_id(0)/(n2-2)+1;
	int i2=gspar_get_block_id(0)%(n2-2)+1;
	int lid=gspar_get_thread_id(0);
	int i1;	

	for(i1=lid; i1<n1; i1+=gspar_get_block_size(0)){
		r1[i1]=r[i3*n2*n1+(i2-1)*n2+i1]
			+r[i3*n2*n1+(i2+1)*n1+i1]
			+r[(i3-1)*n2*n1+i2*n1+i1]
			+r[(i3+1)*n2*n1+i2*n1+i1];
		r2[i1]=r[(i3-1)*n2*n1+(i2-1)*n1+i1]
			+r[(i3-1)*n2*n1+(i2+1)*n1+i1]
			+r[(i3+1)*n2*n1+(i2-1)*n1+i1]
			+r[(i3+1)*n2*n1+(i2+1)*n1+i1];
	} gspar_synchronize_local_threads();
	for(i1=lid+1; i1<n1-1; i1+=gspar_get_block_size(0)){
		u[i3*n2*n1+i2*n1+i1]=u[i3*n2*n1+i2*n1+i1]
			+c[0]*r[i3*n2*n1+i2*n1+i1]
			+c[1]*(r[i3*n2*n1+i2*n1+i1-1]
					+r[i3*n2*n1+i2*n1+i1+1]
					+r1[i1])
			+c[2]*(r2[i1]+r1[i1-1]+r1[i1+1] );
	}

	// END
});

std::string kernel_source_resid = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void resid_gpu_kernel(
		GSPAR_DEVICE_GLOBAL_MEMORY double* base_u,
		GSPAR_DEVICE_GLOBAL_MEMORY double* base_v,
		GSPAR_DEVICE_GLOBAL_MEMORY double* base_r,
		GSPAR_DEVICE_GLOBAL_MEMORY double* a,
		int n1,
		int n2,
		int n3,
		int amount_of_work,
		int offset_1,
		int offset_2,
		int offset_3){
	// BEGIN

	int check=gspar_get_global_id(0);
	if(check>=amount_of_work){return;}

	GSPAR_DEVICE_GLOBAL_MEMORY double* u = base_u + offset_1;
	GSPAR_DEVICE_GLOBAL_MEMORY double* v = base_v + offset_2;
	GSPAR_DEVICE_GLOBAL_MEMORY double* r = base_r + offset_3;
	
	GSPAR_DEVICE_SHARED_MEMORY double u1[M], u2[M];

	int i3=gspar_get_block_id(0)/(n2-2)+1;
	int i2=gspar_get_block_id(0)%(n2-2)+1;
	int lid=gspar_get_thread_id(0);
	int i1;

	for(i1=lid; i1<n1; i1+=gspar_get_block_size(0)){
		u1[i1]=u[i3*n2*n1+(i2-1)*n1+i1]
			+u[i3*n2*n1+(i2+1)*n1+i1]
			+u[(i3-1)*n2*n1+i2*n1+i1]
			+u[(i3+1)*n2*n1+i2*n1+i1];
		u2[i1]=u[(i3-1)*n2*n1+(i2-1)*n1+i1]
			+u[(i3-1)*n2*n1+(i2+1)*n1+i1]
			+u[(i3+1)*n2*n1+(i2-1)*n1+i1]
			+u[(i3+1)*n2*n1+(i2+1)*n1+i1];
	} gspar_synchronize_local_threads();
	for(i1=lid+1; i1<n1-1; i1+=gspar_get_block_size(0)){
		r[i3*n2*n1+i2*n1+i1]=v[i3*n2*n1+i2*n1+i1]
			-a[0]*u[i3*n2*n1+i2*n1+i1]
			-a[2]*(u2[i1]+u1[i1-1]+u1[i1+1])
			-a[3]*(u2[i1-1]+u2[i1+1] );
	}

	// END
});

std::string kernel_source_rprj3 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void rprj3_gpu_kernel(
		GSPAR_DEVICE_GLOBAL_MEMORY double* base_r,
		GSPAR_DEVICE_GLOBAL_MEMORY double* base_s,
		int m1k,
		int m2k,
		int m3k,
		int m1j,
		int m2j,
		int m3j,
		int d1, 
		int d2, 
		int d3,
		int amount_of_work,
		int offset_1,
		int offset_2){
	// BEGIN

	int check=gspar_get_global_id(0);
	if(check>=amount_of_work){return;}

	GSPAR_DEVICE_GLOBAL_MEMORY double* r = base_r + offset_1;
	GSPAR_DEVICE_GLOBAL_MEMORY double* s = base_s + offset_2;

	int j3,j2,j1,i3,i2,i1;
	double x2,y2;
	
	GSPAR_DEVICE_SHARED_MEMORY double x1[M],y1[M];

	j3=gspar_get_block_id(0)/(m2j-2)+1;
	j2=gspar_get_block_id(0)%(m2j-2)+1;
	j1=gspar_get_thread_id(0)+1;

	i3=2*j3-d3;
	i2=2*j2-d2;
	i1=2*j1-d1;
	x1[i1]=r[(i3+1)*m2k*m1k+i2*m1k+i1]
		+r[(i3+1)*m2k*m1k+(i2+2)*m1k+i1]
		+r[i3*m2k*m1k+(i2+1)*m1k+i1]
		+r[(i3+2)*m2k*m1k+(i2+1)*m1k+i1];
	y1[i1]=r[i3*m2k*m1k+i2*m1k+i1]
		+r[(i3+2)*m2k*m1k+i2*m1k+i1]
		+r[i3*m2k*m1k+(i2+2)*m1k+i1]
		+r[(i3+2)*m2k*m1k+(i2+2)*m1k+i1];		
	gspar_synchronize_local_threads();
	if(j1<m1j-1){
		i1=2*j1-d1;
		y2=r[i3*m2k*m1k+i2*m1k+i1+1]
			+r[(i3+2)*m2k*m1k+i2*m1k+i1+1]
			+r[i3*m2k*m1k+(i2+2)*m1k+i1+1]
			+r[(i3+2)*m2k*m1k+(i2+2)*m1k+i1+1];
		x2=r[(i3+1)*m2k*m1k+i2*m1k+i1+1]
			+r[(i3+1)*m2k*m1k+(i2+2)*m1k+i1+1]
			+r[i3*m2k*m1k+(i2+1)*m1k+i1+1]
			+r[(i3+2)*m2k*m1k+(i2+1)*m1k+i1+1];
		s[j3*m2j*m1j+j2*m1j+j1]=
			0.5*r[(i3+1)*m2k*m1k+(i2+1)*m1k+i1+1]
			+0.25*(r[(i3+1)*m2k*m1k+(i2+1)*m1k+i1]
					+r[(i3+1)*m2k*m1k+(i2+1)*m1k+i1+2]+x2)
			+0.125*(x1[i1]+x1[i1+2]+y2)
			+0.0625*(y1[i1]+y1[i1+2]);
	}

	// END
});

std::string kernel_source_zero3 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_KERNEL void zero3_gpu_kernel(
		GSPAR_DEVICE_GLOBAL_MEMORY double* base_z, 
		int n1, 
		int n2, 
		int n3, 
		int amount_of_work,
		int offset){
	// BEGIN

	int thread_id=gspar_get_global_id(0);
	if(thread_id>=(n1*n2*n3)){return;}
	GSPAR_DEVICE_GLOBAL_MEMORY double* z = base_z + offset;
	z[thread_id]=0.0;

	// END
});

std::string source_additional_routines_complete = "\n";

std::string source_additional_routines = 
"\n"
"#define WARP_SIZE 32\n"
"#define MAX_THREADS_PER_BLOCK 1024\n"
"#define THREADS_PER_BLOCK 1024\n"
"\n"
"#define M ";
