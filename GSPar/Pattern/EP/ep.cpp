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
 *      P. O. Frederickson
 *      D. H. Bailey
 *      A. C. Woo
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

#include "GSPar_PatternMap.hpp"
#include "GSPar_PatternReduce.hpp"
#include "GSPar_PatternComposition.hpp"

#if defined(GSPARDRIVER_CUDA)
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
 * --------------------------------------------------------------------
 * this is the serial version of the app benchmark 1,
 * the "embarassingly parallel" benchmark.
 * --------------------------------------------------------------------
 * M is the Log_2 of the number of complex pairs of uniform (0, 1) random
 * numbers. MK is the Log_2 of the size of each batch of uniform random
 * numbers.  MK can be set for convenience on a given system, since it does
 * not affect the results.
 * --------------------------------------------------------------------
 */
#define	MK (16)
#define	MM (M - MK)
#define	NN (1 << MM)
#define	NK (1 << MK)
#define	NQ (10)
#define EPSILON (1.0e-8)
#define	A (1220703125.0)
#define	S (271828183.0)
#define NK_PLUS ((2*NK)+1)
#define PROFILING_TOTAL_TIME (0)
#define THREADS_PER_BLOCK (32)

/* global variables */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static double x[NK_PLUS];
static double q[NQ];
#else
static double (*x)=(double*)malloc(sizeof(double)*(NK_PLUS));
static double (*q)=(double*)malloc(sizeof(double)*(NQ));
#endif

/* gpu data */
string DEVICE_NAME;
Instance* driver;
int amount_of_work;
int threads_per_block;
int blocks_per_grid;
size_t q_size;
size_t sx_size;
size_t sy_size;
double* q_host;
double* sx_host;
double* sy_host;
MemoryObject* q_device;
MemoryObject* sx_device;
MemoryObject* sy_device;
Map* kernel_ep; 
extern std::string source_kernel_ep;
extern std::string source_additional_routines_complete;
extern std::string source_additional_routines_1;
extern std::string source_additional_routines_2;

/* function prototypes */
static void setup_gpu();

/* ep */
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
	double  Mops, t1, t2, t3, t4, x1, x2;
	double  sx, sy, tm, an, tt, gc;
	double  sx_verify_value, sy_verify_value, sx_err, sy_err;
	int     np;
	int     i, ik, kk, l, k, nit;
	int     k_offset, j;
	boolean verified;
	double  dum[3] = {1.0, 1.0, 1.0};
	char    size[16];

	/*
	 * --------------------------------------------------------------------
	 * because the size of the problem is too large to store in a 32-bit
	 * integer for some classes, we put it into a string (for printing).
	 * have to strip off the decimal point put in there by the floating
	 * point print statement (internal file)
	 * --------------------------------------------------------------------
	 */    
	sprintf(size, "%15.0f", pow(2.0, M+1));
	j = 14;
	if(size[j]=='.'){j--;}
	size[j+1] = '\0';
	printf("\n\n NAS Parallel Benchmarks 4.1 Serial C++ version - EP Benchmark\n\n");
	printf(" Number of random numbers generated: %15s\n", size);

	verified = FALSE;

	/*
	 * --------------------------------------------------------------------
	 * compute the number of "batches" of random number pairs generated 
	 * per processor. Adjust if the number of processors does not evenly 
	 * divide the total number
	 * --------------------------------------------------------------------
	 */
	np = NN;

	/*
	 * call the random number generator functions and initialize
	 * the x-array to reduce the effects of paging on the timings.
	 * also, call all mathematical functions that are used. make
	 * sure these initializations cannot be eliminated as dead code.
	 */
	vranlc(0, &dum[0], dum[1], &dum[2]);
	dum[0] = randlc(&dum[1], dum[2]);
	for(i=0; i<NK_PLUS; i++){x[i] = -1.0e99;}
	Mops = log(sqrt(fabs(npb_max(1.0, 1.0))));

	t1 = A;
	vranlc(0, &t1, A, x);

	/* compute AN = A ^ (2 * NK) (mod 2^46) */

	t1 = A;

	for(i=0; i<MK+1; i++){
		t2 = randlc(&t1, t1);
	}

	an = t1;
	tt = S;
	gc = 0.0;
	sx = 0.0;
	sy = 0.0;

	for(i=0; i<=NQ-1; i++){
		q[i] = 0.0;
	}

	/*
	 * each instance of this loop may be performed independently. we compute
	 * the k offsets separately to take into account the fact that some nodes
	 * have more numbers to generate than others
	 */
	k_offset = 0;

	setup_gpu();

	timer_clear(PROFILING_TOTAL_TIME);
	timer_start(PROFILING_TOTAL_TIME);

	/* kernel_ep */
	try {
		kernel_ep->setParameter<double*>("q_global", q_device, GSPAR_PARAM_PRESENT);
		kernel_ep->setParameter<double*>("sx_global", sx_device, GSPAR_PARAM_PRESENT);
		kernel_ep->setParameter<double*>("sy_global", sy_device, GSPAR_PARAM_PRESENT);
		kernel_ep->setParameter("an", an);
		kernel_ep->setParameter("NK", NK);

		kernel_ep->run<Instance>();
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}

	q_device->copyOut();
	sx_device->copyOut();
	sy_device->copyOut();

	for(int block=0; block<blocks_per_grid; block++){
		for(int i=0; i<NQ; i++){
			q[i]+=q_host[block*NQ+i];
		}
		sx+=sx_host[block];
		sy+=sy_host[block];
	}
	for(int i=0; i<=NQ-1; i++){
		gc = gc + q[i];
	}

	timer_stop(PROFILING_TOTAL_TIME);
	tm = timer_read(PROFILING_TOTAL_TIME);

	nit = 0;
	verified = TRUE;
	if(M == 24){
		sx_verify_value = -3.247834652034740e+3;
		sy_verify_value = -6.958407078382297e+3;
	}else if(M == 25){
		sx_verify_value = -2.863319731645753e+3;
		sy_verify_value = -6.320053679109499e+3;
	}else if(M == 28){
		sx_verify_value = -4.295875165629892e+3;
		sy_verify_value = -1.580732573678431e+4;
	}else if(M == 30){
		sx_verify_value =  4.033815542441498e+4;
		sy_verify_value = -2.660669192809235e+4;
	}else if(M == 32){
		sx_verify_value =  4.764367927995374e+4;
		sy_verify_value = -8.084072988043731e+4;
	}else if(M == 36){
		sx_verify_value =  1.982481200946593e+5;
		sy_verify_value = -1.020596636361769e+5;
	}else if (M == 40){
		sx_verify_value = -5.319717441530e+05;
		sy_verify_value = -3.688834557731e+05;
	}else{
		verified = FALSE;
	}
	if(verified){
		sx_err = fabs((sx - sx_verify_value) / sx_verify_value);
		sy_err = fabs((sy - sy_verify_value) / sy_verify_value);
		verified = ((sx_err <= EPSILON) && (sy_err <= EPSILON));
	}
	Mops = pow(2.0, M+1)/tm/1000000.0;

	printf("\n EP Benchmark Results:\n\n");
	printf(" CPU Time =%10.4f\n", tm);
	printf(" N = 2^%5d\n", M);
	printf(" No. Gaussian Pairs = %15.0f\n", gc);
	printf(" Sums = %25.15e %25.15e\n", sx, sy);
	printf(" Counts: \n");
	for(i=0; i<NQ-1; i++){
		printf("%3d%15.0f\n", i, q[i]);
	}

	c_print_results((char*)"EP",
			CLASS,
			M+1,
			0,
			0,
			nit,
			tm,
			Mops,
			(char*)"Random numbers generated",
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

static void setup_gpu(){
	Instance* driver = Instance::getInstance();
	driver->init();

	int numGpus = driver->getGpuCount();
	if (numGpus == 0) {
		std::cout << "No GPU found, interrupting the benchmark" << std::endl;
		exit(-1);
	}

	auto gpus = driver->getGpuList();

	DEVICE_NAME = gpus[0]->getName();	

	auto gpu = driver->getGpu(0);

	amount_of_work = NN;

	threads_per_block = THREADS_PER_BLOCK;

	blocks_per_grid = NN / threads_per_block;
	if(blocks_per_grid == 0){blocks_per_grid=1;}

	q_size = blocks_per_grid * NQ * sizeof(double);
	sx_size = blocks_per_grid * sizeof(double);
	sy_size = blocks_per_grid * sizeof(double);

	q_host=(double*)malloc(q_size);
	sx_host=(double*)malloc(sx_size);
	sy_host=(double*)malloc(sy_size);

	q_device = gpu->malloc(q_size, q_host);
	sx_device = gpu->malloc(sx_size, sx_host);
	sy_device = gpu->malloc(sy_size, sy_host);

	for(int block=0; block<blocks_per_grid; block++){
		for(int i=0; i<NQ; i++){
			q_host[block*NQ+i]=0.0;
		}
		sx_host[block]=0.0;
		sy_host[block]=0.0;
	}

	q_device->copyIn();
	sx_device->copyIn();
	sy_device->copyIn();	

	source_additional_routines_complete.append(source_additional_routines_1);
	source_additional_routines_complete.append(source_additional_routines_2);

	/* kernel_ep */
	double an = 0.0;
	try {
		unsigned long dims[3] = {(long unsigned int)amount_of_work, 0, 0}; 

		kernel_ep = new Map(source_kernel_ep);

		kernel_ep->setStdVarNames({"gspar_thread_id"});			

		kernel_ep->setParameter<double*>("q_global", q_device, GSPAR_PARAM_PRESENT);
		kernel_ep->setParameter<double*>("sx_global", sx_device, GSPAR_PARAM_PRESENT);
		kernel_ep->setParameter<double*>("sy_global", sy_device, GSPAR_PARAM_PRESENT);
		kernel_ep->setParameter("an", an);
		kernel_ep->setParameter("NK", NK);

		kernel_ep->setNumThreadsPerBlockForX(threads_per_block);

		kernel_ep->addExtraKernelCode(source_additional_routines_complete);

		kernel_ep->compile<Instance>(dims);
	} catch (GSPar::GSParException &ex) {
		std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
		exit(-1);
	}
}

std::string source_additional_routines_complete = "\n";

std::string source_additional_routines_1 = 
"\n"
"#define THREADS_PER_BLOCK 32\n"
"#define NQ 10\n"
"#define RECOMPUTING 128\n"
"\n"
"\n"
"\n";

std::string source_additional_routines_2 = GSPAR_STRINGIZE_SOURCE(
GSPAR_DEVICE_CONSTANT double A = (1220703125.0);
GSPAR_DEVICE_CONSTANT double S = (271828183.0);
GSPAR_DEVICE_CONSTANT double R23 = (0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5);
GSPAR_DEVICE_CONSTANT double R46 = ((0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5)*(0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5));
GSPAR_DEVICE_CONSTANT double T23 = (2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0);
GSPAR_DEVICE_CONSTANT double T46 = ((2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0)*(2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0));
double pow2(double a){return(a*a);}
GSPAR_DEVICE_FUNCTION double randlc_device(double* x, double a){
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
GSPAR_DEVICE_FUNCTION void vranlc_device(int n, double* x_seed, double a, double* y){
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

std::string source_kernel_ep = GSPAR_STRINGIZE_SOURCE(
	// BEGIN 		

	double x_local[2*RECOMPUTING];
	double q_local[NQ]; 
	double sx_local, sy_local;
	double t1, t2, t3, t4, x1, x2, seed;
	int i, ii, ik, kk, l;

	q_local[0]=0.0;
	q_local[1]=0.0;
	q_local[2]=0.0;
	q_local[3]=0.0;
	q_local[4]=0.0;
	q_local[5]=0.0;
	q_local[6]=0.0;
	q_local[7]=0.0;
	q_local[8]=0.0;
	q_local[9]=0.0;
	sx_local=0.0;
	sy_local=0.0;	

	kk=gspar_get_global_id(0);

	t1=S;
	t2=an;

	for(i=1; i<=100; i++){
		ik=kk/2;
		if((2*ik)!=kk){t3=randlc_device(&t1, t2);}
		if(ik==0){break;}
		t3=randlc_device(&t2, t2);
		kk=ik;
	} 

	seed=t1;
	for(ii=0; ii<NK; ii=ii+RECOMPUTING){
		vranlc_device(2*RECOMPUTING, &seed, A, x_local);
		for(i=0; i<RECOMPUTING; i++){
			x1=2.0*x_local[2*i]-1.0;
			x2=2.0*x_local[2*i+1]-1.0;
			t1=x1*x1+x2*x2;
			if(t1<=1.0){
				t2=sqrt(-2.0*log(t1)/t1);
				t3=(x1*t2);
				t4=(x2*t2);
				l=max(fabs(t3), fabs(t4));
				q_local[l]+=1.0;
				sx_local+=t3;
				sy_local+=t4;
			}
		}
	}

	gspar_atomic_add_double(q_global+gspar_get_block_id(0)*NQ+0, q_local[0]); 
	gspar_atomic_add_double(q_global+gspar_get_block_id(0)*NQ+1, q_local[1]); 
	gspar_atomic_add_double(q_global+gspar_get_block_id(0)*NQ+2, q_local[2]); 
	gspar_atomic_add_double(q_global+gspar_get_block_id(0)*NQ+3, q_local[3]); 
	gspar_atomic_add_double(q_global+gspar_get_block_id(0)*NQ+4, q_local[4]); 
	gspar_atomic_add_double(q_global+gspar_get_block_id(0)*NQ+5, q_local[5]); 
	gspar_atomic_add_double(q_global+gspar_get_block_id(0)*NQ+6, q_local[6]); 
	gspar_atomic_add_double(q_global+gspar_get_block_id(0)*NQ+7, q_local[7]); 
	gspar_atomic_add_double(q_global+gspar_get_block_id(0)*NQ+8, q_local[8]);
	gspar_atomic_add_double(q_global+gspar_get_block_id(0)*NQ+9, q_local[9]); 
	gspar_atomic_add_double(sx_global+gspar_get_block_id(0), sx_local); 
	gspar_atomic_add_double(sy_global+gspar_get_block_id(0), sy_local);

	//END
);
